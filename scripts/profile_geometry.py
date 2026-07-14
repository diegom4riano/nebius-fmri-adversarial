#!/usr/bin/env python3
"""Per-subject loss-surface geometry profiler (offline MVP for geometry-aware routing).

For every attackable subject (non-target, correctly classified) it computes the local
geometry of the loss w.r.t. the input:

  grad_norm   = ||d/dx CE(f(x), target=0)||           (is there any first-order signal?)
  mu_min/max  = extreme eigenvalues of the loss Hessian (one reorthogonalized Lanczos pass)
  sigma_max   = max(|mu_min|, |mu_max|)                (curvature magnitude)
  kappa@lambda= (mu_max+s+l)/(mu_min+s+l), s=max(0,-mu_min)   (anisotropy at a common lambda)
  pd_ok       = minimal PD damping makes (H+lambda I) > 0

The heavy 3x eigsh of estimate_condition_number is replaced by a single k-step Lanczos
(~k HVPs vs ~600), which is what makes profiling the whole pool feasible on CPU. No new
math: the eigenvalue/kappa formulas mirror estimate_condition_number exactly, so the
per-subject values are directly comparable to the aggregate condition_number already saved.

Output is aligned by subject POSITION to the test loader (shuffle=False), so it joins to
the per_subject attack outcomes in the result JSONs by index. Writes incrementally and
resumes, so an overnight CPU run survives interruption.

Usage:
  python scripts/profile_geometry.py --model stagin --smoke-test          # CPU, synthetic
  python scripts/profile_geometry.py --model stagin \
      --pool-json output/20260713_104957_base_e0.001_0/attack_results.json
  python scripts/profile_geometry.py --model ecg \
      --pool-json output/20260711_170335/attack_results_ecg.json
"""
import argparse
import json
import os
import sys
import time
import types

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hessian import _lanczos_extreme_eigs
# Importing does not run main (both drivers guard it under __main__).
from test_fmri_model import (
    DEVICE, DEFAULTS, ModelSTAGIN, make_loaders, set_attack_mode,
    _SmokeDataset, _smoke_collate, _infer_input_dim,
)
from test_pytorch_model import _load_data, NUM_CLASSES
from model.CNN import CNN
from utils.DataLoader import ECGDataset, ecg_collate_func

TARGET = 0                      # non-target class (Female for STAGIN, Normal for ECG)
REF_LAMBDAS = (1e-6, 1e-3, 1e-1)
LAMBDA_MIN_PD = 0.1             # damping floor for the PD-check, common to both models
SEED = 42


# --------------------------------------------------------------------------- geometry
def _extreme_eigs(grad_graph, x1, k, seed_offset):
    """(mu_min, mu_max) of the loss Hessian via one reorthogonalized Lanczos pass.

    matvec is the exact autodiff HVP: a second backward through the retained graph of the
    first gradient. grad_graph must have been built with create_graph=True.
    """
    gen = torch.Generator(device=x1.device).manual_seed(SEED + seed_offset)
    x0 = torch.randn(x1.shape, generator=gen, device=x1.device, dtype=x1.dtype)

    def matvec(q):
        return torch.autograd.grad(grad_graph, x1, grad_outputs=q, retain_graph=True)[0]

    return _lanczos_extreme_eigs(matvec, x0, k=k, reorth=True)


def _geometry_from_eigs(mu_min, mu_max):
    """Derived quantities, mirroring estimate_condition_number's formulas exactly."""
    sigma_max = max(abs(mu_min), abs(mu_max))
    shift = max(0.0, -mu_min)                         # lifts mu_min to 0 before adding lambda
    kappa_at = {f"{lm:g}": (mu_max + shift + lm) / (mu_min + shift + lm) for lm in REF_LAMBDAS}
    pd_ok = (mu_min + shift + LAMBDA_MIN_PD) > 0
    return sigma_max, kappa_at, pd_ok


def _geom_record(grad_norm, mu_min, mu_max):
    if mu_min is None or mu_max is None or not np.isfinite(mu_min) or not np.isfinite(mu_max):
        return dict(grad_norm=grad_norm, mu_min=None, mu_max=None, sigma_max=None,
                    kappa_at_lambda=None, pd_ok=None)
    sigma_max, kappa_at, pd_ok = _geometry_from_eigs(mu_min, mu_max)
    return dict(grad_norm=grad_norm, mu_min=mu_min, mu_max=mu_max, sigma_max=sigma_max,
                kappa_at_lambda=kappa_at, pd_ok=bool(pd_ok))


# --------------------------------------------------------------- per-model subject profilers
def _profile_stagin_subject(model, v1, a1, t1, endpoints, k, seed_offset):
    v1 = v1.clone().detach().requires_grad_(True)
    target1 = torch.zeros(1, dtype=torch.long, device=v1.device)
    logits = model(v1, a1, t1, endpoints)[0]
    loss = F.cross_entropy(logits, target1, reduction="sum")
    grad_graph = torch.autograd.grad(loss, v1, create_graph=True, retain_graph=True)[0]
    grad_norm = float(grad_graph.detach().norm().item())
    mu_min, mu_max = _extreme_eigs(grad_graph, v1, k, seed_offset)
    return _geom_record(grad_norm, mu_min, mu_max)


def _profile_ecg_subject(model, x1, k, seed_offset):
    x1 = x1.clone().detach().requires_grad_(True)
    target1 = torch.zeros(1, dtype=torch.long, device=x1.device)
    logits = model(x1)
    loss = F.cross_entropy(logits, target1, reduction="sum")
    grad_graph = torch.autograd.grad(loss, x1, create_graph=True, retain_graph=True)[0]
    grad_norm = float(grad_graph.detach().norm().item())
    mu_min, mu_max = _extreme_eigs(grad_graph, x1, k, seed_offset)
    return _geom_record(grad_norm, mu_min, mu_max)


# --------------------------------------------------------------------------- pool + I/O
def _pool_positions(pool_json):
    """Positions of attackable subjects: label != TARGET AND pred_clean != TARGET."""
    d = json.load(open(pool_json))
    ps = d["epsilon_results"][0]["per_subject"]
    lab = np.array(ps["labels"]); pc = np.array(ps["pred_clean"])
    pool = (lab != TARGET) & (pc != TARGET)
    return set(int(i) for i in np.where(pool)[0]), int(len(lab))


def _load_done(out_path):
    """Resume: {pos: record} already computed in a prior (possibly interrupted) run."""
    if not os.path.exists(out_path):
        return {}
    try:
        prev = json.load(open(out_path))
        return {int(s["pos"]): s for s in prev.get("subjects", [])}
    except Exception:
        return {}


def _save(out_path, model_name, n_total, done):
    payload = {
        "model": model_name, "seed": SEED, "target": TARGET,
        "ref_lambdas": [f"{lm:g}" for lm in REF_LAMBDAS], "lambda_min_pd": LAMBDA_MIN_PD,
        "n_total": n_total, "n_profiled": len(done),
        "subjects": [done[p] for p in sorted(done)],
    }
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, out_path)   # atomic: an interrupted write never corrupts the resume file


# --------------------------------------------------------------------------- builders
def _build_stagin(args):
    if args.smoke_test:
        print("SMOKE TEST — synthetic fMRI, random weights (CPU ok)\n")
        input_dim = args.smoke_input_dim
        ds = _SmokeDataset(input_dim, n_samples=args.smoke_samples)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=min(args.smoke_samples, 4), collate_fn=_smoke_collate, shuffle=False)
    else:
        roi = os.path.join(args.data_dir, "roi_timeseries.npy")
        lab = os.path.join(args.data_dir, "labels.npy")
        _, _, loader = make_loaders(roi, lab, batch_size=args.batch, seed=SEED)
        input_dim = np.load(roi).shape[1]
    model = ModelSTAGIN(input_dim=input_dim, hidden_dim=DEFAULTS["hidden_dim"], num_classes=2,
                        num_heads=DEFAULTS["num_heads"], num_layers=DEFAULTS["num_layers"],
                        sparsity=DEFAULTS["sparsity"], dropout=DEFAULTS["dropout"],
                        cls_token=DEFAULTS["cls_token"], readout=DEFAULTS["readout"])
    if not args.smoke_test and os.path.exists(args.ckpt):
        model.load_state_dict(torch.load(args.ckpt, map_location=DEVICE))
        print(f"Checkpoint: {args.ckpt}")
    model = model.to(DEVICE)
    set_attack_mode(model)                          # eval + GRU-train so the HVP backward runs
    return model, loader


def _build_ecg(args):
    if args.smoke_test:
        print("SMOKE TEST — synthetic ECG, random weights (CPU ok)\n")
    # _load_data reproduces the exact eval ordering (fixed permutation + last 3%) so positions
    # align with the ECG attack results.
    la = types.SimpleNamespace(smoke_test=args.smoke_test, seed=SEED,
                               smoke_samples=args.smoke_samples, data_dir=args.data_dir)
    X, y, _dmin, _dmax = _load_data(la, {})
    loader = torch.utils.data.DataLoader(
        ECGDataset(list(X), list(y)), batch_size=args.batch, shuffle=False,
        collate_fn=ecg_collate_func)
    model = CNN(num_classes=NUM_CLASSES)
    if not args.smoke_test and os.path.exists(args.ckpt_ecg):
        sd = torch.load(args.ckpt_ecg, map_location=DEVICE)
        sd = {k.removeprefix("module."): v for k, v in sd.items()}
        model.load_state_dict(sd)
        print(f"Checkpoint: {args.ckpt_ecg}")
    model = model.to(DEVICE).eval()
    return model, loader


# --------------------------------------------------------------------------- main loop
def run(args):
    model_name = args.model
    if model_name == "stagin":
        model, loader = _build_stagin(args)
    else:
        model, loader = _build_ecg(args)

    out_path = args.out or os.path.join(args.output_dir, f"geometry_{model_name}.json")
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine which positions to profile.
    if args.smoke_test or not args.pool_json:
        pool, n_total = set(), None
        profile_all = True
    else:
        pool, n_total = _pool_positions(args.pool_json)
        profile_all = args.all
        print(f"Pool: {len(pool)} attackable subjects (of {n_total})"
              f"{' — profiling ALL' if profile_all else ''}")

    done = _load_done(out_path)
    if done:
        print(f"Resume: {len(done)} subjects already in {out_path}")

    pos = -1
    t_start = time.time()
    for batch in loader:
        if model_name == "stagin":
            v, a, t, endpoints, labels = batch
            v, a, t = v.to(DEVICE), a.to(DEVICE), t.to(DEVICE)
            B = v.shape[0]
        else:
            data, _lengths, labels = batch
            data = data.to(DEVICE)
            B = data.shape[0]

        for j in range(B):
            pos += 1
            in_pool = (pos in pool)
            if not (profile_all or in_pool):
                continue
            if pos in done:
                continue
            t0 = time.time()
            if model_name == "stagin":
                rec = _profile_stagin_subject(
                    model, v[j:j + 1], a[j:j + 1], t[:, j:j + 1, :], endpoints,
                    args.lanczos_k, pos)
            else:
                rec = _profile_ecg_subject(model, data[j:j + 1], args.lanczos_k, pos)
            rec["pos"] = pos
            rec["in_pool"] = bool(in_pool) if n_total is not None else True
            rec["label"] = int(labels[j])
            done[pos] = rec
            _save(out_path, model_name, n_total, done)
            km = rec["kappa_at_lambda"]["0.1"] if rec["kappa_at_lambda"] else float("nan")
            print(f"  pos={pos:>4} pool={rec['in_pool']!s:>5} "
                  f"||g||={rec['grad_norm']:>10.4g} mu_min={_fmt(rec['mu_min'])} "
                  f"mu_max={_fmt(rec['mu_max'])} k@0.1={_fmt(km)} "
                  f"pd={rec['pd_ok']!s:>5} ({time.time()-t0:.1f}s)", flush=True)

    print(f"\nDone: {len(done)} subjects → {out_path}  ({time.time()-t_start:.0f}s total)")


def _fmt(x):
    return "  none" if x is None else f"{x:>9.4g}"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["stagin", "ecg"], required=True)
    p.add_argument("--pool-json", default=None,
                   help="base/results JSON whose per_subject defines the attackable pool")
    p.add_argument("--all", action="store_true",
                   help="profile every subject, not only the attackable pool")
    p.add_argument("--out", default=None)
    p.add_argument("--output-dir", default="output")
    p.add_argument("--lanczos-k", type=int, default=30)
    p.add_argument("--batch", type=int, default=DEFAULTS["batch"])
    p.add_argument("--data-dir", default=DEFAULTS["data_dir"],
                   help="fMRI ROI dir (stagin) or ECG data dir (ecg)")
    p.add_argument("--ckpt", default=DEFAULTS["ckpt"], help="STAGIN checkpoint")
    p.add_argument("--ckpt-ecg", default="saved_model/best_model.pth")
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--smoke-samples", type=int, default=8)
    p.add_argument("--smoke-input-dim", type=int, default=16)
    args = p.parse_args()
    # ECG default data dir differs from the fMRI default.
    if args.model == "ecg" and args.data_dir == DEFAULTS["data_dir"]:
        args.data_dir = "data"
    return args


if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    run(parse_args())
