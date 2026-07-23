#!/usr/bin/env python3
"""Per-subject loss-surface geometry profiler for PathomicFusion MaxNet (TCGA-GBMLGG grade classification).

MaxNet is a 4-layer SNN (ELU + AlphaDropout, NO BatchNorm, NO LayerNorm) that classifies
glioma grade (II/III/IV) from 80 genomic features (IDH mutation + copy number variation).
Expected to have high κ(H+λI) >> 1 due to absence of normalisation layers on correlated
genomic inputs — this script measures that empirically.

Architecture (Mahmood Lab, ICCV 2021):
  input(80) → [Linear→ELU→AlphaDropout] ×4 → [64→48→32→32] → Linear → 3 logits

Data (all publicly available, no credentials required):
  all_datasets.csv   — 769 samples, 80 genomic features + metadata
  grade_data.csv     — WHO grade labels (II=0, III=1, IV=2 after -2 shift)
  Checkpoint .pth    — per-fold trained weights from Mahmood Lab Google Drive

Download (one-time, ~200 MB total):
  pip install gdown
  gdown --folder 1swiMrz84V3iuzk8x99vGIBd5FCVncOlf -O data/TCGA_GBMLGG/

Output: output/geometry_maxnet.json  — same schema as geometry_stagin.json

Usage:
  python scripts/profile_geometry_pathomic.py --smoke-test
  python scripts/profile_geometry_pathomic.py \\
      --data-dir data/TCGA_GBMLGG \\
      --checkpoint data/TCGA_GBMLGG/checkpoints/omic/omic_1.pt \\
      --out output/geometry_maxnet.json
"""
import argparse
import json
import math
import os
import sys

# macOS OpenMP/threadpool import deadlock guard. numpy (OpenBLAS) and torch each ship their
# own libomp; on this box the threadpool init races at import and the main thread blocks
# forever on a pthread condvar (0% CPU hang, confirmed via `sample`). Forcing single-threaded
# BLAS/OMP before any heavy import removes the race; a 80-dim MLP needs no thread parallelism.
# Must be set before torch/numpy import. Import torch before numpy as a second safeguard.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
torch.set_num_threads(1)
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hessian import _lanczos_extreme_eigs

TARGET      = 0          # Grade II (most benign) — adversarial target: push toward benign
REF_LAMBDAS = (1e-6, 1e-3, 1e-1)
LAMBDA_MIN_PD = 0.1
SEED        = 42
LANCZOS_K   = 30


# ─────────────────────────────────────────────── MaxNet (standalone, from Mahmood Lab)

def _init_max_weights(module):
    """Weight initialisation from PathomicFusion utils.py (self-normalising variant)."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            stdv = 1.0 / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()


class MaxNet(nn.Module):
    """Genomic SNN from PathomicFusion (Chenetal., ICCV 2021).

    Re-implemented standalone — no PathomicFusion/torch_geometric dependency.
    Architecture and weight init mirror networks.py exactly.
    """

    def __init__(self, input_dim=80, omic_dim=32, dropout_rate=0.25, label_dim=3):
        super().__init__()
        hidden = [64, 48, 32, omic_dim]

        def _block(in_f, out_f):
            return nn.Sequential(
                nn.Linear(in_f, out_f),
                nn.ELU(),
                nn.AlphaDropout(p=dropout_rate, inplace=False),
            )

        self.encoder = nn.Sequential(
            _block(input_dim, hidden[0]),
            _block(hidden[0], hidden[1]),
            _block(hidden[1], hidden[2]),
            _block(hidden[2], hidden[3]),
        )
        # Original PathomicFusion wraps classifier in Sequential; must match for checkpoint load.
        self.classifier = nn.Sequential(nn.Linear(omic_dim, label_dim))

        # fixed output-range scalars (unused for cross-entropy, kept for weight-load compat)
        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift  = Parameter(torch.FloatTensor([-3]), requires_grad=False)

        _init_max_weights(self)

    def forward(self, x):
        """Accept plain tensor (not **kwargs) for gradient compatibility."""
        features = self.encoder(x)
        return self.classifier(features)

    def load_pathomic_checkpoint(self, path, device="cpu"):
        """Load a checkpoint saved by the original PathomicFusion training loop.

        Original code saved state_dict with keys like 'encoder.0.0.weight' etc.
        This loader handles both flat state_dict and {'model_state_dict': ...} wrappers.
        """
        ckpt = torch.load(path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict):
            sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        else:
            sd = ckpt

        # The original forward used **kwargs; we patched forward to accept plain tensor.
        # State dict keys are unchanged — just load directly.
        missing, unexpected = self.load_state_dict(sd, strict=False)
        if missing:
            print(f"  [ckpt] missing keys: {missing}")
        if unexpected:
            print(f"  [ckpt] unexpected keys: {unexpected}")
        return self


# ──────────────────────────────────────────────────────── geometry primitives (same as profile_geometry.py)

def _extreme_eigs(grad_graph, x1, k, seed_offset):
    gen = torch.Generator(device=x1.device).manual_seed(SEED + seed_offset)
    x0  = torch.randn(x1.shape, generator=gen, device=x1.device, dtype=x1.dtype)

    def matvec(q):
        return torch.autograd.grad(grad_graph, x1, grad_outputs=q, retain_graph=True)[0]

    return _lanczos_extreme_eigs(matvec, x0, k=k, reorth=True)


def _geometry_from_eigs(mu_min, mu_max):
    sigma_max = max(abs(mu_min), abs(mu_max))
    shift     = max(0.0, -mu_min)
    kappa_at  = {f"{lm:g}": (mu_max + shift + lm) / (mu_min + shift + lm) for lm in REF_LAMBDAS}
    pd_ok     = (mu_min + shift + LAMBDA_MIN_PD) > 0
    return sigma_max, kappa_at, pd_ok


def _geom_record(grad_norm, mu_min, mu_max):
    if mu_min is None or mu_max is None or not np.isfinite(mu_min) or not np.isfinite(mu_max):
        return dict(grad_norm=grad_norm, mu_min=None, mu_max=None, sigma_max=None,
                    kappa_at_lambda=None, pd_ok=None)
    sigma_max, kappa_at, pd_ok = _geometry_from_eigs(mu_min, mu_max)
    return dict(grad_norm=grad_norm, mu_min=mu_min, mu_max=mu_max,
                sigma_max=sigma_max, kappa_at_lambda=kappa_at, pd_ok=bool(pd_ok))


def _profile_subject(model, x_feat, label_idx, seed_offset):
    """Profile one subject: grad_norm + Lanczos κ."""
    x = x_feat.clone().detach().requires_grad_(True)
    target = torch.tensor([TARGET], dtype=torch.long, device=x.device)

    logits = model(x)
    loss   = F.cross_entropy(logits, target, reduction="sum")
    grad_g = torch.autograd.grad(loss, x, create_graph=True, retain_graph=True)[0]

    grad_norm = float(grad_g.detach().norm().item())
    mu_min, mu_max = _extreme_eigs(grad_g, x, LANCZOS_K, seed_offset)
    rec = _geom_record(grad_norm, mu_min, mu_max)
    rec["label"] = label_idx
    return rec


# ─────────────────────────────────────────────────────── data loading

# Metadata columns present in all_dataset.csv (everything else is a genomic feature).
# The remaining 80 columns = codeletion + idh mutation + gene mutations + chromosome-arm CNVs.
_META_COLS = {"TCGA ID", "indexes", "censored", "Survival months"}


def _median(vals):
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _load_data(data_dir):
    """Load all_dataset.csv + grade_data.csv via stdlib csv (pandas 3.0.3 deadlocks on import).

    Mirrors PathomicFusion getCleanAllDataset: drop metadata columns, merge Grade from
    grade_data.csv on TCGA ID, map Grade II/III/IV → 0/1/2, median-impute missing features.

    Returns:
        X    — torch.FloatTensor (N, 80)
        y    — list[int] — grade label 0/1/2 (Grade II/III/IV)
        ids  — list[str] — TCGA IDs
    """
    import csv

    feat_path  = os.path.join(data_dir, "all_dataset.csv")
    grade_path = os.path.join(data_dir, "grade_data.csv")

    if not os.path.exists(feat_path):
        raise FileNotFoundError(
            f"Missing: {feat_path}\n"
            "Download + extract from Google Drive:\n"
            "  gdown 1McOK93l31ALaoAJaE8vJGRB3S3NK99MU -O data.zip\n"
            "  python -c \"import zipfile; zipfile.ZipFile('data.zip').extractall()\""
        )

    # grade_data.csv → {TCGA ID: grade_int}, skipping NA
    grade_map = {}
    with open(grade_path) as f:
        r = csv.DictReader(f)
        for row in r:
            g = row.get("Grade", "").strip()
            if g and g.upper() != "NA":
                try:
                    grade_map[row["TCGA ID"]] = int(float(g))
                except ValueError:
                    pass

    # all_dataset.csv → feature rows
    with open(feat_path) as f:
        r = csv.reader(f)
        header = next(r)
        rows = list(r)

    feat_cols = [i for i, c in enumerate(header) if c not in _META_COLS]
    id_col    = header.index("TCGA ID")

    # parse features as float, tracking NaN for later median imputation
    raw_feats, ids, y = [], [], []
    for row in rows:
        tid = row[id_col]
        if tid not in grade_map:          # no grade label → skip (matches dropna)
            continue
        vals = []
        for i in feat_cols:
            v = row[i].strip()
            try:
                vals.append(float(v))
            except ValueError:
                vals.append(float("nan"))
        raw_feats.append(vals)
        ids.append(tid)
        y.append(grade_map[tid] - 2)      # Grade II→0, III→1, IV→2

    # median-impute per column (over non-NaN entries)
    n_feat = len(feat_cols)
    col_medians = []
    for j in range(n_feat):
        present = [raw_feats[i][j] for i in range(len(raw_feats))
                   if not math.isnan(raw_feats[i][j])]
        col_medians.append(_median(present))
    for i in range(len(raw_feats)):
        for j in range(n_feat):
            if math.isnan(raw_feats[i][j]):
                raw_feats[i][j] = col_medians[j]

    X = torch.tensor(raw_feats, dtype=torch.float32)
    return X, y, ids


def _load_done(out_path):
    if not os.path.exists(out_path):
        return {}
    try:
        prev = json.load(open(out_path))
        return {int(s["pos"]): s for s in prev.get("subjects", [])}
    except Exception:
        return {}


def _save(out_path, n_total, done):
    payload = {
        "model": "MaxNet_GBMLGG", "seed": SEED, "target": TARGET,
        "ref_lambdas": [f"{lm:g}" for lm in REF_LAMBDAS], "lambda_min_pd": LAMBDA_MIN_PD,
        "n_total": n_total, "n_profiled": len(done),
        "subjects": [done[p] for p in sorted(done)],
    }
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, out_path)


# ─────────────────────────────────────────────────────── main profiling loop

def _run_smoke(args):
    print("SMOKE TEST — synthetic 80-dim inputs, random MaxNet weights (CPU)\n")
    N   = args.smoke_samples
    rng = torch.Generator().manual_seed(SEED)

    model = MaxNet(input_dim=80, label_dim=3)
    model.eval()

    done = {}
    for i in range(N):
        x     = torch.randn(1, 80, generator=rng)
        label = int(torch.randint(3, (1,), generator=rng).item())
        rec   = _profile_subject(model, x, label, seed_offset=i)
        rec["pos"] = i
        done[i] = rec
        k_str = f"{rec['kappa_at_lambda']['0.1']:.2f}" if rec["kappa_at_lambda"] else "n/a"
        print(f"  [{i:2d}] grad={rec['grad_norm']:.4f}  "
              f"mu_min={rec['mu_min']:.4f}  mu_max={rec['mu_max']:.4f}  "
              f"κ@0.1={k_str}")

    _save(args.out, N, done)
    print(f"\nSmoke PASSED — wrote {args.out}")


def _run_real(args):
    device = torch.device("cpu")  # 80-dim MLP — CPU is fine
    print(f"Loading data from {args.data_dir} …")
    X, y, ids = _load_data(args.data_dir)
    N = len(y)
    print(f"  {N} subjects, {X.shape[1]} features, {len(set(y))} classes")

    model = MaxNet(input_dim=X.shape[1], label_dim=3).to(device)
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        model.load_pathomic_checkpoint(args.checkpoint, device=device)
    else:
        print("WARNING: no checkpoint — using random weights (geometry will be meaningless)")
    model.eval()

    # in_pool: correctly classified by the model (attackable subjects)
    with torch.no_grad():
        logits_all = model(X.to(device))
        preds = logits_all.argmax(dim=1).cpu().tolist()

    in_pool = {i for i, (p, label) in enumerate(zip(preds, y)) if p == label and p != TARGET}
    print(f"  In pool (correct, non-target): {len(in_pool)}/{N}")

    done  = _load_done(args.out)
    todo  = sorted(in_pool - set(done.keys()))
    print(f"  Already done: {len(done)}, remaining: {len(todo)}")

    for count, i in enumerate(todo):
        x = X[i:i+1].to(device)
        try:
            rec = _profile_subject(model, x, int(y[i]), seed_offset=i)
        except Exception as e:
            print(f"  [{i}] ERROR: {e}")
            rec = dict(grad_norm=float("nan"), mu_min=None, mu_max=None,
                       sigma_max=None, kappa_at_lambda=None, pd_ok=None, label=int(y[i]))

        rec["pos"]     = i
        rec["in_pool"] = True
        rec["id"]      = str(ids[i])
        done[i] = rec

        k_str = f"{rec['kappa_at_lambda']['0.1']:.2f}" if rec.get("kappa_at_lambda") else "n/a"
        print(f"  [{i:3d}/{N}] κ@0.1={k_str}  σ_max={rec.get('sigma_max', 'n/a')}"
              f"  grade={y[i]}", flush=True)

        if count % 20 == 0:
            _save(args.out, N, done)

    # also record out-of-pool subjects (not attackable, geometry still useful)
    for i in range(N):
        if i not in done:
            rec = {"pos": i, "in_pool": False, "label": int(y[i]),
                   "grad_norm": None, "mu_min": None, "mu_max": None,
                   "sigma_max": None, "kappa_at_lambda": None, "pd_ok": None}
            done[i] = rec

    _save(args.out, N, done)
    print(f"\nDone — wrote {args.out}")

    # quick κ summary
    kvals = [s["kappa_at_lambda"]["0.1"]
             for s in done.values()
             if s.get("kappa_at_lambda") and s["kappa_at_lambda"].get("0.1")]
    if kvals:
        arr = np.array(kvals)
        print(f"\nκ(H+0.1I) distribution over {len(arr)} in-pool subjects:")
        print(f"  median={np.median(arr):.2f}  p90={np.percentile(arr,90):.2f}"
              f"  max={arr.max():.2f}  min={arr.min():.2f}")


# ─────────────────────────────────────────────────────── CLI

def main():
    p = argparse.ArgumentParser(description="Geometry profiler — PathomicFusion MaxNet")
    p.add_argument("--data-dir",    default="data/TCGA_GBMLGG",
                   help="directory containing all_datasets.csv and grade_data.csv")
    p.add_argument("--checkpoint",  default=None,
                   help=".pt/.pth checkpoint file (one fold); omit for random weights")
    p.add_argument("--out",         default="output/geometry_maxnet.json")
    p.add_argument("--smoke-test",  action="store_true")
    p.add_argument("--smoke-samples", type=int, default=10)
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if args.smoke_test:
        _run_smoke(args)
    else:
        _run_real(args)


if __name__ == "__main__":
    main()
