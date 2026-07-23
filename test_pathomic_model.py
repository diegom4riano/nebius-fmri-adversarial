#!/usr/bin/env python3
"""KAPPA vs PGD adversarial attack on PathomicFusion MaxNet (TCGA-GBMLGG grade classification).

Mirrors test_fmri_model.py but for tabular genomic inputs.
No ForwardWrapper or library-attack normalisation needed: MaxNet is a plain MLP
that accepts a [1, 80] tensor and returns [1, 3] logits — compatible with
targeted_attack and pgd_attack directly.

Pool: correctly classified non-Grade-II subjects (same definition as profiler).
Target class: 0 (Grade II, most benign).
Threat model: L∞, epsilon sweep.
Attacks: KAPPA (Newton-CG, Lanczos damping) and PGD-40 and PGD-500.
Statistics: per-subject flip outcome + McNemar KAPPA vs PGD.

Usage:
  python test_pathomic_model.py \\
      --data-dir data/TCGA_GBMLGG \\
      --checkpoint data/TCGA_GBMLGG/checkpoints_omic/omic_1.pt \\
      --geometry output/geometry_maxnet.json
"""
import argparse
import json
import math
import os
import sys
import time

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hessian import targeted_attack, pgd_attack

TARGET      = 0   # Grade II (most benign) — adversarial target
EPSILON_SWEEP = [0.01, 0.05, 0.1, 0.5]
DATA_MIN    = -2.0
DATA_MAX    = 3.657


# ─────────────────────────────────────────────── MaxNet (identical to profiler)

def _init_max_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            stdv = 1.0 / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()


class MaxNet(nn.Module):
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
        self.classifier = nn.Sequential(nn.Linear(omic_dim, label_dim))
        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift  = Parameter(torch.FloatTensor([-3]), requires_grad=False)
        _init_max_weights(self)

    def forward(self, x):
        return self.classifier(self.encoder(x))

    def load_pathomic_checkpoint(self, path, device="cpu"):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict):
            sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        else:
            sd = ckpt
        missing, unexpected = self.load_state_dict(sd, strict=False)
        if missing:
            print(f"  [ckpt] missing keys: {missing}")
        if unexpected:
            print(f"  [ckpt] unexpected keys: {unexpected}")
        return self


# ─────────────────────────────────────────────── data loading (no pandas)

_META_COLS = {"TCGA ID", "indexes", "censored", "Survival months"}


def _load_data(data_dir):
    import csv
    feat_path  = os.path.join(data_dir, "all_dataset.csv")
    grade_path = os.path.join(data_dir, "grade_data.csv")

    grade_map = {}
    with open(grade_path) as f:
        for row in csv.DictReader(f):
            g = row.get("Grade", "").strip()
            if g and g.upper() != "NA":
                try:
                    grade_map[row["TCGA ID"]] = int(float(g))
                except ValueError:
                    pass

    with open(feat_path) as f:
        r = csv.reader(f)
        header = next(r)
        rows = list(r)

    feat_cols = [i for i, c in enumerate(header) if c not in _META_COLS]
    id_col    = header.index("TCGA ID")

    tcga_ids, raw_feats, labels = [], [], []
    for row in rows:
        tid = row[id_col].strip()
        if tid not in grade_map:
            continue
        grade = grade_map[tid]
        if grade not in (2, 3, 4):
            continue
        try:
            vals = [float(row[i]) if row[i].strip() else float("nan") for i in feat_cols]
        except ValueError:
            continue
        tcga_ids.append(tid)
        raw_feats.append(vals)
        labels.append(grade - 2)   # II→0, III→1, IV→2

    arr = np.array(raw_feats, dtype=np.float32)
    col_medians = np.nanmedian(arr, axis=0)
    for j in range(arr.shape[1]):
        mask = np.isnan(arr[:, j])
        arr[mask, j] = col_medians[j]

    X = torch.from_numpy(arr)
    y = torch.tensor(labels, dtype=torch.long)
    return tcga_ids, X, y


# ─────────────────────────────────────────────── McNemar test

def mcnemar_test(a_flips, b_flips):
    """McNemar test: does attack A flip subjects that B misses?"""
    n01 = sum(1 for a, b in zip(a_flips, b_flips) if a and not b)
    n10 = sum(1 for a, b in zip(a_flips, b_flips) if not a and b)
    total = n01 + n10
    if total == 0:
        return {"n01": 0, "n10": 0, "chi2": 0.0, "p": 1.0, "note": "no discordant pairs"}
    from scipy.stats import chi2 as chi2dist
    chi2 = (abs(n01 - n10) - 1) ** 2 / total  # continuity corrected
    p = chi2dist.sf(chi2, df=1)
    return {"n01": n01, "n10": n10, "chi2": round(chi2, 4), "p": round(p, 6)}


# ─────────────────────────────────────────────── main

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",   default="data/TCGA_GBMLGG")
    p.add_argument("--checkpoint", default="data/TCGA_GBMLGG/checkpoints_omic/omic_1.pt")
    p.add_argument("--geometry",   default="output/geometry_maxnet.json",
                   help="JSON from profile_geometry_pathomic.py (adds per-subject κ to results)")
    p.add_argument("--out",        default="output/attack_results_pathomic.json")
    p.add_argument("--epsilons",   type=float, nargs="+", default=EPSILON_SWEEP)
    p.add_argument("--lambda-reg", type=float, default=0.1)
    p.add_argument("--outer-steps",type=int,   default=5)
    p.add_argument("--cg-iters",   type=int,   default=50)
    p.add_argument("--pgd-steps",        type=int, default=40)
    p.add_argument("--pgd-matched-steps",type=int, default=500,
                   help="PGD steps for matched-budget comparison (default=outer*cg_iters*2)")
    p.add_argument("--seed",             type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── load data
    print(f"Loading data from {args.data_dir} …")
    tcga_ids, X, y = _load_data(args.data_dir)
    print(f"  {len(tcga_ids)} subjects, {X.shape[1]} features, {y.max().item()+1} classes")

    # ── load model
    model = MaxNet(input_dim=X.shape[1]).eval()
    print(f"Loading checkpoint: {args.checkpoint}")
    model.load_pathomic_checkpoint(args.checkpoint)
    model.eval()

    # ── pool: correctly classified, not Grade II
    with torch.no_grad():
        logits = model(X)
    preds = logits.argmax(1)
    pool_mask = (preds == y) & (y != TARGET)
    pool_idx  = pool_mask.nonzero(as_tuple=True)[0].tolist()
    print(f"  Pool (correct, non-target): {len(pool_idx)}/{len(tcga_ids)}")

    # ── load geometry (for per-subject κ annotation)
    kappa_by_id = {}
    if os.path.exists(args.geometry):
        with open(args.geometry) as f:
            geo = json.load(f)
        for s in geo.get("subjects", []):
            sid = s.get("id")
            if sid:
                kappa_by_id[sid] = (s.get("kappa_at_lambda") or {}).get("0.1")
        print(f"  Loaded κ for {len(kappa_by_id)} subjects from {args.geometry}")

    y_target = torch.tensor([TARGET], dtype=torch.long)

    epsilon_results = []
    for epsilon in args.epsilons:
        print(f"\n  ε = {epsilon}")
        t0_eps = time.time()

        kappa_flips = []   # KAPPA flipped this subject?
        pgd40_flips = []
        pgd500_flips= []
        per_subject = []

        for rank, idx in enumerate(pool_idx):
            x_i     = X[idx].unsqueeze(0)    # [1, 80]
            true_lbl = int(y[idx].item())
            tid      = tcga_ids[idx]
            kappa_val= kappa_by_id.get(tid)

            def fwd(x): return model(x)

            # KAPPA
            x_k, _ = targeted_attack(
                fwd, x_i, y_target,
                lambda_reg=args.lambda_reg, epsilon=epsilon,
                num_steps=args.outer_steps, max_iter=args.cg_iters,
                verbose=False, data_min=DATA_MIN, data_max=DATA_MAX,
                return_info=True, damping_mode="lanczos",
            )
            with torch.no_grad():
                pred_k = int(model(x_k).argmax(1).item())
            flipped_k = (pred_k == TARGET)

            # PGD-40
            x_p40 = pgd_attack(
                fwd, x_i, y_target,
                epsilon=epsilon, num_steps=args.pgd_steps,
                data_min=DATA_MIN, data_max=DATA_MAX,
            )
            with torch.no_grad():
                pred_p40 = int(model(x_p40).argmax(1).item())
            flipped_p40 = (pred_p40 == TARGET)

            # PGD matched budget: outer_steps × cg_iters backward passes
            x_p500 = pgd_attack(
                fwd, x_i, y_target,
                epsilon=epsilon, num_steps=args.pgd_matched_steps,
                data_min=DATA_MIN, data_max=DATA_MAX,
            )
            with torch.no_grad():
                pred_p500 = int(model(x_p500).argmax(1).item())
            flipped_p500 = (pred_p500 == TARGET)

            kappa_flips.append(flipped_k)
            pgd40_flips.append(flipped_p40)
            pgd500_flips.append(flipped_p500)

            per_subject.append({
                "tcga_id":    tid,
                "true_label": true_lbl,
                "kappa_at_0.1": kappa_val,
                "kappa_flip":  flipped_k,
                "pgd40_flip":  flipped_p40,
                "pgd500_flip": flipped_p500,
            })

            if (rank + 1) % 50 == 0 or rank == 0:
                k_asr  = sum(kappa_flips) / len(kappa_flips)
                p_asr  = sum(pgd40_flips) / len(pgd40_flips)
                print(f"    [{rank+1:3d}/{len(pool_idx)}]  KAPPA={k_asr:.3f}  PGD40={p_asr:.3f}", flush=True)

        n = len(pool_idx)
        kappa_asr = sum(kappa_flips) / n
        pgd40_asr = sum(pgd40_flips) / n
        pgd500_asr= sum(pgd500_flips) / n

        mc_kappa_pgd40  = mcnemar_test(kappa_flips, pgd40_flips)
        mc_kappa_pgd500 = mcnemar_test(kappa_flips, pgd500_flips)

        eps_entry = {
            "epsilon":    epsilon,
            "n_pool":     n,
            "attacks": {
                "kappa":    {"asr": round(kappa_asr, 6),  "flipped": sum(kappa_flips)},
                "pgd_40":   {"asr": round(pgd40_asr, 6),  "flipped": sum(pgd40_flips)},
                "pgd_500":  {"asr": round(pgd500_asr, 6), "flipped": sum(pgd500_flips)},
            },
            "mcnemar": {
                "kappa_vs_pgd40":  mc_kappa_pgd40,
                "kappa_vs_pgd500": mc_kappa_pgd500,
            },
            "time_s": round(time.time() - t0_eps, 1),
            "per_subject": per_subject,
        }
        epsilon_results.append(eps_entry)

        print(f"    KAPPA ASR={kappa_asr:.4f}  PGD-40 ASR={pgd40_asr:.4f}  PGD-500 ASR={pgd500_asr:.4f}")
        print(f"    McNemar KAPPA vs PGD-40:  χ²={mc_kappa_pgd40['chi2']}  p={mc_kappa_pgd40['p']}")
        print(f"    McNemar KAPPA vs PGD-500: χ²={mc_kappa_pgd500['chi2']}  p={mc_kappa_pgd500['p']}")

        # partial save after each ε
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        partial = args.out.replace(".json", "_partial.json")
        with open(partial, "w") as f:
            json.dump({"epsilon_results": epsilon_results}, f, indent=2)

    output = {
        "checkpoint":   args.checkpoint,
        "n_pool":       len(pool_idx),
        "target_class": TARGET,
        "epsilons":     args.epsilons,
        "attack_config": {
            "kappa_outer_steps": args.outer_steps,
            "kappa_cg_iters":    args.cg_iters,
            "lambda_reg":        args.lambda_reg,
            "damping_mode":      "lanczos",
            "pgd_steps":         args.pgd_steps,
            "pgd_500_steps":     500,
            "data_min":          DATA_MIN,
            "data_max":          DATA_MAX,
            "threat_model":      "L_inf",
        },
        "epsilon_results": epsilon_results,
    }
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved → {args.out}")


if __name__ == "__main__":
    main()
