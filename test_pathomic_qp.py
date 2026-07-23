#!/usr/bin/env python3
"""Box-constrained QP attack on MaxNet — the theoretically-correct second-order L∞ attack.

Motivation
----------
KAPPA (Newton-CG) solves the UNCONSTRAINED damped system (H+λI)δ = −g in L2, then
projects δ onto the L∞ ε-ball afterwards. That projection truncates exactly the
direction the Newton step optimised, so κ predicts CG convergence but NOT how well the
step exploits the L∞ geometry. On MaxNet (κ up to 2261) KAPPA lost to PGD.

This attack removes that mismatch. It solves the second-order model directly UNDER the
L∞ box, so the quadratic is optimised in the same geometry the threat model lives in:

    min_δ   gᵀδ + ½ δᵀ(H+λI)δ
    s.t.    x_adv + δ ∈ [max(x₀−ε, lo), min(x₀+ε, hi)]   (componentwise)

Everything else is IDENTICAL to KAPPA (targeted_attack): same outer loop, same per-step
Lanczos damping λ = max(0,−μ_min)+λ_reg, same exact autodiff HVP. The ONLY change is
CG-then-project  →  box-constrained QP solve. This isolates L2-projection vs
L∞-constrained-solve as the single variable.

The damped operator H+λI ≻ 0 by construction ⇒ the QP is convex ⇒ L-BFGS-B (matrix-free,
one HVP per objective/grad eval) finds its unique global minimum. (The undamped indefinite
box-QP is NP-hard; damping matches KAPPA and keeps the comparison fair.)

If QP ≈ PGD on the high-κ subjects, the routing hypothesis is dead: doing the second-order
attack *correctly* for L∞ still buys nothing. If QP > PGD, the earlier KAPPA loss was an
artefact of the projection, not of second-order information being useless.

Usage:
  python test_pathomic_qp.py --epsilons 0.05 0.1
  python test_pathomic_qp.py --epsilons 0.05 --high-kappa 50   # fast: only κ≥50 subjects
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
import torch.nn.functional as F
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hessian import _lanczos_extreme_eigs, pgd_attack, targeted_attack
from test_pathomic_model import (
    MaxNet, _load_data, mcnemar_test, TARGET, DATA_MIN, DATA_MAX,
)


# ─────────────────────────────────────────────── QP box-constrained attack

def qp_box_attack(model, x, y_target, epsilon, lambda_reg=0.1, num_steps=5,
                  lanczos_iters=30, data_min=None, data_max=None,
                  qp_maxiter=200, seed=20260711, return_info=False):
    """Damped second-order attack that respects the L∞ box DURING the solve.

    Mirrors hessian.targeted_attack exactly except the inner step: instead of
    δ = CG(H+λI, −g) followed by an L∞ projection, we solve the box-constrained QP
    min gᵀδ + ½δᵀ(H+λI)δ over the feasible box directly (L-BFGS-B, matrix-free HVP).
    """
    x_orig = x.clone().detach()
    x_adv  = x.clone().detach().requires_grad_(True)
    shape  = x.shape
    d      = x.numel()

    info = {"mu_min": [], "mu_max": [], "lambda": [], "qp_obj": [], "qp_iters": []}

    for step in range(num_steps):
        out  = model(x_adv)
        loss = F.cross_entropy(out, y_target, reduction="sum")   # targeted → minimise CE to target
        cached_grad = torch.autograd.grad(loss, x_adv, create_graph=True, retain_graph=True)[0]
        g = cached_grad.detach()

        # Exact autodiff HVP (same double-backward as KAPPA). Graph retained across all
        # L-BFGS-B evals this outer step; rebuilt next step when x_adv is recreated.
        def _H(vv):
            return torch.autograd.grad(cached_grad, x_adv, grad_outputs=vv, retain_graph=True)[0]

        # Per-step Lanczos damping — identical formula to KAPPA (targeted_attack).
        gen = torch.Generator(device=x.device).manual_seed(seed + step)
        x0  = torch.randn(g.shape, generator=gen, device=x.device, dtype=g.dtype)
        mu_min, mu_max = _lanczos_extreme_eigs(_H, x0, k=lanczos_iters, reorth=True)
        if not math.isfinite(mu_min):
            smax   = abs(mu_max) if math.isfinite(mu_max) else 1.0
            mu_min = -smax
        lam = max(0.0, -mu_min) + lambda_reg

        # Feasible box for δ so the CUMULATIVE perturbation stays in the ε-ball ∩ domain:
        #   x_adv + δ ∈ [max(x₀−ε, lo), min(x₀+ε, hi)]  (same target set KAPPA projects onto).
        hi_p = x_orig + epsilon
        lo_p = x_orig - epsilon
        if data_max is not None:
            hi_p = torch.clamp(hi_p, max=data_max)
        if data_min is not None:
            lo_p = torch.clamp(lo_p, min=data_min)
        ub = (hi_p - x_adv.detach()).flatten().double().cpu().numpy()
        lb = (lo_p - x_adv.detach()).flatten().double().cpu().numpy()
        # Numerical guard: rounding can leave lb marginally above ub → clip to lb.
        ub = np.maximum(ub, lb)

        g_np = g.flatten().double().cpu().numpy()

        def q_and_grad(delta_np):
            delta  = torch.as_tensor(delta_np, dtype=x.dtype, device=x.device).view(shape)
            Hd     = _H(delta).detach().flatten().double().cpu().numpy()
            grad_q = g_np + Hd + lam * delta_np                       # ∇q(δ) = g + (H+λI)δ
            obj    = float(g_np @ delta_np + 0.5 * delta_np @ Hd
                           + 0.5 * lam * (delta_np @ delta_np))       # q(δ)
            return obj, grad_q

        res = minimize(q_and_grad, x0=np.zeros(d), method="L-BFGS-B", jac=True,
                       bounds=list(zip(lb, ub)),
                       options={"maxiter": qp_maxiter, "ftol": 1e-12, "gtol": 1e-9})
        delta_star = torch.as_tensor(res.x, dtype=x.dtype, device=x.device).view(shape)

        # δ is feasible by construction — no projection needed (that's the whole point).
        x_adv = (x_adv.detach() + delta_star).clone().detach().requires_grad_(True)

        info["mu_min"].append(mu_min)
        info["mu_max"].append(mu_max)
        info["lambda"].append(lam)
        info["qp_obj"].append(float(res.fun))
        info["qp_iters"].append(int(res.nit))

    return (x_adv.detach(), info) if return_info else x_adv.detach()


# ─────────────────────────────────────────────── main

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",   default="data/TCGA_GBMLGG")
    p.add_argument("--checkpoint", default="data/TCGA_GBMLGG/checkpoints_omic/omic_1.pt")
    p.add_argument("--geometry",   default="output/geometry_maxnet.json")
    p.add_argument("--out",        default="output/attack_results_pathomic_qp.json")
    p.add_argument("--epsilons",   type=float, nargs="+", default=[0.05, 0.1])
    p.add_argument("--lambda-reg", type=float, default=0.1)
    p.add_argument("--outer-steps",type=int,   default=5)
    p.add_argument("--qp-maxiter", type=int,   default=200)
    p.add_argument("--pgd-steps",  type=int,   default=40)
    p.add_argument("--high-kappa", type=float, default=None,
                   help="restrict pool to subjects with κ(0.1) ≥ this value (fast focused test)")
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading data from {args.data_dir} …")
    tcga_ids, X, y = _load_data(args.data_dir)
    print(f"  {len(tcga_ids)} subjects, {X.shape[1]} features, {y.max().item()+1} classes")

    model = MaxNet(input_dim=X.shape[1]).eval()
    print(f"Loading checkpoint: {args.checkpoint}")
    model.load_pathomic_checkpoint(args.checkpoint)
    model.eval()

    with torch.no_grad():
        preds = model(X).argmax(1)
    pool_mask = (preds == y) & (y != TARGET)
    pool_idx  = pool_mask.nonzero(as_tuple=True)[0].tolist()

    # κ per subject
    kappa_by_id = {}
    if os.path.exists(args.geometry):
        with open(args.geometry) as f:
            geo = json.load(f)
        for s in geo.get("subjects", []):
            sid = s.get("id")
            if sid:
                kappa_by_id[sid] = (s.get("kappa_at_lambda") or {}).get("0.1")

    # Optional restriction to high-κ subjects (the only stratum with non-zero ASR).
    if args.high_kappa is not None:
        pool_idx = [i for i in pool_idx
                    if (kappa_by_id.get(tcga_ids[i]) or 0) >= args.high_kappa]
        print(f"  Pool restricted to κ≥{args.high_kappa}: {len(pool_idx)} subjects")
    else:
        print(f"  Pool (correct, non-target): {len(pool_idx)}")

    y_target = torch.tensor([TARGET], dtype=torch.long)

    epsilon_results = []
    for epsilon in args.epsilons:
        print(f"\n  ε = {epsilon}")
        t0 = time.time()
        qp_flips, pgd_flips, kappa_flips, per_subject = [], [], [], []

        for rank, idx in enumerate(pool_idx):
            x_i = X[idx].unsqueeze(0)
            tid = tcga_ids[idx]

            def fwd(z): return model(z)

            # QP (box-constrained second-order)
            x_qp = qp_box_attack(fwd, x_i, y_target, epsilon=epsilon,
                                 lambda_reg=args.lambda_reg, num_steps=args.outer_steps,
                                 data_min=DATA_MIN, data_max=DATA_MAX, qp_maxiter=args.qp_maxiter)
            fq = (int(model(x_qp).argmax(1).item()) == TARGET)

            # KAPPA (unconstrained Newton-CG + L∞ projection) — same budget
            x_k, _ = targeted_attack(fwd, x_i, y_target, epsilon=epsilon,
                                     lambda_reg=args.lambda_reg, num_steps=args.outer_steps,
                                     max_iter=args.qp_maxiter, data_min=DATA_MIN, data_max=DATA_MAX,
                                     return_info=True, damping_mode="lanczos")
            fk = (int(model(x_k).argmax(1).item()) == TARGET)

            # PGD-40 baseline
            x_p = pgd_attack(fwd, x_i, y_target, epsilon=epsilon,
                             num_steps=args.pgd_steps, data_min=DATA_MIN, data_max=DATA_MAX)
            fp = (int(model(x_p).argmax(1).item()) == TARGET)

            qp_flips.append(fq); kappa_flips.append(fk); pgd_flips.append(fp)
            per_subject.append({
                "tcga_id": tid, "true_label": int(y[idx].item()),
                "kappa_at_0.1": kappa_by_id.get(tid),
                "qp_flip": fq, "kappa_flip": fk, "pgd40_flip": fp,
            })

            if (rank + 1) % 25 == 0 or rank == 0:
                n = len(qp_flips)
                print(f"    [{rank+1:3d}/{len(pool_idx)}]  QP={sum(qp_flips)/n:.3f}  "
                      f"KAPPA={sum(kappa_flips)/n:.3f}  PGD40={sum(pgd_flips)/n:.3f}", flush=True)

        n = len(pool_idx)
        qp_asr, kappa_asr, pgd_asr = sum(qp_flips)/n, sum(kappa_flips)/n, sum(pgd_flips)/n
        mc_qp_pgd   = mcnemar_test(qp_flips, pgd_flips)
        mc_qp_kappa = mcnemar_test(qp_flips, kappa_flips)

        eps_entry = {
            "epsilon": epsilon, "n_pool": n,
            "attacks": {
                "qp":     {"asr": round(qp_asr, 6),    "flipped": sum(qp_flips)},
                "kappa":  {"asr": round(kappa_asr, 6), "flipped": sum(kappa_flips)},
                "pgd_40": {"asr": round(pgd_asr, 6),   "flipped": sum(pgd_flips)},
            },
            "mcnemar": {"qp_vs_pgd40": mc_qp_pgd, "qp_vs_kappa": mc_qp_kappa},
            "time_s": round(time.time() - t0, 1),
            "per_subject": per_subject,
        }
        epsilon_results.append(eps_entry)

        print(f"    QP ASR={qp_asr:.4f}  KAPPA ASR={kappa_asr:.4f}  PGD-40 ASR={pgd_asr:.4f}")
        print(f"    McNemar QP vs PGD-40: χ²={mc_qp_pgd['chi2']}  p={mc_qp_pgd['p']}  "
              f"(n01={mc_qp_pgd['n01']}, n10={mc_qp_pgd['n10']})")

        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out.replace(".json", "_partial.json"), "w") as f:
            json.dump({"epsilon_results": epsilon_results}, f, indent=2)

    output = {
        "checkpoint": args.checkpoint,
        "n_pool": len(pool_idx),
        "high_kappa_filter": args.high_kappa,
        "target_class": TARGET,
        "epsilons": args.epsilons,
        "attack_config": {
            "qp_outer_steps": args.outer_steps, "qp_maxiter": args.qp_maxiter,
            "lambda_reg": args.lambda_reg, "damping_mode": "lanczos",
            "pgd_steps": args.pgd_steps, "data_min": DATA_MIN, "data_max": DATA_MAX,
            "threat_model": "L_inf",
            "note": "QP solves min gᵀδ+½δᵀ(H+λI)δ over the L∞∩box directly; KAPPA solves "
                    "unconstrained then projects. Only difference is the projection.",
        },
        "epsilon_results": epsilon_results,
    }
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved → {args.out}")


if __name__ == "__main__":
    main()
