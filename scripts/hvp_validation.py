#!/usr/bin/env python3
"""Validate the loss-Hessian spectrum and the condition number κ.

Checks three things:

  (1) Autodiff vs finite-difference HVP — measures ‖Hv_fd − Hv_ad‖/‖Hv_ad‖ across fd_eps
      and dtype. A large error means the finite-difference HVP corrupts the spectrum that
      feeds the damping, so the exact autodiff HVP should be used.

  (2) Multi-input κ — estimates the spectrum over N inputs and reports the distribution of
      μ_min/μ_max/κ(H+λI) at reference λ, to check that rank-deficiency and the ordering are
      robust rather than a single-sample artifact.

  (3) Damping PD-check — verifies empirically that λ = |μ_min| + λ_min makes (H+λI) ≻ 0
      (smallest Ritz value of H+λI > 0).

Runs on GPU with the real checkpoint; use --smoke-test to validate on CPU without HCP data.
  python scripts/hvp_validation.py --config configs/config.yaml --n-inputs 8
  python scripts/hvp_validation.py --smoke-test         # CPU, synthetic data
"""
import argparse
import os
import statistics
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Reuse the driver's building blocks (importing does not run main — it is under __main__).
from test_fmri_model import (
    DEVICE, DEFAULTS, ForwardWrapper, ModelSTAGIN, estimate_condition_number,
    set_attack_mode, _SmokeDataset, _smoke_collate,
)
from utils.fMRILoader import make_loaders

TARGET = 0


# --------------------------------------------------------------------------- HVP paths
def hvp_autodiff(forward_v, v1, direction, target):
    """Hv exato via double-backward (create_graph)."""
    v1 = v1.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(forward_v(v1), target, reduction="sum")
    grad = torch.autograd.grad(loss, v1, create_graph=True, retain_graph=True)[0]
    Hv = torch.autograd.grad(grad, v1, grad_outputs=direction, retain_graph=True)[0]
    return Hv.detach()


def hvp_fd(forward_v, v1, direction, target, fd_eps, dtype):
    """Hv ≈ (∇loss(v+h·d) − ∇loss(v))/h — o caminho de diferenças finitas do KAPPA.

    dtype permite testar float32 (default do repo, sujeito a cancelamento) vs float64.
    """
    def grad_at(x):
        x = x.clone().detach().to(dtype).requires_grad_(True)
        l = F.cross_entropy(forward_v(x.to(v1.dtype)).to(dtype), target, reduction="sum")
        return torch.autograd.grad(l, x)[0].detach()

    d = direction.to(dtype)
    h = fd_eps / (d.flatten().norm() + 1e-12)
    g0 = grad_at(v1)
    g1 = grad_at(v1 + h.to(v1.dtype) * d.to(v1.dtype))
    return ((g1 - g0) / h).to(v1.dtype)


def compare_hvp(forward_v, v1, target, seed=42):
    """(1) Erro relativo FD-vs-autodiff numa direção aleatória, por fd_eps e dtype."""
    gen = torch.Generator(device=v1.device).manual_seed(seed)
    direction = torch.randn(v1.shape, generator=gen, device=v1.device, dtype=v1.dtype)
    direction = direction / (direction.flatten().norm() + 1e-12)

    Hv_ad = hvp_autodiff(forward_v, v1, direction, target)
    ad_norm = Hv_ad.flatten().norm().item()
    print(f"\n[1] HVP autodiff vs FD  (‖Hv_autodiff‖ = {ad_norm:.4g})")
    print(f"    {'fd_eps':>8} {'dtype':>8} {'rel_err':>12} {'‖Hv_fd‖':>12}")
    rows = []
    for dtype in (torch.float32, torch.float64):
        for fd_eps in (1e-1, 1e-2, 1e-3):
            try:
                Hv_fd = hvp_fd(forward_v, v1, direction, target, fd_eps, dtype)
                diff = (Hv_fd - Hv_ad).flatten().norm().item()
                rel = diff / (ad_norm + 1e-12)
                fd_norm = Hv_fd.flatten().norm().item()
                print(f"    {fd_eps:>8.0e} {str(dtype).split('.')[-1]:>8} "
                      f"{rel:>12.4g} {fd_norm:>12.4g}")
                rows.append((dtype, fd_eps, rel))
            except Exception as e:
                print(f"    {fd_eps:>8.0e} {str(dtype).split('.')[-1]:>8}  FALHOU: {e}")
    if rows:
        best = min(rows, key=lambda r: r[2])
        print(f"    → menor erro: fd_eps={best[1]:.0e} {str(best[0]).split('.')[-1]} "
              f"(rel_err={best[2]:.3g}). Erro alto (≫1e-2) => FD corrompe o espectro.")
    return rows


# --------------------------------------------------------------------------- multi-input κ
def _grad_norm(model, v1, a1, t1, endpoints):
    """‖∇_v CE(loss→class 0)‖ for a single input (gradient-masking check)."""
    v1 = v1.clone().detach().requires_grad_(True)
    out = model(v1, a1, t1, endpoints)
    logits = out[0] if isinstance(out, (tuple, list)) else out
    loss = F.cross_entropy(logits, torch.zeros(1, dtype=torch.long, device=v1.device), reduction="sum")
    g = torch.autograd.grad(loss, v1)[0]
    return float(g.detach().norm().item())


def kappa_multi_input(model, batches, lambda_min, n_inputs, seed):
    """(2) κ rigoroso em N inputs → distribuição. (3) PD-check. (4b) ‖∇‖ p/ gradient-masking."""
    print(f"\n[2] κ multi-input (N={n_inputs}, λ_min={lambda_min})  [+ ‖∇‖ p/ Fase 4b]")
    print(f"    {'#':>3} {'μ_min':>12} {'μ_max':>12} {'‖grad‖':>12} "
          f"{'κ(H+λI@0.1)':>12} {'PD?':>5}")
    mu_mins, mu_maxs, kappas, pd_ok, gnorms = [], [], [], [], []
    count = 0
    for (v, a, t, endpoints, _) in batches:
        v, a, t = v.to(DEVICE), a.to(DEVICE), t.to(DEVICE)
        B = v.shape[0]
        for j in range(B):
            if count >= n_inputs:
                break
            gn = _grad_norm(model, v[j:j + 1], a[j:j + 1], t[:, j:j + 1, :], endpoints)
            info = estimate_condition_number(
                model, v[j:j + 1], a[j:j + 1], t[:, j:j + 1, :], endpoints,
                lambda_min=lambda_min, seed=seed + count,
            )
            mm, mx = info.get("mu_min"), info.get("mu_max")
            if mm is None or mx is None:
                print(f"    {count:>3}  eigsh falhou: {info.get('error')}")
                count += 1
                continue
            kap = info.get("kappa_H_reg")
            # PD-check: λ = |μ_min| + λ_min → smallest eigenvalue of H+λI = μ_min+λ
            lam = max(0.0, -mm) + lambda_min
            pd = (mm + lam) > 0
            mu_mins.append(mm); mu_maxs.append(mx); kappas.append(kap); pd_ok.append(pd); gnorms.append(gn)
            print(f"    {count:>3} {mm:>12.4g} {mx:>12.4g} {gn:>12.4g} "
                  f"{kap:>12.4g} {str(pd):>5}")
            count += 1
        if count >= n_inputs:
            break

    def _summ(name, xs):
        if not xs:
            print(f"    {name}: (vazio)")
            return
        m = statistics.mean(xs)
        sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
        print(f"    {name}: média={m:.4g}  dp={sd:.4g}  min={min(xs):.4g}  max={max(xs):.4g}  (n={len(xs)})")

    print(f"\n    --- resumo ---")
    _summ("μ_min      ", mu_mins)
    _summ("μ_max      ", mu_maxs)
    _summ("‖grad‖     ", gnorms)
    _summ("κ(H+λI@0.1)", kappas)
    if pd_ok:
        print(f"    [3] PD-check (Thm 5.1): λ=|μ_min|+λ_min certifica (H+λI)≻0 em "
              f"{sum(pd_ok)}/{len(pd_ok)} inputs "
              f"{'(SEMPRE — consistente com Thm 5.1)' if all(pd_ok) else '(FALHA em algum — heurística, ver C1)'}")
    # [4b] Gradient-masking: inputs com Hessiana≈0 também têm ‖∇‖≈0?
    if gnorms and mu_maxs:
        tol_h, tol_g = 1e-6, 1e-4
        flat_H = [i for i, mx in enumerate(mu_maxs) if abs(mx) < tol_h]
        both0 = [i for i in flat_H if gnorms[i] < tol_g]
        print(f"    [4b] gradient-masking: {len(flat_H)}/{len(mu_maxs)} inputs com Hessiana≈0 "
              f"(|μ_max|<{tol_h}); destes, {len(both0)} têm ‖∇‖<{tol_g} também.")
        if flat_H:
            print(f"         ‖∇‖ nos inputs de Hessiana≈0: "
                  f"{[round(gnorms[i], 6) for i in flat_H]}")
            print("         → ‖∇‖≈0 ⇒ masking p/ AMBAS as ordens (1ª e 2ª); "
                  "‖∇‖>0 ⇒ só a curvatura degenerou (PGD ainda tem sinal).")
    return dict(mu_min=mu_mins, mu_max=mu_maxs, kappa=kappas, pd_ok=pd_ok)


# --------------------------------------------------------------------------- setup
def build(args):
    if args.smoke_test:
        print("SMOKE TEST — dados sintéticos, pesos aleatórios (CPU ok)\n")
        input_dim = args.smoke_input_dim
        ds = _SmokeDataset(input_dim, n_samples=max(args.n_inputs, 4))
        loader = torch.utils.data.DataLoader(
            ds, batch_size=min(args.n_inputs, 4), collate_fn=_smoke_collate, shuffle=False)
    else:
        roi = os.path.join(args.data_dir, "roi_timeseries.npy")
        lab = os.path.join(args.data_dir, "labels.npy")
        _, _, loader = make_loaders(roi, lab, batch_size=args.batch, seed=args.seed)
        input_dim = np.load(roi).shape[1]

    model = ModelSTAGIN(input_dim=input_dim, hidden_dim=args.hidden_dim, num_classes=2,
                        num_heads=args.num_heads, num_layers=args.num_layers,
                        sparsity=args.sparsity, dropout=args.dropout,
                        cls_token=args.cls_token, readout=args.readout)
    if not args.smoke_test and os.path.exists(args.ckpt):
        model.load_state_dict(torch.load(args.ckpt, map_location=DEVICE))
        print(f"Checkpoint: {args.ckpt}")
    model = model.to(DEVICE)
    set_attack_mode(model)
    return model, loader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--data-dir", default=DEFAULTS["data_dir"])
    p.add_argument("--ckpt", default=DEFAULTS["ckpt"])
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--hidden-dim", type=int, default=DEFAULTS["hidden_dim"])
    p.add_argument("--num-heads", type=int, default=DEFAULTS["num_heads"])
    p.add_argument("--num-layers", type=int, default=DEFAULTS["num_layers"])
    p.add_argument("--sparsity", type=int, default=DEFAULTS["sparsity"])
    p.add_argument("--dropout", type=float, default=DEFAULTS["dropout"])
    p.add_argument("--readout", default=DEFAULTS["readout"])
    p.add_argument("--cls-token", default=DEFAULTS["cls_token"])
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    p.add_argument("--lambda-min", type=float, default=0.1)
    p.add_argument("--n-inputs", type=int, default=8)
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--smoke-input-dim", type=int, default=20)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model, loader = build(args)

    # Take the first input for the HVP comparison (single sample).
    v, a, t, endpoints, _ = next(iter(loader))
    v, a, t = v.to(DEVICE), a.to(DEVICE), t.to(DEVICE)
    forward_v = ForwardWrapper(model, a[:1], t[:, :1, :], endpoints)
    v1 = v[:1]
    target = torch.full((1,), TARGET, dtype=torch.long, device=DEVICE)

    compare_hvp(forward_v, v1, target, seed=args.seed)
    kappa_multi_input(model, iter(loader), args.lambda_min, args.n_inputs, args.seed)
    print("\nOK — validação de κ concluída.")


if __name__ == "__main__":
    main()
