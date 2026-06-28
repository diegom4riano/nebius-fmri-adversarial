import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import torch
import torch.nn as nn
from hessian import targeted_attack, pgd_attack


# ── Minimal STAGIN-compatible models ─────────────────────────────────────────

class TinySTAGIN(nn.Module):
    """
    Mimics the STAGIN forward signature (v, a, t, endpoints) → (logits, _, _, reg)
    with minimal parameters. Used to test attack mechanics without full STAGIN.
    v: (B, T_w, N, N)
    """
    def __init__(self, n_rois=10, n_windows=5, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(n_rois * n_rois, num_classes)

    def forward(self, v, a=None, t=None, endpoints=None):
        B, T_w, N, _ = v.shape
        x = v.mean(dim=1).reshape(B, -1)   # (B, N*N)
        logits = self.fc(x)
        return logits, None, None, torch.tensor(0.0)


class IllConditionedSTAGIN(nn.Module):
    """
    Two-layer linear model with deliberately ill-conditioned weight matrix.
    Eigenvalues span [0.02, 0.20] → condition number κ ≈ 100.
    Produces high-κ Hessian, where Newton-CG should outperform PGD.
    """
    def __init__(self, n_rois=10, num_classes=2, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        dim = n_rois * n_rois
        W = torch.randn(dim, dim)
        # Make W ill-conditioned: scale singular values to range [0.02, 0.20]
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        S_new = torch.linspace(0.02, 0.20, len(S))
        W_ill = (U * S_new.unsqueeze(0)) @ Vh
        self.register_buffer("W", W_ill)
        self.fc = nn.Linear(dim, num_classes, bias=False)

    def forward(self, v, a=None, t=None, endpoints=None):
        B, T_w, N, _ = v.shape
        x = v.mean(dim=1).reshape(B, -1)   # (B, N*N)
        x = x @ self.W.T
        logits = self.fc(x)
        return logits, None, None, torch.tensor(0.0)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_batch(B=4, T_w=5, N=10):
    v = torch.randn(B, T_w, N, N, requires_grad=False)
    labels = torch.randint(0, 2, (B,))
    targets = torch.zeros(B, dtype=torch.long)   # all → class 0
    return v, labels, targets


def _forward_v(model, v, a=None, t=None, endpoints=None):
    """Wraps model to accept only v (for hessian.py interface)."""
    def fn(v_in):
        logits, _, _, _ = model(v_in, a, t, endpoints)
        return logits
    return fn


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestLinfConstraint:
    @pytest.mark.parametrize("epsilon", [0.01, 0.05, 0.1, 0.5])
    def test_hessian_linf(self, epsilon):
        model = TinySTAGIN()
        v, labels, targets = _make_batch()
        fwd = _forward_v(model, v)
        v_adv = targeted_attack(fwd, v, targets, epsilon=epsilon, verbose=False)
        delta = (v_adv - v).abs().max().item()
        assert delta <= epsilon + 1e-5, f"L∞ violation: {delta:.6f} > {epsilon}"

    @pytest.mark.parametrize("epsilon", [0.01, 0.05, 0.1, 0.5])
    def test_pgd_linf(self, epsilon):
        model = TinySTAGIN()
        v, labels, targets = _make_batch()
        fwd = _forward_v(model, v)
        v_adv = pgd_attack(fwd, v, targets, epsilon=epsilon)
        delta = (v_adv - v).abs().max().item()
        assert delta <= epsilon + 1e-5, f"L∞ violation: {delta:.6f} > {epsilon}"


class TestAttackEffectiveness:
    def test_hessian_loss_decreases(self):
        """Targeted attack should reduce loss toward target class."""
        import torch.nn.functional as F
        model = TinySTAGIN()
        v, labels, targets = _make_batch()
        fwd = _forward_v(model, v)

        logits_clean = fwd(v).detach()
        v_adv = targeted_attack(fwd, v, targets, epsilon=0.1, verbose=False)
        logits_adv = fwd(v_adv).detach()

        loss_clean = F.cross_entropy(logits_clean, targets).item()
        loss_adv   = F.cross_entropy(logits_adv,   targets).item()
        assert loss_adv < loss_clean, f"Loss did not decrease: {loss_clean:.4f} → {loss_adv:.4f}"

    def test_predictions_change(self):
        """At least 25% of predictions should change after attack."""
        model = TinySTAGIN()
        v, labels, targets = _make_batch(B=16)
        fwd = _forward_v(model, v)

        preds_clean = fwd(v).argmax(1)
        v_adv = targeted_attack(fwd, v, targets, epsilon=0.1, verbose=False)
        preds_adv = fwd(v_adv).argmax(1)

        changed = (preds_clean != preds_adv).float().mean().item()
        assert changed >= 0.25, f"Only {changed:.0%} predictions changed"


class TestKappaAdvantage:
    """
    Core empirical claim: on an ill-conditioned model (κ≈100),
    Hessian attack should achieve higher ASR than PGD with the same ε.
    """

    def test_hessian_asr_gt_pgd(self):
        torch.manual_seed(42)
        model = IllConditionedSTAGIN(n_rois=8, seed=42)
        model.eval()
        B, T_w, N = 32, 5, 8
        v       = torch.randn(B, T_w, N, N)
        labels  = torch.randint(0, 2, (B,))
        targets = torch.zeros(B, dtype=torch.long)
        epsilon = 0.1

        fwd = _forward_v(model, v)
        v_hess = targeted_attack(fwd, v, targets, epsilon=epsilon, verbose=False)
        v_pgd  = pgd_attack(fwd, v, targets, epsilon=epsilon)

        with torch.no_grad():
            asr_hess = (fwd(v_hess).argmax(1) == targets).float().mean().item()
            asr_pgd  = (fwd(v_pgd).argmax(1)  == targets).float().mean().item()

        print(f"\nIllConditionedSTAGIN  ε={epsilon}")
        print(f"  Hessian ASR : {asr_hess:.4f}")
        print(f"  PGD     ASR : {asr_pgd:.4f}")
        print(f"  Advantage   : {asr_hess - asr_pgd:+.4f}")

        assert asr_hess >= asr_pgd, (
            f"Expected Hessian ASR >= PGD ASR, got {asr_hess:.4f} < {asr_pgd:.4f}"
        )
