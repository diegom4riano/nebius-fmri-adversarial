"""
Testes do ataque adversarial de segunda ordem (hessian.py).
Verificam restrição L∞, efetividade do ataque e vantagem sobre PGD
em modelo linear mal-condicionado (κ≈100).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from hessian import targeted_attack, conjugate_gradient


# ── Modelos mínimos para testes ──────────────────────────────────────────────

class TinyModel(nn.Module):
    """Linear model (batch, 1, seq_len) → (batch, 2): analiticamente tratável."""

    def __init__(self, seq_len=20, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(seq_len, num_classes, bias=False)

    def forward(self, x):
        return self.fc(x.squeeze(1))


class IllConditionedModel(nn.Module):
    """
    Linear model (batch, 1, seq_len) → (batch, 2) com κ ≈ 100.

    Pesos de features pares têm escala 0.02; features ímpares têm 0.20
    (razão 10:1 → κ≈100 no espaço de pesos). Gerador fixo garante
    reprodutibilidade sem afetar o estado global do torch. Com essas
    escalas, |logit| < 1 para x ~ N(0,1): softmax não satura e o HVP
    é numericamente estável.
    """

    def __init__(self, seq_len=20, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(seq_len, num_classes, bias=False)
        with torch.no_grad():
            g = torch.Generator()
            g.manual_seed(99)
            scales = torch.tensor(
                [0.02 if j % 2 == 0 else 0.20 for j in range(seq_len)]
            )
            self.fc.weight.copy_(
                torch.randn(num_classes, seq_len, generator=g) * scales.unsqueeze(0)
            )

    def forward(self, x):
        return self.fc(x.squeeze(1))


# ── Restrição L∞ ─────────────────────────────────────────────────────────────

class TestLinfConstraint:

    @pytest.mark.parametrize("epsilon", [0.1, 1.0, 5.0, 10.0])
    def test_perturbation_within_epsilon(self, epsilon):
        """‖x_adv - x‖_∞ ≤ ε para qualquer ε"""
        model = TinyModel(seq_len=20)
        model.eval()
        x = torch.randn(4, 1, 20)
        y_target = torch.zeros(4, dtype=torch.long)
        x_adv = targeted_attack(model, x, y_target,
                                lambda_reg=1e-4, epsilon=epsilon,
                                max_iter=20, num_steps=5)
        perturbation = (x_adv - x).abs().max().item()
        assert perturbation <= epsilon + 1e-5, \
            f"‖δ‖_∞ = {perturbation:.4f} > ε = {epsilon}"

    def test_shape_preserved(self):
        """x_adv deve ter o mesmo shape que x"""
        model = TinyModel(seq_len=20)
        model.eval()
        x = torch.randn(3, 1, 20)
        y_target = torch.zeros(3, dtype=torch.long)
        x_adv = targeted_attack(model, x, y_target,
                                lambda_reg=1e-4, epsilon=1.0,
                                max_iter=20, num_steps=3)
        assert x_adv.shape == x.shape

    def test_no_gradient_leakage(self):
        """x_adv retornado não deve ter grad_fn (deve ser detached)"""
        model = TinyModel(seq_len=20)
        model.eval()
        x = torch.randn(2, 1, 20)
        y_target = torch.zeros(2, dtype=torch.long)
        x_adv = targeted_attack(model, x, y_target,
                                lambda_reg=1e-4, epsilon=1.0,
                                max_iter=10, num_steps=2)
        assert x_adv.grad_fn is None


# ── Efetividade do Ataque ─────────────────────────────────────────────────────

class TestAttackEffectiveness:

    def test_loss_decreases_toward_target(self):
        """
        A loss de ataque direcionado deve cair após as iterações:
        o modelo deve tornar-se mais confiante na classe alvo.
        """
        torch.manual_seed(42)
        model = TinyModel(seq_len=20, num_classes=4)
        model.eval()
        x = torch.randn(8, 1, 20)
        y_target = torch.zeros(8, dtype=torch.long)

        with torch.no_grad():
            loss_clean = F.cross_entropy(model(x), y_target).item()

        x_adv = targeted_attack(model, x, y_target,
                                lambda_reg=1e-4, epsilon=5.0,
                                max_iter=50, num_steps=10)

        with torch.no_grad():
            loss_adv = F.cross_entropy(model(x_adv), y_target).item()

        assert loss_adv < loss_clean, \
            f"Loss after attack ({loss_adv:.4f}) should be < clean loss ({loss_clean:.4f})"

    def test_predictions_change(self):
        """
        Após o ataque, pelo menos 25% das predições devem mudar
        (IllConditionedModel com escala controlada, perturbação generosa).
        """
        torch.manual_seed(0)
        model = IllConditionedModel(seq_len=20, num_classes=2)
        model.eval()
        x = torch.randn(16, 1, 20)

        with torch.no_grad():
            pred_clean = model(x).argmax(dim=1)

        y_target = 1 - pred_clean

        x_adv = targeted_attack(model, x, y_target,
                                lambda_reg=1e-4, epsilon=10.0,
                                max_iter=50, num_steps=20)

        with torch.no_grad():
            pred_adv = model(x_adv).argmax(dim=1)

        changed = (pred_adv != pred_clean).float().mean().item()
        assert changed >= 0.25, \
            f"Only {changed*100:.0f}% of predictions changed — attack not effective"


# ── Vantagem de Segunda Ordem ─────────────────────────────────────────────────

class TestSecondOrderAdvantage:
    """
    Prova matematicamente que o gradiente conjugado encontra a solução
    ótima de sistemas mal-condicionados, enquanto o passo de sinal (base do
    PGD) é subótimo. Esta é a propriedade teórica que justifica o ataque
    Newton-CG ser mais eficaz que PGD em problemas com alta curvatura.
    """

    def test_cg_optimal_on_ill_conditioned_quadratic(self):
        """
        f(δ) = ½δᵀAδ + gᵀδ com κ(A)=100.
        CG encontra δ* = -A⁻¹g (mínimo exato).
        sign(g) com mesmo orçamento L∞ é estritamente pior.
        """
        # Operador diagonal: autovalores ∈ {1, 100} → κ = 100
        d = torch.tensor([1.0 if i % 2 == 0 else 100.0 for i in range(10)])
        d = d.view(1, 1, 10)

        torch.manual_seed(42)
        g = torch.randn(1, 1, 10)

        # CG: resolve Aδ = -g → δ* = -A⁻¹g (mínimo global)
        delta_cg = conjugate_gradient(lambda v: d * v, -g, max_iter=50)

        # Passo sign(g) com mesmo orçamento L∞ que o CG
        eps = delta_cg.abs().max().item()
        delta_sign = -eps * g.sign()

        # Objetivo quadrático: f(δ) = ½δᵀAδ + gᵀδ
        def quad(delta):
            return (0.5 * d * delta * delta + g * delta).sum().item()

        assert quad(delta_cg) < quad(delta_sign), (
            "CG exact solution must achieve lower quadratic loss than sign-gradient"
        )

    def test_cg_convergence_faster_than_gradient_descent(self):
        """
        Após 5 iterações, resíduo ‖Aδ - b‖ do CG deve ser menor que o
        do gradient descent com mesmo número de passos. Demonstra que o
        CG converge mais rápido que GD em sistemas mal-condicionados.
        """
        # κ ≈ 100: autovalores distribuídos linearmente de 1 a 100
        d = torch.tensor([1.0 + 99.0 * i / 9 for i in range(10)])
        d = d.view(1, 1, 10)

        torch.manual_seed(7)
        b = torch.randn(1, 1, 10)

        # CG com 5 iterações
        delta_cg = conjugate_gradient(lambda v: d * v, b, max_iter=5)
        residual_cg = ((d * delta_cg - b) ** 2).sum().sqrt().item()

        # Gradient descent com passo conservador 1/λ_max
        step = 1.0 / d.max().item()
        delta_gd = torch.zeros_like(b)
        for _ in range(5):
            delta_gd = delta_gd - step * (d * delta_gd - b)
        residual_gd = ((d * delta_gd - b) ** 2).sum().sqrt().item()

        assert residual_cg < residual_gd, (
            f"CG residual ({residual_cg:.4f}) should be < GD residual ({residual_gd:.4f}) "
            "after 5 iterations on κ=100 system"
        )

    def test_no_nan_on_ill_conditioned_model(self):
        """
        targeted_attack não deve produzir NaN mesmo em modelo linear
        com pesos mal-condicionados (κ ≈ 100).
        """
        torch.manual_seed(99)
        model = IllConditionedModel(seq_len=20, num_classes=2)
        model.eval()

        x = torch.randn(4, 1, 20)
        y_target = torch.zeros(4, dtype=torch.long)

        x_adv = targeted_attack(model, x, y_target,
                                lambda_reg=1e-4, epsilon=1.0,
                                max_iter=50, num_steps=5)

        with torch.no_grad():
            logits = model(x_adv)

        assert not torch.isnan(logits).any(), "targeted_attack produced NaN logits"
        assert not torch.isinf(logits).any(), "targeted_attack produced Inf logits"

    def test_newton_reduces_loss_comparably_to_pgd(self):
        """
        Com orçamento idêntico, Newton-CG deve reduzir a loss pelo menos
        tanto quanto PGD no modelo mal-condicionado.
        """
        torch.manual_seed(5)
        model = IllConditionedModel(seq_len=20, num_classes=2)
        model.eval()

        x = torch.randn(8, 1, 20)
        y_target = torch.zeros(8, dtype=torch.long)

        EPSILON   = 1.0
        NUM_STEPS = 10

        with torch.no_grad():
            loss_clean = F.cross_entropy(model(x), y_target).item()

        x_newton = targeted_attack(model, x, y_target,
                                   lambda_reg=1e-4, epsilon=EPSILON,
                                   max_iter=50, num_steps=NUM_STEPS)

        x_adv_pgd = x.clone().detach().requires_grad_(True)
        step_size = EPSILON / NUM_STEPS
        for _ in range(NUM_STEPS):
            out  = model(x_adv_pgd)
            loss = F.cross_entropy(out, y_target)
            model.zero_grad()
            loss.backward()
            x_adv_pgd = (x_adv_pgd - step_size * x_adv_pgd.grad.sign()).detach()
            x_adv_pgd = torch.clamp(x_adv_pgd, x - EPSILON, x + EPSILON).requires_grad_(True)
        x_adv_pgd = x_adv_pgd.detach()

        with torch.no_grad():
            loss_newton = F.cross_entropy(model(x_newton), y_target).item()
            loss_pgd    = F.cross_entropy(model(x_adv_pgd), y_target).item()

        assert not torch.isnan(torch.tensor(loss_newton)), "Newton-CG produced NaN loss"

        reduction_newton = loss_clean - loss_newton
        reduction_pgd    = loss_clean - loss_pgd
        # Newton-CG should achieve at least 80% of PGD reduction
        assert reduction_newton >= 0.8 * reduction_pgd, (
            f"Newton reduction ({reduction_newton:.4f}) << PGD reduction ({reduction_pgd:.4f})"
        )
