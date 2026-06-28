"""
Testes matemáticos do método de segunda ordem (hessian.py).
Verificam propriedades analíticas conhecidas, sem necessidade de dados reais.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import torch
from hessian import conjugate_gradient, hessian_vector_product, adaptive_lambda_reg


# ── Gradiente Conjugado ──────────────────────────────────────────────────────

class TestConjugateGradient:

    def test_scalar_operator(self):
        """CG resolve Ax=b para A(v)=2v: solução exata é x=b/2"""
        b = torch.ones(3, 1, 8)
        x = conjugate_gradient(lambda v: 2.0 * v, b, max_iter=50)
        assert torch.allclose(x, b / 2.0, atol=1e-4)

    def test_diagonal_operator(self):
        """CG resolve sistema diagonal A=diag(d): x_i = b_i / d_i"""
        d = torch.tensor([2.0, 3.0, 5.0]).view(1, 1, 3)
        b = torch.ones(2, 1, 3)
        x = conjugate_gradient(lambda v: d * v, b, max_iter=50)
        assert torch.allclose(x, b / d, atol=1e-4)

    def test_residual_small_at_convergence(self):
        """Resíduo ‖Ax - b‖ deve ser < 1e-3 após convergência"""
        c = 5.0
        b = torch.randn(4, 1, 12)
        x = conjugate_gradient(lambda v: c * v, b, max_iter=100, tol=1e-6)
        residual = (c * x - b).norm().item()
        assert residual < 1e-3

    def test_batch_independence(self):
        """Cada amostra do batch é resolvida independentemente"""
        b = torch.zeros(2, 1, 4)
        b[0] = 1.0   # amostra 0: b=[1,1,1,1], solução x=[0.5,0.5,0.5,0.5]
        b[1] = 2.0   # amostra 1: b=[2,2,2,2], solução x=[1,1,1,1]
        x = conjugate_gradient(lambda v: 2.0 * v, b, max_iter=50)
        assert torch.allclose(x[0], torch.full((1, 4), 0.5), atol=1e-4)
        assert torch.allclose(x[1], torch.full((1, 4), 1.0), atol=1e-4)


# ── Produto Hessiana-Vetor ───────────────────────────────────────────────────

class TestHessianVectorProduct:

    def test_identity_hessian(self):
        """Para f(x) = ½‖x‖², H = I → Hv deve ser igual a v"""
        x = torch.randn(2, 1, 10).requires_grad_(True)
        loss = 0.5 * (x * x).sum(dim=(1, 2))   # (batch,)
        v = torch.ones_like(x)
        Hv = hessian_vector_product(x, loss, v)
        assert torch.allclose(Hv, v, atol=1e-4)

    def test_scaled_hessian(self):
        """Para f(x) = ½c‖x‖², H = cI → Hv deve ser igual a cv"""
        c = 4.0
        x = torch.randn(2, 1, 10).requires_grad_(True)
        loss = 0.5 * c * (x * x).sum(dim=(1, 2))
        v = torch.randn_like(x)
        Hv = hessian_vector_product(x, loss, v)
        assert torch.allclose(Hv, c * v, atol=1e-3)

    def test_output_shape_preserved(self):
        """Hv deve ter mesmo shape que v"""
        x = torch.randn(3, 1, 20).requires_grad_(True)
        loss = (x * x).sum(dim=(1, 2))
        v = torch.randn_like(x)
        Hv = hessian_vector_product(x, loss, v)
        assert Hv.shape == v.shape

    def test_linearity_in_v(self):
        """HVP é linear em v: H(v1+v2) = Hv1 + Hv2"""
        x = torch.randn(2, 1, 8).requires_grad_(True)
        loss = 0.5 * (x * x).sum(dim=(1, 2))
        v1 = torch.randn_like(x)
        v2 = torch.randn_like(x)

        # Recomputar x com grad para cada chamada
        x1 = x.detach().requires_grad_(True)
        loss1 = 0.5 * (x1 * x1).sum(dim=(1, 2))
        Hv1 = hessian_vector_product(x1, loss1, v1)

        x2 = x.detach().requires_grad_(True)
        loss2 = 0.5 * (x2 * x2).sum(dim=(1, 2))
        Hv2 = hessian_vector_product(x2, loss2, v2)

        x12 = x.detach().requires_grad_(True)
        loss12 = 0.5 * (x12 * x12).sum(dim=(1, 2))
        Hv12 = hessian_vector_product(x12, loss12, v1 + v2)

        assert torch.allclose(Hv12, Hv1 + Hv2, atol=1e-4)


# ── Regularização Adaptativa (Quociente de Rayleigh) ────────────────────────

class TestAdaptiveLambda:

    def test_positive_definite_returns_min_lambda(self):
        """H = I (PD): Rayleigh = 1 > 0 → lambda deve ser min_lambda"""
        v  = torch.randn(2, 1, 10)
        Hv = v.clone()                         # H = I
        lam = adaptive_lambda_reg(Hv, v, min_lambda=1e-6)
        assert lam == pytest.approx(1e-6, rel=1e-2)

    def test_negative_definite_increases_lambda(self):
        """H = -I (ND): Rayleigh = -1 → lambda deve ser > 1 para garantir PD"""
        v  = torch.randn(2, 1, 10)
        Hv = -v.clone()                        # H = -I
        lam = adaptive_lambda_reg(Hv, v, min_lambda=1e-6)
        assert lam > 1.0

    def test_guarantees_positive_definiteness(self):
        """Com λ retornado, vᵀ(H + λI)v > 0 mesmo para H = -2I"""
        v  = torch.randn(2, 1, 10)
        Hv = -2.0 * v                          # H = -2I, Rayleigh = -2
        lam = adaptive_lambda_reg(Hv, v, min_lambda=1e-6)
        # vᵀ(H + λI)v = (-2 + λ)‖v‖²
        quad = (-2.0 + lam) * (v * v).sum().item()
        assert quad > 0

    def test_scales_with_hessian_magnitude(self):
        """λ deve crescer proporcionalmente com a negatividade de H"""
        v = torch.randn(2, 1, 10)
        lam_small = adaptive_lambda_reg(-1.0 * v, v, min_lambda=1e-6)
        lam_large = adaptive_lambda_reg(-5.0 * v, v, min_lambda=1e-6)
        assert lam_large > lam_small
