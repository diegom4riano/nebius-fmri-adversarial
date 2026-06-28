import time
import torch
import torch.nn.functional as F

def adaptive_lambda_reg(Hv, v, min_lambda=1e-6):
    # Quociente de Rayleigh por amostra: vᵀHv / vᵀv
    extra = tuple(range(1, v.dim()))
    vHv = (v * Hv).sum(dim=extra)        # (batch,)
    vv  = (v * v).sum(dim=extra)          # (batch,)
    rayleigh = vHv / (vv + 1e-8)          # autovalor de H na direção v
    # λ > max(0, -λ_min) garante (H + λI) ≻ 0 — stays on GPU, no .item() sync
    lambda_val = torch.clamp(-rayleigh.min(), min=0) + min_lambda
    return lambda_val

def conjugate_gradient(A, b, max_iter=100, tol=1e-6):
    x = torch.zeros_like(b)
    r = b.clone()
    d = r.clone()
    extra = tuple(range(1, b.dim()))
    rs_old = (r * r).sum(dim=extra)

    for i in range(max_iter):
        Ad = A(d)
        dAd = (d * Ad).sum(dim=extra)
        alpha = rs_old / (dAd + 1e-8)
        shape = (-1,) + (1,) * (b.dim() - 1)
        alpha = alpha.view(*shape)
        x = x + alpha * d
        r = r - alpha * Ad
        rs_new = (r * r).sum(dim=extra)

        # Check convergence every 50 iters to minimize CPU-GPU syncs (80% fewer syncs)
        if i % 50 == 49 and torch.mean(torch.sqrt(rs_new)).item() < tol:
            break

        beta = rs_new / rs_old
        beta = beta.view(*shape)
        d = r + beta * d
        rs_old = rs_new

    return x

def hessian_vector_product(x, loss_per_sample, v, grad=None):
    if grad is None:
        grad = torch.autograd.grad(
            loss_per_sample,
            x,
            grad_outputs=torch.ones_like(loss_per_sample),
            create_graph=True,
            retain_graph=True,
        )[0]

    Hv = torch.autograd.grad(
        grad,
        x,
        grad_outputs=v,
        retain_graph=True,
    )[0]

    return Hv

def targeted_attack(model, x, y_target, lambda_reg=0.1, epsilon=0.1, max_iter=100, num_steps=5, verbose=False):
    x_adv = x.clone().detach().requires_grad_(True)

    for step in range(num_steps):
        _t0 = time.time()
        # Forward pass
        output = model(x_adv)

        # One-hot encode once: after first forward we know num_classes
        if y_target.dim() == 1:
            y_target = F.one_hot(y_target, num_classes=output.size(1)).float().to(x.device)

        loss_per_sample = F.cross_entropy(output, y_target, reduction='none')

        # Computed once per Newton step; reused across all CG iterations
        cached_grad = torch.autograd.grad(
            loss_per_sample,
            x_adv,
            grad_outputs=torch.ones_like(loss_per_sample),
            create_graph=True,
            retain_graph=True,
        )[0]
        g = cached_grad.detach()

        def A(v):
            Hv = hessian_vector_product(x_adv, loss_per_sample, v, grad=cached_grad)
            return Hv + adaptive_lambda_reg(Hv, v, min_lambda=lambda_reg) * v

        # Resolve (H + λI)δ = -g usando Gradiente Conjugado
        delta = conjugate_gradient(A, -g, max_iter=max_iter)

        if verbose:
            elapsed = time.time() - _t0
            eta = elapsed * (num_steps - step - 1)
            print(f'  [Newton] step {step+1}/{num_steps} | {elapsed:.1f}s | ETA: {eta:.1f}s', flush=True)

        # Aplica perturbação e projeta em L∞
        x_adv = x_adv.detach() + delta
        perturbation = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        craft_x_adv = x + perturbation
        x_adv = craft_x_adv.clone().detach().requires_grad_(True)  # Re-enable gradient tracking for next iteration

    return x_adv


def pgd_attack(model, x, y_target, epsilon=0.1, num_steps=40, step_size=None, verbose=False):
    if step_size is None:
        step_size = epsilon / (num_steps ** 0.5)
    x_adv = x.clone().detach()
    for step in range(num_steps):
        x_adv.requires_grad_(True)
        output = model(x_adv)
        if y_target.dim() == 1:
            _y = F.one_hot(y_target, num_classes=output.size(1)).float().to(x.device)
        else:
            _y = y_target
        loss = F.cross_entropy(output, _y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() - step_size * grad.sign()
        x_adv = x + torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
    return x_adv.detach()