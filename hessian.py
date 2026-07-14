import math
import time
import torch
import torch.nn.functional as F

def adaptive_lambda_reg(Hv, v, min_lambda=1e-6):
    # Per-sample Rayleigh quotient vᵀHv / vᵀv as a cheap eigenvalue estimate along v.
    extra = tuple(range(1, v.dim()))
    vHv = (v * Hv).sum(dim=extra)         # (batch,)
    vv  = (v * v).sum(dim=extra)          # (batch,)
    rayleigh = vHv / (vv + 1e-8)
    # λ = max(0, -min_rayleigh) + min_lambda; kept on-GPU (no .item() sync).
    lambda_val = torch.clamp(-rayleigh.min(), min=0) + min_lambda
    return lambda_val

def conjugate_gradient(A, b, max_iter=100, tol=1e-6, return_residual=False):
    """Matrix-free CG for a fixed SPD operator A, solving A x = b batched per-sample.

    return_residual: also return the final relative residual mean_i ‖r_i‖/‖b_i‖. With a large
    condition number and few iterations CG may not converge, so the residual records whether
    the system was actually solved or only approximated by truncated CG.
    """
    x = torch.zeros_like(b)
    r = b.clone()
    d = r.clone()
    extra = tuple(range(1, b.dim()))
    rs_old = (r * r).sum(dim=extra)
    b_norm = torch.sqrt((b * b).sum(dim=extra)) + 1e-12

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

        # +1e-12 avoids 0/0 once CG has converged (rs_old→0); without it beta becomes NaN and
        # propagates, corrupting the solution. After convergence beta≈0 and x stays stable.
        beta = rs_new / (rs_old + 1e-12)
        beta = beta.view(*shape)
        d = r + beta * d
        rs_old = rs_new

    if return_residual:
        rel = torch.mean(torch.sqrt(rs_old) / b_norm).item()
        return x, rel
    return x

def _lanczos_extreme_eigs(matvec, x0, k=30, reorth=True):
    """k-step Lanczos on a symmetric matrix-free operator (matvec: tensor→tensor), treating
    the whole batched tensor as one vector to estimate the extreme eigenvalues of the
    block-diagonal per-sample Hessian. Returns (min_ritz, max_ritz).

    The smallest Ritz value estimates μ_min, so λ = max(0,-μ_min)+λ_min gives (H+λI) ≻ 0.

    reorth=True does full reorthogonalization (stores the Lanczos basis and re-projects each
    new vector against all previous ones). Without it, unreorthogonalized Lanczos loses
    orthogonality in float32 and yields spurious/duplicated Ritz values. Cost: k basis vectors
    in memory and O(k²·n) dot products — cheap next to the k HVPs.
    """
    q = x0 / (x0.norm() + 1e-12)
    q_prev = torch.zeros_like(x0)
    beta = x0.new_zeros(())
    alphas, betas = [], []
    Q = [q] if reorth else None
    for _ in range(k):
        w = matvec(q)
        if not torch.isfinite(w).all():   # guard: NaN/inf from the operator → stop cleanly
            break
        alpha = (w * q).sum()
        w = w - alpha * q - beta * q_prev
        if reorth:                        # full reorthogonalization against the whole basis
            for qi in Q:
                w = w - (w * qi).sum() * qi
        beta = w.norm()
        alphas.append(alpha)
        if beta.item() < 1e-8:
            break
        q_prev, q = q, w / (beta + 1e-12)
        betas.append(beta)
        if reorth:
            Q.append(q)
    if len(alphas) == 0:
        return float("nan"), float("nan")
    # Solve the tiny (≤k×k) tridiagonal eigenproblem on CPU in float64: cuSOLVER's
    # eigvalsh (GPU) fails on tridiagonals with repeated/clustered Ritz values (which
    # unreorthogonalized Lanczos produces), whereas CPU LAPACK is robust. Cheap (≤30×30).
    m = len(alphas)
    a = torch.tensor([float(x) for x in alphas], dtype=torch.float64)
    b = torch.tensor([float(x) for x in betas[:max(m - 1, 0)]], dtype=torch.float64)
    T = torch.diag(a)
    for i in range(m - 1):          # m×m tridiagonal has m-1 off-diagonals
        T[i, i + 1] = T[i + 1, i] = b[i]
    try:
        evals = torch.linalg.eigvalsh(T)
        return float(evals.min()), float(evals.max())
    except Exception:
        return float(a.min()), float(a.max())   # fallback: diagonal (Rayleigh) bounds


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

def targeted_attack(model, x, y_target, lambda_reg=0.1, epsilon=0.1, max_iter=100, num_steps=5,
                    verbose=False, data_min=None, data_max=None, lanczos_iters=30, return_info=False,
                    hvp_mode="autodiff", fd_eps=1e-3, damping_mode="lanczos", cg_tol=1e-6):
    # Damped Newton-CG L_inf attack. Per outer step: solve (H + λI)δ = -g with CG, apply δ,
    # then project onto the L_inf ε-ball ∩ valid box.
    #
    # damping_mode:
    #   'fixed'          → constant Tikhonov λ = lambda_reg (no eigenvalue estimate). Choose
    #                      lambda_reg >= |μ_min| to keep (H+λI) positive definite.
    #   'lanczos'        → estimate μ_min per step (reorthogonalized Lanczos), λ = max(0,-μ_min)+λ_min.
    #   'adaptive_exact' → 'lanczos' but forces the exact autodiff HVP (ignores hvp_mode).
    # hvp_mode: 'autodiff' (exact double-backward; default) or 'fd' (central difference). In
    #   autodiff mode, a non-finite HVP on a given step falls back to FD for that step only.
    # data_min/data_max: valid input box (e.g. [-1,1] for FC); project onto (L_inf ball) ∩ (box).
    # return_info: also return per-step μ_min/μ_max/λ and the CG residual.
    if damping_mode == "adaptive_exact":
        hvp_mode = "autodiff"                     # exact spectrum for the exact damping
    x_adv = x.clone().detach().requires_grad_(True)
    info = {"mu_min": [], "mu_max": [], "lambda": [], "lambda_min_guarantees_PD": [],
            "cg_rel_residual": [], "hvp_fd_fallback": []}

    for step in range(num_steps):
        _t0 = time.time()
        # Forward pass
        output = model(x_adv)

        # One-hot encode once: after first forward we know num_classes
        if y_target.dim() == 1:
            y_target = F.one_hot(y_target, num_classes=output.size(1)).float().to(x.device)

        loss_per_sample = F.cross_entropy(output, y_target, reduction='none')

        # Computed once per Newton step; reused across all CG iterations. create_graph only
        # needed for the autodiff HVP (double backward); FD mode uses first-order grads only.
        cached_grad = torch.autograd.grad(
            loss_per_sample,
            x_adv,
            grad_outputs=torch.ones_like(loss_per_sample),
            create_graph=(hvp_mode == "autodiff"),
            retain_graph=True,
        )[0]
        g = cached_grad.detach()

        # --- HVP operators ---
        # Autodiff is the exact double-backward HVP. The FD fallback uses a central difference
        # (O(h²), fd_eps default 1e-3), which is far more accurate than a one-sided difference
        # when the curvature is small.
        def _grad_at(xp):
            xp = xp.detach().requires_grad_(True)
            l = F.cross_entropy(model(xp), y_target, reduction="sum")
            return torch.autograd.grad(l, xp)[0].detach()

        def _H_fd(vv):
            h = fd_eps / (vv.detach().flatten().norm() + 1e-12)
            return (_grad_at(x_adv + h * vv) - _grad_at(x_adv - h * vv)) / (2.0 * h)

        def _H_ad(vv):
            return hessian_vector_product(x_adv, loss_per_sample, vv, grad=cached_grad)

        fd_fallbacks = [0]
        if hvp_mode == "fd":
            _H = _H_fd
        else:
            def _H(vv):
                Hv = _H_ad(vv)
                if not torch.isfinite(Hv).all():   # non-finite autodiff HVP → FD for this call only
                    fd_fallbacks[0] += 1
                    return _H_fd(vv)
                return Hv

        # --- Damping λ, fixed once per outer step (independent of the CG search direction) ---
        if damping_mode == "fixed":
            # Constant Tikhonov λ = lambda_reg; no eigenvalue estimate. Assumes
            # lambda_reg >= |μ_min| so that (H+λI) ≻ 0.
            mu_min = mu_max = float("nan")
            lam = lambda_reg
            pd_flag = None
        else:
            # 'lanczos'/'adaptive_exact': estimate μ_min via reorthogonalized Lanczos on the
            # HVP, then λ = max(0,-μ_min)+λ_min.
            gen = torch.Generator(device=x.device).manual_seed(20260711 + step)
            x0 = torch.randn(g.shape, generator=gen, device=x.device, dtype=g.dtype)
            mu_min, mu_max = _lanczos_extreme_eigs(_H, x0, k=lanczos_iters, reorth=True)
            if not math.isfinite(mu_min):
                smax = abs(mu_max) if math.isfinite(mu_max) else 1.0
                mu_min = -smax
                print(f"  [KAPPA] step {step+1}: Lanczos μ_min NaN → fallback λ={smax + lambda_reg:.4f}",
                      flush=True)
            lam = max(0.0, -mu_min) + lambda_reg
            pd_flag = bool(lambda_reg >= -mu_min)
        info["mu_min"].append(mu_min)
        info["mu_max"].append(mu_max)
        info["lambda"].append(lam)
        info["lambda_min_guarantees_PD"].append(pd_flag)

        def A(v):
            return _H(v) + lam * v

        # Solve (H + λI)δ = -g with CG (A is fixed and linear); log the relative residual so a
        # non-converged (approximate) step is visible.
        delta, cg_res = conjugate_gradient(A, -g, max_iter=max_iter, tol=cg_tol, return_residual=True)
        info["cg_rel_residual"].append(cg_res)
        info["hvp_fd_fallback"].append(fd_fallbacks[0])

        if verbose:
            elapsed = time.time() - _t0
            eta = elapsed * (num_steps - step - 1)
            print(f'  [Newton] step {step+1}/{num_steps} | {elapsed:.1f}s | ETA: {eta:.1f}s', flush=True)

        # Apply the step and project onto the L_inf ball (and the valid box, if given).
        x_adv = x_adv.detach() + delta
        perturbation = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        craft_x_adv = x + perturbation
        if data_min is not None or data_max is not None:
            craft_x_adv = torch.clamp(craft_x_adv, min=data_min, max=data_max)
        x_adv = craft_x_adv.clone().detach().requires_grad_(True)  # re-enable grad for next step

    return (x_adv, info) if return_info else x_adv


def pgd_attack(model, x, y_target, epsilon=0.1, num_steps=40, step_size=None, verbose=False,
               data_min=None, data_max=None, num_restarts=1, seed=0):
    # num_restarts>1: random-start PGD (Madry). Restart 0 starts at x; restarts 1..R start from
    # a random point in the L_inf ball. Per sample we keep the restart with the lowest targeted
    # cross-entropy, i.e. the strongest of the restarts.
    if step_size is None:
        # Literature-standard PGD step size (Madry et al. 2018 / TRADES / torchattacks):
        # 2.5·ε/num_steps lets the update reach the L∞ ball boundary from any start within
        # num_steps steps, and scales down proportionally as num_steps grows.
        step_size = 2.5 * epsilon / num_steps

    def _y_of(output):
        if y_target.dim() == 1:
            return F.one_hot(y_target, num_classes=output.size(1)).float().to(x.device)
        return y_target

    shape = (-1,) + (1,) * (x.dim() - 1)
    best_adv, best_loss = None, None
    for r in range(max(1, num_restarts)):
        if r == 0:
            x_adv = x.clone().detach()
        else:
            gen = torch.Generator(device=x.device).manual_seed(seed + r)
            noise = (torch.rand(x.shape, generator=gen, device=x.device, dtype=x.dtype) * 2 - 1) * epsilon
            x_adv = x + noise
            if data_min is not None or data_max is not None:
                x_adv = torch.clamp(x_adv, min=data_min, max=data_max)
            x_adv = x_adv.detach()
        for step in range(num_steps):
            x_adv.requires_grad_(True)
            output = model(x_adv)
            loss = F.cross_entropy(output, _y_of(output))
            grad = torch.autograd.grad(loss, x_adv)[0]
            x_adv = x_adv.detach() - step_size * grad.sign()
            x_adv = x + torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
            # Project onto the valid input box, if given.
            if data_min is not None or data_max is not None:
                x_adv = torch.clamp(x_adv, min=data_min, max=data_max)
        x_adv = x_adv.detach()
        if num_restarts <= 1:
            return x_adv
        with torch.no_grad():
            out = model(x_adv)
            per_loss = F.cross_entropy(out, _y_of(out), reduction="none")
        if best_adv is None:
            best_adv, best_loss = x_adv.clone(), per_loss
        else:
            improve = per_loss < best_loss
            best_adv = torch.where(improve.view(*shape), x_adv, best_adv)
            best_loss = torch.where(improve, per_loss, best_loss)
    return best_adv.detach()