"""ECG dilated-CNN adversarial evaluation (control experiment, low-κ regime).

Mirrors test_fmri_model.py:
  - deployable: argparse + YAML config + JSON output + per-sample logging
  - repo-local checkpoint (saved_model/best_model.pth), data from --data-dir
  - metric: untargeted robust-ASR (RobustBench standard) over correctly-classified,
    non-Normal samples: success = pred_adv != true_label
  - baselines: KAPPA (targeted->Normal, scored untargeted), PGD-40, PGD-500 (matched
    compute), APGD-CE, AutoAttack(standard); all domain-correct via the [0,1] wrapper
    (the ECG signal is raw amplitude, far outside [0,1], so the libraries' hardcoded
    clamp(0,1) would destroy it — we attack in a normalized box).
  - rigorous κ (Lanczos on matrix-free HVP): κ(H) and κ(H+λI).
"""
import argparse
import contextlib
import json
import os
import time

import numpy as np
import torch


def _maybe_cudnn_deterministic(enabled):
    """Context manager: cuDNN-deterministic ONLY when enabled, else a no-op.

    IMPORTANT (see the note in main()): setting cuDNN determinism GLOBALLY hangs the KAPPA
    HVP (conv double-backward has no fast deterministic algorithm). This
    helper is used to scope determinism to the SINGLE-backward library attacks + the
    forward-only re-scoring — the KAPPA attack runs entirely OUTSIDE it. Gated by config
    (`deterministic_lib_attacks`, default True) so it is a one-flag revert if a specific
    torch/cuDNN build makes even single conv-backward determinism too slow.
    """
    if enabled:
        return torch.backends.cudnn.flags(deterministic=True, benchmark=False)
    return contextlib.nullcontext()
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from model.CNN import CNN
from utils.DataLoader import ECGDataset, ecg_collate_func
from hessian import targeted_attack, pgd_attack

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 4          # Normal(0), AF, Other, Noisy
NORMAL_CLASS = 0


class _ECGAttackModel(torch.nn.Module):
    """Adapter so image-centric attack libs work on 1D ECG signals.

    Attacks see a normalized 4D 'image' z ∈ [0,1] of shape (B,1,1,L) — torchattacks APGD
    and the autoattack package assume 4D image tensors and clamp to [0,1]. We squeeze the
    dummy height dim and denormalize to the real amplitude x = z*(hi-lo)+lo before the CNN,
    so the [0,1] clamp corresponds to the valid box [lo,hi] and eps_z = eps/(hi-lo)."""
    def __init__(self, base, lo, hi):
        super().__init__()
        self.base = base
        self.lo, self.hi = float(lo), float(hi)

    def forward(self, z):
        if z.dim() == 4:           # (B,1,1,L) from the image-centric attack
            z = z.squeeze(2)
        return self.base(z * (self.hi - self.lo) + self.lo)


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--output-dir", default="output")
    p.add_argument("--run-id", default="")
    p.add_argument("--ckpt", default="saved_model/best_model.pth")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke-test", action="store_true",
                   help="synthetic data + short signals for local CPU validation")
    p.add_argument("--smoke-samples", type=int, default=8)
    return p.parse_args()


def get_gpu_memory_mb():
    return round(torch.cuda.max_memory_allocated() / 1e6, 1) if torch.cuda.is_available() else 0.0


def estimate_condition_number_ecg(model, x, lambda_min=1e-6, n_iter=200, tol=1e-3, seed=42):
    """Rigorous spectrum via Lanczos on the matrix-free per-sample HVP (see N5)."""
    from scipy.sparse.linalg import LinearOperator, eigsh
    model.eval()
    x1 = x[:1].clone().detach().requires_grad_(True)
    target1 = torch.zeros(1, dtype=torch.long, device=x.device)
    logits = model(x1)
    loss = F.cross_entropy(logits, target1, reduction="sum")
    grad = torch.autograd.grad(loss, x1, create_graph=True, retain_graph=True)[0]
    shape, d = x1.shape, int(x1.numel())

    def matvec(v_np):
        vv = torch.as_tensor(v_np, dtype=x1.dtype, device=x1.device).view(shape)
        Hv = torch.autograd.grad(grad, x1, grad_outputs=vv, retain_graph=True)[0]
        return Hv.detach().reshape(-1).double().cpu().numpy()

    H = LinearOperator((d, d), matvec=matvec, dtype=np.float64)

    def _ext(which):
        try:
            return float(eigsh(H, k=1, which=which, return_eigenvectors=False,
                               maxiter=n_iter, tol=tol)[0])
        except Exception as e:
            print(f"  [κ] eigsh({which}) failed: {e}", flush=True)
            return None

    mu_max, mu_min, sig_min = _ext("LA"), _ext("SA"), _ext("SM")
    info = {"dim": d, "protocol": {"method": "Lanczos (scipy eigsh) on matrix-free HVP",
                                   "n_probe_inputs": 1, "maxiter": n_iter, "tol": tol,
                                   "lambda_min": lambda_min, "seed": seed}}
    if mu_max is None or mu_min is None:
        info["error"] = "eigsh LA/SA failed"
        return info
    sigma_max = max(abs(mu_max), abs(mu_min))
    lam = max(0.0, -mu_min) + lambda_min
    info.update({"mu_max": mu_max, "mu_min": mu_min, "sigma_max": sigma_max,
                 "sigma_min": None if sig_min is None else abs(sig_min), "lambda_eff": lam,
                 "kappa_H_reg": (mu_max + lam) / (mu_min + lam),
                 # κ at COMMON reference λ_min → comparable across experiments (raw κ(H) is ∞)
                 "kappa_H_reg_at_ref_lambda": {
                     f"{lm:g}": (mu_max + max(0.0, -mu_min) + lm) / (mu_min + max(0.0, -mu_min) + lm)
                     for lm in (1e-6, 1e-3, 1e-1)}})
    if sig_min is not None and abs(sig_min) > 1e-12:
        info["kappa_H"] = sigma_max / abs(sig_min)
        info["kappa_H_rank_deficient"] = abs(sig_min) < 1e-6
    else:
        info["kappa_H"] = None
        info["kappa_H_rank_deficient"] = True
    return info


def compute_untargeted_asr(labels, preds_clean, preds_adv, exclude_class=NORMAL_CLASS):
    """Untargeted robust-ASR over correctly-classified, non-Normal samples.
    pool = (pred_clean == label) & (label != Normal); success = pred_adv != label.
    A sentinel pred of -1 marks a sample the attack did not run on (skipped) — it is
    NOT counted as a flip; if every pred is -1 the attack is reported as skipped (None)."""
    labels = np.asarray(labels); pc = np.asarray(preds_clean); pa = np.asarray(preds_adv)
    if np.all(pa < 0):
        return None, 0, 0                       # attack skipped/failed
    pool = (pc == labels) & (labels != exclude_class)
    if pool.sum() == 0:
        return 0.0, 0, 0
    flipped = int(((pa[pool] != labels[pool]) & (pa[pool] >= 0)).sum())
    return flipped / int(pool.sum()), flipped, int(pool.sum())


def _run_attacks_ecg(model, x, labels, epsilon, cfg, dmin, dmax):
    """One batch, all attacks. Returns (aggregate per-attack, per-sample preds)."""
    model.eval()   # re-assert determinism EACH batch: library attacks (or a crashed one)
                   # can leave the model in train() → dropout on → non-deterministic preds.
    targets0 = torch.zeros_like(labels)          # KAPPA target class Normal(0)
    # Scoped determinism (library attacks + scoring only; KAPPA stays outside). Config-gated.
    det_eval = bool(cfg.get("deterministic_lib_attacks", True))
    span = dmax - dmin
    model_img = _ECGAttackModel(model, dmin, dmax)   # library attacks: 4D image in [0,1]
    to_z4   = lambda t: ((t - dmin) / span).unsqueeze(2)   # (B,1,L) -> (B,1,1,L)
    from_z4 = lambda z4: z4.squeeze(2) * span + dmin       # (B,1,1,L) -> (B,1,L)
    eps_z   = epsilon / span

    with torch.no_grad():
        pred_clean = model(x).argmax(1)

    n = labels.shape[0]
    attack_keys = ["newton_cg", "pgd_40", "pgd_500", "apgd_ce", "autoattack"]
    per_subject = {"labels": labels.detach().cpu().tolist(),
                   "pred_clean": pred_clean.detach().cpu().tolist()}
    for k in attack_keys:
        per_subject[k] = [-1] * n
    res = {}

    def _score(x_adv, key, t0):
        # Deterministic re-forward: cuDNN conv is nondeterministic by default, which made the
        # re-scoring disagree with the library attacks' internal success checks → AutoAttack
        # could re-score BELOW its own components (the "impossible" AutoAttack<APGD-CE). Forcing
        # determinism here is cheap (single forward, no double-backward) and makes ASR stable.
        with torch.no_grad(), _maybe_cudnn_deterministic(det_eval):
            pred = model(x_adv).argmax(1)
        per_subject[key] = pred.detach().cpu().tolist()
        res[key] = {"time_s": time.time() - t0}
        return pred

    # KAPPA — targeted -> Normal(0), projected onto (L∞) ∩ [dmin,dmax]; scored untargeted.
    # return_info logs the per-step μ_min/λ (Lanczos-exact damping → guaranteed PD).
    t0 = time.time()
    x_adv, dmp = targeted_attack(model, x, targets0, lambda_reg=cfg.get("newton_cg_lambda", 1e-6),
                                 epsilon=epsilon, max_iter=cfg.get("newton_cg_cg_iters", 50),
                                 num_steps=cfg.get("newton_cg_outer_steps", 5),
                                 data_min=dmin, data_max=dmax,
                                 lanczos_iters=cfg.get("lanczos_iters", 30), return_info=True)
    _score(x_adv, "newton_cg", t0)
    res["newton_cg"]["damping"] = dmp

    # PGD-40 / PGD-500 — UNTARGETED (torchattacks, normalized [0,1] 4D-image space)
    try:
        import torchattacks
        for key, steps in (("pgd_40", cfg.get("pgd_steps", 40)),
                           ("pgd_500", cfg.get("pgd_matched_budget_steps", 500))):
            t0 = time.time()
            atk = torchattacks.PGD(model_img, eps=eps_z, alpha=2.5 * eps_z / steps, steps=steps)
            with _maybe_cudnn_deterministic(det_eval):
                z_adv = atk(to_z4(x).clone(), labels.long())
            _score(from_z4(z_adv), key, t0)
    except Exception as e:
        print(f"    [PGD] skipped: {e}")

    # APGD-CE — UNTARGETED. Use AutoAttack's OWN apgd-ce (not torchattacks) so it is the exact
    # same implementation that runs as the first component of the standard ensemble below. This
    # guarantees ASR(AutoAttack) >= ASR(APGD-CE) BY CONSTRUCTION (the ensemble runs this same
    # apgd-ce on the full set first, then only adds flips via apgd-t/fab/square). Removes the
    # torchattacks-vs-autoattack cross-implementation mismatch that (with nondeterminism) let
    # AutoAttack re-score below APGD-CE. Falls back to torchattacks if autoattack is unavailable.
    if cfg.get("run_apgd", True):
        try:
            from autoattack import AutoAttack
            t0 = time.time()
            adv_ce = AutoAttack(model_img, norm="Linf", eps=eps_z, version="custom",
                                attacks_to_run=["apgd-ce"], verbose=False)
            with _maybe_cudnn_deterministic(det_eval):
                z_adv = adv_ce.run_standard_evaluation(to_z4(x).clone(), labels.long(), bs=x.shape[0])
            _score(from_z4(z_adv), "apgd_ce", t0)
        except Exception as e:
            print(f"    [APGD-CE via autoattack] failed ({e}); falling back to torchattacks APGD")
            try:
                import torchattacks
                t0 = time.time()
                apgd = torchattacks.APGD(model_img, norm="Linf", eps=eps_z, steps=100, loss="ce")
                with _maybe_cudnn_deterministic(det_eval):
                    z_adv = apgd(to_z4(x).clone(), labels.long())
                _score(from_z4(z_adv), "apgd_ce", t0)
            except Exception as e2:
                print(f"    [APGD-CE] skipped: {e2}")

    # AutoAttack — UNTARGETED, standard ensemble (4 classes → DLR/apgd-t/fab-t valid)
    if cfg.get("run_autoattack", True):
        try:
            from autoattack import AutoAttack
            t0 = time.time()
            version = cfg.get("autoattack_version", "standard")
            adversary = AutoAttack(model_img, norm="Linf", eps=eps_z, version=version, verbose=False)
            # 4-class model → only NUM_CLASSES-1 target classes exist; the default (9) crashes
            # APGD-T/FAB-T ("index out of bounds"). Cap it to what's valid.
            n_tgt = NUM_CLASSES - 1
            if hasattr(adversary, "apgd_targeted"):
                adversary.apgd_targeted.n_target_classes = n_tgt
            if hasattr(adversary, "fab"):
                adversary.fab.n_target_classes = n_tgt
            with _maybe_cudnn_deterministic(det_eval):
                z_adv = adversary.run_standard_evaluation(to_z4(x).clone(), labels.long(), bs=x.shape[0])
            _score(from_z4(z_adv), "autoattack", t0)
        except Exception as e:
            print(f"    [AutoAttack] skipped: {e}")

    return res, per_subject


def _load_data(args, cfg):
    if args.smoke_test:
        rng = np.random.RandomState(args.seed)
        T = 512
        X = [rng.randn(T).astype(np.float32) * 100 for _ in range(args.smoke_samples)]
        y = np.array([i % NUM_CLASSES for i in range(args.smoke_samples)])
        return X, y, float(np.min([x.min() for x in X])), float(np.max([x.max() for x in X]))
    X = np.load(os.path.join(args.data_dir, "raw_data.npy"), allow_pickle=True)
    y = np.load(os.path.join(args.data_dir, "raw_labels.npy"), allow_pickle=True)
    P = np.load(os.path.join(args.data_dir, "random_permutation.npy"), allow_pickle=True)
    X, y = X[P], y[P]
    # global valid box from the FULL dataset (documented, fixed) BEFORE splitting
    dmin = float(min(np.asarray(xi).min() for xi in X))
    dmax = float(max(np.asarray(xi).max() for xi in X))
    mid = int(len(X) * 0.97)     # last 3% held out (~256 samples, matches paper)
    return X[mid:], y[mid:], dmin, dmax


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # NB: do NOT set cudnn.deterministic=True globally — it makes the KAPPA HVP (conv
    # double-backward) prohibitively slow (no fast deterministic conv-backward algorithm),
    # effectively hanging the run. The residual cuDNN-conv nondeterminism only causes a
    # cosmetic "randomized defense" warning from Square; the conclusion (KAPPA weakest) is
    # robust via the deterministic-enough gradient baselines. If a warning-free AutoAttack
    # is needed, wrap ONLY the library-attack calls in torch.backends.cudnn.flags(
    # deterministic=True) — they are single-backward/forward, so determinism there is cheap.

    cfg = {}
    eps_sweep = [2, 10]
    if args.config and os.path.exists(args.config):
        raw = load_config(args.config)
        cfg = raw.get("attack", {})
        eps_sweep = cfg.get("epsilon_sweep", eps_sweep)
    if args.smoke_test:
        eps_sweep = [10]
        cfg = {"newton_cg_outer_steps": 1, "newton_cg_cg_iters": 3, "pgd_steps": 3,
               "pgd_matched_budget_steps": 3, "run_apgd": True, "run_autoattack": True,
               "autoattack_version": "custom", "newton_cg_lambda": 1e-6}
        print("SMOKE TEST MODE — synthetic ECG (no data/checkpoint required)\n")

    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output_dir, run_id)
    os.makedirs(out_dir, exist_ok=True)

    X, y, dmin, dmax = _load_data(args, cfg)
    dmin = float(cfg.get("data_min", dmin))
    dmax = float(cfg.get("data_max", dmax))
    print(f"{len(X)} eval samples | input box [dmin,dmax]=[{dmin:.3f},{dmax:.3f}]")
    loader = DataLoader(ECGDataset(list(X), list(y)), batch_size=args.batch,
                        shuffle=False, collate_fn=ecg_collate_func)

    model = CNN(num_classes=NUM_CLASSES)
    if not args.smoke_test and os.path.exists(args.ckpt):
        sd = torch.load(args.ckpt, map_location=DEVICE)
        sd = {k.removeprefix("module."): v for k, v in sd.items()}
        model.load_state_dict(sd)
        print(f"Loaded checkpoint: {args.ckpt}")
    elif not args.smoke_test:
        print(f"WARNING: checkpoint not found at {args.ckpt} — random weights")
    model = model.to(DEVICE).eval()

    # Clean eval
    preds_all, labels_all = [], []
    with torch.no_grad():
        for data, _lengths, labels in loader:
            data = data.to(DEVICE)
            preds_all.extend(model(data).argmax(1).cpu().tolist())
            labels_all.extend(labels.tolist())
    acc = accuracy_score(labels_all, preds_all)
    bacc = balanced_accuracy_score(labels_all, preds_all)
    f1 = f1_score(labels_all, preds_all, average="macro", zero_division=0)
    print(f"\nClean — acc={acc:.4f} bacc={bacc:.4f} f1={f1:.4f}  (n={len(labels_all)})")

    # κ
    print("\nEstimating condition number κ …", flush=True)
    kappa_info = None
    try:
        for data, _l, _lab in loader:
            kappa_info = estimate_condition_number_ecg(
                model, data.to(DEVICE), lambda_min=cfg.get("newton_cg_lambda", 1e-6), seed=args.seed)
            break
        kH = (kappa_info or {}).get("kappa_H")
        print(f"  κ(H+λI) : {(kappa_info or {}).get('kappa_H_reg', float('nan')):.2f}   "
              f"κ(H) : {'unbounded' if kH is None else f'{kH:.2f}'}")
    except Exception as e:
        print(f"  κ estimation failed: {e}")

    # Adversarial sweep
    eps_results = []
    for eps in eps_sweep:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        print(f"\n  ε = {eps}", flush=True)
        acc_keys = ["newton_cg", "pgd_40", "pgd_500"]
        if cfg.get("run_apgd", True):
            acc_keys.append("apgd_ce")
        if cfg.get("run_autoattack", True):
            acc_keys.append("autoattack")
        subj = {k: [] for k in (["labels", "pred_clean"] + acc_keys)}
        times = {k: 0.0 for k in acc_keys}
        kappa_damping = None
        for data, _lengths, labels in loader:
            data, labels = data.to(DEVICE), labels.to(DEVICE)
            _res, _subj = _run_attacks_ecg(model, data, labels, eps, cfg, dmin, dmax)
            if kappa_damping is None:
                kappa_damping = _res.get("newton_cg", {}).get("damping")
            for k in subj:
                subj[k].extend(_subj.get(k, [-1] * labels.shape[0]))
            for k in acc_keys:
                times[k] += _res.get(k, {}).get("time_s", 0.0)

        entry = {"epsilon": eps, "attacks": {}, "per_subject": subj,
                 "gpu_memory_mb_peak": get_gpu_memory_mb(), "kappa_damping": kappa_damping}
        for k in acc_keys:
            asr, flipped, pool = compute_untargeted_asr(subj["labels"], subj["pred_clean"], subj[k])
            entry["attacks"][k] = {"asr": None if asr is None else round(asr, 6),
                                   "flipped": flipped, "pool_total": pool,
                                   "time_s": round(times[k], 2),
                                   "skipped": asr is None}
            asr_str = "skipped" if asr is None else f"{asr:.4f}"
            print(f"    {k:<12s}: robustASR={asr_str}  (pool={pool})  t={times[k]:.1f}s")
        eps_results.append(entry)

    attack_config = {
        "threat_model": "L_inf", "input_domain_box": [dmin, dmax],
        "metric": "UNTARGETED robust-ASR: pred_adv != true_label over correctly-classified "
                  "non-Normal samples (pool = pred_clean==label & label!=Normal)",
        "domain_fix_N8": "attacks run in [0,1]-normalized space mapping to [dmin,dmax]; "
                         "eps_z = eps/(dmax-dmin); fixes libraries' hardcoded clamp(0,1)",
        "kappa": {"outer_steps_Kncg": cfg.get("newton_cg_outer_steps", 5),
                  "cg_iters_Mcg": cfg.get("newton_cg_cg_iters", 50),
                  "lambda_min": cfg.get("newton_cg_lambda", 1e-6), "targeted_to_Normal": True},
        "pgd": {"steps": [cfg.get("pgd_steps", 40), cfg.get("pgd_matched_budget_steps", 500)],
                "alpha": "2.5*eps/steps", "targeted": False},
        "apgd_ce": {"lib": "autoattack (apgd-ce component; same impl as the ensemble → "
                    "AutoAttack>=APGD-CE by construction)", "n_iter": 100, "targeted": False,
                    "cudnn_deterministic": True},
        "autoattack": {"version": cfg.get("autoattack_version", "standard"), "targeted": False,
                       "cudnn_deterministic": True},
        "kappa_protocol": (kappa_info or {}).get("protocol"),
    }
    output = {"run_id": run_id, "device": str(DEVICE), "checkpoint": args.ckpt,
              "smoke_test": args.smoke_test,
              "clean": {"accuracy": round(acc, 6), "balanced_accuracy": round(bacc, 6),
                        "macro_f1": round(f1, 6), "n_samples": len(labels_all)},
              "condition_number": kappa_info,
              "condition_number_kappa": (kappa_info or {}).get("kappa_H_reg"),
              "attack_config": attack_config,
              "gpu_memory_mb_peak": get_gpu_memory_mb(),
              "epsilon_results": eps_results}
    json_path = os.path.join(out_dir, "attack_results_ecg.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved → {json_path}\n")


if __name__ == "__main__":
    main()
