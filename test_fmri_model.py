import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    classification_report, confusion_matrix, f1_score,
)

from hessian import targeted_attack, pgd_attack
from model.STAGIN import ModelSTAGIN
from utils.fMRILoader import make_loaders

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULTS = dict(
    data_dir    = "data/fmri/hcp/roi",
    ckpt        = "saved_model/best_model_fmri.pth",
    batch       = 32,
    hidden_dim  = 64,
    num_heads   = 1,
    num_layers  = 4,
    sparsity    = 30,
    dropout     = 0.5,
    readout     = "sero",
    cls_token   = "sum",
    seed        = 42,
    output_dir  = "output",
    run_id      = "",
)

EPSILON_SWEEP_DEFAULT = [0.001, 0.005, 0.01, 0.05, 0.1]


class ForwardWrapper(torch.nn.Module):
    """Wraps STAGIN for single-input attack libraries (AutoAttack, torchattacks).

    These libraries call forward() with variable batch sizes n ≤ B.  We pad v to
    the stored batch size B so a and t always match.  The model's train/eval mode is
    set ONCE by the caller (eval mode with only the GRU in train mode — see
    set_attack_mode) and must NOT be changed here: query-based attacks (Square) and
    restart-based ones (APGD) require deterministic forward passes, so dropout/BatchNorm
    stay in eval; only the cuDNN GRU is kept in train() so its backward works.
    """
    def __init__(self, model, a, t, endpoints):
        super().__init__()
        self.model = model
        self._a = a          # [B, T_w, N, N]
        self._t = t          # [T, B, N_rois]  seq-first
        self.endpoints = endpoints
        self._B = a.shape[0]

    def forward(self, v):
        n = v.shape[0]
        B = self._B
        if n < B:
            pad = torch.zeros((B - n,) + v.shape[1:], device=v.device, dtype=v.dtype)
            v_run = torch.cat([v, pad], dim=0)
        else:
            v_run = v
        logits, _, _, _ = self.model(v_run, self._a, self._t, self.endpoints)
        return logits[:n]


class _NormalizedModel(torch.nn.Module):
    """Wrap a model so library attacks operate in a normalized [0,1] box.

    torchattacks and the autoattack package hardcode clamp(0,1); our inputs are NOT in
    [0,1] (STAGIN FC ∈ [-1,1]). We attack z ∈ [0,1] and let the model see the real input
    x = z*(hi-lo)+lo, so the [0,1] clamp corresponds exactly to the valid box [lo,hi] and
    an L∞ radius eps_z = eps/(hi-lo) in z-space equals eps in x-space.
    """
    def __init__(self, base, lo, hi):
        super().__init__()
        self.base = base
        self.lo, self.hi = float(lo), float(hi)

    def forward(self, z):
        return self.base(z * (self.hi - self.lo) + self.lo)


def _infer_input_dim(ckpt_path):
    """Read input_dim from checkpoint's GRU weight (shape [3*hidden, input_dim])."""
    try:
        sd = torch.load(ckpt_path, map_location="cpu")
        for k, v in sd.items():
            if "timestamp_encoder.rnn.weight_ih_l0" in k:
                return int(v.shape[1])  # [3*hidden_dim, input_dim]
    except Exception:
        pass
    return 333  # default: HCP atlas ROIs in this checkpoint


class _SmokeDataset(torch.utils.data.Dataset):
    """Tiny synthetic dataset for local smoke-testing (no real fMRI files needed)."""
    def __init__(self, input_dim, n_samples=8, n_windows=5, n_time=60):
        self.n_samples = n_samples
        self.v = torch.randn(n_samples, n_windows, input_dim, input_dim)
        self.t_seq = torch.randn(n_time, n_samples, input_dim)
        self.labels = torch.tensor([i % 2 for i in range(n_samples)], dtype=torch.long)
        self.endpoints = list(range(10, 10 * n_windows + 1, 10))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return (
            self.v[idx],
            self.v[idx].clone(),
            self.t_seq[:, idx, :],
            self.endpoints,
            self.labels[idx].item(),
        )


def _smoke_collate(batch):
    vs, as_, ts, labels = [], [], [], []
    endpoints = None
    for v, a, t, ep, lbl in batch:
        vs.append(v)
        as_.append(a)
        ts.append(t)
        labels.append(lbl)
        if endpoints is None:
            endpoints = ep
    return (
        torch.stack(vs),
        torch.stack(as_),
        torch.stack(ts, dim=1),
        endpoints,
        torch.tensor(labels, dtype=torch.long),
    )


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--output-dir", default=DEFAULTS["output_dir"])
    p.add_argument("--run-id", default=DEFAULTS["run_id"])
    p.add_argument("--smoke-test", action="store_true",
                   help="Use synthetic data for local testing (no HCP files required)")
    p.add_argument("--smoke-samples", type=int, default=8)
    p.add_argument("--smoke-epsilons", type=float, nargs="+", default=[0.05])
    p.add_argument("--smoke-input-dim", type=int, default=16,
                   help="Tiny input_dim for fast CPU smoke test (uses random weights)")
    # --- Fan-out overrides: run one job per (λ, ε) without generating N config files ---
    p.add_argument("--lambda-reg", type=float, default=None,
                   help="override newton_cg_lambda (λ do damping)")
    p.add_argument("--epsilon", type=float, nargs="+", default=None,
                   help="override epsilon_sweep (ex.: --epsilon 0.001 0.01)")
    p.add_argument("--damping-mode", default=None,
                   choices=["fixed", "lanczos", "adaptive_exact"])
    p.add_argument("--hvp-mode", default=None, choices=["autodiff", "fd"])
    p.add_argument("--only", default=None,
                   help="lista de ataques a rodar p/ fan-out sem duplicar baselines: "
                        "combinação de kappa,pgd,pgd500,apgd,autoattack (ex.: --only kappa)")
    for k, v in DEFAULTS.items():
        if k in ("output_dir", "run_id"):
            continue
        p.add_argument(f"--{k.replace('_','-')}", type=type(v), default=v)
    return p.parse_args()


def get_gpu_memory_mb():
    if torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated() / 1e6, 1)
    return 0.0


def set_attack_mode(model):
    """Deterministic attack/HVP mode (fix for AutoAttack<APGD-CE and ε-varying denominator).

    Run everything in eval() — dropout off, BatchNorm uses running stats (no padding
    cross-contamination) — EXCEPT the timestamp-encoder GRU, kept in train() so cuDNN
    RNN backward works. The GRU is num_layers=1, dropout=0.0, so its train mode is
    deterministic and identical to eval. This makes every forward pass deterministic,
    which query-based (Square) and restart-based (APGD) attacks require.
    """
    model.eval()
    rnn = getattr(getattr(model, "timestamp_encoder", None), "rnn", None)
    if rnn is not None:
        rnn.train()


def estimate_condition_number(model, v, a, t, endpoints, lambda_min=0.1,
                              n_iter=200, tol=1e-3, seed=42):
    """Rigorous loss-Hessian spectrum via matrix-free Lanczos on the per-sample HVP.

    The loss Hessian is generally indefinite, so the meaningful quantities are singular
    values (σ = |eigenvalue|) and, above all, the condition number of the damped
    operator κ(H+λI) that Conjugate Gradient actually
    solves. We compute, for one representative test input (single-sample Hessian):
      μ_max = largest algebraic eigenvalue  (Lanczos, which='LA')
      μ_min = smallest algebraic eigenvalue (Lanczos, which='SA')
      σ_max = max(|μ_max|, |μ_min|)
      λ     = max(0, -μ_min) + λ_min                       (the damping, as in KAPPA)
      κ(H+λI) = (μ_max + λ) / (μ_min + λ)                  (well-defined, finite, CG-relevant)
      σ_min best-effort (Lanczos which='SM'); κ(H) = σ_max/σ_min (may be ~unbounded if
             H is rank-deficient — expected for FC inputs, so reported with a flag).
    Returns a dict with the full protocol logged. Runs in the deterministic attack mode
    (eval + GRU train) since it backprops through the cuDNN GRU.
    """
    from scipy.sparse.linalg import LinearOperator, eigsh

    set_attack_mode(model)
    v1 = v[:1].clone().detach().requires_grad_(True)
    a1, t1 = a[:1], t[:, :1, :]
    target1 = torch.zeros(1, dtype=torch.long, device=v.device)

    logits, _, _, _ = model(v1, a1, t1, endpoints)
    loss = F.cross_entropy(logits, target1, reduction="sum")
    grad = torch.autograd.grad(loss, v1, create_graph=True, retain_graph=True)[0]

    shape = v1.shape
    d = int(v1.numel())

    def matvec(x_np):
        xv = torch.as_tensor(x_np, dtype=v1.dtype, device=v1.device).view(shape)
        Hv = torch.autograd.grad(grad, v1, grad_outputs=xv, retain_graph=True)[0]
        return Hv.detach().reshape(-1).double().cpu().numpy()

    H = LinearOperator((d, d), matvec=matvec, dtype=np.float64)

    def _extreme(which):
        try:
            val = eigsh(H, k=1, which=which, return_eigenvectors=False,
                        maxiter=n_iter, tol=tol)
            return float(val[0])
        except Exception as e:
            print(f"  [κ] eigsh({which}) failed: {e}", flush=True)
            return None

    mu_max = _extreme("LA")   # largest algebraic
    mu_min = _extreme("SA")   # smallest algebraic (most negative)
    sigma_min = _extreme("SM")  # smallest magnitude (best-effort; matrix-free Lanczos is unreliable here)

    info = {"dim": d, "protocol": {"method": "Lanczos (scipy eigsh) on matrix-free HVP",
                                   "n_probe_inputs": 1, "maxiter": n_iter, "tol": tol,
                                   "lambda_min": lambda_min, "seed": seed}}
    if mu_max is None or mu_min is None:
        info["error"] = "eigsh LA/SA failed"
        return info

    sigma_max = max(abs(mu_max), abs(mu_min))
    lam = max(0.0, -mu_min) + lambda_min
    info["mu_max"] = mu_max
    info["mu_min"] = mu_min
    info["sigma_max"] = sigma_max
    info["sigma_min"] = None if sigma_min is None else abs(sigma_min)
    info["lambda_eff"] = lam
    info["kappa_H_reg"] = (mu_max + lam) / (mu_min + lam)   # κ(H+λI): finite, CG-relevant
    # κ at COMMON reference λ_min values → directly comparable across experiments
    # (raw κ(H) is ∞ for both models; κ(H+λI) alone is confounded by each run's λ_min).
    info["kappa_H_reg_at_ref_lambda"] = {
        f"{lm:g}": (mu_max + max(0.0, -mu_min) + lm) / (mu_min + max(0.0, -mu_min) + lm)
        for lm in (1e-6, 1e-3, 1e-1)
    }
    if sigma_min is not None and abs(sigma_min) > 1e-12:
        info["kappa_H"] = sigma_max / abs(sigma_min)
        info["kappa_H_rank_deficient"] = abs(sigma_min) < 1e-6
    else:
        info["kappa_H"] = None
        info["kappa_H_rank_deficient"] = True   # smallest σ ≈ 0 → κ(H) formally unbounded
    return info


def evaluate_clean(loader, model, device):
    model.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for v, a, t, endpoints, labels in loader:
            v, a, t = v.to(device), a.to(device), t.to(device)
            logits, _, _, _ = model(v, a, t, endpoints)
            preds_all.extend(logits.argmax(1).cpu().tolist())
            labels_all.extend(labels.tolist())
    return preds_all, labels_all


def _run_attacks_batch(forward_v, wrapper, v, labels, epsilon, cfg):
    """
    Run all 6 attacks on one batch. True ASR is measured ONLY over subjects that are
    truly Male (label != 0) AND predicted Male on clean input (pred_clean != 0) — the
    standard robustness convention of evaluating on originally-correctly-classified
    examples. On this set, "targeted (flip→0)" and "untargeted from the true label" are
    the SAME objective in binary classification, so KAPPA/PGD (targeted→0) and the
    baselines are compared on equal footing. Returns (aggregate counts, per-subject preds).
    """
    # Re-assert the attack mode before measuring pred_clean: a library attack on a previous
    # batch can leave the model in train() (dropout on) → nondeterministic clean preds and a
    # drifting "correctly-classified" pool across ε. eval() (dropout off) + GRU-train is
    # deterministic; global cuDNN determinism is avoided (it hangs the double-backward HVP).
    set_attack_mode(wrapper.model)
    targets = torch.zeros_like(labels)
    with torch.no_grad():
        pred_clean = forward_v(v).argmax(1)
    # Fair comparison: only correctly-classified Male subjects (true Male AND predicted Male).
    # Removes the false-Male group (label=Female, pred=Male) that otherwise handed KAPPA/PGD
    # free wins relative to attacks driven from the true label.
    not_target = (pred_clean != targets) & (labels != targets)

    def _asr(v_adv):
        with torch.no_grad():
            pred = forward_v(v_adv).argmax(1)
        fl = ((not_target) & (pred == targets)).sum().item()
        return fl, not_target.sum().item(), pred

    res = {}
    n = labels.shape[0]
    # Per-subject predictions: defensive redundancy so any future metric change can be
    # recomputed offline from attack_results.json without a paid re-run. Sentinel -1 marks
    # an attack that did not run for this batch.
    per_subject = {
        "labels":     labels.detach().cpu().tolist(),
        "pred_clean": pred_clean.detach().cpu().tolist(),
    }
    for k in ("newton_cg", "pgd_40", "pgd_500", "autoattack", "apgd_ce"):
        per_subject[k] = [-1] * n

    # Valid input box for the (L∞ ε-ball) ∩ (domain) projection. FC is Pearson correlation
    # ∈ [dmin, dmax] (default [-1,1]). KAPPA/PGD clamp to this box directly; AutoAttack/APGD
    # (which hardcode clamp(0,1)) run in a [0,1] space that maps to it.
    dmin = float(cfg.get("data_min", -1.0))
    dmax = float(cfg.get("data_max", 1.0))
    span = dmax - dmin
    model_z = _NormalizedModel(wrapper, dmin, dmax)
    to_z   = lambda x: (x - dmin) / span
    from_z = lambda z: z * span + dmin
    eps_z  = epsilon / span
    # Ablation flag: whether KAPPA/PGD also project onto the valid box [dmin,dmax].
    # (The baseline wrapper always maps its [0,1] clamp to [dmin,dmax] regardless.)
    # Set newton_pgd_box_clamp: false to test whether the box clamp penalises KAPPA.
    _clamp = cfg.get("newton_pgd_box_clamp", True)
    clamp_lo = dmin if _clamp else None
    clamp_hi = dmax if _clamp else None

    # Newton-CG (KAPPA) — targeted → class 0, projected onto (L∞) ∩ [dmin,dmax].
    # return_info logs the per-step μ_min/λ actually used (Lanczos-exact damping → PD).
    # Gated by run_kappa_attack so a baselines-only fan-out job can skip it.
    if cfg.get("run_kappa_attack", True):
        t0 = time.time()
        v_adv, dmp = targeted_attack(forward_v, v, targets, epsilon=epsilon,
                                     lambda_reg=cfg.get("newton_cg_lambda", 0.1),
                                     num_steps=cfg.get("newton_cg_outer_steps", 5),
                                     max_iter=cfg.get("newton_cg_cg_iters", 50), verbose=False,
                                     data_min=clamp_lo, data_max=clamp_hi,
                                     lanczos_iters=cfg.get("lanczos_iters", 30), return_info=True,
                                     hvp_mode=cfg.get("hvp_mode", "autodiff"), fd_eps=cfg.get("fd_eps", 1e-3),
                                     damping_mode=cfg.get("damping_mode", "lanczos"))
        fl, nt, pred = _asr(v_adv)
        res["newton_cg"] = {"flipped": fl, "nontarget": nt, "time_s": time.time() - t0, "damping": dmp}
        per_subject["newton_cg"] = pred.detach().cpu().tolist()

    # PGD-40 — targeted → class 0. Gated (run_pgd) for kappa-only fan-out jobs.
    if cfg.get("run_pgd", True):
        t0 = time.time()
        v_adv = pgd_attack(forward_v, v, targets, epsilon=epsilon,
                            num_steps=cfg.get("pgd_steps", 40), data_min=clamp_lo, data_max=clamp_hi,
                            num_restarts=cfg.get("pgd_restarts", 1))
        fl, nt, pred = _asr(v_adv)
        res["pgd_40"] = {"flipped": fl, "nontarget": nt, "time_s": time.time() - t0}
        per_subject["pgd_40"] = pred.detach().cpu().tolist()

    # PGD-500 (matched compute budget: 5 outer × 50 CG iters × 2 = 500 backward passes) — targeted → 0
    if cfg.get("run_pgd500", True):
        t0 = time.time()
        v_adv = pgd_attack(forward_v, v, targets, epsilon=epsilon,
                            num_steps=cfg.get("pgd_matched_budget_steps", 500),
                            data_min=clamp_lo, data_max=clamp_hi)
        fl, nt, pred = _asr(v_adv)
        res["pgd_500"] = {"flipped": fl, "nontarget": nt, "time_s": time.time() - t0}
        per_subject["pgd_500"] = pred.detach().cpu().tolist()

    # AutoAttack — UNTARGETED from the true label, run in the normalized [0,1] space so the
    # library's hardcoded clamp(0,1) corresponds to the true valid box [dmin,dmax].
    # Targeted AutoAttack is unavailable for binary (APGD-T/FAB-T use DLR, undefined for 2
    # classes) → custom ensemble apgd-ce+square; on correctly-classified-Male ≡ targeted→0.
    if cfg.get("run_autoattack", True):
        try:
            from autoattack import AutoAttack
            t0 = time.time()
            adversary = AutoAttack(model_z, norm="Linf", eps=eps_z,
                                   version="custom",
                                   attacks_to_run=["apgd-ce", "square"],
                                   verbose=False)
            z_adv = adversary.run_standard_evaluation(to_z(v).clone(), labels.long(),
                                                      bs=v.shape[0])
            v_adv = from_z(z_adv)
            fl, nt, pred = _asr(v_adv)
            res["autoattack"] = {"flipped": fl, "nontarget": nt, "time_s": time.time() - t0}
            per_subject["autoattack"] = pred.detach().cpu().tolist()
        except Exception as e:
            print(f"    [AutoAttack] skipped: {e}")
            res["autoattack"] = {"flipped": 0, "nontarget": 0, "time_s": 0.0, "error": str(e)}

    # APGD-CE standalone — UNTARGETED from the true label. torchattacks APGD does not
    # support targeted mode; on the correctly-classified-Male (binary) set, untargeted
    # from label==1 is identical to targeted→0, so it is compared on equal footing.
    # (C&W removed: it is an L2 attack, not projected into the L∞ ε-ball, so it is not
    # comparable in an ε-indexed L∞ table.)
    if cfg.get("run_apgd", cfg.get("run_cw", True)):
        try:
            import torchattacks
            t0 = time.time()
            apgd = torchattacks.APGD(model_z, norm="Linf", eps=eps_z,
                                      steps=100, loss="ce")   # normalized [0,1] space
            z_adv = apgd(to_z(v).clone(), labels.long())      # untargeted from true label
            v_adv = from_z(z_adv)
            fl, nt, pred = _asr(v_adv)
            res["apgd_ce"] = {"flipped": fl, "nontarget": nt, "time_s": time.time() - t0}
            per_subject["apgd_ce"] = pred.detach().cpu().tolist()
        except Exception as e:
            print(f"    [APGD-CE] skipped: {e}")
            res["apgd_ce"] = {"flipped": 0, "nontarget": 0, "time_s": 0.0, "error": str(e)}

    return res, per_subject


def run_attack_sweep(loader, model, device, epsilons, cfg_attack, out_dir=None):
    partial_path = os.path.join(out_dir, "attack_results_partial.json") if out_dir else None

    # Resume: load previously completed epsilons so we skip them
    completed = {}
    if partial_path and os.path.exists(partial_path):
        with open(partial_path) as f:
            prev = json.load(f)
        for entry in prev.get("epsilon_results", []):
            completed[entry["epsilon"]] = entry
        if completed:
            print(f"  [RESUME] found {len(completed)} completed ε in {partial_path}", flush=True)

    epsilon_results = list(completed.values())

    for epsilon in epsilons:
        if epsilon in completed:
            print(f"\n  ε = {epsilon:.4f}  [SKIP — already done]", flush=True)
            continue

        print(f"\n  ε = {epsilon:.4f}", flush=True)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Only include attacks that actually run (a disabled one must not report ASR=0).
        attack_keys = []
        if cfg_attack.get("run_kappa_attack", True):
            attack_keys.append("newton_cg")
        if cfg_attack.get("run_pgd", True):
            attack_keys.append("pgd_40")
        if cfg_attack.get("run_pgd500", True):
            attack_keys.append("pgd_500")
        if cfg_attack.get("run_autoattack", True):
            attack_keys.append("autoattack")
        if cfg_attack.get("run_apgd", cfg_attack.get("run_cw", True)):
            attack_keys.append("apgd_ce")
        totals = {k: {"flipped": 0, "nontarget": 0, "time_s": 0.0} for k in attack_keys}
        # Per-subject predictions accumulated across batches (test loader is shuffle=False,
        # so subject order is deterministic and consistent across attacks and epsilons).
        subj = {k: [] for k in (["labels", "pred_clean"] + attack_keys)}

        set_attack_mode(model)  # deterministic: eval everywhere, only GRU in train (cuDNN backward)
        kappa_damping = None     # attack's actual per-step μ_min/λ (first batch, for the JSON)
        max_batches = cfg_attack.get("max_batches")   # limit batches for a fast GPU quick-test
        for _bi, (v, a, t, endpoints, labels) in enumerate(loader):
            if max_batches is not None and _bi >= max_batches:
                break
            v, a, t  = v.to(device), a.to(device), t.to(device)
            labels   = labels.to(device)

            def forward_v(v_in):
                logits, _, _, _ = model(v_in, a, t, endpoints)
                return logits

            wrapper = ForwardWrapper(model, a, t, endpoints)
            # mode already set by set_attack_mode(model): eval + GRU-train (do NOT change it here)

            batch_res, batch_subj = _run_attacks_batch(forward_v, wrapper, v, labels, epsilon, cfg_attack)
            if kappa_damping is None:
                kappa_damping = batch_res.get("newton_cg", {}).get("damping")

            for atk, counts in batch_res.items():
                if atk in totals:
                    totals[atk]["flipped"]  += counts["flipped"]
                    totals[atk]["nontarget"]+= counts["nontarget"]
                    totals[atk]["time_s"]   += counts["time_s"]

            for k, arr in batch_subj.items():
                if k in subj:
                    subj[k].extend(arr)

        eps_entry = {"epsilon": epsilon, "attacks": {}}
        for atk in attack_keys:
            nt  = max(totals[atk]["nontarget"], 1)
            asr = totals[atk]["flipped"] / nt
            eps_entry["attacks"][atk] = {
                "asr": round(asr, 6),
                "flipped": totals[atk]["flipped"],
                "nontarget_total": totals[atk]["nontarget"],
                "time_s": round(totals[atk]["time_s"], 2),
            }
            print(f"    {atk:<15s}: ASR={asr:.4f}  t={totals[atk]['time_s']:.1f}s")

        eps_entry["gpu_memory_mb_peak"] = get_gpu_memory_mb()
        eps_entry["per_subject"] = subj
        eps_entry["kappa_damping"] = kappa_damping  # μ_min/λ per outer step (attack's real PD damping)
        epsilon_results.append(eps_entry)

        # Save partial results after every epsilon so job failures don't lose prior work
        if partial_path:
            with open(partial_path, "w") as f:
                json.dump({"epsilon_results": epsilon_results}, f, indent=2)
            print(f"    [partial saved → {partial_path}]", flush=True)

    # Sort by ε so resumed runs (new ε appended out of order) stay monotonic in the JSON.
    return sorted(epsilon_results, key=lambda e: e["epsilon"])


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # NB: intentionally NOT forcing cudnn.deterministic here — the KAPPA HVP is a double
    # backward through the cuDNN GRU, and forcing determinism can break it. STAGIN does not
    # need it: no AutoAttack (no Square/"randomized" check) and the eval+GRU-train mode
    # already yields a stable pool/denominator.

    # Load config YAML (CLI args take precedence over config for explicitly set values)
    cfg_attack    = {}
    epsilon_sweep = EPSILON_SWEEP_DEFAULT
    if args.config and os.path.exists(args.config):
        raw = load_config(args.config)
        cfg_attack    = raw.get("attack", {})
        cfg_model     = raw.get("model", {})
        epsilon_sweep = cfg_attack.get("epsilon_sweep", EPSILON_SWEEP_DEFAULT)
        if args.hidden_dim == DEFAULTS["hidden_dim"]:
            args.hidden_dim = cfg_model.get("hidden_dim", args.hidden_dim)
        if args.sparsity == DEFAULTS["sparsity"]:
            args.sparsity = cfg_model.get("sparsity", args.sparsity)
        if args.batch == DEFAULTS["batch"]:
            args.batch = cfg_attack.get("batch_size", args.batch)

    if args.smoke_test:
        epsilon_sweep = args.smoke_epsilons
        # Override attack steps to minimum for fast CPU validation
        cfg_attack = {
            "newton_cg_outer_steps": 1,
            "newton_cg_cg_iters": 3,
            "pgd_steps": 3,
            "pgd_matched_budget_steps": 3,
            "run_autoattack": True,
            "run_cw": True,
        }
        print("SMOKE TEST MODE — synthetic data (no HCP files required)\n")

    # --- CLI fan-out overrides: applied LAST so they take precedence over config and smoke ---
    if args.lambda_reg is not None:
        cfg_attack["newton_cg_lambda"] = args.lambda_reg
    if args.damping_mode is not None:
        cfg_attack["damping_mode"] = args.damping_mode
    if args.hvp_mode is not None:
        cfg_attack["hvp_mode"] = args.hvp_mode
    if args.epsilon is not None:
        epsilon_sweep = args.epsilon
    if args.only is not None:
        sel = {s.strip() for s in args.only.split(",") if s.strip()}
        cfg_attack["run_kappa_attack"] = "kappa" in sel
        cfg_attack["run_pgd"]          = "pgd" in sel
        cfg_attack["run_pgd500"]       = "pgd500" in sel
        cfg_attack["run_apgd"]         = "apgd" in sel
        cfg_attack["run_autoattack"]   = ("autoattack" in sel) or ("aa" in sel)
        # κ estimation is expensive+deterministic → only when KAPPA runs (reuse across jobs).
        cfg_attack["run_kappa"] = "kappa" in sel

    run_id  = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output_dir, run_id)
    os.makedirs(out_dir, exist_ok=True)

    # Build data loader
    if args.smoke_test:
        # Use tiny input_dim for fast CPU testing — checkpoint not loaded
        input_dim = args.smoke_input_dim
        print(f"Smoke test input_dim={input_dim} (random weights, no checkpoint)")
        ds = _SmokeDataset(input_dim, n_samples=args.smoke_samples)
        test_loader = torch.utils.data.DataLoader(
            ds, batch_size=min(args.smoke_samples, 4),
            collate_fn=_smoke_collate, shuffle=False,
        )
    else:
        roi_ts_path = os.path.join(args.data_dir, "roi_timeseries.npy")
        labels_path = os.path.join(args.data_dir, "labels.npy")
        _, _, test_loader = make_loaders(roi_ts_path, labels_path,
                                         batch_size=args.batch, seed=args.seed)
        input_dim = np.load(roi_ts_path).shape[1]

    # Build model
    model = ModelSTAGIN(
        input_dim   = input_dim,
        hidden_dim  = args.hidden_dim,
        num_classes = 2,
        num_heads   = args.num_heads,
        num_layers  = args.num_layers,
        sparsity    = args.sparsity,
        dropout     = args.dropout,
        cls_token   = args.cls_token,
        readout     = args.readout,
    )
    if not args.smoke_test and os.path.exists(args.ckpt):
        model.load_state_dict(torch.load(args.ckpt, map_location=DEVICE))
        print(f"Loaded checkpoint: {args.ckpt}")
    elif not args.smoke_test:
        print(f"WARNING: checkpoint not found at {args.ckpt} — using random weights")
    model = model.to(DEVICE)

    # Clean evaluation
    preds, labels = evaluate_clean(test_loader, model, DEVICE)
    acc  = accuracy_score(labels, preds)
    bacc = balanced_accuracy_score(labels, preds)
    f1   = f1_score(labels, preds, average="macro", zero_division=0)

    print(f"\n{'='*60}")
    print(f"Clean evaluation — {len(labels)} subjects")
    print(f"  Accuracy     : {acc:.4f}")
    print(f"  Balanced Acc : {bacc:.4f}")
    print(f"  Macro F1     : {f1:.4f}")
    if not args.smoke_test:
        print(classification_report(labels, preds,
                                    target_names=["Female", "Male"], zero_division=0))
        print(confusion_matrix(labels, preds))

    # Condition number estimation (rigorous — Lanczos on matrix-free HVP). Skippable
    # (run_kappa: false) — it's deterministic, so quick-tests can reuse the full run's κ.
    kappa_info = None
    lambda_min_stagin = float(cfg_attack.get("newton_cg_lambda", 0.1))
    if not cfg_attack.get("run_kappa", True):
        print("\nSkipping κ estimation (run_kappa: false)", flush=True)
    else:
        print("\nEstimating condition number κ …", flush=True)
        try:
            for v, a, t, endpoints, _ in test_loader:
                v, a, t = v.to(DEVICE), a.to(DEVICE), t.to(DEVICE)
                kappa_info = estimate_condition_number(model, v, a, t, endpoints,
                                                       lambda_min=lambda_min_stagin, seed=args.seed)
                break
            if kappa_info and "kappa_H_reg" in kappa_info:
                kH = kappa_info.get("kappa_H")
                print(f"  κ(H+λI) : {kappa_info['kappa_H_reg']:.2f}   "
                      f"κ(H) : {'unbounded (rank-deficient)' if kH is None else f'{kH:.2f}'}   "
                      f"σ_max : {kappa_info['sigma_max']:.4g}")
            else:
                print(f"  κ estimation incomplete: {kappa_info}")
        except Exception as e:
            print(f"  κ estimation failed: {e}")

    # Adversarial sweep
    print(f"\nAdversarial sweep  ε={epsilon_sweep}")
    atk_results = run_attack_sweep(test_loader, model, DEVICE, epsilon_sweep, cfg_attack,
                                   out_dir=out_dir)

    # Attack hyperparameters, recorded so the exact configuration is reproducible.
    attack_config = {
        "threat_model": "L_inf",
        "input_domain_box": [float(cfg_attack.get("data_min", -1.0)),
                             float(cfg_attack.get("data_max", 1.0))],
        "domain_fix_N8": "AutoAttack/APGD run in a [0,1]-normalized space that maps to the "
                         "valid box; eps rescaled by 1/(dmax-dmin); KAPPA/PGD clamp to the box. "
                         "Fixes the libraries' hardcoded clamp(0,1) corrupting FC in [-1,1].",
        "metric": "targeted flip -> class 0 (Female), over correctly-classified Male "
                  "(label==1 & pred_clean==1); binary => equivalent to untargeted-from-label",
        "newton_cg_kappa": {"outer_steps_Kncg": cfg_attack.get("newton_cg_outer_steps", 5),
                             "cg_iters_Mcg": cfg_attack.get("newton_cg_cg_iters", 50),
                             "lambda_min": lambda_min_stagin,
                             "lambda_frozen_per_outer_step": True,
                             "targeted": True},
        "pgd": {"step_size_rule": "2.5*eps/num_steps", "pgd_40_steps": cfg_attack.get("pgd_steps", 40),
                "pgd_500_steps": cfg_attack.get("pgd_matched_budget_steps", 500), "targeted": True},
        "autoattack": {"norm": "Linf", "version": "custom",
                       "attacks_to_run": ["apgd-ce", "square"], "targeted": False,
                       "note": "DLR-based targeted variants undefined for 2 classes"},
        "apgd_ce": {"lib": "torchattacks", "steps": 100, "loss": "ce", "targeted": False,
                    "note": "torchattacks APGD has no targeted mode; binary => equiv. to targeted->0"},
        "cw_removed": "C&W is L2, not projected into the L_inf ball => not comparable in an eps-indexed table",
        "kappa_protocol": (kappa_info or {}).get("protocol"),
    }

    # Save JSON
    output = {
        "run_id": run_id,
        "device": str(DEVICE),
        "checkpoint": args.ckpt,
        "smoke_test": args.smoke_test,
        "clean": {
            "accuracy":          round(acc, 6),
            "balanced_accuracy": round(bacc, 6),
            "macro_f1":          round(f1, 6),
            "n_subjects":        len(labels),
        },
        "condition_number": kappa_info,
        "condition_number_kappa": (kappa_info or {}).get("kappa_H_reg"),  # headline = κ(H+λI), finite
        "attack_config": attack_config,
        "gpu_memory_mb_peak": get_gpu_memory_mb(),
        "epsilon_results": atk_results,
    }
    json_path = os.path.join(out_dir, "attack_results.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved → {json_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
