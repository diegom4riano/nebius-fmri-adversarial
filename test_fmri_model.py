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
    the stored batch size B so a and t always match — slicing a/t is not used because
    BatchNorm cross-contamination from sub-batches still causes shape divergences in
    _collate_adjacency.  Model is kept in train() mode throughout so cuDNN RNN backward
    works; eval() must NOT be restored inside forward because the backward pass runs
    after forward() returns and also requires train() mode.
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
        # cuDNN RNN backward requires train() mode.  Do NOT restore eval after this
        # call — the backward pass happens after forward() returns and also needs
        # train() mode.
        self.model.train()
        logits, _, _, _ = self.model(v_run, self._a, self._t, self.endpoints)
        return logits[:n]


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
    for k, v in DEFAULTS.items():
        if k in ("output_dir", "run_id"):
            continue
        p.add_argument(f"--{k.replace('_','-')}", type=type(v), default=v)
    return p.parse_args()


def get_gpu_memory_mb():
    if torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated() / 1e6, 1)
    return 0.0


def estimate_condition_number(model, v, a, t, endpoints, n_vectors=10):
    """Estimate Hessian condition number κ via Rayleigh quotients on random directions."""
    n = min(4, v.shape[0])
    v_probe = v[:n].clone()
    a_probe  = a[:n]
    t_probe  = t[:, :n, :]
    targets  = torch.zeros(n, dtype=torch.long, device=v.device)
    rq_list  = []

    for _ in range(n_vectors):
        v_in = v_probe.clone().detach().requires_grad_(True)
        logits, _, _, _ = model(v_in, a_probe, t_probe, endpoints)
        loss_per_sample = F.cross_entropy(logits, targets, reduction="none")
        rand_dir = torch.randn_like(v_in)
        rand_dir = rand_dir / (rand_dir.norm() + 1e-8)

        grad = torch.autograd.grad(
            loss_per_sample, v_in,
            grad_outputs=torch.ones_like(loss_per_sample),
            create_graph=True, retain_graph=True,
        )[0]
        Hv = torch.autograd.grad(grad, v_in, grad_outputs=rand_dir, retain_graph=False)[0]

        extra = tuple(range(1, rand_dir.dim()))
        rq = (rand_dir * Hv).sum(dim=extra) / ((rand_dir * rand_dir).sum(dim=extra) + 1e-8)
        rq_list.extend(rq.detach().cpu().tolist())

    rq_abs = [abs(r) for r in rq_list if not np.isnan(r)]
    if len(rq_abs) < 2 or max(rq_abs) < 1e-12:
        return float("nan")
    return float(max(rq_abs) / max(min(rq_abs), 1e-12))


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
    Run all 6 attacks on one batch. True ASR: fraction of non-target subjects
    (pred != 0 before attack) whose prediction flips to class 0 after attack.
    Binary classification: targeted (flip→0) == untargeted for class-1 subjects.
    """
    targets = torch.zeros_like(labels)
    with torch.no_grad():
        pred_clean = forward_v(v).argmax(1)
    not_target = (pred_clean != targets)

    def _asr(v_adv):
        with torch.no_grad():
            pred = forward_v(v_adv).argmax(1)
        fl = ((not_target) & (pred == targets)).sum().item()
        return fl, not_target.sum().item()

    res = {}

    # Newton-CG
    t0 = time.time()
    v_adv = targeted_attack(forward_v, v, targets, epsilon=epsilon,
                             num_steps=cfg.get("newton_cg_outer_steps", 5),
                             max_iter=cfg.get("newton_cg_cg_iters", 50), verbose=False)
    fl, nt = _asr(v_adv)
    res["newton_cg"] = {"flipped": fl, "nontarget": nt, "time_s": time.time() - t0}

    # PGD-40
    t0 = time.time()
    v_adv = pgd_attack(forward_v, v, targets, epsilon=epsilon,
                        num_steps=cfg.get("pgd_steps", 40))
    fl, nt = _asr(v_adv)
    res["pgd_40"] = {"flipped": fl, "nontarget": nt, "time_s": time.time() - t0}

    # PGD-500 (matched compute budget: 5 outer × 50 CG iters × 2 = 500 backward passes)
    t0 = time.time()
    v_adv = pgd_attack(forward_v, v, targets, epsilon=epsilon,
                        num_steps=cfg.get("pgd_matched_budget_steps", 500))
    fl, nt = _asr(v_adv)
    res["pgd_500"] = {"flipped": fl, "nontarget": nt, "time_s": time.time() - t0}

    # AutoAttack  (binary: DLR and FAB-T don't work with 2 classes; use apgd-ce + square)
    if cfg.get("run_autoattack", True):
        try:
            from autoattack import AutoAttack
            t0 = time.time()
            adversary = AutoAttack(wrapper, norm="Linf", eps=epsilon,
                                   version="custom",
                                   attacks_to_run=["apgd-ce", "square"],
                                   verbose=False)
            v_adv = adversary.run_standard_evaluation(v.clone(), labels.long(),
                                                       bs=v.shape[0])
            fl, nt = _asr(v_adv)
            res["autoattack"] = {"flipped": fl, "nontarget": nt, "time_s": time.time() - t0}
        except Exception as e:
            print(f"    [AutoAttack] skipped: {e}")
            res["autoattack"] = {"flipped": 0, "nontarget": 0, "time_s": 0.0, "error": str(e)}

    # APGD-CE standalone
    if cfg.get("run_cw", True):
        try:
            import torchattacks
            t0 = time.time()
            apgd = torchattacks.APGD(wrapper, norm="Linf", eps=epsilon,
                                      steps=100, loss="ce")
            v_adv = apgd(v.clone(), labels.long())
            fl, nt = _asr(v_adv)
            res["apgd_ce"] = {"flipped": fl, "nontarget": nt, "time_s": time.time() - t0}
        except Exception as e:
            print(f"    [APGD-CE] skipped: {e}")
            res["apgd_ce"] = {"flipped": 0, "nontarget": 0, "time_s": 0.0, "error": str(e)}

        # C&W L2 (Carlini-Wagner — gold-standard L2 attack)
        try:
            import torchattacks
            t0 = time.time()
            cw = torchattacks.CW(wrapper, c=1, kappa=0, steps=50, lr=0.01)
            v_adv = cw(v.clone(), labels.long())
            fl, nt = _asr(v_adv)
            res["cw_l2"] = {"flipped": fl, "nontarget": nt, "time_s": time.time() - t0}
        except Exception as e:
            print(f"    [C&W L2] skipped: {e}")
            res["cw_l2"] = {"flipped": 0, "nontarget": 0, "time_s": 0.0, "error": str(e)}

    return res


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

        attack_keys = ["newton_cg", "pgd_40", "pgd_500", "autoattack", "apgd_ce", "cw_l2"]
        totals = {k: {"flipped": 0, "nontarget": 0, "time_s": 0.0} for k in attack_keys}

        model.train()  # keep in train mode: cuDNN RNN backward requires it
        for v, a, t, endpoints, labels in loader:
            v, a, t  = v.to(device), a.to(device), t.to(device)
            labels   = labels.to(device)

            def forward_v(v_in):
                logits, _, _, _ = model(v_in, a, t, endpoints)
                return logits

            wrapper = ForwardWrapper(model, a, t, endpoints)
            # do NOT call wrapper.eval(): model must stay in train() for cuDNN RNN backward

            batch_res = _run_attacks_batch(forward_v, wrapper, v, labels, epsilon, cfg_attack)

            for atk, counts in batch_res.items():
                if atk in totals:
                    totals[atk]["flipped"]  += counts["flipped"]
                    totals[atk]["nontarget"]+= counts["nontarget"]
                    totals[atk]["time_s"]   += counts["time_s"]

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
        epsilon_results.append(eps_entry)

        # Save partial results after every epsilon so job failures don't lose prior work
        if partial_path:
            with open(partial_path, "w") as f:
                json.dump({"epsilon_results": epsilon_results}, f, indent=2)
            print(f"    [partial saved → {partial_path}]", flush=True)

    return epsilon_results


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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

    # Condition number estimation
    print("\nEstimating condition number κ …", flush=True)
    kappa = float("nan")
    try:
        for v, a, t, endpoints, _ in test_loader:
            v, a, t = v.to(DEVICE), a.to(DEVICE), t.to(DEVICE)
            kappa = estimate_condition_number(model, v, a, t, endpoints)
            break
        print(f"  κ estimate : {kappa:.2f}")
    except Exception as e:
        print(f"  κ estimation failed: {e}")

    # Adversarial sweep
    print(f"\nAdversarial sweep  ε={epsilon_sweep}")
    atk_results = run_attack_sweep(test_loader, model, DEVICE, epsilon_sweep, cfg_attack,
                                   out_dir=out_dir)

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
        "condition_number_kappa": None if np.isnan(kappa) else round(kappa, 4),
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
