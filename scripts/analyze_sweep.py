#!/usr/bin/env python3
"""Analyze the damping sweep: KAPPA(λ) vs baselines, paired.

Each KAPPA-only job (--only kappa) saved only `newton_cg` in per_subject; the baselines job
(--only pgd,pgd500,apgd) saved pgd_40/pgd_500/apgd_ce. Per ε, this script:
  * matches jobs by subject (same loader order), using the intersection pool — subjects
    correctly classified as non-target in BOTH jobs (the cuDNN GRU differs on ~2/216 preds
    across machines; the intersection removes that noise without bias);
  * reports, per λ: ASR(KAPPA) + Wilson CI, the actual λ and the CG residual (did it
    converge?), and a paired McNemar test KAPPA-vs-{PGD-40,PGD-500,APGD-CE} (+ ΔASR and
    bootstrap CI).

Usage:  python scripts/analyze_sweep.py output    # scans run dirs *_base_* and *_kappa_*
"""
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paired_stats import wilson_ci, mcnemar, bootstrap_diff_ci

TARGET = 0


def _load(run_dir):
    for name in ("attack_results.json", "attack_results_partial.json"):
        p = os.path.join(run_dir, name)
        if os.path.exists(p):
            return json.load(open(p))
    return None


def _eps_entry(d, eps):
    for e in d.get("epsilon_results", []):
        if abs(float(e["epsilon"]) - eps) < 1e-12:
            return e
    return None


def discover(out_dir):
    """Agrupa run dirs por ε → {'base': dir, lambdas: {λ|'adaptive': dir}}."""
    runs = {}
    for d in glob.glob(os.path.join(out_dir, "*")):
        b = os.path.basename(d)
        mb = re.search(r"_base_e([0-9.]+)_", b)
        mk = re.search(r"_kappa_l([0-9.]+)_e([0-9.]+)_", b)
        ma = re.search(r"_kappa_adaptive_e([0-9.]+)_", b)
        if mb:
            runs.setdefault(float(mb.group(1)), {}).setdefault("base", d)
        elif mk:
            runs.setdefault(float(mk.group(2)), {}).setdefault("lam", {})[mk.group(1)] = d
        elif ma:
            runs.setdefault(float(ma.group(1)), {}).setdefault("lam", {})["adaptive"] = d
    return runs


def analyze_eps(eps, group):
    base_dir = group.get("base")
    lam_dirs = group.get("lam", {})
    if not base_dir or not lam_dirs:
        print(f"\n[ε={eps}] incompleto (base={bool(base_dir)}, λ jobs={len(lam_dirs)}) — pulando")
        return
    base = _eps_entry(_load(base_dir), eps)
    psb = base["per_subject"]
    lab = np.array(psb["labels"]); pcb = np.array(psb["pred_clean"])
    baselines = {k: np.array(psb[k]) for k in ("pgd_40", "pgd_500", "apgd_ce") if k in psb}

    print(f"\n{'='*82}\n  ε = {eps}   (baselines: {base_dir.split('/')[-1]})")
    print(f"{'-'*82}")
    print(f"  {'λ':>9} {'λ_real':>8} {'cg_res':>7} {'ASR_K':>7} {'IC95':>15} "
          f"{'melhor base':>12} {'McNemar p':>10} {'veredito':>16}")

    def _lam_key(k):
        return (1e9 if k == "adaptive" else float(k))

    for lk in sorted(lam_dirs, key=_lam_key):
        ek = _eps_entry(_load(lam_dirs[lk]), eps)
        if ek is None or "newton_cg" not in ek.get("per_subject", {}):
            print(f"  {lk:>9}  (sem dados)")
            continue
        psk = ek["per_subject"]
        pck = np.array(psk["pred_clean"]); nc = np.array(psk["newton_cg"])
        # intersection pool: correctly-classified non-target in BOTH jobs
        pool = (lab != TARGET) & (pcb != TARGET) & (pck != TARGET)
        idx = np.where(pool)[0]
        n = len(idx)
        sk = (nc[idx] == TARGET)
        kap_asr = sk.mean() if n else float("nan")
        lo, hi = wilson_ci(int(sk.sum()), n)
        dmp = ek.get("kappa_damping", {}) or {}
        lam_real = (dmp.get("lambda") or [float("nan")])[0]
        cg_res = np.mean(dmp.get("cg_rel_residual") or [float("nan")])
        # best baseline (highest ASR on the same pool) and McNemar against it
        best_b, best_asr = None, -1
        for bk, arr in baselines.items():
            a = (arr[idx] == TARGET).mean()
            if a > best_asr:
                best_asr, best_b = a, bk
        sb = (baselines[best_b][idx] == TARGET)
        b, c, _stat, p = mcnemar(sk.tolist(), sb.tolist())
        d = kap_asr - best_asr
        verd = ("KAPPA>base" if (p < 0.05 and b > c) else
                "KAPPA<base" if (p < 0.05 and c > b) else "sem dif.")
        print(f"  {lk:>9} {lam_real:>8.3g} {cg_res:>7.2f} {kap_asr:>7.3f} "
              f"[{lo:>5.3f},{hi:>5.3f}] {best_b:>12} {p:>10.3f} {verd:>16}")
    # baselines de referência no pool base inteiro
    poolb = (lab != TARGET) & (pcb != TARGET)
    print(f"  ---- baselines (pool base n={int(poolb.sum())}): " +
          "  ".join(f"{k}={ (v[poolb]==TARGET).mean():.3f}" for k, v in baselines.items()))


def main(argv):
    out_dir = argv[1] if len(argv) > 1 else "output"
    runs = discover(out_dir)
    if not runs:
        print(f"nenhum run *_base_*/_kappa_* em {out_dir}")
        return 1
    for eps in sorted(runs):
        analyze_eps(eps, runs[eps])
    print("\nNota: pool de interseção (correto em ambos os jobs). McNemar exato bilateral (b+c<25).")
    print("cg_res = resíduo relativo médio do CG (≈0 convergiu; ≫0 = passo de Newton só aproximado).")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
