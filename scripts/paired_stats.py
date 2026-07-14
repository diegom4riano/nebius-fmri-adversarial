#!/usr/bin/env python3
"""Offline paired statistics for the attack results (no GPU).

Each attack's per-subject predictions are already saved in every
`output/<run_id>/attack_results.json` under `per_subject` (labels, pred_clean, and each
attack's predictions; a -1 sentinel means the attack did not run). This lets us audit the
claims without re-running anything:

  * Per-attack ASR + Wilson confidence interval (n is small, so a CI is essential).
  * Paired McNemar test, KAPPA vs baseline (same subjects) — the correct test for
    "is KAPPA better than PGD?", far more powerful than comparing marginal ASRs.
  * Bootstrap CI for the ASR difference (resampling subjects).
  * Old metric (pred_clean != target) vs new metric (& labels != target) — how much of
    the gap was just metric bias.

Usage:
  python scripts/paired_stats.py output/<run_a> [output/<run_b> ...]
  python scripts/paired_stats.py output/<run>/attack_results.json
"""
import json
import math
import os
import sys

TARGET = 0                       # targeted-attack target class (flip → 0)
KAPPA_KEY = "newton_cg"          # KAPPA attack key in the JSON
ATTACK_KEYS = ["newton_cg", "pgd_40", "pgd_500", "apgd_ce", "autoattack"]
Z = 1.959963984540054            # z_{0.975} → CI de 95%
N_BOOT = 10000
BOOT_SEED = 20260713


# ----------------------------------------------------------------------------- stats
def wilson_ci(k, n, z=Z):
    """CI de Wilson para proporção k/n (robusto p/ n pequeno e p perto de 0/1)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def mcnemar(succ_a, succ_b):
    """McNemar pareado sobre dois vetores booleanos de sucesso (mesmos sujeitos).

    Retorna (b, c, stat, p_value) onde b = A vence & B falha, c = B vence & A falha.
    Usa a correção de continuidade; p bilateral via qui-quadrado (1 gl) ou binomial
    exato quando b+c é pequeno (< 25), que é o regime deste paper.
    """
    b = sum(1 for a, bb in zip(succ_a, succ_b) if a and not bb)
    c = sum(1 for a, bb in zip(succ_a, succ_b) if bb and not a)
    disc = b + c
    if disc == 0:
        return b, c, 0.0, 1.0
    if disc < 25:
        # binomial exato bilateral: P(X<=min(b,c)) sob p=0.5, *2 (clip em 1)
        k = min(b, c)
        tail = sum(math.comb(disc, i) for i in range(0, k + 1)) / (2 ** disc)
        return b, c, float("nan"), min(1.0, 2 * tail)
    stat = (abs(b - c) - 1) ** 2 / disc          # chi-square with continuity correction
    # p = P(chi2_1 > stat) = erfc(sqrt(stat/2))
    p = math.erfc(math.sqrt(stat / 2.0))
    return b, c, stat, p


def bootstrap_diff_ci(succ_a, succ_b, n_boot=N_BOOT, seed=BOOT_SEED):
    """CI de 95% (percentil) da diferença de ASR (A - B), pareado por sujeito.

    Usa um LCG determinístico (sem numpy) para reprodutibilidade sem dependências.
    """
    n = len(succ_a)
    if n == 0:
        return (float("nan"), float("nan"))
    a = [1 if x else 0 for x in succ_a]
    b = [1 if x else 0 for x in succ_b]
    state = seed & 0xFFFFFFFF
    diffs = []
    for _ in range(n_boot):
        sa = sb = 0
        for _ in range(n):
            state = (1103515245 * state + 12345) & 0x7FFFFFFF
            idx = state % n
            sa += a[idx]
            sb += b[idx]
        diffs.append((sa - sb) / n)
    diffs.sort()
    lo = diffs[int(0.025 * n_boot)]
    hi = diffs[min(n_boot - 1, int(0.975 * n_boot))]
    return (lo, hi)


# ----------------------------------------------------------------------------- pools
def _ran(preds):
    """Um ataque rodou se algum sujeito não tem a sentinela -1."""
    return preds is not None and any(p != -1 for p in preds)


def pools(ps):
    """Retorna (pool_novo, pool_antigo) como listas de índices de sujeito.

    NOVO  : corretamente-classificados non-target  → (pred_clean != T) & (labels != T)
    ANTIGO: só pred_clean != T (inclui o grupo 'falso-Male' que inflava KAPPA/PGD)
    """
    labels, pred_clean = ps["labels"], ps["pred_clean"]
    new = [i for i, (l, pc) in enumerate(zip(labels, pred_clean)) if pc != TARGET and l != TARGET]
    old = [i for i, pc in enumerate(pred_clean) if pc != TARGET]
    return new, old


def success_vec(ps, atk, idxs):
    """Vetor booleano de sucesso (pred == TARGET, i.e. flip para o alvo) no pool."""
    preds = ps[atk]
    return [preds[i] == TARGET for i in idxs]


# ----------------------------------------------------------------------------- report
def analyze_epsilon(entry):
    eps = entry["epsilon"]
    ps = entry["per_subject"]
    ran = [a for a in ATTACK_KEYS if a in ps and _ran(ps[a])]
    new_pool, old_pool = pools(ps)
    n_new, n_old = len(new_pool), len(old_pool)

    print(f"\n{'='*78}\n  ε = {eps}   |   pool NOVO n={n_new}   pool ANTIGO n={n_old}")
    print(f"{'-'*78}")

    # ---- ASR + Wilson CI, métrica NOVA vs ANTIGA
    print(f"  {'ataque':<12} {'ASR_novo':>9} {'IC95_novo':>16} {'ASR_antigo':>11} "
          f"{'Δ(antigo-novo)':>14}")
    asr_new = {}
    for a in ran:
        sn = success_vec(ps, a, new_pool)
        so = success_vec(ps, a, old_pool)
        kn, ko = sum(sn), sum(so)
        pn = kn / n_new if n_new else float("nan")
        po = ko / n_old if n_old else float("nan")
        lo, hi = wilson_ci(kn, n_new)
        asr_new[a] = sn
        print(f"  {a:<12} {pn:>9.3f} [{lo:>6.3f},{hi:>6.3f}] {po:>11.3f} "
              f"{po - pn:>+14.3f}")

    # ---- Paired McNemar + bootstrap: KAPPA vs each baseline (new metric)
    if KAPPA_KEY in asr_new:
        print(f"\n  McNemar pareado (métrica nova) — KAPPA vs baseline:")
        print(f"  {'baseline':<12} {'b(K✓B✗)':>8} {'c(K✗B✓)':>8} {'p-valor':>9} "
              f"{'ΔASR':>8} {'IC95_boot(Δ)':>18}  veredito")
        ka = asr_new[KAPPA_KEY]
        for a in ran:
            if a == KAPPA_KEY:
                continue
            ba = asr_new[a]
            b, c, _stat, p = mcnemar(ka, ba)
            d = (sum(ka) - sum(ba)) / n_new if n_new else float("nan")
            blo, bhi = bootstrap_diff_ci(ka, ba)
            if p < 0.05:
                verd = "KAPPA MELHOR" if b > c else "KAPPA PIOR"
            else:
                verd = "sem dif. signif."
            print(f"  {a:<12} {b:>8} {c:>8} {p:>9.3f} {d:>+8.3f} "
                  f"[{blo:>+6.3f},{bhi:>+6.3f}]  {verd}")


def load_entries(path):
    if os.path.isdir(path):
        for name in ("attack_results.json", "attack_results_ecg.json"):
            p = os.path.join(path, name)
            if os.path.exists(p):
                path = p
                break
    with open(path) as f:
        d = json.load(f)
    return path, d.get("epsilon_results", [])


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1
    for path in argv[1:]:
        resolved, entries = load_entries(path)
        print(f"\n\n########## {resolved} ##########")
        if not entries:
            print("  (sem epsilon_results)")
            continue
        for entry in entries:
            if "per_subject" not in entry:
                print(f"  ε={entry.get('epsilon')}: sem per_subject (run antigo) — pulando")
                continue
            analyze_epsilon(entry)
    print("\n\nNotas: 'b' = sujeitos onde KAPPA acerta e o baseline erra; 'c' = o inverso.")
    print("McNemar usa binomial exato bilateral quando b+c<25 (o regime deste paper).")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
