#!/usr/bin/env python3
"""Generate the paper figures from the results: the STAGIN damping sweep, the ECG untargeted
robust-ASR, and the Hessian-spectrum (σ_max vs κ) comparison.

Okabe--Ito colorblind-safe palette, fixed hue order per attack; thin marks, recessive grid,
direct labels. Writes PNGs to figures/.

Usage: python generate_figures.py [OUTPUT_DIR]   (default: output)
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
from analyze_sweep import discover, _load, _eps_entry   # reuse the same extraction

OUT = sys.argv[1] if len(sys.argv) > 1 else "output"
ECG_JSON = "output/20260711_170335/attack_results_ecg.json"
STAGIN_KAPPA_JSON = "output/20260711_170409/attack_results.json"   # tem condition_number
TARGET = 0
os.makedirs("figures", exist_ok=True)

# Okabe--Ito, fixed hue assignment per attack (categorical, never cycled).
C = {
    "KAPPA":     "#D55E00",   # vermillion — the method under test (highlight)
    "PGD-40":    "#0072B2",   # blue
    "PGD-500":   "#56B4E9",   # sky blue
    "APGD-CE":   "#009E73",   # bluish green
    "AutoAttack":"#E69F00",   # orange
}
INK, MUTED, GRID = "#222222", "#666666", "#dddddd"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 11,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
})


# ---------------------------------------------------------------- STAGIN sweep data
def sweep_data():
    """{eps: {'kappa': [(λ_label, λ_real, asr%)], 'PGD-40':%, 'PGD-500':%, 'APGD-CE':%}}"""
    runs = discover(OUT)
    out = {}
    for eps, grp in sorted(runs.items()):
        base_dir, lam_dirs = grp.get("base"), grp.get("lam", {})
        if not base_dir or not lam_dirs:
            continue
        base = _eps_entry(_load(base_dir), eps); psb = base["per_subject"]
        lab, pcb = np.array(psb["labels"]), np.array(psb["pred_clean"])
        bl = {}
        for bk, key in (("PGD-40", "pgd_40"), ("PGD-500", "pgd_500"), ("APGD-CE", "apgd_ce")):
            if key in psb:
                arr = np.array(psb[key]); pool = (lab != TARGET) & (pcb != TARGET)
                bl[bk] = 100.0 * ((arr[pool] == TARGET).mean())
        kap = []
        def _lk(k): return 1e9 if k == "adaptive" else float(k)
        for lk in sorted(lam_dirs, key=_lk):
            ek = _eps_entry(_load(lam_dirs[lk]), eps)
            if not ek or "newton_cg" not in ek.get("per_subject", {}):
                continue
            psk = ek["per_subject"]; pck = np.array(psk["pred_clean"]); nc = np.array(psk["newton_cg"])
            idx = np.where((lab != TARGET) & (pcb != TARGET) & (pck != TARGET))[0]
            asr = 100.0 * (nc[idx] == TARGET).mean() if len(idx) else np.nan
            lam_real = (ek.get("kappa_damping", {}) or {}).get("lambda", [np.nan])[0]
            kap.append((lk, lam_real, asr))
        out[eps] = {"kappa": kap, **bl}
    return out


# ---------------------------------------------------------------- FIG 1: damping sweep
def fig_damping_sweep(sw):
    epss = sorted(sw)
    fig, axes = plt.subplots(1, len(epss), figsize=(4.6 * len(epss), 4.0), squeeze=False)
    for ax, eps in zip(axes[0], epss):
        d = sw[eps]
        lams = [lr if np.isfinite(lr) else float(lbl) for (lbl, lr, _) in d["kappa"]
                if lbl != "adaptive"]
        asrs = [a for (lbl, _, a) in d["kappa"] if lbl != "adaptive"]
        ax.plot(lams, asrs, "-o", color=C["KAPPA"], lw=2, ms=7, label="KAPPA (ours)", zorder=3)
        for (lbl, lr, a) in d["kappa"]:
            if lbl == "adaptive" and np.isfinite(lr):
                ax.plot([lr], [a], "D", color=C["KAPPA"], ms=8, mfc="white", mew=2, zorder=4)
                ax.annotate("adaptive", (lr, a), textcoords="offset points", xytext=(6, 6),
                            fontsize=8, color=C["KAPPA"])
        for bk, ls in (("PGD-40", "--"), ("APGD-CE", ":")):
            if bk in d:
                ax.axhline(d[bk], color=C[bk], lw=2, ls=ls, label=bk, zorder=2)
        best = max([d.get(k, 0) for k in ("PGD-40", "PGD-500", "APGD-CE")] or [0])
        ax.axhspan(0, best, color=GRID, alpha=0.35, zorder=0)
        y1 = ax.get_ylim()[1]
        ax.axvline(0.12, color=MUTED, lw=1, ls="-.", alpha=0.7, zorder=1)
        ax.annotate("|μ_min|≈0.12\n(PD threshold)", (0.12, y1),
                    textcoords="offset points", xytext=(4, -30), fontsize=7.5, color=MUTED)
        ax.set_title(f"ε = {eps}", fontsize=12)
        ax.set_xlabel("damping λ"); ax.set_ylabel("True ASR (%)")
        ax.set_xlim(0, 0.55)
        ax.legend(frameon=False, fontsize=9, loc="best")
    fig.suptitle("STAGIN: KAPPA stays below the best first-order attack at every valid damping λ",
                 fontsize=12.5, y=1.02)
    fig.tight_layout()
    fig.savefig("figures/stagin_damping_sweep.png", bbox_inches="tight")
    plt.close(fig); print("Saved: figures/stagin_damping_sweep.png")


# ---------------------------------------------------------------- FIG 2: ECG robust-ASR
def fig_ecg():
    d = json.load(open(ECG_JSON))
    order = [("KAPPA", "newton_cg"), ("PGD-40", "pgd_40"), ("PGD-500", "pgd_500"),
             ("APGD-CE", "apgd_ce"), ("AutoAttack", "autoattack")]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    n = len(order); epss = d["epsilon_results"]; w = 0.8 / len(epss)
    x = np.arange(n)
    for j, e in enumerate(epss):
        vals = [100.0 * e["attacks"].get(key, {}).get("asr", np.nan) for _, key in order]
        alpha = 0.55 + 0.45 * j / max(1, len(epss) - 1)
        for i, (lbl, _) in enumerate(order):
            ax.bar(x[i] + j * w, vals[i], w * 0.92, color=C[lbl], alpha=alpha, zorder=3,
                   label=(f"ε = {e['epsilon']}" if i == 0 else None))
            ax.annotate(f"{vals[i]:.0f}", (x[i] + j * w, vals[i]), textcoords="offset points",
                        xytext=(0, 2), ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(x + w * (len(epss) - 1) / 2)
    ax.set_xticklabels([lbl for lbl, _ in order])
    ax.set_ylabel("Untargeted robust-ASR (%)")
    ax.set_title("ECG CNN: KAPPA is the weakest attack at both budgets", fontsize=12.5)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig("figures/ecg_robust_asr.png", bbox_inches="tight")
    plt.close(fig); print("Saved: figures/ecg_robust_asr.png")


# ---------------------------------------------------------------- FIG 3: conditioning
def fig_conditioning():
    st = (json.load(open(STAGIN_KAPPA_JSON)).get("condition_number") or {}) if os.path.exists(STAGIN_KAPPA_JSON) else {}
    ec = (json.load(open(ECG_JSON)).get("condition_number") or {}) if os.path.exists(ECG_JSON) else {}
    if not st.get("kappa_H_reg_at_ref_lambda") or not ec.get("kappa_H_reg_at_ref_lambda"):
        print("  (sem condition_number — pulo fig_conditioning)"); return
    lams = ["1e-06", "0.001", "0.1"]
    st_k = [st["kappa_H_reg_at_ref_lambda"][l] for l in lams]
    ec_k = [ec["kappa_H_reg_at_ref_lambda"][l] for l in lams]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 4.0), gridspec_kw={"width_ratios": [1.5, 1]})
    xx = np.arange(len(lams)); w = 0.38
    a1.bar(xx - w / 2, st_k, w, color=C["KAPPA"], zorder=3, label="STAGIN")
    a1.bar(xx + w / 2, ec_k, w, color=C["PGD-40"], zorder=3, label="ECG")
    a1.set_yscale("log"); a1.set_xticks(xx); a1.set_xticklabels(lams)
    a1.set_xlabel("common damping λ"); a1.set_ylabel("κ(H+λI)  (log)")
    a1.set_title("Condition number at a COMMON λ\n(only mildly higher for STAGIN)", fontsize=10.5)
    a1.legend(frameon=False, fontsize=9)
    for i, (s, e) in enumerate(zip(st_k, ec_k)):
        a1.annotate(f"{s:.0f}" if s >= 10 else f"{s:.2f}", (i - w / 2, s),
                    textcoords="offset points", xytext=(0, 2), ha="center", fontsize=7.5)
        a1.annotate(f"{e:.0f}" if e >= 10 else f"{e:.3f}", (i + w / 2, e),
                    textcoords="offset points", xytext=(0, 2), ha="center", fontsize=7.5)
    smax = [st.get("sigma_max", np.nan), ec.get("sigma_max", np.nan)]
    a2.bar([0, 1], smax, 0.6, color=[C["KAPPA"], C["PGD-40"]], zorder=3)
    a2.set_yscale("log"); a2.set_xticks([0, 1]); a2.set_xticklabels(["STAGIN", "ECG"])
    a2.set_ylabel("σ_max  (log, free of λ)")
    ratio = smax[0] / smax[1] if smax[1] else float("nan")
    a2.set_title(f"Curvature MAGNITUDE\n(σ_max ~{ratio:.0f}× larger)", fontsize=10.5)
    for i, v in enumerate(smax):
        a2.annotate(f"{v:.1e}", (i, v), textcoords="offset points", xytext=(0, 2),
                    ha="center", fontsize=7.5)
    fig.suptitle("What separates the models is curvature magnitude (σ_max), not conditioning (κ)",
                 fontsize=12, y=1.03)
    fig.tight_layout()
    fig.savefig("figures/conditioning_sigma_vs_kappa.png", bbox_inches="tight")
    plt.close(fig); print("Saved: figures/conditioning_sigma_vs_kappa.png")


if __name__ == "__main__":
    sw = sweep_data()
    if sw:
        fig_damping_sweep(sw)
    else:
        print("  (sem dados do barrido em", OUT, "— pulo fig 1)")
    if os.path.exists(ECG_JSON):
        fig_ecg()
    fig_conditioning()
    print("Figuras regeneradas.")
