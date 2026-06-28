import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

os.makedirs("figures", exist_ok=True)

with open("output/attack_results.json") as f:
    data = json.load(f)

results = data["epsilon_results"]
epsilons = [e["epsilon"] for e in results]

ATTACKS = {
    "newton_cg":  ("KAPPA (ours)",  "#e63946", "-",  "o",  2.5),
    "autoattack": ("AutoAttack",     "#457b9d", "--", "s",  1.8),
    "apgd_ce":    ("APGD-CE",        "#2a9d8f", "--", "^",  1.8),
    "pgd_40":     ("PGD-40",         "#f4a261", ":",  "D",  1.6),
    "pgd_500":    ("PGD-500",        "#e9c46a", ":",  "v",  1.6),
    "cw_l2":      ("C&W L2",         "#adb5bd", "-.", "x",  1.4),
}

# ── Figure 1: ASR vs Epsilon (main result) ──────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5.5))

for key, (label, color, ls, marker, lw) in ATTACKS.items():
    asrs = [e["attacks"][key]["asr"] * 100 for e in results]
    ax.plot(epsilons, asrs, color=color, linestyle=ls,
            marker=marker, linewidth=lw, markersize=7,
            label=label, zorder=3 if key == "newton_cg" else 2)

ax.set_xscale("log")
ax.set_xlabel("Epsilon (L∞)", fontsize=12)
ax.set_ylabel("True Attack Success Rate (%)", fontsize=12)
ax.set_title("KAPPA vs State-of-the-Art Attacks on STAGIN fMRI\n(κ = 178,695  |  84 Male test subjects)", fontsize=13)
ax.set_ylim(0, 100)
ax.set_xticks(epsilons)
ax.set_xticklabels([str(e) for e in epsilons])
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
ax.grid(True, alpha=0.3, linestyle="--")
ax.legend(loc="upper left", fontsize=10, framealpha=0.9)

# Annotate KAPPA at ε=0.001
kappa_asr_001 = results[0]["attacks"]["newton_cg"]["asr"] * 100
ax.annotate(f"KAPPA: {kappa_asr_001:.1f}%",
            xy=(0.001, kappa_asr_001), xytext=(0.0012, kappa_asr_001 - 12),
            fontsize=9, color="#e63946",
            arrowprops=dict(arrowstyle="->", color="#e63946", lw=1.2))

aa_asr_001 = results[0]["attacks"]["autoattack"]["asr"] * 100
ax.annotate(f"AutoAttack: {aa_asr_001:.1f}%",
            xy=(0.001, aa_asr_001), xytext=(0.0012, aa_asr_001 + 8),
            fontsize=9, color="#457b9d",
            arrowprops=dict(arrowstyle="->", color="#457b9d", lw=1.2))

plt.tight_layout()
plt.savefig("figures/asr_vs_epsilon_kappa.png", dpi=150, bbox_inches="tight")
print("Saved: figures/asr_vs_epsilon_kappa.png")
plt.close()


# ── Figure 2: Bar chart at ε = 0.001 ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

eps001 = results[0]["attacks"]
keys_ordered = ["newton_cg", "apgd_ce", "pgd_40", "autoattack", "cw_l2", "pgd_500"]
labels_ordered = [ATTACKS[k][0] for k in keys_ordered]
colors_ordered  = [ATTACKS[k][1] for k in keys_ordered]
asrs_ordered    = [eps001[k]["asr"] * 100 for k in keys_ordered]
times_ordered   = [eps001[k]["time_s"] for k in keys_ordered]

bars = ax.barh(labels_ordered, asrs_ordered, color=colors_ordered,
               edgecolor="white", linewidth=0.8, height=0.6)

for bar, asr, t in zip(bars, asrs_ordered, times_ordered):
    ax.text(asr + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{asr:.1f}%  ({t:.0f}s)",
            va="center", fontsize=9.5, color="#333333")

ax.set_xlabel("True Attack Success Rate (%)", fontsize=12)
ax.set_title(f"Attack Comparison at ε = 0.001\nSTAGIN fMRI  |  κ = 178,695", fontsize=13)
ax.set_xlim(0, 75)
ax.axvline(asrs_ordered[0], color="#e63946", linestyle="--", alpha=0.4, linewidth=1)
ax.grid(axis="x", alpha=0.3, linestyle="--")
ax.invert_yaxis()

plt.tight_layout()
plt.savefig("figures/bar_attack_eps001.png", dpi=150, bbox_inches="tight")
print("Saved: figures/bar_attack_eps001.png")
plt.close()


# ── Figure 3: KAPPA vs AutoAttack gap across epsilons ───────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))

kappa_asrs = [e["attacks"]["newton_cg"]["asr"] * 100 for e in results]
aa_asrs    = [e["attacks"]["autoattack"]["asr"] * 100 for e in results]
gaps       = [k - a for k, a in zip(kappa_asrs, aa_asrs)]

ax.fill_between(epsilons, aa_asrs, kappa_asrs,
                alpha=0.15, color="#e63946", label="KAPPA advantage")
ax.plot(epsilons, kappa_asrs, color="#e63946", marker="o", linewidth=2.5,
        markersize=8, label="KAPPA (ours)")
ax.plot(epsilons, aa_asrs, color="#457b9d", marker="s", linewidth=2,
        markersize=7, linestyle="--", label="AutoAttack (SOTA)")

for i, (eps, gap) in enumerate(zip(epsilons, gaps)):
    ax.annotate(f"+{gap:.1f}pp",
                xy=(eps, (kappa_asrs[i] + aa_asrs[i]) / 2),
                ha="center", va="center", fontsize=8.5,
                color="#e63946", fontweight="bold")

ax.set_xscale("log")
ax.set_xlabel("Epsilon (L∞)", fontsize=12)
ax.set_ylabel("True ASR (%)", fontsize=12)
ax.set_title("KAPPA vs AutoAttack: Advantage Across Epsilon Budget\nShaded area = percentage points gained by using KAPPA", fontsize=12)
ax.set_ylim(0, 100)
ax.set_xticks(epsilons)
ax.set_xticklabels([str(e) for e in epsilons])
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
ax.legend(fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle="--")

plt.tight_layout()
plt.savefig("figures/kappa_vs_autoattack_gap.png", dpi=150, bbox_inches="tight")
print("Saved: figures/kappa_vs_autoattack_gap.png")
plt.close()

print("\nAll figures generated successfully.")
