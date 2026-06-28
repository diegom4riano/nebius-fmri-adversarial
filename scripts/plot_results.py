"""
Generate result figures from Nebius VM logs.

Outputs (saved to figures/):
  - asr_vs_epsilon.png  : Hessian vs PGD ASR across epsilon values
  - training_curve.png  : val BACC + train/val loss vs epoch
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

LOGS_DIR   = os.path.join(os.path.dirname(__file__), "..", "logs", "vm_logs")
FIGS_DIR   = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIGS_DIR, exist_ok=True)

# ── 1. ASR vs ε ──────────────────────────────────────────────────────────────
EVAL_LOGS = [
    ("eval_eps0001.log", 0.001),
    ("eval_eps0003.log", 0.003),
    ("eval_eps0005.log", 0.005),
    ("eval_eps001.log",  0.010),
]

epsilons, hess_asr, pgd_asr = [], [], []

for fname, eps in EVAL_LOGS:
    path = os.path.join(LOGS_DIR, fname)
    with open(path) as f:
        text = f.read()
    h = re.search(r"Hessian \(Newton-CG\) ASR\s*:\s*([\d.]+)", text)
    p = re.search(r"PGD\s+ASR\s*:\s*([\d.]+)", text)
    if h and p:
        epsilons.append(eps)
        hess_asr.append(float(h.group(1)))
        pgd_asr.append(float(p.group(1)))

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(epsilons, [v * 100 for v in hess_asr], "o-",  color="#e74c3c", lw=2.5,
        markersize=8, label="Hessian (Newton-CG)")
ax.plot(epsilons, [v * 100 for v in pgd_asr],  "s--", color="#3498db", lw=2.5,
        markersize=8, label="PGD")

# Shade advantage
ax.fill_between(epsilons,
                [v * 100 for v in pgd_asr],
                [v * 100 for v in hess_asr],
                alpha=0.12, color="#e74c3c", label="Advantage (H−PGD)")

# Annotate advantage at each point
for eps, h, p in zip(epsilons, hess_asr, pgd_asr):
    adv = (h - p) * 100
    ax.annotate(f"+{adv:.0f}pp",
                xy=(eps, h * 100),
                xytext=(0, 10), textcoords="offset points",
                ha="center", fontsize=9, color="#c0392b")

ax.set_xscale("log")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(
    lambda x, _: f"{x:.3f}".rstrip("0").rstrip(".")))
ax.set_xticks(epsilons)
ax.set_xlabel("Perturbation budget ε (L∞)", fontsize=12)
ax.set_ylabel("True Attack Success Rate (%)", fontsize=12)
ax.set_title("Adversarial Attack Comparison — STAGIN / HCP (n=216 test)", fontsize=13)
ax.set_ylim(0, 100)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0f}%"))
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
fig.tight_layout()
out = os.path.join(FIGS_DIR, "asr_vs_epsilon.png")
fig.savefig(out, dpi=150)
print(f"Saved: {out}")
plt.close(fig)


# ── 2. Training curve ────────────────────────────────────────────────────────
train_log = os.path.join(LOGS_DIR, "train_v5.log")
epochs, tr_loss, va_loss, va_bacc = [], [], [], []

with open(train_log) as f:
    for line in f:
        # Format: "  42    0.1234    2.3456   0.8065  0.7990   35.3s"
        m = re.match(r"\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+[\d.]+s", line)
        if m:
            epochs.append(int(m.group(1)))
            tr_loss.append(float(m.group(2)))
            va_loss.append(float(m.group(3)))
            va_bacc.append(float(m.group(4)))

best_epoch = epochs[np.argmax(va_bacc)]
best_bacc  = max(va_bacc)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

# Val BACC
ax1.plot(epochs, [v * 100 for v in va_bacc], color="#27ae60", lw=2, label="Val BACC")
ax1.axvline(best_epoch, color="gray", ls=":", lw=1.5, label=f"Best epoch {best_epoch}")
ax1.axhline(best_bacc * 100, color="#27ae60", ls="--", lw=1, alpha=0.5)
ax1.scatter([best_epoch], [best_bacc * 100], color="#27ae60", s=80, zorder=5)
ax1.annotate(f"BACC={best_bacc*100:.1f}%",
             xy=(best_epoch, best_bacc * 100),
             xytext=(8, -14), textcoords="offset points",
             fontsize=9, color="#1e8449")
ax1.set_ylabel("Balanced Accuracy (%)", fontsize=11)
ax1.set_title("Training Curve — STAGIN / HCP (OneCycleLR, reg_λ=1e-5)", fontsize=12)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0f}%"))
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Losses — cap va_loss for visibility (some NaN spikes)
VA_CAP = 20.0
va_loss_capped = [min(v, VA_CAP) for v in va_loss]
ax2.plot(epochs, tr_loss, color="#e67e22", lw=1.8, label="Train loss")
ax2.plot(epochs, va_loss_capped, color="#8e44ad", lw=1.8, alpha=0.8,
         label=f"Val loss (capped at {VA_CAP})")
ax2.axvline(best_epoch, color="gray", ls=":", lw=1.5)
ax2.set_xlabel("Epoch", fontsize=11)
ax2.set_ylabel("Cross-entropy loss", fontsize=11)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

fig.tight_layout()
out = os.path.join(FIGS_DIR, "training_curve.png")
fig.savefig(out, dpi=150)
print(f"Saved: {out}")
plt.close(fig)

print("Done.")
