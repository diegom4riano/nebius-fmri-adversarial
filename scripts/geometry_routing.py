#!/usr/bin/env python3
"""Join per-subject geometry with attack outcomes and test the routing hypothesis.

The MVP question: does the local loss-surface geometry (grad_norm, sigma_max, kappa) of an
input predict WHICH attack succeeds on it? If yes, a robustness test can be ROUTED by
geometry instead of running every attack on every input.

Per attackable subject we join:
  * geometry (from output/geometry_<model>.json, aligned by loader position);
  * attack outcomes (per_subject predictions in the result JSONs) → a "winning attack"
    category: robust / first_order_only / kappa_only / both;
  * a geometry class from data-driven (median) splits: flat_masked / flat_with_grad /
    isotropic_curved / anisotropic.

It then reports separability (geometry class x category contingency, per-feature AUC,
robustness rate + Wilson CI per class, a rule-router vs run-all-oracle comparison) and
writes the headline scatter (sigma_max x kappa, coloured by winning attack) + supporting
figures. Fully offline.

Usage:
  python scripts/geometry_routing.py output/            # scans the sweep run dirs in output/
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_sweep import discover, _load, _eps_entry
from paired_stats import wilson_ci

TARGET = 0
ECG_JSON = "output/20260711_170335/attack_results_ecg.json"
FO_KEYS_STAGIN = ("pgd_40", "pgd_500", "apgd_ce")
FO_KEYS_ECG = ("pgd_40", "pgd_500", "apgd_ce", "autoattack")
CATEGORIES = ("robust", "first_order_only", "kappa_only", "both")
GEOM_CLASSES = ("flat_masked", "flat_with_grad", "isotropic_curved", "anisotropic")

# Okabe--Ito, fixed categorical hue per winning-attack category (matches generate_figures).
CAT_COLOR = {
    "robust":           "#999999",   # grey — nothing flips it
    "first_order_only": "#0072B2",   # blue — PGD/APGD wins
    "kappa_only":       "#D55E00",   # vermillion — second-order is the only tool that works
    "both":             "#009E73",   # green — everything flips it
}


# --------------------------------------------------------------------------- geometry I/O
def load_geometry(model, out_dir):
    """{pos: record} for subjects with a valid (non-null) spectrum."""
    path = os.path.join(out_dir, f"geometry_{model}.json")
    if not os.path.exists(path):
        return None, path
    d = json.load(open(path))
    geo = {}
    for s in d["subjects"]:
        if s.get("sigma_max") is None or s.get("kappa_at_lambda") is None:
            continue
        geo[int(s["pos"])] = s
    return geo, path


# ------------------------------------------------------------------- outcomes per model/eps
def stagin_outcomes(out_dir):
    """{eps: {labels, pred_clean, fo:{k:arr}, kappa_by_lam:{lam:arr}, kappa_union:arr}}."""
    runs = discover(out_dir)
    out = {}
    for eps, grp in sorted(runs.items()):
        base_dir, lam_dirs = grp.get("base"), grp.get("lam", {})
        if not base_dir or not lam_dirs:
            continue
        base = _eps_entry(_load(base_dir), eps)
        psb = base["per_subject"]
        n = len(psb["labels"])
        fo = {k: np.array(psb[k]) for k in FO_KEYS_STAGIN if k in psb}
        kappa_by_lam, union = {}, np.zeros(n, dtype=bool)
        for lam, d in lam_dirs.items():
            ek = _eps_entry(_load(d), eps)
            if not ek or "newton_cg" not in ek.get("per_subject", {}):
                continue
            nc = np.array(ek["per_subject"]["newton_cg"])
            kappa_by_lam[lam] = nc
            union |= (nc == TARGET)
        out[eps] = dict(labels=np.array(psb["labels"]), pred_clean=np.array(psb["pred_clean"]),
                        fo=fo, kappa_by_lam=kappa_by_lam, kappa_union=union)
    return out


def ecg_outcomes(path=ECG_JSON):
    if not os.path.exists(path):
        return {}
    d = json.load(open(path))
    out = {}
    for e in d.get("epsilon_results", []):
        ps = e["per_subject"]
        fo = {k: np.array(ps[k]) for k in FO_KEYS_ECG if k in ps}
        nc = np.array(ps["newton_cg"])
        out[float(e["epsilon"])] = dict(
            labels=np.array(ps["labels"]), pred_clean=np.array(ps["pred_clean"]),
            fo=fo, kappa_by_lam={"single": nc}, kappa_union=(nc == TARGET))
    return out


# --------------------------------------------------------------------------- join + labels
def build_table(geo, oc):
    """Per pool subject with valid geometry: features + winning-attack category."""
    labels, pred_clean = oc["labels"], oc["pred_clean"]
    pool = (labels != TARGET) & (pred_clean != TARGET)
    rows = []
    for pos in sorted(geo):
        if pos >= len(labels) or not pool[pos]:
            continue
        kappa_flip = bool(oc["kappa_union"][pos])
        fo_flip = any(bool(arr[pos] == TARGET) for arr in oc["fo"].values())
        if kappa_flip and fo_flip:
            cat = "both"
        elif kappa_flip:
            cat = "kappa_only"
        elif fo_flip:
            cat = "first_order_only"
        else:
            cat = "robust"
        g = geo[pos]
        rows.append(dict(pos=pos, sigma_max=g["sigma_max"], grad_norm=g["grad_norm"],
                         kappa=g["kappa_at_lambda"]["0.1"], mu_min=g["mu_min"],
                         mu_max=g["mu_max"], kappa_flip=kappa_flip, fo_flip=fo_flip, cat=cat))
    return rows


def assign_geom_class(rows):
    """Two median splits → 4 classes. Descriptive at small n; keeps each class populated."""
    if not rows:
        return
    smax = np.array([r["sigma_max"] for r in rows])
    gnorm = np.array([r["grad_norm"] for r in rows])
    kap = np.array([r["kappa"] for r in rows])
    smax_med, gnorm_med, kap_med = np.median(smax), np.median(gnorm), np.median(kap)
    for r in rows:
        if r["sigma_max"] < smax_med:                       # flat
            r["geom_class"] = "flat_masked" if r["grad_norm"] < gnorm_med else "flat_with_grad"
        else:                                               # curved
            r["geom_class"] = "anisotropic" if r["kappa"] >= kap_med else "isotropic_curved"
    return dict(sigma_max=float(smax_med), grad_norm=float(gnorm_med), kappa=float(kap_med))


# --------------------------------------------------------------------------- separability
def _auc(scores, pos_mask):
    """Rank AUC (Mann--Whitney): P(score higher for a positive than a negative)."""
    pos = [s for s, m in zip(scores, pos_mask) if m]
    neg = [s for s, m in zip(scores, pos_mask) if not m]
    if not pos or not neg:
        return float("nan")
    c = sum((1.0 if p > n else 0.5 if p == n else 0.0) for p in pos for n in neg)
    return c / (len(pos) * len(neg))


def _chi2_p(table):
    """p-value of independence for a class x category count table (scipy if available)."""
    try:
        from scipy.stats import chi2_contingency
        t = np.array(table, dtype=float)
        t = t[t.sum(1) > 0][:, t.sum(0) > 0]               # drop empty rows/cols
        if t.shape[0] < 2 or t.shape[1] < 2:
            return float("nan")
        return float(chi2_contingency(t)[1])
    except Exception:
        return float("nan")


def separability_report(rows, meds, title):
    print(f"\n{'='*78}\n  {title}   (n={len(rows)} pool subjects with valid geometry)")
    if not rows:
        print("  (no rows)"); return
    # category counts
    catc = {c: sum(1 for r in rows if r["cat"] == c) for c in CATEGORIES}
    print("  categories: " + "  ".join(f"{c}={catc[c]}" for c in CATEGORIES))
    print(f"  medians: sigma_max={meds['sigma_max']:.4g}  "
          f"grad_norm={meds['grad_norm']:.4g}  kappa@0.1={meds['kappa']:.4g}")

    # contingency geom_class x category
    print(f"\n  {'geom_class':>17} | " + " ".join(f"{c[:10]:>11}" for c in CATEGORIES) +
          " |  robust-rate [Wilson95]")
    table = []
    for gc in GEOM_CLASSES:
        sub = [r for r in rows if r["geom_class"] == gc]
        counts = [sum(1 for r in sub if r["cat"] == c) for c in CATEGORIES]
        table.append(counts)
        nrob = sum(1 for r in sub if r["cat"] == "robust")
        lo, hi = wilson_ci(nrob, len(sub)) if sub else (float("nan"), float("nan"))
        rate = (nrob / len(sub)) if sub else float("nan")
        print(f"  {gc:>17} | " + " ".join(f"{c:>11}" for c in counts) +
              f" |  {rate:>5.2f} [{lo:.2f},{hi:.2f}] (n={len(sub)})")
    print(f"  chi-square independence (class vs category): p = {_chi2_p(table):.4g}")

    # per-feature AUC for two questions
    for qname, mask in (("KAPPA flips subject", [r["kappa_flip"] for r in rows]),
                        ("subject is robust to ALL", [r["cat"] == "robust" for r in rows])):
        aucs = {f: _auc([r[f] for r in rows], mask) for f in ("sigma_max", "kappa", "grad_norm")}
        print(f"  AUC[{qname:>24}]: " +
              "  ".join(f"{f}={aucs[f]:.3f}" for f in ("sigma_max", "kappa", "grad_norm")))

    # rule-router vs run-all oracle (illustrative; oracle is an upper bound by construction)
    _router_vs_oracle(rows)


def _router_vs_oracle(rows):
    """Rule: anisotropic→KAPPA, else→first-order. Compare ASR to the run-all oracle."""
    def flips(r, use_kappa):
        return r["kappa_flip"] if use_kappa else r["fo_flip"]
    router = sum(1 for r in rows if flips(r, r["geom_class"] == "anisotropic"))
    oracle = sum(1 for r in rows if (r["kappa_flip"] or r["fo_flip"]))
    n = len(rows)
    print(f"  router(aniso→KAPPA else FO) ASR={router/n:.2f}  vs  oracle(run-all) ASR={oracle/n:.2f}"
          f"   (router runs 1 attack/subject, oracle runs all)")


# --------------------------------------------------------------------------- figures
def fig_scatter(rows, model, eps, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, MUTED, GRID = "#222222", "#666666", "#dddddd"
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 200, "font.size": 11,
                         "axes.edgecolor": MUTED, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True,
                         "grid.color": GRID, "axes.axisbelow": True})
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    smax_med = np.median([r["sigma_max"] for r in rows])
    kap_med = np.median([r["kappa"] for r in rows])
    for cat in CATEGORIES:
        pts = [(r["sigma_max"], r["kappa"]) for r in rows if r["cat"] == cat]
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.scatter(xs, ys, s=55, c=CAT_COLOR[cat], label=f"{cat} (n={len(pts)})",
                   edgecolors="white", linewidths=0.8, zorder=3, alpha=0.9)
    ax.axvline(smax_med, color=MUTED, lw=1, ls="--", alpha=0.6, zorder=1)
    ax.axhline(kap_med, color=MUTED, lw=1, ls="--", alpha=0.6, zorder=1)
    ax.set_xscale("log")
    if all(r["kappa"] > 0 for r in rows) and max(r["kappa"] for r in rows) > 3:
        ax.set_yscale("log")
    ax.set_xlabel("σ_max  (curvature magnitude, log)")
    ax.set_ylabel("κ(H+λI) at λ=0.1  (anisotropy)")
    ax.set_title(f"{model.upper()} ε={eps}: does geometry predict the winning attack?",
                 fontsize=12.5)
    ax.legend(frameon=False, fontsize=9, loc="best")
    ax.annotate("anisotropic\n(Newton's regime)", (smax_med, kap_med),
                textcoords="offset points", xytext=(10, 10), fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_png}")


def fig_robustness_by_class(rows, model, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, MUTED, GRID = "#222222", "#666666", "#dddddd"
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 200, "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True})
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    xs, rates, los, his, ns = [], [], [], [], []
    for gc in GEOM_CLASSES:
        sub = [r for r in rows if r["geom_class"] == gc]
        if not sub:
            continue
        nrob = sum(1 for r in sub if r["cat"] == "robust")
        lo, hi = wilson_ci(nrob, len(sub))
        xs.append(gc); rates.append(100 * nrob / len(sub))
        los.append(100 * (nrob / len(sub) - lo)); his.append(100 * (hi - nrob / len(sub)))
        ns.append(len(sub))
    x = np.arange(len(xs))
    ax.bar(x, rates, 0.6, color="#0072B2", zorder=3,
           yerr=[los, his], capsize=4, ecolor=MUTED)
    for i, (r, n) in enumerate(zip(rates, ns)):
        ax.annotate(f"{r:.0f}%\n(n={n})", (i, r), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=8, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(xs, rotation=15, ha="right")
    ax.set_ylabel("robust-to-all rate (%)")
    ax.set_title(f"{model.upper()}: robustness by geometry class (Wilson 95% CI)", fontsize=12)
    fig.tight_layout(); fig.savefig(out_png, bbox_inches="tight"); plt.close(fig)
    print(f"  saved: {out_png}")


# --------------------------------------------------------------------------- driver
def analyze(model, geo, oc_by_eps, primary_eps, out_dir):
    if not geo:
        print(f"[{model}] no geometry file (run profile_geometry.py --model {model} first)")
        return
    for eps in sorted(oc_by_eps):
        rows = build_table(geo, oc_by_eps[eps])
        meds = assign_geom_class(rows)
        separability_report(rows, meds, f"{model.upper()}  ε={eps}")
        if abs(eps - primary_eps) < 1e-9 and rows:
            os.makedirs("figures", exist_ok=True)
            fig_scatter(rows, model, eps, f"figures/geometry_routing_{model}.png")
            fig_robustness_by_class(rows, model, f"figures/geometry_robustness_{model}.png")


def main(argv):
    out_dir = argv[1] if len(argv) > 1 else "output"
    geo_st, path_st = load_geometry("stagin", out_dir)
    geo_ec, path_ec = load_geometry("ecg", out_dir)

    if geo_st:
        print(f"STAGIN geometry: {len(geo_st)} subjects ({path_st})")
        analyze("stagin", geo_st, stagin_outcomes(out_dir), primary_eps=0.001, out_dir=out_dir)
    else:
        print(f"STAGIN geometry not found at {path_st}")

    if geo_ec:
        print(f"\nECG geometry: {len(geo_ec)} subjects ({path_ec})")
        analyze("ecg", geo_ec, ecg_outcomes(), primary_eps=10.0, out_dir=out_dir)
    else:
        print(f"ECG geometry not found at {path_ec}")

    print("\nNote: small n → results are descriptive; AUC>0.5 and a concentrated 'kappa_only' "
          "region support the routing hypothesis. Oracle (run-all) ASR is an upper bound.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
