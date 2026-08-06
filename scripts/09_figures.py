#!/usr/bin/env python
"""
Phase 7. Publication figures.

Generates all main figures from the saved metric tables. Figures use a clean,
print-friendly style and are written as both PNG (300 dpi) and PDF (vector).
Outputs -> results/figures/*.png|pdf
"""
from __future__ import annotations
import sys
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fmscreen import config as C
from fmscreen import figstyle
from fmscreen.figstyle import (NAVY, TEAL, AMBER, CORAL, SLATE, GREEN, PURPLE, INK,
                               NAVY_D, CORAL_D, GREEN_D)

figstyle.apply()
FIG = C.FIGURES


def _system_names():
    """Map UNIFORMAT system codes -> human-readable names, if available."""
    f = C.TABLES / "burden_by_system.csv"
    if not Path(f).exists():
        return {}
    b = pd.read_csv(f)
    if C.COL_SYSTEM not in b.columns or C.COL_SYSTEM_DESC not in b.columns:
        return {}
    m = (b[[C.COL_SYSTEM, C.COL_SYSTEM_DESC]].dropna().drop_duplicates()
         .assign(**{C.COL_SYSTEM_DESC: lambda x: x[C.COL_SYSTEM_DESC].astype(str).str.strip()}))
    return dict(zip(m[C.COL_SYSTEM].astype(str), m[C.COL_SYSTEM_DESC]))


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=300)
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)
    print(f"[fig] {name}", flush=True)


def fig_louo_per_campus():
    f = C.METRICS / "louo_folds.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    sub = d[(d.target_kind == "severity_labour") & (d.pctl == 75) &
            (d.layer == "M1") & (d.model == "gbdt")].copy()
    sub["u"] = sub["held_out_university"].astype(str)
    sub = sub.sort_values("lift_top10", ascending=False)
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    x = np.arange(len(sub))
    yerr = np.vstack([sub["lift_top10"] - sub["lift_top10_ci_lo"],
                      sub["lift_top10_ci_hi"] - sub["lift_top10"]])
    ax.bar(x, sub["lift_top10"], yerr=yerr, capsize=3, color=NAVY,
           ecolor=SLATE, label="Gradient-boosted screen")
    ax.scatter(x, sub["best_rule_lift_top10"], color=CORAL_D, zorder=5,
               marker="D", s=40, edgecolor="white", linewidth=0.6,
               label="Best simple rule")
    ax.axhline(2.0, ls="--", color=SLATE, lw=1.2, label="Two-fold bar")
    ax.set_xticks(x); ax.set_xticklabels("U" + sub["u"])
    ax.set_xlabel("Held-out campus"); ax.set_ylabel("Top-decile precision lift")
    # Legend ordering: main series first, then rule, then reference bar.
    h, lab = ax.get_legend_handles_labels()
    order = ["Gradient-boosted screen", "Best simple rule", "Two-fold bar"]
    idx = [lab.index(o) for o in order if o in lab]
    ax.legend([h[i] for i in idx], [lab[i] for i in idx],
              frameon=False, loc="upper right", ncol=1)
    figstyle.despine(ax)
    save(fig, "fig_louo_per_campus_lift")


def fig_calibration_curve():
    f = C.METRICS / "calibration_curve.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    sub = d[(d.target_kind == "severity_labour") & (d.model == "gbdt")]
    # x exists only at 0,5,10,20 (% local history) -> plot on evenly-spaced
    # categorical positions so no unmeasured intermediate behaviour is implied.
    fracs = sorted(sub["frac"].unique())            # [0.0, 0.05, 0.10, 0.20]
    pcts = [int(round(fr * 100)) for fr in fracs]    # [0, 5, 10, 20]
    pos = {fr: i for i, fr in enumerate(fracs)}      # 0,1,2,3
    n_campus = sub["held_out_university"].nunique()
    fig, ax = plt.subplots(figsize=(6.3, 4.0))
    grey_handle = None
    for u, g in sub.groupby("held_out_university"):
        g = g.sort_values("frac")
        ln, = ax.plot([pos[fr] for fr in g["frac"]], g["lift_top10"],
                      color=SLATE, lw=0.8, alpha=0.25)
        grey_handle = ln
    med = sub.groupby("frac")["lift_top10"].median().sort_index()
    ax.plot([pos[fr] for fr in med.index], med.values, color=NAVY_D, lw=2.5,
            marker="o", ms=6, mfc=NAVY_D, mec="white", mew=0.8, zorder=5,
            label="median across campuses")
    ax.axhline(2.0, ls="--", color=SLATE, lw=1.2, label="Two-fold bar")
    # Annotate the near-flat result.
    delta = med.iloc[-1] - med.iloc[0]
    ax.annotate(f"$\\Delta$(0$\\to$20%) $\\approx$ {delta:+.2f} (flat)",
                xy=(pos[fracs[-1]], med.iloc[-1]), xytext=(-6, -22),
                textcoords="offset points", ha="right", va="top",
                fontsize=9, color=INK,
                bbox=dict(boxstyle="square,pad=0.3", fc="white", ec=SLATE, lw=0.6))
    ax.set_xticks(list(range(len(pcts)))); ax.set_xticklabels(pcts)
    ax.set_xlim(-0.3, len(pcts) - 0.7)
    ax.set_xlabel("Local history added to training (% of campus's earliest data)")
    ax.set_ylabel("Top-decile precision lift")
    if grey_handle is not None:
        grey_handle.set_label(f"individual campuses (n={n_campus})")
    # below the axes: an opaque in-plot legend was punching a white hole through
    # the two-fold reference line and several of the per-campus traces
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=3, fontsize=8.5)
    figstyle.despine(ax)
    save(fig, "fig_calibration_curve")


def fig_decomposition():
    f = C.METRICS / "decomposition.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    order = ["occurrence", "severity_p50", "severity_p75", "severity_p90"]
    d = d[d.target.isin(order)]
    labels = ["Occurrence", "p50", "p75", "p90"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.3, 3.5))
    data_lift = [d[d.target == t]["lift_top10"].dropna().values for t in order]
    data_auc = [d[d.target == t]["pr_auc"].dropna().values for t in order]

    def styled_box(ax, data, fill, accent, fmt="{:.2f}"):
        bx = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.62,
                   medianprops=dict(color=accent, lw=2.0),
                   boxprops=dict(facecolor=fill, edgecolor=INK, lw=1.0),
                   whiskerprops=dict(color=INK, lw=1.2),
                   capprops=dict(color=INK, lw=1.2),
                   flierprops=dict(marker="o", ms=4, mfc="none", mec=accent, alpha=0.8))
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(length=3, color="#9aa3ab")
        # Median value label above each box.
        for i, dat in enumerate(data, start=1):
            if len(dat) == 0:
                continue
            med = float(np.median(dat))
            top = float(np.max(dat))
            ax.annotate(fmt.format(med), (i, top), xytext=(0, 5),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=8.5, fontweight="bold", color=INK)
        # Headroom so the top labels never clip.
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.08 * (hi - lo))

    styled_box(a1, data_lift, NAVY, NAVY_D)
    a1.axhline(2.0, ls="--", color=CORAL_D, lw=1.3)
    a1.text(4.52, 2.06, "2$\\times$ bar", color=CORAL_D, fontsize=8.5,
            va="bottom", ha="right")
    a1.set_ylabel("Top-10% precision lift")
    a1.set_xlabel("Prediction target")
    styled_box(a2, data_auc, GREEN, GREEN_D)
    a2.set_ylabel("PR-AUC")
    a2.set_xlabel("Prediction target")
    save(fig, "fig_occurrence_vs_highburden")


def fig_heterogeneity():
    f = C.METRICS / "heterogeneity_by_system.csv"
    if not f.exists():
        return
    d = pd.read_csv(f).sort_values("lift_top10", ascending=True)
    names = _system_names()

    def status_color(row):
        # colour by the DEPLOYMENT criterion (cross-campus robustness), not pooled lift
        if row["frac_campuses_meet_2x"] < 0.8:
            return CORAL          # unstable across campuses (deployment-weak)
        if row["n_campuses_evaluable"] < 3:
            return AMBER          # too few campuses to judge robustness
        return NAVY               # robustly screenable

    colors = [status_color(r) for _, r in d.iterrows()]
    codes = d["system"].astype(str).tolist()
    have_names = bool(names) and any(c in names for c in codes)
    if have_names:
        ylabels = [f"{c} {names[c]}" if c in names else c for c in codes]
    else:
        ylabels = codes

    fig, ax = plt.subplots(figsize=(6.3, max(4, 0.40 * len(d))))
    y = np.arange(len(d))
    bars = ax.barh(y, d["lift_top10"], color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y); ax.set_yticklabels(ylabels)
    ax.set_ylim(-0.6, len(d) + 0.3)            # headroom above the top bar
    # drawn above the bars: at zorder 0 it sat behind them and the annotation
    # below pointed at a line the reader could not see
    ax.axvline(2.0, ls="--", color=SLATE, lw=1.2, zorder=3)
    ax.set_xlim(0, 8)
    # Numeric lift value at the end of each bar (one decimal).
    figstyle.barlabels(ax, bars, fmt="{:.1f}", pad=4, fontsize=8.5)
    # Clear label anchored to the 2x reference line, in the headroom above bars.
    ax.annotate("Two-fold bar", xy=(2.0, len(d) - 0.55),
                xytext=(2.35, len(d) - 0.1), ha="left", va="center",
                fontsize=8.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK, lw=1.0))
    ax.set_xlabel("Pooled top-decile precision lift (high-burden p75, gradient-boosted)")
    # Status legend (only show categories present in the data).
    from matplotlib.patches import Patch
    used = set(colors)
    spec = [(NAVY, "robust: $\\geq$2× in $\\geq$80% of campuses"),
            (AMBER, "too few campuses to judge"),
            (CORAL, "unstable across campuses (<80%)")]
    legend_items = [Patch(facecolor=c, label=lab) for c, lab in spec if c in used]
    ax.legend(handles=legend_items, frameon=False, loc="lower right", fontsize=8.5)
    figstyle.despine(ax)
    save(fig, "fig_heterogeneity_by_system")


def main():
    for fn in (fig_louo_per_campus, fig_calibration_curve, fig_decomposition,
               fig_heterogeneity):
        try:
            fn()
        except Exception as e:
            print(f"[fig] {fn.__name__} failed: {e}", flush=True)
    print("[fig] done ->", FIG, flush=True)


if __name__ == "__main__":
    main()
