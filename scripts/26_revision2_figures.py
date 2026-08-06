#!/usr/bin/env python
"""Revision-2 figures: GBDT feature importance (SHAP) and the co-escalation
robustness panel (age strata, lag-decay, within-building control, size/activity).
Matches the house style of scripts/21_revision_figures.py. Saves PNG + PDF."""
from __future__ import annotations
import sys, json
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fmscreen import config as C
from fmscreen import figstyle
from fmscreen.figstyle import (NAVY, TEAL, AMBER, CORAL, SLATE, GREEN, PURPLE, INK,
                               NAVY_D, CORAL_D, LGREY)

figstyle.apply()
FIG = C.FIGURES


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=300); fig.savefig(FIG / f"{name}.pdf"); plt.close(fig)
    print("[fig]", name, flush=True)


def fig_feature_importance():
    imp = pd.read_csv(C.TABLES / "feature_importance.csv").head(15).iloc[::-1]
    cmap = {"record_history": NAVY, "premium": GREEN, "taxonomy": AMBER, "weather_season": LGREY}
    colors = [cmap[g] for g in imp["group"]]
    fig, ax = plt.subplots(figsize=(6.3, 4.8))
    bars = ax.barh(np.arange(len(imp)), imp["mean_abs_shap"], color=colors,
                   edgecolor="white", lw=0.5)
    ax.set_yticks(np.arange(len(imp))); ax.set_yticklabels(imp["label"])
    ax.set_xlabel("Mean absolute Shapley attribution  (impact on high-burden score)")
    ax.set_xlim(0, imp["mean_abs_shap"].max() * 1.05)
    # value labels at the end of each (horizontal) bar; annotate explicitly so
    # short bars are not mis-classified by the generic width/height heuristic
    for b in bars:
        w = b.get_width()
        ax.annotate(f"{w:.2f}", (w, b.get_y() + b.get_height() / 2),
                    xytext=(3, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=8.5, color=INK)
    # faint vertical major gridlines only
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=figstyle.GRIDC, lw=0.6, alpha=0.7)
    figstyle.despine(ax)
    # legend: only groups that actually appear in the top-15
    from matplotlib.patches import Patch
    present = set(imp["group"])
    legend = [Patch(fc=NAVY, label="Record history"),
              Patch(fc=AMBER, label="System taxonomy"),
              Patch(fc=LGREY, label="Weather / season")]
    if "premium" in present:
        legend.append(Patch(fc=GREEN, label="Reactive premium"))
    ax.legend(handles=legend, loc="lower right")
    save(fig, "fig_feature_importance")


def fig_coesc_robustness():
    d = json.load(open(C.METRICS / "coescalation_robustness.json"))
    pooled = d["full_panel_cellpair_rr"]["rr"]
    strata = pd.DataFrame(d["strata"])
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 5.8))
    YLAB = "Cell-pair risk ratio"
    YMAX = 7.2          # shared range for the bar panels (a, c, d)
    REF = "#9aa3ab"     # neutral grey for the RR = 1 reference line

    # (a) building-age tertiles
    ax = axes[0, 0]
    age = strata[strata.dimension == "building_age"]
    labs = ["Oldest\n(<1958)", "Mid\n(1958–85)", "Newest\n(≥1986)"]
    x = np.arange(len(age))
    err = [age["rr"] - age["ci_lo"], age["ci_hi"] - age["rr"]]
    bars = ax.bar(x, age["rr"], color=NAVY, edgecolor="white", lw=0.5,
                  yerr=err, capsize=4, width=0.6)
    ax.axhline(pooled, ls="--", color=CORAL_D, lw=1.4, label=f"Pooled risk ratio = {pooled:.2f}")
    ax.axhline(1.0, ls="-", color=REF, lw=1.0)
    figstyle.barlabels(ax, bars, fmt="{:.2f}", pad=3, tops=age["ci_hi"].to_numpy())
    ax.set_xticks(x); ax.set_xticklabels(labs); ax.set_ylabel(YLAB)
    ax.set_title("(a)")
    ax.legend(loc="upper right"); ax.set_ylim(0, YMAX)
    ax.grid(axis="x", visible=False); figstyle.despine(ax)

    # (b) lag-decay (keeps its own scale)
    ax = axes[0, 1]
    lag = pd.DataFrame(d["lag_decay"])
    ax.plot(lag["lag_quarters"], lag["rr"], "-o", color=NAVY_D, lw=1.8, ms=6)
    ax.axhline(1.0, ls="-", color=REF, lw=1.0)
    ax.set_xlabel("Quarters after trigger ($t{+}L$)"); ax.set_ylabel("Risk ratio at lag $L$")
    ax.set_title("(b)")
    ax.set_ylim(0, 3.8); ax.set_xticks(range(1, 9))
    figstyle.despine(ax)

    # (c) within-building control
    ax = axes[1, 0]
    wb = d["within_building"]
    vals = [wb["pooled_rr_contrast"], wb["mh_rr_within_building"]]
    # Two defects here before: the within-building interval was drawn one-sided
    # (its lower half hardcoded to zero, so 0.88 was never shown), and the
    # pooled contrast was given a zero-length bar, which renders as a cap and
    # reads as "measured with no uncertainty". It has no clustered interval, so
    # it now gets none, and the caption says so.
    mh, lo, hi = wb["mh_rr_within_building"], wb["ci95"][0], wb["ci95"][1]
    errs = [[np.nan, mh - lo], [np.nan, hi - mh]]
    cols = [LGREY, NAVY]   # pooled neutral, within-building = key contrast
    bars = ax.bar([0, 1], vals, color=cols, edgecolor="white", lw=0.5, width=0.6,
                  yerr=errs, capsize=4)
    ax.axhline(pooled, ls="--", color=CORAL_D, lw=1.4)
    ax.axhline(1.0, ls="-", color=REF, lw=1.0)
    figstyle.barlabels(ax, bars, fmt="{:.2f}", pad=3,
                       tops=[wb["pooled_rr_contrast"], wb["ci95"][1]])
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pooled\n(cross-building)",
                                               "Within-building\n(own control)"])
    ax.set_ylabel(YLAB); ax.set_ylim(0, YMAX)
    ax.set_title("(c)")
    ax.grid(axis="x", visible=False); figstyle.despine(ax)

    # (d) size & activity strata (two stratifications)
    ax = axes[1, 1]
    sub = strata[strata.dimension.isin(["n_systems", "upm_volume"])].reset_index(drop=True)
    labs2 = ["Systems\n3–8", "Systems\n≥9", "UPM vol.\nmid", "UPM vol.\nhigh"]
    # leave a gap between the two groupings
    x = np.array([0.0, 1.0, 2.4, 3.4])
    err = [sub["rr"] - sub["ci_lo"], sub["ci_hi"] - sub["rr"]]
    bars = ax.bar(x, sub["rr"], color=NAVY, edgecolor="white", lw=0.5,
                  yerr=err, capsize=4, width=0.6)
    ax.axhline(pooled, ls="--", color=CORAL_D, lw=1.4)
    ax.axhline(1.0, ls="-", color=REF, lw=1.0)
    figstyle.barlabels(ax, bars, fmt="{:.2f}", pad=3, tops=sub["ci_hi"].to_numpy())
    ax.set_xticks(x); ax.set_xticklabels(labs2)
    ax.set_ylabel(YLAB); ax.set_ylim(0, YMAX)
    # light visual separation + secondary group labels
    ax.axvline(1.7, color=figstyle.GRIDC, lw=1.0, ls="-")
    grp_y = -0.30   # axes fraction, below the two-line tick labels
    ax.text(0.215, grp_y, "by # systems", transform=ax.transAxes,
            ha="center", va="top", fontsize=8.5, style="italic", color=INK)
    ax.text(0.755, grp_y, "by UPM volume", transform=ax.transAxes,
            ha="center", va="top", fontsize=8.5, style="italic", color=INK)
    ax.set_title("(d)")
    ax.grid(axis="x", visible=False); figstyle.despine(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, "fig_coesc_robustness")


if __name__ == "__main__":
    fig_feature_importance()
    fig_coesc_robustness()
