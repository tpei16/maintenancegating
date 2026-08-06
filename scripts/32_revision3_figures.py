#!/usr/bin/env python
"""Revision-3 figures: building-feature ablation (C1) and captured-vs-missed
extreme-event capture by trajectory (C3). Shared house style. PNG + PDF."""
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
from fmscreen.figstyle import (NAVY, TEAL, AMBER, CORAL, SLATE, GREEN, INK, LGREY,
                               NAVY_D, CORAL_D)

figstyle.apply()
FIG = C.FIGURES


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=300); fig.savefig(FIG / f"{name}.pdf"); plt.close(fig)
    print("[fig]", name, flush=True)


def fig_building_ablation():
    d = json.load(open(C.METRICS / "building_ablation.json"))
    ml = d["median_lift"]
    rule = ml["M1"] - d["delta_M1_minus_rule"]["median"]   # best-rule median
    labels = ["Best simple\nrule", "M0\n(counts)", "M1\n(+cell burden)", "M1b\n(+building)"]
    vals = [rule, ml["M0"], ml["M1"], ml["M1b"]]
    cols = [LGREY, "#bcd2e8", NAVY, GREEN]
    fig, ax = plt.subplots(figsize=(6.3, 4.0))
    x = np.arange(len(vals))
    bars = ax.bar(x, vals, color=cols, edgecolor="white", lw=0.6, width=0.64)
    figstyle.barlabels(ax, bars, fmt="{:.2f}", pad=2)
    ax.axhline(2.0, ls="--", color=CORAL_D, lw=1.2)
    ax.text(0.5, 2.09, "2$\\times$ bar", color=CORAL_D, fontsize=8, va="bottom", ha="center")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Median LOUO top-10% lift")
    ax.set_ylim(0, max(vals) * 1.18)
    # bracket annotations for the two increments
    # Both annotations are the median of the per-campus PAIRED differences, with their
    # CIs. Reporting one as a paired delta and the other as the gap between the plotted
    # bars mixes two estimators in one panel and does not reconcile with the main text.
    d1 = d["delta_M1_minus_M0"]["median"]; ci1 = d["delta_M1_minus_M0"]["ci95"]
    d2 = d["delta_M1b_minus_M1"]["median"]; ci2 = d["delta_M1b_minus_M1"]["ci95"]
    ax.annotate(f"cell-burden gain +{d1:.2f}\n(95% CI {ci1[0]:.2f}, {ci1[1]:.2f})",
                xy=(2, ml["M1"]), xytext=(1.5, max(vals) * 1.10), ha="center",
                fontsize=8.2, color=NAVY_D)
    ax.annotate(f"building gain +{d2:.2f}\n(95% CI {ci2[0]:.2f}, {ci2[1]:.2f}: ~0)",
                xy=(3, ml["M1b"]), xytext=(3.0, max(vals) * 1.10), ha="center",
                fontsize=8.2, color=INK)
    ax.grid(axis="x", visible=False); figstyle.despine(ax)
    save(fig, "fig_building_ablation")


def fig_missed_p90():
    d = json.load(open(C.METRICS / "missed_p90.json"))
    tab = {r["traj"]: r for r in d["capture_by_trajectory"]}
    order = ["escalating", "non_escalating", "zero_prior"]
    labs = ["Escalating\n(rising prior)", "Non-escalating\n(flat/declining)", "Zero prior\nactivity"]
    cap = [tab[k]["capture_rate"] * 100 for k in order]
    ns = [tab[k]["n"] for k in order]
    cols = [NAVY, NAVY, CORAL]
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    x = np.arange(len(order))
    bars = ax.bar(x, cap, color=cols, edgecolor="white", lw=0.6, width=0.62)
    for b, n in zip(bars, ns):
        # One decimal below 10%, where integer rounding would disagree with the value
        # quoted in the main text (4.5%, not 4%).
        h = b.get_height()
        lab = f"{h:.1f}%" if h < 10 else f"{h:.0f}%"
        ax.annotate(f"{lab}\n(n={n:,})", (b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                    fontsize=8.6, color=INK)
    ov = d["overall_capture_rate_at_top10"] * 100
    ax.axhline(ov, ls="--", color=NAVY_D, lw=1.3, label=f"overall capture {ov:.0f}%")
    ax.set_xticks(x); ax.set_xticklabels(labs)
    ax.set_ylabel("Captured by top-decile screen (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False); figstyle.despine(ax)
    save(fig, "fig_missed_p90")


if __name__ == "__main__":
    fig_building_ablation()
    fig_missed_p90()
