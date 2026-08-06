#!/usr/bin/env python
"""CEPI figures: evidence-gate heatmap, priority map, and robustness/sensitivity.

Reads the real CEPI outputs written by 37_cepi.py (cepi_gates.csv,
cepi_priority_map.csv, cepi_summary.json) and renders three house-style figures.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fmscreen import config as C
from fmscreen import figstyle
from fmscreen.figstyle import (NAVY, TEAL, AMBER, CORAL, SLATE, GREEN, INK, GRIDC,
                               NAVY_D, CORAL_D, LGREY)

figstyle.apply()
TAB, MET, FIG = C.TABLES, C.METRICS, C.FIGURES

GATE_LABELS = {
    "data_sufficiency": "Data\nsufficiency",
    "risk_concentration": "Risk\nconcentration",
    "transfer_stability": "Transfer\nstability",
    "calibration_reliability": "Risk-gradient\nreliability",
    "antecedent_signal": "Prior-record\ntrace",
}
GATES = list(GATE_LABELS)
# pass / caution / fail -> muted house tints
CMAP = {1.0: "#d3e3f2", 0.5: "#f6e2c0", 0.0: "#f0cdd2"}
EDGE = {1.0: NAVY_D, 0.5: "#d99a3d", 0.0: CORAL_D}
LAB = {1.0: "pass", 0.5: "caution", 0.0: "fail"}


def gate_heatmap():
    g = pd.read_csv(TAB / "cepi_gates.csv")
    g = g.sort_values("burden_share", ascending=False).reset_index(drop=True)
    n = len(g)
    fig, ax = plt.subplots(figsize=(6.3, 5.2))
    for i, row in g.iterrows():
        y = n - 1 - i
        for j, gate in enumerate(GATES):
            v = row[gate]
            ax.add_patch(Rectangle((j, y), 0.94, 0.9, facecolor=CMAP[v],
                                   edgecolor=EDGE[v], lw=1.1))
            ax.text(j + 0.47, y + 0.45, LAB[v], ha="center", va="center",
                    fontsize=8, color=INK)
        # S column
        s = row["S"]
        ax.add_patch(Rectangle((len(GATES) + 0.25, y), 0.94 * s + 0.02, 0.9,
                               facecolor=NAVY, edgecolor="none", alpha=0.85))
        ax.add_patch(Rectangle((len(GATES) + 0.25, y), 0.96, 0.9,
                               facecolor="none", edgecolor=GRIDC, lw=0.8))
        # value sits outside the fill bar: a label centred in the box straddles the
        # bar edge whenever S is near half, and is unreadable in either colour.
        ax.text(len(GATES) + 0.25 + 0.96 + 0.08, y + 0.45, f"{s:.2f}", ha="left",
                va="center", fontsize=8.6, color=INK, fontweight="bold")
    ax.set_xlim(-0.05, len(GATES) + 2.05)
    ax.set_ylim(-0.15, n)
    ax.set_xticks([j + 0.47 for j in range(len(GATES))] + [len(GATES) + 0.73])
    ax.set_xticklabels(list(GATE_LABELS.values()) + ["Certificate\n$S$"], fontsize=8.6)
    ax.xaxis.set_ticks_position("top"); ax.xaxis.set_label_position("top")
    ax.tick_params(length=0)
    ax.set_yticks([n - 1 - i + 0.45 for i in range(n)])
    ax.set_yticklabels([f"{r.system}  {r.system_desc.strip()}"
                        for _, r in g.iterrows()], fontsize=8.0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(False)
    handles = [Patch(facecolor=CMAP[v], edgecolor=EDGE[v], label=LAB[v]) for v in (1.0, 0.5, 0.0)]
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.02, -0.10),
              ncol=3, fontsize=8, handlelength=1.1, columnspacing=1.0)
    fig.text(0.02, 0.015, "Systems ordered by share of reactive labor (top = highest burden). "
             "Hard gates: data sufficiency, transfer stability, risk-gradient reliability.",
             fontsize=8.6, color=INK)
    fig.savefig(FIG / "fig_cepi_gates.pdf"); fig.savefig(FIG / "fig_cepi_gates.png", dpi=300)
    plt.close(fig); print("[fig] fig_cepi_gates")


def priority_map():
    """Deployment boundary by building system.

    One message: the S = 0.67 line splits the portfolio into systems whose
    high-risk units enter a MAINTENANCE DECISION TASK and systems whose
    high-risk units must VERIFY CONDITION FIRST.  Class is encoded three ways
    (region shading, marker shape, marker fill) so the figure survives
    greyscale printing and colour-vision deficiency.  Marker size is *not*
    used: mean R varies little among the systems that matter and its largest
    excursions belong to the two micro-systems, so size encoding would shrink
    exactly the points the reader must notice.
    """
    pm = pd.read_csv(TAB / "cepi_priority_map.csv").copy()
    pm["x"] = pm["burden_share"] * 100.0
    pm["name"] = pm["system_desc"].str.strip()
    # sentence case, consistently
    pm["name"] = pm["name"].map(lambda s: s[0].upper() + s[1:].lower())
    suff = pm["S"] >= 0.67

    fig, ax = plt.subplots(figsize=(6.3, 4.6))

    # ---- the threshold is the figure ------------------------------------
    ax.axhspan(0.67, 1.34, facecolor=NAVY, alpha=0.05, lw=0, zorder=0)
    ax.axhspan(-0.12, 0.67, facecolor=CORAL, alpha=0.06, lw=0, zorder=0)
    # second, hatched span: keeps the two regions separable in greyscale
    ax.axhspan(-0.12, 0.67, facecolor="none", edgecolor=CORAL, alpha=0.16,
               hatch="///", lw=0.0, zorder=0)
    ax.axhline(0.67, color=CORAL_D, lw=2.1, ls=(0, (5.5, 2.4)), zorder=3)
    ax.text(0.097, 0.672, "$S = 0.67$ decision threshold", ha="left", va="center",
            fontsize=9.8, color=CORAL_D, fontweight="bold", zorder=6,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9))

    # ---- region banners, each inside the region it describes -------------
    # (no proxy glyphs: they would read as data points; the marker shape is
    #  identified by the region each banner sits in)
    ytr = ax.get_yaxis_transform()  # x in axes fraction, y in data units
    ax.text(0.975, 0.775, "CERTIFIED", transform=ytr, ha="right", va="center",
            fontsize=9.6, fontweight="bold", color=NAVY_D, zorder=6)
    ax.text(0.975, 0.722, "high-risk units $\\rightarrow$ MAINTENANCE DECISION TASK",
            transform=ytr, ha="right", va="center", fontsize=8.6, color=NAVY_D, zorder=6)

    # placed low-right, in the empty part of the uncertified region: every
    # uncertified system sits at a small labour share, i.e. on the left.
    ax.text(0.975, 0.300, "NOT CERTIFIED", transform=ytr, ha="right", va="center",
            fontsize=9.6, fontweight="bold", color=CORAL_D, zorder=6)
    ax.text(0.975, 0.247, "high-risk units $\\rightarrow$ VERIFY CONDITION FIRST",
            transform=ytr, ha="right", va="center", fontsize=8.6, color=CORAL_D, zorder=6)

    # ---- the systems -----------------------------------------------------
    ax.scatter(pm.loc[suff, "x"], pm.loc[suff, "S"], marker="o", s=96,
               facecolor=NAVY_D, edgecolor="white", lw=1.2, zorder=5)
    ax.scatter(pm.loc[~suff, "x"], pm.loc[~suff, "S"], marker="s", s=96,
               facecolor="white", edgecolor=CORAL_D, lw=2.1, zorder=5)

    # ---- grouped annotation for the eight systems sitting at S = 1.0 -----
    core = pm[(pm["S"] == 1.0)].sort_values("x")
    x0, x1 = core["x"].min() * 0.86, core["x"].max() * 1.14
    yb = 1.055
    ax.plot([x0, x0, x1, x1], [yb - 0.026, yb, yb, yb - 0.026], color=NAVY_D, lw=0.9,
            alpha=0.55, clip_on=False, zorder=4)
    # share and member names read from the data, never hand-typed: the share was
    # previously a literal "93%" that silently went stale when the denominator
    # was corrected to the nine-campus panel
    core_share = float(core["burden_share"].sum()) * 100
    # lower-case the descriptive names but leave acronyms (HVAC) alone
    names = [str(s) if str(s).isupper() else str(s).lower() for s in
             core.sort_values("burden_share", ascending=False)["system_desc"]]
    half = (len(names) + 1) // 2
    ax.text(np.sqrt(x0 * x1), yb + 0.030,
            f"{len(core)} core systems at $S = 1.0$: {core_share:.0f}% of "
            "portfolio reactive labor\n"
            + " · ".join(names[:half]) + "\n" + " · ".join(names[half:]),
            ha="center", va="bottom", fontsize=8.6, color=NAVY_D, linespacing=1.45,
            clip_on=False, zorder=6)

    # ---- direct labels: every record-insufficient system, plus the four
    #      record-sufficient systems that sit closest to the threshold -----
    off = {   # (dx pt, dy pt, ha, va)
        "Exterior enclosure": (11, -3, "left", "center"),
        "Site improvements": (-11, -3, "right", "center"),
        "Roofing": (11, -3, "left", "center"),
        "Special construction": (12, -3, "left", "center"),
        "Conveying": (12, -3, "left", "center"),
        "Stairs": (12, -3, "left", "center"),
        "Site mechanical utilities": (12, -3, "left", "center"),
        "Selective building demolition": (12, -3, "left", "center"),
    }
    for _, r in pm.iterrows():
        if r["name"] not in off:
            continue
        dx, dy, ha, va = off[r["name"]]
        ax.annotate(r["name"], (r.x, r.S), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va=va, fontsize=9.0, zorder=6,
                    color=NAVY_D if r.S >= 0.67 else CORAL_D)

    ax.set_xscale("log")
    # lower bound set from the data with headroom, so the smallest system's
    # marker is not sliced in half by the axis
    ax.set_xlim(float(pm["x"].min()) * 0.62, 62)
    ax.set_ylim(-0.10, 1.34)
    ax.set_xticks([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 40])
    ax.set_xticklabels(["0.1", "0.2", "0.5", "1", "2", "5", "10", "20", "40"])
    ax.minorticks_off()
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("Share of portfolio reactive labor (%, log scale)", fontsize=10.8)
    ax.set_ylabel("Record-evidence certificate  $S$", fontsize=10.8)
    ax.tick_params(labelsize=10.0)
    figstyle.despine(ax)
    fig.savefig(FIG / "fig_cepi_priority_map.pdf"); fig.savefig(FIG / "fig_cepi_priority_map.png", dpi=300)
    plt.close(fig); print("[fig] fig_cepi_priority_map")


def sensitivity():
    s = json.load(open(MET / "cepi_summary.json"))
    qz = json.load(open(MET / "cepi_queue.json"))
    fig = plt.figure(figsize=(6.3, 6.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.82, 1.0],
                          hspace=0.62, wspace=0.42,
                          left=0.20, right=0.97, top=0.90, bottom=0.09)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    # (a) route shares under the four action labels
    ax = ax_a
    cls_order = ["records_informative_high_risk", "evidence_insufficient_high_risk",
                 "watchlist", "routine_monitoring"]
    labels = ["OPEN MAINTENANCE\nDECISION TASK", "VERIFY CONDITION\nFIRST",
              "EVIDENCE\nWATCHLIST", "CONTINUE STANDARD\nMAINTENANCE"]
    cols = [NAVY, CORAL, AMBER, LGREY]
    vals = [s["class_shares"][k] * 100 for k in cls_order]
    y = np.arange(len(cls_order))[::-1]
    bars = ax.barh(y, vals, color=cols, edgecolor="white", height=0.66)
    for yi, v in zip(y, vals):
        ax.text(v + 1.2, yi, f"{v:.1f}%", va="center", fontsize=8.3, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8.6)
    ax.set_xlim(0, 100); ax.set_xlabel("Share of scored units (%)")
    figstyle.despine(ax); ax.set_title("(a)", fontsize=10)

    # (b) verification-queue size across risk cuts x sufficiency cuts
    ax = ax_b
    grid = qz["queue_size_sensitivity"]
    rlabs = ["top5pct", "top10pct", "top15pct", "top20pct"]
    rticks = ["top 5%", "top 10%", "top 15%", "top 20%"]
    scuts = ["S<0.5", "S<0.67", "S<0.85"]
    scols = ["#d9e5f1", NAVY, "#6f9cc4"]
    w = 0.26
    xs = np.arange(len(rlabs))
    for i, (sc, col) in enumerate(zip(scuts, scols)):
        vals = [grid[r][sc] for r in rlabs]
        bars = ax.bar(xs + (i - 1) * w, vals, width=w, color=col,
                      edgecolor="white", label=sc.replace("S<", "$S<$"))
        for b, v in zip(bars, vals):
            if v >= 100:
                ax.text(b.get_x() + b.get_width() / 2, v * 1.05, f"{v:,}",
                        ha="center", fontsize=6.4, color=INK, rotation=90)
    ax.axvline(1 + 0.0, color=CORAL_D, lw=0.9, ls=(0, (3, 2)), alpha=0.55)
    ax.text(1.02, 0.97, "primary", transform=ax.get_xaxis_transform(),
            fontsize=7.2, color=CORAL_D, ha="left", va="top", rotation=90)
    ax.set_yscale("log"); ax.set_ylim(8, 3e4)
    ax.set_xticks(xs); ax.set_xticklabels(rticks, fontsize=8.4)
    ax.set_xlabel("Risk cut (share of units flagged high-risk)")
    ax.set_ylabel("Verification-queue size (units, log)")
    ax.legend(fontsize=8, loc="upper left")
    figstyle.despine(ax); ax.set_title("(b)", fontsize=10)

    # (c) # evidence-insufficient systems across gate-weight schemes + hard-fail
    ax = ax_c
    gw = s["sensitivity"]["gate_weights"]
    hf = s["sensitivity"]["hard_fail"]
    schemes = ["equal", "reliability_heavy", "evidence_gap_heavy"]
    slab = ["equal", "reliability-\nheavy", "gap-\nheavy"]
    vals = [gw[k]["n_systems_insufficient"] for k in schemes]
    vals += [hf["off"]["n_systems_insufficient"]]
    slab += ["no\nhard-fail"]
    xpos = np.arange(len(vals))
    bars = ax.bar(xpos, vals, color=[NAVY, NAVY, NAVY, LGREY], width=0.62, edgecolor="white")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.08, str(v), ha="center",
                fontsize=9, color=INK, fontweight="bold")
    ax.set_xticks(xpos); ax.set_xticklabels(slab, fontsize=8)
    ax.set_ylabel("Evidence-insufficient systems\n($S<0.67$)")
    ax.set_ylim(0, max(vals) + 1.4)
    figstyle.despine(ax)
    ax.set_title("(c)", fontsize=10)

    fig.savefig(FIG / "fig_cepi_sensitivity.pdf"); fig.savefig(FIG / "fig_cepi_sensitivity.png", dpi=300)
    plt.close(fig); print("[fig] fig_cepi_sensitivity")


if __name__ == "__main__":
    gate_heatmap()
    priority_map()
    sensitivity()
