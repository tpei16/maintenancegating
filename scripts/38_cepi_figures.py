#!/usr/bin/env python
"""Gate-score figure: the five gate scores and the certification score of each
building system on the pooled map.

Reads cepi_gates.csv as written by 47_corrected_figure_inputs.py.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figcheck
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
CMAP = {1.0: "#d3e3f2", 0.5: "#eceef1", 0.0: "#f0cdd2"}
EDGE = {1.0: NAVY_D, 0.5: SLATE, 0.0: CORAL_D}
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
    ax.set_xticklabels(list(GATE_LABELS.values()) + ["Evidence\nscore $S$"], fontsize=8.6)
    ax.xaxis.set_ticks_position("top"); ax.xaxis.set_label_position("top")
    ax.tick_params(length=0)
    ax.set_yticks([n - 1 - i + 0.45 for i in range(n)])
    ax.set_yticklabels([f"{r.system}  {r.system_desc.strip()}"
                        for _, r in g.iterrows()], fontsize=8.0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(False)
    handles = [Patch(facecolor=CMAP[v], edgecolor=EDGE[v], label=LAB[v]) for v in (1.0, 0.5, 0.0)]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.42, -0.085),
              ncol=3, fontsize=8, handlelength=1.1, columnspacing=1.4,
              frameon=True, edgecolor=INK, borderpad=0.5).get_frame().set_linewidth(0.5)
    fig.savefig(FIG / "fig_cepi_gates.pdf"); fig.savefig(FIG / "fig_cepi_gates.png", dpi=300)
    plt.close(fig); print("[fig] fig_cepi_gates")


if __name__ == "__main__":
    gate_heatmap()
