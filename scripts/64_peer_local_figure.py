#!/usr/bin/env python
"""
64_peer_local_figure.py -- evidence from peer campuses against a campus's own
records, drawn from the outputs of 60_local_history_baseline.py and
68_direct_local_validation.py.

Panel (a): for each campus, the number of building systems supported from the
peer campuses and the number validated directly on the campus's own outcomes
dated 2016 or earlier, with the campuses ordered by the length of their record.
Panel (b): the top-decile lift realized in 2017 to 2019 for every evaluable
campus-system pair, grouped by direct local validation and by the peer verdict.

Reads  results/tables/direct_local_validation.csv
       results/metrics/local_history_baseline.json
Writes results/figures/fig_peer_local.{pdf,png}
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fmscreen import config as C
from fmscreen import figstyle
from fmscreen.figstyle import NAVY, CORAL, SLATE, INK, GRIDC, NAVY_D, CORAL_D

figstyle.apply()
TAB, MET, FIG = C.TABLES, C.METRICS, C.FIGURES
BLUE_L, CORAL_L = "#d3e3f2", "#f0cdd2"


def main():
    d = pd.read_csv(TAB / "direct_local_validation.csv", dtype={"campus": str})
    meta = json.load(open(MET / "local_history_baseline.json"))["per_campus"]
    d["peer"] = d.supported_peer.fillna(False).astype(bool)
    d["local"] = d.all_verdict.eq("validated")

    order = sorted(meta, key=lambda c: (-meta[c]["quarters_before_2017"], int(c)))
    peer = [int(d.loc[d.campus == c, "peer"].sum()) for c in order]
    local = [int(d.loc[d.campus == c, "local"].sum()) for c in order]
    qs = [meta[c]["quarters_before_2017"] for c in order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.0),
                                   gridspec_kw={"width_ratios": [1.0, 0.85], "wspace": 0.80})

    # ---- (a) building systems by campus
    x = np.arange(len(order)); w = 0.38
    ax1.bar(x - w / 2, peer, w, color=BLUE_L, edgecolor=NAVY_D, lw=0.8, label="supported from peers")
    ax1.bar(x + w / 2, local, w, color=CORAL_L, edgecolor=CORAL_D, lw=0.8, label="validated on own records")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{c}\n({q})" for c, q in zip(order, qs)], fontsize=8.5)
    ax1.set_xlabel("Campus (quarters of own records before 2017)", fontsize=9)
    ax1.set_ylabel("Building systems", fontsize=9)
    ax1.set_ylim(0, 17.5); ax1.set_yticks([0, 3, 6, 9, 12])
    ax1.legend(loc="upper right", ncol=1, fontsize=8.2, frameon=True, edgecolor=INK,
               fancybox=False, framealpha=1.0).get_frame().set_linewidth(0.5)
    ax1.set_title("(a)", loc="left", fontsize=9.5, fontweight="bold")
    ax1.grid(False); ax1.grid(axis="y", color=GRIDC, lw=0.5); ax1.set_axisbelow(True)

    # ---- (b) realized lift by local validation and peer verdict
    e = d[d.evaluable_2017_2019].copy()
    e["cls"] = np.select(
        [e.local & e.peer, e.local & ~e.peer, ~e.local & e.peer, ~e.local & ~e.peer],
        [0, 1, 2, 3])
    labels = ["validated locally;\nsupported by peers", "validated locally;\nnot supported by peers",
              "no local evaluation;\nsupported by peers", "no local evaluation;\nnot supported by peers"]
    rng = np.random.default_rng(7)
    ypos = [3, 2, 1, 0]
    ticks = []
    for k in range(4):
        v = e.loc[e.cls == k, "lift_2017_2019"].to_numpy()
        jit = rng.uniform(-0.24, 0.24, len(v))
        col, edge = (BLUE_L, NAVY_D) if k in (0, 2) else (CORAL_L, CORAL_D)
        ax2.scatter(v, ypos[k] + jit, s=15, facecolor=col, edgecolor=edge, lw=0.7, zorder=3)
        ax2.plot([np.median(v)] * 2, [ypos[k] - 0.34, ypos[k] + 0.34], color=INK, lw=1.4, zorder=4)
        ticks.append(f"{labels[k]} (n = {len(v)})")
    ax2.axvline(2.0, color=SLATE, lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax2.set_yticks(ypos); ax2.set_yticklabels(ticks, fontsize=8.2)
    ax2.set_ylim(-0.6, 3.6)
    ax2.set_xlabel("Top-decile lift, 2017 to 2019", fontsize=9)
    ax2.set_xlim(0, max(9.0, float(e.lift_2017_2019.max()) + 0.6))
    ax2.set_title("(b)", loc="left", fontsize=9.5, fontweight="bold")
    ax2.grid(False); ax2.grid(axis="x", color=GRIDC, lw=0.5); ax2.set_axisbelow(True)

    for ax in (ax1, ax2):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=8.5)
    ax2.tick_params(axis="y", labelsize=8.2, length=0)

    fig.savefig(FIG / "fig_peer_local.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig_peer_local.png", dpi=220, bbox_inches="tight")
    counts = e.groupby("cls").agg(n=("lift_2017_2019", "size"),
                                  hold=("lift_2017_2019", lambda s: int((s >= 2).sum())),
                                  median=("lift_2017_2019", "median"))
    print(counts.round(3).to_string())
    print("supported from peers by campus", dict(zip(order, peer)), "validated locally", dict(zip(order, local)))


if __name__ == "__main__":
    main()
