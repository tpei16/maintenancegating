#!/usr/bin/env python
"""
Figure: escalating model capacity does not move the ceiling.

Reads the two capacity experiments and draws one exhibit for the manuscript
section "Higher-Capacity Configurations Do Not Improve Held-Out Ranking":

  results/wp1/chronology_value.json  representation escalation (event sequence)
  results/wp3/transformer.json       architecture and adaptation escalation

Every arm is scored on the same nine leave-one-campus-out folds with the same
top-decile budget, so the per-campus points are paired and the medians are
directly comparable. The incumbent's median is drawn as the reference the whole
figure is read against.

Output -> results/figures/fig_capacity_ladder.{pdf,png}
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figcheck
from fmscreen import figstyle
from fmscreen.figstyle import NAVY, TEAL, AMBER, NAVY_D, CORAL_D, SLATE, INK, LGREY

FIGS = ROOT / "results" / "figures"


def load():
    wp1 = json.load(open(ROOT / "results" / "wp1" / "chronology_value.json"))
    p3 = ROOT / "results" / "wp3" / "transformer.json"
    wp3 = json.load(open(p3)) if p3.exists() else None
    return wp1, wp3


def main():
    figstyle.apply()
    wp1, wp3 = load()
    w1 = wp1["per_campus"]

    # (label, per-campus dict, block index). Block 0 reference, 1 representation,
    # 2 architecture and adaptation.
    # (label, per-campus dict, block, is_control)
    rows = [("Incumbent: boosted trees on the quarterly panel", w1["A0"], 0, False)]
    rows += [
        ("Event-sequence encoder on the raw work-order stream", w1["A1"], 1, False),
        ("event order shuffled (control)", w1["A2"], 1, True),
        ("elapsed time removed (control)", w1["A3"], 1, True),
        ("events pooled, no chronology at all (control)", w1["A4"], 1, True),
    ]
    if wp3 is not None:
        pc = wp3["per_campus"]
        add = []
        if "T1" in pc:
            add.append(("Feature-tokenizer transformer on the same features",
                        pc["T1"], 2, False))
        if "T3" in pc:
            add += [
                ("Context-adaptive transport on the incumbent score", pc["T3"], 2, False),
                ("context descriptor withheld (control)", pc["T3n"], 2, True),
                ("campus descriptors deranged (control)", pc["T3p"], 2, True),
            ]
        rows += add

    ref = float(np.median(list(w1["A0"].values())))

    n = len(rows)
    fig, ax = plt.subplots(figsize=(7.2, 0.40 * n + 1.05))
    block_fill = {0: LGREY, 1: NAVY, 2: TEAL}
    block_line = {0: SLATE, 1: NAVY_D, 2: "#4da395"}

    ys = np.arange(n)[::-1]
    # hairline separators between the three escalation blocks
    blocks = [r[2] for r in rows]
    for i in range(1, n):
        if blocks[i] != blocks[i - 1]:
            ax.axhline(ys[i] + 0.5, color="#dfe4e9", lw=0.7, zorder=0)
    for y, (lab, d, blk, _ctl) in zip(ys, rows):
        v = np.array(list(d.values()), dtype=float)
        ax.scatter(v, np.full_like(v, y), s=17, color=block_fill[blk],
                   edgecolor="none", zorder=2, alpha=0.85)
        med = float(np.median(v))
        # no white ring on the median diamond: where a campus point sits at the
        # median the ring bit a crescent out of it, which read as a stray glyph
        ax.scatter([med], [y], s=62, marker="D", color=block_line[blk],
                   edgecolor="none", zorder=4)
        # opaque badge: several medians sit within hundredths of the incumbent,
        # so the reference rule runs vertically through their labels
        ax.annotate(f"{med:.2f}", (med, y), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8.2, color=INK,
                    fontweight="bold", zorder=6,
                    bbox=dict(boxstyle="square,pad=0.12", fc="white",
                              ec="none", alpha=1.0))

    ax.axvline(ref, ls="--", color=CORAL_D, lw=1.4, zorder=1)
    ax.annotate("incumbent median", (ref - 0.06, -0.42), fontsize=8.2,
                color=CORAL_D, ha="right", va="bottom")

    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.6)
    lab_weight = {r[0]: ("normal" if r[3] else "bold") for r in rows}
    for t in ax.get_yticklabels():
        t.set_fontweight(lab_weight.get(t.get_text(), "normal"))
    ax.set_xlabel("Top-10% precision lift on the held-out campus\n"
                  "(one point per campus, diamond marks the median)")
    ax.set_ylim(-0.55, n - 0.12)
    lo = min(min(map(float, d.values())) for _, d, _, _ in rows)
    hi = max(max(map(float, d.values())) for _, d, _, _ in rows)
    ax.set_xlim(lo - 0.45, hi + 0.55)
    ax.grid(axis="x", color="#e3e7eb", lw=0.6)
    ax.set_axisbelow(True)
    figstyle.despine(ax, left=False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    figcheck.assert_clean(fig, ax, "fig_capacity_ladder")
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"fig_capacity_ladder.{ext}", dpi=300,
                    bbox_inches="tight")
    print(f"wrote {FIGS/'fig_capacity_ladder.pdf'}  ({n} arms, ref={ref:.2f})")


if __name__ == "__main__":
    main()
