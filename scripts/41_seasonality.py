#!/usr/bin/env python
"""Descriptive seasonality of reactive demand for the top building services.

Context for the weather/season discussion: the share of each service's UPM work
orders and of its high-burden cell-quarters falling in each calendar quarter,
computed from the full panel. Descriptive only; the screening model needs no
external weather feed because seasonal variation reaches it through the record
features (see the weather ablation).

Outputs -> results/figures/fig_seasonality.(png|pdf)
           results/tables/seasonality_by_service.csv
"""
from __future__ import annotations
import sys
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fmscreen import config as C
from fmscreen import figstyle
from fmscreen.figstyle import NAVY, CORAL, SLATE, INK, LGREY

figstyle.apply()

SERVICES = ["HVAC", "Plumbing", "Electrical"]
COLS = {"HVAC": NAVY, "Plumbing": CORAL, "Electrical": LGREY}


def main():
    p = pd.read_parquet(C.DATA_PROCESSED / "panel_quarter.parquet")
    p = p[p["SystemDescription"].isin(SERVICES)].copy()

    # current-quarter high-burden marker (descriptive, full data): system p75 of
    # positive current UPM labour
    thr = (p[p["upm_labour"] > 0]
           .groupby("SystemDescription", observed=True)["upm_labour"].quantile(0.75))
    p["hb_now"] = (p["upm_labour"] > p["SystemDescription"].map(thr)).astype(int)

    rows = []
    for svc, g in p.groupby("SystemDescription", observed=True):
        orders = g.groupby("quarter", observed=True)["upm_count"].sum()
        hb = g.groupby("quarter", observed=True)["hb_now"].sum()
        for q in (1, 2, 3, 4):
            rows.append({"service": svc, "calendar_quarter": q,
                         "upm_order_share": float(orders.get(q, 0) / orders.sum()),
                         "high_burden_share": float(hb.get(q, 0) / hb.sum())})
    out = pd.DataFrame(rows)
    out.to_csv(C.TABLES / "seasonality_by_service.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.9))
    x = np.arange(4); w = 0.26
    for ax, col, title in [
        (axes[0], "upm_order_share", "Share of the Service's UPM Work Orders"),
        (axes[1], "high_burden_share", "Share of the Service's High-Burden Cell-Quarters"),
    ]:
        for i, svc in enumerate(SERVICES):
            d = out[out["service"] == svc].sort_values("calendar_quarter")
            ax.bar(x + (i - 1) * w, 100 * d[col].to_numpy(), width=w,
                   color=COLS[svc], edgecolor="white", lw=0.8, label=svc)
        ax.axhline(25, color=INK, lw=0.9, ls=(0, (4, 2)))
        # sits above the tallest Q3/Q4 bar in both panels, clear of every bar
        ax.text(3.42, 26.6, "uniform (25%)", ha="right", fontsize=7.5, color=INK)
        ax.set_xticks(x); ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4"])
        ax.set_ylim(0, 36)
        ax.set_ylabel("share of annual total (%)")
        # the title was computed but never applied, so the two panels were
        # indistinguishable on the page
        ax.set_title(title, fontsize=9)
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(C.FIGURES / "fig_seasonality.png", dpi=300)
    fig.savefig(C.FIGURES / "fig_seasonality.pdf")
    plt.close(fig)

    print(out.pivot(index="service", columns="calendar_quarter",
                    values="upm_order_share").round(3).to_string())
    print("[fig] fig_seasonality")


if __name__ == "__main__":
    main()
