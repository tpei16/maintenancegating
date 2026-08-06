#!/usr/bin/env python
"""
Analyses A+B — burden concentration (Lorenz/Gini) + temporal persistence decay.

Structural context (full 2002-2021, no train/test split). NOT headline findings;
they explain WHY simple rules work (chronic recurrence in concentrated hotspots)
and why the model adds only a modest increment.

A (Lorenz/Gini): rank cells (and buildings) by cumulative UPM labour; Gini, and
   top-10%/20% concentration shares.
B (persistence): for cells severe (system p75) at quarter t, fraction still severe
   at t+1/2/4/8; decay curve overall and by system; vs marginal severe rate.

Outputs -> results/metrics/burden_structure.json, results/tables/persistence_by_system.csv
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fmscreen import config as C

CELL = [C.COL_UNIV, C.COL_BUILDING, C.COL_SYSTEM]


def gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return float("nan")
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def lorenz_top_share(x: np.ndarray, top_frac: float) -> float:
    x = np.sort(np.asarray(x, dtype=float))[::-1]
    n = max(1, int(np.ceil(top_frac * len(x))))
    return float(x[:n].sum() / x.sum()) if x.sum() > 0 else float("nan")


def main():
    p = pd.read_parquet(C.DATA_PROCESSED / "panel_quarter.parquet")
    thr = p[p["upm_labour"] > 0].groupby(C.COL_SYSTEM, observed=True)["upm_labour"].quantile(0.75)
    p["sev"] = (p["upm_labour"] > p[C.COL_SYSTEM].map(thr).fillna(np.inf)).astype(int)

    # ---- A: concentration ----
    cell_tot = p.groupby(CELL, observed=True)["upm_labour"].sum().to_numpy()
    bld_tot = p.groupby([C.COL_UNIV, C.COL_BUILDING], observed=True)["upm_labour"].sum().to_numpy()
    conc = {
        "n_cells": int(len(cell_tot)), "n_buildings": int(len(bld_tot)),
        "gini_cells": gini(cell_tot), "gini_buildings": gini(bld_tot),
        "cell_top10_share": lorenz_top_share(cell_tot, 0.10),
        "cell_top20_share": lorenz_top_share(cell_tot, 0.20),
        "building_top10_share": lorenz_top_share(bld_tot, 0.10),
        "building_top20_share": lorenz_top_share(bld_tot, 0.20),
    }
    # Lorenz curve points (cells) for the figure
    xs = np.sort(cell_tot)
    lor = np.concatenate([[0], np.cumsum(xs) / xs.sum()])
    pop = np.linspace(0, 1, len(lor))
    pd.DataFrame({"pop_frac": pop, "labour_frac": lor}).to_csv(
        C.TABLES / "lorenz_curve_cells.csv", index=False)

    # ---- B: persistence decay ----
    p = p.sort_values(CELL + ["period_q"]).reset_index(drop=True)
    gsev = p.groupby(CELL, observed=True)["sev"]
    for k in (1, 2, 4, 8):
        p[f"sev_t{k}"] = gsev.shift(-k)
    sev_now = p[p["sev"] == 1]
    marginal = float(p[p["active"] == 1]["sev"].mean())  # severe rate among active cell-quarters
    decay = {"marginal_severe_rate_active": marginal}
    for k in (1, 2, 4, 8):
        col = sev_now[f"sev_t{k}"]
        decay[f"persist_t{k}"] = float(col.dropna().mean())
    conc["persistence"] = decay

    # persistence by system (t+1)
    rows = []
    for s, sub in sev_now.groupby(C.COL_SYSTEM, observed=True):
        if len(sub) < 100:
            continue
        rows.append({"system": s,
                     "system_desc": p[p[C.COL_SYSTEM] == s]["SystemDescription"].iloc[0],
                     "n_severe": int(len(sub)),
                     "persist_t1": float(sub["sev_t1"].dropna().mean()),
                     "persist_t2": float(sub["sev_t2"].dropna().mean()),
                     "persist_t4": float(sub["sev_t4"].dropna().mean()),
                     "persist_t8": float(sub["sev_t8"].dropna().mean())})
    pers = pd.DataFrame(rows).sort_values("persist_t1", ascending=False)
    pers.to_csv(C.TABLES / "persistence_by_system.csv", index=False)

    json.dump(conc, open(C.METRICS / "burden_structure.json", "w"), indent=2)
    print(json.dumps(conc, indent=2))
    print("\nPersistence by system (top):")
    print(pers.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
