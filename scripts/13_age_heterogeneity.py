#!/usr/bin/env python
"""
Secondary stratification — building-age heterogeneity.

Building metadata (BuiltYear) is present for ~24% of records, so this is a
SUPPLEMENTARY stratification. We run the primary LOUO config (severity p75,
GBDT, M1) with building-tagged held-out predictions, attach building age, and
compare top-10% screening lift across age bands.

Outputs -> results/metrics/age_heterogeneity.csv + age_heterogeneity.json
"""
from __future__ import annotations
import os
N_JOBS = int(os.environ.get("N_JOBS", "3"))
os.environ.setdefault("FMSCREEN_THREADS", str(max(2, 24 // max(N_JOBS, 1))))
import sys, json
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fmscreen import config as C, io as F, validation as V, engine as E
from fmscreen import metrics as MET


def building_age_map() -> pd.DataFrame:
    df = F.load_raw(usecols=[C.COL_UNIV, C.COL_BUILDING, C.COL_BUILT_YEAR])
    df = df.dropna(subset=[C.COL_BUILDING])
    g = (df.groupby([C.COL_UNIV, C.COL_BUILDING], observed=True)[C.COL_BUILT_YEAR]
           .median().reset_index())
    g["building"] = g[C.COL_UNIV].astype("string") + "/" + g[C.COL_BUILDING].astype("string")
    return g[["building", C.COL_BUILT_YEAR]].rename(columns={C.COL_BUILT_YEAR: "built_year"})


def run_campus(u: str) -> pd.DataFrame:
    panel = pd.read_parquet(C.DATA_PROCESSED / "panel_quarter.parquet")
    ucol = panel[C.COL_UNIV].astype("string").to_numpy()
    train, test = panel[ucol != u], panel[ucol == u]
    r = E.evaluate_split(train, test, "M1", "gbdt", "severity_labour", pctl=75,
                         compute_ci=False, return_scores=True)
    if r is None:
        return pd.DataFrame()
    return pd.DataFrame({"held_out_university": u, "y": r["_y"], "score": r["_scores"],
                         "building": r["_test_building"], "system": r["_test_system"]})


def main():
    panel = pd.read_parquet(C.DATA_PROCESSED / "panel_quarter.parquet")
    campuses = V.universities(panel)
    parts = Parallel(n_jobs=N_JOBS, backend="loky")(delayed(run_campus)(u) for u in campuses)
    preds = pd.concat([p for p in parts if len(p)], ignore_index=True)
    age = building_age_map()
    preds = preds.merge(age, on="building", how="left")

    bins = [-np.inf, 1960, 1980, 2000, np.inf]
    labels = ["<1960", "1960-1979", "1980-1999", ">=2000"]
    preds["age_band"] = pd.cut(preds["built_year"], bins=bins, labels=labels)
    preds["age_band"] = preds["age_band"].cat.add_categories(["unknown"]).fillna("unknown")

    rows = []
    for band, g in preds.groupby("age_band", observed=True):
        if len(g) < 200 or g["y"].sum() < 20:
            continue
        m = MET.core_metrics(g["y"].to_numpy(), g["score"].to_numpy(), k=C.TOPK_BUDGET)
        rows.append({"age_band": str(band), "n_cells": int(len(g)), "n_pos": int(g["y"].sum()),
                     "base_rate": m["base_rate"], "lift_top10": m["lift_top10"],
                     "capture_top10": m["capture_top10"], "pr_auc": m["pr_auc"]})
    out = pd.DataFrame(rows)
    out.to_csv(C.METRICS / "age_heterogeneity.csv", index=False)
    coverage = float(preds["built_year"].notna().mean())
    json.dump({"built_year_coverage_in_panel": coverage, "bands": rows},
              open(C.METRICS / "age_heterogeneity.json", "w"), indent=2)
    print(f"[age] built-year coverage in panel cells: {coverage:.1%}", flush=True)
    print(out.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
