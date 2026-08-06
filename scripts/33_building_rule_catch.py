#!/usr/bin/env python
"""
Does the building-level rule catch anything the cell screen misses?

C1 says: when any system in a building is high-burden at t, inspect the whole building. But the
M1b ablation shows building features add no predictive value, and the building's other cells may
already rank high. So we quantify the INCREMENTAL catch of the building rule directly.

For each true high-burden event (cell high-burden at t+1, p75), under leakage-controlled LOUO:
  * captured     = the cell is in the campus top-10% by model score at t (caught by the screen);
  * triggered    = at t, some OTHER system in the same building is currently high-burden
                   (so the building rule would have flagged this cell);
  * building-rule incremental catch = events that are NOT captured by the screen but ARE in a
    triggered building.
We report recall of the screen alone, the incremental recall added by the building rule, the
combined recall, and the extra inspection load the building rule imposes.

Outputs -> results/metrics/building_rule_catch.json
"""
from __future__ import annotations
import os
os.environ.setdefault("FMSCREEN_THREADS", "8")
import sys, json
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fmscreen import config as C, features as FE, models as M, validation as V

CELL = [C.COL_UNIV, C.COL_BUILDING, C.COL_SYSTEM]
TOPK = C.TOPK_BUDGET


def main():
    panel = pd.read_parquet(C.DATA_PROCESSED / "panel_quarter.parquet")
    # current-quarter high-burden (p75 of positive current UPM labour, full data for the descriptive trigger)
    thr = panel[panel["upm_labour"] > 0].groupby(C.COL_SYSTEM, observed=True)["upm_labour"].quantile(0.75)
    panel["sev_now"] = (panel["upm_labour"] > panel[C.COL_SYSTEM].map(thr).fillna(np.inf)).astype(int)
    bq_sev = panel.groupby([C.COL_UNIV, C.COL_BUILDING, "period_q"], observed=True)["sev_now"].transform("sum")
    panel["sib_trigger"] = ((bq_sev - panel["sev_now"]) >= 1).astype(int)   # a sibling system is high-burden now

    rows = []
    for u, train, test in V.louo_folds(panel):
        art = FE.assemble_Xy(train, test, "M1", "severity_labour", pctl=75)
        if art["y_train"].sum() == 0 or len(art["y_test"]) == 0:
            continue
        model = M.make_gbdt(art["num_cols"], art["cat_cols"])
        scores = M.fit_predict(model, art["X_train"], art["y_train"], art["X_test"])
        k = max(1, int(np.ceil(TOPK * len(scores))))
        cut = np.sort(scores)[::-1][k - 1]
        df = test[[C.COL_UNIV, C.COL_BUILDING, C.COL_SYSTEM, "period_q", "sib_trigger"]].copy()
        df["captured"] = (scores >= cut).astype(int)
        df["y"] = art["y_test"]                       # high-burden at t+1
        rows.append(df)
    d = pd.concat(rows, ignore_index=True)

    events = d[d["y"] == 1]
    n = len(events)
    cap = float(events["captured"].mean())
    trig = float(events["sib_trigger"].mean())
    cap_or_trig = float(((events["captured"] == 1) | (events["sib_trigger"] == 1)).mean())
    incremental = float(((events["captured"] == 0) & (events["sib_trigger"] == 1)).mean())
    missed_both = float(((events["captured"] == 0) & (events["sib_trigger"] == 0)).mean())

    # inspection-load cost of the building rule: cells flagged ONLY by the building rule
    # (not in top-10%) as a fraction of all scored cells, and the screen's own 10%.
    extra_load = float(((d["captured"] == 0) & (d["sib_trigger"] == 1)).mean())
    screen_load = float((d["captured"] == 1).mean())

    out = {
        "n_high_burden_events": int(n),
        "recall_screen_alone": cap,
        "recall_building_rule_alone": trig,
        "recall_screen_plus_building_rule": cap_or_trig,
        "incremental_recall_from_building_rule": incremental,
        "events_missed_by_both": missed_both,
        "extra_inspection_load_building_rule_only": extra_load,
        "screen_inspection_load": screen_load,
        "reading": (f"The screen alone captures {cap:.0%} of high-burden events; the building rule "
                    f"adds {incremental:.0%} that the screen missed, raising combined recall to "
                    f"{cap_or_trig:.0%}, at an extra inspection load of {extra_load:.0%} of cells."),
    }
    json.dump(out, open(C.METRICS / "building_rule_catch.json", "w"), indent=2, default=float)
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
