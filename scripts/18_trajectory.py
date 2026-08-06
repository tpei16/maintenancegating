#!/usr/bin/env python
"""
Analysis Y — severe-tail trajectory decomposition (explains the severity ceiling).

Among cells that experience a p90 severe event, what fraction were on an
ESCALATING trajectory (rising activity in the prior 4 quarters, predictable from
trends) versus NON-ESCALATING (flat/declining prior activity, i.e. sudden onset,
not predictable)? A matched-control group (same system, same campus, non-event
cell-quarters with equal history) guards against regression-to-the-mean.

This quantifies the composition of the severity ceiling (PR-AUC 0.40 at p90):
if a substantial fraction of p90 events have no escalating prior signal, that
structurally explains why record-based ranking of the severe tail is hard.

Outputs -> results/metrics/trajectory.json
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fmscreen import config as C

CELL = [C.COL_UNIV, C.COL_BUILDING, C.COL_SYSTEM]


def main():
    p = pd.read_parquet(C.DATA_PROCESSED / "panel_quarter.parquet")
    thr90 = p[p["upm_labour"] > 0].groupby(C.COL_SYSTEM, observed=True)["upm_labour"].quantile(0.90)
    p["p90"] = (p["upm_labour"] > p[C.COL_SYSTEM].map(thr90).fillna(np.inf)).astype(int)

    p = p.sort_values(CELL + ["period_q"]).reset_index(drop=True)
    g = p.groupby(CELL, observed=True)["upm_count"]
    # backward trajectory over the prior 4 quarters [t-4..t-1]
    prior_recent = g.shift(1).rolling(2, min_periods=2).sum()   # t-1, t-2
    prior_older = g.shift(3).rolling(2, min_periods=2).sum()    # t-3, t-4
    p["prior_recent"] = prior_recent
    p["prior_older"] = prior_older
    p["prior_total"] = prior_recent + prior_older
    p["trend_back"] = prior_recent - prior_older
    p["has_history"] = (p["period_q"] - 4 >= p["cell_first_q"]).astype(int)

    def classify(df):
        esc = (df["trend_back"] > 0)
        zero_prior = (df["prior_total"] == 0)
        return {
            "n": int(len(df)),
            "escalating": float(esc.mean()),
            "non_escalating": float((~esc).mean()),
            "zero_prior_activity": float(zero_prior.mean()),
            "declining": float((df["trend_back"] < 0).mean()),
            "flat": float((df["trend_back"] == 0).mean()),
        }

    events = p[(p["p90"] == 1) & (p["has_history"] == 1)]
    # matched controls: non-p90 cell-quarters, same systems, same campuses, with history
    controls = p[(p["p90"] == 0) & (p["has_history"] == 1) & (p["active"] == 1)]

    out = {
        "p90_threshold_note": "system-specific p90 of positive UPM labour (full data)",
        "n_p90_events_with_history": int(len(events)),
        "events": classify(events),
        "controls_active": classify(controls),
    }
    # by-system non-escalating fraction
    bysys = []
    for s, sub in events.groupby(C.COL_SYSTEM, observed=True):
        if len(sub) < 50:
            continue
        bysys.append({"system": s,
                      "system_desc": p[p[C.COL_SYSTEM] == s]["SystemDescription"].iloc[0],
                      "n_events": int(len(sub)),
                      "non_escalating": float((sub["trend_back"] <= 0).mean()),
                      "zero_prior": float((sub["prior_total"] == 0).mean())})
    out["by_system"] = sorted(bysys, key=lambda d: -d["non_escalating"])
    out["headline"] = (f"{out['events']['non_escalating']:.0%} of p90 events were non-escalating "
                       f"(flat/declining prior activity); {out['events']['zero_prior_activity']:.0%} had "
                       f"zero prior-year activity (sudden onset). Controls: "
                       f"{out['controls_active']['non_escalating']:.0%} non-escalating.")
    json.dump(out, open(C.METRICS / "trajectory.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "by_system"}, indent=2))
    print("\nNon-escalating fraction by system (top):")
    for d in out["by_system"][:8]:
        print(f"  {d['system_desc']:24s} n={d['n_events']:5d} non-esc={d['non_escalating']:.0%} zero-prior={d['zero_prior']:.0%}")


if __name__ == "__main__":
    main()
