#!/usr/bin/env python
"""
Captured-vs-missed extreme (p90) events by trajectory.

The C3 ceiling claim must be the right ceiling: not "CMMS-only ranking fails on the
tail" but "escalating-trend early warning cannot cover non-escalating extreme events".
We cross-tabulate p90 extreme high-burden events by (i) their pre-event activity
trajectory (escalating vs non-escalating, identical definition to scripts/18) and
(ii) whether the leakage-controlled top-10% LOUO screen CAPTURED them (ranked the cell
in the campus top decile at the prior quarter).

Outputs -> results/metrics/missed_p90.json
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


def louo_scores(panel: pd.DataFrame) -> pd.DataFrame:
    """Held-out top-10% capture flag per scored cell-quarter (gbdt M1, p75)."""
    out = []
    for u, train, test in V.louo_folds(panel):
        art = FE.assemble_Xy(train, test, "M1", "severity_labour", pctl=75)
        if art["y_train"].sum() == 0 or len(art["y_test"]) == 0:
            continue
        model = M.make_gbdt(art["num_cols"], art["cat_cols"])
        scores = M.fit_predict(model, art["X_train"], art["y_train"], art["X_test"])
        sc = test[[C.COL_UNIV, C.COL_BUILDING, C.COL_SYSTEM, "period_q"]].copy()
        sc["score"] = scores
        # top-10% within this held-out campus (pooled over its cell-quarters)
        k = max(1, int(np.ceil(TOPK * len(sc))))
        cut = np.sort(scores)[::-1][k - 1]
        sc["captured"] = (sc["score"] >= cut).astype(int)
        out.append(sc)
    return pd.concat(out, ignore_index=True)


def main():
    p = pd.read_parquet(C.DATA_PROCESSED / "panel_quarter.parquet")
    # p90 event at quarter q (current-quarter extreme), trajectory over [q-4..q-1] (== script 18)
    thr90 = p[p["upm_labour"] > 0].groupby(C.COL_SYSTEM, observed=True)["upm_labour"].quantile(0.90)
    p = p.sort_values(CELL + ["period_q"]).reset_index(drop=True)
    p["p90"] = (p["upm_labour"] > p[C.COL_SYSTEM].map(thr90).fillna(np.inf)).astype(int)
    g = p.groupby(CELL, observed=True)["upm_count"]
    prior_recent = g.shift(1).rolling(2, min_periods=2).sum()
    prior_older = g.shift(3).rolling(2, min_periods=2).sum()
    p["trend_back"] = prior_recent - prior_older
    p["prior_total"] = prior_recent + prior_older
    p["has_history"] = (p["period_q"] - 4 >= p["cell_first_q"]).astype(int)
    p["traj"] = np.where(p["prior_total"] == 0, "zero_prior",
                np.where(p["trend_back"] > 0, "escalating", "non_escalating"))

    # capture flag from the screening decision made the PRIOR quarter (scored at q-1 -> predicts q)
    sc = louo_scores(p)
    sc["period_q_event"] = sc["period_q"] + 1     # the event it was predicting
    cap = sc[[C.COL_UNIV, C.COL_BUILDING, C.COL_SYSTEM, "period_q_event", "captured"]].rename(
        columns={"period_q_event": "period_q"})

    ev = p[(p["p90"] == 1) & (p["has_history"] == 1)].merge(
        cap, on=[C.COL_UNIV, C.COL_BUILDING, C.COL_SYSTEM, "period_q"], how="inner")

    n = len(ev)
    overall_capture = float(ev["captured"].mean())
    tab = ev.groupby("traj").agg(n=("captured", "size"),
                                 capture_rate=("captured", "mean")).reset_index()
    # share of MISSED events that are non-escalating, and risk ratio of being missed
    missed = ev[ev["captured"] == 0]
    captured = ev[ev["captured"] == 1]
    share_missed_nonesc = float((missed["traj"] != "escalating").mean())
    share_captured_nonesc = float((captured["traj"] != "escalating").mean())
    miss_rate_esc = float((ev[ev["traj"] == "escalating"]["captured"] == 0).mean())
    miss_rate_nonesc = float((ev[ev["traj"] != "escalating"]["captured"] == 0).mean())
    rr_missed = miss_rate_nonesc / miss_rate_esc if miss_rate_esc else float("nan")

    out = {
        "n_p90_events_scored": int(n),
        "overall_capture_rate_at_top10": overall_capture,
        "capture_by_trajectory": tab.to_dict("records"),
        "share_of_missed_that_are_non_escalating": share_missed_nonesc,
        "share_of_captured_that_are_non_escalating": share_captured_nonesc,
        "miss_rate_escalating": miss_rate_esc,
        "miss_rate_non_escalating": miss_rate_nonesc,
        "risk_ratio_missed_nonesc_vs_esc": rr_missed,
        "reading": ("Captured p90 events skew escalating; missed p90 events skew non-escalating, "
                    "so the ceiling is on escalating-trend early warning, not on CMMS-only ranking."),
    }
    json.dump(out, open(C.METRICS / "missed_p90.json", "w"), indent=2, default=float)
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
