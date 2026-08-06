"""
39_queue.py -- gate-then-rank post-processing for the v7 framework.

Implements the locked decision logic:
  risk gate first (R >= 0.90 -> HIGH RISK), then the evidence gate for high-risk
  units (S >= 0.67 -> OPEN MAINTENANCE DECISION TASK, else VERIFY CONDITION FIRST); CEPI = R(1-S)
  is used ONLY to rank the verification queue. No independent CEPI threshold.

Outputs (real numbers only):
  results/metrics/cepi_queue.json
    - route counts under the primary cuts (must reconcile with 37_cepi.py classes)
    - verification-queue composition by system and by campus, ordered by CEPI
    - queue-size sensitivity across risk cuts (top 5/10/15/20%) x sufficiency cuts
"""
from __future__ import annotations
import json, glob
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MET = ROOT / "results" / "metrics"
TAB = ROOT / "results" / "tables"
PRED = ROOT / "data" / "processed" / "pred_louo"

ROUTES = {
    "plan_maintenance": "OPEN MAINTENANCE DECISION TASK (high R, S sufficient)",
    "verify_condition_first": "VERIFY CONDITION FIRST (high R, S insufficient)",
    "continue_standard": "CONTINUE STANDARD MAINTENANCE (low R, S sufficient)",
    "evidence_watchlist": "EVIDENCE WATCHLIST (low R, S insufficient)",
}


def load_units():
    frames = []
    for f in sorted(glob.glob(str(PRED / "*.parquet"))):
        d = pd.read_parquet(f)
        d = d[(d.target_kind == "severity_labour") & (d.pctl == 75)
              & (d.layer == "M1") & (d.model == "gbdt")].copy()
        d["R"] = d.groupby("held_out_university")["score"].rank(pct=True, method="average")
        frames.append(d[["held_out_university", "system", "y", "R"]])
    return pd.concat(frames, ignore_index=True)


def main():
    units = load_units()
    gates = json.load(open(MET / "cepi_gates.json"))
    Smap = {k: float(v) for k, v in gates["S_by_system"].items()}
    name = dict(pd.read_csv(TAB / "burden_by_system.csv")[
        ["SystemCode", "SystemDescription"]].values)
    name = {k: str(v).strip() for k, v in name.items()}

    units = units[units.system.isin(Smap)].copy()
    units["S"] = units.system.map(Smap)
    units["CEPI"] = units.R * (1.0 - units.S)

    R_CUT, S_CUT = 0.90, 0.67
    hiR = units.R >= R_CUT
    hiS = units.S >= S_CUT
    routes = {
        "plan_maintenance": int((hiR & hiS).sum()),
        "verify_condition_first": int((hiR & ~hiS).sum()),
        "continue_standard": int((~hiR & hiS).sum()),
        "evidence_watchlist": int((~hiR & ~hiS).sum()),
    }
    n = len(units)

    # ---- verification queue, ordered by CEPI --------------------------------
    q = units[hiR & ~hiS].copy().sort_values("CEPI", ascending=False)
    by_sys = (q.groupby("system")
              .agg(n=("R", "size"), mean_CEPI=("CEPI", "mean"),
                   mean_R=("R", "mean"), S=("S", "first"))
              .sort_values("mean_CEPI", ascending=False).reset_index())
    by_sys["system_desc"] = by_sys.system.map(name)
    by_camp = (q.groupby("held_out_university").size()
               .sort_values(ascending=False).reset_index(name="n"))
    # head of the queue = highest CEPI tier (demolition G=1.0, then stairs 0.6...)
    head = by_sys[["system", "system_desc", "n", "S", "mean_CEPI"]].to_dict("records")

    # ---- queue-size sensitivity: risk cut x sufficiency cut ------------------
    grid = {}
    for r_cut, klab in [(0.95, "top5pct"), (0.90, "top10pct"),
                        (0.85, "top15pct"), (0.80, "top20pct")]:
        hr = units.R >= r_cut
        grid[klab] = {}
        for s_cut in (0.50, 0.67, 0.85):
            grid[klab][f"S<{s_cut}"] = int((hr & (units.S < s_cut)).sum())

    out = {
        "logic": "gate-then-rank: R>=0.90 first; S>=0.67 routes high-risk; CEPI ranks the verification queue only",
        "primary_cuts": {"R": R_CUT, "S": S_CUT},
        "n_units": int(n),
        "route_counts": routes,
        "route_shares": {k: round(v / n, 4) for k, v in routes.items()},
        "route_labels": ROUTES,
        "verification_queue": {
            "n": int(len(q)),
            "by_system_cepi_order": head,
            "by_campus": by_camp.to_dict("records"),
            "share_of_queue_top2_systems": round(
                float(by_sys.head(2)["n"].sum() / max(len(q), 1)), 3),
        },
        "queue_size_sensitivity": grid,
    }
    json.dump(out, open(MET / "cepi_queue.json", "w"), indent=2, default=str)

    print("route counts:", routes, " (n=%d)" % n)
    print("\nverification queue by system (CEPI order):")
    print(by_sys.to_string(index=False))
    print("\nby campus:")
    print(by_camp.to_string(index=False))
    print("\nqueue-size grid:")
    for k, v in grid.items():
        print(" ", k, v)
    print("\nWrote results/metrics/cepi_queue.json")


if __name__ == "__main__":
    main()
