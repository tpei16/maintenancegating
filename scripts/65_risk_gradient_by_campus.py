#!/usr/bin/env python
"""
65_risk_gradient_by_campus.py -- the risk-gradient gate with campus-specific
and with pooled base rates.

The risk-gradient gate cuts the within-campus-quarter risk percentile into five
bands and asks whether the high-burden rate rises across them.  The study's
definition compares the observed number of high-burden quarters in each band
with the number expected from each certifying campus's own base rate
(GRADIENT_BASE = "campus" in 45_certification_corrected.py).  The alternative
pools the campuses and compares each band's rate with one pooled base rate.
When the base rate of a building system differs strongly between campuses, a
campus with a high base rate lifts every band it occupies and the pooled
gradient flattens.  This script measures what the choice changes in the
certificates, the routes and the held-out check.

Everything else is the main pipeline (double nesting, R_quarter, capped
transfer gate, target-excluded trace table).

Outputs -> results/metrics/risk_gradient_by_campus.json
"""
from __future__ import annotations
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import sys, json, importlib.util
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fmscreen import config as C
from fmscreen import metrics as MET

_spec = importlib.util.spec_from_file_location(
    "cert45", Path(__file__).resolve().parent / "45_certification_corrected.py")
cc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cc)

RCOL = "R_quarter"


def heldout(single, outer, campuses):
    held = {"certified": [0, 0], "withheld": [0, 0]}
    for j in campuses:
        sj = single[single.university == j]
        for s, v in outer[j].items():
            gg = sj[sj.system == s]
            if gg["y"].sum() >= 10 and len(gg) >= 100:
                lift = MET.lift_at_topk(gg["y"].to_numpy(), gg["score"].to_numpy(), C.TOPK_BUDGET)
                k = "certified" if v["S"] >= cc.S_CUT else "withheld"
                held[k][0] += 1; held[k][1] += int(lift >= 2)
    return {k: {"n": v[0], "n_lift_ge_2": v[1], "share": round(v[1] / v[0], 4) if v[0] else None}
            for k, v in held.items()}


def summary(res, single, campuses):
    outer = res["outer"]
    return {
        "n_campus_system_pairs": sum(len(v) for v in outer.values()),
        "n_certified": sum(1 for j in campuses for v in outer[j].values() if v["S"] >= cc.S_CUT),
        "risk_gradient_fails": sum(1 for j in campuses for v in outer[j].values()
                                   if v["scores"]["risk_gradient"] == cc.FAIL),
        "pooled_map_certified": sum(1 for v in res["pooled"].values() if v["S"] >= cc.S_CUT),
        "route_counts": res["routing"]["route_counts"],
        "queue_by_system": res["routing"]["queue_by_system"],
        "heldout_check": heldout(single, outer, campuses),
    }


def main():
    single, pairs = cc.load_single(), cc.load_pairs()
    campuses = sorted(single.university.unique())

    assert cc.GRADIENT_BASE == "campus"
    ref = cc.build("double", RCOL, True, single, pairs, campuses)
    stored = json.load(open(C.METRICS / "cert_corrected.json"))["corrected"]
    assert ref["routing"]["route_counts"] == stored["routing"]["route_counts"], \
        "run 45_certification_corrected.py first"

    cc.GRADIENT_BASE = "pooled"
    alt = cc.build("double", RCOL, True, single, pairs, campuses)
    cc.GRADIENT_BASE = "campus"

    # the same gate with one reversal of two adjacent bands admitted
    cc.GRADIENT_RHO_PASS = 0.85
    rev = cc.build("double", RCOL, True, single, pairs, campuses)
    cc.GRADIENT_RHO_PASS = 0.95
    rev_changes, rev_gate_changes = [], 0
    for j in campuses:
        for s_, a_ in ref["outer"][j].items():
            b_ = rev["outer"][j].get(s_)
            if b_ is None:
                continue
            if a_["scores"]["risk_gradient"] != b_["scores"]["risk_gradient"]:
                rev_gate_changes += 1
            if (a_["S"] >= cc.S_CUT) != (b_["S"] >= cc.S_CUT):
                rev_changes.append({"campus": j, "system": s_, "S_monotone": a_["S"],
                                    "S_one_reversal": b_["S"],
                                    "top_band": a_.get("top_band"), "rho": a_.get("rho")})

    changes, gate_changes = [], 0
    for j in campuses:
        for s, a in ref["outer"][j].items():
            b = alt["outer"][j].get(s)
            if b is None:
                continue
            if a["scores"]["risk_gradient"] != b["scores"]["risk_gradient"]:
                gate_changes += 1
            if (a["S"] >= cc.S_CUT) != (b["S"] >= cc.S_CUT):
                changes.append({"campus": j, "system": s,
                                "S_campus_base": a["S"], "S_pooled_base": b["S"],
                                "top_band_campus_base": a.get("top_band"),
                                "top_band_pooled_base": b.get("top_band")})
    a5, b5 = ref["outer"]["5"]["B20"], alt["outer"]["5"]["B20"]
    res = {
        "design": ("risk-gradient gate with the expectation of each band taken from each "
                   "certifying campus's own base rate (the study's definition) against one "
                   "pooled base rate; all other gates and the aggregation unchanged"),
        "campus_base": summary(ref, single, campuses),
        "pooled_base": summary(alt, single, campuses),
        "n_risk_gradient_scores_differing": gate_changes,
        "verdicts_differing": changes,
        "pooled_map_differing": [
            {"system": s, "S_campus_base": v["S"], "S_pooled_base": alt["pooled"][s]["S"]}
            for s, v in ref["pooled"].items() if alt["pooled"][s]["S"] != v["S"]],
        "campus5_B20": {
            "campus_base": {"scores": a5["scores"], "S": a5["S"],
                            "top_band": a5.get("top_band"), "rho": a5.get("rho")},
            "pooled_base": {"scores": b5["scores"], "S": b5["S"],
                            "top_band": b5.get("top_band"), "rho": b5.get("rho")}},
    }
    res["one_adjacent_reversal_admitted"] = {
        "design": ("pass level of the rank correlation lowered so that one reversal of two "
                   "adjacent bands still passes; the study's definition requires the ratio "
                   "to rise over every band"),
        "summary": summary(rev, single, campuses),
        "n_risk_gradient_scores_differing": rev_gate_changes,
        "verdicts_differing": rev_changes,
        "pooled_map_differing": [
            {"system": s_, "S_monotone": v["S"], "S_one_reversal": rev["pooled"][s_]["S"]}
            for s_, v in ref["pooled"].items() if rev["pooled"][s_]["S"] != v["S"]],
    }
    json.dump(res, open(C.METRICS / "risk_gradient_by_campus.json", "w"), indent=2)
    print(json.dumps(res, indent=1)[:3000])


if __name__ == "__main__":
    main()
