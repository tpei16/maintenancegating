#!/usr/bin/env python
"""
46_manuscript_numbers.py -- every quantity the manuscript quotes about
certification and routing, recomputed from the corrected pipeline
(45_certification_corrected.py) so that one file is the single source of truth.

Route shares are reported over ALL panel units and all 21 strata, not over the
16 gated ones, because a portfolio applying the method inherits the ungated
strata as well.

Output -> results/metrics/manuscript_numbers.json  (+ console table)
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fmscreen import config as C

S_CUT, R_CUT = 0.67, 0.90


def main() -> None:
    cert = json.load(open(C.METRICS / "cert_corrected.json"))
    cor = cert["corrected"]
    pub = cert["single_nested"]
    ev = pd.read_csv(C.TABLES / "gate_evidence_corrected.csv")
    burden = pd.read_csv(C.TABLES / "burden_by_system.csv")
    share = dict(zip(burden.SystemCode, burden.share_of_total_upm_labour))

    gated = list(ev.system)
    out = {}

    # ---------------- burden accounting (the 99.97% claim) ----------------
    out["burden"] = {
        "n_strata_total": int(burden.SystemCode.nunique()),
        "n_strata_gated": len(gated),
        "share_labor_in_gated_strata": round(float(sum(share[s] for s in gated)) * 100, 2),
        "ungated_strata": {str(r.SystemCode): round(float(r.share_of_total_upm_labour) * 100, 3)
                           for r in burden[~burden.SystemCode.isin(gated)].itertuples()},
    }

    # ---------------- certification map ----------------
    certified = ev[ev.S_pooled >= S_CUT]
    uncert = ev[ev.S_pooled < S_CUT]
    out["certification"] = {
        "n_certified": len(certified), "n_gated": len(gated),
        "certified_systems": list(certified.system_desc.str.strip()),
        "uncertified_systems": list(uncert.system_desc.str.strip()),
        "share_labor_certified": round(float(certified.burden_share.sum()) * 100, 2),
        "share_labor_uncertified": round(float(uncert.burden_share.sum()) * 100, 2),
        "n_all_nine_folds": int((ev.folds_certified == 9).sum()),
        "share_labor_all_nine_folds": round(
            float(ev[ev.folds_certified == 9].burden_share.sum()) * 100, 2),
        "folds_certified": {r.system_desc.strip(): f"{int(r.folds_certified)}/{int(r.folds_gated)}"
                            for r in ev.itertuples()},
    }

    # ---------------- pooled-lift range ----------------
    out["pooled_lift"] = {
        "min_all_gated": float(ev.pooled_lift_top10.min()),
        "min_all_gated_system": ev.loc[ev.pooled_lift_top10.idxmin(), "system_desc"].strip(),
        "max_all_gated": float(ev.pooled_lift_top10.max()),
        "max_all_gated_system": ev.loc[ev.pooled_lift_top10.idxmax(), "system_desc"].strip(),
        "min_multi_context": float(ev[ev.campuses_evaluable >= 3].pooled_lift_top10.min()),
        "n_pass_risk_concentration": int((ev.risk_concentration == 1.0).sum()),
    }

    # ---------------- no-antecedent contrast ----------------
    z = ev.dropna(subset=["zero_prior_share"])
    out["no_antecedent"] = {
        "min": float(z.zero_prior_share.min()),
        "min_system": z.loc[z.zero_prior_share.idxmin(), "system_desc"].strip(),
        "max": float(z.zero_prior_share.max()),
        "max_system": z.loc[z.zero_prior_share.idxmax(), "system_desc"].strip(),
        "ratio": round(float(z.zero_prior_share.max() / z.zero_prior_share.min()), 1),
    }

    # ---------------- routing, over all 21 strata ----------------
    for arm, tag in (("corrected", "corrected"), ("single_nested", "single_nested")):
        r = cert[arm]["routing"]
        rc, n = r["route_counts"], r["n_routed"]
        hi = rc["pma"] + rc["verify"]
        out[f"routing_{tag}"] = {
            "n_units_routed": n,
            "counts": rc,
            "shares_pct": {k: round(v / n * 100, 1) for k, v in rc.items()},
            "n_high_risk": hi,
            "pct_high_risk_cleared": round(rc["pma"] / hi * 100, 1),
            "pct_high_risk_diverted": round(rc["verify"] / hi * 100, 1),
            "units_in_gated_strata": r["units_in_gated_strata"],
            "units_in_ungated_strata": r["units_in_ungated_strata"],
            "pct_units_ungated": round(r["units_in_ungated_strata"] / n * 100, 2),
            "queue_by_system": r["queue_by_system"],
        }

    # ---------------- verification queue: stable core vs fold-dependent ----------------
    # A stratum is "stably uncertified" if it is certified in zero outer folds.
    stable = set(ev[ev.folds_certified == 0].system)
    q = cert["corrected"]["routing"]["queue_by_system"]
    core = {s: n for s, n in q.items() if s in stable}
    labile = {s: n for s, n in q.items() if s not in stable}
    desc = {r.system: r.system_desc.strip() for r in ev.itertuples()}
    out["verification_queue"] = {
        "total_tasks": int(sum(q.values())),
        "stable_core_tasks": int(sum(core.values())),
        "stable_core_by_system": {desc.get(s, s): int(n) for s, n in
                                  sorted(core.items(), key=lambda kv: -kv[1])},
        "fold_dependent_tasks": int(sum(labile.values())),
        "fold_dependent_by_system": {desc.get(s, s): int(n) for s, n in
                                     sorted(labile.items(), key=lambda kv: -kv[1])},
        "pct_fold_dependent": round(sum(labile.values()) / max(sum(q.values()), 1) * 100, 1),
    }

    # ---------------- measured effect of each correction ----------------
    delta = json.load(open(C.METRICS / "leakage_delta.json"))
    out["corrections"] = {
        k: {"n_flips": v["n_certification_flips"],
            "max_abs_delta_S": v["max_abs_delta_S"],
            "mean_abs_delta_S": v["mean_abs_delta_S"],
            "n_pairs": v["n_stratum_fold_pairs"],
            "delta_routes": v["delta_route_counts"],
            "cleared_from": v["share_high_risk_cleared_from"],
            "cleared_to": v["share_high_risk_cleared_to"]}
        for k, v in delta.items()
    }
    flips = delta["all_three__single_nested_vs_corrected"]["flips"]
    out["corrections"]["flip_systems"] = sorted({desc.get(f["system"], f["system"])
                                                 for f in flips})
    out["corrections"]["core_strata_stable"] = bool(
        all(int(ev[ev.system == s].folds_certified.iloc[0]) == 9
            for s in ev[ev.folds_certified == 9].system))

    json.dump(out, open(C.METRICS / "manuscript_numbers.json", "w"), indent=2)

    # ---------------- console ----------------
    print("=== BURDEN ACCOUNTING ===")
    print(f"  {out['burden']['n_strata_gated']} gated strata carry "
          f"{out['burden']['share_labor_in_gated_strata']}% of reactive labor "
          f"(paper says 99.97)")
    print(f"  ungated but non-trivial: {out['burden']['ungated_strata']}")
    print("\n=== CERTIFICATION ===")
    c = out["certification"]
    print(f"  {c['n_certified']} of {c['n_gated']} certified, carrying "
          f"{c['share_labor_certified']}% of labor")
    print(f"  uncertified carry {c['share_labor_uncertified']}% of labor")
    print(f"  {c['n_all_nine_folds']} certified in all nine folds "
          f"({c['share_labor_all_nine_folds']}% of labor)")
    print("\n=== ROUTING (all 21 strata, all units) ===")
    for tag in ("single_nested", "corrected"):
        r = out[f"routing_{tag}"]
        print(f"  {tag:10s} n={r['n_units_routed']} high-risk={r['n_high_risk']} "
              f"cleared={r['pct_high_risk_cleared']}% diverted={r['pct_high_risk_diverted']}%")
        print(f"             shares {r['shares_pct']}")
    print("\n=== VERIFICATION QUEUE ===")
    vq = out["verification_queue"]
    print(f"  total {vq['total_tasks']}, stable core {vq['stable_core_tasks']}, "
          f"fold-dependent {vq['fold_dependent_tasks']} ({vq['pct_fold_dependent']}%)")
    print(f"  core: {vq['stable_core_by_system']}")
    print(f"  labile: {vq['fold_dependent_by_system']}")
    print("\n=== POOLED LIFT RANGE ===")
    pl = out["pooled_lift"]
    print(f"  all gated: {pl['min_all_gated']} ({pl['min_all_gated_system']}) to "
          f"{pl['max_all_gated']} ({pl['max_all_gated_system']})")
    print(f"  >=3 contexts: {pl['min_multi_context']} to {pl['max_all_gated']}")
    print(f"  risk concentration passes: {pl['n_pass_risk_concentration']}/{len(ev)}")
    print("\n=== NO-ANTECEDENT CONTRAST ===")
    na = out["no_antecedent"]
    print(f"  {na['min']} ({na['min_system']}) to {na['max']} ({na['max_system']}) "
          f"= {na['ratio']}-fold")


if __name__ == "__main__":
    main()
