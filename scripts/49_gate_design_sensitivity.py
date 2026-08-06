#!/usr/bin/env python
"""
49_gate_design_sensitivity.py -- design sensitivity of the certificate itself.
No new model fits: everything reads the
corrected double-nested predictions.

  18  which three gates are hard?  The published choice is data sufficiency,
      transfer stability and risk-gradient reliability.  Every other subset of
      the five is scored, so the choice is justified by comparison rather than
      asserted.
  24  joint sweep of the two cuts rho (risk) and sigma (certification), which
      govern different resources and were previously swept separately.
  25  the "empty region" argument for sigma = 2/3 holds for the pooled score
      distribution; deployment uses the nested map, where outer-fold
      certificates do reach 0.7.  Count the stratum-fold pairs sitting exactly
      there and move the cut to 0.75.
  26  uncertainty on the individual gate functionals, not only on the screening
      metrics: a campus-cluster bootstrap of pooled lift, the two-fold share and
      the no-antecedent share per stratum.

Output -> results/metrics/gate_design_sensitivity.json
"""
from __future__ import annotations
import sys, json, itertools, importlib.util
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fmscreen import config as C
from fmscreen import metrics as MET

spec = importlib.util.spec_from_file_location(
    "cert", Path(__file__).resolve().parent / "45_certification_corrected.py")
cert = importlib.util.module_from_spec(spec); spec.loader.exec_module(cert)

GATES = cert.GATES
PUBLISHED_HARD = set(cert.HARD)
R_CUT, S_CUT = 0.90, 0.67
RNG = np.random.default_rng(20260806)


def sufficiency_with(scores: dict, hard: set) -> float:
    S = float(np.mean([scores[g] for g in GATES]))
    nhf = sum(1 for g in hard if scores[g] == 0.0)
    if nhf >= 2:
        return 0.0
    if nhf >= 1:
        return min(S, 0.50)
    return S


def main() -> None:
    single, pairs, trace = cert.load_single(), cert.load_pairs(), cert.prior_trace_table()
    campuses = sorted(single.university.unique())

    outer_scores = {}
    for j in campuses:
        dev = cert.dev_corpus(j, campuses, single, pairs, "double")
        outer_scores[j] = cert.gates_for(dev, trace[trace.campus != j], "R_quarter", True)
    pooled = cert.gates_for(single, trace, "R_quarter", True)
    out = {}

    # ---------------- 18: which gates are hard ----------------
    subsets = []
    for r in range(0, 6):
        subsets += [frozenset(c) for c in itertools.combinations(GATES, r)]
    rows = []
    for hs in subsets:
        cert_pooled = {s: sufficiency_with(v["scores"], hs) for s, v in pooled.items()}
        n_cert = sum(v >= S_CUT for v in cert_pooled.values())
        # does this hard set still catch a stratum whose ranking does not transfer?
        # the discriminating case is a stratum failing transfer stability alone.
        probe = {g: 1.0 for g in GATES}; probe["transfer_stability"] = 0.0
        catches_transfer_failure = sufficiency_with(probe, hs) < S_CUT
        probe2 = {g: 1.0 for g in GATES}; probe2["data_sufficiency"] = 0.0
        catches_no_data = sufficiency_with(probe2, hs) < S_CUT
        probe3 = {g: 1.0 for g in GATES}; probe3["risk_gradient"] = 0.0
        catches_flat_gradient = sufficiency_with(probe3, hs) < S_CUT
        rows.append({
            "hard_set": sorted(hs), "size": len(hs), "n_certified": int(n_cert),
            "is_published_choice": set(hs) == PUBLISHED_HARD,
            "catches_transfer_failure": bool(catches_transfer_failure),
            "catches_no_data": bool(catches_no_data),
            "catches_flat_gradient": bool(catches_flat_gradient),
            "catches_all_three_uninterpretable": bool(
                catches_transfer_failure and catches_no_data and catches_flat_gradient),
        })
    minimal = [r for r in rows if r["catches_all_three_uninterpretable"]]
    out["hard_set_search"] = {
        "n_subsets": len(rows),
        "n_catching_all_three": len(minimal),
        "smallest_catching_all_three": min((r["size"] for r in minimal), default=None),
        "sets_catching_all_three": [r["hard_set"] for r in minimal
                                    if r["size"] == min(x["size"] for x in minimal)],
        "certified_count_by_size": {str(k): sorted({r["n_certified"] for r in rows if r["size"] == k})
                                    for k in range(6)},
        "published_row": next(r for r in rows if r["is_published_choice"]),
        "all_rows": rows,
    }

    # ---------------- 24: joint rho-sigma grid ----------------
    grid = []
    for rho in (0.80, 0.85, 0.90, 0.95):
        for sig in (0.50, 0.60, 0.67, 0.75, 0.85):
            routes = {"pma": 0, "verify": 0, "continue": 0, "watch": 0}
            for j in campuses:
                sub = single[single.university == j]
                Smap = {s: v["S"] for s, v in outer_scores[j].items()}
                S = sub.system.map(Smap).fillna(0.0).to_numpy()
                R = sub["R_quarter"].to_numpy()
                hiR, hiS = R >= rho, S >= sig
                routes["pma"] += int((hiR & hiS).sum()); routes["verify"] += int((hiR & ~hiS).sum())
                routes["continue"] += int((~hiR & hiS).sum()); routes["watch"] += int((~hiR & ~hiS).sum())
            hi = routes["pma"] + routes["verify"]
            grid.append({"rho": rho, "sigma": sig, **routes,
                         "pct_cleared": round(routes["pma"] / hi * 100, 1) if hi else None,
                         "verify_queue": routes["verify"]})
    out["rho_sigma_grid"] = grid

    # ---------------- 25: sigma robustness under the nested map ----------------
    vals = [v for j in campuses for v in
            (x["S"] for x in outer_scores[j].values())]
    at_070 = sum(1 for v in vals if abs(v - 0.70) < 1e-9)
    hist = {}
    for v in vals:
        hist[f"{v:.1f}"] = hist.get(f"{v:.1f}", 0) + 1
    g67 = next(x for x in grid if x["rho"] == 0.90 and x["sigma"] == 0.67)
    g75 = next(x for x in grid if x["rho"] == 0.90 and x["sigma"] == 0.75)
    out["sigma_under_nested_map"] = {
        "n_stratum_fold_pairs": len(vals),
        "n_at_exactly_0.70": at_070,
        "pct_at_exactly_0.70": round(at_070 / len(vals) * 100, 1),
        "outer_S_histogram": dict(sorted(hist.items())),
        "routes_at_sigma_0.67": {k: g67[k] for k in ("pma", "verify")},
        "routes_at_sigma_0.75": {k: g75[k] for k in ("pma", "verify")},
        "verify_queue_change": g75["verify"] - g67["verify"],
        "note": ("the pooled distribution has an empty region in (0.60, 0.80]; "
                 "the nested map does not, so the cut is consequential in "
                 "deployment even though it is not in the pooled map"),
    }

    # ---------------- 26: uncertainty on the gate functionals ----------------
    B = 400
    unc = {}
    for sysc, g in single.groupby("system", observed=True):
        if g["y"].sum() < 20 or len(g) < 200:
            continue
        us = g.university.unique()
        lift_b, frac_b = [], []
        for _ in range(B):
            draw = RNG.choice(us, size=len(us), replace=True)      # cluster on campus
            gb = pd.concat([g[g.university == u] for u in draw], ignore_index=True)
            lift_b.append(MET.lift_at_topk(gb["y"].to_numpy(), gb["score"].to_numpy(),
                                           C.TOPK_BUDGET))
            ls = [MET.lift_at_topk(x["y"].to_numpy(), x["score"].to_numpy(), C.TOPK_BUDGET)
                  for _, x in gb.groupby("university", observed=True)
                  if x["y"].sum() >= 10 and len(x) >= 100]
            frac_b.append(float(np.mean([l >= 2 for l in ls])) if ls else np.nan)
        tr = trace[trace.system == sysc]
        ne, nz = int(tr.n_events.sum()), int(tr.n_zero.sum())
        zp_ci = None
        if ne >= 30:
            zb = RNG.binomial(ne, nz / ne, size=B) / ne
            zp_ci = [round(float(np.percentile(zb, 2.5)), 3),
                     round(float(np.percentile(zb, 97.5)), 3)]
        fb = np.asarray(frac_b, dtype=float); fb = fb[~np.isnan(fb)]
        unc[sysc] = {
            "pooled_lift_ci95": [round(float(np.percentile(lift_b, 2.5)), 2),
                                 round(float(np.percentile(lift_b, 97.5)), 2)],
            "frac_2x_ci95": ([round(float(np.percentile(fb, 2.5)), 2),
                              round(float(np.percentile(fb, 97.5)), 2)] if len(fb) else None),
            "no_antecedent_ci95": zp_ci,
        }
    out["gate_uncertainty"] = {"n_bootstrap": B, "cluster": "campus", "by_stratum": unc}

    json.dump(out, open(C.METRICS / "gate_design_sensitivity.json", "w"), indent=2)

    # ---------------- console ----------------
    hs = out["hard_set_search"]
    print("=== 18. WHICH GATES ARE HARD ===")
    print(f"  {hs['n_subsets']} subsets scored; {hs['n_catching_all_three']} catch all "
          f"three uninterpretable failures")
    print(f"  smallest such set has size {hs['smallest_catching_all_three']}: "
          f"{hs['sets_catching_all_three']}")
    print(f"  published choice certifies {hs['published_row']['n_certified']}/16")
    print("\n=== 25. SIGMA UNDER THE NESTED MAP ===")
    sg = out["sigma_under_nested_map"]
    print(f"  {sg['n_at_exactly_0.70']} of {sg['n_stratum_fold_pairs']} stratum-fold pairs "
          f"sit exactly at 0.70 ({sg['pct_at_exactly_0.70']}%)")
    print(f"  verify queue 0.67 -> 0.75: {sg['routes_at_sigma_0.67']['verify']} -> "
          f"{sg['routes_at_sigma_0.75']['verify']} ({sg['verify_queue_change']:+d})")
    print(f"  outer S histogram: {sg['outer_S_histogram']}")
    print("\n=== 24. JOINT GRID (verify-queue size) ===")
    print("  rho\\sigma " + "".join(f"{s:>9}" for s in (0.50, 0.60, 0.67, 0.75, 0.85)))
    for rho in (0.80, 0.85, 0.90, 0.95):
        row = [next(x for x in grid if x["rho"] == rho and x["sigma"] == s)["verify"]
               for s in (0.50, 0.60, 0.67, 0.75, 0.85)]
        print(f"  {rho:<9}" + "".join(f"{v:>9d}" for v in row))
    print("\n=== 26. GATE UNCERTAINTY (a few strata) ===")
    for s in list(unc)[:4]:
        u = unc[s]
        print(f"  {s}: lift {u['pooled_lift_ci95']}, 2x share {u['frac_2x_ci95']}, "
              f"no-antecedent {u['no_antecedent_ci95']}")


if __name__ == "__main__":
    main()
