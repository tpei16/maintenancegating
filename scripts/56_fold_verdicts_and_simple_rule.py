#!/usr/bin/env python
"""
56_fold_verdicts_and_simple_rule.py -- the stratum-fold certificates gate by
gate, their dependence on sample size, a simpler rule that reads no model
output, and the performance of the screen in the campus each certificate never
saw.

  (a) One row per stratum-fold certificate (target campus c, building system k)
      of the main pipeline: the five gate scores, the quantities behind them,
      the plain mean of the five scores, the number h of hard gates scoring
      fail, S, and the certified flag.
  (b) Counts of fail, caution and pass per gate; the stratum-folds with h = 1
      and h >= 2; and the stratum-folds whose certified flag changes when the
      plain mean of the five scores replaces S, which are the cases in which
      the hard-fail rule decides the verdict.
  (c) Spearman correlation of S with the positive count and with the number of
      evaluable certifying contexts, on the pooled map and over the folds.
  (d) A simpler rule: certify a stratum when it is evaluable in at least 3
      certifying contexts, has at least 100 high-burden events, and has a
      no-antecedent share below 0.25.  A stratum with too few extreme events to
      estimate the share does not fail the last condition.  The three levels
      are the fail levels of the data-sufficiency and prior-record-trace gates,
      so the rule has no tuned parameter.  Agreement and Cohen's kappa with the
      certificate are reported, then a post hoc grid search over the three
      levels, labelled as post hoc.
  (e) Out-of-context performance: the top-decile lift of f^(-c) on stratum k in
      campus c.  No quantity entering S^(-c)_k has seen campus c, so the lift
      tests the certificate on a context it did not use.  Five decision rules
      are compared on the same evaluable stratum-folds.
  (f) The gate record of the campus 5 exterior-enclosure (B20) certificate.

Nothing is fitted.  Every gate and the aggregation rule come from
45_certification_corrected.py at the main setting: double nesting, the risk
percentile ranked within campus and quarter, and the transfer gate capped at
caution below six evaluable contexts.

Outputs -> results/tables/fold_gate_verdicts.csv
           results/metrics/fold_verdicts_and_simple_rule.json
"""
from __future__ import annotations
import sys, json, time, itertools, importlib.util
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fmscreen import config as C
from fmscreen import metrics as MET

_spec = importlib.util.spec_from_file_location(
    "cert_corrected", ROOT / "scripts" / "45_certification_corrected.py")
cc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cc)

GATES, HARD = cc.GATES, cc.HARD
R_CUT, S_CUT = cc.R_CUT, cc.S_CUT
RCOL = "R_quarter"
LEVEL = {cc.FAIL: "fail", cc.CAUTION: "caution", cc.PASS: "pass"}
CTX_MIN_POS, CTX_MIN_ROWS = 10, 100        # context evaluability, as in gates_for
TRACE_MIN_EVENTS = 30                      # as in gates_for
PRESPECIFIED = {"min_contexts": 3, "min_events": 100, "max_no_antecedent": 0.25}
GRID_CONTEXTS = list(range(1, 8))
GRID_EVENTS = [50, 100, 150, 200, 300, 400, 500, 750, 1000]
GRID_LEVELS = [0.10, 0.25, None]           # None: the no-antecedent condition is not used
PROVISIONAL = ["B30", "D10", "F10", "G20"]


# --------------------------------------------------------------------------
# the main pipeline and its reference numbers
# --------------------------------------------------------------------------
def final_pipeline(single: pd.DataFrame, pairs: dict, campuses: list[str]):
    outer = {j: cc.gates_for(cc.dev_corpus(j, campuses, single, pairs, "double"),
                             cc.trace_for(j), RCOL, True) for j in campuses}
    pooled = cc.gates_for(single, cc.trace_for(None), RCOL, True)
    routing = cc.route(single, outer, campuses, RCOL)
    return outer, pooled, routing


def check_reference(outer: dict, pooled: dict, routing: dict) -> dict:
    ref = json.load(open(C.METRICS / "cert_corrected.json"))["corrected"]
    n = sum(len(v) for v in outer.values())
    assert n == 141, f"{n} stratum-fold certificates, expected 141"
    assert set(outer) == set(ref["outer_S"]), "campus set differs from the reference"
    for j, m in outer.items():
        assert set(m) == set(ref["outer_S"][j]), f"strata differ for campus {j}"
        for s, v in m.items():
            assert abs(v["S"] - ref["outer_S"][j][s]) < 1e-9, \
                f"S^(-{j})_{s} = {v['S']} != reference {ref['outer_S'][j][s]}"
    n_cert = sum(v["S"] >= S_CUT for v in pooled.values())
    assert len(pooled) == 16 and n_cert == 12, f"pooled map certifies {n_cert}/{len(pooled)}"
    for s, v in pooled.items():
        assert abs(v["S"] - ref["pooled"][s]["S"]) < 1e-9 and v["scores"] == ref["pooled"][s]["scores"]
    assert routing["route_counts"] == ref["routing"]["route_counts"], \
        f"route counts {routing['route_counts']} != reference {ref['routing']['route_counts']}"
    return {"n_stratum_folds": n, "pooled_certified": f"{n_cert}/{len(pooled)}",
            "route_counts": routing["route_counts"], "matches_cert_corrected_json": True}


# --------------------------------------------------------------------------
# small statistics
# --------------------------------------------------------------------------
def cohen_kappa(a, b) -> float | None:
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    po = float(np.mean(a == b))
    pa, pb = float(a.mean()), float(b.mean())
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe >= 1.0 else round((po - pe) / (1 - pe), 4)


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list | None:
    if n == 0:
        return None
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(float(centre - half), 4), round(float(centre + half), 4)]


def spearman(x, y) -> dict:
    r, p = spearmanr(x, y)
    return {"rho": round(float(r), 4), "p_value": float(f"{p:.4g}"), "n": int(len(x))}


def exact_no_antecedent(trace: pd.DataFrame, system: str) -> float | None:
    """The unrounded no-antecedent share the prior-record-trace gate reads, or
    None when the stratum has too few extreme events for the gate to score it."""
    tr = trace[trace.system == system]
    ne, nz = int(tr.n_events.sum()), int(tr.n_zero.sum())
    return None if ne < TRACE_MIN_EVENTS else nz / ne


def simple_rule(n_campuses: int, n_pos: int, zp: float | None,
                min_contexts: int, min_events: int, max_no_antecedent: float | None) -> bool:
    ok = n_campuses >= min_contexts and n_pos >= min_events
    if max_no_antecedent is not None and zp is not None:
        ok = ok and zp < max_no_antecedent
    return bool(ok)


def agreement(a: list, b: list) -> dict:
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    return {"n": int(len(a)), "n_agree": int((a == b).sum()),
            "agreement": round(float(np.mean(a == b)), 4), "kappa": cohen_kappa(a, b)}


def jsonable(o):
    """Plain JSON types; NaN becomes null."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return None if np.isnan(o) else float(o)
    return o


def lift_summary(lifts: list) -> dict:
    lifts = [float(x) for x in lifts if x is not None and not np.isnan(x)]
    n = len(lifts)
    k = int(sum(x >= 2.0 for x in lifts))
    return {"n": n, "n_lift_ge_2": k,
            "share_lift_ge_2": round(k / n, 4) if n else None,
            "share_lift_ge_2_wilson95": wilson(k, n),
            "median_lift": round(float(np.median(lifts)), 3) if n else None}


# --------------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    single, pairs = cc.load_single(), cc.load_pairs()
    campuses = sorted(single.university.unique())
    outer, pooled, routing = final_pipeline(single, pairs, campuses)
    ref_check = check_reference(outer, pooled, routing)
    print(f"[56] reference reproduced: {ref_check}", flush=True)

    burden = pd.read_csv(C.TABLES / "burden_by_system.csv")
    name = {k: str(v).strip() for k, v in burden[["SystemCode", "SystemDescription"]].values}

    # unrounded no-antecedent shares, from the same trace tables the gates read
    zp_fold = {(j, s): exact_no_antecedent(cc.trace_for(j), s)
               for j in campuses for s in outer[j]}
    zp_pool = {s: exact_no_antecedent(cc.trace_for(None), s) for s in pooled}

    # ---------------- (e) realized lift in the held-out campus ----------------
    held = {}
    for (j, s), g in single.groupby(["university", "system"], observed=True):
        if j in outer and s in outer[j]:
            y, sc = g["y"].to_numpy(), g["score"].to_numpy()
            ev = bool(y.sum() >= CTX_MIN_POS and len(y) >= CTX_MIN_ROWS)
            held[(j, s)] = {"rows": int(len(y)), "pos": int(y.sum()), "evaluable": ev,
                            "lift": round(float(MET.lift_at_topk(y, sc, C.TOPK_BUDGET)), 3)
                            if ev else None}
    for j in campuses:                       # a stratum the target campus never records
        for s in outer[j]:
            held.setdefault((j, s), {"rows": 0, "pos": 0, "evaluable": False, "lift": None})

    # ---------------- (a) one row per stratum-fold ----------------
    rows = []
    for j in campuses:
        for s in sorted(outer[j]):
            v = outer[j][s]
            sc = v["scores"]
            mean = float(np.mean([sc[g] for g in GATES]))
            h = int(sum(sc[g] == cc.FAIL for g in HARD))
            rule = simple_rule(v["n_campuses"], v["n_pos"], zp_fold[(j, s)], **PRESPECIFIED)
            # the prespecified rule is "neither data sufficiency nor prior-record
            # trace scores fail"; check that reading against the gate scores
            assert rule == (sc["data_sufficiency"] != cc.FAIL and
                            sc["prior_record_trace"] != cc.FAIL), (j, s)
            if zp_fold[(j, s)] is not None:
                assert abs(round(zp_fold[(j, s)], 3) - v["zero_prior_share"]) < 1e-9
            rows.append({
                "campus": j, "system": s, "system_desc": name.get(s, s),
                **{g: sc[g] for g in GATES},
                "n_campuses": v["n_campuses"], "n_pos": v["n_pos"],
                "pooled_lift": v["pooled_lift"], "frac_2x": v["frac_2x"],
                "top_band": v.get("top_band"), "rho": v.get("rho"),
                "zero_prior_share": v["zero_prior_share"],
                "mean_score": round(mean, 2), "h_hard_fail": h, "S": round(v["S"], 2),
                "certified": bool(v["S"] >= S_CUT),
                "certified_by_mean": bool(mean >= S_CUT),
                "simple_rule_prespecified": rule,
                "heldout_rows": held[(j, s)]["rows"], "heldout_pos": held[(j, s)]["pos"],
                "heldout_evaluable": held[(j, s)]["evaluable"],
                "heldout_lift": held[(j, s)]["lift"],
            })
    fv = pd.DataFrame(rows)

    # ---------------- (b) counts ----------------
    gate_counts = {g: {LEVEL[x]: int((fv[g] == x).sum()) for x in (cc.FAIL, cc.CAUTION, cc.PASS)}
                   for g in GATES}
    flips = fv[fv.certified != fv.certified_by_mean]
    counts = {
        "n_stratum_folds": int(len(fv)),
        "n_certified": int(fv.certified.sum()),
        "gate_score_counts": gate_counts,
        "hard_fail_counts": {"h=0": int((fv.h_hard_fail == 0).sum()),
                             "h=1": int((fv.h_hard_fail == 1).sum()),
                             "h>=2": int((fv.h_hard_fail >= 2).sum())},
        "n_certified_by_plain_mean": int(fv.certified_by_mean.sum()),
        "n_verdicts_changed_by_hard_fail_rule": int(len(flips)),
        "verdicts_changed_by_hard_fail_rule": [
            {"campus": r.campus, "system": r.system, "mean_score": r.mean_score, "S": r.S,
             "hard_gates_failed": [g for g in HARD if getattr(r, g) == cc.FAIL]}
            for r in flips.itertuples()],
    }

    # ---------------- (c) sample size ----------------
    pool_rows = [{"system": s, "S": v["S"], "n_pos": v["n_pos"], "n_campuses": v["n_campuses"]}
                 for s, v in sorted(pooled.items())]
    pdf = pd.DataFrame(pool_rows)
    sample_size = {
        "pooled_map": {"S_vs_n_pos": spearman(pdf.S, pdf.n_pos),
                       "S_vs_n_campuses": spearman(pdf.S, pdf.n_campuses)},
        "stratum_folds": {"S_vs_n_pos": spearman(fv.S, fv.n_pos),
                          "S_vs_n_campuses": spearman(fv.S, fv.n_campuses)},
        "note": ("the 141 stratum-folds are not independent: each stratum contributes up "
                 "to nine folds that share eight of nine campuses"),
    }

    # ---------------- (d) the simpler rule ----------------
    cert_pool = {s: v["S"] >= S_CUT for s, v in pooled.items()}
    cert_fold = {(r.campus, r.system): r.certified for r in fv.itertuples()}
    fold_keys = list(cert_fold)
    pool_keys = sorted(pooled)

    def evaluate_rule(**kw) -> dict:
        rp = {s: simple_rule(pooled[s]["n_campuses"], pooled[s]["n_pos"], zp_pool[s], **kw)
              for s in pool_keys}
        rf = {k: simple_rule(outer[k[0]][k[1]]["n_campuses"], outer[k[0]][k[1]]["n_pos"],
                             zp_fold[k], **kw) for k in fold_keys}
        return {"pooled": rp, "folds": rf,
                "pooled_agreement": agreement([cert_pool[s] for s in pool_keys],
                                              [rp[s] for s in pool_keys]),
                "fold_agreement": agreement([cert_fold[k] for k in fold_keys],
                                            [rf[k] for k in fold_keys])}

    def disagreements(res: dict) -> dict:
        return {
            "pooled": [{"system": s, "certificate": bool(cert_pool[s]), "rule": res["pooled"][s],
                        "S": pooled[s]["S"]}
                       for s in pool_keys if res["pooled"][s] != cert_pool[s]],
            "folds": [{"campus": k[0], "system": k[1], "certificate": bool(cert_fold[k]),
                       "rule": res["folds"][k], "S": outer[k[0]][k[1]]["S"]}
                      for k in fold_keys if res["folds"][k] != cert_fold[k]],
        }

    pre = evaluate_rule(**PRESPECIFIED)
    grid = []
    for mc, me, ml in itertools.product(GRID_CONTEXTS, GRID_EVENTS, GRID_LEVELS):
        r = evaluate_rule(min_contexts=mc, min_events=me, max_no_antecedent=ml)
        grid.append({"min_contexts": mc, "min_events": me, "max_no_antecedent": ml,
                     "fold_agreement": r["fold_agreement"]["agreement"],
                     "fold_kappa": r["fold_agreement"]["kappa"],
                     "pooled_agreement": r["pooled_agreement"]["agreement"],
                     "pooled_kappa": r["pooled_agreement"]["kappa"],
                     "n_fold_certified": int(sum(r["folds"].values())),
                     "n_pooled_certified": int(sum(r["pooled"].values()))})
    # best = highest agreement over the 141 stratum-folds; ties broken by pooled
    # agreement, then by fold kappa, then by grid order (fewest contexts first)
    order = sorted(range(len(grid)), key=lambda i: (-grid[i]["fold_agreement"],
                                                    -grid[i]["pooled_agreement"],
                                                    -(grid[i]["fold_kappa"] or -9), i))
    best = grid[order[0]]
    best_kw = {k: best[k] for k in ("min_contexts", "min_events", "max_no_antecedent")}
    best_res = evaluate_rule(**best_kw)
    tied = [g for g in grid if g["fold_agreement"] == best["fold_agreement"]]
    for r in rows:
        r["simple_rule_best_post_hoc"] = best_res["folds"][(r["campus"], r["system"])]
    fv["simple_rule_best_post_hoc"] = [r["simple_rule_best_post_hoc"] for r in rows]

    simple = {
        "prespecified": {
            "rule": PRESPECIFIED,
            "definition": ("certify when the stratum is evaluable (>=10 positives, >=100 rows) in "
                           "at least min_contexts certifying contexts, has at least min_events "
                           "positives, and has a no-antecedent share below max_no_antecedent; "
                           "a share over fewer than 30 extreme events counts as not failing"),
            "pooled_agreement": pre["pooled_agreement"],
            "fold_agreement": pre["fold_agreement"],
            "n_pooled_certified": int(sum(pre["pooled"].values())),
            "n_fold_certified": int(sum(pre["folds"].values())),
            "disagreements": disagreements(pre),
        },
        "post_hoc_grid": {
            "label": "post hoc: levels searched on the same certificates they are compared with",
            "n_rules": len(grid),
            "selection": ("highest agreement over the 141 stratum-folds; ties broken by pooled "
                          "agreement, then fold kappa, then fewest contexts and events"),
            "best_rule": best_kw,
            "best_pooled_agreement": best_res["pooled_agreement"],
            "best_fold_agreement": best_res["fold_agreement"],
            "best_disagreements": disagreements(best_res),
            "n_rules_tied_on_fold_agreement": len(tied),
            "rules_tied_on_fold_agreement": tied,
            "grid": grid,
        },
    }

    # ---------------- (e) out-of-context performance ----------------
    fv_e = fv[fv.heldout_evaluable]
    decision = {
        "certificate": fv_e.certified,
        "plain_mean_at_cut": fv_e.certified_by_mean,
        "simple_rule_prespecified": fv_e.simple_rule_prespecified,
        "simple_rule_best_post_hoc": fv_e.simple_rule_best_post_hoc,
        "certify_all": pd.Series(True, index=fv_e.index),
    }
    perf = {}
    for k, flag in decision.items():
        flag = flag.astype(bool)
        perf[k] = {"certified": lift_summary(fv_e.heldout_lift[flag].tolist()),
                   "withheld": lift_summary(fv_e.heldout_lift[~flag].tolist())}
    gate_fail = fv[(fv.transfer_stability == cc.FAIL) | (fv.risk_gradient == cc.FAIL)]
    heldout = {
        "definition": ("top-decile lift of f^(-c) on the rows of stratum k in campus c; a pair "
                       "is evaluable with >= 10 positives and >= 100 rows; holds = lift >= 2"),
        "n_stratum_folds": int(len(fv)),
        "n_evaluable": int(len(fv_e)),
        "not_evaluable": [{"campus": r.campus, "system": r.system, "rows": r.heldout_rows,
                           "pos": r.heldout_pos, "S": r.S}
                          for r in fv[~fv.heldout_evaluable].itertuples()],
        "by_decision_rule": perf,
        "spearman_S_vs_heldout_lift": spearman(fv_e.S, fv_e.heldout_lift),
        "transfer_or_gradient_fail": [
            {"campus": r.campus, "system": r.system, "S": r.S,
             "transfer_stability": LEVEL[r.transfer_stability],
             "risk_gradient": LEVEL[r.risk_gradient],
             "heldout_evaluable": bool(r.heldout_evaluable), "heldout_lift": r.heldout_lift,
             "heldout_rows": r.heldout_rows, "heldout_pos": r.heldout_pos}
            for r in gate_fail.itertuples()],
    }

    # ---------------- (f) campus 5, exterior enclosure ----------------
    v5 = outer["5"]["B20"]
    r5 = fv[(fv.campus == "5") & (fv.system == "B20")].iloc[0]
    campus5 = {
        "campus": "5", "system": "B20", "system_desc": name.get("B20", "B20"),
        "scores": {g: LEVEL[v5["scores"][g]] for g in GATES},
        "hard_gates_failed": [g for g in HARD if v5["scores"][g] == cc.FAIL],
        "n_campuses": v5["n_campuses"], "n_pos": v5["n_pos"], "pooled_lift": v5["pooled_lift"],
        "frac_2x": v5["frac_2x"], "top_band": v5.get("top_band"), "rho": v5.get("rho"),
        "zero_prior_share": v5["zero_prior_share"],
        "mean_score": float(r5.mean_score), "S": v5["S"], "certified": bool(v5["S"] >= S_CUT),
        "heldout_lift": r5.heldout_lift, "heldout_rows": int(r5.heldout_rows),
        "heldout_pos": int(r5.heldout_pos),
        "pooled_map_B20": {"scores": {g: LEVEL[pooled["B20"]["scores"][g]] for g in GATES},
                           "S": pooled["B20"]["S"], "frac_2x": pooled["B20"]["frac_2x"],
                           "n_campuses": pooled["B20"]["n_campuses"]},
    }

    # ---------------- write ----------------
    cols = (["campus", "system", "system_desc"] + GATES +
            ["n_campuses", "n_pos", "pooled_lift", "frac_2x", "top_band", "rho",
             "zero_prior_share", "mean_score", "h_hard_fail", "S", "certified",
             "certified_by_mean", "simple_rule_prespecified", "simple_rule_best_post_hoc",
             "heldout_rows", "heldout_pos", "heldout_evaluable", "heldout_lift"])
    fv[cols].to_csv(C.TABLES / "fold_gate_verdicts.csv", index=False)

    out = {
        "design": ("The 141 stratum-fold certificates of the main pipeline (double nesting, "
                   "risk percentile within campus and quarter, capped transfer gate), recomputed "
                   "with the functions of 45_certification_corrected.py, read gate by gate; a "
                   "simpler rule built from the data-sufficiency and prior-record-trace fail "
                   "levels; and the top-decile lift of f^(-c) on each stratum in the campus c "
                   "that the certificate S^(-c)_k did not use."),
        "reference_check": ref_check,
        "provisional_strata": PROVISIONAL,
        "counts": counts,
        "sample_size": sample_size,
        "simple_rule": simple,
        "heldout_performance": heldout,
        "campus5_B20": campus5,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    json.dump(jsonable(out), open(C.METRICS / "fold_verdicts_and_simple_rule.json", "w"),
              indent=2)

    # ---------------- console ----------------
    print(f"\n=== (b) gate scores over {len(fv)} stratum-folds ===")
    for g, c in gate_counts.items():
        print(f"  {g:20s} {c}")
    print(f"  hard fails: {counts['hard_fail_counts']}")
    print(f"  verdicts changed by the hard-fail rule: {counts['n_verdicts_changed_by_hard_fail_rule']}")
    for f in counts["verdicts_changed_by_hard_fail_rule"]:
        print(f"    {f}")
    print("\n=== (c) sample size ===")
    print(f"  {json.dumps({k: v for k, v in sample_size.items() if k != 'note'})}")
    print("\n=== (d) simpler rule ===")
    print(f"  prespecified: pooled {pre['pooled_agreement']}, folds {pre['fold_agreement']}")
    print(f"  best post hoc {best_kw}: pooled {best_res['pooled_agreement']}, "
          f"folds {best_res['fold_agreement']} ({len(tied)} rules tied)")
    print(f"\n=== (e) held-out lift, {len(fv_e)} evaluable of {len(fv)} ===")
    for k, v in perf.items():
        print(f"  {k:28s} certified {v['certified']}\n  {'':28s} withheld  {v['withheld']}")
    print(f"  Spearman S vs lift: {heldout['spearman_S_vs_heldout_lift']}")
    print(f"  transfer/gradient fails: {heldout['transfer_or_gradient_fail']}")
    print(f"\n=== (f) campus 5 B20 ===\n  {campus5}")
    print(f"\n[56] done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
