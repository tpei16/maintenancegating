#!/usr/bin/env python
"""
59_learner_agreement.py -- chance-corrected agreement of the certificate across
three risk scores.

48_learner_invariance.py runs the whole certification under three scores of
very different capability (the deployed boosted screen, an l2-regularised
logistic regression on the same features, and the unfitted prior-year count of
reactive work orders) and reports raw agreement of the certified flag with the
boosted screen.  Raw agreement is inflated when most strata are certified under
every score, so this script reports, for each pair of scores (logistic against
boosted, rule against boosted, rule against logistic):

  * raw agreement and Cohen's kappa of the certified flag on the pooled map and
    over the stratum-fold certificates, with a bootstrap 95% interval for the
    fold-level kappa that resamples building systems (the strata are the
    clusters, because the folds of one stratum share eight of nine campuses);
  * the same restricted to the provisional strata, those certified in some but
    not all folds under the boosted screen;
  * agreement and linearly weighted kappa of the three-level scores of the
    three gates that read the risk score (risk concentration, transfer
    stability, risk-gradient reliability), over the stratum-folds; and the
    number of stratum-folds in which the two gates that do not read the score
    (data sufficiency as coded, prior-record trace) differ between scores.

Nothing is fitted.  The logistic predictions are the fits that script 48 wrote
to data/processed/pred_single_logreg/ and pred_pair_logreg/; the rule frame is
script 48's rule_frame().  The certificates are recomputed with the functions
of 45_certification_corrected.py exactly as script 48 computes them, and the
pooled and fold agreement figures of learner_invariance.json are asserted.

Output -> results/metrics/learner_agreement.json
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import sys, json, time, itertools, importlib.util
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fmscreen import config as C


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cc = _load("cert_corrected", "45_certification_corrected.py")
li = _load("learner_invariance", "48_learner_invariance.py")

GATES, HARD = cc.GATES, cc.HARD
S_CUT = cc.S_CUT
RCOL = "R_quarter"
SCORE_GATES = ["risk_concentration", "transfer_stability", "risk_gradient"]
NONSCORE_GATES = ["data_sufficiency", "prior_record_trace"]
PROVISIONAL = ["B30", "D10", "F10", "G20"]
PAIRS = [("logreg", "gbdt"), ("rule", "gbdt"), ("rule", "logreg")]
N_BOOT = C.N_BOOTSTRAP


# --------------------------------------------------------------------------
# the three arms, built as script 48 builds them
# --------------------------------------------------------------------------
def build_arms() -> tuple[dict, list[str], dict]:
    def load(dirpath):
        d = pd.concat([pd.read_parquet(f) for f in sorted(dirpath.glob("*.parquet"))],
                      ignore_index=True)
        return cc._add_ranks(d)

    def pairs_map(dirpath):
        out = {}
        for f in sorted(dirpath.glob("*.parquet")):
            a, b = f.stem.split("__")
            out[frozenset((a, b))] = pd.read_parquet(f)
        return out

    sd_dir = C.DATA_PROCESSED / "pred_single_logreg"
    pd_dir = C.DATA_PROCESSED / "pred_pair_logreg"
    assert len(list(sd_dir.glob("*.parquet"))) == 9 and len(list(pd_dir.glob("*.parquet"))) == 36
    arms = {"gbdt": (cc.load_single(), cc.load_pairs()),
            "logreg": (load(sd_dir), pairs_map(pd_dir))}
    rf, rule_name = li.rule_frame()
    arms["rule"] = (cc._add_ranks(rf), None)

    campuses = sorted(arms["gbdt"][0].university.unique())
    results = {}
    for name, (single, prs) in arms.items():
        outer = {}
        for j in campuses:
            if prs is None:
                # the rule is not fitted, so every certifying context is read from
                # the same unfitted score, as in script 48
                dev = cc._add_ranks(single[single.university != j])
            else:
                dev = cc.dev_corpus(j, campuses, single, prs, "double")
            outer[j] = cc.gates_for(dev, cc.trace_for(j), RCOL, True)
        pooled = cc.gates_for(single, cc.trace_for(None), RCOL, True)
        results[name] = {"pooled": pooled, "outer": outer}
    routing = cc.route(arms["gbdt"][0], results["gbdt"]["outer"], campuses, RCOL)
    return results, campuses, {"rule_name": rule_name, "routing": routing}


# --------------------------------------------------------------------------
# reference numbers
# --------------------------------------------------------------------------
def check_cert_reference(outer: dict, pooled: dict, routing: dict) -> dict:
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


def check_invariance_reference(results: dict, campuses: list[str], rule_name: str) -> dict:
    """Recompute the figures of learner_invariance.json with script 48's formulas."""
    ref = json.load(open(C.METRICS / "learner_invariance.json"))
    assert ref["rule_used"] == rule_name, f"rule {rule_name} != {ref['rule_used']}"
    pS = {n: {s: v["S"] for s, v in results[n]["pooled"].items()} for n in results}
    oS = {n: {j: {s: v["S"] for s, v in results[n]["outer"][j].items()} for j in campuses}
          for n in results}
    for n in results:
        assert ref["n_gated"][n] == len(pS[n]), f"{n}: gated {len(pS[n])}"
        assert ref["certified_counts"][n] == sum(v >= S_CUT for v in pS[n].values()), n
        assert set(ref["pooled_S"][n]) == set(pS[n])
        for s in pS[n]:
            assert abs(ref["pooled_S"][n][s] - pS[n][s]) < 1e-9, (n, s)
            assert abs(ref["pooled_lift"][n][s] - results[n]["pooled"][s]["pooled_lift"]) < 1e-9
    for n in ("logreg", "rule"):
        r = ref["agreement_vs_gbdt"][n]
        both = [s for s in sorted(pS["gbdt"]) if s in pS[n]]
        same = sum((pS["gbdt"][s] >= S_CUT) == (pS[n][s] >= S_CUT) for s in both)
        dS = [abs(pS["gbdt"][s] - pS[n][s]) for s in both]
        npair, same_f = 0, 0
        for j in campuses:
            for s in set(oS["gbdt"][j]) & set(oS[n][j]):
                npair += 1
                same_f += (oS["gbdt"][j][s] >= S_CUT) == (oS[n][j][s] >= S_CUT)
        assert r["n_strata_compared"] == len(both)
        assert r["pooled_verdict_agreement"] == round(same / len(both), 3), (n, same)
        assert r["pooled_disagreements"] == [s for s in both if (pS["gbdt"][s] >= S_CUT) !=
                                             (pS[n][s] >= S_CUT)]
        assert r["mean_abs_delta_S"] == round(float(np.mean(dS)), 3)
        assert r["max_abs_delta_S"] == round(float(np.max(dS)), 3)
        assert r["n_stratum_fold_pairs"] == npair, (n, npair)
        assert r["outer_verdict_agreement"] == round(same_f / npair, 3), (n, same_f, npair)
    return {"matches_learner_invariance_json": True,
            "certified_counts": ref["certified_counts"],
            "pooled_verdict_agreement_vs_gbdt": {n: ref["agreement_vs_gbdt"][n]["pooled_verdict_agreement"]
                                                 for n in ("logreg", "rule")},
            "outer_verdict_agreement_vs_gbdt": {n: ref["agreement_vs_gbdt"][n]["outer_verdict_agreement"]
                                                for n in ("logreg", "rule")}}


# --------------------------------------------------------------------------
# agreement statistics
# --------------------------------------------------------------------------
def cohen_kappa(a, b) -> float | None:
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    if len(a) == 0:
        return None
    po = float(np.mean(a == b))
    pa, pb = float(a.mean()), float(b.mean())
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe >= 1.0 - 1e-12 else (po - pe) / (1 - pe)


def weighted_kappa_linear(a, b, levels=(0.0, 0.5, 1.0)) -> float | None:
    ia = np.searchsorted(levels, a); ib = np.searchsorted(levels, b)
    k = len(levels)
    O = np.zeros((k, k))
    np.add.at(O, (ia, ib), 1.0)
    O /= O.sum()
    E = np.outer(O.sum(axis=1), O.sum(axis=0))
    W = np.abs(np.subtract.outer(np.arange(k), np.arange(k))) / (k - 1)
    den = float((W * E).sum())
    return None if den <= 1e-12 else 1.0 - float((W * O).sum()) / den


def r4(x):
    return None if x is None else round(float(x), 4)


def flag_agreement(keys: list, A: dict, B: dict) -> dict:
    a = [A[k] >= S_CUT for k in keys]
    b = [B[k] >= S_CUT for k in keys]
    n_agree = int(sum(x == y for x, y in zip(a, b)))
    return {"n": len(keys), "n_agree": n_agree,
            "agreement": r4(n_agree / len(keys)) if keys else None,
            "kappa": r4(cohen_kappa(a, b)),
            "n_certified": [int(sum(a)), int(sum(b))],
            "disagreements": [list(k) if isinstance(k, tuple) else k
                              for k, x, y in zip(keys, a, b) if x != y]}


def cluster_kappa_ci(keys: list, A: dict, B: dict, draws: np.ndarray, strata: list[str]) -> dict:
    """Percentile interval of the fold-level kappa, resampling strata with replacement."""
    by = {s: [k for k in keys if k[1] == s] for s in strata}
    vals, undefined = [], 0
    for row in draws:
        ks = [k for i in row for k in by[strata[i]]]
        kap = cohen_kappa([A[k] >= S_CUT for k in ks], [B[k] >= S_CUT for k in ks])
        if kap is None:
            undefined += 1
        else:
            vals.append(kap)
    return {"n_boot": int(len(draws)), "n_undefined": undefined, "n_clusters": len(strata),
            "ci95": ([r4(np.percentile(vals, 2.5)), r4(np.percentile(vals, 97.5))]
                     if vals else None),
            "note": "undefined replicates (every drawn fold certified, or none, under both scores) are dropped"}


# --------------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    results, campuses, info = build_arms()
    ref_cert = check_cert_reference(results["gbdt"]["outer"], results["gbdt"]["pooled"],
                                    info["routing"])
    ref_inv = check_invariance_reference(results, campuses, info["rule_name"])
    print(f"[59] references reproduced: {ref_cert}\n     {ref_inv}", flush=True)

    # provisional strata from the boosted screen's fold certificates
    go = results["gbdt"]["outer"]
    prov = sorted(s for s in {s for j in campuses for s in go[j]}
                  if 0 < sum(go[j][s]["S"] >= S_CUT for j in campuses if s in go[j])
                  < sum(s in go[j] for j in campuses))
    assert prov == PROVISIONAL, f"provisional strata {prov} != {PROVISIONAL}"

    pS = {n: {s: v["S"] for s, v in results[n]["pooled"].items()} for n in results}
    fS = {n: {(j, s): v["S"] for j in campuses for s, v in results[n]["outer"][j].items()}
          for n in results}
    fG = {n: {(j, s): v["scores"] for j in campuses for s, v in results[n]["outer"][j].items()}
          for n in results}
    pG = {n: {s: v["scores"] for s, v in results[n]["pooled"].items()} for n in results}

    rng = np.random.default_rng(C.RANDOM_SEED)
    all_strata = sorted({k[1] for n in results for k in fS[n]})
    draws_all = rng.integers(0, len(all_strata), size=(N_BOOT, len(all_strata)))
    draws_prov = rng.integers(0, len(PROVISIONAL), size=(N_BOOT, len(PROVISIONAL)))

    comparisons = {}
    for a, b in PAIRS:
        pk = sorted(set(pS[a]) & set(pS[b]))
        fk = sorted(set(fS[a]) & set(fS[b]))
        fk_strata = sorted({k[1] for k in fk})
        pk_p = [s for s in pk if s in PROVISIONAL]
        fk_p = [k for k in fk if k[1] in PROVISIONAL]
        fold = flag_agreement(fk, fS[a], fS[b])
        fold["kappa_cluster_bootstrap"] = cluster_kappa_ci(
            fk, fS[a], fS[b], draws_all if fk_strata == all_strata else
            rng.integers(0, len(fk_strata), size=(N_BOOT, len(fk_strata))), fk_strata)
        fold_p = flag_agreement(fk_p, fS[a], fS[b])
        fold_p["kappa_cluster_bootstrap"] = cluster_kappa_ci(fk_p, fS[a], fS[b], draws_prov,
                                                             PROVISIONAL)
        gate_rows = {}
        for g in SCORE_GATES:
            x = [fG[a][k][g] for k in fk]; y = [fG[b][k][g] for k in fk]
            gate_rows[g] = {"n": len(fk), "agreement": r4(np.mean(np.asarray(x) == np.asarray(y))),
                            "weighted_kappa_linear": r4(weighted_kappa_linear(x, y)),
                            "n_differ": int(sum(u != v for u, v in zip(x, y)))}
        nonscore = {}
        for g in NONSCORE_GATES:
            diff = [list(k) for k in fk if fG[a][k][g] != fG[b][k][g]]
            pdiff = [s for s in pk if pG[a][s][g] != pG[b][s][g]]
            nonscore[g] = {"identical_over_stratum_folds": not diff, "n_stratum_folds_differ": len(diff),
                           "stratum_folds_differ": diff,
                           "identical_on_pooled_map": not pdiff, "pooled_strata_differ": pdiff}
        comparisons[f"{a}_vs_{b}"] = {
            "pooled": flag_agreement(pk, pS[a], pS[b]),
            "folds": fold,
            "stratum_folds_gated_only_in": {a: sorted(list(k) for k in set(fS[a]) - set(fS[b])),
                                            b: sorted(list(k) for k in set(fS[b]) - set(fS[a]))},
            "provisional_strata": {"pooled": flag_agreement(pk_p, pS[a], pS[b]), "folds": fold_p},
            "score_reading_gates": gate_rows,
            "gates_not_reading_the_score": nonscore,
        }

    out = {
        "design": ("The pooled and stratum-fold certificates of 45_certification_corrected.py "
                   "recomputed under three risk scores (boosted screen, logistic regression, "
                   "prior-year reactive work-order count) exactly as 48_learner_invariance.py "
                   "computes them; Cohen's kappa of the certified flag for each pair of scores, "
                   "with a bootstrap interval that resamples strata, and agreement of the "
                   "three-level gate scores."),
        "reference_check": {"cert_corrected": ref_cert, "learner_invariance": ref_inv},
        "scores": {"gbdt": "deployed boosted screen", "logreg": "l2-regularised logistic regression",
                   "rule": info["rule_name"] + " (prior-year reactive work-order count, unfitted)"},
        "provisional_strata": PROVISIONAL,
        "n_boot": N_BOOT,
        "cluster": "stratum (building system)",
        "certified_counts_pooled": {n: int(sum(v >= S_CUT for v in pS[n].values())) for n in results},
        "certified_counts_folds": {n: int(sum(v >= S_CUT for v in fS[n].values())) for n in results},
        "n_stratum_folds": {n: len(fS[n]) for n in results},
        "comparisons": comparisons,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    json.dump(out, open(C.METRICS / "learner_agreement.json", "w"), indent=2)

    # ---------------- console ----------------
    for k, v in comparisons.items():
        print(f"\n=== {k} ===")
        print(f"  pooled  {({x: v['pooled'][x] for x in ('n', 'agreement', 'kappa')})}")
        print(f"  folds   {({x: v['folds'][x] for x in ('n', 'agreement', 'kappa')})} "
              f"CI {v['folds']['kappa_cluster_bootstrap']['ci95']} "
              f"(undefined {v['folds']['kappa_cluster_bootstrap']['n_undefined']})")
        pp = v["provisional_strata"]
        print(f"  provisional pooled {({x: pp['pooled'][x] for x in ('n', 'agreement', 'kappa')})}, "
              f"folds {({x: pp['folds'][x] for x in ('n', 'agreement', 'kappa')})} "
              f"CI {pp['folds']['kappa_cluster_bootstrap']['ci95']}")
        for g, r in v["score_reading_gates"].items():
            print(f"  {g:20s} {r}")
        for g, r in v["gates_not_reading_the_score"].items():
            print(f"  {g:20s} identical={r['identical_over_stratum_folds']} "
                  f"differ={r['n_stratum_folds_differ']}")
    print(f"\n[59] done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
