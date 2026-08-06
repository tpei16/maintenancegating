#!/usr/bin/env python
"""
48_learner_invariance.py -- is the certificate a property of the evidence source
or of the model?

The paper asserts that S measures whether the record supports a decision, not how
good one particular model is.  That is testable: run the whole certification
under three risk scores of very different capability and compare the maps.

  rule    the best simple rule, prior-year UPM work-order count.  No fitting at
          all, so no leakage question arises for it.
  logreg  l2-regularised logistic regression on the deployed feature set.
  gbdt    the deployed boosted screen.

If the certification map is largely invariant, S is a property of the evidence.
If it is not, S must be described throughout as out-of-context model reliability.
Both outcomes are reportable; this script produces whichever holds.

Needs 45 logistic fits (36 pairs + 9 singles) to keep the double-nested protocol
identical across learners.  The rule needs none.

Outputs -> data/processed/pred_pair_logreg/, pred_single_logreg/
           results/metrics/learner_invariance.json
"""
from __future__ import annotations
import os
N_JOBS = int(os.environ.get("N_JOBS", "6"))
_T = os.environ.get("FMSCREEN_THREADS", "4")
os.environ["FMSCREEN_THREADS"] = _T
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = _T

import sys, json, itertools
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fmscreen import config as C
from fmscreen import engine as E
from fmscreen import baselines as BL

LAYER, TARGET, PCTL = "M1", "severity_labour", 75
S_CUT = 0.67


def _panel():
    return pd.read_parquet(C.DATA_PROCESSED / "panel_quarter.parquet")


def _fit(held_out, model_name, outdir):
    panel = _panel()
    ucol = panel[C.COL_UNIV].astype("string").to_numpy()
    keep = ~np.isin(ucol, list(held_out))
    train, test = panel[keep], panel[~keep]
    r = E.evaluate_split(train, test, LAYER, model_name, TARGET, pctl=PCTL,
                         compute_ci=False, return_scores=True)
    if r is None:
        raise RuntimeError(f"degenerate split {held_out}")
    df = pd.DataFrame({
        "university": np.asarray(r["_test_univ"]).astype(str),
        "system": np.asarray(r["_test_system"]).astype(str),
        "period_q": test["period_q"].to_numpy(),
        "y": np.asarray(r["_y"]).astype(np.int8),
        "score": np.asarray(r["_scores"], dtype=np.float32),
    })
    df.to_parquet(outdir / ("__".join(held_out) + ".parquet"), index=False)
    return len(df)


def rule_frame() -> pd.DataFrame:
    """The best simple rule, scored on every panel row.  No model, no folds."""
    panel = _panel()
    from fmscreen import features as FE
    # label with train-fold-free thresholds is not available for a rule that is
    # never trained; use the same system-specific p75 the panel defines.
    _, y_all, _ = FE.make_labels(panel, panel, TARGET, PCTL)
    scores = BL.rule_scores(panel)
    name = "upm_count_prior_year" if "upm_count_prior_year" in scores else list(scores)[0]
    return pd.DataFrame({
        "university": panel[C.COL_UNIV].astype(str).to_numpy(),
        "system": panel[C.COL_SYSTEM].astype(str).to_numpy(),
        "period_q": panel["period_q"].to_numpy(),
        "y": np.asarray(y_all).astype(np.int8),
        "score": np.asarray(scores[name], dtype=np.float32),
    }), name


def main() -> None:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "cert", Path(__file__).resolve().parent / "45_certification_corrected.py")
    cert = importlib.util.module_from_spec(spec); spec.loader.exec_module(cert)

    panel = _panel()
    campuses = sorted(panel[C.COL_UNIV].astype(str).unique())
    pairs = list(itertools.combinations(campuses, 2))

    # ---------------- logistic fits ----------------
    pd_dir = C.DATA_PROCESSED / "pred_pair_logreg"; pd_dir.mkdir(parents=True, exist_ok=True)
    sd_dir = C.DATA_PROCESSED / "pred_single_logreg"; sd_dir.mkdir(parents=True, exist_ok=True)
    if len(list(pd_dir.glob("*.parquet"))) < len(pairs):
        print(f"[inv] fitting {len(campuses)} single + {len(pairs)} pair logistic models",
              flush=True)
        Parallel(n_jobs=N_JOBS, verbose=5)(
            delayed(_fit)((c,), "logreg", sd_dir) for c in campuses)
        Parallel(n_jobs=N_JOBS, verbose=5)(
            delayed(_fit)((a, b), "logreg", pd_dir) for a, b in pairs)

    trace = cert.prior_trace_table()

    def load(dirpath):
        d = pd.concat([pd.read_parquet(f) for f in sorted(dirpath.glob("*.parquet"))],
                      ignore_index=True)
        return cert._add_ranks(d)

    def pairs_map(dirpath):
        out = {}
        for f in sorted(dirpath.glob("*.parquet")):
            a, b = f.stem.split("__")
            out[frozenset((a, b))] = pd.read_parquet(f)
        return out

    arms = {}
    # gbdt: the deployed screen, reusing the fits from script 44
    arms["gbdt"] = (cert.load_single(), cert.load_pairs())
    arms["logreg"] = (load(sd_dir), pairs_map(pd_dir))

    rf, rule_name = rule_frame()
    rf = cert._add_ranks(rf)
    arms["rule"] = (rf, None)

    results = {}
    for name, (single, prs) in arms.items():
        outer = {}
        for j in campuses:
            if prs is None:
                # a rule is not fitted, so no context can leak into it; every
                # certifying context is read from the same unfitted score.
                dev = cert._add_ranks(single[single.university != j])
            else:
                dev = cert.dev_corpus(j, campuses, single, prs, "double")
            outer[j] = cert.gates_for(dev, trace[trace.campus != j],
                                      "R_quarter", True)
        pooled = cert.gates_for(single, trace, "R_quarter", True)
        results[name] = {"pooled": {s: v["S"] for s, v in pooled.items()},
                         "pooled_lift": {s: v["pooled_lift"] for s, v in pooled.items()},
                         "outer": {j: {s: v["S"] for s, v in outer[j].items()}
                                   for j in campuses}}

    # ---------------- agreement ----------------
    ref = "gbdt"
    systems = sorted(results[ref]["pooled"])
    agree = {}
    for name in results:
        if name == ref:
            continue
        both = [s for s in systems if s in results[name]["pooled"]]
        same = sum((results[ref]["pooled"][s] >= S_CUT) ==
                   (results[name]["pooled"][s] >= S_CUT) for s in both)
        dS = [abs(results[ref]["pooled"][s] - results[name]["pooled"][s]) for s in both]
        # outer-fold level agreement
        pairs_cmp, same_f = 0, 0
        for j in campuses:
            for s in set(results[ref]["outer"][j]) & set(results[name]["outer"][j]):
                pairs_cmp += 1
                same_f += ((results[ref]["outer"][j][s] >= S_CUT) ==
                           (results[name]["outer"][j][s] >= S_CUT))
        agree[name] = {
            "n_strata_compared": len(both),
            "pooled_verdict_agreement": round(same / len(both), 3),
            "pooled_disagreements": [s for s in both
                                     if (results[ref]["pooled"][s] >= S_CUT) !=
                                        (results[name]["pooled"][s] >= S_CUT)],
            "mean_abs_delta_S": round(float(np.mean(dS)), 3),
            "max_abs_delta_S": round(float(np.max(dS)), 3),
            "n_stratum_fold_pairs": pairs_cmp,
            "outer_verdict_agreement": round(same_f / pairs_cmp, 3) if pairs_cmp else None,
        }

    out = {"rule_used": rule_name, "certified_counts":
           {n: int(sum(v >= S_CUT for v in results[n]["pooled"].values()))
            for n in results},
           "n_gated": {n: len(results[n]["pooled"]) for n in results},
           "agreement_vs_gbdt": agree,
           "pooled_S": {n: results[n]["pooled"] for n in results},
           "pooled_lift": {n: results[n]["pooled_lift"] for n in results}}
    json.dump(out, open(C.METRICS / "learner_invariance.json", "w"), indent=2)

    print("\n=== CERTIFICATION INVARIANCE ACROSS LEARNERS ===")
    print(f"  best simple rule used: {rule_name}")
    for n in ("rule", "logreg", "gbdt"):
        print(f"  {n:7s} certified {out['certified_counts'][n]}/{out['n_gated'][n]}")
    print("\n  agreement with the deployed boosted screen:")
    for n, a in agree.items():
        print(f"    {n:7s} pooled verdicts {a['pooled_verdict_agreement']*100:.0f}% "
              f"({a['n_strata_compared']} strata), outer-fold "
              f"{a['outer_verdict_agreement']*100:.0f}% ({a['n_stratum_fold_pairs']} pairs), "
              f"mean|dS|={a['mean_abs_delta_S']}, max|dS|={a['max_abs_delta_S']}")
        if a["pooled_disagreements"]:
            print(f"            disagreeing strata: {a['pooled_disagreements']}")


if __name__ == "__main__":
    main()
