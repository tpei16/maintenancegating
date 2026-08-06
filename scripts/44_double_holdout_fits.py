#!/usr/bin/env python
"""
44_double_holdout_fits.py -- model fits for leakage-free certification.

A single-nested Stage 2 certifies target campus c from the held-out
predictions of the other campuses, but those predictions come from f^(-d), which
is trained on C\\{d} and therefore on c. Certification for c was a function of
outcomes observed in c.

This script produces the fits that remove the dependence:

  * 36 pair fits  f^(-{a,b})  scoring both a and b, so certification for target c
    can read campus d through a model that saw neither c nor d.  The same fit
    serves both orderings of the pair, so 36 fits cover all 72 ordered pairs.
  * 9 single fits f^(-c)      scoring c, which is what the risk screen already
    uses.  Refitted here only so that every prediction carries period_q, which
    the published prediction files do not, and which ranking within (c,t)
    requires.

45 fits in total.  One config only: severity_labour|p75|M1|gbdt, the deployed
screen.  Everything else is inherited from fmscreen.config unchanged.

Outputs -> data/processed/pred_pair/{a}__{b}.parquet   (both campuses' rows)
           data/processed/pred_single/{c}.parquet
           results/metrics/double_holdout_fits.json    (provenance + the
                                                        reproduction check)

Env: N_JOBS (default 6), FMSCREEN_THREADS (default 4).  Their product must not
exceed the core count; numerical runtimes size their pool from the box, not from
the affinity mask, so the thread cap is set here before numpy is imported.
"""
from __future__ import annotations
import os

N_JOBS = int(os.environ.get("N_JOBS", "6"))
_THREADS = os.environ.get("FMSCREEN_THREADS", "4")
os.environ["FMSCREEN_THREADS"] = _THREADS
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = _THREADS

import sys, json, time, itertools
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fmscreen import config as C
from fmscreen import engine as E

LAYER, MODEL, TARGET, PCTL = "M1", "gbdt", "severity_labour", 75
PAIR_DIR = C.DATA_PROCESSED / "pred_pair"
SINGLE_DIR = C.DATA_PROCESSED / "pred_single"
for d in (PAIR_DIR, SINGLE_DIR):
    d.mkdir(parents=True, exist_ok=True)


def _panel() -> pd.DataFrame:
    return pd.read_parquet(C.DATA_PROCESSED / "panel_quarter.parquet")


def _fit(held_out: tuple[str, ...]) -> pd.DataFrame:
    """Train on every campus outside `held_out`; score every row inside it."""
    panel = _panel()
    ucol = panel[C.COL_UNIV].astype("string").to_numpy()
    keep = ~np.isin(ucol, list(held_out))
    train, test = panel[keep], panel[~keep]

    r = E.evaluate_split(train, test, LAYER, MODEL, TARGET, pctl=PCTL,
                         compute_ci=False, return_scores=True)
    if r is None:
        raise RuntimeError(f"degenerate split for held_out={held_out}")

    # assemble_Xy must preserve test row order, or period_q would be attached to
    # the wrong score.  Assert it rather than trust it.
    assert np.array_equal(np.asarray(r["_test_univ"]).astype(str),
                          test[C.COL_UNIV].astype(str).to_numpy()), \
        f"test row order not preserved for held_out={held_out}"

    return pd.DataFrame({
        "held_out": "+".join(held_out),
        "n_held_out": len(held_out),
        "university": np.asarray(r["_test_univ"]).astype(str),
        "building": np.asarray(r["_test_building"]).astype(str),
        "system": np.asarray(r["_test_system"]).astype(str),
        "period_q": test["period_q"].to_numpy(),
        "y": np.asarray(r["_y"]).astype(np.int8),
        "score": np.asarray(r["_scores"], dtype=np.float32),
    })


def _run_pair(a: str, b: str) -> dict:
    t0 = time.time()
    df = _fit((a, b))
    df.to_parquet(PAIR_DIR / f"{a}__{b}.parquet", index=False)
    return {"held_out": f"{a}+{b}", "rows": int(len(df)), "secs": round(time.time() - t0, 1)}


def _run_single(c: str) -> dict:
    t0 = time.time()
    df = _fit((c,))
    df.to_parquet(SINGLE_DIR / f"{c}.parquet", index=False)
    from fmscreen import metrics as MET
    lift = MET.lift_at_topk(df["y"].to_numpy(), df["score"].to_numpy(), C.TOPK_BUDGET)
    return {"held_out": c, "rows": int(len(df)), "lift_top10": round(float(lift), 3),
            "secs": round(time.time() - t0, 1)}


def main() -> None:
    panel = _panel()
    campuses = sorted(panel[C.COL_UNIV].astype(str).unique())
    pairs = list(itertools.combinations(campuses, 2))
    print(f"[fits] {len(campuses)} campuses -> {len(pairs)} pair fits + "
          f"{len(campuses)} single fits = {len(pairs) + len(campuses)} "
          f"(N_JOBS={N_JOBS}, threads={_THREADS})", flush=True)

    t0 = time.time()
    singles = Parallel(n_jobs=N_JOBS, verbose=5)(delayed(_run_single)(c) for c in campuses)
    paired = Parallel(n_jobs=N_JOBS, verbose=5)(delayed(_run_pair)(a, b) for a, b in pairs)
    wall = time.time() - t0

    # ---- reproduction check: the single fits must reproduce the published lifts ----
    pub = pd.read_csv(C.METRICS / "louo_folds.csv")
    pub = pub[(pub.regime == "louo") & (pub.target_kind == TARGET) & (pub.pctl == PCTL)
              & (pub.layer == LAYER) & (pub.model == MODEL)]
    pubmap = {str(u): float(v) for u, v in
              zip(pub["held_out_university"], pub["lift_top10"])}
    check = []
    for s in singles:
        got, want = s["lift_top10"], pubmap.get(s["held_out"])
        check.append({"campus": s["held_out"], "published": want, "refit": got,
                      "delta": None if want is None else round(got - want, 4)})
    deltas = [abs(c["delta"]) for c in check if c["delta"] is not None]
    max_delta = max(deltas) if deltas else float("nan")

    prov = {
        "config": f"{TARGET}|p{PCTL}|{LAYER}|{MODEL}",
        "n_pair_fits": len(paired), "n_single_fits": len(singles),
        "wall_seconds": round(wall, 1), "n_jobs": N_JOBS, "threads": _THREADS,
        "reproduction_check": check,
        "max_abs_lift_delta_vs_published": None if not deltas else round(max_delta, 4),
        "singles": singles, "pairs": paired,
    }
    json.dump(prov, open(C.METRICS / "double_holdout_fits.json", "w"), indent=2)

    print(f"\n[fits] done in {wall/60:.1f} min")
    print("[fits] reproduction of published per-campus lifts:")
    for c in check:
        print(f"    campus {c['campus']:>2}  published {c['published']}  "
              f"refit {c['refit']}  delta {c['delta']}")
    print(f"[fits] max |delta| = {max_delta}")
    if deltas and max_delta > 0.01:
        print("[fits] WARNING: refit does not reproduce the published screen; "
              "the harness differs and the certification numbers are not comparable.")


if __name__ == "__main__":
    main()
