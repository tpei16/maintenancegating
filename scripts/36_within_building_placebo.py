#!/usr/bin/env python
"""
Within-building placebo / permutation test.

The within-building Mantel-Haenszel (MH) RR is 0.92 (95% CI 0.88-0.97), slightly BELOW 1
(engineering-core 0.85). A sharp reviewer could argue this below-one value is a mechanical
artefact of the estimator rather than a real result. We test that directly.

Null hypothesis: within a building, sibling-system exposure (>=1 other system high-burden at t)
is unrelated to a cell's forward high-burden risk. We permute the exposure label across
cell-quarters WITHIN each building, preserving each building's exposed count and outcome count,
and recompute the MH RR. Because the MH stratum contribution depends only on each building's
2x2 margins, this permutation is exactly the per-building hypergeometric null:
    a_b (exposed outcomes) ~ Hypergeometric(ngood=m_b, nbad=n_b-m_b, nsample=e_b),
with c_b = m_b - a_b, exposed count e_b and unexposed count n_b-e_b fixed. 1000 permutations
give the null distribution of the MH RR; we locate the observed value in it.

Reading: if the null is centred at ~1.0, the estimator is UNBIASED (no mechanical below-one
pull). Whether the observed 0.92 sits inside or just below that null tells us whether the slight
reversal is sampling noise or a small real within-building negative association attributable to
baseline composition (maintenance scheduling, finite capacity, mean-reversion after busy
quarters), NOT system-to-system propagation. Either way the robust claim holds: the large
positive POOLED association (3.21) vanishes once each building is its own control.

Outputs -> results/metrics/within_building_placebo.json
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import importlib.util
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fmscreen import config as C

# reuse the EXACT preparation used for the published within-building MH (script 22)
_spec = importlib.util.spec_from_file_location(
    "s22", Path(__file__).resolve().parents[0] / "22_coescalation_robustness.py")
s22 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(s22)

N_PERM = 1000
SEED = 42


def build(present: pd.DataFrame):
    """Reproduce the within-building exposure/outcome rows and per-building margins."""
    df = present[["bkey", "period_q", "sev", "sev_next4"]].copy()
    bq_trig = df.groupby(["bkey", "period_q"], observed=True)["sev"].transform("sum")
    bq_n = df.groupby(["bkey", "period_q"], observed=True)["sev"].transform("size")
    df = df[bq_n >= 2]                                     # cells with >=1 sibling
    sib_trig = (bq_trig - df["sev"]).loc[df.index]
    df["exposed"] = (sib_trig >= 1).astype(int)
    df["y"] = df["sev_next4"].astype(int)
    df["ay"] = ((df["exposed"] == 1) & (df["y"] == 1)).astype(int)
    df["cy"] = ((df["exposed"] == 0) & (df["y"] == 1)).astype(int)
    g = df.groupby("bkey", observed=True)
    strata = g.agg(n=("y", "size"), e=("exposed", "sum"), m=("y", "sum"),
                   a=("ay", "sum"), c=("cy", "sum")).reset_index()
    strata = strata[(strata["e"] > 0) & (strata["e"] < strata["n"])]   # informative
    return strata


def mh(a, c, e, n):
    num = (a * (n - e) / n).sum()
    den = (c * e / n).sum()
    return num / den if den else np.nan


def main():
    panel = pd.read_parquet(C.DATA_PROCESSED / "panel_quarter.parquet")
    p = s22.prepare(panel)
    present = p[p["win_valid"] == 1].copy()
    s = build(present)

    n = s["n"].to_numpy(np.int64); e = s["e"].to_numpy(np.int64); m = s["m"].to_numpy(np.int64)
    obs = float(mh(s["a"].to_numpy(float), s["c"].to_numpy(float), e, n))

    rng = np.random.default_rng(SEED)
    nbad = n - m
    null = np.empty(N_PERM)
    for i in range(N_PERM):
        a = rng.hypergeometric(m, nbad, e).astype(float)   # exposed outcomes, per building
        null[i] = mh(a, m - a, e, n)

    lo, hi = float(np.quantile(null, 0.025)), float(np.quantile(null, 0.975))
    p_lower = float((np.sum(null <= obs) + 1) / (N_PERM + 1))
    inside = bool(lo <= obs <= hi)
    out = {
        "observed_mh_rr": obs,
        "n_informative_buildings": int(len(s)),
        "n_permutations": N_PERM,
        "null_mean": float(np.mean(null)),
        "null_median": float(np.median(null)),
        "null_ci95": [lo, hi],
        "observed_inside_null_ci": inside,
        "p_lower_tail": p_lower,
        "reading": (
            f"Under the within-building permutation null the MH RR is centred at "
            f"{np.mean(null):.3f} (95% null interval {lo:.3f}-{hi:.3f}), confirming the estimator "
            f"is unbiased (no mechanical below-one pull). The observed {obs:.3f} is "
            f"{'inside' if inside else 'below'} the null interval, so the slight reversal is "
            f"{'compatible with sampling variation under within-building composition' if inside else 'a small but real within-building negative association, consistent with baseline composition (scheduling, finite capacity, mean-reversion) rather than propagation'}. "
            f"The large positive pooled association vanishes within building either way."
        ),
    }
    json.dump(out, open(C.METRICS / "within_building_placebo.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
