#!/usr/bin/env python
"""
43_synthetic_certification.py -- does the record-evidence sufficiency score S
recover a KNOWN ground truth about whether a stratum's records can be acted on?

The sufficiency score S in 37_cepi.py certifies, per stratum, whether a predictive
screen over maintenance records is trustworthy enough to act on directly. On real
data there is no ground truth to check it against: we never learn whether a stratum
the gates rejected was in fact unusable. This script builds that ground truth by
simulation. It generates panels whose data-generating process is known by
construction to support -- or not to support -- a records-only decision, runs the
paper's own leave-one-context-out protocol and the paper's own gate code over them,
and measures how often the certificate agrees with the truth.

GROUND TRUTH (four regimes, three strata each, 50 replicates)
  sufficient        next-quarter high burden is driven by the unit's own past-only
                    history (a persistent burden process), with the SAME relationship
                    in every context.                              -> ACTIONABLE
  shock             high burden arrives from a memoryless process independent of
                    history, so no past-only feature can rank it.  -> NOT ACTIONABLE
  non-transferable  a genuine, strong past-only driver exists in every context, but
                    WHICH recorded attribute drives it varies: four contexts share
                    one driver, the other four each have a private one. A model
                    pooled over contexts therefore ranks well overall yet fails on
                    half the held-out contexts.                    -> NOT ACTIONABLE
  scarce            the same process as the sufficient regime, but observed in two
                    contexts with fewer than 100 positive events.  -> NOT ACTIONABLE

The non-transferable regime is the discriminating case: it is built so that pooled
top-10% lift (the risk-concentration gate) is comfortably above its 2.0 pass level
while the leave-one-context-out transfer gate fails. The counterfactual below prices
that gate: it recomputes every decision with the transfer gate deleted from the mean
and from the hard-fail set.

GENERATIVE MODEL (one equation set, per-regime parameters; see GEN below)
  context intensity   m_c      deterministic ladder; in the non-transferable regime
                               two levels tied to driver alignment (see NOTE 2)
  unit baseline       a_i      = m_c + N(0, sigma_unit)
  dormancy            d_i      ~ N(0, sigma_dorm), enters ticket frequency ONLY
  latent state        z_it     AR(1) with coefficient phi  (phi = 0 is the shock regime)
  recorded channels   u_kit    six independent past-only recorded attributes
  channel effect      chan_it  = sign_c * w_chan * u_{k(c), i, t-1}
  severity            sev_it   = exp(a_i + w_z z_it + chan_it + sigma_e eps_it)
  label               y_it     = 1{sev_i,t+1 >= stratum quantile at 1 - base_rate}
  ticket count        n_it     = Poisson(exp(log_lam0 + kappa a_i + d_i + theta z_it))
                                 + 1{severe}   (a severe quarter always emits a ticket)
  recorded burden     b_it     = sev_it where n_it > 0, else 0

Features are past-only summaries of the RECORDED series (b, n, u) up to quarter t;
the label is the next quarter. Severity is latent and always defined while the
record trail is a separate Poisson process, so the record-density parameters move
the antecedent gate without weakening the predictive signal.

GATE CODE IS NOT REIMPLEMENTED. scripts/37_cepi.py is loaded as a module and its
own objects are used unchanged:
    THR, GATES, HARD_GATES, PASS/CAUTION/FAIL, build_gates(), calibration_gate(),
    sufficiency()
so gate thresholds, the pass/caution/fail cutoffs, the hard-gate set and the capping
rule (>=1 hard fail -> S capped at 0.50; >=2 -> S = 0) are the paper's, by
construction. The only new combination logic is sufficiency_subset(), needed for the
counterfactual; it is asserted equal to 37_cepi.sufficiency() on the full gate set
for every score combination the run encounters. verify_worked_example() additionally
checks the whole harness against a hand-computed example.

The model is the pipeline's own factory, fmscreen.models.make_gbdt, forced onto CPU
with a four-thread cap (the study's primary configuration otherwise unchanged), and
the protocol is the study's leave-one-context-out design with the same
within-held-out-context percentile rank R.

NOTE 1 (honest, and visible in the output). In the non-transferable regime the
antecedent gate returns caution or fail, because severity there is driven by a
recorded attribute rather than by persistent activity, so the prior-record trace is
thinner than in the sufficient regime. It is a soft gate and never decides that
regime's verdict; the transfer gate is the only hard fail.

NOTE 2 (a design choice a reviewer should see). In the non-transferable regime the
four contexts that share the common driver are also the four higher-intensity ones.
Without that coupling, pooled lift sits at about 2.0 rather than comfortably above
it, because each leave-one-context-out fold produces its own score scale and the
between-context intensity effect does not survive pooling. The coupling is realistic
-- the campuses where a screen works are often the busiest ones -- but it is an
assumption, not a derivation.

NOTE 3 (a sensitivity, reported separately). An alternative rendering of
non-transferability, in which the four idiosyncratic contexts share ONE attribute
with flipped signs rather than holding private ones, is run alongside as
`sensitivity_sign_flip`. The boosted model partially recovers a sign-flipped driver
through a symmetric function of it, so those contexts reach roughly 1.5x rather than
chance and the transfer gate's verdict becomes borderline. That variant is reported,
not hidden, but it is not the headline regime.

Outputs:
  results/metrics/synthetic_certification.json
"""
from __future__ import annotations

import os

# thread caps must precede numpy / xgboost import (OpenMP sizes its pool at load)
N_THREADS = 4
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "FMSCREEN_THREADS"):
    os.environ[_v] = str(N_THREADS)

import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fmscreen import config as C          # noqa: E402
from fmscreen.metrics import lift_at_topk  # noqa: E402
from fmscreen.models import make_gbdt      # noqa: E402

MET = ROOT / "results" / "metrics"
OUT = MET / "synthetic_certification.json"

BASE_SEED = 20260802
N_REPLICATES = 50
S_CUT = 0.67          # the paper's "records are informative" cut (37_cepi.classify)
TOPK = C.TOPK_BUDGET  # 0.10, the study's inspection budget
WARMUP = 4            # quarters consumed by the lag window before the first feature row
N_CHANNELS = 6        # recorded past-only attributes u1..u6 available to every model


# --------------------------------------------------------------- the paper's gate code
def load_cepi():
    """Load scripts/37_cepi.py as a module (its name is not a valid identifier)."""
    path = ROOT / "scripts" / "37_cepi.py"
    spec = importlib.util.spec_from_file_location("cepi37", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)   # module level only defines constants + functions
    return mod


CEPI = load_cepi()
GATES, HARD_GATES = CEPI.GATES, CEPI.HARD_GATES
PASS, CAUTION, FAIL = CEPI.PASS, CEPI.CAUTION, CEPI.FAIL


def sufficiency_subset(scores, gates, hard):
    """37_cepi.sufficiency() generalised to an arbitrary gate subset.

    Identical rule: equal-weighted mean of the retained gate scores, then
    >=2 hard fails -> 0.0, >=1 hard fail -> capped at 0.50. Needed only for the
    counterfactual that deletes a gate; asserted equal to the original on the full
    gate set for every combination this run produces.
    """
    S = float(np.mean([scores[g] for g in gates]))
    n_hard_fail = sum(1 for g in hard if scores[g] == FAIL)
    if n_hard_fail >= 2:
        return 0.0
    if n_hard_fail >= 1:
        return min(S, 0.50)
    return S


# --------------------------------------------------------------- generative parameters
# Shared across regimes; only the fields listed per regime differ.
COMMON = dict(
    n_ctx=8,            # contexts (campus analogue)
    n_units=60,         # units per context (building-system analogue)
    n_q=20,             # quarters observed
    base_rate=0.15,     # share of unit-quarters labelled high burden
    sigma_e=0.50,       # idiosyncratic severity noise (log scale)
    log_lam0=-2.2,      # baseline routine ticket rate per unit-quarter
    kappa=0.5,          # unit burden level -> ticket rate
    theta=0.65,         # latent state -> ticket rate
    sigma_dorm=1.0,     # unit dormancy spread (ticket frequency only)
    jitter=0.15,        # context-intensity jitter where two levels are used
    ctx_levels=None,    # None -> deterministic ladder of width sigma_ctx
    w_chan=0.0,         # recorded-channel coefficient
    chan=None,          # per-context (channel index, sign); None -> no channel
)

GEN = {
    # ---- regime A: history-driven, one relationship everywhere -> ACTIONABLE
    "A_sufficient": {**COMMON, "phi": 0.82, "sigma_ctx": 0.20, "sigma_unit": 0.45,
                     "w_z": 1.30},
    # ---- regime B: memoryless shocks -> NOT ACTIONABLE
    "B_shock": {**COMMON, "phi": 0.00, "sigma_ctx": 0.20, "sigma_unit": 0.00,
                "w_z": 1.30},
    # ---- regime C: strong driver in every context, but not the same one
    #      contexts 0-3 share channel u1 and are the higher-intensity half;
    #      contexts 4-7 each hold a private channel (u2..u5).
    "C_non_transferable": {**COMMON, "phi": 0.82, "sigma_ctx": 0.0, "sigma_unit": 0.00,
                           "w_z": 0.00, "w_chan": 1.50,
                           "ctx_levels": [0.4] * 4 + [-0.4] * 4,
                           "chan": [(1, +1), (1, +1), (1, +1), (1, +1),
                                    (2, +1), (3, +1), (4, +1), (5, +1)]},
    # ---- regime D: regime A's process, two contexts, <100 positives
    "D_scarce": {**COMMON, "phi": 0.82, "sigma_ctx": 0.20, "sigma_unit": 0.45,
                 "w_z": 1.30, "n_ctx": 2, "n_units": 18, "n_q": 16},
}

# Sensitivity variant of regime C (NOTE 3): the four idiosyncratic contexts share one
# attribute with flipped signs instead of holding private ones. Reported separately.
GEN_SENS = {
    "sensitivity_sign_flip": {**GEN["C_non_transferable"],
                              "chan": [(1, +1), (1, +1), (1, +1), (1, +1),
                                       (2, +1), (2, +1), (2, -1), (2, -1)]},
}

GROUND_TRUTH = {          # can this stratum's records support a direct decision?
    "A_sufficient": True,
    "B_shock": False,
    "C_non_transferable": False,
    "D_scarce": False,
}

# Three strata per regime, so a regime is not a single point. Deterministic offsets:
# a stratum is thinner/thicker, rarer/commoner, and weaker/stronger than its siblings.
STRATUM_VARIANTS = [
    dict(tag="1", base_rate=0.12, units_delta=-5, signal=0.92),
    dict(tag="2", base_rate=0.15, units_delta=0, signal=1.00),
    dict(tag="3", base_rate=0.18, units_delta=+5, signal=1.08),
]

FEATURES = (["b_lag1", "b_lag2", "b_mean4", "b_max4", "n_lag1", "n_sum4", "level",
             "n_prior_high", "t_since_high"]
            + [f"u{k}_lag1" for k in range(1, N_CHANNELS + 1)])


def stratum_params(regime: str, variant: dict) -> dict:
    """Apply a stratum variant to a regime's parameters."""
    p = dict(GEN.get(regime) or GEN_SENS[regime])
    p["base_rate"] = variant["base_rate"]
    # the scarce regime is small by design, so its variants move by less
    delta = variant["units_delta"] if p["n_ctx"] > 2 else round(variant["units_delta"] / 2.5)
    p["n_units"] = max(4, p["n_units"] + delta)
    p["w_z"] = p["w_z"] * variant["signal"]
    p["w_chan"] = p["w_chan"] * variant["signal"]
    return p


# --------------------------------------------------------------- simulation
def simulate(p: dict, rng: np.random.Generator) -> dict:
    """One stratum panel: contexts x units x quarters."""
    n_ctx, n_units, n_q = p["n_ctx"], p["n_units"], p["n_q"]
    n = n_ctx * n_units
    ctx = np.repeat(np.arange(n_ctx), n_units)

    if p["ctx_levels"] is not None:                     # two intensity levels
        m_c = np.asarray(p["ctx_levels"], float) + rng.normal(0, p["jitter"], n_ctx)
    else:                                               # deterministic ladder
        ladder = np.linspace(-1.0, 1.0, n_ctx) if n_ctx > 1 else np.zeros(1)
        m_c = p["sigma_ctx"] * ladder

    a = m_c[ctx] + rng.normal(0.0, p["sigma_unit"], n)   # unit burden level
    dorm = rng.normal(0.0, p["sigma_dorm"], n)           # ticket frequency only

    def ar1(phi):
        x = np.empty((n, n_q))
        x[:, 0] = rng.normal(0, 1, n)
        for t in range(1, n_q):
            x[:, t] = phi * x[:, t - 1] + np.sqrt(1 - phi ** 2) * rng.normal(0, 1, n)
        return x

    z = ar1(p["phi"])                                    # persistent (or memoryless) state
    u = [ar1(0.0) for _ in range(N_CHANNELS)]            # recorded attributes, memoryless

    chan = np.zeros((n, n_q))
    if p["chan"] is not None:
        for c, (which, sign) in enumerate(p["chan"]):
            rows = ctx == c
            chan[rows, 1:] = sign * p["w_chan"] * u[which - 1][rows, :-1]

    sev = np.exp(a[:, None] + p["w_z"] * z + chan + p["sigma_e"] * rng.normal(0, 1, (n, n_q)))
    thr = np.quantile(sev, 1 - p["base_rate"])
    severe = (sev >= thr).astype(int)

    lam = np.exp(p["log_lam0"] + p["kappa"] * a[:, None] + dorm[:, None] + p["theta"] * z)
    counts = rng.poisson(lam) + severe                   # a severe quarter emits a ticket
    burden = np.where(counts > 0, sev, 0.0)              # recorded burden

    return dict(ctx=ctx, burden=burden, counts=counts, u=u, severe=severe,
                n_units_total=n, n_q=n_q)


def featurise(sim: dict) -> pd.DataFrame:
    """Past-only feature rows at quarter t, labelled by quarter t+1."""
    b, n, hi = sim["burden"], sim["counts"], sim["severe"]
    rows = []
    for t in range(WARMUP, sim["n_q"] - 1):
        win = slice(t - 3, t + 1)
        h = hi[:, :t + 1]
        last = np.where(h.any(1), t - np.argmax(h[:, ::-1], axis=1), -99)
        d = {
            "ctx": sim["ctx"], "quarter": t,
            "b_lag1": b[:, t], "b_lag2": b[:, t - 1],
            "b_mean4": b[:, win].mean(1), "b_max4": b[:, win].max(1),
            "n_lag1": n[:, t], "n_sum4": n[:, win].sum(1),
            "level": b[:, :t + 1].mean(1),
            "n_prior_high": h.sum(1),
            "t_since_high": np.where(last < 0, 99, t - last),
            "y": hi[:, t + 1],
        }
        for k in range(N_CHANNELS):
            d[f"u{k + 1}_lag1"] = sim["u"][k][:, t]
        rows.append(pd.DataFrame(d))
    return pd.concat(rows, ignore_index=True)


def leave_one_context_out(panel: pd.DataFrame, seed: int) -> pd.DataFrame:
    """The study's protocol: train on every context but one, predict the held-out one."""
    frames = []
    for c in sorted(panel.ctx.unique()):
        train, test = panel[panel.ctx != c], panel[panel.ctx == c]
        model = make_gbdt(None, None, seed=seed)
        model.set_params(device="cpu", n_jobs=N_THREADS)   # n_jobs is xgboost's nthread
        model.fit(train[FEATURES], train["y"])
        out = test[["ctx", "y"]].copy()
        out["score"] = model.predict_proba(test[FEATURES])[:, 1]
        out["held_out_university"] = c
        frames.append(out)
    pred = pd.concat(frames, ignore_index=True)
    # R: within-held-out-context percentile rank of the score (37_cepi.load_units)
    pred["R"] = pred.groupby("held_out_university")["score"].rank(pct=True, method="average")
    return pred


def prior_record_trace(sim: dict) -> tuple[int, float]:
    """Extreme events with >=4 quarters of history, and the share with no prior ticket.

    Mirrors 18_trajectory / 40_outer_s: extreme = above the 90th percentile of
    non-zero recorded burden; zero-prior = no work orders in the prior four quarters.
    """
    b, n = sim["burden"], sim["counts"]
    positive = b[b > 0]
    if positive.size == 0:
        return 0, float("nan")
    thr = np.quantile(positive, 0.90)
    n_events = n_zero = 0
    for t in range(WARMUP, sim["n_q"]):
        m = b[:, t] > thr
        n_events += int(m.sum())
        n_zero += int((n[m, t - 4:t].sum(1) == 0).sum())
    return n_events, (n_zero / n_events if n_events else float("nan"))


def gate_stratum(stratum: str, pred: pd.DataFrame, sim: dict) -> dict:
    """Assemble the gate inputs the paper's build_gates() consumes, then call it."""
    lifts = [lift_at_topk(g["y"].to_numpy(), g["score"].to_numpy(), TOPK)
             for _, g in pred.groupby("held_out_university")
             if g["y"].sum() >= 10 and len(g) >= 100]          # 40_outer_s evaluability
    pooled = float(lift_at_topk(pred["y"].to_numpy(), pred["score"].to_numpy(), TOPK))
    frac_2x = float(np.mean([l >= 2 for l in lifts])) if lifts else float("nan")
    n_events, zero_prior = prior_record_trace(sim)

    units = pred.assign(system=stratum)
    het = pd.DataFrame([{"system": stratum,
                         "n_campuses_evaluable": len(lifts),
                         "n_pos": int(pred["y"].sum()),
                         "lift_top10": pooled,
                         "frac_campuses_meet_2x": frac_2x}])
    traj = {stratum: {"n_events": n_events, "zero_prior": zero_prior}}
    # build_gates ignores its `dep` argument; `name`/`share` are label lookups only.
    row = CEPI.build_gates(units, het, het.copy(), {}, {}, traj)[0]

    scores = {g: row[g] for g in GATES}
    S_full = CEPI.sufficiency(scores)
    # the generalised combiner must agree with the paper's on the full gate set
    assert abs(sufficiency_subset(scores, GATES, HARD_GATES) - S_full) < 1e-12, scores

    keep = [g for g in GATES if g != "transfer_stability"]
    keep_hard = [g for g in HARD_GATES if g != "transfer_stability"]
    return {
        "stratum": stratum, **scores,
        "S": S_full,
        "S_without_transfer_gate": sufficiency_subset(scores, keep, keep_hard),
        "n_rows": int(len(pred)), "n_pos": int(pred["y"].sum()),
        "base_rate": float(pred["y"].mean()),
        "n_contexts_evaluable": len(lifts),
        "pooled_lift_top10": pooled,
        "frac_contexts_meet_2x": frac_2x,
        "per_context_lift": [round(float(x), 3) for x in lifts],
        "top_band_lift": row["_cal"].get("top_band_lift"),
        "band_spearman": row["_cal"].get("band_spearman"),
        "n_extreme_events": n_events,
        "zero_prior_share": None if np.isnan(zero_prior) else float(zero_prior),
    }


# --------------------------------------------------------------- harness verification
def verify_worked_example() -> dict:
    """Check the whole harness against an example whose gates are computed by hand.

    Inputs chosen so every gate is derivable on paper:
      breadth 7 contexts, 620 positives   -> 7 >= 6 and 620 >= 500          -> pass
      pooled top-10% lift 2.4             -> 2.4 >= 2.0                     -> pass
      contexts clearing 2x: 0.55          -> 0.55 < 0.60                    -> FAIL
      500 scored rows, R uniform, positives per score quintile 2/4/8/16/40:
        base rate 70/500 = 0.14, top band 40/100 = 0.40,
        top-band lift 0.40/0.14 = 2.857 >= 2.0 and rho = 1.0 >= 0.9         -> pass
      200 extreme events, 18% with no prior ticket -> 0.10 <= 0.18 < 0.25   -> caution
    Mean = (1 + 1 + 0 + 1 + 0.5)/5 = 0.70; one hard fail (transfer) caps it at 0.50.
    Deleting the transfer gate: mean(1, 1, 1, 0.5) = 0.875, no hard fail    -> 0.875.
    """
    n = 500
    per_band = [2, 4, 8, 16, 40]
    y = np.concatenate([np.concatenate([np.ones(k), np.zeros(100 - k)]) for k in per_band])
    units = pd.DataFrame({"system": "WORKED", "y": y.astype(int),
                          "R": np.linspace(0.0, 1.0, n)})
    het = pd.DataFrame([{"system": "WORKED", "n_campuses_evaluable": 7, "n_pos": 620,
                         "lift_top10": 2.4, "frac_campuses_meet_2x": 0.55}])
    traj = {"WORKED": {"n_events": 200, "zero_prior": 0.18}}
    row = CEPI.build_gates(units, het, het.copy(), {}, {}, traj)[0]
    scores = {g: row[g] for g in GATES}

    expected_scores = {"data_sufficiency": PASS, "risk_concentration": PASS,
                       "transfer_stability": FAIL, "calibration_reliability": PASS,
                       "antecedent_signal": CAUTION}
    expected_S, expected_S_no_transfer = 0.50, 0.875

    S = CEPI.sufficiency(scores)
    keep = [g for g in GATES if g != "transfer_stability"]
    keep_hard = [g for g in HARD_GATES if g != "transfer_stability"]
    S_no_transfer = sufficiency_subset(scores, keep, keep_hard)

    assert scores == expected_scores, (scores, expected_scores)
    assert abs(S - expected_S) < 1e-12, (S, expected_S)
    assert abs(S_no_transfer - expected_S_no_transfer) < 1e-12
    assert abs(row["_cal"]["top_band_lift"] - 2.857) < 5e-4, row["_cal"]
    assert abs(row["_cal"]["band_spearman"] - 1.0) < 1e-12, row["_cal"]
    return {
        "inputs": {"n_contexts_evaluable": 7, "n_pos": 620, "pooled_lift_top10": 2.4,
                   "frac_contexts_meet_2x": 0.55, "positives_per_score_quintile": per_band,
                   "n_extreme_events": 200, "zero_prior_share": 0.18},
        "hand_computed": {"gate_scores": expected_scores, "top_band_lift": 2.857,
                          "band_spearman": 1.0, "S": expected_S,
                          "S_without_transfer_gate": expected_S_no_transfer},
        "computed_by_37_cepi": {"gate_scores": scores,
                                "top_band_lift": row["_cal"]["top_band_lift"],
                                "band_spearman": row["_cal"]["band_spearman"],
                                "S": S, "S_without_transfer_gate": S_no_transfer},
        "agree": True,
    }


# --------------------------------------------------------------- aggregation
def quartiles(v) -> dict:
    v = np.asarray([x for x in v if x is not None and not (isinstance(x, float) and np.isnan(x))],
                   dtype=float)
    if v.size == 0:
        return {"median": None, "q25": None, "q75": None, "min": None, "max": None, "n": 0}
    return {"median": round(float(np.median(v)), 4), "q25": round(float(np.quantile(v, .25)), 4),
            "q75": round(float(np.quantile(v, .75)), 4), "min": round(float(v.min()), 4),
            "max": round(float(v.max()), 4), "n": int(v.size)}


def firing_rates(df: pd.DataFrame) -> dict:
    """Share of strata at fail / caution / pass, per gate, for one regime."""
    out = {}
    for g in GATES:
        v = df[g].to_numpy(float)
        out[g] = {"fail": round(float((v == FAIL).mean()), 4),
                  "caution": round(float((v == CAUTION).mean()), 4),
                  "pass": round(float((v == PASS).mean()), 4)}
    return out


def main():
    t0 = time.time()
    verification = verify_worked_example()
    print(f"harness verified against the hand-computed example "
          f"(S = {verification['hand_computed']['S']}, "
          f"S without the transfer gate = {verification['hand_computed']['S_without_transfer_gate']})")

    regimes = list(GEN) + list(GEN_SENS)
    rows = []
    for rep in range(N_REPLICATES):
        for i, regime in enumerate(regimes):
            # the sensitivity variant uses the middle stratum only (NOTE 3)
            variants = STRATUM_VARIANTS if regime in GEN else STRATUM_VARIANTS[1:2]
            for j, variant in enumerate(variants):
                p = stratum_params(regime, variant)
                # only the replicate index varies the draw; regime/stratum index keeps
                # independent streams within a replicate
                rng = np.random.default_rng([BASE_SEED, rep, i, j])
                sim = simulate(p, rng)
                panel = featurise(sim)
                pred = leave_one_context_out(panel, seed=BASE_SEED % 10 ** 6 + rep)
                r = gate_stratum(regime + variant["tag"], pred, sim)
                r.update(regime=regime, variant=variant["tag"], replicate=rep,
                         is_sensitivity=regime in GEN_SENS,
                         ground_truth_actionable=GROUND_TRUTH.get(regime))
                rows.append(r)
        if (rep + 1) % 10 == 0:
            print(f"  replicate {rep + 1}/{N_REPLICATES}  ({time.time() - t0:.0f}s)")

    df = pd.DataFrame(rows)
    df["decision"] = df["S"] >= S_CUT
    df["decision_without_transfer_gate"] = df["S_without_transfer_gate"] >= S_CUT

    main_df = df[~df.is_sensitivity].copy()
    main_df["correct"] = main_df.decision == main_df.ground_truth_actionable
    main_df["correct_without_transfer_gate"] = (
        main_df.decision_without_transfer_gate == main_df.ground_truth_actionable)

    # ---------------- per-regime blocks
    per_regime = {}
    for regime, g in main_df.groupby("regime"):
        per_regime[regime] = {
            "ground_truth_actionable": bool(GROUND_TRUTH[regime]),
            "n_strata_scored": int(len(g)),
            "S": quartiles(g["S"]),
            "recovery_rate": round(float(g["correct"].mean()), 4),
            "recovery_rate_without_transfer_gate": round(float(g["correct_without_transfer_gate"].mean()), 4),
            "gate_firing_rates": firing_rates(g),
            "evidence": {
                "pooled_lift_top10": quartiles(g["pooled_lift_top10"]),
                "frac_contexts_meet_2x": quartiles(g["frac_contexts_meet_2x"]),
                "calibration_top_band_lift": quartiles(g["top_band_lift"]),
                "n_pos": quartiles(g["n_pos"]),
                "n_contexts_evaluable": quartiles(g["n_contexts_evaluable"]),
                "zero_prior_share": quartiles(g["zero_prior_share"]),
            },
            "S_value_counts": {str(k): int(v) for k, v in g["S"].round(3).value_counts().sort_index().items()},
        }

    # ---------------- the non-redundancy evidence
    c = main_df[main_df.regime == "C_non_transferable"]
    non_redundancy = {
        "claim": ("the non-transferable regime is caught by the transfer-stability gate "
                  "and not by risk concentration: pooled lift stays above its pass level "
                  "while the fraction of held-out contexts clearing 2x falls below it"),
        "n_strata": int(len(c)),
        "risk_concentration_pass_rate": round(float((c.risk_concentration == PASS).mean()), 4),
        "transfer_stability_fail_rate": round(float((c.transfer_stability == FAIL).mean()), 4),
        "calibration_pass_rate": round(float((c.calibration_reliability == PASS).mean()), 4),
        "pooled_lift_top10": quartiles(c["pooled_lift_top10"]),
        "frac_contexts_meet_2x": quartiles(c["frac_contexts_meet_2x"]),
        "share_with_pooled_lift_ge_2": round(float((c.pooled_lift_top10 >= 2.0).mean()), 4),
        "share_with_transfer_as_only_hard_fail": round(float(
            ((c.transfer_stability == FAIL)
             & (c.data_sufficiency != FAIL)
             & (c.calibration_reliability != FAIL)).mean()), 4),
    }

    # ---------------- the counterfactual: delete the transfer gate
    flip = main_df[main_df.decision != main_df.decision_without_transfer_gate]
    counterfactual = {
        "definition": ("S recomputed with transfer_stability removed from the mean AND "
                       "from the hard-fail set; everything else unchanged"),
        "accuracy_with_all_five_gates": round(float(main_df["correct"].mean()), 4),
        "accuracy_without_transfer_gate": round(float(main_df["correct_without_transfer_gate"].mean()), 4),
        "accuracy_change": round(float(main_df["correct_without_transfer_gate"].mean()
                                       - main_df["correct"].mean()), 4),
        "per_regime_recovery_without_transfer_gate": {
            k: round(float(v["correct_without_transfer_gate"].mean()), 4)
            for k, v in main_df.groupby("regime")},
        "n_decisions_flipped": int(len(flip)),
        "flips_by_regime": {k: int(v) for k, v in flip.regime.value_counts().items()},
        "flip_direction": ("all flips are insufficient -> sufficient, i.e. strata whose "
                           "records cannot support a decision are wrongly certified"
                           if bool((flip.decision_without_transfer_gate).all()) or len(flip) == 0
                           else "mixed"),
    }

    # ---------------- the sign-flip sensitivity (NOTE 3)
    s = df[df.is_sensitivity]
    sensitivity = {
        "design": ("the four idiosyncratic contexts share one recorded attribute with "
                   "flipped signs instead of holding private ones; ground truth is still "
                   "not actionable"),
        "n_strata": int(len(s)),
        "S": quartiles(s["S"]),
        "recovery_rate": round(float((s["S"] < S_CUT).mean()), 4),
        "gate_firing_rates": firing_rates(s),
        "pooled_lift_top10": quartiles(s["pooled_lift_top10"]),
        "frac_contexts_meet_2x": quartiles(s["frac_contexts_meet_2x"]),
        "reading": ("the boosted model partially recovers a sign-flipped driver through a "
                    "symmetric function of it, so the idiosyncratic contexts land near 1.5x "
                    "rather than at chance and the transfer gate sits on its "
                    "caution/fail boundary"),
    }

    elapsed = time.time() - t0
    result = {
        "purpose": ("does the record-evidence sufficiency score S recover a known ground "
                    "truth about whether a stratum's records can support a decision?"),
        "decision_rule": f"records are actionable when S >= {S_CUT}",
        "gate_code": {
            "source": "scripts/37_cepi.py, imported and used unchanged",
            "objects_used": ["THR", "GATES", "HARD_GATES", "PASS/CAUTION/FAIL",
                             "build_gates", "calibration_gate", "sufficiency"],
            "thresholds": CEPI.THR,
            "gates": GATES, "hard_gates": HARD_GATES,
            "new_code": ("sufficiency_subset(), needed only for the counterfactual; "
                         "asserted equal to 37_cepi.sufficiency() on the full gate set "
                         "for every combination encountered"),
        },
        "harness_verification": verification,
        "protocol": {
            "design": "leave-one-context-out, mirroring the study's leave-one-campus-out",
            "model": ("fmscreen.models.make_gbdt (the pipeline's own factory: XGBoost, "
                      "500 trees, depth 6, lr 0.06, class-balanced), forced to CPU"),
            "threads": N_THREADS,
            "topk_budget": TOPK,
            "R": "within-held-out-context percentile rank of the model score",
            "context_evaluability": "n >= 100 and n_pos >= 10 (as in 40_outer_s.py)",
        },
        "experiment": {
            "base_seed": BASE_SEED,
            "n_replicates": N_REPLICATES,
            "regimes": {k: ("actionable" if v else "not actionable")
                        for k, v in GROUND_TRUTH.items()},
            "strata_per_regime": len(STRATUM_VARIANTS),
            "stratum_variants": STRATUM_VARIANTS,
            "n_decisions": int(len(main_df)),
            "features": FEATURES,
            "warmup_quarters": WARMUP,
            "generative_parameters": GEN,
            "sensitivity_parameters": GEN_SENS,
            "design_notes": {
                "note_1_antecedent_in_C": ("in the non-transferable regime the antecedent "
                                           "gate returns caution or fail because severity is "
                                           "driven by a recorded attribute rather than by "
                                           "persistent activity; it is soft and never decides "
                                           "that regime's verdict"),
                "note_2_intensity_coupling": ("in the non-transferable regime the four "
                                              "contexts sharing the common driver are also the "
                                              "higher-intensity half; without that coupling "
                                              "pooled lift sits at about 2.0 rather than "
                                              "comfortably above it"),
                "note_3_sign_flip": ("an alternative rendering of non-transferability is "
                                     "reported under `sensitivity_sign_flip`"),
            },
        },
        "headline": {
            "overall_recovery_rate": round(float(main_df["correct"].mean()), 4),
            "per_regime_recovery_rate": {k: round(float(v["correct"].mean()), 4)
                                         for k, v in main_df.groupby("regime")},
            "per_regime_S_median": {k: quartiles(v["S"])["median"]
                                    for k, v in main_df.groupby("regime")},
        },
        "per_regime": per_regime,
        "non_redundancy": non_redundancy,
        "counterfactual_drop_transfer_gate": counterfactual,
        "sensitivity_sign_flip": sensitivity,
        "runtime_seconds": round(elapsed, 1),
        "environment": {"python": platform.python_version(),
                        "numpy": np.__version__, "pandas": pd.__version__,
                        "xgboost": __import__("xgboost").__version__},
    }

    MET.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(result, fh, indent=2, default=str)

    # ---------------- console report
    print("\n=== S BY REGIME (median [IQR] over strata x replicates) ===")
    for regime in GEN:
        b = per_regime[regime]
        q = b["S"]
        truth = "actionable" if b["ground_truth_actionable"] else "not actionable"
        print(f"  {regime:22s} truth={truth:15s} S={q['median']:.2f} "
              f"[{q['q25']:.2f}-{q['q75']:.2f}]  recovery={b['recovery_rate']:.3f}  n={b['n_strata_scored']}")
    print(f"  {'OVERALL':22s} {'':22s} recovery={result['headline']['overall_recovery_rate']:.3f} "
          f"over {len(main_df)} decisions")

    print("\n=== GATE FIRING RATES (share of strata at FAIL) ===")
    hdr = "  " + " " * 22 + "".join(f"{g.split('_')[0][:9]:>11s}" for g in GATES)
    print(hdr)
    for regime in GEN:
        line = f"  {regime:22s}"
        for g in GATES:
            line += f"{per_regime[regime]['gate_firing_rates'][g]['fail']:>11.2f}"
        print(line)

    print("\n=== NON-REDUNDANCY: the non-transferable regime ===")
    nr = non_redundancy
    print(f"  pooled top-10% lift          median {nr['pooled_lift_top10']['median']:.2f} "
          f"[{nr['pooled_lift_top10']['q25']:.2f}-{nr['pooled_lift_top10']['q75']:.2f}]  "
          f"(>= 2.0 in {nr['share_with_pooled_lift_ge_2']:.0%} of strata)")
    print(f"  risk-concentration gate      passes in {nr['risk_concentration_pass_rate']:.0%}")
    print(f"  contexts clearing 2x         median {nr['frac_contexts_meet_2x']['median']:.3f}")
    print(f"  transfer-stability gate      fails  in {nr['transfer_stability_fail_rate']:.0%}")
    print(f"  transfer is the only hard fail in {nr['share_with_transfer_as_only_hard_fail']:.0%}")

    print("\n=== COUNTERFACTUAL: delete the transfer-stability gate ===")
    cf = counterfactual
    print(f"  accuracy with all five gates   {cf['accuracy_with_all_five_gates']:.3f}")
    print(f"  accuracy without the gate      {cf['accuracy_without_transfer_gate']:.3f} "
          f"({cf['accuracy_change']:+.3f})")
    print(f"  decisions flipped              {cf['n_decisions_flipped']}  {cf['flips_by_regime']}")

    print("\n=== SENSITIVITY (sign-flipped driver instead of private drivers) ===")
    print(f"  S median {sensitivity['S']['median']:.2f}  "
          f"contexts clearing 2x median {sensitivity['frac_contexts_meet_2x']['median']:.3f}  "
          f"recovery {sensitivity['recovery_rate']:.3f}  n={sensitivity['n_strata']}")

    print(f"\nWrote {OUT}  ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
