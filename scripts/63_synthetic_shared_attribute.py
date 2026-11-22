#!/usr/bin/env python
"""
63_synthetic_shared_attribute.py -- the shared-attribute construction of the
synthetic benchmark, reported with the quantities of the main synthetic table,
and two simpler certification rules scored on the same simulated strata.

43_synthetic_certification.py builds two renderings of a stratum whose records
cannot support a decision because the recorded driver does not transfer:

  C_non_transferable      four contexts share one driver, the other four each
                          hold a private one (the headline regime);
  sensitivity_sign_flip   four contexts share one driver, the other four share
                          ONE other attribute with signs flipped in two of them.

For the headline regime script 43 stores the pooled top-decile lift, the share
of held-out contexts clearing two-fold, S, the recovery rate, and the recovery
rate with the transfer-stability gate removed from the mean and from the
hard-fail set.  For the shared-attribute construction it stores all of these
except the last.

This script reruns all five constructions (the four main regimes, three strata
each, and the shared-attribute construction, one stratum) with script 43's own
generator, seeds, protocol and gate code, imported and not copied: stratum j of
construction i in replicate r draws from
numpy.random.default_rng([BASE_SEED, r, i, j]) and fits with seed
BASE_SEED % 10**6 + r, exactly as in script 43's main loop.  It asserts that the
rerun reproduces every quantity script 43 stored for each construction, then
adds the recovery without the transfer-stability gate for the shared-attribute
construction.

On the same per-stratum values it scores two comparison rules that read no
held-out transfer evidence, by their recovery rate (share of strata whose
verdict equals the ground truth):

  count rule         certify when at least 3 contexts are evaluable, there are
                     at least 100 positives, and the no-antecedent share is
                     below 0.25 (a missing share counts as not below 0.25);
  pooled-lift rule   certify when the pooled top-decile lift is at least 2.

Overall accuracy over the 600 main-regime strata is reported for the
certificate and for both rules.

Outputs -> results/metrics/synthetic_shared_attribute.json
"""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["FMSCREEN_DEVICE"] = "cpu"
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "FMSCREEN_THREADS"):
    os.environ[_v] = "4"

import sys, json, time, importlib.util
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fmscreen import config as C

_spec = importlib.util.spec_from_file_location(
    "synthetic43", ROOT / "scripts" / "43_synthetic_certification.py")
S43 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S43)

SHARED = "sensitivity_sign_flip"
PASS, CAUTION, FAIL = S43.PASS, S43.CAUTION, S43.FAIL
COUNT_RULE = dict(min_contexts=3, min_pos=100, max_zero_prior=0.25)
LIFT_RULE = dict(min_pooled_lift=2.0)


def run_construction(regime: str, i: int, variants: list[dict], t0: float) -> pd.DataFrame:
    """Every stratum of one construction, with script 43's seeds and harness."""
    rows = []
    for rep in range(S43.N_REPLICATES):
        for j, variant in enumerate(variants):
            p = S43.stratum_params(regime, variant)
            rng = np.random.default_rng([S43.BASE_SEED, rep, i, j])
            sim = S43.simulate(p, rng)
            panel = S43.featurise(sim)
            pred = S43.leave_one_context_out(panel, seed=S43.BASE_SEED % 10 ** 6 + rep)
            r = S43.gate_stratum(regime + variant["tag"], pred, sim)
            r.update(regime=regime, variant=variant["tag"], replicate=rep)
            rows.append(r)
    print(f"  {regime}: {len(rows)} strata  ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def rule_verdicts(df: pd.DataFrame) -> pd.DataFrame:
    zp = pd.to_numeric(df["zero_prior_share"], errors="coerce")
    out = pd.DataFrame(index=df.index)
    out["ctx_ok"] = df["n_contexts_evaluable"] >= COUNT_RULE["min_contexts"]
    out["pos_ok"] = df["n_pos"] >= COUNT_RULE["min_pos"]
    out["trace_ok"] = zp.notna() & (zp < COUNT_RULE["max_zero_prior"])   # missing -> not below
    out["count_rule"] = out.ctx_ok & out.pos_ok & out.trace_ok
    out["lift_rule"] = df["pooled_lift_top10"] >= LIFT_RULE["min_pooled_lift"]
    out["certificate"] = df["S"] >= S43.S_CUT
    out["certificate_without_transfer_gate"] = df["S_without_transfer_gate"] >= S43.S_CUT
    return out


def table_row(recovery, recovery_no_transfer, S, lift, frac, truth, n) -> dict:
    return {"ground_truth_actionable": truth, "n_strata": n,
            "pooled_lift_top10_median": lift, "frac_contexts_meet_2x_median": frac,
            "S_median": S, "recovery_rate": recovery,
            "recovery_rate_without_transfer_gate": recovery_no_transfer}


def main() -> None:
    t0 = time.time()
    stored = json.load(open(C.METRICS / "synthetic_certification.json"))

    regimes = list(S43.GEN) + list(S43.GEN_SENS)
    frames = {}
    for i, regime in enumerate(regimes):
        variants = S43.STRATUM_VARIANTS if regime in S43.GEN else S43.STRATUM_VARIANTS[1:2]
        frames[regime] = run_construction(regime, i, variants, t0)

    # ---- the rerun must reproduce everything script 43 stored ----
    for regime in S43.GEN:
        g, want = frames[regime], stored["per_regime"][regime]
        truth = S43.GROUND_TRUTH[regime]
        got = {
            "n_strata_scored": int(len(g)),
            "S": S43.quartiles(g["S"]),
            "recovery_rate": round(float(((g["S"] >= S43.S_CUT) == truth).mean()), 4),
            "recovery_rate_without_transfer_gate": round(float(
                ((g["S_without_transfer_gate"] >= S43.S_CUT) == truth).mean()), 4),
            "gate_firing_rates": S43.firing_rates(g),
            "evidence": {
                "pooled_lift_top10": S43.quartiles(g["pooled_lift_top10"]),
                "frac_contexts_meet_2x": S43.quartiles(g["frac_contexts_meet_2x"]),
                "calibration_top_band_lift": S43.quartiles(g["top_band_lift"]),
                "n_pos": S43.quartiles(g["n_pos"]),
                "n_contexts_evaluable": S43.quartiles(g["n_contexts_evaluable"]),
                "zero_prior_share": S43.quartiles(g["zero_prior_share"]),
            },
        }
        for k, v in got.items():
            assert v == want[k], f"{regime}: rerun differs from the stored {k}: {v} vs {want[k]}"
    s, want = frames[SHARED], stored["sensitivity_sign_flip"]
    got_shared = {
        "n_strata": int(len(s)),
        "S": S43.quartiles(s["S"]),
        "recovery_rate": round(float((s["S"] < S43.S_CUT).mean()), 4),
        "gate_firing_rates": S43.firing_rates(s),
        "pooled_lift_top10": S43.quartiles(s["pooled_lift_top10"]),
        "frac_contexts_meet_2x": S43.quartiles(s["frac_contexts_meet_2x"]),
    }
    for k, v in got_shared.items():
        assert v == want[k], f"{SHARED}: rerun differs from the stored {k}: {v} vs {want[k]}"
    main_all = pd.concat([frames[r] for r in S43.GEN], ignore_index=True)
    main_truth = main_all["regime"].map(S43.GROUND_TRUTH).astype(bool)
    overall_cert = round(float(((main_all["S"] >= S43.S_CUT) == main_truth).mean()), 4)
    assert overall_cert == stored["headline"]["overall_recovery_rate"], overall_cert
    print("[63] rerun reproduces every stored quantity of all five constructions", flush=True)

    # ---- the shared-attribute construction, with the quantity script 43 does not store ----
    rec_nt = round(float((s["S_without_transfer_gate"] < S43.S_CUT).mean()), 4)
    flips = s[(s["S"] >= S43.S_CUT) != (s["S_without_transfer_gate"] >= S43.S_CUT)]
    shared = {
        **table_row(got_shared["recovery_rate"], rec_nt, got_shared["S"]["median"],
                    got_shared["pooled_lift_top10"]["median"],
                    got_shared["frac_contexts_meet_2x"]["median"], False, got_shared["n_strata"]),
        "S": got_shared["S"],
        "S_without_transfer_gate": S43.quartiles(s["S_without_transfer_gate"]),
        "pooled_lift_top10": got_shared["pooled_lift_top10"],
        "frac_contexts_meet_2x": got_shared["frac_contexts_meet_2x"],
        "gate_firing_rates": got_shared["gate_firing_rates"],
        "risk_concentration_pass_rate": round(float((s.risk_concentration == PASS).mean()), 4),
        "transfer_stability_fail_rate": round(float((s.transfer_stability == FAIL).mean()), 4),
        "share_with_pooled_lift_ge_2": round(float((s.pooled_lift_top10 >= 2.0).mean()), 4),
        "n_decisions_flipped_without_transfer_gate": int(len(flips)),
        "flip_direction": ("not certified -> certified" if len(flips) and
                           bool((flips["S_without_transfer_gate"] >= S43.S_CUT).all())
                           else ("none" if not len(flips) else "mixed")),
        "S_value_counts": {str(k): int(v) for k, v in
                           s["S"].round(3).value_counts().sort_index().items()},
        "S_without_transfer_gate_value_counts": {
            str(k): int(v) for k, v in
            s["S_without_transfer_gate"].round(3).value_counts().sort_index().items()},
        "per_context_lift_pooled": S43.quartiles([x for v in s["per_context_lift"] for x in v]),
    }

    table = {}
    for reg, b in stored["per_regime"].items():
        table[reg] = table_row(b["recovery_rate"], b["recovery_rate_without_transfer_gate"],
                               b["S"]["median"], b["evidence"]["pooled_lift_top10"]["median"],
                               b["evidence"]["frac_contexts_meet_2x"]["median"],
                               b["ground_truth_actionable"], b["n_strata_scored"])
    table["shared_attribute_sign_flip"] = table_row(
        shared["recovery_rate"], shared["recovery_rate_without_transfer_gate"],
        shared["S_median"], shared["pooled_lift_top10_median"],
        shared["frac_contexts_meet_2x_median"], False, shared["n_strata"])

    # ---- two comparison rules on the same strata ----
    names = {**{r: r for r in S43.GEN}, SHARED: "shared_attribute_sign_flip"}
    rules = {}
    for regime, df in frames.items():
        truth = bool(S43.GROUND_TRUTH.get(regime, False))
        v = rule_verdicts(df)
        rules[names[regime]] = {
            "ground_truth_actionable": truth, "n_strata": int(len(df)),
            "recovery_certificate": round(float((v.certificate == truth).mean()), 4),
            "recovery_count_rule": round(float((v.count_rule == truth).mean()), 4),
            "recovery_pooled_lift_rule": round(float((v.lift_rule == truth).mean()), 4),
            "share_certified": {"certificate": round(float(v.certificate.mean()), 4),
                                "count_rule": round(float(v.count_rule.mean()), 4),
                                "pooled_lift_rule": round(float(v.lift_rule.mean()), 4)},
            "count_rule_conditions_met": {
                "contexts_evaluable_ge_3": round(float(v.ctx_ok.mean()), 4),
                "positives_ge_100": round(float(v.pos_ok.mean()), 4),
                "zero_prior_share_lt_0.25": round(float(v.trace_ok.mean()), 4),
                "zero_prior_share_missing": round(float(
                    pd.to_numeric(df["zero_prior_share"], errors="coerce").isna().mean()), 4)},
        }
    vm = rule_verdicts(main_all)
    overall = {
        "n_strata": int(len(main_all)),
        "certificate": round(float((vm.certificate == main_truth).mean()), 4),
        "count_rule": round(float((vm.count_rule == main_truth).mean()), 4),
        "pooled_lift_rule": round(float((vm.lift_rule == main_truth).mean()), 4),
    }
    assert overall["certificate"] == overall_cert

    out = {
        "design": ("The shared-attribute rendering of non-transferability (four contexts share one "
                   "driver, the other four share one other attribute with signs flipped in two), "
                   "rerun with script 43's generator, seeds, leave-one-context-out protocol and gate "
                   "code beside the four main regimes, reported with the quantities of the main "
                   "synthetic table, and two certification rules without transfer evidence (a count "
                   "rule and a pooled-lift rule) scored on the same strata by recovery of the "
                   "ground truth."),
        "source": {"generator_and_harness": "scripts/43_synthetic_certification.py (imported)",
                   "constructions": regimes,
                   "shared_attribute_parameters": S43.GEN_SENS[SHARED],
                   "shared_attribute_stratum_variant": S43.STRATUM_VARIANTS[1],
                   "seeds": {"simulation": f"default_rng([{S43.BASE_SEED}, r, i, j])",
                             "model": f"{S43.BASE_SEED % 10 ** 6} + r"},
                   "n_replicates": S43.N_REPLICATES},
        "reproduced_stored_quantities": {
            "main_regimes": ["n_strata_scored", "S", "recovery_rate",
                             "recovery_rate_without_transfer_gate", "gate_firing_rates",
                             "evidence"],
            "shared_attribute": sorted(got_shared),
            "overall_recovery_rate": overall_cert},
        "table": table,
        "shared_attribute": shared,
        "comparison_rules": {
            "definitions": {
                "count_rule": ("certify when n_contexts_evaluable >= 3 and n_pos >= 100 and "
                               "zero_prior_share < 0.25; a missing zero_prior_share counts as "
                               "not below 0.25"),
                "pooled_lift_rule": "certify when pooled top-decile lift >= 2.0",
                "recovery": "share of strata whose verdict equals the ground truth",
            },
            "by_regime": rules,
            "overall_main_regimes": overall,
        },
        "runtime_seconds": round(time.time() - t0, 1),
    }
    json.dump(out, open(C.METRICS / "synthetic_shared_attribute.json", "w"), indent=2, default=str)

    print("\n regime                       truth  lift   frac2x  S     recovery  w/o transfer")
    for reg, r in table.items():
        print(f"  {reg:28s} {str(r['ground_truth_actionable']):5s}  "
              f"{r['pooled_lift_top10_median']:.2f}  {r['frac_contexts_meet_2x_median']:.3f}  "
              f"{r['S_median']:.2f}  {r['recovery_rate']:.3f}     "
              f"{r['recovery_rate_without_transfer_gate']:.3f}")
    print("\n recovery                     certificate  count rule  pooled-lift rule")
    for reg, r in rules.items():
        print(f"  {reg:28s} {r['recovery_certificate']:.4f}       {r['recovery_count_rule']:.4f}      "
              f"{r['recovery_pooled_lift_rule']:.4f}")
    print(f"  {'overall (600 main strata)':28s} {overall['certificate']:.4f}       "
          f"{overall['count_rule']:.4f}      {overall['pooled_lift_rule']:.4f}")
    print(f"[63] done in {out['runtime_seconds']} s")


if __name__ == "__main__":
    main()
