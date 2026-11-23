#!/usr/bin/env python
"""
71_peer_comparability.py -- outcome-free comparability of a target campus with
its peers, a labour-recording audit, and the record flow from the raw file to
the analytical panel.

The verdict for a target campus c and building system s is formed from the
records of the other eight campuses, without any outcome of c.  Whether c
resembles its peers is a condition of that verdict.  This script measures the
resemblance from features and record counts only, relates it to what the
ranking later achieved on c, audits how labour hours, which define the outcome,
are recorded on each campus, and reconciles the record counts of the corpus.

PART 1. COMPARABILITY DIAGNOSTICS (no outcome label of the target is read).
For each target c, against the pooled units of its eight peers:
  size              units, buildings, quarters, building systems of individual
                    buildings (pair level: units, buildings, quarters of s on c)
  record density    reactive and preventive work orders per unit (a unit is one
                    building system of one building in one quarter)
  preventive share  preventive / (preventive + reactive) work orders
  empty reactive    share of units with no reactive work order in the four
                    quarters up to and including the anchor quarter
                    (upm_cnt_w4 == 0)
  labour recorded   share of work orders with positive labour hours
  system mix        total-variation distance between the target's and the
                    peers' distribution of units over building systems (pair
                    level: |log| of the ratio of the system's share of units)
  deviation         |log(target / peers)| for the densities and |target - peers|
                    for the shares
  feature shift     cross-validated AUC of a gradient-boosted classifier (CPU,
                    fixed settings) that separates the target's units from peer
                    units.  Peer units are subsampled to the target's count
                    (each class capped at CAP_PER_CLASS units); five folds,
                    stratified by class and grouped by building.  Features: the
                    screening model's features without the four weather
                    variables (record features; weather is a campus-quarter
                    mean and identifies the campus by itself) and with them (all
                    features).  The two reactive-premium features are constants
                    of the building system computed on the training campuses and
                    carry nothing beyond the system code, so they are omitted.
                    The AUC is reported pooled over the out-of-fold scores with
                    its range over folds, overall and per building system (from
                    the same out-of-fold scores), and at a common size (every
                    campus subsampled to the smallest campus's unit count, five
                    draws) so that it does not move with sample size.
Two windows.  "heldout": all units of c against all peer units; related to the
held-out check (top-decile lift of f^(-c) on c, scripts 44 and 56).  "forward":
the units of c anchored in 2017 to 2019 against the peer units anchored in 2016
or earlier, the training rows of f_h^(-c); related to the forward test
(script 60).  Rank correlations over the nine campuses use the exact
permutation distribution (9! orderings).  Pair-level associations use the
evaluable pairs and permute the outcome within building system.  Eligibility
rules that read one diagnostic are evaluated by the counts of pairs short of
two-fold and of pairs at or above it that they would withhold.

PART 2. LABOUR-RECORDING AUDIT of the reactive work orders that enter the panel,
by campus and by campus and building system: share with positive, zero, missing
and negative labour hours; median and upper percentiles of positive hours;
share above the system's 99th percentile of positive hours over the nine
campuses; the high-burden base rate with the whole-panel thresholds and with
the thresholds fitted on the other campuses.

PART 3. RECORD FLOW from the raw file to the analytical panel, step by step,
for work orders, building systems of individual buildings and buildings, with
every count of Table 1 and of the reconstructed work-order stream checked.

INVARIANTS (asserted before anything new is computed): the verdicts of the
held-out maps are reproduced (script 56's pipeline and reference check against
cert_corrected.json, and results/tables/fold_gate_verdicts.csv); the held-out
lifts of the 141 campus-system pairs and the 2017 to 2019 lifts of the 159
pairs of the forward test are recomputed from the stored predictions and equal
the stored tables; the work orders aggregated by unit equal the panel's
counts and labour sums; the diagnostic frames hold no outcome column.

Outputs -> results/metrics/peer_comparability.json
           results/tables/peer_comparability_by_campus.csv
           results/tables/peer_comparability_by_pair.csv
           results/tables/labor_recording_audit.csv
           results/tables/record_flow.csv
"""
from __future__ import annotations
import os

# CPU only, with the thread caps set before numpy or fmscreen is imported
N_THREADS = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["FMSCREEN_DEVICE"] = "cpu"
os.environ["FMSCREEN_THREADS"] = N_THREADS
_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS")
for _v in _THREAD_VARS:
    os.environ[_v] = N_THREADS

import sys, json, time, itertools, math, importlib.util
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import rankdata, binom, beta
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fmscreen import config as C
from fmscreen import io as F
from fmscreen import features as FE
from fmscreen import metrics as MET


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cc = _load("cert_corrected", "45_certification_corrected.py")
fvm = _load("fold_verdicts", "56_fold_verdicts_and_simple_rule.py")
lh = _load("local_history_baseline", "60_local_history_baseline.py")
# script 60 sets its own thread caps on import; restore the caps of this run
os.environ["FMSCREEN_THREADS"] = N_THREADS
for _v in _THREAD_VARS:
    os.environ[_v] = N_THREADS

SEED = C.RANDOM_SEED
HIST_END, EVAL_YEARS = lh.HIST_END, lh.EVAL_YEARS
MIN_POS, MIN_ROWS = lh.MIN_POS, lh.MIN_ROWS          # evaluability of a pair outcome
LIFT_LEVEL = 2.0
S_CUT = cc.S_CUT
SYS = C.COL_SYSTEM
WEATHER = ["min_temp", "max_temp", "precip", "humidity"]
PREMIUMS = ["prem_labour_sys", "prem_cost_sys"]
_NUM = [c for c in FE.M0_NUMERIC + FE.M1_EXTRA_NUMERIC if c not in PREMIUMS]
FEATURES = {"record": [c for c in _NUM if c not in WEATHER] + [SYS],
            "all": _NUM + [SYS]}
OUTCOME_COLS = ["upm_count_next", "upm_labour_next", "upm_cost_next", "occurrence_next", "y"]
CAP_PER_CLASS = 50_000
N_FOLDS = 5
N_SIZE_DRAWS = 5
MIN_PAIR_AUC_UNITS = 30         # per class, pooled out-of-fold, for a pair-level AUC
MIN_FOLD_AUC_UNITS = 10         # per class within a fold, for the fold range
N_PERM = 10_000
EXTREME_PCTL = 0.99
AUDIT_MIN_WO = 100              # reactive work orders of a pair for the cross-campus spread
XGB = dict(device="cpu", tree_method="hist", enable_categorical=True, n_estimators=200,
           max_depth=4, learning_rate=0.1, min_child_weight=5.0, subsample=0.8,
           colsample_bytree=0.8, reg_lambda=1.0, eval_metric="logloss",
           n_jobs=int(N_THREADS), verbosity=0)
WINDOWS = {
    "heldout": {"target": lambda yr: np.ones(len(yr), bool), "peer": lambda yr: np.ones(len(yr), bool),
                "label": "all units of the target against all peer units"},
    "forward": {"target": lambda yr: (yr >= EVAL_YEARS[0]) & (yr <= EVAL_YEARS[-1]),
                "peer": lambda yr: yr <= HIST_END,
                "label": ("units of the target anchored in 2017 to 2019 against peer units "
                          "anchored in 2016 or earlier")},
}

# values printed in the manuscript (Table 1 and the reconstructed work-order stream table)
TABLE1 = {"work_orders_fmucd": 3731442, "preventive_fmucd": 2224663, "reactive_fmucd": 1469204,
          "no_category_fmucd": 37575, "campuses": 9, "panel_preventive": 1909259,
          "panel_reactive": 886723, "buildings": 3400, "building_systems_of_buildings": 22546,
          "uniformat_systems": 21, "units": 514975, "quarters": 75,
          "first_quarter": "2002 Q3", "last_quarter": "2021 Q1",
          "occurrence_pct": 29.0, "high_burden_p75_pct": 6.8, "high_burden_p50_pct": 13.5,
          "high_burden_p90_pct": 2.7}
TABLE_STREAM = {"work_orders_read": 3731442, "work_orders_retained": 2854673,
                "building_systems_with_a_work_order": 23593, "time_of_day_pct": 61.8,
                "distinct_times_of_day": 74943, "sharing_exact_timestamp_pct": 48.4,
                "at_least_one_day_apart_pct": 37.6, "median_gap_days": 0.0,
                "median_prior_work_orders": 8, "decision_points_ge8_pct": 50.8,
                "p95_prior_work_orders": 329, "subsystem_component_pct": 100.0,
                "labour_recorded_pct": 94.1}
TEXT = {"excluded_universities_record_pct": 22.0, "labour_positive_pct": 82, "cost_positive_pct": 44}


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def qlabel(q: int) -> str:
    return f"{int(q) // 4} Q{int(q) % 4 + 1}"


def r4(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), 4)


def absdiff(a, b):
    return abs(a - b) if np.isfinite(a) and np.isfinite(b) else np.nan


def abslogratio(a, b):
    return abs(math.log(a / b)) if (np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0) else np.nan


# --------------------------------------------------------------------------
# PART 3 first: raw records and the record flow
# --------------------------------------------------------------------------
def load_records() -> pd.DataFrame:
    """The raw file with the columns the panel reads, system codes normalized as
    in fmscreen.panel.build."""
    cols = [C.COL_UNIV, C.COL_BUILDING, C.COL_SYSTEM, C.COL_PPMUPM, C.COL_START,
            C.COL_LABORHOURS, C.COL_TOTALCOST, C.COL_SUBSYS, C.COL_COMPONENT]
    df = F.load_raw(usecols=cols)
    df = F.add_time_keys(df, anchor=C.COL_START)
    df[SYS] = df[SYS].astype("string").str.strip().str.upper()
    df["university"] = df[C.COL_UNIV].astype(str)
    df["is_upm"] = F.is_upm(df).to_numpy()
    df["is_ppm"] = F.is_ppm(df).to_numpy()
    return df


def record_flow(raw: pd.DataFrame, panel: pd.DataFrame) -> tuple[list, dict, pd.DataFrame]:
    """Sequential exclusions from the raw file to the panel's work orders, with
    cells (building systems of individual buildings) and buildings at each step.
    Returns the step table, the checks, and the work orders of the panel units."""
    n_raw = len(raw)
    panel_univ = set(panel["university"].unique())
    excluded = sorted(set(raw["university"].unique()) - panel_univ, key=int)
    assert excluded == sorted(C.UNIVERSITIES_NO_BUILDING, key=int), excluded
    # the excluded universities record no building identifier at all
    assert raw.loc[raw.university.isin(excluded), C.COL_BUILDING].notna().sum() == 0

    steps = []

    def cells_bld(m):
        d = raw.loc[m, ["university", C.COL_BUILDING, SYS]]
        d = d.dropna()
        return (int(d.drop_duplicates().shape[0]),
                int(d[["university", C.COL_BUILDING]].drop_duplicates().shape[0]))

    def add(step, desc, m_prev, m_now, cells=None, blds=None):
        steps.append({"step": step, "description": desc,
                      "work_orders_removed": int(m_prev.sum() - m_now.sum()),
                      "work_orders_remaining": int(m_now.sum()),
                      "preventive_remaining": int((m_now & raw.is_ppm).sum()),
                      "reactive_remaining": int((m_now & raw.is_upm).sum()),
                      "uncategorized_remaining": int((m_now & ~raw.is_ppm & ~raw.is_upm).sum()),
                      "building_systems_remaining": cells, "buildings_remaining": blds})

    m0 = np.ones(n_raw, bool)
    add(0, "work orders in the raw file, 12 universities", m0, m0)
    m1 = raw.university.isin(panel_univ).to_numpy()
    add(1, f"records of the {len(excluded)} universities without building identifiers "
           f"({', '.join(excluded)}) removed", m0, m1)
    m2 = m1 & raw[C.COL_BUILDING].notna().to_numpy()
    add(2, "records without a building identifier removed (nine campuses)", m1, m2)
    m3 = m2 & raw[SYS].notna().to_numpy()
    add(3, "records without a building-system code removed", m2, m3, *cells_bld(m3))
    m4 = m3 & raw["period_q"].notna().to_numpy()
    add(4, "records without a parseable start date removed", m3, m4, *cells_bld(m4))
    m5 = m4 & (raw.is_ppm | raw.is_upm).to_numpy()
    add(5, "records with no maintenance category removed; the reconstructed work-order stream",
        m4, m5, *cells_bld(m5))

    # the panel's grid is built from every record left after step 4
    a = raw.loc[m4, ["university", C.COL_BUILDING, SYS, "period_q", "year", "is_upm", "is_ppm",
                     C.COL_LABORHOURS, C.COL_START, C.COL_SUBSYS, C.COL_COMPONENT]].copy()
    a["period_q"] = a["period_q"].astype(int)
    bkey = ["university", C.COL_BUILDING]
    ckey = bkey + [SYS]
    blast = a.groupby(bkey, observed=True)["period_q"].max().rename("building_last_q")
    bfirst = a.groupby(bkey, observed=True)["period_q"].min().rename("building_first_q")
    cfirst = a.groupby(ckey, observed=True)["period_q"].min().rename("cell_first_q")
    a = a.merge(blast.reset_index(), on=bkey, how="left")
    in_panel = (a["period_q"] < a["building_last_q"]).to_numpy()
    m6 = m5.copy()
    idx4 = np.flatnonzero(m4)
    m6[idx4[~in_panel]] = False
    cells = cfirst.reset_index().merge(blast.reset_index(), on=bkey)
    cells_kept = cells[cells.cell_first_q < cells.building_last_q]
    blds = bfirst.reset_index().merge(blast.reset_index(), on=bkey)
    add(6, ("work orders dated in the last active quarter of their building removed: that "
            "quarter has no next quarter on record, so its units carry no outcome and leave "
            "the panel"), m5, m6, int(len(cells_kept)),
        int((blds.building_first_q < blds.building_last_q).sum()))
    panel_wo = a[in_panel & (a.is_ppm | a.is_upm).to_numpy()].copy()

    # units: the grid runs from a cell's first quarter to its building's last
    # active quarter, and the last row of every cell is dropped
    grid_rows = int((cells.building_last_q - cells.cell_first_q + 1).sum())
    units = grid_rows - int(len(cells))
    n_cells4, n_blds4 = cells_bld(m4)
    single_q_cells = int((cells.cell_first_q == cells.building_last_q).sum())
    single_q_blds = int((blds.building_first_q == blds.building_last_q).sum())

    # the work orders of the panel units, aggregated by unit, equal the panel
    lab = panel_wo[C.COL_LABORHOURS].fillna(0.0)
    agg = (panel_wo.assign(upm=panel_wo.is_upm.astype(int), ppm=panel_wo.is_ppm.astype(int),
                           upm_lab=np.where(panel_wo.is_upm, lab, 0.0))
           .groupby(ckey + ["period_q"], observed=True)
           .agg(upm=("upm", "sum"), ppm=("ppm", "sum"), upm_lab=("upm_lab", "sum")).reset_index())
    pk = panel[["university", C.COL_BUILDING, SYS, "period_q", "upm_count", "ppm_count",
                "upm_labour"]]
    j = pk.merge(agg, on=ckey + ["period_q"], how="left", indicator=True)
    assert len(j) == len(panel)
    assert int((j["_merge"] == "both").sum()) == len(agg), "a unit with work orders is not in the panel"
    for c in ("upm", "ppm", "upm_lab"):
        j[c] = j[c].fillna(0.0)
    assert (j.upm.to_numpy() == j.upm_count.to_numpy()).all()
    assert (j.ppm.to_numpy() == j.ppm_count.to_numpy()).all()
    assert np.allclose(j.upm_lab.to_numpy(), j.upm_labour.to_numpy(), rtol=1e-9, atol=1e-6)

    # ---- every printed count, reproduced ----
    s5 = raw.loc[m5]
    ts = s5[C.COL_START]
    tod = ts.dt.hour * 3600 + ts.dt.minute * 60 + ts.dt.second
    ev = s5[["university", C.COL_BUILDING, SYS, C.COL_START, "period_q"]].copy()
    ev = ev.sort_values(["university", C.COL_BUILDING, SYS, C.COL_START], kind="mergesort")
    gap = (ev[C.COL_START] - ev.groupby(["university", C.COL_BUILDING, SYS], observed=True)
           [C.COL_START].shift(1)).dt.total_seconds() / 86400.0
    n_pairs = int(gap.notna().sum())
    # history at a decision point: work orders of the cell up to and including t
    evq = (ev.groupby(["university", C.COL_BUILDING, SYS, "period_q"], observed=True).size()
             .rename("n").reset_index().sort_values(["university", C.COL_BUILDING, SYS, "period_q"]))
    evq["period_q"] = evq["period_q"].astype(int)
    evq["cum"] = evq.groupby(["university", C.COL_BUILDING, SYS], observed=True)["n"].cumsum()
    anchors = panel[["university", C.COL_BUILDING, SYS, "period_q"]]
    hj = anchors.merge(evq[["university", C.COL_BUILDING, SYS, "period_q", "cum"]],
                       on=["university", C.COL_BUILDING, SYS, "period_q"], how="left")
    assert len(hj) == len(panel)
    # each cell's first unit has work orders, so a forward fill stays within the cell
    hist = hj["cum"].ffill().fillna(0)

    rates = {}
    for pc in (50, 75, 90):
        t, g = FE.severity_thresholds(panel, target="labour", pctl=pc)
        rates[pc] = float(FE.apply_severity(panel, t, g, "labour").mean())

    raw_lab_pos = float((raw[C.COL_LABORHOURS].fillna(0) > 0).mean())
    raw_cost_pos = float((raw[C.COL_TOTALCOST].fillna(0) > 0).mean())
    reproduced_t1 = {
        "work_orders_fmucd": n_raw, "preventive_fmucd": int(raw.is_ppm.sum()),
        "reactive_fmucd": int(raw.is_upm.sum()),
        "no_category_fmucd": int((~raw.is_ppm & ~raw.is_upm).sum()),
        "campuses": int(panel.university.nunique()),
        "panel_preventive": int(panel.ppm_count.sum()), "panel_reactive": int(panel.upm_count.sum()),
        "buildings": int(panel[["university", C.COL_BUILDING]].drop_duplicates().shape[0]),
        "building_systems_of_buildings": int(panel[["university", C.COL_BUILDING, SYS]]
                                             .drop_duplicates().shape[0]),
        "uniformat_systems": int(panel[SYS].nunique()), "units": int(len(panel)),
        "quarters": int(panel.period_q.nunique()),
        "first_quarter": qlabel(panel.period_q.min()), "last_quarter": qlabel(panel.period_q.max()),
        "occurrence_pct": round(100 * float(panel.occurrence_next.mean()), 1),
        "high_burden_p75_pct": round(100 * rates[75], 1),
        "high_burden_p50_pct": round(100 * rates[50], 1),
        "high_burden_p90_pct": round(100 * rates[90], 1),
    }
    reproduced_stream = {
        "work_orders_read": n_raw, "work_orders_retained": int(m5.sum()),
        "building_systems_with_a_work_order": cells_bld(m5)[0],
        "time_of_day_pct": round(100 * float((tod > 0).mean()), 1),
        "distinct_times_of_day": int(tod.nunique()),
        "sharing_exact_timestamp_pct": round(100 * float((gap == 0).sum()) / len(ev), 1),
        "at_least_one_day_apart_pct": round(100 * float((gap >= 1).sum()) / len(ev), 1),
        "median_gap_days": round(float(gap.median()), 1),
        "median_prior_work_orders": int(hist.median()),
        "decision_points_ge8_pct": round(100 * float((hist >= 8).mean()), 1),
        "p95_prior_work_orders": int(hist.quantile(0.95)),
        "subsystem_component_pct": round(100 * float((s5[C.COL_SUBSYS].notna() &
                                                      s5[C.COL_COMPONENT].notna()).mean()), 1),
        "labour_recorded_pct": round(100 * float(s5[C.COL_LABORHOURS].notna().mean()), 1),
    }
    reproduced_text = {
        "excluded_universities_record_pct": round(100 * float((~m1).mean()), 1),
        "labour_positive_pct": round(100 * raw_lab_pos), "cost_positive_pct": round(100 * raw_cost_pos),
    }

    def compare(printed, got, table):
        return [{"table": table, "quantity": k, "printed": printed[k], "reproduced": got[k],
                 "match": printed[k] == got[k]} for k in printed]

    table_checks = (compare(TABLE1, reproduced_t1, "Table 1") +
                    compare(TABLE_STREAM, reproduced_stream, "work-order stream table") +
                    compare(TEXT, reproduced_text, "Data section text"))
    flow = {
        "steps": steps,
        "difference_work_orders": {
            "stream_minus_panel": int(m5.sum() - (reproduced_t1["panel_preventive"] +
                                                  reproduced_t1["panel_reactive"])),
            "explained_by_step_6": steps[6]["work_orders_removed"],
            "explained_by_step_5": steps[5]["work_orders_removed"],
        },
        "difference_building_systems": {
            "stream_minus_panel": cells_bld(m5)[0] - reproduced_t1["building_systems_of_buildings"],
            "cells_whose_first_quarter_is_their_buildings_last": single_q_cells,
            "cells_with_only_uncategorized_records": int(n_cells4 - cells_bld(m5)[0]),
        },
        "difference_buildings": {
            "stream_buildings": cells_bld(m5)[1], "panel_buildings": reproduced_t1["buildings"],
            "buildings_active_in_one_quarter_only": single_q_blds,
        },
        "units": {"grid_rows": grid_rows, "cells": int(len(cells)),
                  "units_grid_minus_last_row_per_cell": units,
                  "matches_panel": units == len(panel)},
        "panel_work_orders_by_category": {
            "preventive": int(panel_wo.is_ppm.sum()), "reactive": int(panel_wo.is_upm.sum())},
        "labour_hours_in_panel_work_orders": {
            "missing": int(panel_wo[C.COL_LABORHOURS].isna().sum()),
            "zero": int((panel_wo[C.COL_LABORHOURS] == 0).sum()),
            "negative": int((panel_wo[C.COL_LABORHOURS] < 0).sum()),
            "note": ("work orders without labour hours are not excluded: their hours count as "
                     "zero in the unit's labour sum, and negative entries are summed as recorded")},
        "consecutive_pairs": {
            "n_work_orders": int(len(ev)), "n_consecutive_pairs": n_pairs,
            "share_exact_timestamp_over_work_orders": r4((gap == 0).sum() / len(ev)),
            "share_exact_timestamp_over_pairs": r4((gap == 0).sum() / n_pairs),
            "share_ge_1_day_over_work_orders": r4((gap >= 1).sum() / len(ev)),
            "share_ge_1_day_over_pairs": r4((gap >= 1).sum() / n_pairs)},
        "table_checks": table_checks,
    }
    assert units == len(panel), (units, len(panel))
    assert flow["difference_work_orders"]["stream_minus_panel"] == steps[6]["work_orders_removed"]
    assert steps[6]["building_systems_remaining"] == reproduced_t1["building_systems_of_buildings"]
    assert steps[6]["work_orders_remaining"] == len(panel_wo)
    return steps, flow, panel_wo


# --------------------------------------------------------------------------
# reference: verdicts, held-out lifts and forward-test lifts
# --------------------------------------------------------------------------
def heldout_outcomes(single: pd.DataFrame, campuses: list[str]) -> tuple[pd.DataFrame, dict]:
    """Per pair and per campus, the top-decile lift of f^(-c) on c, recomputed
    from the stored predictions and checked against the stored tables."""
    stored = pd.read_csv(C.TABLES / "fold_gate_verdicts.csv", dtype={"campus": str})
    assert len(stored) == 141
    rows = []
    for (j, s), g in single.groupby(["university", "system"], observed=True):
        y, sc = g["y"].to_numpy(), g["score"].to_numpy()
        ev = bool(y.sum() >= MIN_POS and len(y) >= MIN_ROWS)
        m = max(1, int(np.ceil(C.TOPK_BUDGET * len(y))))
        k = int(y[MET._topk_mask(sc, C.TOPK_BUDGET)].sum())
        rows.append({"campus": j, "system": s, "rows": int(len(y)), "pos": int(y.sum()),
                     "evaluable": ev, "top_m": m, "top_pos": k,
                     "lift": float(MET.lift_at_topk(y, sc, C.TOPK_BUDGET)) if ev else np.nan})
    h = pd.DataFrame(rows)
    chk = stored.merge(h, on=["campus", "system"], how="left")
    chk["rows"] = chk["rows"].fillna(0).astype(int)
    chk["pos"] = chk["pos"].fillna(0).astype(int)
    chk["evaluable"] = chk["evaluable"].fillna(False).astype(bool)
    assert (chk.rows == chk.heldout_rows).all() and (chk.pos == chk.heldout_pos).all()
    assert (chk.evaluable == chk.heldout_evaluable).all()
    e = chk[chk.evaluable]
    assert np.allclose(e.lift.round(3), e.heldout_lift, atol=1e-9)
    assert len(e) == 95
    # campus-level lift over all building systems, as stored by script 44
    dh = {r["held_out"]: r["lift_top10"] for r in
          json.load(open(C.METRICS / "double_holdout_fits.json"))["singles"]}
    camp = {}
    for j in campuses:
        g = single[single.university == j]
        lf = float(MET.lift_at_topk(g.y.to_numpy(), g.score.to_numpy(), C.TOPK_BUDGET))
        assert abs(round(lf, 3) - dh[j]) < 1e-9, (j, lf, dh[j])
        camp[j] = {"lift": lf, **lift_ceiling(len(g), int(g.y.sum()))}
    out = stored[["campus", "system", "S", "certified"]].merge(h, on=["campus", "system"],
                                                               how="outer")
    out["has_verdict"] = out["S"].notna()
    out["supported"] = out["certified"].fillna(False).astype(bool)
    for c in ("rows", "pos", "top_m", "top_pos"):
        out[c] = out[c].fillna(0).astype(int)
    out["evaluable"] = out["evaluable"].fillna(False).astype(bool)
    return out.drop(columns=["certified"]), camp


def forward_outcomes(campuses: list[str]) -> tuple[pd.DataFrame, dict]:
    """Per pair and per campus, the 2017 to 2019 top-decile lift of f_h^(-c),
    recomputed from the stored pre-2017 predictions and checked."""
    stored = pd.read_csv(C.TABLES / "local_history_baseline.csv", dtype={"campus": str})
    ref = {r["campus"]: r for r in json.load(open(C.METRICS / "local_history_baseline.json"))
           ["reproduction_vs_temporal_louo"]["per_campus"]}
    rows, camp = [], {}
    for j in campuses:
        s = pd.read_parquet(lh.SINGLE_DIR / f"{j}.parquet")
        e = s[(s.period_q // 4).isin(EVAL_YEARS)]
        lf = float(MET.lift_at_topk(e.y.to_numpy(), e.score.to_numpy(), C.TOPK_BUDGET))
        assert abs(round(lf, 4) - ref[j]["lift_top10_cpu"]) < 1e-9 and len(e) == ref[j]["n_test"]
        camp[j] = {"lift": lf, **lift_ceiling(len(e), int(e.y.sum()))}
        for k in sorted(s.system.unique()):
            g = e[e.system == k]
            y, sc = g.y.to_numpy(), g.score.to_numpy()
            ev = bool(y.sum() >= MIN_POS and len(y) >= MIN_ROWS)
            m = max(1, int(np.ceil(C.TOPK_BUDGET * len(y)))) if len(y) else 0
            kk = int(y[MET._topk_mask(sc, C.TOPK_BUDGET)].sum()) if len(y) else 0
            rows.append({"campus": j, "system": k, "rows": int(len(y)), "pos": int(y.sum()),
                         "evaluable": ev, "top_m": m, "top_pos": kk,
                         "lift": float(MET.lift_at_topk(y, sc, C.TOPK_BUDGET)) if ev else np.nan})
    f = pd.DataFrame(rows)
    chk = stored.merge(f, on=["campus", "system"], how="left", validate="one_to_one")
    assert len(chk) == len(stored) == len(f) == 159
    assert (chk.rows == chk.n_eval_rows).all() and (chk.pos == chk.n_eval_pos).all()
    assert (chk["evaluable_x"].astype(bool) == chk["evaluable_y"].astype(bool)).all()
    e = chk[chk.evaluable_x.astype(bool)]
    assert len(e) == 91 and np.allclose(e.lift.round(3), e.lift_eval_model, atol=1e-9)
    out = f.merge(stored[["campus", "system", "S_peer", "verdict_peer", "cert_peer"]],
                  on=["campus", "system"], how="left")
    out = out.rename(columns={"S_peer": "S", "verdict_peer": "has_verdict", "cert_peer": "supported"})
    out["supported"] = out["supported"].astype(bool)
    return out, camp


def lift_ceiling(n: int, pos: int) -> dict:
    """Largest top-decile lift attainable on n units with pos high-burden units:
    a top decile of m units holds at most min(pos, m) of them."""
    m = max(1, int(np.ceil(C.TOPK_BUDGET * n)))
    base = pos / n if n else np.nan
    return {"base_rate": base, "ceiling": (min(pos, m) / m) / base if pos else np.nan}


def sampling_check(row) -> tuple[float, float]:
    """For an evaluable pair: the binomial probability of at most the observed
    high-burden count in the top decile if the true lift were 2, and the
    one-sided 95% Clopper-Pearson upper bound of the lift.  Units are treated as
    independent, which understates the uncertainty of clustered records."""
    n, pos, m, k = row["rows"], row["pos"], row["top_m"], row["top_pos"]
    base = pos / n
    p2 = binom.cdf(k, m, min(1.0, LIFT_LEVEL * base))
    ub = beta.ppf(0.95, k + 1, m - k) if k < m else 1.0
    return float(p2), float(ub / base)


# --------------------------------------------------------------------------
# PART 1: diagnostics
# --------------------------------------------------------------------------
def unit_frame(panel: pd.DataFrame) -> pd.DataFrame:
    """Keys, features and current-quarter record counts only; no outcome column."""
    cols = (["university", "building", SYS, "period_q", "year", "upm_count", "ppm_count"] +
            [c for c in FEATURES["all"] if c not in ("upm_count", "ppm_count", SYS, "year")])
    u = panel[list(dict.fromkeys(cols))].copy()
    assert not (set(OUTCOME_COLS) & set(u.columns)), "an outcome column entered the diagnostics"
    u[SYS] = u[SYS].astype(str)
    return u


def wo_frame(panel_wo: pd.DataFrame) -> pd.DataFrame:
    w = panel_wo[["university", C.COL_BUILDING, SYS, "period_q", "year", "is_upm", "is_ppm",
                  C.COL_LABORHOURS]].copy()
    w["building"] = w["university"] + "/" + w[C.COL_BUILDING].astype(str)
    w[SYS] = w[SYS].astype(str)
    w["year"] = (w["period_q"] // 4).astype(int)
    w["hours"] = w[C.COL_LABORHOURS]
    return w.drop(columns=[C.COL_LABORHOURS, C.COL_BUILDING])


def descriptive(T: pd.DataFrame, TW: pd.DataFrame) -> dict:
    n = len(T)
    upm, ppm = float(T.upm_count.sum()), float(T.ppm_count.sum())
    return {"n_units": n, "n_buildings": int(T.building.nunique()),
            "n_quarters": int(T.period_q.nunique()),
            "n_building_systems": int(T[["building", SYS]].drop_duplicates().shape[0]),
            "react_per_unit": upm / n if n else np.nan, "prev_per_unit": ppm / n if n else np.nan,
            "ppm_share": ppm / (upm + ppm) if (upm + ppm) else np.nan,
            "empty_react4_share": float((T.upm_cnt_w4 == 0).mean()) if n else np.nan,
            "labor_pos_share": float((TW.hours.fillna(0) > 0).mean()) if len(TW) else np.nan}


def contrast(t: dict, p: dict) -> dict:
    return {"react_dev": abslogratio(t["react_per_unit"], p["react_per_unit"]),
            "prev_dev": abslogratio(t["prev_per_unit"], p["prev_per_unit"]),
            "ppm_dev": absdiff(t["ppm_share"], p["ppm_share"]),
            "empty_react4_dev": absdiff(t["empty_react4_share"], p["empty_react4_share"]),
            "labor_pos_dev": absdiff(t["labor_pos_share"], p["labor_pos_share"])}


def _xframe(df: pd.DataFrame, feats: list[str], cats) -> pd.DataFrame:
    X = df[feats].copy()
    X[SYS] = pd.Categorical(X[SYS], categories=cats)
    return X


def shift_classifier(T: pd.DataFrame, P: pd.DataFrame, feats: list[str], cats, n_cap: int,
                     seed: int) -> dict:
    """Cross-validated AUC of target-versus-peer classification on balanced
    samples, folds stratified by class and grouped by building."""
    rng = np.random.default_rng(seed)
    n = int(min(len(T), len(P), n_cap))
    ti = np.sort(rng.choice(len(T), n, replace=False))
    pi = np.sort(rng.choice(len(P), n, replace=False))
    D = pd.concat([T.iloc[ti], P.iloc[pi]], ignore_index=True)
    y = np.r_[np.ones(n, int), np.zeros(n, int)]
    X = _xframe(D, feats, cats)
    oof, fold = np.full(len(D), np.nan), np.full(len(D), -1)
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    fold_auc = []
    for k, (tr, te) in enumerate(cv.split(X, y, D["building"].to_numpy())):
        m = xgb.XGBClassifier(random_state=seed, **XGB)
        m.fit(X.iloc[tr], y[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
        fold[te] = k
        if 0 < y[te].sum() < len(te):
            fold_auc.append(float(roc_auc_score(y[te], oof[te])))
    assert np.isfinite(oof).all()
    return {"auc": float(roc_auc_score(y, oof)), "fold_auc": fold_auc, "n_per_class": n,
            "oof": pd.DataFrame({SYS: D[SYS].to_numpy(), "y": y, "score": oof, "fold": fold})}


def per_system_auc(oof: pd.DataFrame) -> dict:
    out = {}
    for s, g in oof.groupby(SYS, observed=True):
        nt, npr = int(g.y.sum()), int((1 - g.y).sum())
        if nt < MIN_PAIR_AUC_UNITS or npr < MIN_PAIR_AUC_UNITS:
            out[s] = {"auc": np.nan, "fold_min": np.nan, "fold_max": np.nan, "n_t": nt, "n_p": npr}
            continue
        fa = []
        for _, gg in g.groupby("fold"):
            if gg.y.sum() >= MIN_FOLD_AUC_UNITS and (1 - gg.y).sum() >= MIN_FOLD_AUC_UNITS:
                fa.append(float(roc_auc_score(gg.y, gg.score)))
        out[s] = {"auc": float(roc_auc_score(g.y, g.score)),
                  "fold_min": min(fa) if fa else np.nan, "fold_max": max(fa) if fa else np.nan,
                  "n_t": nt, "n_p": npr}
    return out


def diagnostics(units: pd.DataFrame, wos: pd.DataFrame, campuses: list[str],
                window: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    spec = WINDOWS[window]
    uy, wy = units["year"].to_numpy(), wos["year"].to_numpy()
    cats = sorted(units[SYS].unique())
    n_match = min(int((spec["target"](uy) & (units.university == c).to_numpy()).sum())
                  for c in campuses)
    crow, prow, aucs = [], [], {}
    for ci, c in enumerate(campuses):
        tm = (units.university == c).to_numpy() & spec["target"](uy)
        pm = (units.university != c).to_numpy() & spec["peer"](uy)
        twm = (wos.university == c).to_numpy() & spec["target"](wy)
        pwm = (wos.university != c).to_numpy() & spec["peer"](wy)
        T, P, TW, PW = units[tm], units[pm], wos[twm], wos[pwm]
        assert not (set(T.university) & set(P.university))
        t, p = descriptive(T, TW), descriptive(P, PW)
        mix_t = T[SYS].value_counts(normalize=True)
        mix_p = P[SYS].value_counts(normalize=True)
        allsys = sorted(set(mix_t.index) | set(mix_p.index))
        tvd = 0.5 * float(np.abs(mix_t.reindex(allsys, fill_value=0) -
                                 mix_p.reindex(allsys, fill_value=0)).sum())
        seed = SEED + 101 * ci + (0 if window == "heldout" else 7)
        res = {fs: shift_classifier(T, P, FEATURES[fs], cats, CAP_PER_CLASS, seed)
               for fs in FEATURES}
        size = [shift_classifier(T, P, FEATURES["record"], cats, n_match, seed + 1000 + d)["auc"]
                for d in range(N_SIZE_DRAWS)]
        psa = per_system_auc(res["record"]["oof"])
        aucs[c] = {fs: {"auc": r4(res[fs]["auc"]), "fold_min": r4(min(res[fs]["fold_auc"])),
                        "fold_max": r4(max(res[fs]["fold_auc"])), "n_per_class": res[fs]["n_per_class"]}
                   for fs in FEATURES}
        aucs[c]["record_size_matched"] = {"n_per_class": n_match, "draws": [r4(x) for x in size],
                                          "median": r4(np.median(size))}
        crow.append({"window": window, "campus": c,
                     **{k: v for k, v in t.items()},
                     **{f"peer_{k}": v for k, v in p.items() if k not in ("n_quarters",)},
                     **contrast(t, p), "system_mix_tvd": tvd,
                     "auc_record": res["record"]["auc"],
                     "auc_record_fold_min": min(res["record"]["fold_auc"]),
                     "auc_record_fold_max": max(res["record"]["fold_auc"]),
                     "auc_record_size_matched": float(np.median(size)),
                     "auc_record_size_matched_min": float(min(size)),
                     "auc_record_size_matched_max": float(max(size)),
                     "auc_all": res["all"]["auc"],
                     "auc_all_fold_min": min(res["all"]["fold_auc"]),
                     "auc_all_fold_max": max(res["all"]["fold_auc"]),
                     "n_per_class_auc": res["record"]["n_per_class"]})
        # pair level
        tg, pg = T.groupby(SYS), P.groupby(SYS)
        twg, pwg = TW.groupby(SYS), PW.groupby(SYS)
        for s in sorted(T[SYS].unique()):
            Ts = tg.get_group(s)
            Ps = pg.get_group(s) if s in pg.groups else P.iloc[0:0]
            TWs = twg.get_group(s) if s in twg.groups else TW.iloc[0:0]
            PWs = pwg.get_group(s) if s in pwg.groups else PW.iloc[0:0]
            ts, ps = descriptive(Ts, TWs), descriptive(Ps, PWs)
            a = psa.get(s, {"auc": np.nan, "fold_min": np.nan, "fold_max": np.nan, "n_t": 0, "n_p": 0})
            prow.append({"window": window, "campus": c, "system": s,
                         **{k: v for k, v in ts.items() if k != "n_building_systems"},
                         "n_building_systems": ts["n_building_systems"],
                         **{f"peer_{k}": v for k, v in ps.items()
                            if k in ("n_units", "react_per_unit", "prev_per_unit", "ppm_share",
                                     "empty_react4_share", "labor_pos_share")},
                         **contrast(ts, ps),
                         "system_share_target": float(mix_t.get(s, 0.0)),
                         "system_share_peers": float(mix_p.get(s, 0.0)),
                         "system_share_dev": abslogratio(float(mix_t.get(s, 0.0)),
                                                         float(mix_p.get(s, 0.0))),
                         "auc_record": a["auc"], "auc_record_fold_min": a["fold_min"],
                         "auc_record_fold_max": a["fold_max"],
                         "auc_n_target": a["n_t"], "auc_n_peer": a["n_p"]})
        print(f"[71] {window} campus {c}: n={len(T)}, auc_record={res['record']['auc']:.3f}, "
              f"size-matched={np.median(size):.3f}, auc_all={res['all']['auc']:.3f}", flush=True)
    return pd.DataFrame(crow), pd.DataFrame(prow), aucs


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------
_PERMS: dict = {}


def exact_spearman(x, y) -> dict:
    """Spearman coefficient with its exact two-sided permutation p-value."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 4 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return {"rho": None, "p_exact": None, "n": int(n)}
    rx, ry = rankdata(x), rankdata(y)
    if n not in _PERMS:
        _PERMS[n] = np.array(list(itertools.permutations(range(n))), dtype=np.int8)
    P = _PERMS[n]
    xc = rx - rx.mean()
    yp = ry[P] - ry.mean()
    rho_p = (yp @ xc) / (np.sqrt((xc ** 2).sum()) * np.sqrt(((ry - ry.mean()) ** 2).sum()))
    rho = float(np.corrcoef(rx, ry)[0, 1])
    p = float(np.mean(np.abs(rho_p) >= abs(rho) - 1e-12))
    return {"rho": round(rho, 4), "p_exact": round(p, 4), "n": int(n)}


def _within_perm_index(strata: np.ndarray, n_perm: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = np.tile(np.arange(len(strata)), (n_perm, 1))
    for s in np.unique(strata):
        cols = np.flatnonzero(strata == s)
        if len(cols) > 1:
            idx[:, cols] = rng.permuted(np.tile(cols, (n_perm, 1)), axis=1)
    return idx


def within_system_assoc(x, y, short, strata, seed: int) -> dict:
    """Pooled Spearman of a pair diagnostic with the pair lift, and the AUC of
    the diagnostic for the pairs short of two-fold, each with a two-sided
    p-value from permutations of the outcome within building system."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    short = np.asarray(short, bool)
    strata = np.asarray(strata)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, short, strata = x[ok], y[ok], short[ok], strata[ok]
    n = len(x)
    out = {"n": int(n), "n_short": int(short.sum())}
    if n < 6 or np.ptp(x) == 0:
        return {**out, "rho": None, "p_within_system": None, "auc_short": None,
                "p_auc_within_system": None}
    idx = _within_perm_index(strata, N_PERM, seed)
    rx, ry = rankdata(x), rankdata(y)
    xc = rx - rx.mean(); yc = ry - ry.mean()
    den = np.sqrt((xc ** 2).sum() * (yc ** 2).sum())
    rho = float(xc @ yc / den)
    rho_p = (yc[idx] @ xc) / den
    out["rho"] = round(rho, 4)
    out["p_within_system"] = round(float((1 + np.sum(np.abs(rho_p) >= abs(rho) - 1e-12)) /
                                         (1 + N_PERM)), 4)
    ns = int(short.sum())
    if 0 < ns < n:
        r = rankdata(x)
        scale = ns * (n - ns)
        a = float((r[short].sum() - ns * (ns + 1) / 2) / scale)
        ap = ((short[idx] * r).sum(axis=1) - ns * (ns + 1) / 2) / scale
        out["auc_short"] = round(a, 4)
        out["p_auc_within_system"] = round(float((1 + np.sum(np.abs(ap - 0.5) >= abs(a - 0.5) - 1e-12))
                                                 / (1 + N_PERM)), 4)
    else:
        out["auc_short"] = out["p_auc_within_system"] = None
    return out


# risky direction of each diagnostic: +1 when a larger value means less like the peers
CAMPUS_DIAG = {"n_units": -1, "n_buildings": -1, "n_quarters": -1, "n_building_systems": -1,
               "react_per_unit": 0, "prev_per_unit": 0, "ppm_share": 0, "empty_react4_share": 0,
               "labor_pos_share": 0, "react_dev": 1, "prev_dev": 1, "ppm_dev": 1,
               "empty_react4_dev": 1, "labor_pos_dev": 1, "system_mix_tvd": 1,
               "auc_record": 1, "auc_record_size_matched": 1, "auc_all": 1}
PAIR_DIAG = {"n_units": -1, "n_buildings": -1, "n_quarters": -1, "react_per_unit": 0,
             "prev_per_unit": 0, "ppm_share": 0, "empty_react4_share": 0, "labor_pos_share": 0,
             "react_dev": 1, "prev_dev": 1, "ppm_dev": 1, "empty_react4_dev": 1,
             "labor_pos_dev": 1, "system_share_dev": 1, "auc_record": 1}


def campus_relations(cdf: pd.DataFrame) -> dict:
    outcomes = ["outcome_campus_lift", "outcome_lift_share_of_ceiling", "outcome_share_2x",
                "outcome_share_2x_supported", "outcome_median_pair_lift"]
    rel = {}
    for d in CAMPUS_DIAG:
        rel[d] = {o: exact_spearman(cdf[d], cdf[o]) for o in outcomes}
    return rel


def extremeness(cdf: pd.DataFrame, target: str = "7") -> dict:
    """Rank of one campus on each diagnostic (1 = least like the peers in the
    risky direction; for diagnostics without a direction, the rank from the
    smallest value), with the values of the two nearest campuses."""
    out = {}
    for d, sgn in CAMPUS_DIAG.items():
        v = cdf.set_index("campus")[d].astype(float)
        key = -v if sgn == 1 else v
        order = key.sort_values(kind="mergesort")
        rank = int(list(order.index).index(target)) + 1
        out[d] = {"value": r4(v[target]), "rank_of_9": rank,
                  "direction": {1: "larger is less like the peers", -1: "smaller is less like the peers",
                                0: "rank from the smallest value"}[sgn],
                  "most_extreme_campus": order.index[0], "most_extreme_value": r4(v[order.index[0]]),
                  "second_campus": order.index[1], "second_value": r4(v[order.index[1]])}
    return out


def rule_counts(withhold: np.ndarray, ev: pd.DataFrame) -> dict:
    short = (ev.lift < LIFT_LEVEL).to_numpy()
    sup = ev.supported.to_numpy()
    c7 = (ev.campus == "7").to_numpy()
    w = np.asarray(withhold, bool)
    return {"short_withheld": int((w & short).sum()), "short_total": int(short.sum()),
            "held_withheld": int((w & ~short).sum()), "held_total": int((~short).sum()),
            "supported_short_withheld": int((w & short & sup).sum()),
            "supported_short_total": int((short & sup).sum()),
            "supported_held_withheld": int((w & ~short & sup).sum()),
            "supported_held_total": int((~short & sup).sum()),
            "campus7_pairs_withheld": int((w & c7).sum()), "campus7_pairs_total": int(c7.sum())}


RULES = (
    [("campus units below", "campus_n_units", lt) for lt in (2000, 5000, 10000, 20000)] +
    [("campus quarters below", "campus_n_quarters", lt) for lt in (8, 12, 16, 20)] +
    [("campus buildings below", "campus_n_buildings", lt) for lt in (50, 100, 200)] +
    [("campus record-feature AUC above", "campus_auc_record", gt) for gt in (0.90, 0.95, 0.98)] +
    [("campus size-matched AUC above", "campus_auc_record_size_matched", gt) for gt in (0.90, 0.95, 0.98)] +
    [("campus system-mix distance above", "campus_system_mix_tvd", gt) for gt in (0.10, 0.20, 0.30)] +
    [("pair units below", "n_units", lt) for lt in (200, 300, 500, 1000)] +
    [("pair buildings below", "n_buildings", lt) for lt in (10, 20, 30)] +
    [("pair quarters below", "n_quarters", lt) for lt in (8, 12, 16)] +
    [("pair record-feature AUC above", "auc_record", gt) for gt in (0.80, 0.90, 0.95)] +
    [("pair reactive-density deviation above", "react_dev", gt) for gt in (math.log(1.5), math.log(2.0))]
)


def eligibility(ev: pd.DataFrame) -> dict:
    out = []
    for name, col, level in RULES:
        v = ev[col].to_numpy(float)
        below = "below" in name
        w = (v < level) if below else (v > level)
        w = np.where(np.isfinite(v), w, False)
        out.append({"rule": f"withhold when {name} {round(level, 4)}", "column": col,
                    "level": round(level, 4), **rule_counts(w, ev),
                    "n_pairs_without_value": int((~np.isfinite(v)).sum())})
    # for each pair diagnostic, the least strict level that withholds every short pair
    least = {}
    short = (ev.lift < LIFT_LEVEL).to_numpy()
    for d, sgn in {**{f"campus_{k}": v for k, v in CAMPUS_DIAG.items() if v != 0},
                   **{k: v for k, v in PAIR_DIAG.items() if v != 0}}.items():
        v = ev[d].to_numpy(float)
        if not np.isfinite(v[short]).all():
            least[d] = {"note": "a short pair has no value"}
            continue
        if sgn == -1:
            lev = float(np.max(v[short]))
            w = v <= lev
        else:
            lev = float(np.min(v[short]))
            w = v >= lev
        w = np.where(np.isfinite(v), w, False)
        least[d] = {"level": r4(lev), "withhold": "at or below" if sgn == -1 else "at or above",
                    **rule_counts(w, ev)}
    return {"rules": out, "withhold_every_short_pair": least,
            "note": ("levels are round values set for illustration; none is fitted, and no level "
                     "was chosen to separate the pairs that fell short")}


def campus_block_exact(ev: pd.DataFrame, col: str) -> dict:
    """Pooled Spearman of a campus-level diagnostic with the pair lift and the
    AUC of the diagnostic for the pairs short of two-fold, with exact two-sided
    p-values from all orderings of the diagnostic over the campuses: the pairs of
    a campus move together, so a campus effect on lift is not read as an
    association with the diagnostic."""
    camps = sorted(ev.campus.unique())
    k = len(camps)
    v = np.array([float(ev.loc[ev.campus == c, col].iloc[0]) for c in camps])
    nvec = np.array([int((ev.campus == c).sum()) for c in camps])
    ry = rankdata(ev.lift.to_numpy(float))
    yc = ry - ry.mean()
    S = np.array([yc[(ev.campus == c).to_numpy()].sum() for c in camps])
    short = (ev.lift < LIFT_LEVEL).to_numpy()
    nsv = np.array([int(short[(ev.campus == c).to_numpy()].sum()) for c in camps])
    ns, n = int(short.sum()), int(len(ev))
    if k not in _PERMS:
        _PERMS[k] = np.array(list(itertools.permutations(range(k))), dtype=np.int8)
    A = v[_PERMS[k]]
    L = np.zeros(A.shape); T = np.zeros(A.shape)
    for j in range(k):
        L += nvec[j] * (A[:, [j]] < A)
        T += nvec[j] * (A[:, [j]] == A)
    pooled = L + (T + 1) / 2
    m = (n + 1) / 2
    cov = (pooled - m) @ S
    varx = ((pooled - m) ** 2) @ nvec
    rho_p = cov / np.sqrt(varx * (yc ** 2).sum())
    ident = np.all(_PERMS[k] == np.arange(k), axis=1)
    rho = float(rho_p[ident][0])
    out = {"n_pairs": n, "n_campuses": k, "rho": round(rho, 4),
           "p_campus_block_exact": round(float(np.mean(np.abs(rho_p) >= abs(rho) - 1e-12)), 4)}
    if 0 < ns < n:
        auc_p = (pooled @ nsv - ns * (ns + 1) / 2) / (ns * (n - ns))
        a = float(auc_p[ident][0])
        out["auc_short"] = round(a, 4)
        out["p_auc_campus_block_exact"] = round(float(np.mean(np.abs(auc_p - 0.5) >=
                                                              abs(a - 0.5) - 1e-12)), 4)
    return out


def pair_relations(ev: pd.DataFrame, seed: int) -> dict:
    short = (ev.lift < LIFT_LEVEL).to_numpy()
    rel = {}
    for i, d in enumerate(PAIR_DIAG):
        a = within_system_assoc(ev[d], ev.lift, short, ev.system.to_numpy(), seed + i)
        b = within_system_assoc(ev[d], ev.lift, short, ev.campus.to_numpy(), seed + 50 + i)
        rel[d] = {**a, "p_within_campus": b["p_within_system"],
                  "p_auc_within_campus": b["p_auc_within_system"]}
    for i, d in enumerate(CAMPUS_DIAG):
        col = f"campus_{d}"
        a = within_system_assoc(ev[col], ev.lift, short, ev.system.to_numpy(), seed + 100 + i)
        b = campus_block_exact(ev, col)
        assert a["rho"] is None or abs(a["rho"] - b["rho"]) < 1e-3, (col, a["rho"], b["rho"])
        rel[col] = {**b, "p_within_system": a["p_within_system"],
                    "p_auc_within_system": a["p_auc_within_system"]}
    return rel


def short_pair_positions(ev: pd.DataFrame) -> list:
    """For each pair short of two-fold, its percentile among the evaluable pairs
    on each directional pair diagnostic (100 = least like the peers)."""
    out = []
    for r in ev[ev.lift < LIFT_LEVEL].itertuples():
        pos = {}
        for d, sgn in PAIR_DIAG.items():
            if sgn == 0:
                continue
            v = ev[d].to_numpy(float)
            x = getattr(r, d)
            if not np.isfinite(x):
                pos[d] = None
                continue
            vv = v[np.isfinite(v)]
            pos[d] = round(100 * float(np.mean(vv >= x) if sgn == -1 else np.mean(vv <= x)), 1)
        out.append({"campus": r.campus, "system": r.system, "lift": r4(r.lift),
                    "supported": bool(r.supported), "n_units": int(r.n_units),
                    "auc_record": r4(r.auc_record),
                    "p_count_if_lift_2": r4(r.p_short_if_lift2),
                    "lift_upper95": r4(r.lift_upper95),
                    "percentile_least_like_peers": pos})
    return out


# --------------------------------------------------------------------------
# PART 2: labour-recording audit
# --------------------------------------------------------------------------
def labor_audit(wos: pd.DataFrame, panel: pd.DataFrame, single: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    r = wos[wos.is_upm].copy()
    pos = r[r.hours > 0]
    p99 = pos.groupby(SYS)["hours"].quantile(EXTREME_PCTL)
    r["extreme"] = r.hours > r[SYS].map(p99).astype(float).fillna(np.inf)
    thr, gthr = FE.severity_thresholds(panel, target="labour", pctl=75)
    pan = panel[["university", SYS]].copy()
    pan[SYS] = pan[SYS].astype(str)
    pan["y_panel"] = FE.apply_severity(panel, thr, gthr, "labour")
    peer_y = single[["university", "system", "y"]].rename(columns={"system": SYS})

    def block(g, gp, gy):
        h = g.hours
        hp = h[h > 0]
        q = hp.quantile([0.5, 0.75, 0.9, 0.99]) if len(hp) else pd.Series([np.nan] * 4,
                                                                           index=[0.5, 0.75, 0.9, 0.99])
        return {"n_reactive_wo": int(len(g)),
                "share_hours_positive": float((h > 0).mean()) if len(g) else np.nan,
                "share_hours_zero": float((h == 0).mean()) if len(g) else np.nan,
                "share_hours_missing": float(h.isna().mean()) if len(g) else np.nan,
                "share_hours_negative": float((h < 0).mean()) if len(g) else np.nan,
                "median_hours_positive": float(q[0.5]), "p75_hours_positive": float(q[0.75]),
                "p90_hours_positive": float(q[0.9]), "p99_hours_positive": float(q[0.99]),
                "mean_hours_positive": float(hp.mean()) if len(hp) else np.nan,
                "share_extreme": float(g.extreme.mean()) if len(g) else np.nan,
                "n_units": int(len(gp)),
                "base_rate_panel_thresholds": float(gp.y_panel.mean()) if len(gp) else np.nan,
                "base_rate_peer_thresholds": float(gy.y.mean()) if len(gy) else np.nan}

    rows = []
    for c in sorted(r.university.unique()):
        g, gp, gy = r[r.university == c], pan[pan.university == c], peer_y[peer_y.university == c]
        rows.append({"campus": c, "system": "all", **block(g, gp, gy)})
        for s in sorted(set(gp[SYS]) | set(g[SYS])):
            rows.append({"campus": c, "system": s,
                         **block(g[g[SYS] == s], gp[gp[SYS] == s], gy[gy[SYS] == s])})
    nine = {"campus": "all", "system": "all", **block(r, pan, peer_y)}
    rows.append(nine)
    for s in sorted(r[SYS].unique()):
        rows.append({"campus": "all", "system": s,
                     **block(r[r[SYS] == s], pan[pan[SYS] == s], peer_y[peer_y[SYS] == s])})
    a = pd.DataFrame(rows)
    a["extreme_threshold_hours"] = a.system.map(p99).astype(float)

    spread = {}
    pr = a[(a.campus != "all") & (a.system != "all") & (a.n_reactive_wo >= AUDIT_MIN_WO)]
    for s, g in pr.groupby("system"):
        if len(g) < 2:
            continue
        def rng_(col):
            v = g[col].astype(float)
            return {"min": r4(v.min()), "median": r4(v.median()), "max": r4(v.max()),
                    "max_over_min": r4(v.max() / v.min()) if v.min() > 0 else None,
                    "campus_min": g.loc[v.idxmin(), "campus"], "campus_max": g.loc[v.idxmax(), "campus"]}
        spread[s] = {"n_campuses": int(len(g)),
                     "median_hours_positive": rng_("median_hours_positive"),
                     "p90_hours_positive": rng_("p90_hours_positive"),
                     "share_hours_positive": rng_("share_hours_positive"),
                     "share_hours_zero": rng_("share_hours_zero"),
                     "base_rate_panel_thresholds": rng_("base_rate_panel_thresholds"),
                     "base_rate_peer_thresholds": rng_("base_rate_peer_thresholds")}
    return a, {"extreme_threshold_by_system": {k: r4(v) for k, v in p99.items()},
               "nine_campuses": {k: r4(v) if isinstance(v, float) else v for k, v in nine.items()},
               "system_spread": spread,
               "spread_rule": f"pairs with at least {AUDIT_MIN_WO} reactive work orders"}


# --------------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    panel = lh._panel()
    panel["university"] = panel[C.COL_UNIV].astype(str)
    panel["building"] = panel["university"] + "/" + panel[C.COL_BUILDING].astype(str)
    panel["system"] = panel[SYS].astype(str)

    # ---- invariants: verdicts and stored outcomes ----
    single, pairs = cc.load_single(), cc.load_pairs()
    campuses = sorted(single.university.unique())
    outer, pooled, routing = fvm.final_pipeline(single, pairs, campuses)
    fvm.check_reference(outer, pooled, routing)
    stored_fv = pd.read_csv(C.TABLES / "fold_gate_verdicts.csv", dtype={"campus": str})
    for r in stored_fv.itertuples():
        S = outer[r.campus][r.system]["S"]
        assert round(S, 2) == r.S and bool(S >= S_CUT) == bool(r.certified), (r.campus, r.system)
    held, camp_lift_h = heldout_outcomes(single, campuses)
    fwd, camp_lift_f = forward_outcomes(campuses)
    print(f"[71] verdicts of the held-out maps reproduced; held-out and forward lifts equal the "
          f"stored tables ({time.time() - t0:.0f} s)", flush=True)

    # ---- PART 3: record flow ----
    raw = load_records()
    steps, flow, panel_wo = record_flow(raw, panel)
    del raw
    pd.DataFrame(steps).to_csv(C.TABLES / "record_flow.csv", index=False)
    mism = [c for c in flow["table_checks"] if not c["match"]]
    print(f"[71] record flow: {[(s['step'], s['work_orders_remaining']) for s in steps]}; "
          f"mismatches: {mism}", flush=True)

    # ---- PART 1: diagnostics ----
    units = unit_frame(panel)
    wos = wo_frame(panel_wo)
    by_c, by_p, auc_detail, rel_c, rel_p, elig, c7, shorts = [], [], {}, {}, {}, {}, {}, {}
    for w, (out_pairs, out_camp) in {"heldout": (held, camp_lift_h),
                                     "forward": (fwd, camp_lift_f)}.items():
        cdf, pdf, aucs = diagnostics(units, wos, campuses, w)
        auc_detail[w] = aucs
        op = out_pairs.copy()
        pdf = pdf.merge(op, on=["campus", "system"], how="outer", validate="one_to_one")
        pdf["window"] = w
        # the pair's units in the window are the units its outcome is read on
        both = pdf.n_units.notna() & (pdf.rows > 0)
        assert (pdf.loc[both, "n_units"].astype(int) == pdf.loc[both, "rows"]).all(), w
        assert (pdf.n_units.isna() == (pdf.rows.fillna(0) == 0)).all(), w
        evm = pdf.evaluable.fillna(False).astype(bool)
        assert int(evm.sum()) == {"heldout": 97, "forward": 91}[w]
        sc = pdf[evm].apply(sampling_check, axis=1)
        pdf["p_short_if_lift2"] = np.nan
        pdf["lift_upper95"] = np.nan
        pdf["lift_ceiling"] = [lift_ceiling(int(r), int(q))["ceiling"] if e else np.nan
                               for r, q, e in zip(pdf.rows.fillna(0), pdf.pos.fillna(0), evm)]
        pdf.loc[evm, "p_short_if_lift2"] = [x[0] for x in sc]
        pdf.loc[evm, "lift_upper95"] = [x[1] for x in sc]
        # campus outcomes
        for col in ("supported", "evaluable", "has_verdict"):
            pdf[col] = pdf[col].fillna(False).astype(bool)
        # the pairs of the stored check: those with a verdict in the held-out
        # check (95 evaluable), every evaluable pair in the forward test (91)
        pdf["in_check"] = pdf.evaluable & (pdf.has_verdict if w == "heldout" else True)
        assert int(pdf.in_check.sum()) == {"heldout": 95, "forward": 91}[w]
        oc = []
        for c in campuses:
            e = pdf[(pdf.campus == c) & pdf.in_check]
            es = e[e.supported]
            oc.append({"campus": c, "outcome_campus_lift": out_camp[c]["lift"],
                       "outcome_base_rate": out_camp[c]["base_rate"],
                       "outcome_lift_ceiling": out_camp[c]["ceiling"],
                       "outcome_lift_share_of_ceiling": out_camp[c]["lift"] / out_camp[c]["ceiling"],
                       "outcome_n_evaluable": int(len(e)),
                       "outcome_n_2x": int((e.lift >= LIFT_LEVEL).sum()),
                       "outcome_share_2x": float((e.lift >= LIFT_LEVEL).mean()) if len(e) else np.nan,
                       "outcome_n_evaluable_supported": int(len(es)),
                       "outcome_n_2x_supported": int((es.lift >= LIFT_LEVEL).sum()),
                       "outcome_share_2x_supported": float((es.lift >= LIFT_LEVEL).mean()) if len(es) else np.nan,
                       "outcome_median_pair_lift": float(e.lift.median()) if len(e) else np.nan})
        cdf = cdf.merge(pd.DataFrame(oc), on="campus", validate="one_to_one")
        camp_cols = {f"campus_{k}": cdf.set_index("campus")[k] for k in CAMPUS_DIAG}
        for k, v in camp_cols.items():
            pdf[k] = pdf.campus.map(v)
        ev = pdf[pdf.in_check].copy()
        rel_c[w] = campus_relations(cdf)
        rel_p[w] = pair_relations(ev, SEED + (0 if w == "heldout" else 500))
        elig[w] = eligibility(ev)
        c7[w] = extremeness(cdf, "7")
        shorts[w] = short_pair_positions(ev)
        by_c.append(cdf)
        by_p.append(pdf)

    cdf_all = pd.concat(by_c, ignore_index=True)
    pdf_all = pd.concat(by_p, ignore_index=True)
    cdf_all.round(6).to_csv(C.TABLES / "peer_comparability_by_campus.csv", index=False)
    lead = ["window", "campus", "system", "has_verdict", "S", "supported", "evaluable", "in_check", "rows",
            "pos", "lift", "top_m", "top_pos", "lift_ceiling", "p_short_if_lift2", "lift_upper95"]
    pdf_all = pdf_all[lead + [c for c in pdf_all.columns if c not in lead]]
    pdf_all.sort_values(["window", "campus", "system"]).round(6).to_csv(
        C.TABLES / "peer_comparability_by_pair.csv", index=False)

    # ---- PART 2: labour audit ----
    audit, audit_sum = labor_audit(wos, panel, single)
    audit.round(6).to_csv(C.TABLES / "labor_recording_audit.csv", index=False)
    ca = audit[(audit.campus != "all") & (audit.system == "all")].set_index("campus")
    ch = cdf_all[cdf_all.window == "heldout"].set_index("campus")
    pa = audit[(audit.campus != "all") & (audit.system != "all")]
    ph = pdf_all[(pdf_all.window == "heldout") & pdf_all.in_check][["campus", "system", "lift"]]
    pj = ph.merge(pa, on=["campus", "system"], how="left")
    order = ca.share_hours_positive.sort_values()
    audit_rel = {
        "campus_share_hours_positive_vs": {
            o: exact_spearman(ca.loc[campuses, "share_hours_positive"], ch.loc[campuses, o])
            for o in ("outcome_campus_lift", "outcome_share_2x", "outcome_median_pair_lift")},
        "campus_median_hours_vs_campus_lift": exact_spearman(ca.loc[campuses, "median_hours_positive"],
                                                             ch.loc[campuses, "outcome_campus_lift"]),
        "lowest_completeness_campuses": [
            {"campus": c, "share_hours_positive": r4(order[c]),
             "heldout_campus_lift": r4(ch.loc[c, "outcome_campus_lift"]),
             "heldout_lift_rank_from_lowest": int(ch.outcome_campus_lift.rank(method="min")[c]),
             "heldout_share_2x": r4(ch.loc[c, "outcome_share_2x"])} for c in order.index[:3]],
        "pair_share_hours_positive_vs_lift": within_system_assoc(
            pj.share_hours_positive, pj.lift, (pj.lift < LIFT_LEVEL).to_numpy(),
            pj.system.to_numpy(), SEED + 900),
        "pair_share_hours_positive_vs_lift_within_campus": within_system_assoc(
            pj.share_hours_positive, pj.lift, (pj.lift < LIFT_LEVEL).to_numpy(),
            pj.campus.to_numpy(), SEED + 903),
        "pair_median_hours_vs_lift": within_system_assoc(
            pj.median_hours_positive, pj.lift, (pj.lift < LIFT_LEVEL).to_numpy(),
            pj.system.to_numpy(), SEED + 901),
        "pair_base_rate_vs_lift": within_system_assoc(
            pj.base_rate_peer_thresholds, pj.lift, (pj.lift < LIFT_LEVEL).to_numpy(),
            pj.system.to_numpy(), SEED + 902),
    }

    def camp_table(w):
        d = cdf_all[cdf_all.window == w].set_index("campus")
        keep = ["n_units", "n_buildings", "n_quarters", "n_building_systems", "react_per_unit",
                "peer_react_per_unit", "prev_per_unit", "peer_prev_per_unit", "ppm_share",
                "peer_ppm_share", "empty_react4_share", "peer_empty_react4_share",
                "labor_pos_share", "peer_labor_pos_share", "system_mix_tvd", "auc_record",
                "auc_record_fold_min", "auc_record_fold_max", "auc_record_size_matched",
                "auc_all", "outcome_campus_lift", "outcome_base_rate", "outcome_lift_ceiling",
                "outcome_lift_share_of_ceiling", "outcome_n_evaluable", "outcome_n_2x",
                "outcome_share_2x", "outcome_n_evaluable_supported", "outcome_n_2x_supported"]
        return {c: {k: (r4(d.loc[c, k]) if isinstance(d.loc[c, k], (float, np.floating)) else
                        int(d.loc[c, k])) for k in keep} for c in campuses}

    out = {
        "design": ("Outcome-free comparability of each target campus with its eight peers "
                   "(size, record density, preventive share, empty reactive record, labour "
                   "recorded, building-system mix, feature shift), related to the held-out check "
                   "and the forward test of the peer-trained ranking; a labour-recording audit by "
                   "campus and building system; and the record flow from the raw file to the "
                   "analytical panel."),
        "definitions": {
            "unit": "one building system of one building in one quarter (a panel row)",
            "windows": {w: v["label"] for w, v in WINDOWS.items()},
            "feature_shift": (f"gradient-boosted classifier (XGBoost, CPU, {XGB['n_estimators']} "
                              f"trees of depth {XGB['max_depth']}, learning rate "
                              f"{XGB['learning_rate']}), target units against an equal number of "
                              f"peer units (each class at most {CAP_PER_CLASS}), {N_FOLDS} folds "
                              "stratified by class and grouped by building; AUC pooled over the "
                              "out-of-fold scores, with its range over folds; per building system "
                              f"from the same out-of-fold scores when each class has at least "
                              f"{MIN_PAIR_AUC_UNITS} units; size-matched: every campus subsampled to "
                              f"the smallest campus's unit count, median of {N_SIZE_DRAWS} draws"),
            "feature_sets": {k: v for k, v in FEATURES.items()},
            "omitted_features": ("the two reactive-premium features, constants of the building "
                                 "system computed on the training campuses"),
            "empty_react4_share": ("share of units with no reactive work order in the four quarters "
                                   "up to and including the anchor quarter"),
            "labor_pos_share": "share of the unit's work orders (both categories) with positive labour hours",
            "deviation": ("|log(target / peers)| for record densities and the system share; "
                          "|target - peers| for shares"),
            "system_mix_tvd": "total-variation distance between the distributions of units over building systems",
            "heldout_outcome": ("top-decile lift of f^(-c) on the units of c (campus) or of s on c "
                                "(pair), over the whole window; a pair is evaluable with at least "
                                f"{MIN_POS} high-burden units and {MIN_ROWS} units; it reaches two-fold "
                                "when the lift is at least 2"),
            "forward_outcome": ("top-decile lift in 2017 to 2019 of f_h^(-c), trained on the other "
                                "campuses' units anchored in 2016 or earlier; same evaluability"),
            "lift_ceiling": ("largest attainable top-decile lift, min(high-burden units, m) / m "
                             "divided by the base rate, with m the size of the top decile; it is "
                             "below 10 when the base rate exceeds 0.10"),
            "supported": ("evidence score of at least 2/3: the verdict of the held-out map "
                          "(heldout) or the peer verdict formed from pre-2017 records (forward)"),
            "campus_rank_correlation": "Spearman coefficient over the nine campuses; exact two-sided p-value from all 9! orderings",
            "pair_association": (f"Spearman coefficient over the evaluable pairs and the AUC of the "
                                 f"diagnostic for the pairs short of two-fold (the probability that a "
                                 f"short pair has the larger value, so below 0.5 means short pairs "
                                 f"have smaller values), each with two-sided p-values from {N_PERM} "
                                 f"permutations of the outcome within building system and within "
                                 f"campus; for campus-level diagnostics, the exact campus-block "
                                 f"p-value from all orderings of the diagnostic over the campuses "
                                 f"replaces the within-campus one, because the within-system "
                                 f"permutation treats the pairs of one campus as independent"),
            "pairs_in_check": ("held-out check: the 95 evaluable pairs with a verdict (two further "
                               "evaluable pairs, 11-F20 and 5-G30, have no verdict and are left "
                               "out); forward test: the 91 evaluable pairs of the stored table"),
            "sampling_check": ("for an evaluable pair, the binomial probability of at most the observed "
                               "number of high-burden units in the top decile if the true lift were 2, "
                               "and the one-sided 95% Clopper-Pearson upper bound of the lift; units "
                               "are treated as independent, so both understate the uncertainty of "
                               "clustered records"),
            "labour_extreme": f"reactive work order above the system's {int(EXTREME_PCTL * 100)}th percentile of positive hours over the nine campuses",
            "base_rate_panel_thresholds": "high-burden share with the system-specific 75th percentile fitted on the whole panel (the Table 1 definition)",
            "base_rate_peer_thresholds": "high-burden share with the threshold fitted on the other eight campuses (the labels of the held-out check)",
        },
        "checks": {
            "heldout_map_verdicts_reproduced": True,
            "fold_gate_verdicts_csv_reproduced": True,
            "heldout_pair_lifts_equal_stored": {"n_pairs": 141, "n_evaluable": 95},
            "heldout_campus_lifts_equal_double_holdout_fits": True,
            "forward_pair_lifts_equal_stored": {"n_pairs": 159, "n_evaluable": 91},
            "forward_campus_lifts_equal_stored": True,
            "work_orders_by_unit_equal_panel_counts_and_labour": True,
            "diagnostic_frames_hold_no_outcome_column": True,
            "pair_units_equal_outcome_rows": True,
        },
        "record_flow": flow,
        "comparability": {
            w: {"by_campus": camp_table(w), "auc_detail": auc_detail[w],
                "campus_rank_correlations": rel_c[w], "campus7_position": c7[w],
                "pair_associations": rel_p[w], "short_pairs": shorts[w],
                "eligibility": elig[w]} for w in WINDOWS},
        "labor_audit": {**audit_sum, "relation_to_heldout_lift": audit_rel},
        "runtime_seconds": None,
    }
    out["runtime_seconds"] = round(time.time() - t0, 1)
    json.dump(fvm.jsonable(out), open(C.METRICS / "peer_comparability.json", "w"), indent=2)

    # ---- console ----
    pd.set_option("display.width", 250, "display.max_columns", 40)
    for w in WINDOWS:
        d = cdf_all[cdf_all.window == w]
        print(f"\n=== {w} ===")
        print(d[["campus", "n_units", "n_quarters", "react_per_unit", "ppm_share",
                 "empty_react4_share", "labor_pos_share", "system_mix_tvd", "auc_record",
                 "auc_record_size_matched", "auc_all", "outcome_campus_lift",
                 "outcome_share_2x"]].round(3).to_string(index=False))
        print("campus rank correlations (outcome_share_2x):",
              {k: v["outcome_share_2x"] for k, v in rel_c[w].items()})
        print("short pairs:", [(s["campus"], s["system"], s["lift"], s["n_units"], s["auc_record"])
                               for s in shorts[w]])
    print(f"\n[71] done in {out['runtime_seconds']} s")


if __name__ == "__main__":
    main()
