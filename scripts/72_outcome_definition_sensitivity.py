#!/usr/bin/env python
"""
72_outcome_definition_sensitivity.py -- the verdicts of the evidence gating
under alternative definitions of the outcome.

The study defines a high-burden quarter from reactive labour hours: a unit (one
building system of one building in one quarter t) is high-burden when its
reactive labour in quarter t+1 exceeds the system-specific 75th percentile of
positive next-quarter reactive labour, fitted on the training campuses of each
model.  Labour hours are recorded with different completeness across campuses,
so this script re-runs the pipeline from the outcome definition to the verdicts
with the outcome defined in other ways, and compares the verdicts.

OUTCOME DEFINITIONS
  V0  the study's definition, passed through the code path of this script.
  V1  high burden from the number of reactive work orders in quarter t+1, with
      the same rule: above the system-specific 75th percentile of positive
      next-quarter counts, fitted on the training campuses of each model.
  V2  reactive labour as in V0, with the 75th percentile fitted per campus and
      system on that campus's own rows.  A peer campus read by the gates is
      labelled with its own threshold, as is every training campus.  In the
      held-out check the target campus is labelled with the peer-pooled
      threshold of V0 (fitted on the training campuses of f^(-c)), so that no
      outcome of the target campus defines its own label; the lift under the
      target's own per-campus threshold is reported beside it.
  V3  occurrence: any reactive work order in quarter t+1.  The base rate is
      about four times that of V0, so a top-decile lift of 2 is close to the
      largest attainable value; V3 is a boundary case and its levels are not
      comparable with those of V0.
  The extreme event of the prior-record-trace gate (a quarter above the
  system-specific 90th percentile of the burden quantity, with the target campus
  excluded from the threshold) follows the outcome definition: reactive labour
  with a pooled threshold (V0), reactive work-order count with a pooled
  threshold (V1, V3), reactive labour with a per-campus threshold (V2).  Each
  variant is also scored with the trace gate held at the study's definition.

WHAT IS HELD FIXED
  Panel rows, the campus splits of scripts/44_double_holdout_fits.py (9 fits
  with one campus left out, 36 fits with two left out), the features of layer
  M1 (assembled by fmscreen.features.assemble_Xy as in every stored fit), the
  boosted model of fmscreen.models.make_gbdt with its hyperparameters and seed,
  and the gates, the aggregation rule and the routing of
  scripts/45_certification_corrected.py at the main setting (double nesting,
  risk percentile within campus and quarter, transfer gate capped below six
  evaluable campuses).  The features include labour quantities of past quarters
  and the training-campus labour premium; none of them is derived from the
  outcome definition, so they are the stored features in every variant.  Only
  the label changes, and with it the class weight the model sets from the label
  counts.

FITS.  Every fit runs on the CPU.  The stored predictions were fitted on a GPU,
and the two devices do not give bit-identical boosted models.  V0 is therefore
fitted on the CPU as well and serves as the reference for V1 to V3; V0 is also
compared with the stored verdicts.

INVARIANTS (the runner aborts on a mismatch)
  - The resolved model configuration differs from that of the stored fits only
    in the compute device and the thread count.
  - Every fit scores the same rows in the same order as the stored prediction
    file of its split, and its V0 labels equal the stored labels and those of
    fmscreen.features.make_labels.
  - The gate code path of this script, applied to the stored predictions,
    reproduces the stored pooled map, the 141 verdicts of the held-out maps,
    their gate scores and the route counts exactly; its prior-record-trace
    table at the study's definition equals that of script 45 for every map.
  - A repeated fit returns identical scores.
  - The stored panel, predictions and result files read here are unchanged at
    the end of the run (SHA-256).

Usage:  python scripts/72_outcome_definition_sensitivity.py [--variants V0,V1,V2,V3] [--reuse]
        --reuse loads prediction files of an earlier run of this script instead
        of refitting; every check above is still applied to them.

Outputs -> data/processed/pred_outcome_sensitivity/{variant}/single/{c}.parquet
           data/processed/pred_outcome_sensitivity/{variant}/pair/{a}__{b}.parquet
           results/metrics/outcome_definition_sensitivity.json
           results/tables/outcome_definition_sensitivity.csv  (one row per
               variant and verdict: the 141 campus-system pairs of the held-out
               maps and the systems of the pooled map)
"""
from __future__ import annotations
import os

# CPU only, with the thread caps set before numpy or fmscreen is imported
N_THREADS = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["FMSCREEN_DEVICE"] = "cpu"
os.environ["FMSCREEN_THREADS"] = N_THREADS
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = N_THREADS

import sys, json, time, argparse, hashlib, itertools, importlib.util
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fmscreen import config as C
from fmscreen import features as FE
from fmscreen import models as M
from fmscreen import metrics as MET


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cc = _load("cert_corrected", "45_certification_corrected.py")
sr = _load("fold_verdicts", "56_fold_verdicts_and_simple_rule.py")

LAYER, MODEL, TARGET, PCTL = "M1", "gbdt", "severity_labour", 75
TRACE_PCTL = 0.90
STORED_DEVICE, STORED_THREADS = "cuda", 4      # settings of scripts/44 (GPU, FMSCREEN_THREADS=4)
RCOL = "R_quarter"
GATES, HARD = cc.GATES, cc.HARD
S_CUT = cc.S_CUT
LEVEL = sr.LEVEL
CTX_MIN_POS, CTX_MIN_ROWS = sr.CTX_MIN_POS, sr.CTX_MIN_ROWS
PROVISIONAL = list(sr.PROVISIONAL)
UNIV, SYSTEM = C.COL_UNIV, C.COL_SYSTEM
KEYS = ["university", "building", "system", "period_q"]

VARIANTS = {
    "V0": {"name": "study definition",
           "outcome": "upm_labour_next", "rule": "percentile", "scope": "pooled",
           "trace_quantity": "upm_labour", "trace_scope": "pooled",
           "definition": ("reactive labour hours in quarter t+1 above the system-specific "
                          "75th percentile of positive next-quarter reactive labour, fitted "
                          "on the training campuses of each model")},
    "V1": {"name": "work-order count",
           "outcome": "upm_count_next", "rule": "percentile", "scope": "pooled",
           "trace_quantity": "upm_count", "trace_scope": "pooled",
           "definition": ("number of reactive work orders in quarter t+1 above the "
                          "system-specific 75th percentile of positive next-quarter counts, "
                          "fitted on the training campuses of each model")},
    "V2": {"name": "per-campus labour threshold",
           "outcome": "upm_labour_next", "rule": "percentile", "scope": "campus",
           "trace_quantity": "upm_labour", "trace_scope": "campus",
           "definition": ("reactive labour hours in quarter t+1 above the 75th percentile of "
                          "positive next-quarter reactive labour of the same campus and "
                          "system; in the held-out check the target campus is labelled with "
                          "the peer-pooled threshold of V0")},
    "V3": {"name": "occurrence",
           "outcome": "occurrence_next", "rule": "occurrence", "scope": None,
           "trace_quantity": "upm_count", "trace_scope": "pooled",
           "definition": "at least one reactive work order in quarter t+1"},
}
STUDY_TRACE = ("upm_labour", "pooled")

PRED_ROOT = C.DATA_PROCESSED / "pred_outcome_sensitivity"
STORED_SINGLE = C.DATA_PROCESSED / "pred_single"
STORED_PAIR = C.DATA_PROCESSED / "pred_pair"
OUT_JSON = C.METRICS / "outcome_definition_sensitivity.json"
OUT_CSV = C.TABLES / "outcome_definition_sensitivity.csv"
STORED_INPUTS = [C.DATA_PROCESSED / "panel_quarter.parquet",
                 C.METRICS / "cert_corrected.json",
                 C.METRICS / "double_holdout_fits.json",
                 C.METRICS / "fold_verdicts_and_simple_rule.json",
                 C.TABLES / "fold_gate_verdicts.csv",
                 C.TABLES / "gate_evidence_corrected.csv",
                 C.TABLES / "burden_by_system.csv"]


def rel(p: Path) -> str:
    return str(Path(p).resolve().relative_to(ROOT))


def stored_input_files() -> list[Path]:
    return (STORED_INPUTS + sorted(STORED_SINGLE.glob("*.parquet"))
            + sorted(STORED_PAIR.glob("*.parquet")))


def checksums() -> dict:
    out = {}
    for f in stored_input_files():
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        out[rel(f)] = h.hexdigest()
    return out


_PANEL: pd.DataFrame | None = None


def panel() -> pd.DataFrame:
    global _PANEL
    if _PANEL is None:
        _PANEL = pd.read_parquet(C.DATA_PROCESSED / "panel_quarter.parquet")
    return _PANEL


# --------------------------------------------------------------------------
# outcome definitions
# --------------------------------------------------------------------------
def pooled_thresholds(df: pd.DataFrame, col: str, q: float) -> tuple[pd.Series, float]:
    """System-specific percentile of the positive values of `col` over the rows
    of `df` (the training campuses), with the all-system percentile as fallback."""
    pos = df[df[col] > 0]
    thr = pos.groupby(SYSTEM, observed=True)[col].quantile(q)
    glob = float(pos[col].quantile(q)) if len(pos) else 0.0
    return thr, glob


def apply_pooled(df: pd.DataFrame, col: str, thr: pd.Series, glob: float) -> np.ndarray:
    t = df[SYSTEM].map(thr).astype("float64").fillna(glob).to_numpy()
    v = df[col].to_numpy()
    return ((v > t) & (v > 0)).astype(int)


def campus_thresholds(df: pd.DataFrame, col: str, q: float) -> pd.Series:
    """Percentile of the positive values of `col` per campus and system, each
    fitted on that campus's own rows."""
    pos = df[df[col] > 0]
    return pos.groupby([UNIV, SYSTEM], observed=True)[col].quantile(q)


def apply_campus(df: pd.DataFrame, col: str, thr: pd.Series) -> np.ndarray:
    idx = pd.MultiIndex.from_arrays([df[UNIV].astype(str), df[SYSTEM].astype(str)])
    thr = thr.copy()
    thr.index = pd.MultiIndex.from_arrays([thr.index.get_level_values(0).astype(str),
                                           thr.index.get_level_values(1).astype(str)])
    t = thr.reindex(idx).to_numpy(dtype="float64")
    t = np.where(np.isnan(t), np.inf, t)      # no positive row: no high-burden quarter
    v = df[col].to_numpy()
    return ((v > t) & (v > 0)).astype(int)


def make_labels(spec: dict, train: pd.DataFrame, test: pd.DataFrame):
    """Labels of one split: (training rows, scored rows as read by the gates,
    scored rows as read by the held-out check)."""
    col, q = spec["outcome"], PCTL / 100.0
    if spec["rule"] == "occurrence":
        assert np.array_equal(train[col].to_numpy() == 1, train["upm_count_next"].to_numpy() > 0)
        ytr = train[col].to_numpy().astype(int)
        yte = test[col].to_numpy().astype(int)
        return ytr, yte, yte
    if spec["scope"] == "pooled":
        thr, glob = pooled_thresholds(train, col, q)
        yte = apply_pooled(test, col, thr, glob)
        return apply_pooled(train, col, thr, glob), yte, yte
    # per-campus thresholds: every campus is labelled from its own rows; the
    # held-out check reads the scored campus with the peer-pooled threshold
    ytr = apply_campus(train, col, campus_thresholds(train, col, q))
    ygate = apply_campus(test, col, campus_thresholds(test, col, q))
    thr, glob = pooled_thresholds(train, col, q)
    return ytr, ygate, apply_pooled(test, col, thr, glob)


# --------------------------------------------------------------------------
# model configuration
# --------------------------------------------------------------------------
def _same(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    return a == b


def resolved_configuration() -> dict:
    """Parameters of the boosted model as resolved here and as resolved under the
    settings of the stored fits; the runner asserts that only the compute device
    and the thread count differ."""
    num, cat = FE.LAYERS[LAYER]
    cur = M.MODEL_FACTORY[MODEL](num, cat).get_params()
    saved = (M._DEVICE, M._THREADS)
    M._DEVICE, M._THREADS = STORED_DEVICE, STORED_THREADS
    try:
        ref = M.MODEL_FACTORY[MODEL](num, cat).get_params()
    finally:
        M._DEVICE, M._THREADS = saved
    diff = {k: {"stored_fits": ref.get(k), "this_run": cur.get(k)}
            for k in sorted(set(ref) | set(cur)) if not _same(ref.get(k), cur.get(k))}
    assert set(diff) <= {"device", "n_jobs"}, f"model configuration differs: {diff}"
    assert cur["device"] == "cpu", cur["device"]
    keep = {k: v for k, v in cur.items() if isinstance(v, (int, float, str, bool)) or v is None}
    return {"layer": LAYER, "model": MODEL, "pctl": PCTL, "seed": C.RANDOM_SEED,
            "numeric_features": list(num), "categorical_features": list(cat),
            "n_features": len(num) + len(cat),
            "parameters_this_run": keep, "differences_from_stored_fits": diff,
            "note": ("scale_pos_weight is set at fit time from the label counts "
                     "(n_negative / n_positive) and so follows the outcome definition")}


# --------------------------------------------------------------------------
# fits
# --------------------------------------------------------------------------
def stored_file(held_out: tuple[str, ...]) -> Path:
    return (STORED_SINGLE / f"{held_out[0]}.parquet" if len(held_out) == 1
            else STORED_PAIR / f"{held_out[0]}__{held_out[1]}.parquet")


def pred_file(variant: str, held_out: tuple[str, ...]) -> Path:
    return (PRED_ROOT / variant / "single" / f"{held_out[0]}.parquet" if len(held_out) == 1
            else PRED_ROOT / variant / "pair" / f"{held_out[0]}__{held_out[1]}.parquet")


def fit(variant: str, held_out: tuple[str, ...]) -> tuple[pd.DataFrame, dict]:
    """Train on every campus outside `held_out` under the outcome definition of
    `variant`; score every row inside it."""
    t0 = time.time()
    p = panel()
    ucol = p[UNIV].astype(str).to_numpy()
    keep = ~np.isin(ucol, list(held_out))
    train, test = p[keep], p[~keep]
    assert not (set(train[UNIV].astype(str)) & set(held_out)), "held-out campus in training"

    # features exactly as in every stored fit; the labels are replaced below
    art = FE.assemble_Xy(train, test, LAYER, TARGET, PCTL)
    ytr, ygate, yheld = make_labels(VARIANTS[variant], train, test)
    if variant == "V0":
        assert np.array_equal(ytr, art["y_train"]) and np.array_equal(ygate, art["y_test"]), \
            f"V0 labels differ from fmscreen.features.make_labels for {held_out}"
    if len(ygate) == 0 or ytr.sum() == 0 or ytr.sum() == len(ytr) or ygate.sum() == 0:
        raise RuntimeError(f"degenerate split for {variant} held_out={held_out}")

    model = M.MODEL_FACTORY[MODEL](art["num_cols"], art["cat_cols"])
    scores = M.fit_predict(model, art["X_train"], ytr, art["X_test"])
    df = pd.DataFrame({
        "held_out": "+".join(held_out), "n_held_out": len(held_out),
        "university": np.asarray(art["test_univ"]).astype(str),
        "building": np.asarray(art["test_building"]).astype(str),
        "system": np.asarray(art["test_system"]).astype(str),
        "period_q": test["period_q"].to_numpy(),
        "y": np.asarray(ygate).astype(np.int8),
        "y_heldout": np.asarray(yheld).astype(np.int8),
        "score": np.asarray(scores, dtype=np.float32),
    })
    rec = {"held_out": "+".join(held_out), "n_train": int(len(train)), "n_test": int(len(test)),
           "train_positives": int(ytr.sum()), "train_base_rate": round(float(ytr.mean()), 5),
           "scale_pos_weight": round(float(model.get_params()["scale_pos_weight"]), 5),
           "secs": round(time.time() - t0, 1)}
    return df, rec


def check_against_stored(variant: str, held_out: tuple[str, ...], df: pd.DataFrame) -> dict:
    """Same rows in the same order as the stored split; for V0 the same labels."""
    st = pd.read_parquet(stored_file(held_out))
    assert len(st) == len(df), f"{variant} {held_out}: {len(df)} rows, stored {len(st)}"
    for k in KEYS:
        assert np.array_equal(st[k].astype(str).to_numpy() if k != "period_q" else st[k].to_numpy(),
                              df[k].astype(str).to_numpy() if k != "period_q" else df[k].to_numpy()), \
            f"{variant} {held_out}: column {k} differs from the stored split"
    out = {}
    if variant == "V0":
        assert np.array_equal(st["y"].to_numpy(), df["y"].to_numpy()), \
            f"V0 labels differ from the stored labels for {held_out}"
        a, b = st["score"].to_numpy(np.float64), df["score"].to_numpy(np.float64)
        out = {"max_abs_score_diff_vs_stored": round(float(np.max(np.abs(a - b))), 4),
               "spearman_score_vs_stored": round(float(spearmanr(a, b).correlation), 4)}
    return out


def run_fits(variant: str, campuses: list[str], reuse: bool) -> dict:
    splits = [(c,) for c in campuses] + list(itertools.combinations(campuses, 2))
    for d in (PRED_ROOT / variant / "single", PRED_ROOT / variant / "pair"):
        d.mkdir(parents=True, exist_ok=True)
    recfile = PRED_ROOT / variant / "fits.json"
    old = json.load(open(recfile)) if (reuse and recfile.exists()) else {}
    recs, t0 = {}, time.time()
    for i, ho in enumerate(splits):
        f = pred_file(variant, ho)
        key = "+".join(ho)
        if reuse and f.exists() and key in old:
            df, rec = pd.read_parquet(f), dict(old[key])
        else:
            df, rec = fit(variant, ho)
            df.to_parquet(f, index=False)
        rec.update(check_against_stored(variant, ho, df))
        recs[key] = rec
        if (i + 1) % 9 == 0 or i + 1 == len(splits):
            print(f"[72] {variant}: {i + 1}/{len(splits)} fits ({time.time() - t0:.0f} s)",
                  flush=True)
    json.dump(recs, open(recfile, "w"), indent=1)
    return recs


def load_predictions(variant: str, campuses: list[str]):
    single = pd.concat([pd.read_parquet(pred_file(variant, (c,))) for c in campuses],
                       ignore_index=True)
    single = cc._add_ranks(single)
    pairs = {frozenset(ab): pd.read_parquet(pred_file(variant, ab))
             for ab in itertools.combinations(campuses, 2)}
    return single, pairs


# --------------------------------------------------------------------------
# prior-record-trace table under an outcome definition
# --------------------------------------------------------------------------
_TRACE: dict = {}


def trace_table(quantity: str, scope: str, exclude: str | None) -> pd.DataFrame:
    """Per-campus, per-system extreme events and their no-antecedent share, as in
    script 45's prior_trace_table, with the extreme event defined on `quantity`
    (reactive labour or reactive work-order count of the quarter) and its 90th
    percentile fitted over the peer campuses together (pooled) or per campus."""
    key = (quantity, scope, exclude)
    if key in _TRACE:
        return _TRACE[key]
    p = panel()
    if exclude is not None:
        p = p[p[UNIV].astype(str) != str(exclude)]
    p = p.copy()
    if scope == "pooled":
        thr = p[p[quantity] > 0].groupby(SYSTEM, observed=True)[quantity].quantile(TRACE_PCTL)
        t = p[SYSTEM].map(thr).astype(float).fillna(np.inf).to_numpy()
    else:
        thr = campus_thresholds(p, quantity, TRACE_PCTL)
        idx = pd.MultiIndex.from_arrays([p[UNIV].astype(str), p[SYSTEM].astype(str)])
        thr.index = pd.MultiIndex.from_arrays([thr.index.get_level_values(0).astype(str),
                                               thr.index.get_level_values(1).astype(str)])
        t = thr.reindex(idx).to_numpy(dtype="float64")
        t = np.where(np.isnan(t), np.inf, t)
    p["p90"] = (p[quantity].to_numpy() > t).astype(int)
    p = p.sort_values(cc.CELL + ["period_q"]).reset_index(drop=True)
    g = p.groupby(cc.CELL, observed=True)["upm_count"]
    p["prior_total"] = g.shift(1).rolling(2, min_periods=2).sum() + \
                       g.shift(3).rolling(2, min_periods=2).sum()
    p["has_history"] = (p["period_q"] - 4 >= p["cell_first_q"]).astype(int)
    ev = p[(p["p90"] == 1) & (p["has_history"] == 1)].copy()
    ev["zero_prior"] = (ev["prior_total"] == 0).astype(int)
    tt = (ev.groupby([UNIV, SYSTEM], observed=True)
            .agg(n_events=("zero_prior", "size"), n_zero=("zero_prior", "sum")).reset_index())
    tt.columns = ["campus", "system", "n_events", "n_zero"]
    tt["campus"] = tt["campus"].astype(str)
    _TRACE[key] = tt
    return tt


def check_trace_tables(campuses: list[str]) -> dict:
    """At the study's definition the table equals script 45's for every map."""
    for ex in [None] + campuses:
        a = trace_table(*STUDY_TRACE, ex).sort_values(["campus", "system"]).reset_index(drop=True)
        b = cc.trace_for(ex).sort_values(["campus", "system"]).reset_index(drop=True)
        assert a.astype(str).equals(b.astype(str)), f"trace table differs for map {ex}"
    return {"maps_checked": 1 + len(campuses), "equal_to_script_45": True}


# --------------------------------------------------------------------------
# verdicts, held-out check, routing
# --------------------------------------------------------------------------
def verdict_maps(single: pd.DataFrame, pairs: dict, campuses: list[str],
                 trace: tuple[str, str]) -> dict:
    outer = {j: cc.gates_for(cc.dev_corpus(j, campuses, single, pairs, "double"),
                             trace_table(*trace, j), RCOL, True) for j in campuses}
    pooled = cc.gates_for(single, trace_table(*trace, None), RCOL, True)
    routing = cc.route(single, outer, campuses, RCOL)
    return {"outer": outer, "pooled": pooled, "routing": routing}


def heldout_table(single: pd.DataFrame, outer: dict, ycol: str) -> dict:
    """Top-decile lift of f^(-c) on system k in campus c, for every verdict."""
    held = {}
    for (j, s), g in single.groupby(["university", "system"], observed=True):
        if j in outer and s in outer[j]:
            y, sc = g[ycol].to_numpy(), g["score"].to_numpy()
            ev = bool(y.sum() >= CTX_MIN_POS and len(y) >= CTX_MIN_ROWS)
            held[(j, s)] = {"rows": int(len(y)), "pos": int(y.sum()), "evaluable": ev,
                            "lift": round(float(MET.lift_at_topk(y, sc, C.TOPK_BUDGET)), 3)
                            if ev else None}
    for j in outer:
        for s in outer[j]:
            held.setdefault((j, s), {"rows": 0, "pos": 0, "evaluable": False, "lift": None})
    return held


def heldout_check(outer: dict, held: dict) -> dict:
    keys = [k for k in held if held[k]["evaluable"]]
    sup = [held[k]["lift"] for k in keys if outer[k[0]][k[1]]["S"] >= S_CUT]
    nsup = [held[k]["lift"] for k in keys if outer[k[0]][k[1]]["S"] < S_CUT]
    Ss = [outer[k[0]][k[1]]["S"] for k in keys]
    lifts = [held[k]["lift"] for k in keys]
    return {"n_pairs": len(held), "n_evaluable": len(keys),
            "supported": sr.lift_summary(sup), "not_supported": sr.lift_summary(nsup),
            "accept_every_system": sr.lift_summary(lifts),
            "spearman_S_vs_heldout_lift": sr.spearman(Ss, lifts) if len(keys) > 2 else None}


def campus_lifts(single: pd.DataFrame, ycol: str) -> dict:
    out = {}
    for j, g in single.groupby("university", observed=True):
        y, sc = g[ycol].to_numpy(), g["score"].to_numpy()
        out[str(j)] = {"lift_top10": round(float(MET.lift_at_topk(y, sc, C.TOPK_BUDGET)), 3),
                       "base_rate": round(float(y.mean()), 4), "rows": int(len(y))}
    lifts = [v["lift_top10"] for v in out.values()]
    return {"by_campus": out, "median": round(float(np.median(lifts)), 3),
            "min": round(float(min(lifts)), 3),
            "n_campuses_lift_ge_2": int(sum(x >= 2 for x in lifts))}


# --------------------------------------------------------------------------
# comparisons
# --------------------------------------------------------------------------
def verdict_dict(maps: dict) -> tuple[dict, dict]:
    pool = {s: v for s, v in maps["pooled"].items()}
    pairs = {(j, s): v for j, m in maps["outer"].items() for s, v in m.items()}
    return pool, pairs


def agreement(ref: dict, new: dict, keys: list) -> dict:
    """Verdict agreement over `keys`; a key without a verdict counts as not
    supported, as in the routing."""
    a = [bool(ref.get(k) is not None and ref[k]["S"] >= S_CUT) for k in keys]
    b = [bool(new.get(k) is not None and new[k]["S"] >= S_CUT) for k in keys]
    if not keys:
        return {"n": 0}
    out = sr.agreement(a, b)
    out["n_supported_reference"] = int(sum(a))
    out["n_supported_variant"] = int(sum(b))
    return out


def gate_changes(ref: dict, new: dict, keys: list) -> dict:
    out = {}
    both = [k for k in keys if k in ref and k in new]
    for g in GATES:
        trans = {}
        for k in both:
            a, b = ref[k]["scores"][g], new[k]["scores"][g]
            if a != b:
                t = f"{LEVEL[a]} -> {LEVEL[b]}"
                trans[t] = trans.get(t, 0) + 1
        out[g] = {"n_changed": int(sum(trans.values())), "transitions": trans}
    out["n_compared"] = len(both)
    return out


def group_of(system: str, stable: list) -> str:
    return "stable" if system in stable else ("provisional" if system in PROVISIONAL else "other")


def compare(ref_maps: dict, new_maps: dict, stable: list) -> dict:
    rp, rf = verdict_dict(ref_maps)
    npool, nf = verdict_dict(new_maps)
    pool_keys = sorted(set(rp) | set(npool))
    fold_keys = sorted(set(rf) | set(nf))
    by_group = {g: [k for k in fold_keys if group_of(k[1], stable) == g]
                for g in ("stable", "provisional", "other")}
    changes = []
    for k in fold_keys:
        a, b = rf.get(k), nf.get(k)
        sa = None if a is None else a["S"]
        sb = None if b is None else b["S"]
        ca, cb = sa is not None and sa >= S_CUT, sb is not None and sb >= S_CUT
        if ca != cb:
            gch = ([] if (a is None or b is None) else
                   [f"{g}: {LEVEL[a['scores'][g]]} -> {LEVEL[b['scores'][g]]}"
                    for g in GATES if a["scores"][g] != b["scores"][g]])
            changes.append({"campus": k[0], "system": k[1], "group": group_of(k[1], stable),
                            "S_reference": sa, "S_variant": sb,
                            "supported_reference": ca, "supported_variant": cb,
                            "gates_changed": gch})
    pool_changes = []
    for s in pool_keys:
        a, b = rp.get(s), npool.get(s)
        if a is None or b is None or a["S"] != b["S"] or a["scores"] != b["scores"]:
            pool_changes.append({
                "system": s, "S_reference": None if a is None else a["S"],
                "S_variant": None if b is None else b["S"],
                "gates_changed": ([] if (a is None or b is None) else
                                  [f"{g}: {LEVEL[a['scores'][g]]} -> {LEVEL[b['scores'][g]]}"
                                   for g in GATES if a["scores"][g] != b["scores"][g]])})
    return {
        "pooled_map": agreement(rp, npool, pool_keys),
        "pairs_all": agreement(rf, nf, fold_keys),
        "pairs_stable": agreement(rf, nf, by_group["stable"]),
        "pairs_provisional": agreement(rf, nf, by_group["provisional"]),
        "pairs_other": agreement(rf, nf, by_group["other"]),
        "pairs_without_verdict": {"reference_only": [list(k) for k in fold_keys if k not in nf],
                                  "variant_only": [list(k) for k in fold_keys if k not in rf]},
        "n_pairs_S_changed": int(sum(1 for k in fold_keys if k in rf and k in nf
                                     and abs(rf[k]["S"] - nf[k]["S"]) > 1e-12)),
        "gate_changes_pairs": gate_changes(rf, nf, fold_keys),
        "gate_changes_pooled_map": gate_changes(rp, npool, pool_keys),
        "verdict_changes_pairs": changes,
        "pooled_map_changes": pool_changes,
        "route_counts_reference": ref_maps["routing"]["route_counts"],
        "route_counts_variant": new_maps["routing"]["route_counts"],
    }


def summary(maps: dict, held: dict, single: pd.DataFrame, ycol: str) -> dict:
    pool, folds = verdict_dict(maps)
    per_sys = {}
    for (j, s), v in folds.items():
        n, k = per_sys.get(s, (0, 0))
        per_sys[s] = (n + 1, k + int(v["S"] >= S_CUT))
    r = maps["routing"]
    return {
        "pooled_map": {"n_systems_scored": len(pool),
                       "supported": sorted(s for s, v in pool.items() if v["S"] >= S_CUT),
                       "not_supported": sorted(s for s, v in pool.items() if v["S"] < S_CUT),
                       "S": {s: v["S"] for s, v in sorted(pool.items())},
                       "gate_scores": {s: {g: LEVEL[v["scores"][g]] for g in GATES}
                                       for s, v in sorted(pool.items())},
                       "n_pos": {s: v["n_pos"] for s, v in sorted(pool.items())}},
        "held_out_maps": {"n_pairs": len(folds),
                          "n_supported": int(sum(v["S"] >= S_CUT for v in folds.values())),
                          "supported_maps_by_system": {s: f"{k}/{n}" for s, (n, k)
                                                       in sorted(per_sys.items())},
                          "gate_score_counts": {g: {LEVEL[x]: int(sum(v["scores"][g] == x
                                                                      for v in folds.values()))
                                                    for x in (cc.FAIL, cc.CAUTION, cc.PASS)}
                                                for g in GATES}},
        "heldout_check": heldout_check(maps["outer"], held),
        "heldout_lift_by_campus": campus_lifts(single, ycol),
        "routing": {"route_counts": r["route_counts"], "n_routed": r["n_routed"],
                    "n_high_risk": r["n_high_risk"],
                    "share_high_risk_to_decision_task": r["share_high_risk_cleared"]},
        "base_rate_heldout_labels": round(float(single[ycol].mean()), 4),
    }


def stored_maps() -> dict:
    """The stored verdicts, in the structure of verdict_maps."""
    ref = json.load(open(C.METRICS / "cert_corrected.json"))["corrected"]
    fv = pd.read_csv(C.TABLES / "fold_gate_verdicts.csv", dtype={"campus": str})
    outer = {}
    for r in fv.itertuples():
        outer.setdefault(r.campus, {})[r.system] = {
            "S": ref["outer_S"][r.campus][r.system],
            "scores": {g: float(getattr(r, g)) for g in GATES}}
    pooled = {s: {"S": v["S"], "scores": v["scores"]} for s, v in ref["pooled"].items()}
    return {"outer": outer, "pooled": pooled, "routing": ref["routing"]}


def check_code_path(campuses: list[str]) -> dict:
    """The gate code path of this script on the stored predictions reproduces the
    stored verdicts exactly."""
    single, pairs = cc.load_single(), cc.load_pairs()
    maps = verdict_maps(single, pairs, campuses, STUDY_TRACE)
    chk = sr.check_reference(maps["outer"], maps["pooled"], maps["routing"])
    fv = pd.read_csv(C.TABLES / "fold_gate_verdicts.csv", dtype={"campus": str})
    for r in fv.itertuples():
        v = maps["outer"][r.campus][r.system]
        for g in GATES:
            assert v["scores"][g] == float(getattr(r, g)), (r.campus, r.system, g)
        assert bool(v["S"] >= S_CUT) == bool(r.certified), (r.campus, r.system)
    held = heldout_table(single, maps["outer"], "y")
    for r in fv.itertuples():
        h = held[(r.campus, r.system)]
        assert h["evaluable"] == bool(r.heldout_evaluable), (r.campus, r.system)
        if h["evaluable"]:
            assert abs(h["lift"] - r.heldout_lift) < 1e-9, (r.campus, r.system)
    chk["gate_scores_and_heldout_lifts_match_fold_gate_verdicts_csv"] = True
    return chk


def relabelled(base_single: pd.DataFrame, base_pairs: dict,
               lab_single: pd.DataFrame, lab_pairs: dict):
    """Scores of one run with the labels of another (same rows, same order)."""
    assert base_single[KEYS].astype(str).equals(lab_single[KEYS].astype(str))
    s = base_single.copy()
    s["y"], s["y_heldout"] = lab_single["y"].to_numpy(), lab_single["y_heldout"].to_numpy()
    p = {}
    for k, df in base_pairs.items():
        assert df[KEYS].astype(str).equals(lab_pairs[k][KEYS].astype(str))
        d = df.copy()
        d["y"], d["y_heldout"] = lab_pairs[k]["y"].to_numpy(), lab_pairs[k]["y_heldout"].to_numpy()
        p[k] = d
    return s, p


# --------------------------------------------------------------------------
# labour-based against count-based outcome
# --------------------------------------------------------------------------
def labour_vs_count(s0: pd.DataFrame, s1: pd.DataFrame) -> dict:
    assert s0[KEYS].astype(str).equals(s1[KEYS].astype(str))
    # the single-fit predictions hold each campus's panel rows in panel order
    p = panel()
    parts = [p[p[UNIV].astype(str) == c] for c in s0["university"].astype(str).unique()]
    q = pd.concat(parts, ignore_index=True)
    assert np.array_equal(q["period_q"].to_numpy(), s0["period_q"].to_numpy())
    assert np.array_equal(q[SYSTEM].astype(str).to_numpy(), s0["system"].astype(str).to_numpy())
    assert np.array_equal((q[UNIV].astype(str) + "/" + q[C.COL_BUILDING].astype(str)).to_numpy(),
                          s0["building"].astype(str).to_numpy())
    lab, cnt = q["upm_labour_next"].to_numpy(), q["upm_count_next"].to_numpy()
    d = pd.DataFrame({"campus": s0["university"].astype(str).to_numpy(), "lab": lab, "cnt": cnt,
                      "y0": s0["y_heldout"].to_numpy(), "y1": s1["y_heldout"].to_numpy()})

    def block(x: pd.DataFrame) -> dict:
        act = x[x.cnt > 0]
        a, b = x.y0.to_numpy().astype(bool), x.y1.to_numpy().astype(bool)
        both = int((a & b).sum())
        return {"units": int(len(x)),
                "spearman_all_units": round(float(spearmanr(x.lab, x.cnt).correlation), 4),
                "units_with_reactive_work_order": int(len(act)),
                "spearman_units_with_reactive_work_order":
                    round(float(spearmanr(act.lab, act.cnt).correlation), 4) if len(act) > 2 else None,
                "share_with_work_order_and_zero_labour":
                    round(float(((x.cnt > 0) & (x.lab <= 0)).mean()), 4),
                "base_rate_labour": round(float(a.mean()), 4),
                "base_rate_count": round(float(b.mean()), 4),
                "label_agreement": round(float((a == b).mean()), 4),
                "label_kappa": sr.cohen_kappa(a, b),
                "high_burden_by_both": both,
                "share_of_labour_high_also_count_high": round(both / a.sum(), 4) if a.sum() else None,
                "share_of_count_high_also_labour_high": round(both / b.sum(), 4) if b.sum() else None}

    return {"definition": ("units of the nine campuses, each labelled as in the held-out check "
                           "(thresholds fitted on the other eight campuses); Spearman "
                           "correlation of next-quarter reactive labour hours with the "
                           "next-quarter number of reactive work orders, and agreement of "
                           "the V0 and V1 high-burden labels"),
            "all_campuses": block(d),
            "by_campus": {c: block(g) for c, g in d.groupby("campus")}}


# --------------------------------------------------------------------------
def rows_for(variant: str, maps: dict, held: dict, held_own: dict | None,
             maps_trace_study: dict, ref_cpu: dict, ref_st: dict, stable: list,
             name: dict) -> list[dict]:
    rows = []
    vp, vf = verdict_dict(maps)
    cp, cf = verdict_dict(ref_cpu)
    sp, sf = verdict_dict(ref_st)
    tp, tf = verdict_dict(maps_trace_study)

    def sup(d, k):
        return None if d.get(k) is None else bool(d[k]["S"] >= S_CUT)

    def one(mapname, campus, s, v, kc, dc, ds, dt, h, ho):
        r = {"variant": variant, "map": mapname, "campus": campus, "system": s,
             "system_desc": name.get(s, s), "system_group": group_of(s, stable),
             "has_verdict": v is not None}
        for g in GATES:
            r[g] = None if v is None else v["scores"][g]
        for f in ("n_campuses", "n_pos", "pooled_lift", "frac_2x", "top_band", "rho",
                  "zero_prior_share"):
            r[f] = None if v is None else v.get(f)
        r["S"] = None if v is None else round(v["S"], 4)
        r["supported"] = bool(v is not None and v["S"] >= S_CUT)
        r["S_V0_cpu"] = None if dc.get(kc) is None else round(dc[kc]["S"], 4)
        r["supported_V0_cpu"] = bool(sup(dc, kc))
        r["supported_stored"] = bool(sup(ds, kc))
        r["verdict_changed_vs_V0_cpu"] = r["supported"] != r["supported_V0_cpu"]
        r["verdict_changed_vs_stored"] = r["supported"] != r["supported_stored"]
        r["gates_changed_vs_V0_cpu"] = ("" if (v is None or dc.get(kc) is None) else
                                         ";".join(g for g in GATES
                                                  if v["scores"][g] != dc[kc]["scores"][g]))
        r["supported_trace_at_study_definition"] = bool(sup(dt, kc))
        r["heldout_rows"] = None if h is None else h["rows"]
        r["heldout_pos"] = None if h is None else h["pos"]
        r["heldout_evaluable"] = None if h is None else h["evaluable"]
        r["heldout_lift"] = None if h is None else h["lift"]
        r["heldout_lift_own_campus_threshold"] = None if ho is None else ho["lift"]
        return r

    for k in sorted(set(vf) | set(cf) | set(sf)):
        rows.append(one("held_out", k[0], k[1], vf.get(k), k, cf, sf, tf, held.get(k),
                        None if held_own is None else held_own.get(k)))
    for s in sorted(set(vp) | set(cp) | set(sp)):
        rows.append(one("pooled", "all", s, vp.get(s), s, cp, sp, tp, None, None))
    return rows


def jsonable(o):
    if isinstance(o, dict):
        return {(k if isinstance(k, str) else "|".join(map(str, k)) if isinstance(k, tuple)
                 else str(k)): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return None if np.isnan(o) else float(o)
    return o


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="V0,V1,V2,V3")
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    assert variants[0] == "V0" and set(variants) <= set(VARIANTS), variants
    t_start = time.time()

    sums_before = checksums()
    for out in (OUT_JSON, OUT_CSV, PRED_ROOT):
        assert not str(out.resolve()).startswith(str(STORED_SINGLE.resolve())) and \
               not str(out.resolve()).startswith(str(STORED_PAIR.resolve()))

    campuses = sorted(panel()[UNIV].astype(str).unique())
    assert len(campuses) == 9 and len(panel()) == 514975
    config = resolved_configuration()
    print(f"[72] configuration: differs from the stored fits only in "
          f"{sorted(config['differences_from_stored_fits'])}", flush=True)

    trace_check = check_trace_tables(campuses)
    code_path = check_code_path(campuses)
    print(f"[72] code path reproduces the stored verdicts: {code_path}", flush=True)

    ref_st = stored_maps()
    fv = pd.read_csv(C.TABLES / "fold_gate_verdicts.csv", dtype={"campus": str})
    per = fv.groupby("system").certified.agg(["size", "sum"])
    stable = sorted(per.index[per["sum"] == per["size"]])
    assert len(stable) == 9 and not (set(stable) & set(PROVISIONAL)), stable
    burden = pd.read_csv(C.TABLES / "burden_by_system.csv")
    name = {k: str(v).strip() for k, v in burden[["SystemCode", "SystemDescription"]].values}

    fits, preds, maps, maps_trace, held, held_own = {}, {}, {}, {}, {}, {}
    for v in variants:
        spec = VARIANTS[v]
        fits[v] = run_fits(v, campuses, args.reuse)
        single, pairs = load_predictions(v, campuses)
        preds[v] = (single, pairs)
        trace = (spec["trace_quantity"], spec["trace_scope"])
        maps[v] = verdict_maps(single, pairs, campuses, trace)
        maps_trace[v] = (maps[v] if trace == STUDY_TRACE else
                         verdict_maps(single, pairs, campuses, STUDY_TRACE))
        held[v] = heldout_table(single, maps[v]["outer"], "y_heldout")
        if spec["scope"] == "campus":
            held_own[v] = heldout_table(single, maps[v]["outer"], "y")
        print(f"[72] {v}: pooled map supports "
              f"{sum(x['S'] >= S_CUT for x in maps[v]['pooled'].values())}/{len(maps[v]['pooled'])}, "
              f"held-out maps {sum(x['S'] >= S_CUT for m in maps[v]['outer'].values() for x in m.values())}"
              f"/{sum(len(m) for m in maps[v]['outer'].values())}, routes "
              f"{maps[v]['routing']['route_counts']}", flush=True)

    # a repeated fit returns identical scores
    rep, _ = fit("V0", (campuses[0],))
    assert np.array_equal(rep["score"].to_numpy(), preds["V0"][0].loc[
        preds["V0"][0].university == campuses[0], "score"].to_numpy()), "repeated fit differs"

    # V2 reads the held-out campus with the peer-pooled threshold, i.e. the V0 label
    if "V2" in variants:
        assert np.array_equal(preds["V2"][0]["y_heldout"].to_numpy(),
                              preds["V0"][0]["y"].to_numpy()), "V2 held-out labels differ from V0"

    # ---------------- results per variant ----------------
    results, rows = {}, []
    s0, p0 = preds["V0"]
    for v in variants:
        single, pairs = preds[v]
        res = {"definition": VARIANTS[v]["definition"], "name": VARIANTS[v]["name"],
               "prior_record_trace_extreme_event": {
                   "quantity": {"upm_labour": "reactive labour hours",
                                "upm_count": "reactive work-order count"}[VARIANTS[v]["trace_quantity"]],
                   "threshold": {"pooled": "90th percentile over the peer campuses",
                                 "campus": "90th percentile per campus"}[VARIANTS[v]["trace_scope"]]},
               **summary(maps[v], held[v], single, "y_heldout"),
               "vs_V0_cpu": compare(maps["V0"], maps[v], stable),
               "vs_stored": compare(ref_st, maps[v], stable)}
        if v in held_own:
            res["heldout_check_own_campus_threshold"] = heldout_check(maps[v]["outer"], held_own[v])
            res["heldout_lift_by_campus_own_campus_threshold"] = campus_lifts(single, "y")
        if v != "V0":
            tr = maps_trace[v]
            res["trace_gate_at_study_definition"] = {
                "pooled_supported": sorted(s for s, x in tr["pooled"].items() if x["S"] >= S_CUT),
                "n_supported_pairs": int(sum(x["S"] >= S_CUT for m in tr["outer"].values()
                                             for x in m.values())),
                "vs_V0_cpu": {k: val for k, val in compare(maps["V0"], tr, stable).items()
                              if k.startswith("pairs_") or k == "pooled_map"},
                "vs_this_variant": {k: val for k, val in compare(maps[v], tr, stable).items()
                                    if k.startswith("pairs_") or k == "pooled_map"},
                "route_counts": tr["routing"]["route_counts"]}
            # labels of the variant on the scores of the V0 refit: separates the
            # change of the evaluated outcome from the change of the trained model
            ls, lp = relabelled(s0, p0, single, pairs)
            lm = verdict_maps(ls, lp, campuses, (VARIANTS[v]["trace_quantity"],
                                                 VARIANTS[v]["trace_scope"]))
            lh = heldout_table(ls, lm["outer"], "y_heldout")
            cmp_v = compare(maps["V0"], lm, stable)
            res["variant_labels_on_V0_scores"] = {
                "pooled_supported": sorted(s for s, x in lm["pooled"].items() if x["S"] >= S_CUT),
                "n_supported_pairs": int(sum(x["S"] >= S_CUT for m in lm["outer"].values()
                                             for x in m.values())),
                "vs_V0_cpu": {k: val for k, val in cmp_v.items()
                              if k.startswith("pairs_") or k == "pooled_map"},
                "vs_this_variant": {k: val for k, val in compare(maps[v], lm, stable).items()
                                    if k.startswith("pairs_") or k == "pooled_map"},
                "heldout_check": heldout_check(lm["outer"], lh)}
        results[v] = res
        rows += rows_for(v, maps[v], held[v], held_own.get(v), maps_trace[v], maps["V0"],
                         ref_st, stable, name)

    # ---------------- V0 on the CPU against the stored GPU fits ----------------
    sc = [r for r in fits["V0"].values()]
    v0_vs_stored = {
        "labels_equal_stored": True, "rows_and_order_equal_stored": True,
        "max_abs_score_diff": max(r["max_abs_score_diff_vs_stored"] for r in sc),
        "min_spearman_score_vs_stored": min(r["spearman_score_vs_stored"] for r in sc),
        "median_spearman_score_vs_stored": float(np.median([r["spearman_score_vs_stored"] for r in sc])),
        "verdicts": results["V0"]["vs_stored"],
        "heldout_lift_by_campus_stored": {
            c["campus"]: round(c["stored"], 3) for c in
            json.load(open(C.METRICS / "double_holdout_fits.json"))["reproduction_check"]},
    }
    exact = (v0_vs_stored["verdicts"]["pairs_all"]["n_agree"] == v0_vs_stored["verdicts"]["pairs_all"]["n"]
             and not v0_vs_stored["verdicts"]["pooled_map_changes"]
             and v0_vs_stored["verdicts"]["n_pairs_S_changed"] == 0
             and v0_vs_stored["verdicts"]["route_counts_reference"] == v0_vs_stored["verdicts"]["route_counts_variant"])
    v0_vs_stored["reproduces_stored_exactly"] = bool(exact)

    lvc = labour_vs_count(preds["V0"][0], preds["V1"][0]) if "V1" in variants else None

    sums_after = checksums()
    assert sums_before == sums_after, "a stored input file changed during the run"

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    out = {
        "design": ("The pipeline from the outcome definition to the verdicts (45 boosted fits, "
                   "five gates, evidence score S, routing) re-run under alternative outcome "
                   "definitions, all on the CPU, with features, model settings, seeds and "
                   "campus splits of the stored fits. V0 passes the study's definition through "
                   "the same code and is the reference for V1 to V3; it is also compared with "
                   "the stored verdicts, which come from GPU fits. A pair without a verdict "
                   "under one definition counts as not supported in agreement statistics, as "
                   "in the routing."),
        "variants": {v: {k: VARIANTS[v][k] for k in ("name", "definition")} for v in VARIANTS
                     if v in variants},
        "system_groups": {"stable": stable, "provisional": PROVISIONAL,
                          "definition": ("stable: supported in all nine held-out maps of the "
                                         "stored pipeline; provisional: the four systems "
                                         "supported in some held-out maps")},
        "fit_configuration": {**config, "device": "cpu", "threads": int(N_THREADS),
                              "xgboost_version": M.xgb.__version__,
                              "fit_records": fits},
        "invariants": {
            "model_configuration_differs_only_in": sorted(config["differences_from_stored_fits"]),
            "rows_order_and_V0_labels_equal_stored_splits": True,
            "code_path_on_stored_predictions": code_path,
            "trace_table_at_study_definition": trace_check,
            "repeated_fit_identical": True,
            "stored_inputs_unchanged_sha256": True,
            "stored_inputs_checked": sorted(sums_after),
        },
        "V0_cpu_vs_stored": v0_vs_stored,
        "results": results,
        "labour_vs_count_outcome": lvc,
        "runtime_seconds": round(time.time() - t_start, 1),
    }
    json.dump(jsonable(out), open(OUT_JSON, "w"), indent=2)

    # ---------------- console ----------------
    print(f"\n[72] V0 (CPU) against the stored verdicts: exact={exact}; pairs "
          f"{v0_vs_stored['verdicts']['pairs_all']}; pooled {v0_vs_stored['verdicts']['pooled_map']}")
    for c in v0_vs_stored["verdicts"]["verdict_changes_pairs"]:
        print(f"    {c}")
    for v in variants:
        r = results[v]
        print(f"\n=== {v} ({r['name']}) ===")
        print(f"  pooled supported: {r['pooled_map']['supported']}")
        print(f"  held-out maps: {r['held_out_maps']['n_supported']}/{r['held_out_maps']['n_pairs']} "
              f"{r['held_out_maps']['supported_maps_by_system']}")
        for k in ("pooled_map", "pairs_all", "pairs_stable", "pairs_provisional", "pairs_other"):
            print(f"  vs V0 cpu {k}: {r['vs_V0_cpu'][k]}")
        print(f"  gate changes: { {g: x['n_changed'] for g, x in r['vs_V0_cpu']['gate_changes_pairs'].items() if g in GATES} }")
        hc = r["heldout_check"]
        print(f"  held-out: evaluable {hc['n_evaluable']}, supported {hc['supported']}, "
              f"not supported {hc['not_supported']}")
        print(f"  campus lifts: {r['heldout_lift_by_campus']['median']} (min {r['heldout_lift_by_campus']['min']})")
        print(f"  routes: {r['routing']}")
    if lvc:
        print(f"\n[72] labour vs count: {lvc['all_campuses']}")
    print(f"\n[72] done in {time.time() - t_start:.0f} s")


if __name__ == "__main__":
    main()
