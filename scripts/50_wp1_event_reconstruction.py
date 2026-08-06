#!/usr/bin/env python
"""
WP1 (plan sec 20.1, items 1-5 and 10; sec 38 WP1) -- event-history feasibility audit.

Answers, from the raw work-order file and nothing else:

  1. timestamp precision      -- how finely are events actually dated?
  2. event reconstruction     -- can a per-cell chronological history be rebuilt?
  3. tied events              -- how often do events share an instant, so that
                                 order within the tie is manufactured?
  4. sequence lengths         -- how much history does a decision unit carry?
  5. field completeness       -- are subsystem / component / labour usable as
                                 event identity?
 10. context shift            -- how different are the nine campuses as contexts?

The gate this feeds is sec 20.1: TRAIL development proceeds only when event
chronology, event identity, or context adaptation carries measurable value.
This script establishes whether the *inputs* to that question exist. Whether
chronology carries predictive value is measured separately, in 51.

Everything is restricted to the nine panel universities and the panel's own
cell definition, so the event stream lines up row for row with the incumbent
aggregate panel. That alignment is asserted, not assumed.

Outputs -> results/wp1/event_audit.json
           results/wp1/event_sequences.parquet   (one row per event, ordered)
           results/wp1/context_profile.csv
"""
from __future__ import annotations
import sys, json, os
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fmscreen import config as C
from fmscreen import io as F

OUT = C.ROOT / "results" / "wp1"
OUT.mkdir(parents=True, exist_ok=True)

CELL = [C.COL_UNIV, C.COL_BUILDING, C.COL_SYSTEM]


def _atomic_json(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    json.dump(obj, open(tmp, "w"), indent=2, default=str)
    os.replace(tmp, path)


def main():
    panel = pd.read_parquet(C.DATA_PROCESSED / "panel_quarter.parquet")
    panel_univ = set(panel[C.COL_UNIV].astype(str).unique())
    print(f"[wp1] panel: {len(panel):,} rows, {len(panel_univ)} universities")

    cols = [C.COL_UNIV, C.COL_BUILDING, C.COL_SYSTEM, C.COL_SUBSYS,
            C.COL_COMPONENT, C.COL_START, C.COL_PPMUPM, C.COL_LABORHOURS]
    df = F.load_raw(usecols=cols)
    n_raw = len(df)

    # ---- restrict to the panel's own universe -------------------------------
    df[C.COL_UNIV] = df[C.COL_UNIV].astype(str)
    df = df[df[C.COL_UNIV].isin(panel_univ)]
    df = df[df[C.COL_BUILDING].notna() & df[C.COL_SYSTEM].notna()]
    df[C.COL_SYSTEM] = df[C.COL_SYSTEM].astype("string").str.strip().str.upper()
    df["is_upm"] = F.is_upm(df).to_numpy()
    df["is_ppm"] = F.is_ppm(df).to_numpy()
    df = df[df["is_upm"] | df["is_ppm"]]
    df = df[df[C.COL_START].notna()]

    assert set(df[C.COL_UNIV].unique()) == panel_univ, \
        "reconstructed event set does not cover exactly the panel universities"

    # ---- 1. timestamp precision --------------------------------------------
    ts = df[C.COL_START]
    tod = ts.dt.hour * 3600 + ts.dt.minute * 60 + ts.dt.second
    precision = {
        "n_events": int(len(df)),
        "share_with_nonzero_time_of_day": float((tod > 0).mean()),
        "share_at_exact_midnight": float((tod == 0).mean()),
        "distinct_times_of_day": int(tod.nunique()),
        "distinct_dates": int(ts.dt.normalize().nunique()),
        "distinct_seconds_resolution": int(ts.nunique()),
        "date_min": str(ts.min()), "date_max": str(ts.max()),
    }

    # ---- 2. reconstruct per-cell chronological histories --------------------
    df = F.add_time_keys(df, anchor=C.COL_START)
    df = df.sort_values(CELL + [C.COL_START]).reset_index(drop=True)
    df["event_ix"] = df.groupby(CELL, observed=True).cumcount()
    # elapsed time since the previous event in the same cell, in days
    prev = df.groupby(CELL, observed=True)[C.COL_START].shift(1)
    df["gap_days"] = (df[C.COL_START] - prev).dt.total_seconds() / 86400.0

    # ---- 3. tied events -----------------------------------------------------
    tie_exact = df["gap_days"] == 0
    tie_sameday = (df["gap_days"] < 1) & (df["gap_days"] > 0)
    ties = {
        "share_tied_exact_instant": float(tie_exact.mean()),
        "share_same_day_not_tied": float(tie_sameday.mean()),
        "share_gap_ge_1day": float((df["gap_days"] >= 1).mean()),
        "median_gap_days": float(df["gap_days"].median()),
        "note": ("events sharing an exact instant within a cell have no "
                 "recoverable order; the plan requires an order-invariant "
                 "representation for them (sec 10.4)"),
    }

    # ---- 4. sequence lengths ------------------------------------------------
    per_cell = df.groupby(CELL, observed=True).size()
    q = [0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    seqlen = {
        "n_cells_with_events": int(len(per_cell)),
        "events_per_cell_mean": float(per_cell.mean()),
        "events_per_cell_quantiles": {str(k): float(v) for k, v in
                                      per_cell.quantile(q).items()},
        "events_per_cell_max": int(per_cell.max()),
        "share_cells_with_lt10_events": float((per_cell < 10).mean()),
    }
    # history available at each panel anchor: events strictly before quarter t+1
    ev_by_cq = (df.groupby(CELL + ["period_q"], observed=True).size()
                  .rename("n_events").reset_index())
    ev_by_cq = ev_by_cq.sort_values(CELL + ["period_q"])
    ev_by_cq["cum_events"] = (ev_by_cq.groupby(CELL, observed=True)["n_events"]
                                      .cumsum())
    anchors = panel[CELL + ["period_q"]].copy()
    anchors[C.COL_UNIV] = anchors[C.COL_UNIV].astype(str)
    joined = anchors.merge(ev_by_cq[CELL + ["period_q", "cum_events"]],
                           on=CELL + ["period_q"], how="left")
    assert len(joined) == len(panel), "anchor join changed the row count"
    hist = joined["cum_events"].ffill().fillna(0)
    seqlen["history_at_anchor_quantiles"] = {
        str(k): float(v) for k, v in hist.quantile(q).items()}
    seqlen["share_anchors_with_ge8_events"] = float((hist >= 8).mean())

    # ---- 5. field completeness ---------------------------------------------
    completeness = {
        "subsystem_present": float(df[C.COL_SUBSYS].notna().mean()),
        "component_present": float(df[C.COL_COMPONENT].notna().mean()),
        "labour_present": float(df[C.COL_LABORHOURS].notna().mean()),
        "labour_positive": float((df[C.COL_LABORHOURS].fillna(0) > 0).mean()),
        "n_distinct_subsystem": int(df[C.COL_SUBSYS].nunique()),
        "n_distinct_component": int(df[C.COL_COMPONENT].nunique()),
    }

    # ---- 10. context shift across the nine campuses -------------------------
    rows = []
    for u, g in df.groupby(C.COL_UNIV, observed=True):
        cells = g.groupby(CELL, observed=True).size()
        lab = g[C.COL_LABORHOURS].fillna(0)
        rows.append({
            "university": u,
            "n_events": int(len(g)),
            "n_cells": int(len(cells)),
            "events_per_cell_median": float(cells.median()),
            "ppm_to_upm_ratio": float(g["is_ppm"].sum() / max(g["is_upm"].sum(), 1)),
            "median_gap_days": float(g["gap_days"].median()),
            "labour_median": float(lab[lab > 0].median()),
            "subsystem_present": float(g[C.COL_SUBSYS].notna().mean()),
            "component_present": float(g[C.COL_COMPONENT].notna().mean()),
            "span_quarters": int(g["period_q"].nunique()),
        })
    ctx = pd.DataFrame(rows).sort_values("n_events", ascending=False)
    ctx.to_csv(OUT / "context_profile.csv", index=False)

    def spread(col):
        v = ctx[col].to_numpy(dtype=float)
        return {"min": float(v.min()), "median": float(np.median(v)),
                "max": float(v.max()),
                "max_over_min": float(v.max() / v.min()) if v.min() > 0 else None}

    context_shift = {c: spread(c) for c in
                     ["events_per_cell_median", "ppm_to_upm_ratio",
                      "median_gap_days", "labour_median", "component_present"]}

    # ---- persist the event stream for 51 ------------------------------------
    keep = CELL + [C.COL_SUBSYS, C.COL_COMPONENT, C.COL_START, "period_q",
                   "event_ix", "gap_days", "is_upm", C.COL_LABORHOURS]
    df[keep].to_parquet(OUT / "event_sequences.parquet", index=False)

    audit = {
        "provenance": {
            "raw_rows_read": int(n_raw),
            "events_kept": int(len(df)),
            "universities": sorted(panel_univ),
            "note": ("restricted to the nine panel universities and to rows "
                     "with a building, a system and a parseable start date"),
        },
        "timestamp_precision": precision,
        "tied_events": ties,
        "sequence_lengths": seqlen,
        "field_completeness": completeness,
        "context_shift": context_shift,
    }
    _atomic_json(audit, OUT / "event_audit.json")

    # ---- readable summary ---------------------------------------------------
    print("\n=== 1. timestamp precision ===")
    print(f"  events with a real time of day : {precision['share_with_nonzero_time_of_day']:.1%}")
    print(f"  distinct times of day          : {precision['distinct_times_of_day']:,}")
    print(f"  span                           : {precision['date_min'][:10]} .. {precision['date_max'][:10]}")
    print("\n=== 3. tied events (order not recoverable) ===")
    print(f"  share tied at an exact instant : {ties['share_tied_exact_instant']:.1%}")
    print(f"  share same day, ordered        : {ties['share_same_day_not_tied']:.1%}")
    print(f"  median gap between events      : {ties['median_gap_days']:.1f} days")
    print("\n=== 4. sequence lengths ===")
    print(f"  events per cell, median        : {seqlen['events_per_cell_quantiles']['0.5']:.0f}")
    print(f"  events per cell, 95th          : {seqlen['events_per_cell_quantiles']['0.95']:.0f}")
    print(f"  anchors with >=8 prior events  : {seqlen['share_anchors_with_ge8_events']:.1%}")
    print("\n=== 5. field completeness ===")
    for k, v in completeness.items():
        if k.startswith("n_"):
            print(f"  {k:24s}: {v:,}")
        else:
            print(f"  {k:24s}: {v:.1%}")
    print("\n=== 10. context shift across nine campuses (max/min) ===")
    for k, v in context_shift.items():
        r = v["max_over_min"]
        print(f"  {k:24s}: {v['min']:.2f} .. {v['max']:.2f}"
              + (f"   ({r:.1f}x)" if r else ""))
    print(f"\n[wp1] wrote {OUT/'event_audit.json'}")
    print(f"[wp1] wrote {OUT/'event_sequences.parquet'} ({len(df):,} events)")


if __name__ == "__main__":
    main()
