"""
Panel construction — building x system x quarter.

Leakage discipline (the credibility keystone):
  * Each row is an ANCHOR period t for one (university, building, system) cell.
  * ALL features use only information available through the END of period t.
  * The LABEL describes period t+1 only.
  * The current period t's own aggregates ARE valid features (known at end of t);
    only periods > t are forbidden in features.
  * System-specific severity THRESHOLDS and train-derived PREMIUM features are NOT
    baked in here — the panel stores the raw next-period burden (`upm_labour_next`,
    `upm_cost_next`) and the fold machinery thresholds them on the training fold
    only. This keeps thresholding leakage-safe across every validation split.

Cell-inclusion (Section 12a, locked):
  * MAIN  = known-system rule: a cell is in the panel from the first period it
            appears (cell_first_q) until the building's last active period
            (building_last_q). Dormant-but-known cells remain (negatives), which
            is the honest portfolio-wide framing and avoids making the task easy.
  * SENS  = recent-activity rule: a boolean `active_prev4` flag selects the subset
            of rows with activity in the previous 4 periods (applied at eval time).

Horizon rationale: bounding by the building's last active period (not the
university's) keeps dormant-system negatives (the hard part) while avoiding
padding cells for buildings that have entirely left the records.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from . import config as C
from . import io as F

# US CPI-U annual averages (BLS, 1982-84=100). Used to deflate TotalCost to the
# base year. Cost is a SECONDARY, inflation-sensitive target; labour hours (primary)
# need no deflation. Swap for a construction-cost index if desired.
CPI_U = {
    2002: 179.9, 2003: 184.0, 2004: 188.9, 2005: 195.3, 2006: 201.6,
    2007: 207.342, 2008: 215.303, 2009: 214.537, 2010: 218.056, 2011: 224.939,
    2012: 229.594, 2013: 232.957, 2014: 236.736, 2015: 237.017, 2016: 240.007,
    2017: 245.120, 2018: 251.107, 2019: 255.657, 2020: 258.811, 2021: 270.970,
}


def _cost_deflator(years: pd.Series, base: int = C.COST_DEFLATION_BASE_YEAR) -> pd.Series:
    base_cpi = CPI_U[base]
    return years.map(lambda y: base_cpi / CPI_U.get(int(y), base_cpi) if pd.notna(y) else 1.0)


# --------------------------------------------------------------------------- #
# Step 1: raw -> cell x period aggregates
# --------------------------------------------------------------------------- #
def build_cell_period_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw work orders to one row per (univ, building, system, quarter)."""
    upm = F.is_upm(df).to_numpy()
    ppm = F.is_ppm(df).to_numpy()
    lh = df[C.COL_LABORHOURS].fillna(0.0).to_numpy()
    tc = df[C.COL_TOTALCOST].to_numpy()
    defl = _cost_deflator(df["year"]).to_numpy()
    tc_def = np.where(np.isnan(tc), 0.0, tc) * defl

    work = pd.DataFrame({
        C.COL_UNIV: df[C.COL_UNIV].astype("string"),
        C.COL_BUILDING: df[C.COL_BUILDING].astype("string"),
        C.COL_SYSTEM: df[C.COL_SYSTEM].astype("string"),
        "period_q": df["period_q"].astype("Int64"),
        "year": df["year"].astype("Int64"),
        "upm": upm.astype("int32"),
        "ppm": ppm.astype("int32"),
        "upm_labour": np.where(upm, lh, 0.0),
        "ppm_labour": np.where(ppm, lh, 0.0),
        "upm_cost": np.where(upm, tc_def, 0.0),
        "ppm_cost": np.where(ppm, tc_def, 0.0),
        # weather (period context) — averaged later at the cell-period level
        "min_temp": df[C.COL_MINTEMP].to_numpy(),
        "max_temp": df[C.COL_MAXTEMP].to_numpy(),
        "precip": df[C.COL_PRECIP].to_numpy(),
        "humidity": df[C.COL_HUMIDITY].to_numpy(),
    })
    grp = work.groupby([C.COL_UNIV, C.COL_BUILDING, C.COL_SYSTEM, "period_q"],
                       observed=True, sort=False)
    agg = grp.agg(
        n_records=("upm", "size"),
        upm_count=("upm", "sum"),
        ppm_count=("ppm", "sum"),
        upm_labour=("upm_labour", "sum"),
        ppm_labour=("ppm_labour", "sum"),
        upm_cost=("upm_cost", "sum"),
        ppm_cost=("ppm_cost", "sum"),
        min_temp=("min_temp", "mean"),
        max_temp=("max_temp", "mean"),
        precip=("precip", "mean"),
        humidity=("humidity", "mean"),
    ).reset_index()
    return agg


# --------------------------------------------------------------------------- #
# Step 2: build the dense known-system grid
# --------------------------------------------------------------------------- #
def build_grid(agg: pd.DataFrame) -> pd.DataFrame:
    """Dense (cell x period) grid from cell_first_q .. building_last_q (known-system)."""
    cellkey = [C.COL_UNIV, C.COL_BUILDING, C.COL_SYSTEM]
    # cell first period
    cell_first = agg.groupby(cellkey, observed=True)["period_q"].min().rename("cell_first_q")
    # building last period (any system in that building)
    bkey = [C.COL_UNIV, C.COL_BUILDING]
    build_last = agg.groupby(bkey, observed=True)["period_q"].max().rename("building_last_q")
    build_first = agg.groupby(bkey, observed=True)["period_q"].min().rename("building_first_q")

    cells = cell_first.reset_index().merge(
        build_last.reset_index(), on=bkey, how="left").merge(
        build_first.reset_index(), on=bkey, how="left")
    cells["cell_first_q"] = cells["cell_first_q"].astype(int)
    cells["building_last_q"] = cells["building_last_q"].astype(int)

    # generate periods cell_first_q .. building_last_q (inclusive) for each cell
    lo = cells["cell_first_q"].to_numpy()
    hi = cells["building_last_q"].to_numpy()
    lengths = (hi - lo + 1).clip(min=0)
    total = int(lengths.sum())
    periods = np.empty(total, dtype=np.int64)
    pos = 0
    for i in range(len(cells)):
        L = lengths[i]
        if L > 0:
            periods[pos:pos + L] = np.arange(lo[i], hi[i] + 1)
            pos += L
    rep = np.repeat(np.arange(len(cells)), lengths)
    grid = pd.DataFrame({
        C.COL_UNIV: cells[C.COL_UNIV].to_numpy()[rep],
        C.COL_BUILDING: cells[C.COL_BUILDING].to_numpy()[rep],
        C.COL_SYSTEM: cells[C.COL_SYSTEM].to_numpy()[rep],
        "period_q": periods,
        "cell_first_q": cells["cell_first_q"].to_numpy()[rep],
        "building_last_q": cells["building_last_q"].to_numpy()[rep],
    })
    return grid


# --------------------------------------------------------------------------- #
# Step 3: merge aggregates onto grid; compute past-only features + labels
# --------------------------------------------------------------------------- #
def _q_to_year(period_q: np.ndarray) -> np.ndarray:
    return (period_q // 4).astype(int)


def _q_to_quarter(period_q: np.ndarray) -> np.ndarray:
    return (period_q % 4 + 1).astype(int)


def assemble_panel(agg: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    cellkey = [C.COL_UNIV, C.COL_BUILDING, C.COL_SYSTEM]
    aggcols = ["n_records", "upm_count", "ppm_count", "upm_labour", "ppm_labour",
               "upm_cost", "ppm_cost", "min_temp", "max_temp", "precip", "humidity"]
    panel = grid.merge(agg[cellkey + ["period_q"] + aggcols],
                       on=cellkey + ["period_q"], how="left")
    # active period = had any record this quarter
    panel["active"] = panel["n_records"].notna().astype("int8")
    for c in ["n_records", "upm_count", "ppm_count", "upm_labour", "ppm_labour",
              "upm_cost", "ppm_cost"]:
        panel[c] = panel[c].fillna(0.0)
    # weather: fill inactive cell-periods with the university-period mean (regional context)
    panel = _fill_weather(panel)

    panel["year"] = _q_to_year(panel["period_q"].to_numpy())
    panel["quarter"] = _q_to_quarter(panel["period_q"].to_numpy())

    panel = panel.sort_values(cellkey + ["period_q"]).reset_index(drop=True)
    g = panel.groupby(cellkey, observed=True, sort=False)

    # ---- LABELS (period t+1) ----
    panel["upm_count_next"] = g["upm_count"].shift(-1)
    panel["upm_labour_next"] = g["upm_labour"].shift(-1)
    panel["upm_cost_next"] = g["upm_cost"].shift(-1)
    panel["occurrence_next"] = (panel["upm_count_next"] > 0).astype("Int8")
    # rows whose t+1 is unobserved (last row per cell) get NaN labels -> dropped later
    panel.loc[panel["upm_count_next"].isna(), "occurrence_next"] = pd.NA

    # ---- FEATURES (periods <= t) ----
    # rolling windows ending at and INCLUDING t (current period is past info at end of t)
    for w in C.FEATURE_LAGS:  # 1, 2, 4
        panel[f"upm_cnt_w{w}"] = g["upm_count"].transform(lambda s: s.rolling(w, min_periods=1).sum())
        panel[f"ppm_cnt_w{w}"] = g["ppm_count"].transform(lambda s: s.rolling(w, min_periods=1).sum())
    # previous-year (4-quarter) burden magnitudes
    panel["upm_labour_y1"] = g["upm_labour"].transform(lambda s: s.rolling(4, min_periods=1).sum())
    panel["ppm_labour_y1"] = g["ppm_labour"].transform(lambda s: s.rolling(4, min_periods=1).sum())
    panel["upm_cost_y1"] = g["upm_cost"].transform(lambda s: s.rolling(4, min_periods=1).sum())
    # cumulative burden up to and including t
    panel["upm_labour_cum"] = g["upm_labour"].cumsum()
    panel["ppm_labour_cum"] = g["ppm_labour"].cumsum()
    panel["upm_cnt_cum"] = g["upm_count"].cumsum()
    panel["ppm_cnt_cum"] = g["ppm_count"].cumsum()
    # PPM:UPM ratio (cumulative, smoothed)
    panel["ppm_upm_ratio"] = (panel["ppm_cnt_cum"] + 1.0) / (panel["upm_cnt_cum"] + 1.0)
    # data-density indicator: periods of history & active-period share
    panel["periods_since_first"] = (panel["period_q"] - panel["cell_first_q"]).astype(int) + 1
    panel["active_cum"] = g["active"].cumsum()
    panel["active_share_hist"] = panel["active_cum"] / panel["periods_since_first"]
    # recency: time since last UPM / PPM (periods), as of end of t
    panel["time_since_upm"] = _time_since(panel, g, "upm_count")
    panel["time_since_ppm"] = _time_since(panel, g, "ppm_count")
    # recent-activity flag for the sensitivity rule (active in previous 4 periods, incl. t)
    panel["active_prev4"] = (g["active"].transform(lambda s: s.rolling(4, min_periods=1).sum()) > 0).astype("int8")

    # ---- TRENDS (dynamic) ----
    # UPM frequency trend: last 2 quarters minus the 2 before (sum diff)
    upm_last2 = g["upm_count"].transform(lambda s: s.rolling(2, min_periods=1).sum())
    upm_prev2 = g["upm_count"].transform(lambda s: s.shift(2).rolling(2, min_periods=1).sum())
    panel["upm_freq_trend"] = (upm_last2 - upm_prev2.fillna(0))
    ppm_last2 = g["ppm_count"].transform(lambda s: s.rolling(2, min_periods=1).sum())
    ppm_prev2 = g["ppm_count"].transform(lambda s: s.shift(2).rolling(2, min_periods=1).sum())
    panel["ppm_activity_trend"] = (ppm_last2 - ppm_prev2.fillna(0))
    # dynamic reactive-burden trend: rising share of reactive labour
    react_share = panel["upm_labour"] / (panel["upm_labour"] + panel["ppm_labour"] + 1e-6)
    panel["_react_share"] = react_share
    panel["react_share_last2"] = (panel.groupby(cellkey, observed=True, sort=False)["_react_share"]
                                  .transform(lambda s: s.rolling(2, min_periods=1).mean()))
    panel["react_share_prev2"] = (panel.groupby(cellkey, observed=True, sort=False)["_react_share"]
                                  .transform(lambda s: s.shift(2).rolling(2, min_periods=1).mean()))
    panel["react_burden_trend"] = (panel["react_share_last2"] - panel["react_share_prev2"].fillna(0))
    panel.drop(columns=["_react_share", "react_share_last2", "react_share_prev2"], inplace=True)

    return panel


def _fill_weather(panel: pd.DataFrame) -> pd.DataFrame:
    wcols = ["min_temp", "max_temp", "precip", "humidity"]
    up = panel.groupby([C.COL_UNIV, "period_q"], observed=True)[wcols].transform("mean")
    for c in wcols:
        panel[c] = panel[c].fillna(up[c])
        panel[c] = panel[c].fillna(panel[c].median())
    return panel


def _time_since(panel: pd.DataFrame, g, count_col: str) -> pd.Series:
    """Periods since the most recent active (count>0) period, inclusive of t.

    0 if the current period t has the event; grows by 1 each dormant period;
    NaN-> large sentinel if never seen yet.
    """
    active = (panel[count_col] > 0)
    # within each cell, index of period (dense): use cumcount
    idx = g.cumcount()
    # last active idx seen so far (ffill within cell)
    last_active_idx = idx.where(active.to_numpy())
    last_active_idx = panel.assign(_lai=last_active_idx).groupby(
        [C.COL_UNIV, C.COL_BUILDING, C.COL_SYSTEM], observed=True, sort=False)["_lai"].ffill()
    ts = (idx - last_active_idx)
    ts = ts.fillna(999).astype("float32")  # never-seen sentinel
    return ts


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def build(nrows: int | None = None) -> pd.DataFrame:
    cols = [C.COL_UNIV, C.COL_BUILDING, C.COL_SYSTEM, C.COL_SYSTEM_DESC, C.COL_PPMUPM,
            C.COL_START, C.COL_LABORHOURS, C.COL_TOTALCOST,
            C.COL_MINTEMP, C.COL_MAXTEMP, C.COL_PRECIP, C.COL_HUMIDITY]
    df = F.load_raw(nrows=nrows, usecols=cols)
    df = F.add_time_keys(df, anchor=C.COL_START)
    # normalize system codes (fixes case/whitespace inconsistencies, e.g. g20->G20)
    df[C.COL_SYSTEM] = df[C.COL_SYSTEM].astype("string").str.strip().str.upper()
    df[C.COL_SYSTEM_DESC] = df[C.COL_SYSTEM_DESC].astype("string").str.strip()
    # panel requires non-null building & system & period
    keep = df[C.COL_BUILDING].notna() & df[C.COL_SYSTEM].notna() & df["period_q"].notna()
    df = df[keep].copy()
    # attach a system->description map for readability
    sysmap = (df[[C.COL_SYSTEM, C.COL_SYSTEM_DESC]].dropna()
                .drop_duplicates(C.COL_SYSTEM).astype("string"))
    agg = build_cell_period_aggregates(df)
    grid = build_grid(agg)
    panel = assemble_panel(agg, grid)
    panel = panel.merge(sysmap, on=C.COL_SYSTEM, how="left")
    return panel
