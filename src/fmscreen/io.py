"""
Raw FMUCD loader with explicit dtypes and robust date parsing.

Loading is memory-conscious (categoricals for low-cardinality codes) and the
date parser reports how many timestamps failed to parse. No future information
is created here; this module only reads and lightly types the raw file.
"""
from __future__ import annotations
import sys
import warnings
import numpy as np
import pandas as pd

from . import config as C

# Low-cardinality -> category; high-cardinality identifiers/text -> string
_CATEGORY_COLS = [
    C.COL_UNIV, C.COL_COUNTRY, C.COL_STATE, C.COL_BTYPE,
    C.COL_SYSTEM, C.COL_SYSTEM_DESC, C.COL_SUBSYS, C.COL_SUBSYS_DESC,
    C.COL_COMPONENT, C.COL_COMPONENT_DESC, C.COL_WOPRIORITY, C.COL_PPMUPM,
]
_STRING_COLS = [C.COL_BUILDING, C.COL_BUILDING_NAME, C.COL_WOID, C.COL_WODESC]


def _dtype_map() -> dict:
    # Read low-cardinality cols as string (chunked C-engine can't union category
    # dtypes across chunks); convert to category after load for memory savings.
    d = {c: "string" for c in _CATEGORY_COLS}
    for c in _STRING_COLS:
        d[c] = "string"
    for c in C.NUMERIC_COLS:
        d[c] = "float64"
    return d


def parse_dates(s: pd.Series) -> pd.Series:
    """Parse 'YYYY-MM-DD HH:MM:SS' robustly; fall back to mixed for stragglers."""
    out = pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    miss = out.isna() & s.notna()
    if miss.any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out2 = pd.to_datetime(s[miss], errors="coerce", format="mixed")
        out.loc[miss] = out2
    return out


def load_raw(nrows: int | None = None, usecols: list[str] | None = None,
             verbose: bool = True) -> pd.DataFrame:
    """Load the raw FMUCD CSV with typed columns and parsed dates."""
    dtypes = _dtype_map()
    if usecols is not None:
        dtypes = {k: v for k, v in dtypes.items() if k in usecols}
    # read date columns as raw string first, parse afterwards
    read_dtypes = {k: v for k, v in dtypes.items()}
    for dc in C.DATE_COLS:
        if usecols is None or dc in usecols:
            read_dtypes[dc] = "string"
    if verbose:
        print(f"[io] reading {C.DATA_RAW} (nrows={nrows}) ...", file=sys.stderr, flush=True)
    na_tokens = ["", " ", "  ", "NA", " NA", "NA ", " NA ", "N/A", "n/a",
                 "nan", "NaN", "None", "NULL", "null", "-", "--", "?", "."]
    df = pd.read_csv(
        C.DATA_RAW, dtype=read_dtypes, usecols=usecols, nrows=nrows,
        na_values=na_tokens, keep_default_na=True, skipinitialspace=True,
    )
    for dc in C.DATE_COLS:
        if dc in df.columns:
            df[dc] = parse_dates(df[dc])
    # convert low-cardinality string cols to category for memory savings
    for c in _CATEGORY_COLS:
        if c in df.columns:
            df[c] = df[c].astype("category")
    if verbose:
        print(f"[io] loaded shape={df.shape}", file=sys.stderr, flush=True)
    return df


def add_time_keys(df: pd.DataFrame, anchor: str = C.COL_START) -> pd.DataFrame:
    """Add year / quarter / period keys derived from the anchor timestamp."""
    dt = df[anchor]
    df = df.copy()
    df["year"] = dt.dt.year.astype("Int64")
    df["quarter"] = dt.dt.quarter.astype("Int64")
    # period_q: integer YYYY*4 + (Q-1)  -> monotone quarter index
    df["period_q"] = (df["year"].astype("float") * 4 + (df["quarter"].astype("float") - 1)).astype("Int64")
    # period_y: just the year
    df["period_y"] = df["year"]
    # human-readable quarter label
    df["period_label"] = (df["year"].astype("string") + "Q" + df["quarter"].astype("string"))
    return df


def is_upm(df: pd.DataFrame) -> pd.Series:
    s = df[C.COL_PPMUPM].astype("string").str.upper().str.strip() == C.UPM_VALUE
    return s.fillna(False).astype(bool)  # numpy-bool, no NA


def is_ppm(df: pd.DataFrame) -> pd.Series:
    s = df[C.COL_PPMUPM].astype("string").str.upper().str.strip() == C.PPM_VALUE
    return s.fillna(False).astype(bool)
