"""
Central configuration for the CMMS-only reactive-maintenance screening study.

This module encodes:
  * file paths,
  * the exact FMUCD column names (9 attribute groups),
  * the LOCKED design decisions, and
  * Phase-0-resolved gate parameters (filled in after 00_phase0_inspect.py).

Nothing here uses future information; it only declares names and constants.
"""
from __future__ import annotations
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw" / "FMUCD.csv"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
PHASE0_DIR = RESULTS / "phase0"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
METRICS = RESULTS / "metrics"
NOTES = ROOT / "notes"
LOGS = ROOT / "logs"

for _p in (DATA_INTERIM, DATA_PROCESSED, PHASE0_DIR, TABLES, FIGURES, METRICS, NOTES, LOGS):
    _p.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# FMUCD columns (exact header strings, 38 columns, 9 attribute groups)
# --------------------------------------------------------------------------- #
# University / location
COL_UNIV = "UniversityID"
COL_COUNTRY = "Country"
COL_STATE = "State/Province"
# Building
COL_BUILDING = "BuildingID"
COL_BUILDING_NAME = "BuildingName"
COL_SIZE = "Size"
COL_BTYPE = "Type"
COL_BUILT_YEAR = "BuiltYear"
COL_FCI = "FCI (facility condition index)"
COL_CRV = "CRV (current replacement value)"
COL_DMC = "DMC (deferred maintenance cost)"
# System hierarchy (Uniformat II / FMUCO)
COL_SYSTEM = "SystemCode"
COL_SYSTEM_DESC = "SystemDescription"
COL_SUBSYS = "SubsystemCode"
COL_SUBSYS_DESC = "SubsystemDescription"
COL_COMPONENT = "DescriptiveCode"
COL_COMPONENT_DESC = "ComponentDescription"
# Work order
COL_WOID = "WOID"
COL_WODESC = "WODescription"
COL_WOPRIORITY = "WOPriority"
COL_START = "WOStartDate"
COL_END = "WOEndDate"
COL_DURATION = "WODuration"
COL_PPMUPM = "PPM/UPM"
# Cost
COL_LABORCOST = "LaborCost"
COL_MATERIALCOST = "MaterialCost"
COL_OTHERCOST = "OtherCost"
COL_TOTALCOST = "TotalCost"
# Labour
COL_LABORHOURS = "LaborHours"
# Weather
COL_MINTEMP = "MinTemp.(°C)"
COL_MAXTEMP = "MaxTemp.(°C)"
COL_PRESSURE = "Atmospheric pressure(hPa)"
COL_HUMIDITY = "Humidity(%)"
COL_WINDSPEED = "WindSpeed(m/s)"
COL_WINDDEG = "WindDegree"
COL_PRECIP = "Precipitation(mm)"
COL_SNOW = "Snow(mm)"
COL_CLOUD = "Cloudness(%)"

WEATHER_COLS = [
    COL_MINTEMP, COL_MAXTEMP, COL_PRESSURE, COL_HUMIDITY,
    COL_WINDSPEED, COL_WINDDEG, COL_PRECIP, COL_SNOW, COL_CLOUD,
]

STRING_COLS = [
    COL_UNIV, COL_COUNTRY, COL_STATE, COL_BUILDING, COL_BUILDING_NAME, COL_BTYPE,
    COL_SYSTEM, COL_SYSTEM_DESC, COL_SUBSYS, COL_SUBSYS_DESC, COL_COMPONENT,
    COL_COMPONENT_DESC, COL_WOID, COL_WODESC, COL_WOPRIORITY, COL_PPMUPM,
]
NUMERIC_COLS = [
    COL_SIZE, COL_BUILT_YEAR, COL_FCI, COL_CRV, COL_DMC, COL_DURATION,
    COL_LABORCOST, COL_MATERIALCOST, COL_OTHERCOST, COL_TOTALCOST, COL_LABORHOURS,
] + WEATHER_COLS
DATE_COLS = [COL_START, COL_END]

PPM_VALUE = "PPM"
UPM_VALUE = "UPM"

# --------------------------------------------------------------------------- #
# LOCKED design decisions
# --------------------------------------------------------------------------- #
RANDOM_SEED = 42

# Unit of analysis: building x system x period
CELL_KEYS = [COL_UNIV, COL_BUILDING, COL_SYSTEM]

# Sufficiency criterion (Section 21, locked)
TOPK_BUDGET = 0.10          # top 10% inspection budget (primary)
TOPK_BUDGET_SECONDARY = 0.05
SUFFICIENCY_MIN_LIFT = 2.0  # >= 2x precision lift over base rate
# "stable": meets 2x lift in the MAJORITY of held-out campuses/folds,
# with no extreme dependence on a single institution.
STABILITY_MIN_FRACTION = 0.5

# Severity target percentiles (Section 13): 75 primary; 50/90 sensitivity
SEVERITY_PCTL_PRIMARY = 75
SEVERITY_PCTLS = [50, 75, 90]

# Cell-inclusion rules (Section 12a, locked)
#   main  = "known_system"   : system has appeared up to & incl. period t
#   sens  = "recent_activity": cell active in previous 4 periods
CELL_RULE_MAIN = "known_system"
CELL_RULE_SENS = "recent_activity"
RECENT_ACTIVITY_WINDOW = 4  # periods

# Feature lag windows (Section 12): previous 1, 2, 4 periods
FEATURE_LAGS = [1, 2, 4]

# Local-calibration augmentation fractions (Section 18, primary novelty)
CALIB_FRACTIONS = [0.0, 0.05, 0.10, 0.20]

# Temporal validation cutoff -- REVISED by Phase 0 (early years 2002-2014 are thin;
# data concentrates 2015-2021). Train <= 2018, test 2019-2021 keeps ~34% in test.
TEMPORAL_TRAIN_END_YEAR = 2018

# Cost-completeness threshold to include a university in cost-based analysis
COST_COMPLETENESS_MIN = 0.30
LABOUR_COMPLETENESS_MIN = 0.30

# Bootstrap settings for confidence intervals (clustered)
N_BOOTSTRAP = 1000
BOOTSTRAP_CI = 0.95

# --------------------------------------------------------------------------- #
# Phase-0-RESOLVED gate parameters
# (defaults below; overwritten/confirmed by results/phase0/gate_resolution.json)
# --------------------------------------------------------------------------- #
# --- RESOLVED by scripts/00_phase0_inspect.py (see results/phase0/) ---
# Gate 1: month-or-finer timestamps (99.7% non-Jan-1) -> quarter panel
PERIOD = "quarter"
# Gate 2: genuine TotalCost (=Labor+Material+Other in 97.9%) -> both targets;
# but labour is far more complete (82% vs 44%) so labour is PRIMARY.
BURDEN_TARGETS = "both"
COST_DEFLATION_BASE_YEAR = 2021

# Panel scope: building x system requires a non-null BuildingID. 22.5% of records
# (3 universities -- 8, 9, 12 -- that report no BuildingID) are EXCLUDED from the
# building-level panel and reported in dataset-count attribution. Result: the panel
# covers 9 universities.
UNIVERSITIES_NO_BUILDING = {"8", "9", "12"}
# Building-age/asset metadata is sparse (BuiltYear ~24%); heterogeneity by building
# age is a supplementary subset, system-level heterogeneity is the main stratification.
