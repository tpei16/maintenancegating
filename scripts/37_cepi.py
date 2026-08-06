"""
37_cepi.py -- Condition-Evidence Priority Index (CEPI).

Turns the existing leakage-controlled screening pipeline into an evidence-gated
prioritisation layer, WITHOUT introducing any new modelling or fabricated numbers.

  R    = future unplanned-maintenance burden risk, per building-system-quarter unit.
         Source: out-of-fold LOUO predictions (severity_labour p75, layer M1, gbdt),
         converted to a within-held-out-campus percentile rank in [0,1]. Because the
         scoring campus is held out, R is leakage-controlled by construction.
  S    = CMMS evidence sufficiency, per UNIFORMAT system, the equal-weighted mean of
         five gate scores in {0.0 fail, 0.5 caution, 1.0 pass}, after hard-fail caps.
  G    = 1 - S  (evidence gap)
  CEPI = R * G  (per unit; G taken from the unit's system)

Gates (all computed from out-of-fold pipeline outputs):
  1 data_sufficiency     breadth/volume of usable history for the stratum
  2 risk_concentration   pooled top-10% lift  (heterogeneity_by_system)
  3 transfer_stability   fraction of held-out campuses clearing 2x (deployment_matrix)
  4 calibration_reliab.  monotone risk bands + top-band lift, from pred_louo
  5 antecedent_signal    escalating / no-antecedent share of p90 events (trajectory)

Hard gates = {data_sufficiency, transfer_stability, calibration_reliability}:
  >=1 hard fail -> S capped at 0.50 ; >=2 hard fail -> S = 0.

Outputs (real numbers only):
  results/tables/cepi_gates.csv
  results/tables/cepi_priority_map.csv
  results/metrics/cepi_gates.json
  results/metrics/cepi_summary.json
"""
from __future__ import annotations
import json, glob
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
MET = ROOT / "results" / "metrics"
TAB = ROOT / "results" / "tables"
PRED = ROOT / "data" / "processed" / "pred_louo"

PASS, CAUTION, FAIL = 1.0, 0.5, 0.0
HARD_GATES = ["data_sufficiency", "transfer_stability", "calibration_reliability"]

# the data-sufficiency pass level; a two-fold share over fewer contexts than this
# is not an estimate of transfer.
MIN_CTX_FOR_TRANSFER = 6

# code-side gate name -> paper-side gate name, so a reader starting from this
# repository can map the two without consulting the supplement.
GATE_NAME_MAP = {
    "data_sufficiency":        "data sufficiency",
    "risk_concentration":      "risk concentration",
    "transfer_stability":      "transfer stability",
    "calibration_reliability": "risk-gradient reliability",
    "antecedent_signal":       "prior-record trace",
}
SOFT_GATES = ["risk_concentration", "antecedent_signal"]
GATES = ["data_sufficiency", "risk_concentration", "transfer_stability",
         "calibration_reliability", "antecedent_signal"]


def score_label(x):
    return {PASS: "pass", CAUTION: "caution", FAIL: "fail"}[x]


# ---------------------------------------------------------------- load R (per unit)
def load_units():
    frames = []
    for f in sorted(glob.glob(str(PRED / "*.parquet"))):
        d = pd.read_parquet(f)
        d = d[(d.target_kind == "severity_labour") & (d.pctl == 75)
              & (d.layer == "M1") & (d.model == "gbdt")].copy()
        # within-held-out-campus percentile rank of the score -> R in [0,1]
        d["R"] = d.groupby("held_out_university")["score"].rank(pct=True, method="average")
        frames.append(d[["held_out_university", "university", "system", "y", "score", "R"]])
    u = pd.concat(frames, ignore_index=True)
    return u


# ---------------------------------------------------------------- gate inputs
def load_gate_inputs():
    het = pd.read_csv(MET / "heterogeneity_by_system.csv")
    dep = pd.read_csv(TAB / "deployment_matrix.csv")
    burden = pd.read_csv(TAB / "burden_by_system.csv")
    traj = json.load(open(MET / "trajectory.json"))
    trajd = {r["system"]: r for r in traj["by_system"]}
    name = dict(zip(burden.SystemCode, burden.SystemDescription.str.strip()))

    # Burden share is computed on the NINE-campus analytical panel, not on the
    # raw twelve-campus file. burden_by_system.csv is a corpus-level descriptive
    # whose denominator includes the three campuses excluded from the panel for
    # reporting no building identifier; using it here put a 12-campus quantity
    # in a row whose every other column is a 9-campus quantity, and the
    # "share of portfolio reactive labor" the paper reports is the portfolio it
    # actually analyses.
    panel = pd.read_parquet(ROOT / "data" / "processed" / "panel_quarter.parquet")
    lab = panel.groupby("SystemCode", observed=True)["upm_labour"].sum()
    lab = lab[lab > 0]
    share = (lab / lab.sum()).to_dict()
    return het, dep, name, share, trajd


# ---------------------------------------------------------------- gate 4: risk gradient
# default within-gate threshold levels (perturbed only in the sensitivity appendix)
THR = dict(data_npos=500, lift_pass=2.0, lift_caution=1.5, transfer_pass=0.8,
           transfer_caution=0.6, cal_top=2.0, cal_top_caution=1.5,
           zp_pass=0.10, zp_fail=0.25)


def calibration_gate(units_sys, top_pass=2.0, top_caution=1.5):
    """Monotone observed-burden bands + top-band lift, from out-of-fold R."""
    d = units_sys.dropna(subset=["R"])
    if len(d) < 200 or d.y.sum() < 30:
        return CAUTION, {"note": "sparse", "n": int(len(d)), "n_pos": int(d.y.sum())}
    q = pd.qcut(d["R"], 5, labels=False, duplicates="drop")
    obs = d.groupby(q)["y"].mean()
    base = d.y.mean()
    top_lift = obs.iloc[-1] / base if base > 0 else np.nan
    rho, _ = spearmanr(obs.index.values, obs.values)
    info = {"top_band_lift": round(float(top_lift), 3), "band_spearman": round(float(rho), 3),
            "n_bands": int(len(obs))}
    if top_lift >= top_pass and rho >= 0.9:
        return PASS, info
    if top_lift >= top_caution and rho >= 0.5:
        return CAUTION, info
    return FAIL, info


# ---------------------------------------------------------------- gate assembly
def build_gates(units, het, dep, name, share, trajd, thr=None):
    thr = {**THR, **(thr or {})}
    het = het.set_index("system")
    dep = dep.set_index("system")
    rows = []
    for sys in het.index:
        h = het.loc[sys]
        # 1 data sufficiency: breadth (campuses) and event support
        nc, npos = int(h.n_campuses_evaluable), int(h.n_pos)
        if nc >= 6 and npos >= thr["data_npos"]:
            g1 = PASS
        elif nc < 3 or npos < 100:
            g1 = FAIL
        else:
            g1 = CAUTION
        # 2 risk concentration: pooled top-10% lift
        lift = float(h.lift_top10)
        g2 = PASS if lift >= thr["lift_pass"] else (CAUTION if lift >= thr["lift_caution"] else FAIL)
        # 3 transfer stability: fraction of held-out campuses meeting 2x.
        # A share computed over fewer contexts than the data-sufficiency gate
        # requires is not an estimate of transfer, so it is capped at caution
        # whatever its value.
        frac = float(h.frac_campuses_meet_2x)
        g3 = PASS if frac >= thr["transfer_pass"] else (CAUTION if frac >= thr["transfer_caution"] else FAIL)
        if int(getattr(h, "n_campuses_evaluable", 0)) < MIN_CTX_FOR_TRANSFER:
            g3 = min(g3, CAUTION)
        # 4 calibration reliability: monotone bands from pred_louo
        g4, cal_info = calibration_gate(units[units.system == sys], thr["cal_top"], thr["cal_top_caution"])
        # 5 antecedent signal: escalating / no-antecedent share of p90 events
        t = trajd.get(sys)
        if t is None or t.get("n_events", 0) < 30:
            g5, ant_info = CAUTION, {"note": "few p90 events"}
        else:
            zp = float(t["zero_prior"])
            ant_info = {"zero_prior": round(zp, 3), "n_events": int(t["n_events"])}
            if zp < thr["zp_pass"]:
                g5 = PASS
            elif zp >= thr["zp_fail"]:
                g5 = FAIL
            else:
                g5 = CAUTION
        scores = {"data_sufficiency": g1, "risk_concentration": g2,
                  "transfer_stability": g3, "calibration_reliability": g4,
                  "antecedent_signal": g5}
        rows.append({"system": sys, "system_desc": name.get(sys, sys),
                     "burden_share": float(share.get(sys, np.nan)),
                     "lift_top10": lift, "frac_campuses_meet_2x": frac,
                     "n_campuses_evaluable": nc, "n_pos": npos,
                     **scores,
                     "_cal": cal_info, "_ant": ant_info})
    return rows


def sufficiency(scores, weights=None, hard_fail=True):
    """Combine gate scores into S in [0,1]."""
    if weights is None:
        weights = {g: 1.0 for g in GATES}
    wsum = sum(weights[g] for g in GATES)
    S = sum(weights[g] * scores[g] for g in GATES) / wsum
    if hard_fail:
        n_hard_fail = sum(1 for g in HARD_GATES if scores[g] == FAIL)
        if n_hard_fail >= 2:
            S = 0.0
        elif n_hard_fail >= 1:
            S = min(S, 0.50)
    return S


# ---------------------------------------------------------------- classes
def classify(R, S, r_hi=0.90, s_hi=0.67):
    hiR, hiS = R >= r_hi, S >= s_hi
    if hiR and hiS:
        return "records_informative_high_risk"
    if hiR and not hiS:
        return "evidence_insufficient_high_risk"
    if (not hiR) and hiS:
        return "routine_monitoring"
    return "watchlist"


CLASS_ACTION = {
    "records_informative_high_risk": "CMMS-led action (supervisor review, planned intervention, replacement assessment)",
    "evidence_insufficient_high_risk": "Prioritise condition evidence (inspection, BAS/BMS review, asset survey, targeted sensing)",
    "watchlist": "Watchlist unless safety/criticality overrides",
    "routine_monitoring": "Routine monitoring",
}


def main():
    units = load_units()
    het, dep, name, share, trajd = load_gate_inputs()
    grows = build_gates(units, het, dep, name, share, trajd)
    gdf = pd.DataFrame(grows)

    # primary S / G (equal weight, hard-fail on)
    gdf["S"] = [sufficiency({g: r[g] for g in GATES}) for _, r in gdf.iterrows()]
    gdf["G"] = 1.0 - gdf["S"]
    Smap = dict(zip(gdf.system, gdf.S))
    Gmap = dict(zip(gdf.system, gdf.G))

    # per-unit CEPI
    units = units[units.system.isin(Smap)].copy()
    units["S"] = units.system.map(Smap)
    units["G"] = units.system.map(Gmap)
    units["CEPI"] = units["R"] * units["G"]
    units["cls"] = [classify(r, s) for r, s in zip(units.R, units.S)]

    # ------- gate table (systems ordered by burden share)
    gdf = gdf.sort_values("burden_share", ascending=False)
    gate_cols = ["system", "system_desc", "burden_share"] + GATES + ["S", "G"]
    gtab = gdf[gate_cols].copy()
    gtab.to_csv(TAB / "cepi_gates.csv", index=False)

    # ------- priority-map table (system level)
    pm = (units.groupby("system")
          .agg(n_units=("R", "size"), n_pos=("y", "sum"),
               mean_R=("R", "mean"), mean_CEPI=("CEPI", "mean"))
          .reset_index())
    pm = pm.merge(gdf[["system", "system_desc", "burden_share", "S", "G"]], on="system")
    pm["frac_high_CEPI_0.45"] = (units.assign(hi=units.CEPI >= 0.45)
                                 .groupby("system")["hi"].mean().reindex(pm.system).values)
    # dominant quadrant per system at unit level
    dom = (units.groupby(["system", "cls"]).size().reset_index(name="n")
           .sort_values("n", ascending=False).drop_duplicates("system")
           .set_index("system")["cls"])
    pm["dominant_class"] = pm.system.map(dom)
    pm = pm.sort_values("burden_share", ascending=False)
    pm.to_csv(TAB / "cepi_priority_map.csv", index=False)

    # ------- class counts (primary)
    cls_counts = units.cls.value_counts().to_dict()
    n = len(units)
    hi_cepi = int((units.CEPI >= 0.45).sum())

    # ------- sensitivity
    sens = {}
    # R threshold sweep (class shares)
    sens["R_threshold"] = {}
    for r_hi in [0.95, 0.90, 0.85, 0.80]:
        c = units.apply(lambda x: classify(x.R, x.S, r_hi=r_hi), axis=1).value_counts()
        sens["R_threshold"][f"top{int(round((1-r_hi)*100))}pct"] = {
            k: int(c.get(k, 0)) for k in CLASS_ACTION}
    # CEPI threshold sweep (# high-CEPI units)
    sens["CEPI_threshold"] = {f"{t}": int((units.CEPI >= t).sum())
                              for t in [0.35, 0.45, 0.55]}
    # gate-weight sweep (S per system + # systems insufficient S<0.67)
    weight_schemes = {
        "equal": {g: 1.0 for g in GATES},
        "reliability_heavy": {**{g: 1.0 for g in GATES},
                              "calibration_reliability": 2.0, "transfer_stability": 2.0},
        "evidence_gap_heavy": {**{g: 1.0 for g in GATES},
                               "antecedent_signal": 2.0, "risk_concentration": 2.0},
    }
    sens["gate_weights"] = {}
    for wname, w in weight_schemes.items():
        Sw = {r.system: sufficiency({g: r[g] for g in GATES}, weights=w) for _, r in gdf.iterrows()}
        n_insuff = sum(1 for s in Sw.values() if s < 0.67)
        sens["gate_weights"][wname] = {
            "n_systems_insufficient": int(n_insuff),
            "S_by_system": {s: round(v, 3) for s, v in Sw.items()}}
    # hard-fail on/off (S per system)
    sens["hard_fail"] = {}
    for hf in [True, False]:
        Sh = {r.system: sufficiency({g: r[g] for g in GATES}, hard_fail=hf) for _, r in gdf.iterrows()}
        sens["hard_fail"]["on" if hf else "off"] = {
            "n_systems_insufficient": int(sum(1 for s in Sh.values() if s < 0.67)),
            "S_by_system": {s: round(v, 3) for s, v in Sh.items()}}

    # ------- within-gate threshold sensitivity (Appendix D): perturb each gate's
    # level around the primary spec; report the set of insufficient systems.
    CORE_INSUFF = {"D10", "G30", "C20", "F20"}  # conveying, site-mech, stairs, demolition
    perturb = {
        "risk_concentration lift 1.8": {"lift_pass": 1.8},
        "risk_concentration lift 2.2": {"lift_pass": 2.2},
        "transfer 70%": {"transfer_pass": 0.7},
        "transfer 90%": {"transfer_pass": 0.9},
        "calibration top 1.8x": {"cal_top": 1.8},
        "calibration top 2.2x": {"cal_top": 2.2},
        "data support 300": {"data_npos": 300},
        "data support 700": {"data_npos": 700},
        "prior-trace -5pp": {"zp_pass": 0.05, "zp_fail": 0.20},
        "prior-trace +5pp": {"zp_pass": 0.15, "zp_fail": 0.30},
    }
    wgs = {}
    for pname, pthr in perturb.items():
        prows = build_gates(units, het.reset_index(), dep.reset_index(), name, share, trajd, thr=pthr)
        Sp = {r["system"]: sufficiency({g: r[g] for g in GATES}) for r in prows}
        insuff = sorted([s for s, v in Sp.items() if v < 0.67])
        wgs[pname] = {"n_insufficient": len(insuff), "insufficient_systems": insuff,
                      "core4_all_insufficient": CORE_INSUFF.issubset(set(insuff))}
    sens["within_gate_threshold"] = wgs

    # ------- burden / high-risk share carried by the records-insufficient systems
    insuff_sys = set(gdf[gdf.S < 0.67].system)
    hi_units = units[units.R >= 0.90]
    burden_insuff = float(gdf[gdf.system.isin(insuff_sys)].burden_share.sum())
    context = {
        "n_insufficient_systems": len(insuff_sys),
        "insufficient_systems": sorted(insuff_sys),
        "reactive_labour_share_of_insufficient_systems": round(burden_insuff, 4),
        "share_of_high_risk_units_in_insufficient_systems":
            round(float(hi_units.system.isin(insuff_sys).mean()), 4),
        "share_of_all_units_in_insufficient_systems":
            round(float(units.system.isin(insuff_sys).mean()), 4),
    }

    summary = {
        "unit_of_analysis": "building-system-quarter (out-of-fold LOUO scored units)",
        "target": "severity_labour p75, layer M1, gbdt",
        "n_units": int(n),
        "n_positive": int(units.y.sum()),
        "base_rate": round(float(units.y.mean()), 4),
        "R_definition": "within-held-out-campus percentile rank of model score",
        "S_definition": "equal-weighted mean of 5 gate scores {0,0.5,1}, hard-fail capped",
        "class_thresholds": {"R_high": 0.90, "S_high": 0.67},
        "class_counts": {k: int(cls_counts.get(k, 0)) for k in CLASS_ACTION},
        "class_shares": {k: round(cls_counts.get(k, 0) / n, 4) for k in CLASS_ACTION},
        "class_actions": CLASS_ACTION,
        "high_CEPI_ge_0.45": {"n_units": hi_cepi, "share": round(hi_cepi / n, 4)},
        "n_systems": int(len(gdf)),
        "n_systems_fully_sufficient_S_ge_0.67": int((gdf.S >= 0.67).sum()),
        "n_systems_insufficient_S_lt_0.67": int((gdf.S < 0.67).sum()),
        "systems_lowest_S": gdf.sort_values("S").head(4)[["system", "system_desc", "S"]].to_dict("records"),
        "insufficient_systems_context": context,
        "sensitivity": sens,
    }

    json.dump({"gates": grows, "S_by_system": {r.system: round(r.S, 3) for _, r in gdf.iterrows()}},
              open(MET / "cepi_gates.json", "w"), indent=2, default=str)
    json.dump(summary, open(MET / "cepi_summary.json", "w"), indent=2, default=str)

    # ------- console report
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("\n=== GATE TABLE (pass=1 caution=.5 fail=0) ===")
    show = gtab.copy()
    for g in GATES:
        show[g] = show[g].map(score_label)
    print(show.to_string(index=False))
    print("\n=== S / G by system ===")
    print(gdf[["system", "system_desc", "S", "G"]].to_string(index=False))
    print("\n=== CEPI CLASS COUNTS (units) ===")
    for k in CLASS_ACTION:
        print(f"  {k:38s} {cls_counts.get(k,0):>8d}  ({cls_counts.get(k,0)/n:6.1%})")
    print(f"  high-CEPI (>=0.45): {hi_cepi} units ({hi_cepi/n:.1%})")
    print("\n=== PRIORITY MAP (system) ===")
    print(pm[["system_desc", "burden_share", "mean_R", "S", "G", "mean_CEPI",
              "frac_high_CEPI_0.45", "dominant_class"]].round(3).to_string(index=False))
    print("\n=== SENSITIVITY: gate weights -> # systems insufficient ===")
    for w, v in sens["gate_weights"].items():
        print(f"  {w:20s} {v['n_systems_insufficient']}")
    print("  hard-fail on/off:",
          sens["hard_fail"]["on"]["n_systems_insufficient"],
          sens["hard_fail"]["off"]["n_systems_insufficient"])
    print("\n=== WITHIN-GATE THRESHOLD SENSITIVITY (Appendix D) ===")
    for p, v in sens["within_gate_threshold"].items():
        print(f"  {p:26s} n_insuff={v['n_insufficient']}  core4_all_insuff={v['core4_all_insufficient']}  {v['insufficient_systems']}")
    print("\n=== INSUFFICIENT-SYSTEMS CONTEXT ===")
    print("  reactive-labour share:", context["reactive_labour_share_of_insufficient_systems"],
          "| high-risk-unit share:", context["share_of_high_risk_units_in_insufficient_systems"],
          "| all-unit share:", context["share_of_all_units_in_insufficient_systems"])
    print("\nWrote cepi_gates.csv, cepi_priority_map.csv, cepi_gates.json, cepi_summary.json")


if __name__ == "__main__":
    main()
