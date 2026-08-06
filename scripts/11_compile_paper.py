#!/usr/bin/env python
"""
Phase 7 — compile every metric file into paper-ready artefacts.

Produces:
  paper/results_summary.md     — narrative results organised by RQ + abstract sentence
  paper/tables/*.tex           — LaTeX tables for the manuscript
  paper/key_numbers.json       — every headline number in one machine-readable file

Robust to missing inputs (skips sections whose metric files are absent).
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fmscreen import config as C

PAPER = C.ROOT / "paper"; (PAPER / "tables").mkdir(parents=True, exist_ok=True)


def _load_json(p):
    return json.load(open(p)) if Path(p).exists() else None


def _load_csv(p):
    return pd.read_csv(p) if Path(p).exists() else None


def main():
    K = {}  # key numbers
    L = []   # markdown lines
    A = L.append

    phase0 = _load_json(C.PHASE0_DIR / "summary.json")
    base = _load_json(C.PHASE0_DIR / "panel_base_rates.json")
    head = _load_json(C.METRICS / "louo_primary_headline.json")
    temporal = _load_csv(C.METRICS / "temporal_results.csv")
    tabl = _load_json(C.METRICS / "temporal_ablation.json")
    louo = _load_csv(C.METRICS / "louo_folds.csv")
    louo_sum = _load_csv(C.METRICS / "louo_summary.csv")
    decomp = _load_json(C.METRICS / "decomposition_summary.json")
    het = _load_csv(C.METRICS / "heterogeneity_by_system.csv")
    deploy = _load_json(C.METRICS / "deployment_matrix.json")
    calib = _load_csv(C.METRICS / "calibration_summary.csv")
    cost = _load_json(C.METRICS / "cost_summary.json")
    burden = _load_json(C.METRICS / "burden_descriptive.json")

    A("# Results summary — CMMS-only reactive-maintenance screening\n")
    A("_Auto-compiled from `results/metrics/`. Numbers are leakage-controlled and, where "
      "applicable, carry clustered bootstrap CIs or across-campus IQRs._\n")

    # ---------- Dataset ----------
    if phase0 and base:
        b = phase0["basic"]
        K["dataset"] = {"n_records": b["n_records"], "n_universities": b["n_universities"],
                        "n_ppm": b["n_PPM"], "n_upm": b["n_UPM"],
                        "panel_rows": base["panel_rows_known_system"],
                        "panel_universities": base["n_universities"],
                        "panel_cells": base["n_cells"],
                        "occurrence_base_rate": base["occurrence_base_rate_known_system"],
                        "severity_p75_base_rate": base["severity_base_rate_known_system_p75"]}
        A("## Dataset & panel\n")
        A(f"- FMUCD: **{b['n_records']:,}** work orders, **{b['n_universities']}** universities "
          f"({b['n_PPM']:,} PPM / {b['n_UPM']:,} UPM). Building×system panel built on the "
          f"**{base['n_universities']}** universities with BuildingID "
          f"(others record no building; reported in attribution).")
        A(f"- Panel: **{base['panel_rows_known_system']:,}** building×system×quarter rows, "
          f"**{base['n_cells']:,}** cells, **{base['n_systems']}** systems, 2002–2021.")
        A(f"- Base rates: occurrence **{base['occurrence_base_rate_known_system']:.1%}**, "
          f"severity-p75 **{base['severity_base_rate_known_system_p75']:.1%}** "
          f"(p50 {base['severity_base_rate_known_system_p50']:.1%}, "
          f"p90 {base['severity_base_rate_known_system_p90']:.1%}).\n")

    # ---------- RQ1: sufficiency (within-sample) + ablation ----------
    if temporal is not None and tabl:
        A("## RQ1 — Sufficiency (within-sample temporal validation) & RQ1c ablation\n")
        g = tabl["gbdt"]
        K["temporal_gbdt"] = g
        A(f"- GBDT severity-p75 (test 2019–2021): **lift@10 = {g['M1_lift']:.2f}** "
          f"(M0 {g['M0_lift']:.2f}); best simple rule {g['best_rule_lift']:.2f}; "
          f"M1 beats rule = **{g['M1_beats_rule']}**.")
        A(f"- **Reactive-burden gain** (M1−M0) = +{g['M1_minus_M0 (reactive-burden gain)']:.2f} lift; "
          f"**taxonomy gain** (M1−M1_nosys) = {g['M1_minus_M1nosys (taxonomy gain)']:+.2f} "
          f"(system code is largely redundant *within* a campus).\n")

    # ---------- RQ2a: LOUO transfer + sufficiency ----------
    if head:
        A("## RQ2a — Cross-campus transfer (leave-one-university-out) & sufficiency verdict\n")
        g = head.get("gbdt_M1", {})
        K["louo_primary_gbdt_M1"] = g
        A(f"- **PRIMARY ANALYSIS (severity-p75, known-system, LOUO, top-10%): GBDT M1 "
          f"median lift = {g.get('lift_median', float('nan')):.2f}×** "
          f"(IQR {g.get('lift_iqr',[0,0])[0]:.2f}–{g.get('lift_iqr',[0,0])[1]:.2f}, "
          f"range {g.get('lift_min',float('nan')):.2f}–{g.get('lift_max',float('nan')):.2f}), "
          f"capture {g.get('capture_median',float('nan')):.0%}.")
        A(f"- Meets 2× in **{g.get('frac_folds_meet_2x',0):.0%}** of campuses; beats best rule in "
          f"**{g.get('frac_folds_beat_rule',0):.0%}**; **MEETS_SUFFICIENCY = "
          f"{g.get('MEETS_SUFFICIENCY')}**.")
        for m in ("gbdt_M0", "logreg_M1"):
            if m in head:
                h = head[m]
                A(f"- {m}: median lift {h['lift_median']:.2f}, beats-rule "
                  f"{h['frac_folds_beat_rule']:.0%}, sufficient={h['MEETS_SUFFICIENCY']}.")
        A("")

    # ---------- RQ2b: taxonomy contribution in transfer ----------
    if louo is not None:
        sub = louo[(louo.target_kind == "severity_labour") & (louo.pctl == 75)]
        tax = {}
        for model in ("gbdt", "logreg"):
            m1 = sub[(sub.model == model) & (sub.layer == "M1")].set_index("held_out_university")["lift_top10"]
            mn = sub[(sub.model == model) & (sub.layer == "M1_nosys")].set_index("held_out_university")["lift_top10"]
            common = m1.index.intersection(mn.index)
            if len(common):
                tax[model] = float((m1[common] - mn[common]).median())
        K["louo_taxonomy_gain_median"] = tax
        if tax:
            A("## RQ2b — Standardized-taxonomy contribution to transfer\n")
            A(f"- Median LOUO lift gain from including the standardized system code "
              f"(M1 − M1_nosys): " + ", ".join(f"{m}={v:+.2f}" for m, v in tax.items()) + ".")
            A("- Interpretation: the FMUCO taxonomy's primary value is enabling a comparable "
              "cross-institutional unit of analysis (the panel is built on standardized codes); "
              "as an extra predictive feature its marginal lift is small because cell-level "
              "history already carries the signal.\n")

    # ---------- RQ2c: calibration curve ----------
    if calib is not None:
        sev = calib[(calib.target_kind == "severity_labour") & (calib.model == "gbdt")].sort_values("frac")
        if len(sev):
            l0 = sev[sev.frac == 0.0]["lift_median"].iloc[0] if (sev.frac == 0.0).any() else np.nan
            l20 = sev[sev.frac == 0.20]["lift_median"].iloc[0] if (sev.frac == 0.20).any() else np.nan
            K["calibration"] = {"lift_frac0": float(l0), "lift_frac20": float(l20),
                                "delta": float(l20 - l0)}
            A("## RQ2c — Local-history calibration curve (primary novelty C2)\n")
            A(f"- GBDT severity-p75 median lift: frac0 = {l0:.2f} → frac0.20 = {l20:.2f} "
              f"(**Δ = {l20-l0:+.2f}**).")
            A("- The curve is **near-flat**: because standardized-coded records transfer "
              "strongly zero-shot, a new campus is screenable with **near-zero local history**; "
              "small additions of local data give marginal gains.\n")

    # ---------- RQ3a: occurrence vs severity ----------
    if decomp:
        A("## RQ3a — Occurrence-vs-severity decomposition\n")
        K["decomposition"] = decomp
        for t in ("occurrence", "severity_p50", "severity_p75", "severity_p90"):
            if t in decomp:
                d = decomp[t]
                A(f"- {t}: base {d['base_rate_median']:.1%}, lift {d['lift_median']:.2f}, "
                  f"capture {d['capture_median']:.0%}, **PR-AUC {d['prauc_median']:.3f}**, "
                  f"ROC-AUC {d['rocauc_median']:.3f}.")
        A("- **Records predict WHETHER UPM occurs strongly (occurrence PR-AUC "
          f"{decomp['occurrence']['prauc_median']:.2f}) but the severe tail has a lower "
          f"information ceiling (severity-p90 PR-AUC {decomp['severity_p90']['prauc_median']:.2f}).** "
          "This is the C3 finding; candidate causes (intrinsic variability, missing severity "
          "drivers, inconsistent costing, absent condition data) are not disentangled.\n")

    # ---------- RQ3b/3c: heterogeneity + deployment ----------
    if het is not None and deploy:
        A("## RQ3b/3c — System heterogeneity & deployment boundary\n")
        weak = het[het["frac_campuses_meet_2x"].fillna(0) < 0.8]
        K["weak_systems"] = weak["system"].tolist()
        A(f"- Per-system LOUO lift ranges **{het['lift_top10'].min():.2f}–{het['lift_top10'].max():.2f}** "
          f"(severity-p75, GBDT, pooled).")
        A(f"- Robustly screenable systems clear 2× in ≥80% of campuses; "
          f"**weak/unstable strata: {', '.join(weak['system'].tolist()) or 'none'}** "
          "(candidates for richer information).")
        for cell, info in deploy.items():
            A(f"  - {cell}: {info['n']} systems → _{info['action']}_")
        A("")

    # ---------- cost (secondary) ----------
    if cost:
        A("## Secondary — high-cost target\n")
        K["cost"] = cost
        for m, v in cost.items():
            A(f"- {m}: median lift {v['lift_median']:.2f}, meets-2× {v['frac_folds_meet_2x']:.0%}, "
              f"beats-rule {v['frac_folds_beat_rule']:.0%} "
              f"({v['n_campuses_evaluable']} campuses, median cost-completeness {v['median_cost_pos_share']:.0%}).")
        A("")

    # ---------- robustness: recent-activity, age, text ----------
    recent = _load_json(C.METRICS / "recent_activity_summary.json")
    age = _load_json(C.METRICS / "age_heterogeneity.json")
    m2 = _load_json(C.METRICS / "m2_summary.json")
    if recent or age or m2:
        A("## Robustness & sensitivity\n")
    if recent:
        rk = recent.get("severity_labour_p75_gbdt_M1", {})
        K["recent_activity"] = rk
        A(f"- **Recent-activity rule** (harder active-cell subset, base "
          f"{rk.get('base_rate_median',0):.1%}): GBDT M1 lift {rk.get('lift_median',0):.2f}, "
          f"meets-2× {rk.get('frac_meet_2x',0):.0%}, beats-rule {rk.get('frac_beat_rule',0):.0%} "
          "— the signal survives among active cells, not merely dormant-vs-active.")
    if age:
        K["age"] = age
        A(f"- **Building-age heterogeneity** ({age.get('built_year_coverage_in_panel',0):.0%} "
          "metadata coverage): older buildings carry higher burden but screen slightly worse; "
          "all age bands remain screenable (see `age_heterogeneity.csv`).")
    if m2:
        K["m2"] = m2
        d = m2.get("m2_minus_m1_louo_lift_median", 0.0)
        A(f"- **Work-order text (M2)** adds ~0 over M1 (LOUO Δlift = {d:+.2f}); structured "
          "counts/burden/system features already capture the signal — text is descriptively "
          "useful but not necessary for screening.")
        A("")

    # ---------- premiums ----------
    if burden:
        A("## Reactive-burden premiums (Section 14) & burden concentration (RQ1a)\n")
        A(f"- Median labour premium across systems = {burden['median_lab_premium_across_systems']:.2f} "
          f"({burden['systems_with_lab_premium_gt_1']} systems > 1×).")
        tops = ", ".join(f"{t['SystemDescription']} {t['share_of_total_upm_labour']:.0%}"
                         for t in burden["top5_systems_by_upm_labour"][:3])
        A(f"- Reactive labour concentrates in {tops}.\n")

    # ---------- abstract sentence ----------
    if head and decomp:
        g = head["gbdt_M1"]
        A("## Abstract-ready sentence\n")
        A(f"> Inspecting the top 10% highest-risk building-system-quarter cells, record-only "
          f"screening captures a median **{g['capture_median']:.0%}** of next-quarter high-labour "
          f"reactive events across held-out campuses — a median **{g['lift_median']:.1f}× lift** over "
          f"base rate (≥2× in {g['frac_folds_meet_2x']:.0%} of campuses, beating the best simple "
          f"rule in {g['frac_folds_beat_rule']:.0%}) — while the severe-tail information ceiling "
          f"(occurrence PR-AUC {decomp['occurrence']['prauc_median']:.2f} vs severity-p90 "
          f"{decomp['severity_p90']['prauc_median']:.2f}) bounds how finely severity can be ranked.\n")

    (PAPER / "results_summary.md").write_text("\n".join(L))
    json.dump(K, open(PAPER / "key_numbers.json", "w"), indent=2)

    # ---------- LaTeX tables ----------
    _latex_tables(temporal, louo_sum, het, decomp)
    print("[compile] wrote paper/results_summary.md, paper/key_numbers.json, paper/tables/*.tex")
    print("\n".join(L[:60]))


def _latex_tables(temporal, louo_sum, het, decomp):
    if louo_sum is not None:
        t = louo_sum[(louo_sum.target_kind == "severity_labour") & (louo_sum.pctl == 75)][
            ["model", "layer", "lift_median", "lift_q1", "lift_q3", "capture_median",
             "frac_folds_meet_2x", "frac_folds_beat_rule", "MEETS_SUFFICIENCY"]]
        t.to_latex(PAPER / "tables" / "louo_severity_p75.tex", index=False, float_format="%.2f",
                   caption="Leave-one-university-out transfer, severity p75 (top-10\\% lift).",
                   label="tab:louo")
    if decomp:
        rows = [{"target": k, **v} for k, v in decomp.items()]
        pd.DataFrame(rows)[["target", "base_rate_median", "lift_median", "capture_median",
                            "prauc_median", "rocauc_median"]].to_latex(
            PAPER / "tables" / "decomposition.tex", index=False, float_format="%.3f",
            caption="Occurrence-vs-severity decomposition (median across campuses).",
            label="tab:decomp")
    if het is not None:
        het[["system", "n_cells", "base_rate", "lift_top10", "capture_top10",
             "frac_campuses_meet_2x"]].to_latex(
            PAPER / "tables" / "heterogeneity.tex", index=False, float_format="%.3f",
            caption="System-level heterogeneity (severity p75, GBDT).", label="tab:het")


if __name__ == "__main__":
    main()
