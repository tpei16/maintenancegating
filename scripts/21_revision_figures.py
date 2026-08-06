#!/usr/bin/env python
"""Revision figures: CONSORT flow, Lorenz, persistence, co-escalation heatmap,
budget curve, trajectory decomposition, pre-COVID ablation. Saves PNG+PDF."""
from __future__ import annotations
import sys, json
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fmscreen import config as C
from fmscreen import figstyle

figstyle.apply()
FIG = C.FIGURES
NAVY, TEAL, AMBER = figstyle.NAVY, figstyle.TEAL, figstyle.AMBER
CORAL, SLATE, GREEN = figstyle.CORAL, figstyle.SLATE, figstyle.GREEN
NAVY_D, CORAL_D = figstyle.NAVY_D, figstyle.CORAL_D
PURPLE, INK = figstyle.PURPLE, figstyle.INK
BLUE, RED = NAVY, CORAL   # back-compat aliases


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=300); fig.savefig(FIG / f"{name}.pdf"); plt.close(fig)
    print("[fig]", name, flush=True)


def fig_lorenz():
    d = pd.read_csv(C.TABLES / "lorenz_curve_cells.csv")
    bs = json.load(open(C.METRICS / "burden_structure.json"))
    bsys = pd.read_csv(C.TABLES / "burden_by_system.csv").head(8).iloc[::-1]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.3, 3.1))
    a1.plot(d["pop_frac"] * 100, d["labour_frac"] * 100, color=NAVY_D, lw=2.4, zorder=3)
    a1.plot([0, 100], [0, 100], ls="--", color=SLATE, lw=1.2, label="equality")
    a1.fill_between(d["pop_frac"] * 100, d["labour_frac"] * 100, d["pop_frac"] * 100,
                    color=NAVY, alpha=0.45, zorder=2)
    a1.set_xlabel("Cumulative share of building-system cells (%)")
    a1.set_ylabel("Cumulative share of reactive labor (%)")
    a1.set_xlim(0, 100); a1.set_ylim(0, 100)
    # anchor annotation to the actual curve point near x=90
    idx90 = (d["pop_frac"] - 0.90).abs().idxmin()
    x90 = d.loc[idx90, "pop_frac"] * 100
    y90 = d.loc[idx90, "labour_frac"] * 100
    a1.scatter([x90], [y90], s=28, color=NAVY_D, zorder=4)
    a1.annotate(f"top decile of cells\n= {bs['cell_top10_share']:.0%} of labor",
                xy=(x90, y90), xytext=(32, 56), fontsize=9,
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    a1.legend(frameon=False, loc="upper left")
    bars = a2.barh(bsys[C.COL_SYSTEM_DESC].astype(str),
                   bsys["share_of_total_upm_labour"] * 100, color=NAVY)
    a2.set_xlabel("Share of total reactive labor (%)")
    a2.set_xlim(0, max(bsys["share_of_total_upm_labour"] * 100) * 1.18)
    figstyle.barlabels(a2, bars, fmt="{:.0f}%")
    figstyle.despine(a1); figstyle.despine(a2)
    fig.tight_layout()
    save(fig, "fig_burden_structure")


def fig_persistence():
    bs = json.load(open(C.METRICS / "burden_structure.json"))["persistence"]
    pers = pd.read_csv(C.TABLES / "persistence_by_system.csv")
    ks = [1, 2, 4, 8]
    xpos = np.arange(len(ks))   # ordinal x positions (1,2,4,8 are not linear)
    markers = ["o", "s", "^", "D", "v"]
    fig, ax = plt.subplots(figsize=(6.3, 4.2))
    # all-systems series (heavier)
    ax.plot(xpos, [bs[f"persist_t{k}"] * 100 for k in ks], marker=markers[0], color=NAVY_D,
            lw=2.8, ms=7, label="all systems", zorder=4)
    for i, (_, r) in enumerate(pers.head(4).iterrows()):
        c = figstyle.LINE_CYCLE[(i + 1) % len(figstyle.LINE_CYCLE)]
        ax.plot(xpos, [r[f"persist_t{k}"] * 100 for k in ks], marker=markers[(i + 1) % len(markers)],
                color=c, ls="--", lw=1.8, ms=6, alpha=0.95, label=r["system_desc"])
    ax.set_xlabel("Quarters after a high-burden event, t+k  (1, 2, 4, 8; ordinal)")
    ax.set_ylabel("P(high-burden at t+k | high-burden now) (%)")
    ax.set_xticks(xpos); ax.set_xticklabels([str(k) for k in ks])
    ax.set_ylim(44, 66)
    # The unconditional rate is ~15%, far below this axis. A reference line at
    # 45.2 was previously drawn here with the 15% label sitting just above it,
    # which invited the reader to take the line FOR the 15% rate. It marked
    # nothing, so it is gone; the note alone carries the comparison.
    ax.text(len(ks) - 1, 44.6,
            f"unconditional high-burden rate $\\approx$ {bs['marginal_severe_rate_active']:.0%}"
            ", far below this axis",
            ha="right", va="bottom", fontsize=8.6, color=INK, style="italic")
    ax.set_xlim(-0.25, len(ks) - 0.75)
    figstyle.despine(ax)
    ax.legend(frameon=False, fontsize=8.8, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    save(fig, "fig_persistence")


def fig_coescalation():
    m = pd.read_csv(C.TABLES / "coescalation_matrix.csv")
    # short labels
    SHORT = {"HVAC": "HVAC", "Plumbing": "Plumb", "Electrical": "Elec", "Interior Construction": "IntCon",
             "Interior Finishes": "IntFin", "Equipment": "Equip", "Furnishings": "Furn",
             "Fire Protection": "Fire", "Exterior Enclosure": "ExtEnc", "Roofing": "Roof",
             "Superstructure": "Super", "Conveying": "Conv", "Stairs": "Stair", "Site Improvements": "SiteImp",
             "Special Construction": "SpecCon", "Site Mechanical Utilities": "SiteMech",
             "Site Electrical Utilities": "SiteElec", "Foundations": "Found", "Site Preparation": "SitePrep",
             "Selective Building Demolition": "Demo"}
    m["As"] = m["A_desc"].map(lambda s: SHORT.get(str(s).strip(), str(s)[:6]))
    m["Bs"] = m["B_desc"].map(lambda s: SHORT.get(str(s).strip(), str(s)[:6]))
    piv = m.pivot_table(index="As", columns="Bs", values="risk_ratio", aggfunc="mean")
    # Order by total burden, but keep EVERY system that appears in the matrix.
    # A previously hardcoded list of 16 short names silently dropped four
    # systems and with them 41 of the 238 measured pairs, while the caption
    # described the figure as if it showed them all.
    order = ["HVAC", "Plumb", "Elec", "IntCon", "IntFin", "Equip", "Fire", "ExtEnc", "Roof", "Furn",
             "Conv", "Stair", "SiteImp", "SpecCon", "SiteMech", "Super"]
    present = list(piv.index) + list(piv.columns)
    order = [o for o in order if o in present]
    order += sorted(set(present) - set(order))          # anything not listed above
    piv = piv.reindex(index=[o for o in order if o in piv.index],
                      columns=[o for o in order if o in piv.columns])
    # 17 of the 238 pairs have no risk ratio at all: the follower system never
    # reaches high burden anywhere in the panel, so the ratio is 0/0. Those are
    # genuinely unmeasurable and stay blank; every pair that HAS a value is now
    # on the page (221, up from 197 before the ordering was taken from the data).
    n_measurable = int(np.isfinite(m["risk_ratio"]).sum())
    shown = int(piv.notna().sum().sum())
    assert shown == n_measurable, \
        f"heatmap shows {shown} of {n_measurable} measurable pairs"
    fig, ax = plt.subplots(figsize=(6.3, 5.5))
    vals = piv.values.astype(float)
    masked = np.ma.masked_invalid(vals)
    vmax = float(np.ceil(np.nanmax(vals) * 10) / 10)   # true rounded max (~5.6)
    norm = mcolors.TwoSlopeNorm(vmin=0.5, vcenter=1.0, vmax=vmax)
    # truncate coolwarm so the extremes stay medium-depth (no heavy ink blocks)
    base = plt.get_cmap("coolwarm")
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "coolwarm_soft", base(np.linspace(0.22, 0.80, 256)))
    cmap.set_bad("#eaecee")   # light-grey for empty cells
    im = ax.imshow(masked, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index, fontsize=9)
    for i in range(len(piv.index)):
        for j in range(len(piv.columns)):
            v = vals[i, j]
            if not np.isnan(v):
                # adaptive text colour from the rendered cell luminance
                r, g, b, _ = cmap(norm(v))
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7.2,
                        color="white" if lum < 0.5 else INK)
    ax.set_xlabel("Follower system (high-burden within t+1..t+4)"); ax.set_ylabel("Trigger system (high-burden at t)")
    fig.colorbar(im, ax=ax, label="risk ratio vs population baseline", shrink=0.8)
    ax.grid(False)
    save(fig, "fig_coescalation_heatmap")


def fig_budget():
    """Two stacked panels on one budget axis (no dual y-axis): precision lift on
    top, capture below. The top-10% operating point is marked in both."""
    d = pd.read_csv(C.METRICS / "budget_curve.csv")
    agg = d.groupby("budget").agg(lift=("lift", "median"), cap=("capture", "median"),
                                  lift_q1=("lift", lambda s: s.quantile(.25)),
                                  lift_q3=("lift", lambda s: s.quantile(.75)),
                                  cap_q1=("capture", lambda s: s.quantile(.25)),
                                  cap_q3=("capture", lambda s: s.quantile(.75))).reset_index()
    x = agg["budget"] * 100
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.3, 5.4), sharex=True,
                                 gridspec_kw=dict(hspace=0.14))

    a1.plot(x, agg["lift"], "o-", color=NAVY_D, lw=2.4, ms=6.5,
            label="median across held-out campuses", zorder=4)
    a1.fill_between(x, agg["lift_q1"], agg["lift_q3"], color=NAVY, alpha=0.5,
                    label="interquartile range across campuses", lw=0)
    a1.axhline(2.0, ls="--", color=SLATE, lw=1.4, label="Two-fold bar")
    lift10 = float(agg.loc[agg["budget"] == 0.10, "lift"].iloc[0])
    # placed below-left of the marked point: to the upper right it collided
    # with the legend's "interquartile range across campuses" entry
    a1.annotate(f"top 10%: lift $\\approx$ {lift10:.1f}", xy=(10, lift10),
                xytext=(11.0, lift10 - 2.0), ha="left", fontsize=8.8, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.1))
    a1.set_xticks([5, 10, 15, 20])
    a1.set_ylabel("Precision lift over base rate")
    a1.set_ylim(0, None)
    a1.legend(frameon=False, loc="upper right", fontsize=8.6)
    figstyle.despine(a1)

    a2.plot(x, agg["cap"] * 100, "o-", color=NAVY_D, lw=2.4, ms=6.5, zorder=4)
    a2.fill_between(x, agg["cap_q1"] * 100, agg["cap_q3"] * 100, color=NAVY,
                    alpha=0.15, lw=0)
    cap10 = float(agg.loc[agg["budget"] == 0.10, "cap"].iloc[0]) * 100
    a2.annotate(f"top 10%: capture $\\approx$ {cap10:.0f}%", xy=(10, cap10),
                xytext=(11.5, cap10 - 18), ha="left", fontsize=8.8, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.1))
    a2.set_ylabel("High-burden events\ncaptured (%)")
    a2.set_ylim(0, 100)
    a2.set_xlabel("Screening capacity (top-k % of units)")
    figstyle.despine(a2)

    fig.tight_layout()
    save(fig, "fig_budget_curve")


def fig_trajectory():
    t = json.load(open(C.METRICS / "trajectory.json"))
    # "zero prior activity" is a SUBSET of "non-escalating" (no prior work orders
    # implies a flat back-trend), so plotting the three raw shares side by side
    # summed to 103.3% for events and 128.2% for controls and read as a
    # partition that it was not. Subtract it out to get three exclusive classes,
    # which is also how 29_missed_p90.py counts them, so the two figures now
    # use one definition of these category names.
    def split(d):
        return [d["escalating"] * 100,
                (d["non_escalating"] - d["zero_prior_activity"]) * 100,
                d["zero_prior_activity"] * 100]

    labels = ["escalating\n(rising prior)", "non-escalating\n(flat/declining)",
              "zero prior\nactivity"]
    ev, ct = split(t["events"]), split(t["controls_active"])
    for name, v in (("events", ev), ("controls", ct)):
        assert abs(sum(v) - 100.0) < 0.15, f"{name} classes sum to {sum(v):.1f}%"
    cats = labels
    x = np.arange(len(cats)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.3, 4.0))
    b1 = ax.bar(x - w / 2, ev, w, color=CORAL,
                label=f"p90 extreme events (n={t['n_p90_events_with_history']:,})")
    b2 = ax.bar(x + w / 2, ct, w, color=TEAL, label="matched controls (active)")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Share of group (%)")
    ax.set_ylim(0, max(ev + ct) * 1.16)
    figstyle.barlabels(ax, b1, fmt="{:.1f}%"); figstyle.barlabels(ax, b2, fmt="{:.1f}%")
    figstyle.despine(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    save(fig, "fig_trajectory")


def fig_ablation_precovid():
    d = pd.read_csv(C.METRICS / "temporal_precovid.csv")
    cov = json.load(open(C.METRICS / "covid_sensitivity.json"))
    rule = cov["test_2017_2019"]["best_rule_lift"]
    fig, ax = plt.subplots(figsize=(6.3, 4.0))
    layers = ["M0", "M1"]; x = np.arange(len(layers)); w = 0.35
    model_colors = {"gbdt": NAVY, "logreg": AMBER}
    for i, model in enumerate(["gbdt", "logreg"]):
        vals = [d[(d.model == model) & (d.layer == l)]["lift_top10"].iloc[0] for l in layers]
        MODEL_NAME = {"gbdt": "Gradient-boosted trees", "logreg": "Logistic regression"}
        bars = ax.bar(x + i * w, vals, w, color=model_colors[model],
                      label=MODEL_NAME.get(model, model.upper()))
        figstyle.barlabels(ax, bars, fmt="{:.2f}")
    l_rule = ax.axhline(rule, ls=":", color=CORAL_D, lw=1.8,
                        label=f"best simple rule (lift = {rule:.2f})")
    l_bar = ax.axhline(2.0, ls="--", color=SLATE, lw=1.4, label="2× bar")
    ax.set_xticks(x + w / 2); ax.set_xticklabels(["M0\n(history only)", "M1\n(+ reactive burden)"])
    ax.set_ylabel("Top-10% precision lift")
    ax.set_ylim(0, max(rule, d["lift_top10"].max()) * 1.18)
    figstyle.despine(ax)
    # two-column legend: models (bars) separated from threshold lines
    handles, labs = ax.get_legend_handles_labels()
    ax.legend(handles, labs, frameon=False, fontsize=8.6, ncol=2, loc="upper left")
    fig.tight_layout()
    save(fig, "fig_ablation_precovid")


def main():
    for fn in (fig_lorenz, fig_persistence, fig_coescalation,
               fig_budget, fig_trajectory, fig_ablation_precovid):
        try:
            fn()
        except Exception as e:
            import traceback; print(f"[fig] {fn.__name__} FAILED: {e}"); traceback.print_exc()
    print("[fig] revision figures done")


if __name__ == "__main__":
    main()
