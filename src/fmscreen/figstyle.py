"""Shared publication-grade figure style for the manuscript figures.

A single, consistent, colour-blind-friendly palette and Matplotlib rcParams so
every figure in the paper looks like it came from the same hand. Import and call
``apply()`` at the top of a figure script; use ``despine(ax)`` on line/bar axes
(not on heatmaps) for the clean, open look.
"""
from __future__ import annotations
import glob
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

# Colour-blind-safe, journal-grade palette.
# Fills are deliberately LIGHT (pastel): large areas (bars, patches, spans) must
# not read as heavy ink blocks. The _D variants are medium-depth versions of the
# two lead hues, reserved for thin marks that would wash out in pastel: reference
# lines, line plots, markers, and emphasis text. Dark colours never fill an area.
NAVY   = "#a8c6e3"   # primary fill (light blue)
TEAL   = "#9fd6cc"   # secondary fill
AMBER  = "#f3cf8f"   # accent / taxonomy fill
CORAL  = "#eda9b0"   # alert fill (light rose)
SLATE  = "#7d8a97"   # neutral text / annotation grey (not a fill)
GREEN  = "#aed6b2"   # positive fill
PURPLE = "#c3b7d4"   # extra fill
LGREY  = "#c3cbd3"   # light neutral fill (use instead of SLATE for areas)
NAVY_D  = "#4f81ad"  # medium blue: lines, markers, emphasis text
CORAL_D = "#c25b6a"  # medium rose: reference lines, alert text
GREEN_D = "#66a173"  # medium green: lines and box edges
INK    = "#000000"   # text / axes: pure black, for print contrast
GRIDC  = "#c9ced4"   # gridlines
CYCLE  = [NAVY, AMBER, TEAL, CORAL, GREEN, PURPLE, LGREY]
# medium-depth cycle for LINE plots, where pastel strokes would wash out
LINE_CYCLE = [NAVY_D, "#d99a3d", "#4da395", CORAL_D, GREEN_D, "#8b7d9e", SLATE]

# The manuscript body is typeset in TeX Gyre Termes (the Times clone shipped with
# TeX; Times New Roman itself is proprietary and absent on Linux). Register the
# same OTF files with Matplotlib so the figures and the text use one identical
# face, and map mathtext onto it so no second family (STIX) leaks into the PDF.
_SERIF = "Liberation Serif"          # metric-identical Times fallback


def _register_termes() -> str:
    """Register TeX Gyre Termes with Matplotlib; return the family to request."""
    # Home-relative so the lookup works for whoever runs this, not only on the
    # machine the figures were first built on.
    pats = (os.path.expanduser("~/.cache/Tectonic/bundles/data/*/texgyretermes-*.otf"),
            "/usr/share/texmf*/fonts/opentype/public/tex-gyre/texgyretermes-*.otf",
            os.path.expanduser("~/.TinyTeX/texmf-dist/fonts/opentype/public/tex-gyre/texgyretermes-*.otf"))
    found = [f for p in pats for f in glob.glob(p)]
    for f in found:
        try:
            font_manager.fontManager.addfont(f)
        except Exception:
            pass
    if any("texgyretermes-regular" in f for f in found):
        return "TeX Gyre Termes"
    return _SERIF


def apply():
    fam = _register_termes()
    plt.rcParams.update({
        # One Times face for text AND math, identical to the manuscript body.
        "font.family": "serif",
        "font.serif": [fam, _SERIF, "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "custom",
        "mathtext.rm": fam,
        "mathtext.it": f"{fam}:italic",
        "mathtext.bf": f"{fam}:bold",
        "mathtext.default": "it",
        "pdf.fonttype": 42,          # embed TrueType (Type 42), not Type 3 (publisher checks)
        "ps.fonttype": 42,
        "font.size": 10,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.titlepad": 8,
        "axes.labelsize": 10,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.9,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRIDC,
        "grid.linewidth": 0.7,
        "grid.alpha": 0.9,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "text.color": INK,
        "legend.frameon": False,
        "legend.fancybox": False,   # hard-corner legend frames where a frame is used
        "legend.fontsize": 9,
        "figure.dpi": 120,
        "figure.facecolor": "white",
        "savefig.dpi": 320,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "axes.prop_cycle": mpl.cycler(color=CYCLE),
    })


def despine(ax, left=True, bottom=True):
    """Open the axis box (remove top/right spines) for the clean look."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    ax.tick_params(length=3, color="#9aa3ab")


def barlabels(ax, bars, fmt="{:.2f}", pad=3, fontsize=8.5, color=INK, tops=None):
    """Annotate bar ends with their values.

    `tops` gives the anchor coordinate to place each label against, overriding the
    bar end. Pass the upper error-bar caps when the bars carry yerr/xerr, otherwise
    the label is drawn at the bar end and collides with the whisker.

    Every label carries an opaque badge and sits above the reference lines. A bar
    whose value lands near a reference line puts its label on that line, and a
    dashed rule drawn through the digits is unreadable: in the co-escalation
    robustness panel the pooled rule ran straight through "2.11".
    """
    badge = dict(boxstyle="square,pad=0.10", fc="white", ec="none", alpha=1.0)
    for i, b in enumerate(bars):
        w = b.get_width(); h = b.get_height()
        anchor = None if tops is None else float(tops[i])
        if abs(w) >= abs(h):   # horizontal bars
            ax.annotate(fmt.format(w), (w if anchor is None else anchor, b.get_y() + h / 2),
                        xytext=(pad, 0), textcoords="offset points",
                        va="center", ha="left", fontsize=fontsize, color=color,
                        zorder=6, bbox=badge)
        else:                   # vertical bars
            ax.annotate(fmt.format(h), (b.get_x() + w / 2, h if anchor is None else anchor),
                        xytext=(0, pad), textcoords="offset points",
                        va="bottom", ha="center", fontsize=fontsize, color=color,
                        zorder=6, bbox=badge)
