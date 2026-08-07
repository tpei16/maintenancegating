#!/usr/bin/env python
"""check_figure_overlap.py -- read the rendered figure PDFs and report overlaps.

Two kinds, because the earlier text-only sweep passed a figure whose threshold
rule was drawn straight through a data label:

  text / text    two words printed over each other
  text / stroke  a rule, leader or marker outline crossing the middle of a word

Gridlines (the house tint #c9ced4 and anything lighter) are excluded: they are
meant to sit behind everything.  Rotated text is grouped by its own baseline
direction, since characters in a 45-degree tick label overlap in page
coordinates while reading perfectly.

Usage:  python scripts/check_figure_overlap.py [pdf ...]
Exits non-zero if any figure has a collision.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pdfplumber

ROOT = Path(__file__).resolve().parents[1]
PAD = 0.4                 # points of slack
LIGHT = 0.74              # strokes at or above this mean brightness are background


def _upright(ch) -> bool:
    """True only for unrotated text.  The `upright` flag is set for a 45-degree
    label as well, so test the text matrix's shear terms directly."""
    if not ch.get("upright", True):
        return False
    m = ch.get("matrix")
    if m and len(m) >= 4:
        return abs(m[1]) < 1e-6 and abs(m[2]) < 1e-6
    return True


def words(page) -> list[dict]:
    out = []
    for w in page.extract_words(use_text_flow=False, keep_blank_chars=False,
                                extra_attrs=["matrix", "upright"]):
        if not str(w["text"]).strip():
            continue
        out.append(w)
    return out


def _brightness(c) -> float:
    if c is None:
        return 0.0
    if isinstance(c, (int, float)):
        return float(c)
    try:
        v = [float(x) for x in c]
    except TypeError:
        return 0.0
    if len(v) == 1:
        return v[0]
    if len(v) == 3:
        return sum(v) / 3.0
    if len(v) == 4:                      # CMYK
        c_, m_, y_, k_ = v
        return ((1 - c_) * (1 - k_) + (1 - m_) * (1 - k_) + (1 - y_) * (1 - k_)) / 3
    return sum(v) / len(v)


def strokes(page) -> list[dict]:
    out = []
    for obj in list(page.lines) + list(page.curves) + list(page.rects):
        if not obj.get("stroke"):
            continue
        if _brightness(obj.get("stroking_color")) >= LIGHT:
            continue                     # gridline / background rule
        out.append(obj)
    return out


MIN_AREA = 0.25   # fraction of the smaller glyph box that must be covered


def _overlap(a, b) -> bool:
    """Real collision, not kerning.

    In Times the boxes of "Va", "To", "Wo" and "fo" legitimately interpenetrate
    by a sliver, so an area threshold is needed: two labels printed over each
    other cover most of one another, a kerned pair covers a few percent.
    """
    ix = min(a["x1"], b["x1"]) - max(a["x0"], b["x0"])
    iy = min(a["bottom"], b["bottom"]) - max(a["top"], b["top"])
    if ix <= PAD or iy <= PAD:
        return False
    small = min((a["x1"] - a["x0"]) * (a["bottom"] - a["top"]),
                (b["x1"] - b["x0"]) * (b["bottom"] - b["top"]))
    return small > 0 and (ix * iy) / small >= MIN_AREA


def _segments(obj, page_h) -> list[tuple]:
    """The stroke itself, as segments.

    A rect or a marker outline is a frame, not a filled area: its bounding box
    contains the text it encloses, so testing the box would flag every label
    inside a heatmap cell. Walk the actual path instead.
    """
    pts = obj.get("pts")
    if not pts:
        return [((obj["x0"], obj["top"]), (obj["x1"], obj["bottom"]))]
    p = [(float(x), page_h - float(y)) for x, y in pts]
    segs = [(p[i], p[i + 1]) for i in range(len(p) - 1)]
    if obj["object_type"] in ("rect", "curve") and len(p) > 2:
        segs.append((p[-1], p[0]))
    return segs


def _seg_hits(p0, p1, w) -> bool:
    """Liang-Barsky against the word box, inset so a stroke that merely grazes
    a glyph's outer edge is not reported as striking through it."""
    ix = (w["x1"] - w["x0"]) * 0.12
    iy = (w["bottom"] - w["top"]) * 0.18
    xmin, xmax = w["x0"] + ix, w["x1"] - ix
    ymin, ymax = w["top"] + iy, w["bottom"] - iy
    if xmin >= xmax or ymin >= ymax:
        return False
    (x0, y0), (x1, y1) = p0, p1
    dx, dy = x1 - x0, y1 - y0
    t0, t1 = 0.0, 1.0
    for pp, qq in ((-dx, x0 - xmin), (dx, xmax - x0),
                   (-dy, y0 - ymin), (dy, ymax - y0)):
        if pp == 0:
            if qq < 0:
                return False
        else:
            r = qq / pp
            if pp < 0:
                if r > t1:
                    return False
                t0 = max(t0, r)
            else:
                if r < t0:
                    return False
                t1 = min(t1, r)
    return t0 < t1


def check(pdf_path: Path, do_strokes: bool = False) -> list[str]:
    """Text-on-text is decided here; text-on-stroke is only advisory.

    pdfplumber reports the graphics state it can see, and matplotlib sets stroke
    colour and alpha in states it does not track, so a pale hatch or a clip path
    arrives looking like a solid black rule. The authoritative stroke check runs
    inside matplotlib, where zorder and alpha are known: see scripts/figcheck.py,
    called from the figure functions themselves.
    """
    bad = []
    with pdfplumber.open(pdf_path) as pdf:
        for pno, page in enumerate(pdf.pages):
            ws = words(page)
            up = [w for w in ws if all(_upright(c) for c in [w])
                  or _upright(w)]
            for i in range(len(up)):
                for j in range(i + 1, len(up)):
                    if _overlap(up[i], up[j]):
                        bad.append(f"p{pno+1} text/text: "
                                   f"{up[i]['text']!r} <-> {up[j]['text']!r}")
            for st in (strokes(page) if do_strokes else []):
                segs = _segments(st, page.height)
                for w in up:
                    if any(_seg_hits(a, b, w) for a, b in segs):
                        bad.append(f"p{pno+1} text/stroke: {w['text']!r} crossed by "
                                   f"{st['object_type']} colour={st.get('stroking_color')} "
                                   f"lw={st.get('linewidth')}")
    return bad


def main(argv: list[str]) -> int:
    args = [a for a in argv[1:] if a != "--strokes"]
    do_strokes = "--strokes" in argv
    targets = [Path(a) for a in args] or sorted(
        (ROOT / "results" / "figures").glob("*.pdf"))
    total = 0
    for p in targets:
        try:
            bad = check(p, do_strokes)
        except Exception as exc:                      # noqa: BLE001
            print(f"[skip] {p.name}: {exc}")
            continue
        if bad:
            total += len(bad)
            print(f"[FAIL] {p.name}: {len(bad)}")
            for b in bad[:14]:
                print("        " + b)
            if len(bad) > 14:
                print(f"        ... {len(bad)-14} more")
        else:
            print(f"[ok]   {p.name}")
    print(f"\n{total} collision(s) across {len(targets)} figure(s)")
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
