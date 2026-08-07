"""figcheck -- assert that nothing in a figure overlaps anything else.

The earlier check ran on the rendered PDF and compared text bounding boxes with
each other only.  It passed the priority map even though the dashed threshold
rule was drawn straight through the word "Roofing" and through its own label,
because a line is not text.  This module checks inside matplotlib instead, where
every artist's geometry and draw order are known exactly, and it checks the two
pairings the PDF check could not:

  text vs. stroke   (axhline, plot lines, annotation leaders)
  text vs. marker   (scatter points)

Masking is honoured the way the renderer honours it: a text with an opaque bbox
patch legitimately covers anything drawn beneath it, so an artist of lower
zorder under such a text is not a collision.  Everything else is.

Usage, at the end of a figure function and before savefig:

    figcheck.assert_clean(fig, ax, name="fig_cepi_priority_map")
"""
from __future__ import annotations

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.collections import PathCollection
from matplotlib.text import Text, Annotation

PAD = 0.5          # pixels of slack; touching is allowed, overlapping is not


def _opaque_bbox(t: Text) -> bool:
    """True when the text carries a fully opaque background patch."""
    p = t.get_bbox_patch()
    if p is None:
        return False
    a = p.get_alpha()
    if a is not None and a < 0.98:
        return False
    fc = p.get_facecolor()
    return not (len(fc) == 4 and fc[3] < 0.98)


def _rect(t: Text, rend) -> tuple[float, float, float, float] | None:
    """Bounding box of the glyphs alone.

    Annotation.get_window_extent unions the text box with its leader arrow, so
    an annotation offset 49 pt from its anchor reports a box 49 pt taller than
    the words.  Every leader-labelled point then looks like a collision.  Take
    the text box directly instead, after letting the annotation place itself.
    """
    if not t.get_visible() or not str(t.get_text()).strip():
        return None
    if isinstance(t, Annotation) and t.arrow_patch is not None:
        t.update_positions(rend)
        b = Text.get_window_extent(t)
    else:
        b = t.get_window_extent(rend)
    if b.width <= 0 or b.height <= 0:
        return None
    return (b.x0, b.y0, b.x1, b.y1)


def _rects_overlap(a, b, pad=PAD) -> bool:
    return (a[0] < b[2] - pad and b[0] < a[2] - pad
            and a[1] < b[3] - pad and b[1] < a[3] - pad)


def _seg_hits_rect(p0, p1, r, pad=PAD) -> bool:
    """Liang-Barsky: does segment p0->p1 pass through the interior of rect r?"""
    x0, y0 = p0
    x1, y1 = p1
    xmin, ymin, xmax, ymax = r[0] + pad, r[1] + pad, r[2] - pad, r[3] - pad
    if xmin >= xmax or ymin >= ymax:
        return False
    dx, dy = x1 - x0, y1 - y0
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x0 - xmin), (dx, xmax - x0),
                 (-dy, y0 - ymin), (dy, ymax - y0)):
        if p == 0:
            if q < 0:
                return False
        else:
            r_ = q / p
            if p < 0:
                if r_ > t1:
                    return False
                t0 = max(t0, r_)
            else:
                if r_ < t0:
                    return False
                t1 = min(t1, r_)
    return t0 < t1


def _gridlines(ax) -> set[int]:
    return {id(ln) for ln in list(ax.get_xgridlines()) + list(ax.get_ygridlines())}


def collisions(fig, ax) -> list[str]:
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    skip = _gridlines(ax) | {id(s) for s in ax.spines.values()}

    texts, arrows = [], []
    for t in fig.findobj(Text):
        if isinstance(t, Annotation) and t.arrow_patch is not None:
            arrows.append(t)
        r = _rect(t, rend)
        if r is not None:
            texts.append((t, r))

    # tick labels and axis titles live outside the axes; keep them in the
    # text-vs-text check but do not test them against data artists
    ax_bbox = ax.get_window_extent(rend)

    def inside(r):
        return (r[2] > ax_bbox.x0 and r[0] < ax_bbox.x1
                and r[3] > ax_bbox.y0 and r[1] < ax_bbox.y1)

    bad = []

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            (ta, ra), (tb, rb) = texts[i], texts[j]
            if _rects_overlap(ra, rb):
                bad.append(f"text/text: {ta.get_text()!r} <-> {tb.get_text()!r}")

    lines = [ln for ln in fig.findobj(Line2D)
             if id(ln) not in skip and ln.get_visible()
             and ln.get_linestyle() not in ("None", "none", "")]
    for t, r in texts:
        if not inside(r):
            continue
        for ln in lines:
            if ln.zorder < t.zorder and _opaque_bbox(t):
                continue
            xy = ln.get_transform().transform(np.column_stack(ln.get_data()))
            for k in range(len(xy) - 1):
                if _seg_hits_rect(xy[k], xy[k + 1], r):
                    bad.append(f"text/line: {t.get_text()!r} struck by "
                               f"{ln.get_label() or 'line'} (zorder {ln.zorder})")
                    break
            else:
                continue
            break

    # annotation leaders
    for a in arrows:
        patch = a.arrow_patch
        seg = patch.get_path().transformed(patch.get_transform()).vertices
        for t, r in texts:
            if t is a or not inside(r):
                continue
            if _opaque_bbox(t) and patch.zorder < t.zorder:
                continue
            if any(_seg_hits_rect(seg[k], seg[k + 1], r)
                   for k in range(len(seg) - 1)):
                bad.append(f"leader/text: leader of {a.get_text()!r} "
                           f"crosses {t.get_text()!r}")

    for coll in fig.findobj(PathCollection):
        if not coll.get_visible():
            continue
        offs = coll.get_offsets()
        if offs is None or len(offs) == 0:
            continue
        pts = coll.get_offset_transform().transform(offs)
        sizes = coll.get_sizes()
        for n, (px, py) in enumerate(pts):
            s = sizes[n % len(sizes)] if len(sizes) else 36.0
            rad = np.sqrt(s) / 2.0 * fig.dpi / 72.0
            mr = (px - rad, py - rad, px + rad, py + rad)
            for t, r in texts:
                if not inside(r):
                    continue
                if _opaque_bbox(t) and coll.zorder < t.zorder:
                    continue
                if _rects_overlap(mr, r):
                    bad.append(f"marker/text: marker at data index {n} "
                               f"overlaps {t.get_text()!r}")
    return bad


def assert_clean(fig, ax, name: str = "figure") -> None:
    bad = collisions(fig, ax)
    if bad:
        raise AssertionError(
            f"{name}: {len(bad)} collision(s)\n  " + "\n  ".join(bad))
    print(f"[figcheck] {name}: no overlapping elements")
