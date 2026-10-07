"""Print-size layout for the journal (PAPER) version of a figure.

AASTeX two-column: 10 pt body text, 7.1 in text width (``figure*``), 3.4 in column
width (``figure``). Each figure is included at its final width, so in PAPER mode a
``make()`` sizes the figure to that width, keeps every text at 8-10 pt, and saves
the page at exactly the figure size (a tight bounding box would change the page
width and so rescale the text on inclusion).
"""

import matplotlib as mpl

from . import palette

TWO_COLUMN = 7.0
ONE_COLUMN = 3.4
# text sizes at print scale: axis labels 10, tick labels 9, panel titles 10,
# legends and direct line labels 8.5, in-plot notes 8
FS = dict(title=10, label=10, tick=9, legend=8.5, line=8.5, note=8)


def figsize(paper, paper_wh, detailed_wh):
    """The figure size for this mode.

    PAPER also pins the saved page to the figure size (no tight-bbox cropping) and
    the panel titles to 10 pt; the detailed version keeps the harness defaults.
    """
    if not paper:
        return detailed_wh
    mpl.rcParams["savefig.bbox"] = "standard"
    mpl.rcParams["axes.titlesize"] = FS["title"]
    return paper_wh


def label(fw, dev, mode=None, extra=""):
    """Two-line direct label, "jax jit" over "TITAN Xp (Dopri8)" -- the paper-wide
    label (palette.label) folded so that a label column stays narrow at print
    size. ``dev=None`` omits the device (a panel title already names it)."""
    fw = fw.lower()
    name = {"c": "galpy C", "numpy": "numpy", "jax": "jax", "torch": "torch",
            "galax": "galax"}[fw]
    first = name + (f" {mode}" if mode and fw in ("jax", "torch") else "")
    second = ""
    if dev is not None and fw not in ("numpy", "c"):
        second = palette.DEVICE_NAME[palette._device(dev)]
    if extra:
        second = (second + f" ({extra})").strip()
    return first + ("\n" + second if second else "")


def fit_right(fig, pad_in=0.04):
    """Make room for texts drawn right of an axes (the direct line labels of
    palette.label_lines): shrink the subplot area so they stay inside the page,
    and widen the gap between side-by-side panels so they clear the next panel's
    y-axis. Call last, after tight_layout and label_lines."""
    pad = pad_in * fig.dpi
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    # a label column spread past the top or bottom of its axes: slide it back
    for ax in fig.axes:
        col = [t for t in ax.texts if getattr(t, "xyann", None) is not None
               and t.get_anncoords() == "axes fraction" and t.xyann[0] >= 1.0]
        if not col:
            continue
        y1 = max(t.get_window_extent(r).y1 for t in col)
        y0 = min(t.get_window_extent(r).y0 for t in col)
        dy = min(0.0, ax.bbox.y1 - y1) + max(0.0, ax.bbox.y0 - y0)
        if abs(dy) > 0.5:
            for t in col:
                t.xyann = (t.xyann[0], t.xyann[1] + dy / ax.bbox.height)
    for _ in range(4):
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        W = fig.bbox.width
        changed = False
        # right edge of the page
        x1 = max((t.get_window_extent(r).x1 for ax in fig.axes for t in ax.texts
                  if t.get_visible() and t.get_text()), default=0.0)
        over = x1 + pad - W
        if over > 0.5:
            fig.subplots_adjust(right=fig.subplotpars.right - over / W)
            changed = True
        # gaps between panels (same row, by subplot geometry)
        axes = [ax for ax in fig.axes if ax.get_subplotspec() is not None]
        extra = 0.0
        for ax in axes:
            bb = ax.bbox
            hang = max((t.get_window_extent(r).x1 for t in ax.texts
                        if t.get_visible() and t.get_text()), default=bb.x1) - bb.x1
            right_of = [a for a in axes if a is not ax and a.bbox.x0 > bb.x1 - 1
                        and a.bbox.y1 > bb.y0 and a.bbox.y0 < bb.y1]
            if not right_of or hang <= 0:
                continue
            nxt = min(right_of, key=lambda a: a.bbox.x0)
            ybb = nxt.yaxis.get_tightbbox(r)  # None when the y-axis is shared/hidden
            free = (ybb.x0 if ybb is not None else nxt.bbox.x0) - bb.x1
            extra = max(extra, hang + pad - free)
        if extra > 0.5:
            sp = fig.subplotpars
            avg_w = sum(a.bbox.width for a in axes) / len(axes)
            fig.subplots_adjust(wspace=sp.wspace + extra / avg_w)
            changed = True
        if not changed:
            return
