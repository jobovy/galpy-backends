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


def ref_decades(costs):
    """Decade-spaced reference values 10^k bracketing ``costs``: the smallest is
    at or below the cheapest and the largest at or above the dearest, so the
    measured curves lie between reference lines."""
    import math

    c = [float(v) for v in costs if v is not None and math.isfinite(v) and v > 0]
    lo = math.floor(math.log10(min(c)) + 1e-9)
    hi = math.ceil(math.log10(max(c)) - 1e-9)
    return [10.0**k for k in range(lo, hi + 1)]


def last_costs(ax, kind="time"):
    """The per-unit cost at the right end of every labelled data line in ``ax``:
    y/x (``kind="time"``: wall time vs N) or x/y (``kind="rate"``: throughput)."""
    import numpy

    out = []
    for ln in ax.get_lines():
        if ln.get_label().startswith("_ref"):
            continue
        x, y = (numpy.asarray(v, dtype=float) for v in ln.get_data())
        ok = numpy.isfinite(x) & numpy.isfinite(y) & (x > 0) & (y > 0)
        if ok.sum() < 2:
            continue
        i = numpy.flatnonzero(ok)[-1]
        out.append(y[i] / x[i] if kind == "time" else x[i] / y[i])
    return out


def cost_lines(ax, costs, kind="time", x_range=None, scale=1.0):
    """Light-grey dotted constant-cost diagonals behind the data: y = c*x
    (``kind="time"``) or y = x/c (``"rate"``) for each decade value c returned by
    ref_decades(costs); ``scale`` multiplies y (e.g. 1e3 for a ms axis). The axis
    limits are kept. Returns the decade values (per unit, in the units of
    ``costs``)."""
    import numpy

    vals = ref_decades(costs)
    xl, yl = ax.get_xlim(), ax.get_ylim()
    lo, hi = x_range if x_range is not None else xl
    x = numpy.geomspace(lo, hi, 2)
    for c in vals:
        y = c * x if kind == "time" else x / c
        ax.plot(x, scale * y, ls=":", lw=0.8, color="0.75", zorder=0, label="_ref")
    ax.set_xlim(xl)
    ax.set_ylim(yl)
    return vals


def fmt_seconds(vals, unit=None, wrap=False):
    """'0.01, 0.1, 1, 10 ms' for decade values in seconds: one unit (``unit``, or
    the largest of s/ms/us/ns that keeps every value in [0.01, 100]), else each
    value in its own unit, grouped: '100 us, 1, 10, 100 ms, 1 s'."""
    units = [("s", 1.0), ("ms", 1e-3), (r"$\mu$s", 1e-6), ("ns", 1e-9)]
    if unit is not None:
        f = dict(units)[unit]
        return ", ".join(f"{v / f:g}" for v in vals) + f" {unit}"
    for u, f in units:
        s = [v / f for v in vals]
        if min(s) >= 0.01 - 1e-12 and max(s) <= 100 + 1e-9:
            return ", ".join(f"{x:g}" for x in s) + f" {u}"
    groups = []
    for v in vals:
        u, f = next(((u, f) for u, f in units if v >= f * (1 - 1e-9)), units[-1])
        if groups and groups[-1][0] == u:
            groups[-1][1].append(v / f)
        else:
            groups.append((u, [v / f]))
    return (",\n" if wrap else ", ").join(", ".join(f"{x:g}" for x in xs) + f" {u}" for u, xs in groups)
