"""One palette for every figure that compares backends, plus direct line labels.

Colour = framework x device, line style = execution mode, marker = device:

=========  =========  ==============  ============
framework  CPU        GPU (TITAN Xp)  GPU (H100)
=========  =========  ==============  ============
numpy      black      --              --
C (galpy)  grey       --              --
jax        sky blue   blue            navy
torch      orange     vermillion      dark red
galax      green      green           dark green
CUDA       --         reddish purple  dark purple
=========  =========  ==============  ============

jit / torch.compile = solid, eager = dashed. CPU = square, TITAN Xp = circle,
H100 = diamond. galpy C on several OpenMP threads: lighter grey, dash-dot
(``c_threads_style``). Okabe-Ito / Wong (2011) hues, colour-blind safe, for the CPU and
TITAN shades. A device-agnostic jax/torch line uses the TITAN shade
(``style.BACKEND_COLORS``).

"CUDA" is the experimental one-thread-per-orbit CUDA kernel compiled from galpy's
C potentials and dop853 (experiments/cuda_orbits; not a galpy backend). Its
galpy-arithmetic build is solid; the fast-arithmetic build (division/pow-free
special functions) is long-dashed, ``CUDA_FAST_LS``.
"""

import re

import numpy

_COLOURS = {
    ("numpy", "cpu"): "#000000",
    ("c", "cpu"): "#666666",
    ("jax", "cpu"): "#56B4E9",
    ("jax", "titan"): "#0072B2",
    ("jax", "h100"): "#08306B",
    ("torch", "cpu"): "#E69F00",
    ("torch", "titan"): "#D55E00",
    ("torch", "h100"): "#7F2704",
    ("galax", "cpu"): "#009E73",
    ("galax", "titan"): "#009E73",
    ("galax", "h100"): "#00563F",
    ("cuda", "cpu"): "#CC79A7",  # (host build; not plotted)
    ("cuda", "titan"): "#CC79A7",  # Okabe-Ito reddish purple
    ("cuda", "h100"): "#7B2D6B",
}
# CUDA kernel, fast-arithmetic variant: long dashes (distinct from eager "--",
# Dopri5 "-." and the multi-thread C dash-dot styles)
CUDA_FAST_LS = (0, (7, 2))
_LS = {"jit": "-", "eager": "--", None: "-"}
_MARKER = {"cpu": "s", "titan": "o", "h100": "D"}
DEVICE_NAME = {"cpu": "CPU", "titan": "TITAN Xp", "h100": "H100"}


def _device(device):
    d = (device or "cpu").lower()
    if "h100" in d:
        return "h100"
    if d in ("gpu", "cuda", "titan", "titan xp") or "gpu" in d or "cuda" in d:
        return "titan"
    return "cpu"


def style(framework, device="cpu", mode=None):
    """matplotlib kwargs (color, ls, marker) for one series."""
    fw = framework.lower()
    dev = _device(device)
    if fw in ("numpy", "c"):
        dev = "cpu"
    return {
        "color": _COLOURS[(fw, dev)],
        "ls": _LS[mode],
        "marker": _MARKER[dev],
    }


def c_threads_style(nthreads, node_threads=72):
    """galpy C on ``nthreads`` OpenMP threads (its orbit and action-angle loops thread
    over objects): 1 thread is style("c") (grey, solid); the whole node is a lighter
    grey, dash-dot; anything in between a mid grey, dash-dot-dot."""
    st = style("c")
    if nthreads >= node_threads:
        st.update(color="#A6A6A6", ls="-.")
    elif nthreads > 1:
        st.update(color="#878787", ls=(0, (4, 1.5, 1, 1.5, 1, 1.5)))
    return st


def parse(name, default_mode=None):
    """(framework, device, mode) from a series name such as "jax (GPU, eager)",
    "torch-gpu-jit", "galpy (C, CPU)", "numpy (CPU)" or "jax H100 jit"."""
    n = name.lower()
    if "galax" in n:
        fw = "galax"
    elif "torch" in n:
        fw = "torch"
    elif "jax" in n:
        fw = "jax"
    elif "cuda kernel" in n:
        fw = "cuda"
    elif "numpy" in n:
        fw = "numpy"
    elif re.search(r"(\bc\b|_c\b)", n):
        fw = "c"
    else:
        fw = "numpy"
    if "h100" in n:
        dev = "h100"
    elif "gpu" in n or "cuda" in n or "titan" in n:
        dev = "titan"
    else:
        dev = "cpu"
    if "eager" in n:
        mode = "eager"
    elif "jit" in n or "compile" in n:
        mode = "jit"
    else:
        mode = default_mode
    return fw, dev, mode


def style_of(name, default_mode=None):
    return style(*parse(name, default_mode))


def label(framework, device="cpu", mode=None, extra=""):
    """Canonical short label: "jax H100 jit (Dopri8)"."""
    fw = framework.lower()
    parts = [{"c": "galpy C", "numpy": "numpy", "jax": "jax", "torch": "torch", "galax": "galax",
              "cuda": "CUDA kernel"}[fw]]
    if fw not in ("numpy", "c"):
        parts.append(DEVICE_NAME[_device(device)])
    if mode and fw in ("jax", "torch"):
        parts.append(mode)
    s = " ".join(parts)
    return f"{s} ({extra})" if extra else s


def label_lines(ax, lines=None, fontsize=7, min_sep_pt=None, inline=None, pad_pt=6,
                order="end"):
    """Direct line labels in the line's colour, instead of a legend.

    Every line (default: each with a label not starting with "_") is labelled in a
    column just right of the axes, at the height of its last finite point, spread
    to ``min_sep_pt`` per text line apart (a two-line label takes twice the room), with a thin leader from the line end. ``inline``
    maps a label to ``(x, dy_pt)``: that line is instead labelled on the plot,
    next to its point at data ``x`` (lines that end well inside the axes).
    ``order="edge"``: order (and aim) the labels by each line's height extrapolated
    along its last segment to the rightmost data point of all lines, instead of the
    height of its own last point (fewer leader crossings when lines end at
    different x). ``order="elbow"``: keep the end-point order but draw each leader
    as a thin horizontal run from the line's end to the axes' right edge, then a
    short slanted run to the label -- leaders can then never cross each other.
    Call after the axis limits are final.
    """
    inline = inline or {}
    if lines is None:
        lines = [(ln, ln.get_label()) for ln in ax.get_lines() if not ln.get_label().startswith("_")]
    if min_sep_pt is None:
        min_sep_pt = 1.2 * fontsize
    fig = ax.figure
    fig.canvas.draw()  # final transforms
    pt = fig.dpi / 72.0
    to_axes = ax.transAxes.inverted()
    col = []
    for ln, text in lines:
        x, y = (numpy.asarray(v, dtype=float) for v in ln.get_data())
        ok = numpy.isfinite(x) & numpy.isfinite(y)
        if ax.get_yscale() == "log":
            ok &= y > 0
        if ax.get_xscale() == "log":
            ok &= x > 0
        if not ok.any():
            continue
        c = ln.get_color()
        if text in inline:
            xi, dy = inline[text]
            k = numpy.flatnonzero(ok)
            if ax.get_xscale() == "log":
                yi = numpy.exp(numpy.interp(numpy.log(xi), numpy.log(x[k]), numpy.log(y[k])))
            else:
                yi = numpy.interp(xi, x[k], y[k])
            ax.annotate(text, xy=(xi, yi), xytext=(0, dy), textcoords="offset points",
                        fontsize=fontsize, color=c, ha="center", va="center")
            continue
        k = numpy.flatnonzero(ok)
        i = k[-1]
        P = ax.transData.transform(numpy.c_[x[k[-2:]], y[k[-2:]]])
        Y = P[-1, 1]
        col.append({"xy": (x[i], y[i]), "Y": Y, "Ynew": Y, "text": text, "color": c, "P": P})
    if order == "edge" and col:
        X_edge = max(a["P"][-1, 0] for a in col)
        for a in col:
            P = a["P"]
            if len(P) > 1 and P[-1, 0] > P[-2, 0]:
                slope = (P[-1, 1] - P[-2, 1]) / (P[-1, 0] - P[-2, 0])
                a["Y"] = a["Ynew"] = P[-1, 1] + slope * (X_edge - P[-1, 0])
    col.sort(key=lambda a: a["Y"])

    def sep(lo, hi):  # centre-to-centre spacing; a multi-line label is taller
        nl = lambda a: a["text"].count("\n") + 1
        return 0.5 * (nl(lo) + nl(hi)) * min_sep_pt * pt

    for k in range(1, len(col)):
        col[k]["Ynew"] = max(col[k]["Y"], col[k - 1]["Ynew"] + sep(col[k - 1], col[k]))
    # re-centre so the labels sit around their lines rather than all above them
    if col:
        shift = 0.5 * numpy.mean([a["Ynew"] - a["Y"] for a in col])
        for a in col:
            a["Ynew"] -= shift
        for k in range(1, len(col)):
            col[k]["Ynew"] = max(col[k]["Ynew"], col[k - 1]["Ynew"] + sep(col[k - 1], col[k]))
        # keep the whole column inside the axes' height (leaders connect the lines)
        y_lo, y_hi = ax.transAxes.transform((0.0, 0.0))[1], ax.transAxes.transform((0.0, 1.0))[1]
        half = lambda a: 0.5 * (a["text"].count("\n") + 1) * min_sep_pt * pt
        top = col[-1]["Ynew"] + half(col[-1])
        if top > y_hi:
            for a in col:
                a["Ynew"] -= top - y_hi
        bot = col[0]["Ynew"] - half(col[0])
        if bot < y_lo:
            for a in col:
                a["Ynew"] += y_lo - bot
    x0 = ax.transAxes.transform((1.0, 0.0))[0] + pad_pt * pt
    for a in col:
        xt, yt = to_axes.transform((x0, a["Ynew"]))
        xy, xycoords = a["xy"], "data"
        if order == "elbow":
            ye = to_axes.transform(ax.transData.transform(a["xy"]))[1]
            xe = to_axes.transform(ax.transData.transform(a["xy"]))[0]
            ax.plot([xe, 1.0], [ye, ye], transform=ax.transAxes, color=a["color"], lw=0.5,
                    alpha=0.6, clip_on=False, zorder=1, label="_leader")
            xy, xycoords = (1.0, ye), "axes fraction"
        ax.annotate(
            a["text"], xy=xy, xycoords=xycoords, xytext=(xt, yt), textcoords="axes fraction",
            fontsize=fontsize, color=a["color"], va="center", ha="left", annotation_clip=False,
            arrowprops=dict(arrowstyle="-", lw=0.5, color=a["color"], shrinkA=1, shrinkB=2, alpha=0.8,
                            **({"relpos": (0.0, 0.5)} if order == "elbow" else {})),
        )


def extend_right(ax, factor=None, frac=0.25):
    """Widen the x-range on the right to make room for direct labels."""
    lo, hi = ax.get_xlim()
    if ax.get_xscale() == "log":
        ax.set_xlim(lo, hi * (factor or (hi / lo) ** frac))
    else:
        ax.set_xlim(lo, hi + (factor or frac) * (hi - lo))
