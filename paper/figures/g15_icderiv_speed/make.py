"""G15 -- cost of the initial-condition derivative d(orbit)/d(x0), by mechanism.

Companion to g12 (in-backend orbit speed): how expensive is it to *differentiate*
an orbit w.r.t. its initial conditions?

  * C-STM        -- galpy's compiled C variational integrator. The state-
                    transition matrix M=dx(t)/dx0 comes out of the (variational)
                    forward solve, so the IC-gradient is a free read of M. The
                    current implementation builds M one column at a time (6
                    integrate_dxdv solves); a batched single solve (task #59)
                    would cut that ~6x, and a single directional sensitivity
                    (e.g. dR/dR0) needs only ONE column = one solve (forward
                    mode, task #60). The forward solve runs in C on the CPU.
  * in-backend AD (diffrax / torchdiffeq) -- reverse-mode autodiff through the
                    ODE solve; batches over orbits with jax.vmap, and runs on GPU.

What is plotted is the *full* gradient of a scalar (summed final R) w.r.t. all
six initial conditions -- the common ML use case.

Panel (a): single-orbit IC-gradient wall-clock, per mechanism and device.
Panel (b): IC-gradient wall-clock vs the number of orbits N (batch scaling).

Reads data/icderiv.json (machine-stamped by measure/measure_icderiv.py); the plot
is deterministic given the data, so the PDF is byte-stable. TITAN Xp GPU; an H100
measurement in data/h100/icderiv.json, when present, adds H100 bars and lines.

Colours follow _harness/palette.py (colour = framework x device; jit solid,
eager dashed / hatched); panel (b) is labelled directly.
"""

import json
import os
import pathlib

import matplotlib.pyplot as plt
import numpy

from paper.figures._harness import layout, palette
from paper.figures._harness.figure import figure

_HERE = pathlib.Path(__file__).resolve().parent
_DATA = _HERE / "data" / "icderiv.json"
# the same measurement on an H100 (Trillium), drawn as extra GPU series if present
_DATA_H100 = _HERE / "data" / "h100" / "icderiv.json"
# journal version (default): no suptitle, short panel titles, no stamps; the
# caption carries the explanations. GALPY_JAX_PAPER=0 restores the annotated figure.
PAPER = os.environ.get("GALPY_JAX_PAPER", "1") == "1"

# palette colour = framework x device; diffrax runs under jax.jit (solid),
# torchdiffeq is eager torch (dashed); C-STM is galpy's C (grey).
# (mechanism key, framework, mode, label suffix)
_MECH = [
    ("cstm", "c", None, "C-STM"),
    ("diffrax", "jax", "jit", "diffrax"),
    ("torchdiffeq", "torch", "eager", "torchdiffeq"),
]


def _series(d, d_h100):
    """(mechanism, framework, mode, suffix, data-device, palette-device, source)."""
    out = []
    for mk, fw, mode, suf in _MECH:
        devs = ["cpu"] if fw == "c" else ["cpu", "gpu"]
        for dev in devs:
            out.append((mk, fw, mode, suf, dev, "titan" if dev == "gpu" else "cpu", d))
        if fw != "c" and d_h100 is not None:
            out.append((mk, fw, mode, suf, "gpu", "h100", d_h100))
    return out


@figure(
    name="g15_icderiv_speed",
    track="G",
    seed=0,
    description="cost of the IC derivative d(orbit)/d(x0): C-STM vs in-backend AD (CPU + GPU)",
)
def make():
    d = json.loads(_DATA.read_text())
    d_h100 = json.loads(_DATA_H100.read_text()) if _DATA_H100.exists() else None
    series = _series(d, d_h100)

    # print size: 7.0 in wide (figure*), text 8-10 pt; (a)'s bars are then too
    # narrow for three-line tick labels, so the device goes on the (rotated) tick
    # label and the mechanism once per group below it; (b) has two-line labels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=layout.figsize(PAPER, (7.0, 3.5), (11.0, 4.4)),
                                   gridspec_kw={"width_ratios": [1.0, 1.25]})
    fs_line = layout.FS["line"] if PAPER else 7
    fs_note = layout.FS["note"] if PAPER else 6.5

    # ---- (a) single-orbit IC-gradient, per mechanism x device -------------------
    xs, heights, colors, labels, hatches, devs, groups = [], [], [], [], [], [], {}
    for mk, fw, mode, suf, dk, pdev, src in series:
        v = src["single"].get(mk, {}).get(dk, {}).get("grad")
        if v is None:
            continue
        xs.append(len(heights))
        heights.append(v * 1e3)
        colors.append(palette.style(fw, pdev, mode)["color"])
        hatches.append("//" if mode == "eager" else None)
        dev_lab = "CPU" if fw == "c" else palette.DEVICE_NAME[pdev]
        labels.append(f"{suf}\n{'C' if fw == 'c' else fw}\n{dev_lab}")
        devs.append(dev_lab)
        groups.setdefault(f"{suf} ({'C' if fw == 'c' else fw})", []).append(xs[-1])
    bars = ax1.bar(xs, heights, color=colors, edgecolor="k", lw=0.6)
    for b, h in zip(bars, hatches):
        if h:
            b.set_hatch(h)
            b.set_edgecolor("white")
            b.set_linewidth(0.0)
    ax1.set_yscale("log")
    ax1.set_xticks(xs)
    if PAPER:
        ax1.set_xticklabels(devs, rotation=90, fontsize=layout.FS["tick"])
        # mechanism once per bar group, under the device names
        for g, v in groups.items():
            ax1.annotate(g.replace(" (", "\n("), xy=(numpy.mean(v), 0.0),
                         xycoords=("data", "axes fraction"), xytext=(0, -52),
                         textcoords="offset points", ha="center", va="top",
                         fontsize=layout.FS["legend"], annotation_clip=False)
    else:
        ax1.set_xticklabels(labels, fontsize=7 if len(labels) <= 5 else 5.8)
    ax1.set_ylabel("IC-gradient wall-clock  [ms]")
    ax1.set_title("(a) single orbit" if PAPER else "(a) single orbit: full IC-gradient")
    for x, h in zip(xs, heights):
        # PAPER: values in seconds, rotated, so neighbouring bars do not collide
        if PAPER:
            ax1.text(x, h * 1.15, f"{h / 1e3:.2g} s" if h >= 1e3 else f"{h:.0f} ms",
                     ha="center", va="bottom", fontsize=fs_note, rotation=90)
        else:
            ax1.text(x, h * 1.15, f"{h:.0f}" if h >= 10 else f"{h:.1f}",
                     ha="center", va="bottom", fontsize=fs_note)
    ax1.set_ylim(top=max(heights) * (30 if PAPER else 3))
    ax1.text(0.02, 0.97, "hatched = eager", transform=ax1.transAxes,
             fontsize=fs_note, color="0.4", va="top")

    # ---- (b) IC-gradient wall-clock vs number of orbits N -----------------------
    for mk, fw, mode, suf, dk, pdev, src in series:
        scaling = src["scaling"]
        key = f"{mk}_{dk}"
        Ns = sorted(int(k) for k in scaling)
        xsN = [n for n in Ns if scaling[str(n)].get(key) is not None]
        ysN = [scaling[str(n)][key] * 1e3 for n in xsN]
        if len(xsN) >= 2:
            ax2.loglog(xsN, ysN, ms=4, lw=1.6,
                       label=(layout.label if PAPER else palette.label)(fw, pdev, mode, suf),
                       **palette.style(fw, pdev, mode))
    ax2.set_xlabel("number of orbits  N")
    ax2.set_ylabel("IC-gradient wall-clock  [ms]")
    ax2.set_title("(b) batch scaling" if PAPER else "(b) batch scaling of the IC-gradient")
    ax2.grid(True, ls=":", lw=0.5, alpha=0.5)
    lo, hi = ax2.get_xlim()
    ax2.set_xlim(lo, hi * 1.5)
    # the C-STM line ends mid-plot: label it in the empty region below, not in the
    # right-hand column (its leader would cross the other lines)
    c_lines = [ln for ln in ax2.get_lines() if ln.get_label().startswith("galpy C")]
    for ln in c_lines:
        x, y = ln.get_data()
        k = min(range(len(x)), key=lambda i: abs(x[i] - 100.0))
        ax2.annotate(ln.get_label(), xy=(x[k], y[k]), xytext=(250.0, y[k] / 8.0),
                     fontsize=fs_line, color=ln.get_color(), va="center", ha="left",
                     arrowprops=dict(arrowstyle="-", lw=0.5, color=ln.get_color(), shrinkA=1, shrinkB=2))

    if not PAPER:
        gpus = [d["meta"].get("gpu", "?")] + ([d_h100["meta"].get("gpu", "H100")] if d_h100 else [])
        fig.suptitle("Cost of differentiating an orbit w.r.t. its initial conditions "
                     "(full 6-vector gradient; MWPotential2014)", fontsize=11)
        fig.text(0.01, 0.005, "GPU: " + "; ".join(gpus), fontsize=6.5, color="0.5")
    fig.tight_layout()
    # after tight_layout: label positions are in axes fraction of the final axes
    palette.label_lines(ax2, [(ln, ln.get_label()) for ln in ax2.get_lines()
                              if not ln.get_label().startswith(("_", "galpy C"))],
                        fontsize=fs_line, min_sep_pt=1.3 * fs_line if PAPER else None)
    if PAPER:
        layout.fit_right(fig)
    return fig
