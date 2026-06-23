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
is deterministic given the data, so the PDF is byte-stable. TITAN Xp GPU.
"""

import json
import pathlib

import matplotlib.pyplot as plt
import numpy

from paper.figures._harness.figure import figure
from paper.figures._harness.style import BACKEND_COLORS

_DATA = pathlib.Path(__file__).resolve().parent / "data" / "icderiv.json"
# distinct colour per series so CPU vs GPU (and jax vs torch) read unambiguously
_C_CSTM = "#222222"   # C-STM (C, CPU)
_C_JCPU = "#009E73"   # diffrax jax CPU (green)
_C_JGPU = "#0072B2"   # diffrax jax GPU (blue)
_C_TCPU = "#E69F00"   # torchdiffeq CPU (amber)
_C_TGPU = "#D55E00"   # torchdiffeq GPU (vermilion)

# (data-key, device-key, label, color, marker)
_BARS = [
    ("cstm", "cpu", "C-STM\n(C, CPU)", _C_CSTM, "o"),
    ("diffrax", "cpu", "diffrax\n(jax, CPU)", _C_JCPU, "s"),
    ("diffrax", "gpu", "diffrax\n(jax, GPU)", _C_JGPU, "^"),
    ("torchdiffeq", "cpu", "torchdiffeq\n(CPU)", _C_TCPU, "v"),
    ("torchdiffeq", "gpu", "torchdiffeq\n(GPU)", _C_TGPU, "P"),
]
# (scaling-key, label, color, linestyle, marker)
_LINES = [
    ("cstm_cpu", "C-STM (C, CPU)", _C_CSTM, "-", "o"),
    ("diffrax_cpu", "diffrax (jax, CPU)", _C_JCPU, "-", "s"),
    ("diffrax_gpu", "diffrax (jax, GPU)", _C_JGPU, "-", "^"),
    ("torchdiffeq_cpu", "torchdiffeq (CPU)", _C_TCPU, "--", "v"),
    ("torchdiffeq_gpu", "torchdiffeq (GPU)", _C_TGPU, "--", "P"),
]


@figure(
    name="g15_icderiv_speed",
    track="G",
    seed=0,
    description="cost of the IC derivative d(orbit)/d(x0): C-STM vs in-backend AD (CPU + GPU)",
)
def make():
    d = json.loads(_DATA.read_text())
    single, scaling = d["single"], d["scaling"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.2))

    # ---- (a) single-orbit IC-gradient, per mechanism x device -------------------
    xs, heights, colors, labels = [], [], [], []
    for mk, dk, lab, col, _mark in _BARS:
        v = single.get(mk, {}).get(dk, {}).get("grad")
        if v is None:
            continue
        xs.append(len(heights))
        heights.append(v * 1e3)
        colors.append(col)
        labels.append(lab)
    ax1.bar(xs, heights, color=colors, edgecolor="k", lw=0.6)
    ax1.set_yscale("log")
    ax1.set_xticks(xs)
    ax1.set_xticklabels(labels, fontsize=7.5)
    ax1.set_ylabel("IC-gradient wall-clock  [ms]")
    ax1.set_title("(a) single orbit: full IC-gradient")
    for x, h in zip(xs, heights):
        ax1.text(x, h * 1.15, f"{h:.0f}" if h >= 10 else f"{h:.1f}",
                 ha="center", va="bottom", fontsize=7)
    ax1.set_ylim(top=max(heights) * 3)

    # ---- (b) IC-gradient wall-clock vs number of orbits N -----------------------
    Ns = sorted(int(k) for k in scaling)
    for key, lab, col, lsty, mark in _LINES:
        xsN = [n for n in Ns if scaling[str(n)].get(key) is not None]
        ysN = [scaling[str(n)][key] * 1e3 for n in xsN]
        if len(xsN) >= 2:
            ax2.loglog(xsN, ysN, marker=mark, ms=4.5, lw=1.8, color=col, ls=lsty, label=lab)
    ax2.set_xlabel("number of orbits  N")
    ax2.set_ylabel("IC-gradient wall-clock  [ms]")
    ax2.set_title("(b) batch scaling of the IC-gradient")
    ax2.legend(fontsize=7.4, loc="upper left", framealpha=0.95)
    ax2.grid(True, which="both", ls=":", lw=0.5, alpha=0.5)

    fig.suptitle("Cost of differentiating an orbit w.r.t. its initial conditions "
                 "(full 6-vector gradient; MWPotential2014)", fontsize=11)
    fig.tight_layout()
    return fig
