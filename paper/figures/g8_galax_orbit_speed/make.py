"""G8 -- orbit-integration throughput: galpy (C, CPU) vs galax (JAX, CPU & GPU).

MWPotential2014 / BovyMWPotential2014, a Sun-like orbit integrated for 2 Gyr with
1000 output points, over a batch of N (slightly perturbed) initial conditions.

  * galpy (C, CPU): the compiled ``dop853_c`` integrator, one orbit at a time in a
    sequential C loop (single-threaded build).
  * galax (JAX, CPU) and galax (JAX, GPU): the modern ``OrbitSolver`` (diffrax
    Dopri8), jit-compiled and batched over the N ICs (compile time excluded).
    GPU = NVIDIA TITAN Xp (sm_61).

The story: galpy-C has ~zero startup, so it wins decisively at small N (a single
orbit: ~10 ms vs galax-GPU's ~1.6 s of kernel-launch/compile-bound overhead). But
its per-orbit cost is ~constant (sequential), while galax amortises the vectorised
solve across the batch -- on the GPU down to ~0.09 ms/orbit at N=3e4 (~70x faster
per orbit than galpy-C). Crossover is ~N=300 (GPU) / ~N=40 (CPU). The GPU OOMs at
N=1e5 (12 GiB), saving 1000 dense steps x N x 6 doubles.

Reproducibility: the two backends are measured in SEPARATE environments (galpy in
the main py313 env; galax in its own uv-locked venv with a matched CUDA-12 plugin
-- see ../_setup/galax_gpu_setup.md), so unlike the single-process G-figures the
*measure* step here is the two standalone scripts under measure/, which write the
per-backend data/*.json (machine- and version-stamped). The *plot* below is
byte-stable from those JSONs.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy

from paper.figures._harness.figure import figure

_DATA = os.path.join(os.path.dirname(__file__), "data")

# (json file, legend label, colour, linestyle, marker)
_SERIES = [
    ("galpy_cpu-C.json", "galpy (C, CPU)", "#000000", "-", "o"),
    ("galax_cpu.json", "galax (JAX, CPU)", "#009E73", "--", "s"),
    ("galax_gpu.json", "galax (JAX, GPU: TITAN Xp)", "#D55E00", "-", "D"),
]


def _load(fname):
    path = os.path.join(_DATA, fname)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


@figure(
    name="g8_galax_orbit_speed",
    track="G",
    seed=0,
    description="MWPotential2014 orbit-integration wall time vs batch size N: galpy(C,CPU) vs galax(JAX,CPU/GPU) [cross-env measure->json, plot byte-stable]",
)
def make():
    fig, ax = plt.subplots(figsize=(6.0, 4.4))

    for fname, label, col, ls, mk in _SERIES:
        d = _load(fname)
        if d is None or not d.get("results"):
            continue
        res = sorted(d["results"], key=lambda r: r["N"])
        Ns = numpy.array([r["N"] for r in res], dtype=float)
        wall = numpy.array([r["wall_s"] for r in res], dtype=float)
        ax.loglog(Ns, wall, marker=mk, ms=4.5, color=col, ls=ls, label=label, zorder=3)

    ax.set_xlabel(r"batch size  $N$  (number of orbits)")
    ax.set_ylabel(r"wall time for the batch  (s)")
    ax.set_title(r"MWPotential2014 orbit: 2 Gyr, 1000 steps")

    # constant per-orbit-cost reference lines (slope-1 in log-log): wall = N * tau
    ax.set_ylim(*ax.get_ylim())
    xl = ax.get_xlim()
    Nlo, Nhi = xl
    for tau, lab in ((1e-2, "10 ms"), (1e-3, "1 ms"), (1e-4, "0.1 ms")):
        ax.plot([Nlo, Nhi], [Nlo * tau, Nhi * tau], ls=":", lw=0.8, color="0.8", zorder=0)
    ax.set_xlim(*xl)
    ax.text(
        0.025, 0.975,
        "dotted: constant cost / orbit\n(0.1, 1, 10 ms)",
        transform=ax.transAxes, fontsize=6.5, color="0.5", va="top", ha="left",
    )
    ax.legend(fontsize=7.5, loc="lower right", framealpha=0.92)

    # env stamp from the GPU JSON (the headline machine)
    g = _load("galax_gpu.json") or {}
    ax.text(
        0.025, 0.30,
        f"galax {g.get('solver', 'Dopri8')} on {g.get('device', '?')}\n"
        "galpy dop853_c (single-thread C)",
        transform=ax.transAxes, fontsize=6.5, color="0.5", va="bottom", ha="left",
    )
    fig.tight_layout()
    return fig
