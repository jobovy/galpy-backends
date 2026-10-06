"""G8 -- orbit-integration throughput: galpy (C, CPU & JAX CPU/GPU) vs galax (JAX, CPU & GPU).

MWPotential2014 / BovyMWPotential2014, a Sun-like orbit integrated for 2 Gyr over a
batch of N (slightly perturbed) initial conditions.  This figure shares its galpy
series (galpy-C, galpy-jax-CPU, galpy-jax-GPU) byte-for-byte with G12 -- both load
the SAME canonical JSONs from ``paper/figures/_shared/orbit_speed/data/`` so the two
figures' orbit, tolerance, and galpy curves are identical and can be read together.

Shared parameterization (G8 == G12 on these series):
  * same orbit: MWPotential2014, Sun-like IC, 2 Gyr (= linspace(0, 56.2, nt) in
    galpy internal units), per-N-seeded Gaussian perturbations;
  * same tolerance: rtol = atol = 1e-8 (the galax OrbitSolver default; galpy's
    integrators called with the same instead of their tighter 1e-10 default);
  * same nt convention -- lowN regime: nt=1000 dense saves (resolved small-N
    crossover); hiN regime: nt=16 sparse saves (fits the batch in 12 GiB at large
    N).  Integration is adaptive-step-bound not save-bound, so per-orbit cost is the
    same in both regimes and the overlap points agree -> one consistent curve each.

  * galpy (C, CPU): the compiled ``dop853_c`` integrator, one orbit at a time in a
    sequential C loop (numcores=1).  ~0.1 ms/orbit after a ~3 ms per-call setup.
  * galpy (JAX, CPU) and galpy (JAX, GPU): galpy's OWN in-backend differentiable
    integrator (``galpy.backend._reference.integrate_orbit`` -> diffrax Dopri8),
    jit-fused and ``jax.vmap``-ed over the N ICs (compile excluded), per-orbit
    adaptive stepping -- the common galpy reference against which to place galax.
  * galax (JAX, CPU) and galax (JAX, GPU): the modern ``OrbitSolver`` (diffrax
    Dopri8), jit-compiled and batched over the N ICs (compile time excluded).
    GPU = NVIDIA TITAN Xp (sm_61).

The story (TITAN Xp, float64 at 1/32 rate): single-thread galpy-C, ~0.1 ms/orbit,
is the fastest line at every measured N; the JAX GPU backends amortise their ~1.6 s
launch/compile floor down to ~0.07 ms/orbit (galax, batched stepping) and ~0.21
ms/orbit (galpy in-backend, per-orbit stepping) at N=3e5, i.e. galax's batched GPU
line only reaches C's per-orbit cost at the largest N.
The galpy-C numbers were re-measured on 2026-10-06 at ~0.1 ms/orbit; the first
version of this figure (June) had ~2.1 ms/orbit, ~20x too slow -- a slow build in
that environment (both a June-25 and the current galpy measure ~0.1 ms/orbit today).

The GPU lowN (dense) run OOMs at N=1e5 (12 GiB), saving 1000 dense steps x N x 6
doubles; the hiN (sparse, 16 saves) run reaches N=3e5.

Reproducibility: the galpy series come from the shared ``_shared/orbit_speed``
measure scripts (galpy_c_measure.py, galpy_jax_measure.py) run against the
backend-migrated galpy worktree (/var/tmp/backends-merge); galax is measured in its
own uv-locked venv with a matched CUDA-12 plugin -- see ../_setup/galax_gpu_setup.md.
Each writes a machine- and version-stamped JSON.  The *plot* below is byte-stable
from those JSONs.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy

from paper.figures._harness.figure import figure

_DATA = os.path.join(os.path.dirname(__file__), "data")
_SHARED = os.path.join(
    os.path.dirname(__file__), "..", "_shared", "orbit_speed", "data"
)

# (json files [low-N first, optional high-N second], legend label, colour, ls, marker).
# GPU backends merge a low-N run (1000 dense saves, shows the small-N crossover)
# with a high-N run (16 saves, fits up to ~3e5 orbits in 12 GiB) -- per-orbit
# integration cost is the same (integration-bound, saves are cheap), and the
# overlap points agree.
# All GPU lines at MATCHED tolerances (rtol=atol=1e-8, the galax OrbitSolver
# default; galpy's integrate_orbit called with rtol=atol=1e-8 instead of its
# tighter 1e-10 default). Two galax-GPU lines: its native batched mode (ONE
# shared PIDController for the whole batch -> one dt sequence) and a vmapped
# per-orbit-adaptive run (same stepping semantics as galpy's vmap-of-solve) --
# the remaining gap between the per-orbit lines is implementation overhead;
# the gap between the galax lines is the batched-stepping architecture.
# The galpy series (galpy_c_*, jax_*) live in ../_shared/orbit_speed/data and are
# shared byte-for-byte with G12; the galax series are local to G8.
_SERIES = [
    (["galpy_c_cpu.json", "galpy_c_cpu_hiN.json"], "galpy (C, CPU)", "#000000", "-", "o"),
    (["galax_cpu.json"], "galax (JAX, CPU)", "#009E73", "--", "s"),
    (["jax_cpu.json", "jax_cpu_hiN.json"], "galpy (JAX, CPU; per-orbit steps)", "#56B4E9", "--", "P"),
    (["jax_gpu.json", "jax_gpu_hiN.json"], "galpy (JAX, GPU; per-orbit steps)", "#0072B2", "-", "^"),
    (["galax_gpu_perorbit.json", "galax_gpu_perorbit_hiN.json"], "galax (JAX, GPU; per-orbit steps)", "#D55E00", "--", "v"),
    (["galax_gpu.json", "galax_gpu_hiN.json"], "galax (JAX, GPU; batched steps)", "#D55E00", "-", "D"),
]


def _load_one(fname):
    # shared galpy JSONs live in _shared/orbit_speed/data; galax JSONs are local.
    for base in (_SHARED, _DATA):
        path = os.path.join(base, fname)
        if os.path.exists(path):
            with open(path) as fh:
                return json.load(fh)
    return None


def _load(fnames):
    # merge results by N across the listed files (later files win on overlap)
    merged = {}
    meta = None
    for fname in fnames:
        d = _load_one(fname)
        if d is None:
            continue
        meta = d
        for r in d.get("results", []):
            merged[r["N"]] = r
    if meta is None:
        return None
    return {**meta, "results": [merged[n] for n in sorted(merged)]}


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
    ax.set_title(r"MWPotential2014 orbit integration, 2 Gyr")

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
    # legend OUTSIDE the axes (upper right, anchored to the figure margin) so it
    # overlaps no data line.
    ax.legend(
        fontsize=7.0, loc="upper left", bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0, framealpha=0.95,
    )

    # env stamp from the GPU JSON (the headline machine)
    g = _load_one("galax_gpu.json") or {}
    ax.text(
        0.025, 0.03,
        f"galax {g.get('solver', 'Dopri8')} on {g.get('device', '?')}\n"
        "galpy dop853_c (single-thread C)\n"
        "all lines: diffrax/dop853, rtol=atol=1e-8 (matched)",
        transform=ax.transAxes, fontsize=6.5, color="0.5", va="bottom", ha="left",
    )
    fig.tight_layout()
    return fig
