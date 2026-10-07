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
line only reaches C's per-orbit cost at the largest N.  On an H100 (Trillium,
data/h100 of the shared orbit data; galpy's jax GPU line only -- galax was not run
there) galpy's per-orbit in-backend line reaches ~0.025 ms/orbit at N=3e5 (~4x
single-thread C, crossing it near N~2e4).  Line convention: _harness/palette.py
(colour = framework x device, jit solid; galax per-orbit dash-dotted).
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

from paper.figures._harness import palette
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
# (json files, framework, device, mode, label, line-style override); colours from
# _harness/palette.py (galpy C grey, jax blue shades, galax green).
_SERIES = [
    (["galpy_c_cpu.json", "galpy_c_cpu_hiN.json"], "c", "cpu", None, "galpy C (dop853_c)", None),
    (["galax_cpu.json"], "galax", "cpu", "jit", "galax CPU", None),
    (["jax_cpu.json", "jax_cpu_hiN.json"], "jax", "cpu", "jit", "galpy jax CPU (per-orbit steps)", None),
    (["jax_gpu.json", "jax_gpu_hiN.json"], "jax", "gpu", "jit", "galpy jax {dev} (per-orbit steps)", None),
    (["galax_gpu_perorbit.json", "galax_gpu_perorbit_hiN.json"], "galax", "gpu", "jit",
     "galax {dev} (per-orbit steps)", "-."),
    (["galax_gpu.json", "galax_gpu_hiN.json"], "galax", "gpu", "jit", "galax {dev} (batched steps)", None),
]


def _load_one(fname, sub=""):
    # shared galpy JSONs live in _shared/orbit_speed/data; galax JSONs are local.
    # sub="h100": the same measurement on an H100 (data/h100/; galax not run there)
    for base in (_SHARED, _DATA):
        path = os.path.join(base, sub, fname)
        if os.path.exists(path):
            with open(path) as fh:
                return json.load(fh)
    return None


def _load(fnames, sub=""):
    # merge results by N across the listed files (later files win on overlap)
    merged = {}
    meta = None
    for fname in fnames:
        d = _load_one(fname, sub)
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
    fig, ax = plt.subplots(figsize=(6.6, 4.6))

    h100_meta = _load_one("jax_gpu.json", "h100") or {}
    for fnames, fw, dev, mode, lab, ls in _SERIES:
        for sub, d_ in (("", dev), ("h100", "h100")):
            if sub and dev != "gpu":
                continue
            d = _load(fnames, sub)
            if d is None or not d.get("results"):
                continue
            res = sorted(d["results"], key=lambda r: r["N"])
            Ns = numpy.array([r["N"] for r in res], dtype=float)
            wall = numpy.array([r["wall_s"] for r in res], dtype=float)
            st = palette.style(fw, d_, mode)
            if ls:
                st["ls"] = ls
            ax.loglog(Ns, wall, ms=4, zorder=3,
                      label=lab.format(dev=palette.DEVICE_NAME[palette._device(d_)]), **st)

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
    # env stamp from the GPU JSON (the headline machine)
    g = _load_one("galax_gpu.json") or {}
    ax.text(
        0.975, 0.03,
        "galpy C: dop853_c, single thread\n"
        "all lines: diffrax Dopri8 / dop853, rtol=atol=1e-8 (matched)\n"
        f"TITAN Xp: jax {_load_one('jax_gpu.json').get('jax_version', '?')}"
        + (f"  ·  H100 (galpy only): jax {h100_meta.get('jax_version', '?')}" if h100_meta else ""),
        transform=ax.transAxes, fontsize=6.5, color="0.5", va="bottom", ha="right",
    )
    fig.tight_layout()
    palette.label_lines(ax, fontsize=6.5, inline={"galpy C (dop853_c)": (300.0, 9)})
    return fig
