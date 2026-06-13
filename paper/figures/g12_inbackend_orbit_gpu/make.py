"""G12 -- galpy's OWN in-backend orbit integrator: GPU vs CPU vs compiled C.

The analogue of G8, but for galpy's #906 differentiable in-backend integrator
(``galpy.backend._reference.integrate_orbit``) rather than galax. Same experiment
as G8: MWPotential2014, a Sun-like orbit integrated for 2 Gyr over a batch of N
(slightly perturbed) initial conditions; wall time for the whole batch vs N.

Four series, all integrating the *same* backend-agnostic galpy forces:

  * galpy (C, CPU): the compiled ``dop853_c`` integrator, one orbit at a time in a
    single-threaded C loop -- the classical baseline. The in-backend EOM is modelled
    on this C integrator (rectangular variables), so it is the apples-to-apples
    sequential reference.
  * galpy in-backend (JAX, CPU): the diffrax Dopri8 path, ``jax.vmap``-ed over the
    batch and jit-fused, run on the CPU -- isolates the framework/vectorisation cost
    from the device.
  * galpy in-backend (JAX, GPU): the same, on the NVIDIA TITAN Xp (sm_61).
  * galpy in-backend (torch, GPU): the torchdiffeq Dopri5 path. Only a few small-N
    points: torchdiffeq runs EAGER (no jit-fuse of the adaptive-step loop), so every
    step pays full Python dispatch overhead and the GPU is starved -- correct and
    differentiable, but impractically slow at scale (see docstring/caption).

All lines use rtol=atol=1e-8 and per-orbit adaptive stepping. The galpy-C and
galpy-jax (CPU/GPU) series are loaded from ``paper/figures/_shared/orbit_speed/data``
and are byte-for-byte IDENTICAL to G8's -- same orbit (MWPotential2014, Sun-like IC,
2 Gyr = linspace(0, 56.2, nt)), same tolerance (rtol=atol=1e-8), and same nt
convention (lowN regime nt=1000 dense saves; hiN regime nt=16 sparse saves; per-orbit
cost matches and the overlap points agree), so G8 and G12 can be read together.

The story mirrors G8: galpy-C has ~zero startup and wins at small N (one orbit
~3-10 ms vs the GPU's ~1.5 s compile/launch-bound floor), but its per-orbit cost is
constant (sequential); the jit-fused JAX-GPU line amortises the vectorised solve
across the batch down to ~0.21-0.25 ms/orbit, crossing galpy-C around N~few-hundred.
The JAX-CPU line shows the device contribution.

The **torch-GPU line is the eager-vs-compute story**: torchdiffeq runs EAGER (no
jit-fuse of the adaptive-step loop). At small N the batch is *overhead-bound* -- the
wall time is a near-constant ~12 s set by the Python-driven step loop, the GPU is
starved, and ms/orbit falls as ~1/N. Pushed to large N the line crosses into the
*compute-bound* regime: the per-step GPU kernels finally dominate the Python
overhead, wall time starts rising ~linearly, and ms/orbit flattens to torch's true
per-orbit asymptote. Correct and fully differentiable, but ~40x slower per orbit than
the jit-fused JAX path -- exactly why G8 omits a torch line; G12 shows it (out to the
OOM/time cap) to make the eager overhead->compute transition visible.

POTENTIAL SUPPORT: as of the backends worktree (#949, 36c92e8f), MWPotential2014
(PowerSphericalCutoff bulge + MiyamotoNagai disk + NFW halo) is fully
in-backend-integrable on jax AND torch -- the earlier "Plummer/Isochrone only"
limitation no longer holds, so this figure uses the same MW model as G8.

Reproducibility: the shared galpy series come from ../_shared/orbit_speed
(galpy_c_measure.py, galpy_jax_measure.py for {CPU,GPU} x {lowN dense, hiN}), run
against the backend-migrated galpy worktree (/var/tmp/backends-merge, which now
ships libgalpy so the C baseline runs from the SAME galpy as the jax/torch lines);
torch is measured by measure/torch_measure.py. Each writes a machine- and
version-stamped data/*.json. This plot is byte-stable from those JSONs; ``verify``
only re-renders the (deterministic) plot.
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
# The GPU/CPU jax lines merge a low-N run (1000 dense saves, small-N crossover) with
# a high-N run (16 saves, fits the batch in 12 GiB) -- integration is adaptive-step-
# bound not save-bound, so per-orbit cost matches and the overlap points agree.
# galpy_c_* and jax_* are SHARED with G8 (../_shared/orbit_speed/data); torch is local.
_SERIES = [
    (["galpy_c_cpu.json", "galpy_c_cpu_hiN.json"], "galpy (C, CPU; sequential)", "#000000", "-", "o"),
    (["jax_cpu.json", "jax_cpu_hiN.json"], "galpy in-backend (JAX, CPU)", "#009E73", "--", "s"),
    (["jax_gpu.json", "jax_gpu_hiN.json"], "galpy in-backend (JAX, GPU)", "#0072B2", "-", "^"),
    (["torch_cuda.json"], "galpy in-backend (torch, GPU; eager)", "#D55E00", ":", "v"),
]


def _load_one(fname):
    # shared galpy JSONs live in _shared/orbit_speed/data; torch JSON is local.
    for base in (_SHARED, _DATA):
        path = os.path.join(base, fname)
        if os.path.exists(path):
            with open(path) as fh:
                return json.load(fh)
    return None


def _load(fnames):
    merged, meta = {}, None
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
    name="g12_inbackend_orbit_gpu",
    track="G",
    seed=0,
    description="MWPotential2014 in-backend orbit integration wall time vs batch size N: galpy(C,CPU) vs galpy in-backend (JAX CPU/GPU, torch GPU) [cross-env measure->json, plot byte-stable]",
)
def make():
    fig, ax = plt.subplots(figsize=(6.0, 4.4))

    gpu_meta = _load_one("jax_gpu.json") or _load_one("jax_gpu_hiN.json") or {}
    torch_meta = _load_one("torch_cuda.json") or {}
    for fnames, label, col, ls, mk in _SERIES:
        d = _load(fnames)
        if d is None or not d.get("results"):
            continue
        res = sorted(d["results"], key=lambda r: r["N"])
        Ns = numpy.array([r["N"] for r in res], dtype=float)
        wall = numpy.array([r["wall_s"] for r in res], dtype=float)
        ax.loglog(Ns, wall, marker=mk, ms=4.5, color=col, ls=ls, label=label, zorder=3)

    ax.set_xlabel(r"batch size  $N$  (number of orbits)")
    ax.set_ylabel(r"wall time for the batch  (s)")
    ax.set_title(r"galpy in-backend orbit integration (MWPotential2014, 2 Gyr)")

    # constant per-orbit-cost reference lines (slope-1 in log-log): wall = N * tau
    ax.set_ylim(*ax.get_ylim())
    xl = ax.get_xlim()
    Nlo, Nhi = xl
    for tau in (1e-2, 1e-3, 1e-4):
        ax.plot([Nlo, Nhi], [Nlo * tau, Nhi * tau], ls=":", lw=0.8, color="0.8", zorder=0)
    ax.set_xlim(*xl)
    ax.text(
        0.025, 0.975,
        "dotted: constant cost / orbit\n(0.1, 1, 10 ms)",
        transform=ax.transAxes, fontsize=6.5, color="0.5", va="top", ha="left",
    )
    # legend OUTSIDE the axes (upper right, anchored to the figure margin) so it
    # overlaps no data line -- notably the near-flat eager torch line across the top.
    ax.legend(
        fontsize=7.0, loc="upper left", bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0, framealpha=0.95,
    )
    ax.text(
        0.025, 0.02,
        "galpy in-backend integrate_orbit (#906)\n"
        "JAX: diffrax Dopri8 (jit+vmap)  ·  torch: torchdiffeq dopri5 (eager)\n"
        f"rtol=atol=1e-8  ·  GPU: {gpu_meta.get('gpu', '?')}\n"
        f"jax {gpu_meta.get('jax_version', '?')} · torch {torch_meta.get('torch_version', '?')}",
        transform=ax.transAxes, fontsize=6.0, color="0.5", va="bottom", ha="left",
    )
    fig.tight_layout()
    return fig
