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
    sequential C loop (numcores=1).  ~0.1 ms/orbit after a ~3 ms per-call setup,
    ~0.08 ms/orbit at N>=1e5 (measured to N=1e6).
  * galpy (JAX, CPU) and galpy (JAX, GPU): galpy's OWN in-backend differentiable
    integrator (``galpy.backend._reference.integrate_orbit`` -> diffrax Dopri8),
    jit-fused and ``jax.vmap``-ed over the N ICs (compile excluded), per-orbit
    adaptive stepping -- the common galpy reference against which to place galax.
  * galax (JAX, CPU) and galax (JAX, GPU): the modern ``OrbitSolver`` (diffrax
    Dopri8), jit-compiled and batched over the N ICs (compile time excluded).
    GPU = NVIDIA TITAN Xp (sm_61) and NVIDIA H100 (data/h100/, galpy and galax).

The story, per orbit at each line's largest N (all GPU lines end at N=3e5):
  * galpy C (single thread): 0.079 ms/orbit at N=1e6 (79 s); 0.084 at 3e5 (25 s).
  * TITAN Xp (float64 at 1/32 rate), ~1-2 s launch/compile floor: galpy jax
    per-orbit 0.215 ms/orbit (64 s), galax per-orbit 0.209 (63 s), galax batched
    0.068 (20 s).  galax batched is level with C from N=3e4 to 1e5 (2.8 vs 3.1 s,
    8.0 vs 8.1 s) and faster at 3e5 (20 vs 25 s); the per-orbit TITAN lines stay
    ~2.5x slower than C.
  * H100 (Trillium; galpy in _shared/orbit_speed/data/h100, galax in data/h100):
    galpy jax per-orbit 0.0247 ms/orbit (7.4 s; ~3.4x faster than C, ~9x the TITAN
    Xp), galax per-orbit 0.0246 (7.4 s; equal to galpy), galax batched 0.0130
    (3.9 s; ~6x C).  All three H100 lines cross below C at N~1-3e4.
So C is the fastest line only up to N~1e4; past that the H100 lines are faster
and galax's batched TITAN line is level with it, then faster at N=3e5.
Line convention: _harness/palette.py (colour = framework x device, jit solid;
galax per-orbit dash-dotted).
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

from paper.figures._harness import layout, palette
from paper.figures._harness.figure import figure

_DATA = os.path.join(os.path.dirname(__file__), "data")
_SHARED = os.path.join(
    os.path.dirname(__file__), "..", "_shared", "orbit_speed", "data"
)
# journal version (default): no suptitle, short panel titles, no stamps; the
# caption carries the explanations. GALPY_JAX_PAPER=0 restores the annotated figure.
PAPER = os.environ.get("GALPY_JAX_PAPER", "1") == "1"

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
    # sub="h100": the same measurement on an H100 (data/h100/; galax too, measured 2026-10-07)
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
    # print size: 7.0 in wide (figure*; nine directly-labelled lines do not fit a
    # 3.4 in column), text 8-10 pt, two-line labels
    fig, ax = plt.subplots(figsize=layout.figsize(PAPER, (7.0, 3.8), (6.6, 4.6)))
    fs_line = layout.FS["line"] if PAPER else 6.5
    fs_note = layout.FS["note"] if PAPER else 6.5

    def _lab(lab, dev):
        lab = lab.format(dev=palette.DEVICE_NAME[palette._device(dev)])
        return lab.replace(" (", "\n(") if PAPER else lab

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
            ax.loglog(Ns, wall, ms=4, zorder=3, label=_lab(lab, d_), **st)

    ax.set_xlabel(r"batch size  $N$  (number of orbits)")
    ax.set_ylabel(r"wall time for the batch  (s)")
    if not PAPER:
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
        "dotted: constant cost per orbit" if PAPER
        else "dotted: constant cost / orbit\n(0.1, 1, 10 ms)",
        transform=ax.transAxes, fontsize=fs_note, color="0.5", va="top", ha="left",
    )
    # env stamp from the GPU JSON (the headline machine)
    if not PAPER:
        ax.text(
            0.975, 0.03,
            "galpy C: dop853_c, single thread\n"
            "all lines: diffrax Dopri8 / dop853, rtol=atol=1e-8 (matched)\n"
            f"TITAN Xp: jax {_load_one('jax_gpu.json').get('jax_version', '?')}"
            + (f"  ·  H100: jax {h100_meta.get('jax_version', '?')}" if h100_meta else ""),
            transform=ax.transAxes, fontsize=6.5, color="0.5", va="bottom", ha="right",
        )
    fig.tight_layout()
    palette.label_lines(ax, fontsize=fs_line, min_sep_pt=1.3 * fs_line if PAPER else None,
                        inline={_lab("galpy C (dop853_c)", "cpu"): (300.0, 14 if PAPER else 9)})
    if PAPER:
        layout.fit_right(fig)
    return fig
