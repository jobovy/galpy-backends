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

  * galpy (C, CPU): the compiled ``dop853_c`` integrator. Its orbit loop is
    OpenMP-parallel over the orbits (omp_get_max_threads(); ``numcores`` does not
    reach it), so it is drawn twice: on ONE core (OMP_NUM_THREADS=1 + taskset;
    galpy_c_cpu_1cores*.json): 2.2-2.4 ms/orbit; and on the whole node (unpinned,
    72 hardware threads; galpy_c_cpu*.json): ~0.1 ms/orbit after a ~3 ms per-call
    setup, ~0.08 ms/orbit at N>=1e5 (measured to N=1e6).
  * galpy (JAX, CPU) and galpy (JAX, GPU): galpy's OWN in-backend differentiable
    integrator (``galpy.backend._reference.integrate_orbit`` -> diffrax Dopri8),
    jit-fused and ``jax.vmap``-ed over the N ICs (compile excluded), per-orbit
    adaptive stepping -- the common galpy reference against which to place galax.
  * galax (JAX, CPU) and galax (JAX, GPU): the modern ``OrbitSolver`` (diffrax
    Dopri8), jit-compiled and batched over the N ICs (compile time excluded).
    GPU = NVIDIA TITAN Xp (sm_61) and NVIDIA H100 (data/h100/, galpy and galax).

The story, per orbit at each line's largest N (all GPU lines end at N=3e5):
  * galpy C, 1 core: 2.2-2.4 ms/orbit (716 s at 3e5). 72 threads: 0.079 ms/orbit
    at N=1e6 (79 s); 0.084 at 3e5 (25 s). "C" below = the 72-thread node.
  * TITAN Xp (float64 at 1/32 rate), ~1-2 s launch/compile floor: galpy jax
    per-orbit 0.215 ms/orbit (64 s), galax per-orbit 0.209 (63 s), galax batched
    0.068 (20 s).  galax batched is level with C from N=3e4 to 1e5 (2.8 vs 3.1 s,
    8.0 vs 8.1 s) and faster at 3e5 (20 vs 25 s); the per-orbit TITAN lines stay
    ~2.5x slower than C.
  * H100 (Trillium; galpy in _shared/orbit_speed/data/h100, galax in data/h100):
    galpy jax per-orbit 0.0247 ms/orbit (7.4 s; ~3.4x faster than C, ~9x the TITAN
    Xp), galax per-orbit 0.0246 (7.4 s; equal to galpy), galax batched 0.0130
    (3.9 s; ~6x C).  All three H100 lines cross below C at N~1-3e4.
So the 72-thread C node is the fastest line only up to N~1e4; past that the H100
lines are faster and galax's batched TITAN line is level with it, then faster at
N=3e5. Every GPU line overtakes ONE-core C at N~500-1e3 (11-35x faster per orbit
at 3e5 on the TITAN, ~97-180x on the H100).
Line convention: _harness/palette.py (colour = framework x device, jit solid;
galax per-orbit dash-dotted).
The galpy-C numbers re-measured on 2026-10-06 (~0.1 ms/orbit) ran unpinned on all
72 threads; the first version of this figure (June) had ~2.1-3.4 ms/orbit, which
matches the one-core numbers point by point (N=1, 10, 100: June 3.4, 3.3, 2.6;
one core 3.4, 3.2, 2.2 ms/orbit) -- most likely a run without OpenMP threading,
not a slow build.

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
# (json files, framework, device, mode, label, line-style or style override); colours
# from _harness/palette.py (galpy C grey, jax blue shades, galax green).
# CPU thread counts (paper/ms/notes/CPU_CORES_AUDIT.md): galpy C threads its orbit
# loop with OpenMP (omp_get_max_threads(); ``numcores`` does not reach it), so the
# unpinned galpy_c_cpu*.json ran on all 72 hardware threads; the *_1cores* files are
# OMP_NUM_THREADS=1 + taskset to one core. XLA:CPU's thread pool spans the CPU
# affinity, so the jax/galax CPU labels state the affinity stamped in their JSON
# ("cpu_affinity"; an unstamped JSON is the original unpinned 72-thread run).
# A series whose JSONs do not exist (yet) is skipped.
_C1 = "galpy C (1 core)"
_C_NODE = "galpy C (72 threads)"
_C_NODE_FILES = ["galpy_c_cpu.json", "galpy_c_cpu_hiN.json"]
_CUDA_FAST = "CUDA kernel (fast arith.)"  # unlabelled, de-emphasised (named in the caption)
# Three panels (2026-10-28, JB's review): the CPU series, then the TITAN Xp and
# the H100 series, the GPU panels each repeating the whole-node C line in grey.
_CPU_SERIES = [
    (["galpy_c_cpu_1cores.json", "galpy_c_cpu_1cores_hiN.json"], "c", "cpu", None, _C1,
     palette.c_threads_style(1)),
    (_C_NODE_FILES, "c", "cpu", None, _C_NODE, palette.c_threads_style(72)),
    (["galax_cpu.json"], "galax", "cpu", "jit", "galax ({cpu})", None),
    (["jax_cpu.json", "jax_cpu_hiN.json"], "jax", "cpu", "jit",
     "galpy jax ({cpu}, per-orbit steps)", None),
]
_GPU_SERIES = [
    (["jax_gpu.json", "jax_gpu_hiN.json"], "jax", "gpu", "jit", "galpy jax (per-orbit steps)", None),
    (["galax_gpu_perorbit.json", "galax_gpu_perorbit_hiN.json"], "galax", "gpu", "jit",
     "galax (per-orbit steps)", "-."),
    (["galax_gpu.json", "galax_gpu_hiN.json"], "galax", "gpu", "jit", "galax (batched steps)", None),
    # experimental one-thread-per-orbit CUDA kernel compiled from galpy's C forces +
    # dop853 (experiments/cuda_orbits; not a galpy backend, not differentiable), shared
    # with G12: a direct port (solid, labelled) and fast arithmetic (long-dashed,
    # named in the caption); wall incl. H2D + D2H (pinned)
    (["cuda_kernel_lowN.json", "cuda_kernel_hiN.json"], "cuda", "gpu", None, "CUDA kernel (direct port)", None),
    (["cuda_kernel_fast_lowN.json", "cuda_kernel_fast_hiN.json"], "cuda", "gpu", None, _CUDA_FAST, palette.CUDA_FAST_STYLE),
]
_PANELS = [("(a) CPU", "cpu", ""), ("(b) NVIDIA TITAN Xp", "gpu", ""), ("(c) NVIDIA H100", "gpu", "h100")]
_NMAX = 3e6  # common right edge (the H100 CUDA scans also have N=1e7)


def _cpu_tag(d):
    """Cores a jax/galax CPU series could use: its JSON's ``cpu_affinity`` stamp
    (XLA:CPU's pool spans the affinity); unstamped = the unpinned 72-thread run."""
    aff = d.get("cpu_affinity")
    if aff is None:
        return "≤72 threads"
    return "1 core" if aff == 1 else f"≤{aff} cores"


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
    # print size: 7.0 in wide (figure*), three stacked panels of the original
    # panel's proportions, each with its lines labelled directly in a column at
    # the right (elbow leaders); the frame is shared with G12
    fig, axes = plt.subplots(
        3, 1, figsize=layout.figsize(PAPER, (7.0, 9.0), (9.0, 13.0)), sharex=True
    )
    fs_line = layout.FS["line"] if PAPER else 6.5
    fs_note = layout.FS["note"] if PAPER else 6.5

    def _two_line(lab):
        return lab.replace(" (", "\n(") if PAPER else lab

    def _series_xy(d):
        res = sorted((r for r in d["results"] if r["N"] <= _NMAX), key=lambda r: r["N"])
        Ns = numpy.array([r["N"] for r in res], dtype=float)
        wall = numpy.array([r["wall_s"] for r in res], dtype=float)
        return Ns, wall

    def _plot(ax, series, sub):
        for fnames, fw, dev, mode, lab, ls in series:
            d = _load(fnames, sub)
            if d is None or not d.get("results"):
                continue
            Ns, wall = _series_xy(d)
            d_ = "h100" if sub == "h100" else dev
            st = palette.style(fw, d_, mode)
            if isinstance(ls, dict):
                st.update(ls)
            elif ls:
                st["ls"] = ls
            ax.loglog(Ns, wall, ms=4, zorder=3, label=_two_line(lab.format(cpu=_cpu_tag(d))), **st)

    node_label = _two_line(_C_NODE)
    for ax, (title, kind, sub) in zip(axes, _PANELS):
        if kind == "cpu":
            _plot(ax, _CPU_SERIES, "")
        else:
            d = _load(_C_NODE_FILES)
            if d is not None:
                Ns, wall = _series_xy(d)
                ax.loglog(Ns, wall, ms=4, zorder=2, label=node_label, **palette.c_threads_style(72))
            _plot(ax, _GPU_SERIES, sub)
        ax.set_title(title, loc="left")
        ax.set_ylabel(r"wall time for the batch  (s)")
        layout.apply_frame(ax, "orbit_speed")
        # constant per-orbit-cost reference lines (slope-1 in log-log): wall = N * tau
        refs = layout.cost_lines(ax, layout.last_costs(ax, kind="time"), kind="time")
        ax.text(
            0.025, 0.975,
            "grey dotted: " + layout.fmt_seconds(refs) + " per orbit",
            transform=ax.transAxes, fontsize=fs_note, color="0.5", va="top", ha="left",
        )
    axes[-1].set_xlabel(r"batch size  $N$  (number of orbits)")
    layout.faster_arrow(axes[0], "down", xy=(0.94, 0.42), length=0.16)
    if not PAPER:
        fig.suptitle(r"MWPotential2014 orbit integration, 2 Gyr: galpy vs galax")
    fig.tight_layout()
    for ax in axes:
        # the whole-node C line runs below every other line: label it on the plot,
        # to the lower right of its N=100 point (clear of the line, which rises to
        # the right); the rest are labelled in a column right of the axes
        node = [ln for ln in ax.get_lines() if ln.get_label() == node_label]
        for ln in node:
            x, y = (numpy.asarray(v, dtype=float) for v in ln.get_data())
            k = int(numpy.argmin(abs(numpy.log(x / 100.0))))
            ax.annotate(ln.get_label(), xy=(x[k], y[k]), xytext=(6, -6),
                        textcoords="offset points", fontsize=fs_line,
                        color=ln.get_color(), ha="left", va="top")
        palette.label_lines(ax, [(ln, ln.get_label()) for ln in ax.get_lines()
                                 if not ln.get_label().startswith("_") and ln not in node
                                 and "fast arith." not in ln.get_label()],
                            fontsize=fs_line, min_sep_pt=1.3 * fs_line if PAPER else None,
                            order="elbow", pad_pt=18)
    if PAPER:
        layout.fit_right(fig)
    return fig
