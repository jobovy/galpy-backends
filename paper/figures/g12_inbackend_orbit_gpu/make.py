"""G12 -- galpy's OWN in-backend orbit integrator: GPU vs CPU vs compiled C.

The analogue of G8, but for galpy's #906 differentiable in-backend integrator
(`galpy.backend._reference.integrate_orbit`). Same setup and protocol
as G8: MWPotential2014, a Sun-like orbit integrated for 2 Gyr over a batch of N
initial conditions, wall time for the whole batch vs N.

Series, all integrating the *same* backend-agnostic galpy forces:
  * galpy C (dop853_c): one core, 16 threads, the whole node (OpenMP over orbits)
  * galpy in-backend jax (diffrax Dopri8), compiled, on one CPU core
  * galpy in-backend jax (Dopri8 and, dash-dotted, Dopri5), compiled, on the GPUs
  * galpy in-backend torch (torchdiffeq dopri5), eager, on the GPUs
  * an experimental one-thread-per-orbit CUDA kernel (not part of galpy)
  * galax (its diffrax solve; G8's series, folded in here on JB's review of
    2026-10-29): on one CPU core, and on the GPUs with its default batched
    step-size control and with per-orbit control as galpy's

Layout (2026-10-28, JB's review): three panels, (a) the CPU series, (b) the
TITAN Xp series and (c) the H100 series, the GPU panels each repeating the
whole-node C line in grey for reference, so that no panel carries more than six
lines. All three share the axes of the orbit_speed frame (as G8 does) and a
"faster" arrow marks the direction of lower wall time.

WORK COUNTS (measure/work_counts.py -> data/work_counts.json), for ONE Sun-like orbit
at rtol=atol=1e-8 (accepted steps / total RHS evals = nfev):
  * jax Dopri8 (diffrax, shipped):   ~250 steps, ~3250 nfev
  * jax Dopri5 (diffrax, same-work): ~900 steps, ~5600 nfev
  * torch dopri5 (torchdiffeq):      ~900 steps, ~5600 nfev
The jax-Dopri5 line is the apples-to-apples (same 5th-order method, matching
nfev/steps) comparison to torch-dopri5; jax ships Dopri8 (8th order), torch ships
torchdiffeq dopri5 (torchdiffeq's dopri8 backward is buggy).

All lines use rtol=atol=1e-8 and per-orbit adaptive stepping. The galpy-C and
galpy-jax (CPU/GPU) series are loaded from ``paper/figures/_shared/orbit_speed/data``
and are byte-for-byte IDENTICAL to G8's -- same orbit (MWPotential2014, Sun-like IC,
2 Gyr), tolerance (1e-8), and nt convention (lowN regime nt=1000 dense saves; hiN
regime nt=16 sparse saves; per-orbit cost matches and the overlap points agree),
so G8 and G12 can be read together.

Reproducibility: the shared galpy series (C, jax-Dopri8 CPU/GPU) come from
``_shared/orbit_speed/{galpy_c_measure,galpy_jax_measure}.py``; the jax-Dopri5 and
torch series from ``measure/`` here; the CUDA kernel from ``experiments/cuda_orbits``
(copies under ``_shared/orbit_speed/data/[h100/]cuda_kernel*``). Every series is a
machine-stamped JSON; this script only plots them (byte-stable).
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
_GALAX = os.path.join(os.path.dirname(__file__), "..", "g8_galax_orbit_speed", "data")
# journal version (default): no title / env stamp / long notes; the caption
# carries them. GALPY_JAX_PAPER=0 restores the annotated figure.
PAPER = os.environ.get("GALPY_JAX_PAPER", "1") == "1"

# (json files [low-N first, optional high-N second], framework, device, mode,
# label suffix, line-style or style override). The GPU/CPU jax lines merge a
# low-N run (1000 dense saves, small-N crossover) with a high-N run (16 saves,
# fits the batch in 12 GiB) -- integration is adaptive-step-bound not save-bound,
# so per-orbit cost matches and the overlap points agree.
# CPU thread counts (paper/ms/notes/CPU_CORES_AUDIT.md): galpy C threads its orbit
# loop with OpenMP (omp_get_max_threads(); ``numcores`` does not reach it), so the
# unpinned galpy_c_cpu*.json ran on all 72 hardware threads; *_1cores* =
# OMP_NUM_THREADS=1 + taskset to one core, *_16cores* = 16 threads on 16 pinned
# cores. XLA:CPU's pool spans the CPU affinity, so the jax CPU label states the
# affinity stamped in its JSON ("cpu_affinity"; unstamped = the unpinned 72-thread
# run). A series whose JSONs do not exist (yet) is skipped.
_C1, _C16, _C_NODE = "1 core", "16 threads", "36 threads"
_SYMPLEC_LS = (0, (4, 1.5, 1, 1.5, 1, 1.5))  # dash-dot-dot
_CUDA_FAST = "GPU special functions"  # labelled since 2026-10-28 (JB)
_C_NODE_FILES = ["galpy_c_cpu_36cores.json", "galpy_c_cpu_36cores_hiN.json"]
_CPU_SERIES = [
    (["galpy_c_cpu_1cores.json", "galpy_c_cpu_1cores_hiN.json"], "c", "cpu", None, _C1,
     palette.c_threads_style(1)),
    (["galpy_c_cpu_16cores.json", "galpy_c_cpu_16cores_hiN.json"], "c", "cpu", None, _C16,
     palette.c_threads_style(16)),
    (_C_NODE_FILES, "c", "cpu", None, _C_NODE, palette.c_threads_style(36, node_threads=36)),
    (["jax_cpu.json", "jax_cpu_hiN.json"], "jax", "cpu", "jit", "Dopri8, {cpu}", None),
    # the torch CPU series (measure/torch_measure.py with CUDA_VISIBLE_DEVICES=)
    (["torch_cpu.json"], "torch", "cpu", "eager", "dopri5, 1 core", None),
    # the in-backend constant-step sixth-order symplectic integrator
    # (_shared/orbit_speed/galpy_symplec_measure.py: 450 steps, the step at which
    # its worst-case error matches the Dopri8 line's), dash-dot-dot
    (["symplec6_jax_cpu.json"], "jax", "cpu", "jit", "symplec6, {cpu}", _SYMPLEC_LS),
    (["galax_cpu.json"], "galax", "cpu", "jit", "{cpu}", None),
]
_GPU_SERIES = [
    (["jax_gpu.json", "jax_gpu_hiN.json"], "jax", "gpu", "jit", "Dopri8", None),
    (["jax_dopri5_gpu.json", "jax_dopri5_gpu_hiN.json"], "jax", "gpu", "jit", "Dopri5", "-."),
    (["symplec6_jax_gpu.json"], "jax", "gpu", "jit", "symplec6", _SYMPLEC_LS),
    (["torch_cuda.json"], "torch", "gpu", "eager", "dopri5", None),
    # experimental one-thread-per-orbit CUDA kernel (experiments/cuda_orbits): galpy's
    # C forces + dop853 compiled for the GPU; a direct port (_noinl+refill) and the
    # fast-arithmetic build (_fast_noinl+refill+sort). Not differentiable. Wall time
    # incl. H2D + D2H into a reused pinned buffer. Shared with G8.
    (["cuda_kernel_lowN.json", "cuda_kernel_hiN.json"], "cuda", "gpu", None, "direct port", None),
    (["cuda_kernel_fast_lowN.json", "cuda_kernel_fast_hiN.json"], "cuda", "gpu", None, _CUDA_FAST,
     {"ls": "--", "mfc": "white"}),
    # galax: per-orbit step-size control as galpy's (dash-dotted) and its default
    # control of the whole batch by one step sequence (solid)
    (["galax_gpu_perorbit.json", "galax_gpu_perorbit_hiN.json"], "galax", "gpu", "jit",
     "per-orbit steps", "-."),
    (["galax_gpu.json", "galax_gpu_hiN.json"], "galax", "gpu", "jit", "batched steps", None),
]
_NMAX = 3e6  # common right edge (the H100 CUDA scans also have N=1e7)
_PANELS = [("(a) CPU", "cpu", ""), ("(b) NVIDIA TITAN Xp GPU", "gpu", ""),
           ("(c) NVIDIA H100 GPU", "gpu", "h100")]


def _cpu_tag(d):
    """Cores a jax CPU series could use: its JSON's ``cpu_affinity`` stamp (XLA:CPU's
    pool spans the affinity); unstamped = the unpinned 72-thread run."""
    aff = d.get("cpu_affinity")
    if aff is None:
        return "≤72 threads"
    return "1 core" if aff == 1 else f"≤{aff} cores"


def _load_one(fname, sub=""):
    # shared galpy JSONs live in _shared/orbit_speed/data; torch JSON is local.
    # sub="h100": the same measurement on an H100 (data/h100/)
    for base in (_SHARED, _DATA, _GALAX):
        path = os.path.join(base, sub, fname)
        if os.path.exists(path):
            with open(path) as fh:
                return json.load(fh)
    return None


def _load(fnames, sub=""):
    merged, meta = {}, None
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


def _series_xy(d):
    res = sorted((r for r in d["results"] if r["N"] <= _NMAX), key=lambda r: r["N"])
    Ns = numpy.array([r["N"] for r in res], dtype=float)
    wall = numpy.array([r["wall_s"] for r in res], dtype=float)
    return Ns, wall


def _plot(ax, series, sub, label_fn):
    """Draw ``series`` (measured on the device ``sub`` selects) on ``ax``; the
    device is left out of the labels because the panel title names it."""
    for fnames, fw, dev, mode, extra, ls in series:
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
        ax.loglog(Ns, wall, ms=4, zorder=3,
                  label=label_fn(fw, None, mode, extra.format(cpu=_cpu_tag(d))), **st)


@figure(
    name="g12_inbackend_orbit_gpu",
    track="G",
    seed=0,
    description="MWPotential2014 in-backend orbit integration wall time vs batch size N, in three stacked panels (CPU, TITAN Xp, H100): galpy(C,CPU) vs galpy in-backend (JAX CPU/GPU, torch GPU) [cross-env measure->json, plot byte-stable]",
)
def make():
    # print size: 7.0 in wide (figure*), three stacked panels, each named inside
    # its axes (2026-11-02, JB: the figure and its caption must fit one page),
    # each with its lines labelled directly in a column at the right, with a
    # sample of the line's style next to each label
    fig, axes = plt.subplots(
        3, 1, figsize=layout.figsize(PAPER, (7.0, 7.2), (9.0, 13.0)), sharex=True
    )
    fs_line = layout.FS["line"] if PAPER else 7
    fs_note = layout.FS["note"] if PAPER else 6.5
    _label = layout.label if PAPER else palette.label

    node_label = _label("c", None, None, _C_NODE)
    for ax, (title, kind, sub) in zip(axes, _PANELS):
        if kind == "cpu":
            _plot(ax, _CPU_SERIES, "", _label)
        else:
            # the whole-node C line again, for reference, in the C grey
            d = _load(_C_NODE_FILES)
            if d is not None:
                Ns, wall = _series_xy(d)
                st = palette.c_threads_style(36, node_threads=36)
                ax.loglog(Ns, wall, ms=4, zorder=2, label=node_label, **st)
            _plot(ax, _GPU_SERIES, sub, _label)
        if PAPER:
            ax.text(0.025, 0.975, title, transform=ax.transAxes, fontsize=layout.FS["title"],
                    va="top", ha="left")
        else:
            ax.set_title(title, loc="left")
        ax.set_ylabel(r"wall time for the batch  (s)")
        # constant per-orbit-cost reference lines (slope-1 in log-log): wall = N * tau
        # same frame as G8, so the two read as a pair
        layout.apply_frame(ax, "orbit_speed")
        costs = layout.last_costs(ax, kind="time")
        if sub == "h100":
            # no 0.1 us line: the fastest H100 series is just under 1 us per orbit
            costs = [c for c in costs if c >= 1e-6]
        refs = layout.cost_lines(ax, costs, kind="time")
        ax.text(
            0.025, 0.905 if PAPER else 0.975,
            "grey dotted: " + layout.fmt_seconds(refs) + " per orbit",
            transform=ax.transAxes, fontsize=fs_note, color="0.5", va="top", ha="left",
        )
    axes[-1].set_xlabel(r"batch size  $N$  (number of orbits)")
    for ax in axes:
        layout.faster_arrow(ax, "down", xy=(0.94, 0.42), length=0.16)
    if not PAPER:
        fig.suptitle(r"galpy in-backend orbit integration (MWPotential2014, 2 Gyr)")
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
                                 if not ln.get_label().startswith("_") and ln not in node],
                            # nine two-line labels per panel: a smaller font (and
                            # the default spacing), so that the column fits the panel
                            fontsize=6.5 if PAPER else fs_line, min_sep_pt=1.25 * 6.5 if PAPER else None,
                            order="edge", pad_pt=12, swatch_pt=20 if PAPER else None)
    if PAPER:
        layout.fit_right(fig)
    return fig
