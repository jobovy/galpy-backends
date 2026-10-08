"""G12 -- galpy's OWN in-backend orbit integrator: GPU vs CPU vs compiled C.

The analogue of G8, but for galpy's #906 differentiable in-backend integrator
(``galpy.backend._reference.integrate_orbit``) rather than galax. Same experiment
as G8: MWPotential2014, a Sun-like orbit integrated for 2 Gyr over a batch of N
(slightly perturbed) initial conditions; wall time for the whole batch vs N.

Five series, all integrating the *same* backend-agnostic galpy forces:

  * galpy (C, CPU): the compiled ``dop853_c`` integrator -- the classical baseline.
    The in-backend EOM is modelled on this C integrator (rectangular variables).
    Its orbit loop is OpenMP-parallel (omp_get_max_threads(); ``numcores`` does not
    reach it), so it is drawn on ONE core (OMP_NUM_THREADS=1 + taskset;
    galpy_c_cpu_1cores*.json, 2.2-2.4 ms/orbit), on 16 pinned threads when
    galpy_c_cpu_16cores*.json exists (_SHOW_C16), and on the whole node (unpinned,
    72 hardware threads; galpy_c_cpu*.json, ~0.08-0.1 ms/orbit).
  * galpy in-backend (JAX, CPU): the diffrax Dopri8 path, ``jax.vmap``-ed over the
    batch and jit-fused, run on the CPU -- isolates the framework/vectorisation cost
    from the device.
  * galpy in-backend (JAX, GPU; Dopri8, shipped): the same, on the NVIDIA TITAN Xp
    (sm_61). This is what galpy SHIPS on jax (diffrax Dopri8, 8th order).
  * galpy in-backend (JAX, GPU; Dopri5, same-work): diffrax Dopri5 (5th order)
    through the SAME backend forces, jit+vmap, GPU. This is the apples-to-apples
    match to the torch line: galpy ships Dopri8 on jax but dopri5 on torch, so the
    shipped jax-vs-torch crossover is NOT same-work; this line IS (same 5th-order
    method, matching nfev/steps -- see below). Local to g12 (measure/jax_dopri5_*).
  * CUDA kernel (TITAN Xp / H100; galpy arith. solid, fast arith. long-dashed): the
    experimental one-thread-per-orbit CUDA port of galpy's C forces and dop853
    (experiments/cuda_orbits), NOT a galpy backend and not differentiable; shown
    for scale (what GPU-native kernels would buy). Same orbit/tolerance/nt=16.
  * galpy in-backend (torch, GPU; dopri5, eager): the torchdiffeq dopri5 path
    (5th order -- torchdiffeq's dopri8 *backward* is buggy, so galpy ships dopri5 on
    torch). torchdiffeq runs EAGER (no jit-fuse of the adaptive-step loop), so every
    step pays full Python dispatch overhead and the GPU is starved at small N --
    correct and differentiable, but slow until the compute-bound regime (see below).

WORK COUNTS (measure/work_counts.py -> data/work_counts.json), for ONE Sun-like orbit
at rtol=atol=1e-8 (accepted steps / total RHS evals = nfev):
  * jax  diffrax Dopri8 : 189 / 3056  (8th order: few big steps, ~13 evals/step)
  * jax  diffrax Dopri5 : 545 / 3547  (5th order: ~3x more steps, ~6 evals/step)
  * torch torchdiffeq dopri5: ~578 total steps / 3470  -- i.e. jax-Dopri5 and
    torch-dopri5 do the SAME WORK (matching step + nfev counts), so any wall-time gap
    between THOSE two lines is pure backend kernel/dispatch speed, not method. Dopri8
    takes ~3x fewer steps than dopri5 but ~13 vs ~6 evals/step, netting a similar (a
    little lower) nfev.

All lines use rtol=atol=1e-8 and per-orbit adaptive stepping. The galpy-C and
galpy-jax (CPU/GPU) series are loaded from ``paper/figures/_shared/orbit_speed/data``
and are byte-for-byte IDENTICAL to G8's -- same orbit (MWPotential2014, Sun-like IC,
2 Gyr = linspace(0, 56.2, nt)), same tolerance (rtol=atol=1e-8), and same nt
convention (lowN regime nt=1000 dense saves; hiN regime nt=16 sparse saves; per-orbit
cost matches and the overlap points agree), so G8 and G12 can be read together.

The story mirrors G8 (re-measured 2026-10-06): 72-thread galpy-C costs ~0.1
ms/orbit after a ~3 ms per-call setup and is the fastest line at every measured N
on this float64-limited consumer GPU (one-core C: 2.2-2.4 ms/orbit, overtaken by
every GPU line at N~1e3); the jit-fused JAX-GPU lines amortise their
~1.5 s compile/launch floor down to ~0.21 ms/orbit (Dopri8) and ~0.13 ms/orbit
(same-work Dopri5) at large N. (The June version had ~2.1 ms/orbit for C: the
one-core cost, not a slow build.) The JAX-CPU line (one core, 2026-10-08) shows
the device contribution. On the H100 (Trillium) the GPU lines drop 9-12x at large
N and pass the 72-thread C node at N~3e4: jax Dopri8 0.025, Dopri5 0.018, torch dopri5 0.009
ms/orbit at the largest N (see caption.md).

SAME-WORK VERDICT: at the matched 5th-order method, the jit+vmap jax-Dopri5 GPU line
reaches its compute-bound asymptote ~0.13 ms/orbit at N=3e5; the eager torch-dopri5
line bottoms at ~0.07 ms/orbit at N=3e5 in its overhead->compute crossover. So torch's
apparent edge at very large N is NOT a method artefact (same nfev/steps) -- it is the
single shared (6,N) adaptive controller torchdiffeq runs (one dt sequence for the whole
batch, one fused per-step kernel) vs jax's per-orbit vmap-of-solve (N independent
controllers). Same trade-off as g8's galax batched-vs-per-orbit lines; jax keeps
per-orbit stepping for per-orbit error guarantees.

The **torch-GPU line is the eager-vs-compute story**: torchdiffeq runs EAGER (no
jit-fuse of the adaptive-step loop). At small N the batch is *overhead-bound* -- the
wall time is a near-constant ~12 s set by the Python-driven step loop, the GPU is
starved, and ms/orbit falls as ~1/N (here torch is ~tens of x slower per orbit than
the jit-fused jax lines -- exactly why G8 omits a torch line). Pushed to large N the
line crosses into the *compute-bound* regime: the per-step GPU kernels finally
dominate the Python overhead, wall time starts rising ~linearly, and ms/orbit flattens
to torch's true per-orbit asymptote (~0.07 ms/orbit; see the same-work verdict above
for why this beats jax-Dopri5's ~0.13 -- shared vs per-orbit controller, not method).
Correct and fully differentiable; G12 shows it out to the OOM/time cap to make the
eager overhead->compute transition visible.

POTENTIAL SUPPORT: as of the backends worktree (#949, 36c92e8f), MWPotential2014
(PowerSphericalCutoff bulge + MiyamotoNagai disk + NFW halo) is fully
in-backend-integrable on jax AND torch -- the earlier "Plummer/Isochrone only"
limitation no longer holds, so this figure uses the same MW model as G8.

Reproducibility: the shared galpy series (C, jax-Dopri8 CPU/GPU) come from
../_shared/orbit_speed (galpy_c_measure.py, galpy_jax_measure.py for {CPU,GPU} x
{lowN dense, hiN}) and are shared byte-for-byte with G8, run against the
backend-migrated galpy worktree (/var/tmp/backends-merge, which now ships libgalpy so
the C baseline runs from the SAME galpy as the jax/torch lines). The g12-local series
are measure/jax_dopri5_measure.py (the same-work jax-Dopri5 GPU line, {lowN,hiN}),
measure/torch_measure.py (torch dopri5), and measure/work_counts.py (the step/nfev
counts in data/work_counts.json). Each writes a machine- and version-stamped
data/*.json. This plot is byte-stable from those JSONs; ``verify`` only re-renders the
(deterministic) plot. GPU jobs are run STRICTLY SERIALLY (one device-timed process at
a time).
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
# journal version (default): no title / env stamp / long notes; the caption
# carries them. GALPY_JAX_PAPER=0 restores the annotated figure.
PAPER = os.environ.get("GALPY_JAX_PAPER", "1") == "1"

# (json files [low-N first, optional high-N second], legend label, colour, ls, marker).
# The GPU/CPU jax lines merge a low-N run (1000 dense saves, small-N crossover) with
# a high-N run (16 saves, fits the batch in 12 GiB) -- integration is adaptive-step-
# bound not save-bound, so per-orbit cost matches and the overlap points agree.
# galpy_c_* and jax_* (the SHIPPED Dopri8 curve) are SHARED with G8
# (../_shared/orbit_speed/data); the jax_dopri5_* SAME-WORK match and the torch curve
# are local to g12. jax ships diffrax Dopri8 (8th order); torch ships torchdiffeq
# dopri5 (5th order -- torchdiffeq's dopri8 backward is buggy). The jax-Dopri5 line is
# the apples-to-apples (same 5th-order method, matching nfev/steps -- see
# measure/work_counts.py -> data/work_counts.json) comparison to torch-dopri5.
# (json files, framework, device, mode, label suffix, line-style or style override)
# CPU thread counts (paper/ms/notes/CPU_CORES_AUDIT.md): galpy C threads its orbit
# loop with OpenMP (omp_get_max_threads(); ``numcores`` does not reach it), so the
# unpinned galpy_c_cpu*.json ran on all 72 hardware threads; *_1cores* =
# OMP_NUM_THREADS=1 + taskset to one core, *_16cores* = 16 threads on 16 pinned
# cores. XLA:CPU's pool spans the CPU affinity, so the jax CPU label states the
# affinity stamped in its JSON ("cpu_affinity"; unstamped = the unpinned 72-thread
# run). A series whose JSONs do not exist (yet) is skipped.
_C1, _C16, _C_NODE = "1 core", "16 threads", "72 threads"
_CUDA_FAST = "fast arith."  # unlabelled, de-emphasised (named in the caption)
_SHOW_C16 = True
_SERIES = [
    (["galpy_c_cpu_1cores.json", "galpy_c_cpu_1cores_hiN.json"], "c", "cpu", None, _C1,
     palette.c_threads_style(1)),
] + ([
    (["galpy_c_cpu_16cores.json", "galpy_c_cpu_16cores_hiN.json"], "c", "cpu", None, _C16,
     palette.c_threads_style(16)),
] if _SHOW_C16 else []) + [
    (["galpy_c_cpu.json", "galpy_c_cpu_hiN.json"], "c", "cpu", None, _C_NODE,
     palette.c_threads_style(72)),
    (["jax_cpu.json", "jax_cpu_hiN.json"], "jax", "cpu", "jit", "Dopri8, {cpu}", None),
    (["jax_gpu.json", "jax_gpu_hiN.json"], "jax", "gpu", "jit", "Dopri8", None),
    (["jax_dopri5_gpu.json", "jax_dopri5_gpu_hiN.json"], "jax", "gpu", "jit", "Dopri5", "-."),
    (["torch_cuda.json"], "torch", "gpu", "eager", "dopri5", None),
    # experimental one-thread-per-orbit CUDA kernel (experiments/cuda_orbits): galpy's
    # C forces + dop853 compiled for the GPU; galpy arithmetic (_noinl+refill) and the
    # fast-arithmetic build (_fast_noinl+refill+sort). Not differentiable. Wall time
    # incl. H2D + D2H into a reused pinned buffer. Shared with G8 (_shared/orbit_speed/
    # data/[h100/]cuda_kernel*_hiN.json, copies of experiments/cuda_orbits/data/
    # {titan,h100/v2}/scan_{noinl_refill,fast_noinl_refill_sort}_pinned.json)
    (["cuda_kernel_lowN.json", "cuda_kernel_hiN.json"], "cuda", "gpu", None, "galpy arith.", None),
    (["cuda_kernel_fast_lowN.json", "cuda_kernel_fast_hiN.json"], "cuda", "gpu", None, _CUDA_FAST, palette.CUDA_FAST_STYLE),
]
_NMAX = 3e6  # common right edge (the H100 CUDA scans also have N=1e7)


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
    for base in (_SHARED, _DATA):
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


@figure(
    name="g12_inbackend_orbit_gpu",
    track="G",
    seed=0,
    description="MWPotential2014 in-backend orbit integration wall time vs batch size N: galpy(C,CPU) vs galpy in-backend (JAX CPU/GPU, torch GPU) [cross-env measure->json, plot byte-stable]",
)
def make():
    # print size: 7.0 in wide (figure*), text 8-10 pt, two-line labels
    fig, ax = plt.subplots(figsize=layout.figsize(PAPER, (7.0, 3.8), (9.0, 4.5)))
    fs_line = layout.FS["line"] if PAPER else 7
    fs_note = layout.FS["note"] if PAPER else 6.5
    _label = layout.label if PAPER else palette.label

    gpu_meta = _load_one("jax_gpu.json") or _load_one("jax_gpu_hiN.json") or {}
    torch_meta = _load_one("torch_cuda.json") or {}
    h100_meta = _load_one("jax_gpu.json", "h100") or {}
    for fnames, fw, dev, mode, extra, ls in _SERIES:
        for sub, d_ in (("", dev), ("h100", "h100")):
            if sub and dev != "gpu":
                continue
            d = _load(fnames, sub)
            if d is None or not d.get("results"):
                continue
            res = sorted((r for r in d["results"] if r["N"] <= _NMAX), key=lambda r: r["N"])
            Ns = numpy.array([r["N"] for r in res], dtype=float)
            wall = numpy.array([r["wall_s"] for r in res], dtype=float)
            st = palette.style(fw, d_, mode)
            if isinstance(ls, dict):
                st.update(ls)
            elif ls:
                st["ls"] = ls
            ax.loglog(Ns, wall, ms=4, zorder=3,
                      label=_label(fw, d_, mode, extra.format(cpu=_cpu_tag(d))), **st)

    ax.set_xlabel(r"batch size  $N$  (number of orbits)")
    ax.set_ylabel(r"wall time for the batch  (s)")
    if not PAPER:
        ax.set_title(r"galpy in-backend orbit integration (MWPotential2014, 2 Gyr)")

    # constant per-orbit-cost reference lines (slope-1 in log-log): wall = N * tau
    ax.set_ylim(*ax.get_ylim())
    xl = ax.get_xlim()
    # extend the x-axis to the right so the rightmost markers have room and the
    # reference lines / legend region do not crowd the data
    # (stop short of the next decade: a tick label at the right edge would collide
    # with the bottom direct label)
    xl = (xl[0], min(xl[1] * 1.6, 0.9 * 10.0 ** numpy.ceil(numpy.log10(xl[1]))))
    ax.set_xlim(*xl)
    # decade-spaced to bracket every curve's cost per orbit at its largest N
    refs = layout.cost_lines(ax, layout.last_costs(ax, kind="time"), kind="time")
    ax.text(
        0.025, 0.975,
        "grey dotted: " + layout.fmt_seconds(refs) + " per orbit",
        transform=ax.transAxes, fontsize=fs_note, color="0.5", va="top", ha="left",
    )
    # Minimal in-figure stamp: tolerance, GPU, library versions. The integrator
    # details (Dopri8 vs dopri5, same-work nfev counts, eager-vs-compute,
    # controller architecture, compile amortization) live in caption.md.
    if not PAPER:
        ax.text(
            0.17, 0.02,
            f"rtol=atol=1e-8  ·  {gpu_meta.get('gpu', '?')}: jax {gpu_meta.get('jax_version', '?')}"
            f" · torch {torch_meta.get('torch_version', '?')}"
            + (f"\n{h100_meta.get('gpu', 'H100')}: jax {h100_meta.get('jax_version', '?')}"
               f" · torch {(_load_one('torch_cuda.json', 'h100') or {}).get('torch_version', '?')}"
               if h100_meta else ""),
            transform=ax.transAxes, fontsize=6.5, color="0.5", va="bottom", ha="left",
        )
    fig.tight_layout()
    # the all-threads C line runs below every other line: label it on the plot, to
    # the lower right of its N=100 point (clear of the line, which rises to the right)
    node = [ln for ln in ax.get_lines() if ln.get_label() == _label("c", "cpu", None, _C_NODE)]
    for ln in node:
        x, y = (numpy.asarray(v, dtype=float) for v in ln.get_data())
        k = int(numpy.argmin(abs(numpy.log(x / 100.0))))
        ax.annotate(ln.get_label(), xy=(x[k], y[k]), xytext=(6, -6), textcoords="offset points",
                    fontsize=fs_line, color=ln.get_color(), ha="left", va="top")
    palette.label_lines(ax, [(ln, ln.get_label()) for ln in ax.get_lines()
                             if not ln.get_label().startswith("_") and ln not in node
                             and _CUDA_FAST not in ln.get_label()],
                        fontsize=fs_line, min_sep_pt=1.3 * fs_line if PAPER else None,
                        order="elbow", pad_pt=18)
    if PAPER:
        layout.fit_right(fig)
    return fig
