"""gT1 -- CPU thread scaling of orbit integration: galpy C vs jax vs torch.

Every other CPU timing series in the paper runs on ONE pinned core (apples to
apples); galpy C additionally has a 72-thread OpenMP line in g8/g12. This figure
asks what each framework does with more cores, for the g12 high-N case:
MWPotential2014, N Sun-like orbits (the g8/g12 ICs) for 2 Gyr at
rtol = atol = 1e-8 with 16 saved times, N = 1e4 (top row) and 1e5 (bottom row),
on K = 1, 2, 4, 8, 16, 36 and 72 CPUs of the dev box (2 x Xeon E5-2697 v4: 2 x 18
physical cores, 72 hardware threads; CPU n and n+36 are hyper-thread siblings,
even CPUs are socket 0).

  * galpy C: dop853_c, whose orbit loop is OpenMP-parallel (OMP_NUM_THREADS=K).
  * jax jit: galpy's in-backend diffrax Dopri8, jax.vmap over orbits, jit (the g8/g12
    jax line). jaxlib 0.10.1 sizes the XLA:CPU pool from the CPU affinity and ignores
    XLA_FLAGS thread settings, so the affinity is the only control.
  * torch eager: galpy's in-backend torchdiffeq dopri5 (integrate_orbit's default
    torch engine; the batch is one (N, 6) state with a shared controller -- the g12
    torch line and the g15 CPU torch line), torch.set_num_threads(K). The compilable
    torchode path is not used: the paper's orbit torch series are torchdiffeq, and a
    CPU inductor compile per shape takes minutes (gS5).

Core sets (taskset -c; CPU 8 is reserved for other one-core runs): K=1: 2; 2: 2,4; 4: 2,4,6,10; 8: 2,4,6,10-18 even; 16: 0-6 and 10-32 even
(physical cores of socket 0, no hyper-thread siblings); 36: 0-35 (every physical
core, both sockets); 72: 0-71 (all hardware threads). NUMPY_MADVISE_HUGEPAGE=0 throughout;
K<=16 (all node-0 CPUs) under numactl --membind=0, K=36/72 first-touch. A K<=16 run starts only
when the pinned CPUs' hyper-thread twins (CPU+36) are idle (<10%); each call records how busy
the twins and the other CPUs were (busy_twin_cpus, busy_other_cpus). One process per (framework,
K) times N=1e4 then 1e5: the median of 3 warm calls, or 1 call when a call takes
over 60 s (``n_calls`` in the JSON). jax is compiled ahead of time
(.lower().compile(); compile_s recorded, not plotted).

Left: wall time for the batch. Right: speed-up over the same code on one core,
wall(1)/wall(K) (markers), and the cores the process kept busy, CPU time / wall
time over all its threads (thin line, no markers); grey dotted = ideal (= K).
CPU time includes spin-waiting worker threads, so busy cores above the speed-up
are cycles spent waiting, not integrating.

measure/measure_threads.py -> data/{c,jax,torch}.json (per-point wall/cpu, every
call, final positions as a cross-check, env stamps: affinity, thread env, galpy
commit, versions, load averages). The plot is byte-stable from those JSONs.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy

from paper.figures._harness import layout, palette
from paper.figures._harness.figure import figure

_DATA = os.path.join(os.path.dirname(__file__), "data")
PAPER = os.environ.get("GALPY_JAX_PAPER", "1") == "1"

# (file, framework, mode, label extra)
_SERIES = [
    ("c", "c", None, "dop853_c"),
    ("jax", "jax", "jit", "Dopri8"),
    ("torch", "torch", "eager", "dopri5"),
]
_NS = (10000, 100000)
_KS = (1, 2, 4, 8, 16, 36, 72)


def _load(fname):
    path = os.path.join(_DATA, fname + ".json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def _series(d, N):
    pts = sorted((p for p in d["points"] if p["N"] == N), key=lambda p: p["k"])
    k = numpy.array([p["k"] for p in pts], dtype=float)
    wall = numpy.array([p["wall_s"] for p in pts])
    util = numpy.array([p["cpu_s"] / p["wall_s"] for p in pts])
    return k, wall, util


@figure(
    name="gT1_cpu_thread_scaling",
    track="G",
    seed=0,
    description="CPU thread scaling of MWPotential2014 orbit integration (N=1e4, 1e5; 1-72 CPUs): galpy C dop853_c (OpenMP) vs jax jit Dopri8 (XLA:CPU) vs torch eager dopri5 [measure->json, plot byte-stable]",
)
def make():
    fs_line = layout.FS["line"] if PAPER else 7
    fs_note = layout.FS["note"] if PAPER else 6.5
    fig, axs = plt.subplots(2, 2, figsize=layout.figsize(PAPER, (7.0, 5.0), (9.5, 6.5)),
                            sharex=True)
    data = {f: _load(f) for f, *_ in _SERIES}
    for row, N in enumerate(_NS):
        axw, axs_ = axs[row]
        for f, fw, mode, extra in _SERIES:
            d = data[f]
            if d is None:
                continue
            k, wall, util = _series(d, N)
            if not len(k):
                continue
            st = palette.style(fw, "cpu", mode)
            lab = layout.label(fw, None, mode, extra) if PAPER else palette.label(fw, "cpu", mode, extra)
            axw.loglog(k, wall, ms=4, zorder=3, label=lab, **st)
            if k[0] == 1:
                axs_.loglog(k, wall[0] / wall, ms=4, zorder=3, label=lab, **st)
                axs_.loglog(k, util, color=st["color"], ls=st["ls"], lw=0.8, alpha=0.55,
                            marker="None", zorder=2, label="_busy")
        nlab = r"$N = 10^{%d}$ orbits" % round(numpy.log10(N))
        axw.text(0.03, 0.05, nlab, transform=axw.transAxes, fontsize=fs_note + 1,
                 va="bottom", ha="left")
        axw.set_ylabel("wall time for the batch  (s)")
        axs_.set_ylabel("speed-up over 1 core")
        kk = numpy.array([1.0, 72.0])
        axs_.loglog(kk, kk, ls=":", lw=0.9, color="0.55", zorder=1, label="_ideal")
        axs_.set_ylim(0.7, 100)
        if row == 0:
            axs_.text(0.97 * 72 ** 0.5 * 0 + 1.15, 1.6, "ideal", rotation=0, fontsize=fs_note,
                      color="0.5", va="bottom", ha="left", transform=axs_.transData)
            axs_.text(0.03, 0.97, "markers: wall(1 core) / wall\nthin: cores busy (CPU / wall)",
                      transform=axs_.transAxes, fontsize=fs_note, color="0.4", va="top", ha="left")
        for ax in axw, axs_:
            ax.axvspan(36, 80, color="0.93", zorder=0, lw=0)
            ax.axvline(18, color="0.8", lw=0.6, ls="-", zorder=0)
    for ax in axs[0]:
        ax.text(18 * 1.06, 0.97, "1 socket", transform=ax.get_xaxis_transform(), fontsize=fs_note - 0.5,
                color="0.5", va="top", ha="left")
        ax.text(36 * 1.06, 0.97, "hyper-\nthreads", transform=ax.get_xaxis_transform(),
                fontsize=fs_note - 0.5, color="0.5", va="top", ha="left")
    for ax in axs[1]:
        ax.set_xlabel("CPU cores  $K$")
        ax.set_xticks(_KS)
        ax.set_xticklabels([str(k) for k in _KS])
        ax.minorticks_off()
        ax.set_xlim(0.85, 80)
    axs[0][0].set_title("(a) wall time", fontsize=layout.FS["title"] if PAPER else None)
    axs[0][1].set_title("(b) speed-up and busy cores", fontsize=layout.FS["title"] if PAPER else None)
    fig.tight_layout()
    for ax in axs[:, 1]:
        palette.label_lines(ax, [(ln, ln.get_label()) for ln in ax.get_lines()
                                 if not ln.get_label().startswith("_")],
                            fontsize=fs_line, min_sep_pt=1.3 * fs_line if PAPER else None)
    if PAPER:
        layout.fit_right(fig)
    return fig
