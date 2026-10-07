"""GV2 -- actionAngleVerticalInverse speed: evaluation throughput and construction
cost, numpy vs native jax/torch (CPU & GPU).

The 1D inverse action-angle map (J, theta) -> (z, v_z) of
``actionAngleVerticalInverse`` (momentum-matched torus map) is now built and
evaluated natively under jax/torch (see gV1 for what that buys: derivatives in
the potential's parameters, J and theta). This figure is what it costs.

Setup: ``IsothermalDiskPotential(amp=1, sigma=0.5)``, energies linspace(0.05, 2,
nE), setup_interp=True, default mm_npt=40. The jax/torch families are built from
a potential whose parameters are jax/torch arrays on the target device (the
native construction), the numpy family from Python floats (the numpy/scipy
construction). float64 throughout.

(a) Evaluation: wall time of ``xvFreqs(J=0.5, angles)`` -- ONE torus at N angles
    -- vs N, for numpy (CPU), jax eager, jax under the caller's ``jax.jit``
    (compile excluded; galpy adds no internal jit), torch eager, and on the GPU
    jax-jit and torch. "Many tori" (dash-dot): ``jax.jit(jax.vmap(...))`` over
    M = N/16 actions of the per-torus map at 16 fixed angles (works as is: one
    batched root find over all tori), against a Python loop over the M tori on
    numpy. (``torch.func.vmap`` of the same map failed when measured before
    #1641: the root finder's ``xp.sign`` is array_api_compat.torch's sign, which
    indexes with a boolean mask -- a dynamic-shape op torch's vmap cannot batch;
    the Newton step still calls ``xp.sign``, so no torch vmap series.)

    The anomaly root find dominates the evaluation. Since galpy #1641 the backend
    inverts the angle relation with a safeguarded Newton iteration started at the
    requested angle (backend brentq(guess=...), 8 steps; numpy does the same),
    replacing the fixed-schedule bisection (43 halvings eagerly, 100 under jit)
    of BUG_vi_backend_eval_bisection.md. Single-threaded, re-measured 2026-10-07
    on an idle box: numpy is fastest up to N~3000 (N=10: numpy 0.8 ms, jax jit
    2.3 ms, torch 22 ms, jax eager 0.20 s); jax jit overtakes numpy at N~1e4 and
    at N=1e6 takes 2.0 s vs numpy 9.6 s, jax eager 14 s, torch 51 s. jit beats
    eager at every N. The many-tori vmap costs ~4.5x the one-torus jit per point
    on the CPU (9.1 s at N=1e6) but beats the numpy per-torus loop (~0.9 ms per
    torus, 57 s) from ~200 tori on; on the GPU it costs nearly the same as the
    one-torus jit. N=1e6 on the GPU: jax jit 0.31 s (TITAN Xp) / 11 ms (H100),
    torch 0.94 s / 70 ms.

(b) Construction: wall time to build the family vs the number of energies nE,
    numpy vs native jax (eager and jitted) and torch, CPU and GPU. Open markers
    are the FIRST call at each nE: for eager jax this includes compiling every
    op for the new shapes, for jit the full trace+compile (both one-time per
    shape); filled markers are warm calls. At nE=256 (warm, CPU): numpy 2.5 s,
    torch 0.25 s, jax eager 0.36 s, jax jit 0.48 s; jax jit on the GPU: 9-54 ms
    (TITAN Xp) and 3.6-5.8 ms (H100) over the whole nE range.

Plot: colours from _harness/palette.py (framework x device; jit solid, eager
dashed, many-tori variants dash-dot), lines labelled directly. The GPU series
re-measured on an H100 (data/h100/, Trillium) are drawn as extra lines.

CPU threading: every CPU series is SINGLE-THREADED (OPENBLAS_NUM_THREADS=
OMP_NUM_THREADS=1, torch.set_num_threads(1), XLA eigen/intra-op threads = 1).
The measuring box is shared; when it is oversubscribed, multi-threaded BLAS and
torch thrash (measured at load ~160 on 72 hardware threads: numpy construction
nE=31 0.6 s -> 19 s with 4 OpenBLAS threads, torch N=1e3 evaluation 0.08 s ->
3.3 s with 4 torch threads), so single-thread numbers are the stable, comparable
per-core measure. The current data were measured with the box idle (load ~5).
GPU series are unaffected.

Timing protocol: one warm-up (jit compile timed separately and excluded), then
the median of 3 calls (bench.time_call, device-synchronised); a call longer
than 5 s is timed once, and a series stops after a call exceeding 60 s (NaN
beyond).

Reproducibility: measure -> data/gV2_vertical_inverse_speed.json (stamped with
versions, CPU, GPU, thread settings, load average); the plot is byte-stable
from the JSON. Re-measure all by deleting the JSON and running
``OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m paper.figures._harness run
gV2_vertical_inverse_speed``; measure only the GPU series into the existing JSON
with ``OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python
paper/figures/gV2_vertical_inverse_speed/make.py --devices gpu``
(merges, then re-run ``run`` to re-plot). Add GPU evaluation points at larger N
(e.g. on a bigger GPU) with ``... make.py --devices gpu --Ns 1e7``: only the GPU
evaluation series are measured, at those N only, and merged into the JSON (its
N grid is extended, other series NaN there; env stamps kept, the run recorded in
``extra_points``).
"""

import json
import math
import os
import platform
import sys
import time

import matplotlib.pyplot as plt
import numpy

from paper.figures._harness import bench, layout, palette
from paper.figures._harness.figure import figure

_DATA = os.path.join(os.path.dirname(__file__), "data")
_JSON = os.path.join(_DATA, "gV2_vertical_inverse_speed.json")
# the GPU series re-measured on an H100 (Trillium), drawn as extra lines
_JSON_H100 = os.path.join(_DATA, "h100", "gV2_vertical_inverse_speed.json")
# journal version (default): printed size, no env stamp / long notes; the caption
# carries them. GALPY_JAX_PAPER=0 restores the annotated figure.
PAPER = os.environ.get("GALPY_JAX_PAPER", "1") == "1"

_AMP, _SIG = 1.0, 0.5
_EMIN, _EMAX = 0.05, 2.0
_NE_EVAL = 31
_J = 0.5
_NS = [10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000, 300000, 1000000]
_NANG_VMAP = 16
_MS = [max(1, round(n / 16)) for n in _NS]   # tori per call for the many-tori series
_JRANGE_VMAP = (0.05, 0.6)
_NES = [8, 16, 32, 64, 128, 256]
_SLOW = 5.0     # a call slower than this is timed once
_STOP = 60.0    # a call slower than this ends its series

# palette.py: colour = framework x device; jit solid, eager dashed; the
# many-tori variants (vmap / Python loop over tori) dash-dot.
# name: (framework, mode, variant label or None)
_SPEC = {
    "numpy (CPU)": ("numpy", None, None),
    "numpy (CPU, loop over tori)": ("numpy", None, "loop over tori"),
    "jax (CPU, eager)": ("jax", "eager", None),
    "jax (CPU, jit)": ("jax", "jit", None),
    "jax (CPU, jit, vmap over tori)": ("jax", "jit", "vmap over tori"),
    "torch (CPU)": ("torch", "eager", None),
    "jax (GPU, eager)": ("jax", "eager", None),
    "jax (GPU, jit)": ("jax", "jit", None),
    "jax (GPU, jit, vmap over tori)": ("jax", "jit", "vmap over tori"),
    "torch (GPU)": ("torch", "eager", None),
}
_EVAL = ["numpy (CPU)", "numpy (CPU, loop over tori)", "jax (CPU, eager)", "jax (CPU, jit)",
         "jax (CPU, jit, vmap over tori)", "torch (CPU)", "jax (GPU, jit)",
         "jax (GPU, jit, vmap over tori)", "torch (GPU)"]
_CONS = ["numpy (CPU)", "jax (CPU, eager)", "jax (CPU, jit)", "torch (CPU)",
         "jax (GPU, eager)", "jax (GPU, jit)", "torch (GPU)"]


def _style(name, dev):
    fw, mode, var = _SPEC[name]
    st = palette.style(fw, dev, mode)
    if var:
        st["ls"] = "-."
        st["marker"] = "x" if fw == "numpy" else st["marker"]
    if PAPER:  # short variant tags; the caption explains them
        var = {"loop over tori": "loop", "vmap over tori": "vmap"}.get(var, var)
    return st, palette.label(fw, dev, mode, var or "")


def _sources(data):
    """[(data, device override)]: the main JSON, then the H100 GPU series."""
    out = [(data, None)]
    if os.path.exists(_JSON_H100):
        with open(_JSON_H100) as fh:
            out.append((json.load(fh), "h100"))
    return out


def _es(n):
    return numpy.linspace(_EMIN, _EMAX, n)


def _timed(fn, sync="none"):
    """(seconds, stop): warm-up call timed; slow -> one more timed call; else the
    median of 3 (bench.time_call)."""
    t0 = time.perf_counter()
    out = fn()
    if sync == "jax":
        bench._sync_jax(out)
    elif sync == "torch":
        bench._sync_torch()
    tw = time.perf_counter() - t0
    if tw > _STOP:
        return tw, True
    if tw > _SLOW:
        t0 = time.perf_counter()
        out = fn()
        if sync == "jax":
            bench._sync_jax(out)
        elif sync == "torch":
            bench._sync_torch()
        t1 = time.perf_counter() - t0
        return t1, t1 > _STOP
    return bench.time_call(fn, warmup=0, repeats=3, sync=sync)["median"], False


def _sweep(Ns, make_call, sync, label):
    """make_call(N) -> zero-arg callable (inputs ready on device). NaN after a
    >60 s call or an error (e.g. OOM)."""
    out, stop = [], False
    for N in Ns:
        if stop:
            out.append(float("nan"))
            continue
        try:
            t, stop = _timed(make_call(N), sync)
        except Exception as e:  # OOM etc.
            print(f"  {label} N={N}: skip ({type(e).__name__}: {str(e)[:60]})")
            _empty_cache()
            t, stop = float("nan"), True
        print(f"  {label} N={N}: {t:.3e} s", flush=True)
        out.append(t)
    return out


def _empty_cache():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _env():
    env = {"python": sys.version.split()[0], "machine": platform.machine()}
    for m in ("numpy", "scipy", "jax", "torch"):
        try:
            env[m] = __import__(m).__version__
        except Exception:
            pass
    try:
        import galpy

        env["galpy_path"] = os.path.dirname(galpy.__file__)
    except Exception:
        pass
    try:
        with open("/proc/cpuinfo") as fh:
            env["cpu"] = next(
                ln.split(":", 1)[1].strip() for ln in fh if ln.startswith("model name")
            )
    except Exception:
        env["cpu"] = "?"
    try:
        env["cpu_affinity"] = len(os.sched_getaffinity(0))
    except Exception:
        pass
    for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "XLA_FLAGS"):
        env[v] = os.environ.get(v, "unset")
    try:
        import torch

        env["torch_threads"] = torch.get_num_threads()
    except Exception:
        pass
    env["loadavg"] = list(os.getloadavg())
    return env


_XLA_1T = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"


def _measure(devices=("cpu", "gpu"), Ns=None):
    """``Ns`` given: the evaluation series only, at those N (``--Ns``)."""
    eval_only = Ns is not None
    Ns = _NS if Ns is None else Ns
    Ms = _MS if not eval_only else [max(1, round(n / 16)) for n in Ns]
    # single-threaded CPU for every library (see the module docstring); jax reads
    # XLA_FLAGS at backend initialisation, so set it before the first import
    if "jax" not in sys.modules:
        os.environ.setdefault("XLA_FLAGS", _XLA_1T)
    for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
        if os.environ.get(v) != "1":
            print(f"WARNING: {v} is not 1; numpy/BLAS may thrash on a shared box")
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import torch

    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    from galpy.actionAngle import actionAngleVerticalInverse
    from galpy.potential import IsothermalDiskPotential

    cpu = jax.devices("cpu")[0]
    try:
        jgpu = jax.devices("gpu")[0]
    except Exception:
        jgpu = None
    tgpu = torch.cuda.is_available()
    ev, cons, extra = {}, {}, {}

    def pot_jax(dev):
        return IsothermalDiskPotential(
            amp=jax.device_put(jnp.asarray(_AMP), dev),
            sigma=jax.device_put(jnp.asarray(_SIG), dev),
        )

    def pot_torch(dev):
        return IsothermalDiskPotential(
            amp=torch.tensor(_AMP, device=dev), sigma=torch.tensor(_SIG, device=dev)
        )

    def build(pot, nE):
        return actionAngleVerticalInverse(pot=pot, Es=_es(nE), setup_interp=True)

    angles = lambda N: numpy.linspace(0.0, 2.0 * numpy.pi, N, endpoint=False)  # noqa: E731
    a16 = numpy.linspace(0.0, 2.0 * numpy.pi, _NANG_VMAP, endpoint=False)
    Jv = lambda M: numpy.linspace(*_JRANGE_VMAP, M)  # noqa: E731

    def jax_series(dev, tag):
        with jax.default_device(dev):
            aA = build(pot_jax(dev), _NE_EVAL)
            jJ = jax.device_put(jnp.asarray(_J), dev)
            if tag == "CPU":
                ev["jax (CPU, eager)"] = _sweep(
                    Ns, lambda N: (lambda a=jax.device_put(jnp.asarray(angles(N)), dev):
                                    aA.xvFreqs(jJ, a)), "jax", "jax eager")
            comp = []

            def jit_call(N):
                a = jax.device_put(jnp.asarray(angles(N)), dev)
                f = jax.jit(lambda j, a: aA.xvFreqs(j, a))
                t0 = time.perf_counter()
                jax.block_until_ready(f(jJ, a))
                comp.append(time.perf_counter() - t0)  # compile + first run
                return lambda: f(jJ, a)

            ev[f"jax ({tag}, jit)"] = _sweep(Ns, jit_call, "jax", f"jax {tag} jit")
            extra[f"jax ({tag}, jit) first call"] = comp + [float("nan")] * (len(Ns) - len(comp))
            a16d = jax.device_put(jnp.asarray(a16), dev)
            fv = jax.jit(jax.vmap(lambda j: aA.xvFreqs(j, a16d)))

            def vmap_call(M):
                Js = jax.device_put(jnp.asarray(Jv(M)), dev)
                jax.block_until_ready(fv(Js))  # compile for this M (excluded)
                return lambda: fv(Js)

            ev[f"jax ({tag}, jit, vmap over tori)"] = _sweep(
                Ms, vmap_call, "jax", f"jax {tag} vmap M")
            if eval_only:
                return
            # construction
            warm, first = [], []
            stop = False
            for nE in _NES:
                if stop:
                    warm.append(float("nan")); first.append(float("nan"))
                    continue
                p = pot_jax(dev)
                t0 = time.perf_counter()
                jax.block_until_ready(build(p, nE)._mm_bk["js"])
                first.append(time.perf_counter() - t0)
                t, stop = _timed(lambda: build(p, nE)._mm_bk["tabs"], "jax")
                warm.append(t)
                print(f"  jax {tag} eager construct nE={nE}: first {first[-1]:.2e} warm {t:.2e}", flush=True)
            cons[f"jax ({tag}, eager)"] = {"warm": warm, "first": first}
            warm, first = [], []
            stop = False
            for nE in _NES:
                if stop:
                    warm.append(float("nan")); first.append(float("nan"))
                    continue

                def bj(amp, sig, nE=nE):
                    aa = build(IsothermalDiskPotential(amp=amp, sigma=sig), nE)
                    return aa._mm_bk["tabs"]

                f = jax.jit(bj)
                amp = jax.device_put(jnp.asarray(_AMP), dev)
                sig = jax.device_put(jnp.asarray(_SIG), dev)
                t0 = time.perf_counter()
                jax.block_until_ready(f(amp, sig))
                first.append(time.perf_counter() - t0)
                t, stop = _timed(lambda: f(amp, sig), "jax")
                warm.append(t)
                print(f"  jax {tag} jit construct nE={nE}: compile {first[-1]:.2e} warm {t:.2e}", flush=True)
            cons[f"jax ({tag}, jit)"] = {"warm": warm, "first": first}

    def torch_series(dev, tag):
        sync = "torch" if dev == "cuda" else "none"
        aA = build(pot_torch(dev), _NE_EVAL)
        jJ = torch.tensor(_J, device=dev)
        ev[f"torch ({tag})"] = _sweep(
            Ns, lambda N: (lambda a=torch.as_tensor(angles(N), device=dev):
                           aA.xvFreqs(jJ, a)), sync, f"torch {tag}")
        if eval_only:
            return
        warm, first = [], []
        stop = False
        for nE in _NES:
            if stop:
                warm.append(float("nan")); first.append(float("nan"))
                continue
            p = pot_torch(dev)
            t0 = time.perf_counter()
            build(p, nE)
            bench._sync_torch()
            first.append(time.perf_counter() - t0)
            t, stop = _timed(lambda: build(p, nE), sync)
            warm.append(t)
            print(f"  torch {tag} construct nE={nE}: first {first[-1]:.2e} warm {t:.2e}", flush=True)
        cons[f"torch ({tag})"] = {"warm": warm, "first": first}

    if "cpu" in devices:
        # numpy
        aN = build(IsothermalDiskPotential(amp=_AMP, sigma=_SIG), _NE_EVAL)
        ev["numpy (CPU)"] = _sweep(
            Ns, lambda N: (lambda a=angles(N): aN.xvFreqs(_J, a)), "none", "numpy")
        ev["numpy (CPU, loop over tori)"] = _sweep(
            Ms, lambda M: (lambda Js=Jv(M): [aN.xvFreqs(float(j), a16) for j in Js]),
            "none", "numpy loop M")
        if not eval_only:
            warm, stop = [], False
            for nE in _NES:
                if stop:
                    warm.append(float("nan"))
                    continue
                t, stop = _timed(lambda: build(IsothermalDiskPotential(amp=_AMP, sigma=_SIG), nE))
                warm.append(t)
                print(f"  numpy construct nE={nE}: {t:.2e}", flush=True)
            cons["numpy (CPU)"] = {"warm": warm}
        torch_series("cpu", "CPU")
        jax_series(cpu, "CPU")
    if "gpu" in devices:
        if tgpu:
            torch_series("cuda", "GPU")
        if jgpu is not None:
            jax_series(jgpu, "GPU")

    env = _env()
    try:
        env["gpu"] = torch.cuda.get_device_name(0) if tgpu else "none"
    except Exception:
        env["gpu"] = "?"
    return {"Ns": Ns, "Ms": Ms, "nEs": _NES, "nang_vmap": _NANG_VMAP, "J": _J,
            "nE_eval": _NE_EVAL, "eval": ev, "construct": cons, "extra": extra,
            "env": {("gpu_run" if devices == ("gpu",) else "cpu_run"): env}}


def _merge(old, new):
    out = dict(old)
    for k in ("eval", "construct", "extra"):
        out[k] = dict(old.get(k, {}))
        out[k].update(new.get(k, {}))
    out["env"] = dict(old.get("env", {}))
    out["env"].update(new.get("env", {}))
    return out


def _merge_Ns(old, new, Ns):
    """``--Ns``: ``new``'s evaluation series at ``Ns`` into ``old`` (grid extended,
    every other eval series NaN at the new N; ``old``'s env stamps kept)."""
    from paper.figures._harness import extra_points

    out = dict(old)
    grid = sorted(set(old["Ns"]) | set(Ns))
    m_of = dict(zip(old["Ns"], old["Ms"]))
    m_of.update(zip(new["Ns"], new["Ms"]))
    out["Ns"], out["Ms"] = grid, [m_of[N] for N in grid]
    replaced = {}
    for k in ("eval", "extra"):
        out[k] = {name: extra_points.regrid(old["Ns"], v, grid) for name, v in old[k].items()}
        for name, v in new[k].items():
            replaced.update({f"{name} N={N}": out[k][name][grid.index(N)]
                             for N in Ns if N in old["Ns"] and name in old[k]})
            ser = out[k].setdefault(name, [float("nan")] * len(grid))
            extra_points.put(ser, grid, Ns, v)
    rec = extra_points.note(Ns, next(iter(new["env"].values())))
    rec["series"] = sorted(new["eval"]) + sorted(new["extra"])
    rec["replaced"] = replaced
    out["extra_points"] = list(old.get("extra_points", [])) + [rec]
    return out


def _data():
    if os.path.exists(_JSON):
        with open(_JSON) as fh:
            return json.load(fh)
    data = _measure()
    os.makedirs(_DATA, exist_ok=True)
    with open(_JSON, "w") as fh:
        json.dump(data, fh, indent=1, sort_keys=True)
        fh.write("\n")
    return data


def _nan(x):
    return numpy.asarray([numpy.nan if v is None else v for v in x], dtype=float)


@figure(
    name="gV2_vertical_inverse_speed",
    track="E",
    seed=0,
    description="actionAngleVerticalInverse speed: xvFreqs evaluation wall time vs N angles (numpy, jax eager/jit/vmap, torch; CPU & GPU) and family construction time vs nE [measure->json, plot byte-stable]",
)
def make():
    data = _data()
    srcs = _sources(data)
    Ns = numpy.asarray(data["Ns"], dtype=float)
    # PAPER: printed width of a figure* (7.0 in); the panels are stacked so each
    # keeps a column of direct labels at its right
    figsize = layout.figsize(PAPER, (7.0, 6.4), (13.4, 5.0))
    if PAPER:
        fig, (axa, axb) = plt.subplots(2, 1, figsize=figsize,
                                       gridspec_kw={"height_ratios": [1.25, 1.0]})
    else:
        fig, (axa, axb) = plt.subplots(1, 2, figsize=figsize,
                                       gridspec_kw={"width_ratios": [1.2, 1.0]})
    fs_line = layout.FS["line"] if PAPER else 6.2
    fs_note = layout.FS["note"] if PAPER else 6.3
    for src, override in srcs:
        for name in _EVAL:
            if name not in src["eval"]:
                continue
            dev = palette.parse(name)[1]
            if override:
                if dev == "cpu":
                    continue  # only the GPU series are re-measured on the H100
                dev = override
            st, lab = _style(name, dev)
            t = _nan(src["eval"][name])
            x = (numpy.asarray(src["Ms"], dtype=float) * src["nang_vmap"]
                 if ("vmap" in name or "loop" in name) else numpy.asarray(src["Ns"], dtype=float))
            axa.loglog(x, t, ms=3.5, label=lab, lw=1.3, **st)
    axa.set_ylim(1.5e-4 if PAPER else 2e-4, 400.0)
    # constant cost-per-point references (1 us and 10 us per point)
    for us, lab in ((1e-6, r"1 $\mu$s/point"), (1e-5, r"10 $\mu$s/point")):
        axa.plot(Ns, us * Ns, ":", color="0.75", lw=0.9, zorder=0, label="_ref")
    if PAPER:
        axa.set_xlabel(r"points per call $N$ (vmap/loop: $N/%d$ tori $\times$ %d angles)"
                       % (data["nang_vmap"], data["nang_vmap"]))
        axa.set_title("(a) evaluation")
        axa.text(0.02, 0.97, r"dotted: 1, 10 $\mu$s per point", transform=axa.transAxes,
                 fontsize=fs_note, color="0.45", ha="left", va="top")
    else:
        axa.set_xlabel(r"points per call $N$ (angles on one torus; vmap/loop: $N/%d$ tori $\times$ %d angles)"
                       % (data["nang_vmap"], data["nang_vmap"]), fontsize=8)
        axa.set_title(rf"(a) evaluation: xvFreqs($J={data['J']:g}$, angles)")
    axa.set_ylabel("wall time per call (s)")
    axa.grid(True, which="both", alpha=0.25)

    nEs = numpy.asarray(data["nEs"], dtype=float)
    for src, override in srcs:
        for name in _CONS:
            if name not in src["construct"]:
                continue
            dev = palette.parse(name)[1]
            if override:
                if dev == "cpu":
                    continue
                dev = override
            st, lab = _style(name, dev)
            c = src["construct"][name]
            axb.loglog(nEs, _nan(c["warm"]), ms=4, lw=1.3, label=lab, **st)
            if "first" in c:
                fst = dict(st, ls=":")
                axb.loglog(nEs, _nan(c["first"]), ms=4, mfc="w", lw=0.8, label="_first", **fst)
    axb.text(0.02, 0.97, "open markers: first call" if PAPER else
             "thin dotted, open markers: first call at each $n_E$\n"
             "(eager jax: op compiles; jit: trace+compile)",
             transform=axb.transAxes, fontsize=fs_note, color="0.4", va="top", ha="left")
    axb.set_xlabel(r"number of energies $n_E$ in the family")
    axb.set_ylabel("construction wall time (s)" if PAPER
                   else "construction wall time (s) [warm]")
    axb.set_title("(b) construction" if PAPER
                  else r"(b) construction (setup_interp=True, mm_npt=40)")
    axb.set_ylim(top=axb.get_ylim()[1] * (3.0 if PAPER else 8.0))
    axb.grid(True, which="both", alpha=0.25)
    if not PAPER:
        env = data.get("env", {})
        cpu_env = env.get("cpu_run", {})
        gpu_env = env.get("gpu_run", cpu_env)
        stamp = (f"CPU: {cpu_env.get('cpu', '?')}, 1 thread\n"
                 f"{gpu_env.get('gpu', cpu_env.get('gpu', 'none'))}: jax {gpu_env.get('jax', '?')}"
                 f" · torch {gpu_env.get('torch', '?')}")
        if len(srcs) > 1:
            he = srcs[1][0].get("env", {}).get("gpu_run", {})
            stamp += f"\n{he.get('gpu', 'H100')}: jax {he.get('jax', '?')} · torch {he.get('torch', '?')}"
        axa.text(0.02, 0.98, stamp + "\nfloat64 · dotted grey: 1 and 10 $\\mu$s per point",
                 transform=axa.transAxes, fontsize=5.8, color="0.45", ha="left", va="top")
        fig.tight_layout(w_pad=12.0)
    else:
        fig.tight_layout()
    sep = 1.3 * fs_line if PAPER else None
    palette.label_lines(axa, fontsize=fs_line, min_sep_pt=sep)
    palette.label_lines(axb, fontsize=fs_line, min_sep_pt=sep)
    if PAPER:
        layout.fit_right(fig)
    return fig


if __name__ == "__main__":  # re-measure a subset of devices and merge into the JSON
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", default="gpu", help="comma list of cpu,gpu")
    ap.add_argument("--Ns", metavar="N1,N2,...",
                    help="with --devices gpu: measure ONLY the GPU evaluation series at these N "
                         "(ints or e-notation, e.g. 1e7) and MERGE them into the existing JSON "
                         "(N grid extended; env stamps kept; run recorded in 'extra_points')")
    args = ap.parse_args()
    devs = tuple(d.strip() for d in args.devices.split(",") if d.strip())
    if args.Ns is not None:
        from paper.figures._harness import extra_points

        if devs != ("gpu",):
            ap.error("--Ns measures the GPU evaluation series only: use it with --devices gpu")
        try:
            Ns = extra_points.parse_Ns(args.Ns)
        except ValueError as e:
            ap.error(str(e))
        old = extra_points.load(_JSON, required=True)
        import torch

        extra_points.check_same(old.get("env", {}).get("gpu_run", {}).get("gpu"),
                                torch.cuda.get_device_name(0) if torch.cuda.is_available()
                                else "none", "gpu", _JSON)
        new = _measure(devices=devs, Ns=Ns)
        if not new["eval"]:
            sys.exit("--Ns: no GPU visible, nothing measured; JSON left untouched")
        with open(_JSON, "w") as fh:
            json.dump(_merge_Ns(old, new, Ns), fh, indent=1, sort_keys=True)
            fh.write("\n")
        print("wrote", _JSON)
        sys.exit(0)
    new = _measure(devices=devs)
    old = {}
    if os.path.exists(_JSON):
        with open(_JSON) as fh:
            old = json.load(fh)
    os.makedirs(_DATA, exist_ok=True)
    with open(_JSON, "w") as fh:
        json.dump(_merge(old, new) if old else new, fh, indent=1, sort_keys=True)
        fh.write("\n")
    print("wrote", _JSON)
