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
    numpy. (``torch.func.vmap`` of the same map fails: the bisection's
    ``xp.sign`` is array_api_compat.torch's sign, which indexes with a boolean
    mask -- a dynamic-shape op torch's vmap cannot batch.)

    What dominates the backend evaluation is the anomaly root find: the backend
    path inverts the angle relation with the generic bracketed bisection
    (galpy.backend.optimize.brentq) -- 43 halvings eagerly, 100 under jit (the
    traced bracket hides its width, so the halving count falls back to maxiter)
    -- each a full pass of the (N, 40)-harmonic kernel, whereas numpy uses a
    safeguarded Newton iteration that converges in a handful of passes. Hence
    numpy beats jax-jit on the CPU by ~10x per angle; see
    BUG_vi_backend_eval_bisection.md (an in-process A/B with the eager count
    under jit is 2.35x faster at identical values). The per-angle cost of the
    jitted path is not constant in N on the CPU because XLA only multi-threads
    the larger fusions.

(b) Construction: wall time to build the family vs the number of energies nE,
    numpy vs native jax (eager and jitted) and torch, CPU and GPU. Open markers
    are the FIRST call at each nE: for eager jax this includes compiling every
    op for the new shapes, for jit the full trace+compile (both one-time per
    shape); filled markers are warm calls.

CPU threading: every CPU series is SINGLE-THREADED (OPENBLAS_NUM_THREADS=
OMP_NUM_THREADS=1, torch.set_num_threads(1), XLA eigen/intra-op threads = 1).
The measuring box is shared and was heavily oversubscribed (load average
~160 on 72 hardware threads); there, multi-threaded BLAS and torch thrash
(measured: numpy construction nE=31 0.6 s -> 19 s with 4 OpenBLAS threads, torch
N=1e3 evaluation 0.08 s -> 3.3 s with 4 torch threads), so single-thread numbers
are the only stable, comparable per-core measure. GPU series are unaffected.

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
(merges, then re-run ``run`` to re-plot).
"""

import json
import math
import os
import platform
import sys
import time

import matplotlib.pyplot as plt
import numpy

from paper.figures._harness import bench
from paper.figures._harness.figure import figure

_DATA = os.path.join(os.path.dirname(__file__), "data")
_JSON = os.path.join(_DATA, "gV2_vertical_inverse_speed.json")

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

_STYLE = {  # name: (colour, linestyle, marker)
    "numpy (CPU)": ("#000000", "-", "s"),
    "numpy (CPU, loop over tori)": ("#000000", "-.", "x"),
    "jax (CPU, eager)": ("#56B4E9", "-", "^"),
    "jax (CPU, jit)": ("#56B4E9", "--", "P"),
    "jax (CPU, jit, vmap over tori)": ("#56B4E9", "-.", "D"),
    "torch (CPU)": ("#E69F00", "-", "v"),
    "jax (GPU, jit)": ("#0072B2", "--", "o"),
    "jax (GPU, jit, vmap over tori)": ("#0072B2", "-.", "D"),
    "torch (GPU)": ("#D55E00", "-", "o"),
}
_CSTYLE = {  # construction series
    "numpy (CPU)": ("#000000", "-", "s"),
    "jax (CPU, eager)": ("#56B4E9", "-", "^"),
    "jax (CPU, jit)": ("#56B4E9", "--", "P"),
    "torch (CPU)": ("#E69F00", "-", "v"),
    "jax (GPU, eager)": ("#0072B2", "-", "^"),
    "jax (GPU, jit)": ("#0072B2", "--", "P"),
    "torch (GPU)": ("#D55E00", "-", "o"),
}


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


def _measure(devices=("cpu", "gpu")):
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
                    _NS, lambda N: (lambda a=jax.device_put(jnp.asarray(angles(N)), dev):
                                    aA.xvFreqs(jJ, a)), "jax", "jax eager")
            comp = []

            def jit_call(N):
                a = jax.device_put(jnp.asarray(angles(N)), dev)
                f = jax.jit(lambda j, a: aA.xvFreqs(j, a))
                t0 = time.perf_counter()
                jax.block_until_ready(f(jJ, a))
                comp.append(time.perf_counter() - t0)  # compile + first run
                return lambda: f(jJ, a)

            ev[f"jax ({tag}, jit)"] = _sweep(_NS, jit_call, "jax", f"jax {tag} jit")
            extra[f"jax ({tag}, jit) first call"] = comp + [float("nan")] * (len(_NS) - len(comp))
            a16d = jax.device_put(jnp.asarray(a16), dev)
            fv = jax.jit(jax.vmap(lambda j: aA.xvFreqs(j, a16d)))

            def vmap_call(M):
                Js = jax.device_put(jnp.asarray(Jv(M)), dev)
                jax.block_until_ready(fv(Js))  # compile for this M (excluded)
                return lambda: fv(Js)

            ev[f"jax ({tag}, jit, vmap over tori)"] = _sweep(
                _MS, vmap_call, "jax", f"jax {tag} vmap M")
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
            _NS, lambda N: (lambda a=torch.as_tensor(angles(N), device=dev):
                            aA.xvFreqs(jJ, a)), sync, f"torch {tag}")
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
            _NS, lambda N: (lambda a=angles(N): aN.xvFreqs(_J, a)), "none", "numpy")
        ev["numpy (CPU, loop over tori)"] = _sweep(
            _MS, lambda M: (lambda Js=Jv(M): [aN.xvFreqs(float(j), a16) for j in Js]),
            "none", "numpy loop M")
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
    return {"Ns": _NS, "Ms": _MS, "nEs": _NES, "nang_vmap": _NANG_VMAP, "J": _J,
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
    Ns = numpy.asarray(data["Ns"], dtype=float)
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11.2, 4.6),
                                   gridspec_kw={"width_ratios": [1.25, 1.0]})
    for name, (col, ls, mk) in _STYLE.items():
        if name not in data["eval"]:
            continue
        t = _nan(data["eval"][name])
        x = (numpy.asarray(data["Ms"], dtype=float) * data["nang_vmap"]
             if ("vmap" in name or "loop" in name) else Ns)
        axa.loglog(x, t, ls=ls, marker=mk, ms=4, color=col, label=name, lw=1.4)
    # constant cost-per-point references (1 us and 10 us per point)
    for us, lab in ((1e-6, r"1 $\mu$s/point"), (1e-5, r"10 $\mu$s/point")):
        axa.plot(Ns, us * Ns, ":", color="0.75", lw=0.9, zorder=0)
        axa.text(Ns[-1], us * Ns[-1], lab, fontsize=6, color="0.55", ha="right", va="bottom")
    axa.set_xlabel(r"points per call $N$ (angles on one torus; vmap/loop: $N/%d$ tori $\times$ %d angles)"
                   % (data["nang_vmap"], data["nang_vmap"]), fontsize=8)
    axa.set_ylabel("wall time per call (s)")
    axa.set_title(rf"(a) evaluation: xvFreqs($J={data['J']:g}$, angles)")
    axa.legend(fontsize=6.5, loc="upper left", framealpha=0.92)
    axa.grid(True, which="both", alpha=0.25)

    nEs = numpy.asarray(data["nEs"], dtype=float)
    for name, (col, ls, mk) in _CSTYLE.items():
        if name not in data["construct"]:
            continue
        c = data["construct"][name]
        axb.loglog(nEs, _nan(c["warm"]), ls=ls, marker=mk, ms=4.5, color=col, lw=1.4,
                   label=name)
        if "first" in c:
            axb.loglog(nEs, _nan(c["first"]), ls=":", marker=mk, ms=4.5, color=col,
                       mfc="w", lw=0.9)
    axb.plot([], [], ":", marker="o", mfc="w", color="0.4",
             label="first call (eager: op compiles; jit: trace+compile)")
    axb.set_xlabel(r"number of energies $n_E$ in the family")
    axb.set_ylabel("construction wall time (s)")
    axb.set_title(r"(b) construction (setup_interp=True, mm_npt=40)")
    axb.legend(fontsize=6.5, loc="upper left", framealpha=0.92)
    axb.grid(True, which="both", alpha=0.25)
    env = data.get("env", {})
    cpu_env = env.get("cpu_run", {})
    gpu_env = env.get("gpu_run", cpu_env)
    axb.text(0.98, 0.02,
             f"CPU: {cpu_env.get('cpu', '?')}, 1 thread\n"
             f"GPU: {gpu_env.get('gpu', cpu_env.get('gpu', 'none'))}\n"
             f"jax {cpu_env.get('jax', '?')} · torch {cpu_env.get('torch', '?')} · float64",
             transform=axb.transAxes, fontsize=5.8, color="0.45", ha="right", va="bottom")
    fig.tight_layout()
    return fig


if __name__ == "__main__":  # re-measure a subset of devices and merge into the JSON
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", default="gpu", help="comma list of cpu,gpu")
    args = ap.parse_args()
    devs = tuple(d.strip() for d in args.devices.split(",") if d.strip())
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
