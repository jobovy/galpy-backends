"""GS5 -- particle-spray stream generation speed: numpy/C vs jax vs torch, CPU & GPU.

gS2/gS3 show that a streamspraydf stream is DIFFERENTIABLE in the halo parameters;
this figure shows what generating such a stream COSTS on each backend, as a function
of the number of sprayed particles N.

The model: fardal15spraydf, a 2x10^4 Msun progenitor on galpy's standard inclined
stream orbit (the gS1/gS3 progenitor, z=0.887, vz=-0.477 in natural units) in a
flattened LogarithmicHaloPotential (q=0.9), disrupting for 4.5 Gyr, leading arm.
Timed: ``spdf.sample(n=N, return_orbit=False)`` -- draw N stripping times, spray N
particles off the progenitor and integrate every one of them to today. The
construction (one progenitor integration) is done once, outside the timer; its
cost per backend is stored in the JSON (``construct``).

What runs on each backend is DIFFERENT, and that is the point of the figure:

  * numpy (CPU, C): a numpy potential. galpy's default C integrator (symplec4_c,
    OpenMP over particles) on a 10001-point output grid per particle -- the
    historical path. It stores the whole 10001-point orbit per particle
    (~0.5 MB/particle), so the sample is drawn in chunks of 2000 particles to keep
    memory bounded (memory, not time, is what limits a single numpy call).
  * jax / torch: a DIFFERENTIABLE stream -- the halo flattening q is a jax/torch
    scalar and the progenitor IC a backend array, exactly the gS2/gS3 setup. The
    sprayed particles are then integrated IN-BACKEND, vectorised over particles on
    a per-particle [-dt_i, 0] grid: jax with diffrax (Dopri8 under jax.vmap, each
    particle with its own adaptive controller), torch with torchode (Dopri5,
    batched, per-particle controller; the default since galpy#1584). Default
    tolerances rtol=atol=1e-12 (galpy's in-backend default, matching C). The
    output is a backend array carrying d(stream)/d(q).
      - jax (eager): plain calls. diffrax's solve is equinox-jitted internally, so
        "eager" here already runs a compiled ODE loop (one compile per shape in the
        warm-up); the rest of the sampler is op-by-op.
      - jax (jit): the user's ``jax.jit(lambda key: spdf.sample(N, key=key,
        return_orbit=False))`` -- galpy adds no internal jit; it is made
        jit-compatible and the caller compiles. Compile time is EXCLUDED from the
        timing (stored separately as ``compile``); the warm call is timed.
      - torch (eager): torch.compile is not used (it does not support the sm_61
        TITAN Xp; on CPU the per-shape inductor compile is minutes). Eager torchode
        is dispatch-bound: the batch takes a common number of adaptive steps
        (~2800 Dopri5 steps = ~17,000 force evaluations at rtol=1e-12, the SAME for
        N=100 and N=1000 -- set by the hardest particle, not by N), each ~a hundred
        small kernels. CPU torch runs with ONE intra-op thread: for these small
        kernels 4 threads only add barrier syncs (N=1000: 17 s with 1 thread vs
        38 s with 4, and 165-200 s with 4 under heavier contention).
  * GPU series (TITAN Xp, and NVIDIA H100 80GB on Trillium: data/h100/, same
    script, jax 0.11.2 / torch 2.14.1+cu130): the same jax/torch code with the spdf built on the
    device (jax.default_device / a torch.device('cuda') context, which is how
    galpy's own --device cuda test mode places torch work); sync via
    block_until_ready / cuda.synchronize.
  * Not shown: a numpy-parameter spdf sampled under a merely FORCED backend
    (``with use('torch')`` + a key) integrates the particles with the C
    integrator + its state-transition matrix (C-STM: differentiable w.r.t. the
    progenitor IC only, not the potential): ~4.3 ms/particle at N=1000 (torch),
    i.e. ~2x the plain numpy/C cost.

What it shows (CPU numbers, measured 2026-10-05/06): (a) wall time per call vs N;
(b) cost per particle. numpy/C costs ~2.1 ms/particle at every N (0.30 s at
N=100, 21 s at 10^4), strictly linear. The jitted, differentiable jax spray has NO
fixed floor on CPU and costs ~1.0-1.5 ms/particle (0.098 s at N=100, 13 s at
10^4, 153 s at 10^5) -- i.e. it is as fast as or faster than the non-differentiable
C path (whose cost is set by its 10001-point output grid per particle; the
adaptive Dopri8 at 1e-12 needs fewer, larger steps). Eager jax adds a ~6 s per-call
floor (per-call overhead of the un-jitted sampler around the diffrax solve) that is amortised by N~10^4. Eager torch
has a ~12 s floor (the adaptive-step loop is dispatched from Python) and
~7 ms/particle at 10^4. The compile of the jitted sampler (excluded) is ~36-50 s
per shape up to N=10^4 (stored in the JSON).

GPU: the jitted jax sampler has a flat ~0.2-0.3 s floor on both GPUs; on the
TITAN Xp (float64 at 1/32 rate) it turns linear past N~10^4 (0.15 ms/particle at
10^6, 149 s), on the H100 only past ~10^5 (7 us/particle at 10^6: 7.2 s, ~20x
the TITAN). Eager floors: jax ~9-10 s (TITAN) / ~3.5-4 s (H100); torch ~30-45 s
(TITAN) / ~11-14 s (H100), torch reaching 22 us/particle at 10^6 on the H100.
The jit compile (excluded) is 24-184 s on the TITAN box and 5-15 s on the H100
node (a faster host CPU).

Colours/line styles: the shared palette (paper/figures/_harness/palette.py) --
colour = framework x device (numpy black; jax sky blue / blue / navy and torch
orange / vermillion / dark red for CPU / TITAN Xp / H100), jit solid, eager dashed.

CAVEAT: the CPU numbers were measured on a heavily shared machine (load average
~65-150 on 72 logical CPUs, other users' threads running on the same 4 pinned
cores; the load is stamped in ``env_cpu``), so absolute CPU times are inflated
and noisy at the tens-of-percent level; the orders of magnitude and the
scalings are robust.

Timing protocol: one warm-up call, median of 3 timed calls (bench.time_call), all
CPU series pinned to the same 4 cores (taskset); a series stops once one call
exceeds ~60 s (later N recorded as NaN). Large-N GPU calls that run out of the
12 GB device memory are recorded as NaN.

Reproducibility: measure -> data/gS5_streamspray_speed.json, plot byte-stable from
it. Re-measure CPU: delete the JSON (or ``python -m paper.figures.gS5_streamspray_speed.make
--measure cpu``); GPU-only re-measure merged into the JSON: ``... --measure gpu``.
"""

import contextlib
import json
import os
import platform
import sys

import numpy

from paper.figures._harness import bench, palette
from paper.figures._harness.figure import figure

_DATA = os.path.join(os.path.dirname(__file__), "data")
_JSON = os.path.join(_DATA, "gS5_streamspray_speed.json")

_IC = [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596]
_Q = 0.9
_MASS_MSUN = 2e4
_TD_GYR = 4.5
_NS = [100, 300, 1000, 3000, 10000, 30000, 100000, 300000, 1000000]
_NUMPY_CHUNK = 2000
_TSTOP = 60.0  # stop a series once one call exceeds this (s)
_SEED = 20261005
_TORCH_CPU_THREADS = 1

# series name in the JSON -> (framework, device, mode, label suffix); colours and
# line styles from the shared palette (paper/figures/_harness/palette.py)
_SERIES = {
    "numpy (CPU, C)": ("numpy", "cpu", None, "C integrator"),
    "jax (CPU, eager)": ("jax", "cpu", "eager", ""),
    "jax (CPU, jit)": ("jax", "cpu", "jit", ""),
    "torch (CPU, eager)": ("torch", "cpu", "eager", ""),
    "jax (GPU, eager)": ("jax", "gpu", "eager", ""),
    "jax (GPU, jit)": ("jax", "gpu", "jit", ""),
    "torch (GPU, eager)": ("torch", "gpu", "eager", ""),
}
_ORDER = list(_SERIES)
_JSON_H100 = os.path.join(_DATA, "h100", "gS5_streamspray_speed.json")
# journal version (default): no env stamp / long titles or notes; the caption
# carries the explanations. GALPY_JAX_PAPER=0 restores the annotated figure.
PAPER = os.environ.get("GALPY_JAX_PAPER", "1") == "1"


def _env(torch, jax):
    env = {"python": sys.version.split()[0], "machine": platform.machine()}
    for m in ("numpy", "jax", "torch", "galpy", "diffrax", "torchode"):
        try:
            env[m] = __import__(m).__version__
        except Exception:
            pass
    try:
        with open("/proc/cpuinfo") as fh:
            env["cpu"] = next(
                line.split(":", 1)[1].strip() for line in fh if line.startswith("model name")
            )
    except Exception:
        env["cpu"] = "?"
    try:
        env["cpu_cores_used"] = len(os.sched_getaffinity(0))
    except Exception:
        pass
    try:
        env["loadavg_1_5_15"] = list(os.getloadavg())  # shared box: contention context
        env["logical_cpus"] = os.cpu_count()
    except Exception:
        pass
    try:
        env["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    except Exception:
        env["gpu"] = "?"
    return env


def _measure(devices=("cpu",), which=("numpy", "jax", "torch")):
    """Time the CPU and/or GPU series; merges into an existing JSON."""
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import torch

    torch.set_default_dtype(torch.float64)
    from galpy.backend import random as grandom
    from galpy.df import fardal15spraydf
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential
    from galpy.util import conversion

    mass = _MASS_MSUN / conversion.mass_in_msol(220.0, 8.0)
    td = _TD_GYR / conversion.time_in_Gyr(220.0, 8.0)

    def build(bk):
        if bk == "numpy":
            pot = LogarithmicHaloPotential(normalize=1.0, q=_Q)
            prog = Orbit(_IC)
        elif bk == "jax":  # on jax's current default device
            pot = LogarithmicHaloPotential(normalize=1.0, q=jnp.asarray(_Q))
            prog = Orbit(jnp.asarray(_IC))
        else:  # torch's current default device
            pot = LogarithmicHaloPotential(normalize=1.0, q=torch.tensor(_Q))
            prog = Orbit(torch.tensor(_IC))
        return fardal15spraydf(mass, progenitor=prog, pot=pot, tdisrupt=td)

    data = json.load(open(_JSON)) if os.path.exists(_JSON) else {}
    data.setdefault("series", {})
    data.setdefault("construct", {})
    data.setdefault("compile", {})
    data.setdefault("integrator", {})
    data["Ns"] = _NS
    series = data["series"]

    def _sweep(name, call_for_N, sync, compile_for_N=None):
        """call_for_N(N) -> zero-arg callable. Stops once a call exceeds _TSTOP."""
        out, comp, stopped = [], [], False
        for N in _NS:
            if stopped:
                out.append(float("nan"))
                comp.append(float("nan"))
                continue
            try:
                if compile_for_N is not None:
                    fn, tc = compile_for_N(N)
                    comp.append(tc)
                else:
                    fn = call_for_N(N)
                r = bench.time_call(fn, warmup=1, repeats=3, sync=sync)
                t = r["median"]
            except Exception as e:  # OOM on the 12 GB GPU, etc.
                print(f"  {name} N={N}: skip ({type(e).__name__}: {str(e)[:90]})")
                t = float("nan")
                stopped = True
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            print(f"  {name} N={N}: {t:.4g} s", flush=True)
            out.append(t)
            if t > _TSTOP:
                stopped = True
        series[name] = out
        if compile_for_N is not None:
            data["compile"][name] = comp

    import time

    def _timed_build(bk, sync):
        t0 = time.perf_counter()
        s = build(bk)
        if sync == "torch":
            bench._sync_torch()
        t = time.perf_counter() - t0
        return s, t

    def _jax_series(dev, tag):
        with jax.default_device(dev):
            spdf, tb = _timed_build("jax", "jax")
            data["construct"][f"jax ({tag})"] = tb
            data["integrator"][f"jax ({tag})"] = spdf._bsamp[2]
            key = jax.device_put(grandom.key(_SEED, backend="jax"), dev)

            def eager(N):
                return lambda: spdf.sample(N, return_orbit=False, key=key)

            _sweep(f"jax ({tag}, eager)", eager, "jax")

            def compiled(N):
                f = jax.jit(lambda k: spdf.sample(N, return_orbit=False, key=k))
                t0 = time.perf_counter()
                jax.block_until_ready(f(key))
                tc = time.perf_counter() - t0
                return (lambda: f(key)), tc

            _sweep(f"jax ({tag}, jit)", None, "jax", compile_for_N=compiled)

    def _torch_series(device, tag):
        # CPU is torch's default device: no torch.device() context there -- that
        # context is a TorchFunctionMode that adds Python overhead to EVERY op
        # (measured ~2x on this dispatch-bound path). On the GPU it is how the spdf
        # is built on cuda (same mechanism as galpy's --device cuda test mode).
        ctx = contextlib.nullcontext() if device == "cpu" else torch.device(device)
        nthr = torch.get_num_threads()
        if device == "cpu":
            # one intra-op thread: these are small (N-element) kernels, so 4 threads
            # only add barrier syncs -- at N=1000 measured 17 s (1 thread) vs 38 s
            # (4 threads), and 165-200 s (4 threads) under heavier contention on
            # this shared box.
            torch.set_num_threads(_TORCH_CPU_THREADS)
            data["torch_cpu_threads"] = _TORCH_CPU_THREADS
        try:
            _torch_body(device, tag, ctx)
        finally:
            torch.set_num_threads(nthr)

    def _torch_body(device, tag, ctx):
        with ctx:
            spdf, tb = _timed_build("torch", "torch" if device == "cuda" else "none")
            data["construct"][f"torch ({tag})"] = tb
            data["integrator"][f"torch ({tag})"] = spdf._bsamp[2]
            key = grandom.key(_SEED, backend="torch")

            def eager(N):
                return lambda: spdf.sample(N, return_orbit=False, key=key)

            _sweep(f"torch ({tag}, eager)", eager,
                   "torch" if device == "cuda" else "none")

    if "cpu" in devices and "numpy" in which:
        print("numpy (CPU, C)", flush=True)
        spdf_np, tb = _timed_build("numpy", "none")
        data["construct"]["numpy (CPU)"] = tb
        data["integrator"]["numpy (CPU)"] = "symplec4_c (10001-pt grid / particle)"

        def numpy_call(N):
            def f():
                numpy.random.seed(_SEED)
                outs = [
                    spdf_np.sample(min(_NUMPY_CHUNK, N - i), return_orbit=False)
                    for i in range(0, N, _NUMPY_CHUNK)
                ]
                return numpy.hstack(outs)

            return f

        _sweep("numpy (CPU, C)", numpy_call, "none")
    if "cpu" in devices and "jax" in which:
        print("jax (CPU)", flush=True)
        _jax_series(jax.devices("cpu")[0], "CPU")
    if "cpu" in devices and "torch" in which:
        print("torch (CPU)", flush=True)
        _torch_series("cpu", "CPU")
    if "cpu" in devices:
        data["env_cpu"] = _env(torch, jax)
    if "gpu" in devices:
        gpus = [d for d in jax.devices() if d.platform == "gpu"]
        if gpus and "jax" in which:
            print("jax (GPU)", flush=True)
            _jax_series(gpus[0], "GPU")
        if torch.cuda.is_available() and "torch" in which:
            print("torch (GPU)", flush=True)
            _torch_series("cuda", "GPU")
        data["env_gpu"] = _env(torch, jax)

    os.makedirs(_DATA, exist_ok=True)
    with open(_JSON, "w") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return data


def _data():
    if os.path.exists(_JSON):
        with open(_JSON) as fh:
            return json.load(fh)
    return _measure(devices=("cpu", "gpu"))


@figure(
    name="gS5_streamspray_speed",
    track="S",
    seed=0,
    description="particle-spray stream generation (fardal15spraydf.sample) wall time and per-particle cost vs N: numpy/C vs differentiable jax (eager/jit) and torch, CPU & GPU [measure->json, plot byte-stable]",
)
def make():
    import matplotlib.pyplot as plt

    data = _data()
    h100 = None
    if os.path.exists(_JSON_H100):
        with open(_JSON_H100) as fh:
            h100 = json.load(fh)
    fig, (axw, axp) = plt.subplots(1, 2, figsize=(10.4, 4.3))
    for name in _ORDER:
        fw, dev, mode, extra = _SERIES[name]
        runs = [(data, dev)]
        if dev == "gpu" and h100 is not None:
            runs.append((h100, "h100"))
        for d, dv in runs:
            if name not in d["series"]:
                continue
            Ns = numpy.asarray(d["Ns"], dtype=float)
            t = numpy.asarray(d["series"][name], dtype=float)
            if not numpy.any(numpy.isfinite(t)):
                continue
            st = palette.style(fw, dv, mode)
            lab = palette.label(fw, dv, mode, extra)
            axw.loglog(Ns, t, ms=4, label=lab, zorder=3, **st)
            axp.loglog(Ns, 1e3 * t / Ns, ms=4, label=lab, zorder=3, **st)
    axw.set_xlabel(r"number of sprayed particles $N$")
    axw.set_ylabel(r"wall time for spdf.sample($N$)  (s)")
    axw.set_title("(a) wall time per call")
    if not PAPER:
        axw.set_ylim(top=2e3)  # headroom for the stamp
    axp.set_xlabel(r"number of sprayed particles $N$")
    axp.set_ylabel(r"cost per particle  (ms)")
    axp.set_title("(b) per-particle cost")
    ec, eg = data.get("env_cpu", {}), data.get("env_gpu", {})
    eh = (h100 or {}).get("env_gpu", {})
    txt = (
        f"CPU: {ec.get('cpu', '?').replace('Intel(R) Xeon(R) CPU ', 'Xeon ')}"
        f" ({ec.get('cpu_cores_used', '?')} cores): jax {ec.get('jax', '?')} · torch {ec.get('torch', '?')}\n"
        f"{eg.get('gpu', 'GPU not measured')}: jax {eg.get('jax', '?')} · torch {eg.get('torch', '?')}"
        + (f"\n{eh.get('gpu', 'H100')}: jax {eh.get('jax', '?')} · torch {eh.get('torch', '?')}" if eh else "")
    )
    if PAPER:
        fig.tight_layout()
    else:
        axw.text(0.03, 0.97, txt, transform=axw.transAxes, fontsize=6.0, color="0.45",
                 va="top", ha="left")
        fig.suptitle(
            r"Particle-spray stream: fardal15spraydf.sample($N$), 4.5 Gyr of stripping "
            r"(numpy/C vs differentiable jax/torch)", y=0.99)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
    # direct labels on (b) (same lines as (a)); (a) carries the stamp
    # the CPU lines end early (series stop at ~60 s/call): label them inline
    inline = {
        "torch CPU eager": (1e4, 8), "numpy (C integrator)": (1e3, 9),
        "jax CPU eager": (1.5e5, 9), "jax CPU jit": (3e3, -9),
    }
    palette.label_lines(axp, inline=inline)
    for t in axp.texts:
        if t.get_text() in inline:
            t.set_bbox(dict(fc="white", ec="none", alpha=0.85, pad=0.4))
    return fig


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--measure", choices=["cpu", "gpu", "all"], required=True)
    p.add_argument("--which", default="numpy,jax,torch")
    a = p.parse_args()
    _measure(devices=("cpu", "gpu") if a.measure == "all" else (a.measure,),
             which=tuple(a.which.split(",")))
