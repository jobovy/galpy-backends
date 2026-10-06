"""GS4 -- streamdf (action-angle stream model) speed: numpy/C vs jax vs torch, CPU & GPU.

gS1 shows that the streamdf stream TRACK is differentiable under jax. This figure
shows what the streamdf model COSTS per backend, split into the three operations a
user actually performs:

(a) CONSTRUCTION incl. the stream track, vs the number of track chunks
    (nTrackChunks = 4 ... 32; galpy's default for this stream is 10). This is the
    expensive, one-off part: progenitor actions/frequencies + the AD (jax) or
    finite-difference (numpy) dOmega/dJ, then per chunk an auxiliary-orbit point,
    its actions/angles (actionAngleIsochroneApprox: an orbit integration per point)
    and its 6x6 action-angle Jacobian, then the track interpolation and the
    spread (covariance) setup.
      * numpy (CPU): numpy progenitor + potential; C orbit integration
        (actionAngleIsochroneApprox's default C integrator), serial chunk loop,
        finite-difference Jacobians. Median of 3 after one warm-up.
      * jax (CPU/GPU): the backend-native, differentiable construction -- jax
        progenitor IC, LogarithmicHalo q as a jax scalar, diffrax-integrated
        actionAngleIsochroneApprox; the chunk loop is jax.vmap'ed (galpy#1602) and
        the Jacobians are exact AD. Each nTrackChunks is measured in a FRESH
        process, twice: "cold" (first construction: eager jax compiles every
        op of a new shape -- profiling shows ~60% of the time in XLA compilation,
        ~1400 small compiles) and "warm" (an identical second construction in the
        same process, all op compilations cached = the cost inside a fitting loop).
        Single shots (each is minutes).
      * NOT measured, because they fail (see the BUG_*.md notes in this directory):
          - torch: a torch-progenitor streamdf cannot be constructed eagerly
            (BUG_torch_streamdf_construct.md: numpy.sqrt on a grad-carrying tensor
            in _offset_setup).
          - jax.jit (or jax.grad) of the construction w.r.t. a potential parameter:
            TracerArrayConversionError in meanOmega
            (BUG_jax_streamdf_construct_traced.md).

(b) SAMPLING N stream stars, ``sdf.sample(n=N, key=key)`` -> (R,vR,vT,z,vz,phi): the
    backend-native inverse-CDF frequency/angle/time draw + the linearised
    track inverse (_approxaAInv). No orbit integration -- cheap per point.
(c) EVALUATING the log DF at N phase-space points, ``sdf(R,vR,vT,z,vz,phi)`` (the
    likelihood of N stars): the linearised track forward map (_approxaA, the
    useInterp=True default) + the closed-form DF. The N points are a 10^4-star
    sample of this stream, tiled to N (the per-point cost does not depend on
    the values).

    Series in (b) and (c), all from a stream built ONCE outside the timer
    (nTrackChunks=10, nTrackIterations=0 = what numpy picks for this stream):
      * numpy (CPU): numpy-built streamdf, numpy inputs.
      * jax (CPU/GPU, eager / jit): the jax-native streamdf of (a). jit = the
        user's ``jax.jit(lambda k: sdf.sample(n=N, key=k))`` /
        ``jax.jit(lambda *xv: sdf(*xv))``; compile excluded (stored in the JSON).
      * torch (CPU/GPU, eager): since a torch-native streamdf cannot be built
        (bug above), the numpy-built streamdf is used under ``use("torch")`` with
        a torch key / torch inputs (CPU: one intra-op thread, see gS5): its frozen tables are coerced to torch
        constants and the sampling/evaluation runs on torch (outputs are torch
        tensors; differentiable w.r.t. the inputs and the key's draws, not w.r.t.
        the potential).
    Eager calls are timed WARM (one warm-up call first): eager jax compiles each op
    once per shape, so the very first call at a new N is ~6-10 s regardless of N.

What it shows (CPU numbers, measured 2026-10-06):
  (a) numpy/C construction grows linearly with the chunk count (4.4, 7.0, 13, 27 s
      for 4, 8, 16, 32 chunks: a serial per-chunk loop). The jax construction is
      FLAT in nTrackChunks (the chunk loop is vmapped) but expensive: ~200-230 s
      cold and ~90-113 s warm, i.e. eager jax is compile/dispatch-bound, and the
      construction cannot (yet) be jitted to remove that.
  (b) sampling: jax.jit is ~70-150x faster than numpy (N=10^5: 0.51 s vs 77 s;
      N=100: 1.2 ms vs 77 ms); eager jax (0.03 s floor, 7.3 s at 10^5) and eager
      torch on the numpy-built track (23 s at 10^5) also beat numpy, whose
      linearised track inverse (_approxaAInv) is a Python loop over the N points,
      while the backend twin is vectorised.
  (c) DF evaluation: same picture -- jit 0.13 s vs numpy 17 s at N=10^5 (~130x);
      eager jax 3.6 s, eager torch 11 s; numpy's forward map (_approxaA) is also a
      per-point Python loop.
  So construction is where the backends pay, but once built, sampling and
  likelihood evaluation under jax.jit are two orders of magnitude faster than the
  numpy path. Checks: the jax-native track matches the numpy track to 3.9e-7
  (AD vs finite-difference Jacobians); the torch-forced DF evaluation matches numpy
  to 2e-13 and its d(logDF)/dR matches a central FD to 1.8e-7 (no numpy island:
  grad_fn present, no fallback warnings).
  CPU caps: the eager jax 10^6-point sample was OOM-killed on the shared box (the
  closest-track-point search holds (N, 1001) temporaries), so the jax CPU series
  stop at 10^5.

CAVEAT: the CPU numbers were measured on a heavily shared machine (load average
~65-150 on 72 logical CPUs, other users' threads on the same 4 pinned cores; the
load is stamped in ``env_cpu``), so absolute CPU times are inflated and noisy at
the tens-of-percent level; the orders of magnitude and the scalings are robust.

Timing protocol: median of 3 timed calls after one warm-up (bench.time_call),
CPU series pinned to the same 4 cores (taskset); a (b)/(c) series stops once one
call exceeds ~60 s (NaN beyond); GPU OOM -> NaN.

Reproducibility: measure -> data/gS4_streamdf_speed.json, plot byte-stable from it.
Re-measure CPU: ``python -m paper.figures.gS4_streamdf_speed.make --measure cpu``;
GPU-only re-measure merged into the JSON: ``... --measure gpu`` (needs the GPU
visible; jax runs on its default device = the GPU).
"""

import json
import os
import platform
import subprocess
import sys

import numpy

from paper.figures._harness import bench
from paper.figures._harness.figure import figure

_DATA = os.path.join(os.path.dirname(__file__), "data")
_JSON = os.path.join(_DATA, "gS4_streamdf_speed.json")

_IC = [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596]
_Q = 0.9
_SIGV = 0.365 / 220.0
_TD_GYR = 4.5
_TINTJ = 100
_NCH_DEFAULT = 10
_NCHS = [4, 8, 16, 32]
_NS = [100, 1000, 10000, 100000, 1000000]
_NBASE = 10000
_TSTOP = 60.0
_NMAX_JAX_CPU = 100000
_SEED = 20261005

_STYLE = {  # colour, linestyle, marker
    "numpy (CPU)": ("#000000", "-", "s"),
    "jax (CPU, eager cold)": ("#56B4E9", ":", "^"),
    "jax (CPU, eager)": ("#56B4E9", "--", "^"),
    "jax (CPU, jit cold)": ("#56B4E9", "-.", "P"),
    "jax (CPU, jit)": ("#56B4E9", "-", "P"),
    "torch (CPU, eager)": ("#E69F00", "--", "v"),
    "jax (GPU, eager cold)": ("#0072B2", ":", "o"),
    "jax (GPU, eager)": ("#0072B2", "--", "o"),
    "jax (GPU, jit)": ("#0072B2", "-", "D"),
    "torch (GPU, eager)": ("#D55E00", "--", "o"),
}
_ORDER = list(_STYLE)
_LABEL = {  # the torch series run the numpy-built streamdf under use("torch")
    "torch (CPU, eager)": "torch (CPU, eager; numpy-built sdf)",
    "torch (GPU, eager)": "torch (GPU, eager; numpy-built sdf)",
}

# --- construction: one fresh process per (backend, nTrackChunks) -------------------
_BUILD_SCRIPT = r"""
import json, sys, time, warnings
warnings.simplefilter("ignore")
import numpy
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from galpy.actionAngle import actionAngleIsochroneApprox
from galpy.df import streamdf
from galpy.orbit import Orbit
from galpy.potential import LogarithmicHaloPotential
from galpy.util import conversion
bk, nch, ic, q, sigv, tdg, tintJ = sys.argv[1], int(sys.argv[2]), json.loads(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]), int(sys.argv[7])
td = tdg / conversion.time_in_Gyr(220.0, 8.0)
def build(qq=None):
    if bk == "numpy":
        lp = LogarithmicHaloPotential(normalize=1.0, q=q)
        aA = actionAngleIsochroneApprox(pot=lp, b=0.8, tintJ=tintJ)
        prog = Orbit(ic)
    elif bk == "torch":
        import torch
        torch.set_default_dtype(torch.float64)
        lp = LogarithmicHaloPotential(normalize=1.0, q=torch.tensor(q))
        aA = actionAngleIsochroneApprox(pot=lp, b=0.8, tintJ=tintJ, integrate_method="torchode")
        prog = Orbit(torch.tensor(ic))
    else:
        lp = LogarithmicHaloPotential(normalize=1.0, q=jnp.asarray(q) if qq is None else qq)
        aA = actionAngleIsochroneApprox(pot=lp, b=0.8, tintJ=tintJ, integrate_method="diffrax")
        prog = Orbit(jnp.asarray(ic))
    s = streamdf(sigv, progenitor=prog, pot=lp, aA=aA, leading=True, nTrackChunks=nch,
                 nTrackIterations=0, tdisrupt=td)
    jax.block_until_ready(s._interpolatedObsTrackXY) if bk == "jax" else None
    return s
ts = []
if bk == "jaxjit":  # the whole construction under jax.jit: cold = trace+compile+run
    f = jax.jit(lambda qq: build(qq)._ObsTrack)
    for i in range(2):
        t0 = time.perf_counter(); jax.block_until_ready(f(jnp.asarray(q))); ts.append(time.perf_counter() - t0)
    print("RESULT", json.dumps({"times": ts, "track_type": "jax", "device": "jit"}))
    sys.exit(0)
for i in range({"numpy": 4, "torch": 1}.get(bk, 2)):  # torch: ~20 min a build
    t0 = time.perf_counter(); s = build(); ts.append(time.perf_counter() - t0)
print("RESULT", json.dumps({"times": ts, "track_type": type(s._interpolatedObsTrackXY).__module__,
      "device": str(getattr(s._ObsTrack, "devices", lambda: ["cpu"])())}))
"""


def _build_in_subprocess(bk, nch, gpu=False):
    env = dict(os.environ)
    if not gpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["JAX_PLATFORMS"] = "cpu"
    args = [sys.executable, "-c", _BUILD_SCRIPT, bk, str(nch), json.dumps(_IC), str(_Q),
            repr(_SIGV), str(_TD_GYR), str(_TINTJ)]
    out = subprocess.run(args, env=env, capture_output=True, text=True, timeout=3600)
    for line in out.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    print(out.stderr[-2000:])
    return None


def _env(torch, jax):
    env = {"python": sys.version.split()[0], "machine": platform.machine()}
    for m in ("numpy", "jax", "torch", "galpy", "diffrax"):
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


def _measure(devices=("cpu",), parts=("construct", "sample", "eval"),
             which=("numpy", "jax", "torch")):
    import time

    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import torch

    torch.set_default_dtype(torch.float64)
    from galpy.actionAngle import actionAngleIsochroneApprox
    from galpy.backend import random as grandom
    from galpy.backend import use
    from galpy.df import streamdf
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential
    from galpy.util import conversion

    td = _TD_GYR / conversion.time_in_Gyr(220.0, 8.0)
    data = json.load(open(_JSON)) if os.path.exists(_JSON) else {}
    for k in ("construct", "sample", "eval", "compile", "notes"):
        data.setdefault(k, {})
    data.update(Ns=_NS, nchs=_NCHS, nch_default=_NCH_DEFAULT, tintJ=_TINTJ)

    def save():
        os.makedirs(_DATA, exist_ok=True)
        with open(_JSON, "w") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
            fh.write("\n")

    # ---------------- (a) construction ----------------
    if "construct" in parts:
        runs = []
        if "cpu" in devices:
            # --which jax measures eager jax; jaxjit (the whole construction under
            # jax.jit) is opt-in by name
            runs += [(b, "CPU", False) for b in ("numpy", "jax", "jaxjit", "torch") if b in which]
        if "gpu" in devices and any(d.platform == "gpu" for d in jax.devices()):
            runs += [("jax", "GPU", True)]
        for bk, tag, gpu in runs:
            cold, warm = [], []
            for nch in _NCHS:
                r = _build_in_subprocess(bk, nch, gpu=gpu)
                if r is None:
                    cold.append(float("nan"))
                    warm.append(float("nan"))
                    continue
                ts = r["times"]
                if bk == "numpy":  # one warm-up, median of 3
                    warm.append(float(numpy.median(ts[1:])))
                    cold.append(ts[0])
                elif bk == "torch":  # no compile to separate: one build
                    cold.append(ts[0])
                    warm.append(ts[0])
                else:
                    cold.append(ts[0])
                    warm.append(ts[1])
                print(f"  construct {bk} ({tag}) nch={nch}: {ts} [{r['device']}]", flush=True)
            if bk == "numpy":
                data["construct"]["numpy (CPU)"] = warm
            elif bk == "torch":
                data["construct"][f"torch ({tag}, eager)"] = warm
            elif bk == "jaxjit":
                data["construct"][f"jax ({tag}, jit cold)"] = cold
                data["construct"][f"jax ({tag}, jit)"] = warm
            else:
                data["construct"][f"jax ({tag}, eager cold)"] = cold
                data["construct"][f"jax ({tag}, eager)"] = warm
            save()
        for k, key in (("construct_torch", "torch (CPU, eager)"), ("construct_jit", "jax (CPU, jit)")):
            if key in data["construct"]:
                data["notes"].pop(k, None)

    if not ({"sample", "eval"} & set(parts)):
        save()
        return data

    # ---------------- (b) sampling, (c) DF evaluation ----------------
    def build(bk):
        if bk == "numpy":
            lp = LogarithmicHaloPotential(normalize=1.0, q=_Q)
            aA = actionAngleIsochroneApprox(pot=lp, b=0.8, tintJ=_TINTJ)
            prog = Orbit(_IC)
        else:
            lp = LogarithmicHaloPotential(normalize=1.0, q=jnp.asarray(_Q))
            aA = actionAngleIsochroneApprox(pot=lp, b=0.8, tintJ=_TINTJ,
                                            integrate_method="diffrax")
            prog = Orbit(jnp.asarray(_IC))
        return streamdf(_SIGV, progenitor=prog, pot=lp, aA=aA, leading=True,
                        nTrackChunks=_NCH_DEFAULT, nTrackIterations=0, tdisrupt=td)

    sdf_np = build("numpy")
    numpy.random.seed(_SEED)
    base = numpy.asarray(sdf_np.sample(n=_NBASE), dtype=float)  # (6, NBASE)

    def pts(N):
        reps = N // _NBASE + 1
        return numpy.tile(base, (1, reps))[:, :N]

    def _sweep(part, name, make_call, sync, compiled=False, nmax=None):
        out, comp, stopped = [], [], False
        for N in _NS:
            if stopped or (nmax is not None and N > nmax):
                out.append(float("nan"))
                comp.append(float("nan"))
                continue
            try:
                if compiled:
                    fn, tc = make_call(N)
                    comp.append(tc)
                else:
                    fn = make_call(N)
                t = bench.time_call(fn, warmup=1, repeats=3, sync=sync)["median"]
            except Exception as e:
                print(f"  {part} {name} N={N}: skip ({type(e).__name__}: {str(e)[:100]})")
                t, stopped = float("nan"), True
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            print(f"  {part} {name} N={N}: {t:.4g} s", flush=True)
            out.append(t)
            stopped = stopped or t > _TSTOP
        data[part][name] = out
        if compiled:
            data["compile"][f"{part}: {name}"] = comp
        save()

    def jit_call(f, *args):
        t0 = time.perf_counter()
        jax.block_until_ready(f(*args))
        return (lambda: f(*args)), time.perf_counter() - t0

    if "cpu" in devices and "numpy" in which:
        # numpy
        def np_sample(N):
            def f():
                numpy.random.seed(_SEED)
                return sdf_np.sample(n=N)
            return f

        if "sample" in parts:
            _sweep("sample", "numpy (CPU)", np_sample, "none")
        if "eval" in parts:
            _sweep("eval", "numpy (CPU)", lambda N: (lambda p=pts(N): sdf_np(*p)), "none")
    if "cpu" in devices and "torch" in which:
        # torch (forced; numpy-built track). One intra-op thread: small kernels, and
        # 4 threads only add barrier syncs (2-10x slower on this shared box, see gS5)
        _nthr = torch.get_num_threads()
        torch.set_num_threads(1)
        data["torch_cpu_threads"] = 1
        with use("torch"):
            key_t = grandom.key(_SEED, backend="torch")
            if "sample" in parts:
                _sweep("sample", "torch (CPU, eager)",
                       lambda N: (lambda: sdf_np.sample(n=N, key=key_t)), "none")
            if "eval" in parts:
                _sweep("eval", "torch (CPU, eager)",
                       lambda N: (lambda p=[torch.as_tensor(x) for x in pts(N)]: sdf_np(*p)),
                       "none")
        torch.set_num_threads(_nthr)

    jax_devs = []
    if "cpu" in devices and "jax" in which:
        jax_devs.append((jax.devices("cpu")[0], "CPU"))
    if "gpu" in devices:
        g = [d for d in jax.devices() if d.platform == "gpu"]
        if g and "jax" in which:
            jax_devs.append((g[0], "GPU"))
    for dev, tag in jax_devs:
        with jax.default_device(dev):
            t0 = time.perf_counter()
            sdf_j = build("jax")
            jax.block_until_ready(sdf_j._interpolatedObsTrackXY)
            data["notes"][f"jax ({tag}) build for (b)/(c) [s, cold]"] = time.perf_counter() - t0
            key = jax.device_put(grandom.key(_SEED, backend="jax"), dev)

            # CPU: the 10^6 eager jax sample was OOM-killed on the shared box (the
            # closest-track-point search holds (N, 1001) temporaries); cap at 1e5.
            # GPU: no cap -- an XLA RESOURCE_EXHAUSTED is caught and recorded as NaN.
            nmax = _NMAX_JAX_CPU if tag == "CPU" else None

            def jpts(N):
                return [jax.device_put(jnp.asarray(x), dev) for x in pts(N)]

            if "sample" in parts:
                _sweep("sample", f"jax ({tag}, eager)",
                       lambda N: (lambda: sdf_j.sample(n=N, key=key)), "jax", nmax=nmax)
                _sweep("sample", f"jax ({tag}, jit)",
                       lambda N: jit_call(jax.jit(lambda k: sdf_j.sample(n=N, key=k)), key),
                       "jax", compiled=True, nmax=nmax)
            if "eval" in parts:
                _sweep("eval", f"jax ({tag}, eager)",
                       lambda N: (lambda p=jpts(N): sdf_j(*p)), "jax", nmax=nmax)
                _sweep("eval", f"jax ({tag}, jit)",
                       lambda N: jit_call(jax.jit(lambda *a: sdf_j(*a)), *jpts(N)),
                       "jax", compiled=True, nmax=nmax)

    if "gpu" in devices and torch.cuda.is_available() and "torch" in which:
        with torch.device("cuda"), use("torch"):
            key_t = grandom.key(_SEED, backend="torch")
            if "sample" in parts:
                _sweep("sample", "torch (GPU, eager)",
                       lambda N: (lambda: sdf_np.sample(n=N, key=key_t)), "torch")
            if "eval" in parts:
                _sweep("eval", "torch (GPU, eager)",
                       lambda N: (lambda p=[torch.as_tensor(x, device="cuda") for x in pts(N)]:
                                  sdf_np(*p)), "torch")

    if "cpu" in devices:
        data["env_cpu"] = _env(torch, jax)
    if "gpu" in devices:
        data["env_gpu"] = _env(torch, jax)
    save()
    return data


def _data():
    if os.path.exists(_JSON):
        with open(_JSON) as fh:
            return json.load(fh)
    return _measure(devices=("cpu", "gpu"))


def _panel(ax, xs, block, names, per_point=False):
    for name in names:
        if name not in block:
            continue
        t = numpy.asarray(block[name], dtype=float)
        if not numpy.any(numpy.isfinite(t)):
            continue
        col, ls, mk = _STYLE[name]
        ax.loglog(xs, t, color=col, ls=ls, marker=mk, ms=4, label=_LABEL.get(name, name),
                  zorder=3)


@figure(
    name="gS4_streamdf_speed",
    track="S",
    seed=0,
    description="streamdf cost per backend: construction (incl. track) vs nTrackChunks, sampling and DF evaluation vs N; numpy/C vs jax (eager/jit) vs torch, CPU & GPU [measure->json, plot byte-stable]",
)
def make():
    import matplotlib.pyplot as plt

    data = _data()
    fig, (axa, axb, axc) = plt.subplots(1, 3, figsize=(13.2, 4.3))
    nchs = numpy.asarray(data["nchs"], dtype=float)
    Ns = numpy.asarray(data["Ns"], dtype=float)
    _panel(axa, nchs, data["construct"], _ORDER)
    axa.set_xscale("log", base=2)
    axa.set_xticks(nchs)
    axa.set_xticklabels([str(int(n)) for n in nchs])
    axa.axvline(data.get("nch_default", 10), color="0.8", lw=0.8, zorder=0)
    axa.set_xlabel("number of track chunks (nTrackChunks)")
    axa.set_ylabel("wall time (s)")
    axa.set_title("(a) construction incl. stream track")
    axa.legend(fontsize=6.5, loc="center left", framealpha=0.92)
    notes = [data["notes"][k] for k in ("construct_torch", "construct_jit") if data["notes"].get(k)]
    if notes:
        import textwrap

        axa.text(0.97, 0.03, "\n".join(textwrap.fill(n, 48) for n in notes),
                 transform=axa.transAxes, fontsize=5.6, color="0.45", va="bottom", ha="right")
    _panel(axb, Ns, data["sample"], _ORDER)
    axb.set_xlabel(r"number of sampled stars $N$")
    axb.set_ylabel("wall time (s)")
    axb.set_title(r"(b) sdf.sample($N$)")
    _panel(axc, Ns, data["eval"], _ORDER)
    axc.set_xlabel(r"number of phase-space points $N$")
    axc.set_ylabel("wall time (s)")
    axc.set_title(r"(c) log DF at $N$ points")
    handles, labels = [], []
    for ax in (axb, axc):
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    axc.legend(handles, labels, fontsize=6.5, loc="upper left", framealpha=0.92)
    ec, eg = data.get("env_cpu", {}), data.get("env_gpu", {})
    axb.text(0.97, 0.03,
             f"CPU: {ec.get('cpu', '?').replace('Intel(R) Xeon(R) CPU ', 'Xeon ')}"
             f" ({ec.get('cpu_cores_used', '?')} cores)\nGPU: {eg.get('gpu', 'not measured')}\n"
             f"jax {ec.get('jax', '?')} · torch {ec.get('torch', '?')}",
             transform=axb.transAxes, fontsize=6.0, color="0.45", va="bottom", ha="right")
    fig.suptitle("streamdf (action-angle stream model): build once, then sample / evaluate",
                 y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--measure", choices=["cpu", "gpu", "all"], required=True)
    p.add_argument("--parts", default="construct,sample,eval")
    p.add_argument("--which", default="numpy,jax,torch")
    a = p.parse_args()
    _measure(devices=("cpu", "gpu") if a.measure == "all" else (a.measure,),
             parts=tuple(a.parts.split(",")), which=tuple(a.which.split(",")))
