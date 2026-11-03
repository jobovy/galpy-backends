"""Extend gS4's jax (CPU) sampling and DF-evaluation series to N = 10^6.

The jax CPU series were capped at N = 10^5 (make.py's _NMAX_JAX_CPU) because the
10^6 eager sample was OOM-killed on the shared box; the capped jit points cost
seconds. This re-measures, with make.py's model, seed, points and timing helper,
the jax CPU points at the given N (default 10^6): "jax (CPU, jit)" sample and eval
(compile excluded, as in make.py's jit_call), then "jax (CPU, eager)" sample and
eval (an exception, e.g. an allocation failure under the memory cap, leaves NaN).
Only those entries of data/gS4_streamdf_speed.json change; the protocol, load and
galpy commit are recorded under notes["jax_cpu_largeN_extended"].

Run pinned like the CPU series were (4 cores) under a memory cap:
  CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONPATH=<galpy worktree> \\
    taskset -c <4 cores> memcap.sh 48 python extend_jax_cpu_largeN.py [N ...]
"""
import datetime
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
import numpy
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from paper.figures._harness import bench

m = __import__("paper.figures.gS4_streamdf_speed.make", fromlist=["_JSON"])
from galpy.actionAngle import actionAngleIsochroneApprox
from galpy.backend import random as grandom
from galpy.df import streamdf
from galpy.orbit import Orbit
from galpy.potential import LogarithmicHaloPotential
from galpy.util import conversion

Ns = [int(float(a)) for a in sys.argv[1:]] or [1000000]
td = m._TD_GYR / conversion.time_in_Gyr(220.0, 8.0)
wt = os.environ.get("PYTHONPATH", "").split(os.pathsep)[0]
sha = subprocess.run(["git", "-C", wt, "rev-parse", "HEAD"], capture_output=True,
                     text=True).stdout.strip() if wt else "?"


def build(bk):  # == make.py's build
    if bk == "numpy":
        lp = LogarithmicHaloPotential(normalize=1.0, q=m._Q)
        aA = actionAngleIsochroneApprox(pot=lp, b=0.8, tintJ=m._TINTJ)
        prog = Orbit(m._IC)
    else:
        lp = LogarithmicHaloPotential(normalize=1.0, q=jnp.asarray(m._Q))
        aA = actionAngleIsochroneApprox(pot=lp, b=0.8, tintJ=m._TINTJ,
                                        integrate_method="diffrax")
        prog = Orbit(jnp.asarray(m._IC))
    return streamdf(m._SIGV, progenitor=prog, pot=lp, aA=aA, leading=True,
                    nTrackChunks=m._NCH_DEFAULT, nTrackIterations=0, tdisrupt=td)


t0 = time.perf_counter()
sdf_np = build("numpy")
numpy.random.seed(m._SEED)
base = numpy.asarray(sdf_np.sample(n=m._NBASE), dtype=float)  # (6, NBASE)
print(f"numpy sdf built in {time.perf_counter() - t0:.0f} s", flush=True)


def pts(N):  # == make.py's pts
    reps = N // m._NBASE + 1
    return numpy.tile(base, (1, reps))[:, :N]


with open(m._JSON) as fh:
    data = json.load(fh)
assert data["Ns"] == m._NS
notes = data["notes"].setdefault("jax_cpu_largeN_extended", {})


def save():
    with open(m._JSON, "w") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
        fh.write("\n")


def record(part, name, i, t, proto, extra=None):
    ser = data[part].setdefault(name, [float("nan")] * len(m._NS))
    ser[i] = t
    notes[f"{part}: {name} N={m._NS[i]}"] = {
        "protocol": proto, "loadavg": list(os.getloadavg()),
        "cores": len(os.sched_getaffinity(0)), "galpy_sha": sha,
        "date": datetime.datetime.now().isoformat(timespec="seconds"), **(extra or {})}
    save()
    print(f"  {part} {name} N={m._NS[i]}: {t:.4g} s  {extra or ''}", flush=True)


cpu = jax.devices("cpu")[0]
with jax.default_device(cpu):
    t0 = time.perf_counter()
    sdf_j = build("jax")
    jax.block_until_ready(sdf_j._interpolatedObsTrackXY)
    print(f"jax sdf built in {time.perf_counter() - t0:.0f} s", flush=True)
    key = jax.device_put(grandom.key(m._SEED, backend="jax"), cpu)
    for N in Ns:
        i = m._NS.index(N)
        p = [jax.device_put(jnp.asarray(x), cpu) for x in pts(N)]
        # jit: compile excluded (make.py's jit_call), warm-up 1, median of 3
        f = jax.jit(lambda k: sdf_j.sample(n=N, key=k))
        t0 = time.perf_counter()
        jax.block_until_ready(f(key))
        tc = time.perf_counter() - t0
        data["compile"].setdefault("sample: jax (CPU, jit)", [float("nan")] * len(m._NS))[i] = tc
        t = bench.time_call(lambda: f(key), warmup=1, repeats=3, sync="jax")["median"]
        record("sample", "jax (CPU, jit)", i, t, "warmup=1, median of 3", {"compile_s": tc})
        g = jax.jit(lambda *a: sdf_j(*a))
        t0 = time.perf_counter()
        jax.block_until_ready(g(*p))
        tc = time.perf_counter() - t0
        data["compile"].setdefault("eval: jax (CPU, jit)", [float("nan")] * len(m._NS))[i] = tc
        t = bench.time_call(lambda: g(*p), warmup=1, repeats=3, sync="jax")["median"]
        record("eval", "jax (CPU, jit)", i, t, "warmup=1, median of 3", {"compile_s": tc})
        # eager: an exception (allocation failure under the memory cap) leaves NaN
        for part, call in (("sample", lambda: sdf_j.sample(n=N, key=key)),
                           ("eval", lambda: sdf_j(*p))):
            try:
                t = bench.time_call(call, warmup=1, repeats=3, sync="jax")["median"]
                record(part, "jax (CPU, eager)", i, t, "warmup=1, median of 3")
            except Exception as e:
                notes[f"{part}: jax (CPU, eager) N={N}"] = {
                    "error": f"{type(e).__name__}: {str(e)[:200]}", "galpy_sha": sha,
                    "date": datetime.datetime.now().isoformat(timespec="seconds")}
                save()
                print(f"  {part} jax (CPU, eager) N={N}: ERROR {type(e).__name__}: {str(e)[:120]}",
                      flush=True)
print("written", m._JSON)
