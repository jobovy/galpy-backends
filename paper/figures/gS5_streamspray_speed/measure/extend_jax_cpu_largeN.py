"""Extend gS5's "jax (CPU, eager)" and "jax (CPU, jit)" series to the N the 60 s
stop rule left NaN (N = 3e5, 1e6), so they can be compared with the GPU series at
the same N (the companion of extend_numpy_largeN.py).

Same model, seed and timing helper as make.py's _measure (imported from it); only
the named points of those two series (and the jit compile times) are written into
data/gS5_streamspray_speed.json -- every other entry is left untouched; the
protocol, load and galpy commit go under env_cpu["jax_largeN_extended"].

Run pinned under a memory cap (the CPU series were measured on ONE pinned core on
an idle box; the 2026-11-03 extension ran on a 4-core mask on a busy node, which
env_cpu["jax_largeN_extended"] records -- see make.py's docstring):
  CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONPATH=<galpy worktree> \\
    OMP_NUM_THREADS=1 taskset -c <cores> memcap.sh 32 python extend_jax_cpu_largeN.py [N ...]
Protocol: jit -- compile excluded (make.py's compiled()), then 1 warm-up + median
of 3 for N <= 3e5 and a single timed call at N >= 1e6 (one call is ~23 min);
eager -- 1 warm-up (diffrax's internal compile for the new shape) + median of 3
for N <= 3e5, 1 warm-up + a single timed call at N >= 1e6 (~27 min).
"""
import datetime
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from paper.figures._harness import bench

m = __import__("paper.figures.gS5_streamspray_speed.make", fromlist=["_JSON"])
from galpy.backend import random as grandom
from galpy.df import fardal15spraydf
from galpy.orbit import Orbit
from galpy.potential import LogarithmicHaloPotential
from galpy.util import conversion

Ns = [int(float(a)) for a in sys.argv[1:]] or [300000, 1000000]
mass = m._MASS_MSUN / conversion.mass_in_msol(220.0, 8.0)
td = m._TD_GYR / conversion.time_in_Gyr(220.0, 8.0)
wt = os.environ.get("PYTHONPATH", "").split(os.pathsep)[0]
sha = subprocess.run(["git", "-C", wt, "rev-parse", "HEAD"], capture_output=True,
                     text=True).stdout.strip() if wt else "?"

with open(m._JSON) as fh:
    data = json.load(fh)
assert data["Ns"] == m._NS
notes = data.setdefault("env_cpu", {}).setdefault("jax_largeN_extended", {})


def save():
    with open(m._JSON, "w") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
        fh.write("\n")


def record(name, i, t, proto, extra=None):
    ser = data["series"].setdefault(name, [float("nan")] * len(m._NS))
    ser[i] = t
    notes[f"{name} N={m._NS[i]}"] = {
        "protocol": proto, "loadavg": list(os.getloadavg()),
        "cores": len(os.sched_getaffinity(0)), "galpy_sha": sha,
        "date": datetime.datetime.now().isoformat(timespec="seconds"), **(extra or {})}
    save()
    print(f"  {name} N={m._NS[i]}: {t:.4g} s  ({1e3 * t / m._NS[i]:.3f} ms/particle)  {extra or ''}",
          flush=True)


def timed(call, N):
    """1 warm-up + median of 3 (make.py's) for N <= 3e5; 1 warm-up + one timed call above."""
    if N >= 1000000:
        jax.block_until_ready(call())
        t0 = time.perf_counter()
        jax.block_until_ready(call())
        return time.perf_counter() - t0, "warmup=1, single timed call"
    return bench.time_call(call, warmup=1, repeats=3, sync="jax")["median"], "warmup=1, median of 3"


cpu = jax.devices("cpu")[0]
with jax.default_device(cpu):
    pot = LogarithmicHaloPotential(normalize=1.0, q=jnp.asarray(m._Q))  # == make.py's build("jax")
    spdf = fardal15spraydf(mass, progenitor=Orbit(jnp.asarray(m._IC)), pot=pot, tdisrupt=td)
    key = jax.device_put(grandom.key(m._SEED, backend="jax"), cpu)
    for N in Ns:
        i = m._NS.index(N)
        f = jax.jit(lambda k: spdf.sample(N, return_orbit=False, key=k))
        t0 = time.perf_counter()
        jax.block_until_ready(f(key))
        tc = time.perf_counter() - t0
        data["compile"].setdefault("jax (CPU, jit)", [float("nan")] * len(m._NS))[i] = tc
        t, proto = timed(lambda: f(key), N)
        record("jax (CPU, jit)", i, t, proto, {"compile_s": tc})
        t, proto = timed(lambda: spdf.sample(N, return_orbit=False, key=key), N)
        record("jax (CPU, eager)", i, t, proto)
print("written", m._JSON)
