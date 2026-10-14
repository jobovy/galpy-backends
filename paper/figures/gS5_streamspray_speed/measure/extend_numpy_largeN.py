"""Extend gS5's "numpy (CPU, C)" series to the N the 60 s stop rule left NaN
(N=1e5, 3e5, 1e6) so it can be compared with the GPU series at the same N.

Same model, seed, chunking and timing helper as make.py's _measure (imported from
it); only the named points of that one series are written into
data/gS5_streamspray_speed.json -- every other entry is left untouched.

Run pinned like the CPU series were (4 cores):
  CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 PYTHONPATH=<worktree> \\
    taskset -c <4 cores> python extend_numpy_largeN.py [N ...]      (default: 1e5 3e5 1e6)
Protocol: 1 warm-up + median of 3 (make.py's) for N <= 3e5; N >= 1e6 is timed
once after one warm-up chunk (a single call is ~35 min) -- recorded in the JSON.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
import numpy

from paper.figures._harness import bench

m = __import__("paper.figures.gS5_streamspray_speed.make", fromlist=["_JSON"])
from galpy.df import fardal15spraydf
from galpy.orbit import Orbit
from galpy.potential import LogarithmicHaloPotential
from galpy.util import conversion

Ns = [int(float(a)) for a in sys.argv[1:]] or [100000, 300000, 1000000]
mass = m._MASS_MSUN / conversion.mass_in_msol(220.0, 8.0)
td = m._TD_GYR / conversion.time_in_Gyr(220.0, 8.0)
pot = LogarithmicHaloPotential(normalize=1.0, q=m._Q)
spdf = fardal15spraydf(mass, progenitor=Orbit(m._IC), pot=pot, tdisrupt=td)


def numpy_call(N):  # == make.py's numpy_call
    def f():
        numpy.random.seed(m._SEED)
        outs = [
            spdf.sample(min(m._NUMPY_CHUNK, N - i), return_orbit=False)
            for i in range(0, N, m._NUMPY_CHUNK)
        ]
        return numpy.hstack(outs)

    return f


spdf.sample(m._NUMPY_CHUNK, return_orbit=False)  # warm the caches once
with open(m._JSON) as fh:
    data = json.load(fh)
ser = data["series"]["numpy (CPU, C)"]
assert data["Ns"] == m._NS and len(ser) == len(m._NS)
notes = data.setdefault("env_cpu", {}).setdefault("numpy_largeN_extended", {})
for N in Ns:
    i = m._NS.index(N)
    if N >= 1000000:
        t0 = time.perf_counter()
        numpy_call(N)()
        t, proto = time.perf_counter() - t0, "single timed call (no warm-up)"
    else:
        t, proto = bench.time_call(numpy_call(N), warmup=1, repeats=3)["median"], "warmup=1, median of 3"
    ser[i] = t
    notes[str(N)] = {"protocol": proto, "loadavg": list(os.getloadavg()),
                     "cores": len(os.sched_getaffinity(0)), "date": "2026-10-07"}
    print(f"numpy (CPU, C) N={N}: {t:.4g} s  ({1e3 * t / N:.3f} ms/particle)", flush=True)
    with open(m._JSON, "w") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
        fh.write("\n")
print("written", m._JSON)
