"""galpy C dop853_c: RHS-evaluation count per orbit, and a threading sanity check.

(1) nfev. galpy's dop853_c (galpy/util/leung_dop853.c) has a line-by-line Python
    port (galpy/util/leung_dop853.py, same Hairer DOP853 tableau, initial-step
    heuristic and controller). The C code exposes no counter, so the count is
    taken by running the PORT with a counting rectangular RHS (state
    [x,y,z,vx,vy,vz], as integrateFullOrbit.c integrates) for the g8/g12 orbit
    (MWPotential2014, 2 Gyr, rtol=atol=1e-8) over a sample of the measured ICs.
(2) Threading check (a short timing, NOT a benchmark): the C orbit loop is an
    OpenMP ``parallel for`` with omp_get_max_threads() threads (the ``numcores``
    kwarg does not reach it), so galpy_c_measure.py's "single-thread" label holds
    only if OMP_NUM_THREADS=1 was in effect. This times dop853_c on ONE pinned
    core with OMP_NUM_THREADS=1 and compares with galpy_c_cpu_hiN.json.

Run (pin to one core; the timing part must be on an idle core):
  taskset -c 30 nice -n 19 env OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
    PYTHONPATH=/var/tmp/fbp python c_dop853.py
Writes c_dop853.json next to this script.
"""

import json
import os
import time

assert os.environ.get("OMP_NUM_THREADS") == "1", "run with OMP_NUM_THREADS=1"
import numpy

from galpy.orbit import Orbit
from galpy.potential import MWPotential2014
from galpy.potential.Potential import _evaluatephitorques, _evaluateRforces, _evaluatezforces
from galpy.util.leung_dop853 import dop853

HERE = os.path.dirname(os.path.abspath(__file__))
ic0 = numpy.array([1.0, 0.0, 1.0, 0.0125, 0.036, 0.0])


def make_ics(N, rng):  # galpy_c_measure.py's IC draw
    base = numpy.tile(ic0, (N, 1))
    base[:, 0] += rng.normal(0, 0.05, N)
    base[:, 1] += rng.normal(0, 0.01, N)
    base[:, 3] += rng.normal(0, 0.01, N)
    base[:, 4] += rng.normal(0, 0.01, N)
    return base


calls = {"n": 0}


def rhs(w, t):
    calls["n"] += 1
    x, y, z, vx, vy, vz = w
    R = numpy.sqrt(x * x + y * y)
    phi = numpy.arctan2(y, x)
    cp, sp = x / R, y / R
    FR = _evaluateRforces(MWPotential2014, R, z, phi=phi, t=t)
    Fp = _evaluatephitorques(MWPotential2014, R, z, phi=phi, t=t)
    Fz = _evaluatezforces(MWPotential2014, R, z, phi=phi, t=t)
    return [vx, vy, vz, cp * FR - sp / R * Fp, sp * FR + cp / R * Fp, Fz]


def to_rect(v):
    R, vR, vT, z, vz, phi = v
    c, s = numpy.cos(phi), numpy.sin(phi)
    return numpy.array([R * c, R * s, z, vR * c - vT * s, vR * s + vT * c, vz])


out = {"what": "dop853 (galpy C tableau, via its Python port) RHS evaluations per orbit"}
for nt in (16, 1000):
    ts = numpy.linspace(0.0, 56.2, nt)
    ics = make_ics(24, numpy.random.default_rng(12345))
    nf = []
    for v in ics:
        calls["n"] = 0
        dop853(rhs, to_rect(v), ts, rtol=1e-8, atol=1e-8)
        nf.append(calls["n"])
    out[f"nfev_nt{nt}"] = dict(mean=float(numpy.mean(nf)), min=int(min(nf)), max=int(max(nf)),
                               n_orbits=len(nf))
    print("nt", nt, out[f"nfev_nt{nt}"], flush=True)

# (2) threading check: one core, OMP_NUM_THREADS=1, the measure's protocol (best of 3)
ts = numpy.linspace(0.0, 56.2, 16)
for N in (300, 3000):
    ics = make_ics(N, numpy.random.default_rng(N))
    o = Orbit(ics)
    o.integrate(ts, MWPotential2014, method="dop853_c", rtol=1e-8, atol=1e-8)
    best = 1e9
    for _ in range(3):
        o = Orbit(ics)
        t0 = time.perf_counter()
        o.integrate(ts, MWPotential2014, method="dop853_c", rtol=1e-8, atol=1e-8)
        best = min(best, time.perf_counter() - t0)
    out[f"one_thread_N{N}"] = dict(wall_s=best, per_orbit_ms=best / N * 1e3)
    print(N, out[f"one_thread_N{N}"], flush=True)
out["affinity"] = sorted(os.sched_getaffinity(0))
with open(os.path.join(HERE, "c_dop853.json"), "w") as fh:
    json.dump(out, fh, indent=1)
