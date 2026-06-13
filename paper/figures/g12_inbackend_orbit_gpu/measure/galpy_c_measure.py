"""g12 measure -- galpy's compiled C integrator (dop853_c), CPU, sequential.

The classical baseline for the in-backend GPU lines: the SAME MWPotential2014 /
2 Gyr / Sun-like batch, integrated by galpy's compiled ``dop853_c`` one orbit at a
time in a single-threaded C loop. The in-backend integrator's rectangular EOM is
modelled on this C integrator (integrateFullOrbit.c), so dop853_c is the
apples-to-apples sequential-CPU reference. Tolerances matched to the GPU lines
(rtol=atol=1e-8) instead of galpy's tighter 1e-10 default.

Run this in the MAIN env (galpy with the C extension BUILT -- /home/bovy/Repos/galpy
on the default path; do NOT put the pure-backend worktree first, it has no libgalpy):

    python galpy_c_measure.py

Writes data/galpy_c_cpu.json (machine + version stamped).
"""

import json
import os
import socket
import time

import numpy as np

import galpy  # noqa: E402
from galpy.orbit import Orbit  # noqa: E402
from galpy.potential import MWPotential2014  # noqa: E402

_DATA = os.path.join(os.path.dirname(__file__), "..", "data")

ts = np.linspace(0.0, 56.2, 16)  # 2 Gyr; saves are cheap for C, few are enough
ic0 = np.array([1.0, 0.0, 1.0, 0.0125, 0.036, 0.0])  # Sun-like [R,vR,vT,z,vz,phi]


def make_ics(N, rng):
    base = np.tile(ic0, (N, 1))
    base[:, 0] += rng.normal(0, 0.05, N)  # R
    base[:, 1] += rng.normal(0, 0.01, N)  # vR
    base[:, 3] += rng.normal(0, 0.01, N)  # z
    base[:, 4] += rng.normal(0, 0.01, N)  # vz
    return base


out = {
    "backend": "galpy",
    "device": "cpu-C",
    "integrator": "dop853_c (compiled, single-thread, sequential)",
    "nt": 16,
    "tspan_Gyr": 2.0,
    "rtol": 1e-8,
    "atol": 1e-8,
    "potential": "MWPotential2014",
    "host": socket.gethostname(),
    "galpy_version": galpy.__version__,
    "galpy_file": galpy.__file__,
    "results": [],
}
for N in [1, 10, 100, 1000, 10000, 30000]:
    rng = np.random.default_rng(N)
    ics = make_ics(N, rng)
    o = Orbit(ics if N > 1 else ics[0])  # warmup (caches)
    o.integrate(ts, MWPotential2014, method="dop853_c", rtol=1e-8, atol=1e-8)
    best = 1e18
    for _ in range(3):
        o = Orbit(ics if N > 1 else ics[0])
        t0 = time.perf_counter()
        o.integrate(ts, MWPotential2014, method="dop853_c", rtol=1e-8, atol=1e-8)
        best = min(best, time.perf_counter() - t0)
    out["results"].append({"N": N, "wall_s": best, "per_orbit_ms": best / N * 1e3})
    print(f"galpy[cpu-C] N={N:7d}: {best * 1e3:9.2f} ms  ({best / N * 1e3:.5f} ms/orbit)", flush=True)
    os.makedirs(_DATA, exist_ok=True)
    with open(os.path.join(_DATA, "galpy_c_cpu.json"), "w") as fh:
        json.dump(out, fh, indent=2)
print("done -> galpy_c_cpu.json")
