import numpy as np, time, json
from galpy.orbit import Orbit
from galpy.potential import MWPotential2014
import galpy
# 2 Gyr in galpy internal units (ro=8, vo=220 -> 1 t.u. = 35.6 Myr), 1000 points
TU_GYR = 0.035561  # ro/vo in Gyr
ts = np.linspace(0.0, 2.0 / TU_GYR, 1000)
# Sun-like IC in internal units [R, vR, vT, z, vz, phi]
ic0 = np.array([1.0, 0.0, 1.0, 0.0125, 0.036, 0.0])

def make_ics(N, rng):
    base = np.tile(ic0, (N, 1))
    base[:, 0] += rng.normal(0, 0.05, N)   # R
    base[:, 1] += rng.normal(0, 0.01, N)   # vR
    base[:, 3] += rng.normal(0, 0.01, N)   # z
    base[:, 4] += rng.normal(0, 0.01, N)   # vz
    return base

out = {"backend": "galpy", "device": "cpu-C", "method": "dop853_c",
       "nt": 1000, "tspan_Gyr": 2.0, "galpy_version": galpy.__version__, "results": []}
for N in [1, 10, 100, 1000, 10000, 30000]:
    rng = np.random.default_rng(N)
    ics = make_ics(N, rng)
    # warmup (first call compiles nothing for C, but warms caches)
    o = Orbit(ics if N > 1 else ics[0]); o.integrate(ts, MWPotential2014, method="dop853_c")
    best = 1e9
    for _ in range(3):
        o = Orbit(ics if N > 1 else ics[0])
        t0 = time.perf_counter()
        o.integrate(ts, MWPotential2014, method="dop853_c")
        best = min(best, time.perf_counter() - t0)
    out["results"].append({"N": N, "wall_s": best, "per_orbit_ms": best/N*1e3})
    print(f"galpy[cpu-C] N={N:7d}: {best*1e3:9.2f} ms  ({best/N*1e3:.5f} ms/orbit)", flush=True)
    json.dump(out, open("/var/tmp/galax-bench/galpy_cpu-C.json", "w"), indent=2)
print("done -> galpy_cpu-C.json")
print("done -> galpy_cpu-C.json")
