"""Time galpy's dop853_c exactly like galpy_c_measure.py hiN (best of 3), N orbits."""
import os, sys, time
import numpy as np
from galpy.orbit import Orbit
from galpy.potential import MWPotential2014
N = int(sys.argv[1])
ts = np.linspace(0.0, 56.2, 16)
ic0 = np.array([1.0, 0.0, 1.0, 0.0125, 0.036, 0.0])
rng = np.random.default_rng(N)
base = np.tile(ic0, (N, 1))
base[:, 0] += rng.normal(0, 0.05, N); base[:, 1] += rng.normal(0, 0.01, N)
base[:, 3] += rng.normal(0, 0.01, N); base[:, 4] += rng.normal(0, 0.01, N)
o = Orbit(base); o.integrate(ts, MWPotential2014, method="dop853_c", rtol=1e-8, atol=1e-8, progressbar=False)
best = 1e18
for _ in range(3):
    o = Orbit(base); t0 = time.perf_counter()
    o.integrate(ts, MWPotential2014, method="dop853_c", rtol=1e-8, atol=1e-8, progressbar=False)
    best = min(best, time.perf_counter() - t0)
print(f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')} affinity={len(os.sched_getaffinity(0))} N={N}: {best:.3f}s = {best/N*1e3:.4f} ms/orbit")
