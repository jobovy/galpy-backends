"""galpy C reference: Orbit.integrate(..., method='dop853_c', rtol=atol=1e-8) for
N orbits drawn like paper/figures/_shared/orbit_speed/galpy_c_measure.py (hiN:
nt=16 over 2 Gyr). Saves ref_N{N}.npz (ics, ts, galpy getOrbit()).

    OMP_NUM_THREADS=1 PYTHONPATH=/var/tmp/fbp python make_reference.py [N]
"""
import sys
import time

import numpy as np

import galpy
from galpy.orbit import Orbit
from galpy.potential import MWPotential2014

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
TOL = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-8
nt = 16
ts = np.linspace(0.0, 56.2, nt)
ic0 = np.array([1.0, 0.0, 1.0, 0.0125, 0.036, 0.0])


def make_ics(N, rng):  # verbatim from galpy_c_measure.py
    base = np.tile(ic0, (N, 1))
    base[:, 0] += rng.normal(0, 0.05, N)
    base[:, 1] += rng.normal(0, 0.01, N)
    base[:, 3] += rng.normal(0, 0.01, N)
    base[:, 4] += rng.normal(0, 0.01, N)
    return base


ics = make_ics(N, np.random.default_rng(N))
o = Orbit(ics)
t0 = time.perf_counter()
o.integrate(ts, MWPotential2014, method="dop853_c", rtol=TOL, atol=TOL)
dt = time.perf_counter() - t0
orb = o.getOrbit()
print(f"galpy {galpy.__file__}: N={N} in {dt:.3f}s ({dt/N*1e3:.4f} ms/orbit)")
np.savez(f"ref_N{N}.npz" if TOL == 1e-8 else f"ref_N{N}_tol{TOL:g}.npz", ics=ics, ts=ts, orbit=orb, wall_s=dt)
