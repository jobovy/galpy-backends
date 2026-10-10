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
METHOD = sys.argv[3] if len(sys.argv) > 3 else "dop853_c"
DT = float(sys.argv[4]) if len(sys.argv) > 4 else None  # fixed step (symplectic/RK)
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
o.integrate(ts, MWPotential2014, method=METHOD, rtol=TOL, atol=TOL, dt=DT)
dt = time.perf_counter() - t0
orb = o.getOrbit()
print(f"galpy {galpy.__file__}: N={N} in {dt:.3f}s ({dt/N*1e3:.4f} ms/orbit)")
if METHOD == "dop853_c" and DT is None:
    out = f"ref_N{N}.npz" if TOL == 1e-8 else f"ref_N{N}_tol{TOL:g}.npz"
else:
    out = f"ref_N{N}_{METHOD}_tol{TOL:g}" + (f"_dt{DT:g}" if DT else "") + ".npz"
np.savez(out, ics=ics, ts=ts, orbit=orb, wall_s=dt, method=METHOD, tol=TOL, dtfix=DT or 0.0)
print("wrote", out)
