"""Canonical shared measure -- galpy compiled C integrator (dop853_c), CPU, sequential.

This is the SINGLE source of truth for the galpy-C orbit-integration curve that
BOTH g8 (galpy vs galax) and g12 (galpy in-backend GPU/CPU/C) load, so the two
figures show a byte-identical C line.  Same orbit/tolerance/nt convention as the
shared jax measure (galpy_jax_measure.py):

  * MWPotential2014, Sun-like IC [R,vR,vT,z,vz,phi]=[1,0,1,0.0125,0.036,0] in galpy
    internal units, perturbed by a per-N-seeded Gaussian (scale below);
  * 2 Gyr span = linspace(0, 56.2, nt) internal time units (ro=8, vo=220);
  * rtol = atol = 1e-8 (matched to the galax OrbitSolver default and the in-backend
    GPU lines, NOT galpy's tighter 1e-10 default);
  * nt convention: lowN regime -> nt=1000 dense saves (resolved small-N crossover);
    hiN regime -> nt=16 sparse saves (the large-N convention the GPU lines need to
    fit 12 GiB).  C saves are cheap so the two regimes agree on per-orbit cost at
    their overlap -> ONE consistent C curve.

Run in an env where galpy has the C extension BUILT.  The rebased backend worktree
(/var/tmp/backends-merge, feat/backends @36c92e8f) now ships libgalpy.*.so, so it
serves as the single galpy for ALL shared series:

    PYTHONPATH=/var/tmp/backends-merge python galpy_c_measure.py lowN
    PYTHONPATH=/var/tmp/backends-merge python galpy_c_measure.py hiN
    PYTHONPATH=<worktree> GALPY_WORKTREE=<worktree> python galpy_c_measure.py hiN_ext
      (appends N=1e5, 3e5, 1e6 to galpy_c_cpu_hiN.json; existing points are kept)
    ... galpy_c_measure.py <mode> --numcores K     (multi-core variant)
      C orbits run on OpenMP threads (omp_get_max_threads; the ``numcores``
      kwarg is ignored on the C path), so this sets OMP_NUM_THREADS=K before
      galpy loads and writes data/galpy_c_cpu_Kcores[_hiN].json instead; pin it
      (e.g. ``taskset -c 16-31`` for K=16). Without --numcores nothing changes.

Writes data/galpy_c_cpu[_hiN].json (machine + version stamped).  The plots
(../../g8.../make.py, ../../g12.../make.py) are byte-stable from these JSONs.
"""

import json
import os
import socket
import sys
import time

import numpy as np

_argv = list(sys.argv[1:])
_ncores = None
if "--numcores" in _argv:  # multi-core variant: OpenMP threads, set before galpy loads
    _i = _argv.index("--numcores")
    _ncores = int(_argv[_i + 1])
    del _argv[_i : _i + 2]
    os.environ["OMP_NUM_THREADS"] = str(_ncores)

import galpy  # noqa: E402
from galpy.orbit import Orbit  # noqa: E402
from galpy.potential import MWPotential2014  # noqa: E402

assert galpy.__file__.startswith(
    os.environ.get("GALPY_WORKTREE", "/var/tmp/backends-merge")
), galpy.__file__

mode = _argv[0] if _argv else "lowN"  # 'lowN', 'hiN' or 'hiN_ext'
if mode == "hiN":
    nt, Ns, suffix = 16, [1000, 10000, 30000], "_hiN"
elif mode == "hiN_ext":  # extend the hiN curve to the GPU series' largest N, appending
    nt, Ns, suffix = 16, [100000, 300000, 1000000], "_hiN"  # to the existing JSON
else:
    nt, Ns, suffix = 1000, [1, 10, 100, 1000, 10000], ""

if _ncores is not None:
    suffix = f"_{_ncores}cores{suffix}"

_DATA = os.path.join(os.path.dirname(__file__), "data")
ts = np.linspace(0.0, 56.2, nt)  # 2 Gyr in galpy internal units
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
    "nt": nt,
    "tspan_Gyr": 2.0,
    "rtol": 1e-8,
    "atol": 1e-8,
    "potential": "MWPotential2014",
    "host": socket.gethostname(),
    "galpy_version": galpy.__version__,
    "galpy_file": galpy.__file__,
    "results": [],
}
if _ncores is not None:
    out["integrator"] = f"dop853_c (compiled, OpenMP, OMP_NUM_THREADS={_ncores})"
    out["omp_num_threads"] = _ncores
    out["cpu_affinity"] = len(os.sched_getaffinity(0))
_OUT = os.path.join(_DATA, f"galpy_c_cpu{suffix}.json")
if mode == "hiN_ext" and os.path.exists(_OUT):  # keep the existing points + stamps
    with open(_OUT) as fh:
        prev = json.load(fh)
    out.update({k: v for k, v in prev.items() if k != "results"})
    out["results"] = [r for r in prev["results"] if r["N"] not in Ns]
    out["extended"] = {"Ns": Ns, "host": socket.gethostname(),
                       "galpy_version": galpy.__version__, "galpy_file": galpy.__file__}
for N in Ns:
    rng = np.random.default_rng(N)
    ics = make_ics(N, rng)
    o = Orbit(ics if N > 1 else ics[0])  # warmup (caches)
    o.integrate(ts, MWPotential2014, method="dop853_c", rtol=1e-8, atol=1e-8, numcores=_ncores or 1)
    best = 1e18
    for _ in range(3):
        o = Orbit(ics if N > 1 else ics[0])
        t0 = time.perf_counter()
        o.integrate(ts, MWPotential2014, method="dop853_c", rtol=1e-8, atol=1e-8, numcores=_ncores or 1)
        best = min(best, time.perf_counter() - t0)
    out["results"].append({"N": N, "wall_s": best, "per_orbit_ms": best / N * 1e3})
    print(
        f"galpy[cpu-C,{mode}] N={N:7d}: {best * 1e3:9.2f} ms  ({best / N * 1e3:.5f} ms/orbit)",
        flush=True,
    )
    os.makedirs(_DATA, exist_ok=True)
    with open(os.path.join(_DATA, f"galpy_c_cpu{suffix}.json"), "w") as fh:
        json.dump(out, fh, indent=2)
print(f"done -> galpy_c_cpu{suffix}.json")
