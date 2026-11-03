"""Shared measure -- galpy's in-backend constant-step symplectic integrator
(method='symplec6_jax', galpy PR "in-backend symplectic integrators", 2026-10-28)
on JAX, CPU or GPU: the symplectic line of the orbit-speed figure (g12) that JB's
review asked for, next to the adaptive in-backend Dopri8 line of
galpy_jax_measure.py.

Same experiment as the other shared series: MWPotential2014, 2 Gyr, a batch of N
per-N-seeded Sun-like ICs, wall time for the whole batch (steady state, compile
excluded, best of 3) and the first-call compile+run time. The step: NSTEPS
constant steps over the 2 Gyr, chosen such that the worst-case position error
of the batch after 2 Gyr matches that of the adaptive Dopri8 line at
rtol=atol=1e-8 (measured with --calibrate: the sixth-order error falls as
nsteps^-6, so this is a sharp choice). One batched jax.jit of Orbit.integrate
(the public route), no vmap: the constant steps are the same for every orbit.

nt convention: lowN -> nt=1000 (a multiple-of-999 step count), hiN -> nt=16.

    PYTHONPATH=<galpy worktree with the PR> GALPY_WORKTREE=<same> \\
        python galpy_symplec_measure.py gpu lowN
    ... gpu hiN / cpu lowN / cpu hiN
    ... --calibrate      (prints the worst-case error of symplec6 vs dop853_c at
                          a few step counts and of the Dopri8 line, then exits)

Writes data/symplec6_jax_<device>.json (one regime, nt=16, for every N).
"""

import argparse
import json
import os
import socket
import sys
import time

_ap = argparse.ArgumentParser()
_ap.add_argument("device", nargs="?", default="gpu", help="'gpu' (default) or 'cpu'")
_ap.add_argument("mode", nargs="?", default="lowN", help="'lowN' (nt=1000) or 'hiN' (nt=16)")
_ap.add_argument("--nsteps", type=int, default=None, help="override the step count")
_ap.add_argument("--calibrate", action="store_true")
_args = _ap.parse_args()
device, mode = _args.device, _args.mode
if device == "cpu":
    os.environ["JAX_PLATFORMS"] = "cpu"  # force CPU before jax import

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy  # noqa: E402

jax.config.update("jax_enable_x64", True)
import galpy  # noqa: E402
from galpy.orbit import Orbit  # noqa: E402
from galpy.potential import MWPotential2014  # noqa: E402

assert galpy.__file__.startswith(
    os.environ.get("GALPY_WORKTREE", "/var/tmp/backends-merge")
), galpy.__file__

_DATA = os.path.join(os.path.dirname(__file__), "data")
TSPAN = 56.2  # 2 Gyr in galpy internal units (ro=8, vo=220)
# One regime for the whole N range: 16 sparse saves (the hiN convention), since a
# constant-step method's cost does not depend on the saves and the lowN
# convention's 1000 saves would force >= 999 steps. The step count (a multiple
# of nt - 1 = 15): 450 steps of 2 Gyr, 0.125 internal units (4.5 Myr), at which
# the worst-case position error of the batch after 2 Gyr (--calibrate, 300
# orbits) matches the adaptive Dopri8 line at rtol=atol=1e-8 (9e-6): the
# sixth-order error falls as nsteps^-6, so the match is sharp, and 450 x 7 =
# 3150 force evaluations against Dopri8's ~3250.
NSTEPS = {16: 450}
nt, suffix = 16, ""
Ns = [3000000, 10000000]  # H100 extension (only change)
if mode == "hiN":  # kept for the command-line convention of the other measures
    pass
nsteps = _args.nsteps or NSTEPS[nt]
assert nsteps % (nt - 1) == 0, (nsteps, nt)
ts = numpy.linspace(0.0, TSPAN, nt)
ic0 = numpy.array([1.0, 0.0, 1.0, 0.0125, 0.036, 0.0])  # Sun-like [R,vR,vT,z,vz,phi]
scale = numpy.array([0.05, 0.01, 0.0, 0.01, 0.01, 0.0])


def _ics(N):
    return ic0[None] + numpy.asarray(jax.random.normal(jax.random.PRNGKey(N), (N, 6))) * scale


@jax.jit
def batch(ics):
    o = Orbit(ics)
    o.integrate(ts, MWPotential2014, method="symplec6_jax", inbackend_kwargs={"nsteps": nsteps})
    return o.getOrbit()


def _cart(y):
    R, vR, vT, z, vz, phi = (y[..., i] for i in range(6))
    c, s = numpy.cos(phi), numpy.sin(phi)
    return numpy.stack([R * c, R * s, z], -1)


if _args.calibrate:
    ics = _ics(300)
    ref = Orbit(ics)
    ref.integrate(ts, MWPotential2014, method="dop853_c", rtol=1e-13, atol=1e-13)
    pref = _cart(ref.getOrbit()[:, -1])
    o8 = Orbit(ics)
    o8.integrate(ts, MWPotential2014, method="dop853_c", rtol=1e-8, atol=1e-8)
    print(f"dop853 rtol=atol=1e-8 worst-case position error: {numpy.max(numpy.abs(_cart(o8.getOrbit()[:, -1]) - pref)):.3e}")
    for n in (300, 450, 600, 900, 1800):
        o = Orbit(jnp.asarray(ics))
        o.integrate(ts, MWPotential2014, method="symplec6_jax", inbackend_kwargs={"nsteps": n})
        err = numpy.max(numpy.abs(_cart(numpy.asarray(o.getOrbit())[:, -1]) - pref))
        print(f"symplec6_jax nsteps={n}: worst-case position error {err:.3e}")
    sys.exit(0)

plat = jax.devices()[0].platform
out = {
    "backend": "galpy-jax",
    "device": plat,
    "integrator": f"symplec6_jax (in-backend, constant step, nsteps={nsteps})",
    "stepping": "constant, shared by the batch",
    "nsteps": nsteps,
    "nt": nt,
    "tspan_Gyr": 2.0,
    "potential": "MWPotential2014",
    "host": socket.gethostname(),
    "cpu_affinity": len(os.sched_getaffinity(0)),
    "thread_env": {v: os.environ.get(v, "unset")
                   for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "XLA_FLAGS")},
    "results": [],
}
for m in ("jax", "galpy", "numpy"):
    out[m + "_version"] = __import__(m).__version__
try:
    import torch

    out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else plat
except Exception:
    out["gpu"] = plat
out["galpy_file"] = galpy.__file__


def _dump(out):
    os.makedirs(_DATA, exist_ok=True)
    with open(os.path.join(_DATA, f"symplec6_jax_{plat}{suffix}.json"), "w") as fh:
        json.dump(out, fh, indent=2)


for N in Ns:
    ics = jnp.asarray(_ics(N))
    try:
        t0 = time.perf_counter()
        batch(ics).block_until_ready()  # first call: XLA compile + run
        compile_run_s = time.perf_counter() - t0
        best = min(
            (lambda t0: (batch(ics).block_until_ready(), time.perf_counter() - t0)[1])(time.perf_counter())
            for _ in range(3)
        )
        out["results"].append({"N": N, "wall_s": best, "per_orbit_ms": best / N * 1e3,
                               "compile_run_s": compile_run_s})
        print(f"galpy-symplec6[{plat},{mode}] N={N:8d}: steady={best * 1e3:9.2f} ms "
              f"({best / N * 1e3:.5f} ms/orbit)  compile+run={compile_run_s:6.2f} s", flush=True)
        _dump(out)
    except Exception as e:  # OOM: stop and record where
        print(f"galpy-symplec6[{plat}] N={N}: STOP {type(e).__name__}: {str(e)[:70]}", flush=True)
        out["stopped_at"] = {"N": N, "error": type(e).__name__}
        _dump(out)
        break
print(f"done -> symplec6_jax_{plat}{suffix}.json")
