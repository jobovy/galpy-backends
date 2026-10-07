"""g12 measure -- the SAME-WORK match to torch: galpy in-backend on jax with
diffrax **Dopri5** (5th order) instead of the shipped Dopri8 (8th order).

galpy ships diffrax Dopri8 on jax and torchdiffeq dopri5 on torch (torchdiffeq's
dopri8 *backward* is buggy -- see galpy.backend._torch.orbit_ode). A Dopri8-vs-dopri5
wall-time comparison is therefore NOT same-work: the work_counts.py measure shows
Dopri8 takes ~180 accepted steps/orbit vs dopri5's ~560, i.e. a different number of
RHS evaluations. This script runs the in-backend jax path with diffrax **Dopri5** --
the same 5th-order method (and the work_counts measure confirms matching nfev/steps
to torchdiffeq dopri5) -- so g12 can show the apples-to-apples pair
jax-Dopri5 vs torch-dopri5 alongside the production jax-Dopri8 curve.

Identical to the shared galpy_jax_measure.py in EVERYTHING except the solver:
same orbit (MWPotential2014, Sun-like IC, 2 Gyr = linspace(0,56.2,nt)), same
rtol=atol=1e-8, same per-orbit adaptive vmap-of-solve, same nt convention
(lowN nt=1000 dense; hiN nt=16 sparse), same per-N seeded ICs. We re-implement the
diffrax solve here (rather than call integrate_orbit, which hardcodes Dopri8) but
through the SAME backend-agnostic forces (_eom_rhs) and the SAME to/from-EOM
transforms, so only the Butcher tableau differs.

Run in the GPU env (jax 0.10.1 CUDA), backend-migrated galpy on PYTHONPATH, GPU jobs
STRICTLY SERIALLY (one device-timed process at a time):

    PYTHONPATH=/var/tmp/backends-merge python jax_dopri5_measure.py gpu lowN
    PYTHONPATH=/var/tmp/backends-merge python jax_dopri5_measure.py gpu hiN

Writes data/jax_dopri5_<device>[_hiN].json (machine + version stamped); local to g12
(NOT shared -- the shared jax_*.json stay the production Dopri8 curve g8 also uses).

Extra points at larger N (e.g. on a bigger GPU), merged into the existing JSON
(its other points and env stamps kept; the run is noted in its ``extra_points``):

    python jax_dopri5_measure.py gpu hiN --Ns 1e6,3e6
"""

import argparse
import json
import os
import socket
import sys
import time

_ap = argparse.ArgumentParser(description="galpy-jax with diffrax Dopri5 (g12): orbit-integration speed vs N")
_ap.add_argument("device", nargs="?", default="gpu", help="'gpu' (default) or 'cpu'")
_ap.add_argument("mode", nargs="?", default="lowN", help="'lowN' (nt=1000; default) or 'hiN' (nt=16)")
_ap.add_argument("--Ns", metavar="N1,N2,...",
                 help="measure ONLY these N (ints or e-notation, e.g. 1e6,3e6) and MERGE them "
                      "into the existing data/jax_dopri5_<device>[_hiN].json (other points and "
                      "its env stamps kept; this run is recorded in its 'extra_points')")
_args = _ap.parse_args()
device, mode = _args.device, _args.mode
if _args.Ns is not None:  # paper/figures/_harness/extra_points.py
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 4)))
    from paper.figures._harness import extra_points

    try:
        _extra_Ns = extra_points.parse_Ns(_args.Ns)
    except ValueError as e:
        _ap.error(str(e))

if device == "cpu":
    os.environ["JAX_PLATFORMS"] = "cpu"  # force CPU before jax import

import diffrax  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

import galpy  # noqa: E402
from galpy.backend._reference.inbackend_ode import (  # noqa: E402
    _eom_rhs,
    _from_eom,
    _to_eom,
)
from galpy.potential import MWPotential2014  # noqa: E402

assert galpy.__file__.startswith(
    os.environ.get("GALPY_WORKTREE", "/var/tmp/backends-merge")
), galpy.__file__

_DATA = os.path.join(os.path.dirname(__file__), "..", "data")

if mode == "hiN":
    nt, Ns, suffix = 16, [1000, 10000, 100000, 300000], "_hiN"
else:
    nt, suffix = 1000, ""
    Ns = [1, 10, 100, 1000, 10000] if device == "cpu" else [1, 10, 100, 1000, 10000, 30000]
if _args.Ns is not None:
    Ns = _extra_Ns

ts = jnp.linspace(0.0, 56.2, nt)  # 2 Gyr in galpy internal units
ic0 = jnp.array([1.0, 0.0, 1.0, 0.0125, 0.036, 0.0])  # Sun-like [R,vR,vT,z,vz,phi]
scale = jnp.array([0.05, 0.01, 0.0, 0.01, 0.01, 0.0])


def _integrate_dopri5(vxvv):
    """diffrax Dopri5 (5th order) integration through the SAME backend forces as
    the shipped integrate_orbit (which uses Dopri8); only the solver differs."""
    def field(t, y, args):
        return jnp.stack(_eom_rhs(y, MWPotential2014, t, jnp))

    y0 = _to_eom(jnp, vxvv)
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(field),
        diffrax.Dopri5(),
        t0=ts[0],
        t1=ts[-1],
        dt0=None,
        y0=y0,
        saveat=diffrax.SaveAt(ts=ts),
        stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-8),
        max_steps=100000,
    )
    return _from_eom(jnp, sol.ys, 6)


@jax.jit
def batch(ics):
    return jax.vmap(_integrate_dopri5)(ics)


plat = jax.devices()[0].platform
out = {
    "backend": "galpy-jax",
    "device": plat,
    "integrator": "diffrax Dopri5 (in-backend; same-work match to torch dopri5)",
    "stepping": "per-orbit adaptive (vmap-of-solve)",
    "nt": nt,
    "tspan_Gyr": 2.0,
    "rtol": 1e-8,
    "atol": 1e-8,
    "potential": "MWPotential2014",
    "host": socket.gethostname(),
    "results": [],
}
for m in ("jax", "diffrax", "galpy", "numpy"):
    try:
        out[m + "_version"] = __import__(m).__version__
    except Exception:
        pass
try:
    import torch

    out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else plat
except Exception:
    out["gpu"] = plat


def _dump(out):
    if _args.Ns is not None:
        out = extra_points.merge_results(_base, out, _note)
    with open(os.path.join(_DATA, f"jax_dopri5_{plat}{suffix}.json"), "w") as fh:
        json.dump(out, fh, indent=2)


if _args.Ns is not None:
    _base = extra_points.load(os.path.join(_DATA, f"jax_dopri5_{plat}{suffix}.json"), required=False)
    for _k in ("device", "gpu", "nt"):
        extra_points.check_same((_base or {}).get(_k), out[_k], _k, f"jax_dopri5_{plat}{suffix}.json")
    _note = extra_points.note(Ns, {k: v for k, v in out.items() if k != "results"})

for N in Ns:
    ics = ic0[None] + jax.random.normal(jax.random.PRNGKey(N), (N, 6)) * scale
    try:
        t0 = time.perf_counter()
        batch(ics).block_until_ready()  # first call: XLA compile + run
        compile_run_s = time.perf_counter() - t0
        best = min(
            (
                lambda t0: (batch(ics).block_until_ready(), time.perf_counter() - t0)[1]
            )(time.perf_counter())
            for _ in range(3)
        )
        out["results"].append(
            {
                "N": N,
                "wall_s": best,
                "per_orbit_ms": best / N * 1e3,
                "compile_run_s": compile_run_s,
            }
        )
        print(
            f"galpy-jax-Dopri5[{plat},{mode}] N={N:8d}: steady={best * 1e3:9.2f} ms "
            f"({best / N * 1e3:.5f} ms/orbit)  compile+run={compile_run_s:6.2f} s",
            flush=True,
        )
        os.makedirs(_DATA, exist_ok=True)
        _dump(out)
    except Exception as e:  # OOM or step-cap: stop and record where
        print(f"galpy-jax-Dopri5[{plat}] N={N}: STOP {type(e).__name__}: {str(e)[:70]}", flush=True)
        out["stopped_at"] = {"N": N, "reason": type(e).__name__, "msg": str(e)[:120]}
        _dump(out)
        break
print("done")
