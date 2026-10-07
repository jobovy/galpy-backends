"""Canonical shared measure -- galpy's in-backend (diffrax) orbit integrator on JAX.

The SINGLE source of truth for the galpy-jax (CPU and GPU) orbit-integration curves
that BOTH g8 and g12 load, so the two figures show byte-identical jax lines.

Integrates MWPotential2014 for 2 Gyr over a batch of N (per-N-seeded, slightly
perturbed) Sun-like ICs with galpy's OWN differentiable in-backend integrator
(galpy.backend._reference.integrate_orbit -> diffrax Dopri8), jax.vmap-ed over the
batch and jit-fused.  Per-orbit adaptive stepping (vmap-of-solve), rtol=atol=1e-8.
Records, per N, the steady-state batch wall time (compile excluded, median-of-3
best) and the first-call compile+run time (compile_run_s, the amortization story).

nt convention (shared with the C and torch measures, IDENTICAL in g8 and g12):
  * lowN regime -> nt=1000 dense saves  (resolved orbit; small-N crossover);
  * hiN  regime -> nt=16   sparse saves (fits the batch in 12 GiB at large N).
Integration is adaptive-step-bound, not save-bound, so per-orbit cost is the same
in both regimes and the overlap points agree.

Run in the GPU env (jax 0.10.1 CUDA) with the backend-migrated galpy on PYTHONPATH.
Run GPU jobs STRICTLY SERIALLY (one device-timed process at a time):

    PYTHONPATH=/var/tmp/backends-merge python galpy_jax_measure.py gpu lowN
    PYTHONPATH=/var/tmp/backends-merge python galpy_jax_measure.py gpu hiN
    PYTHONPATH=/var/tmp/backends-merge python galpy_jax_measure.py cpu lowN
    PYTHONPATH=/var/tmp/backends-merge python galpy_jax_measure.py cpu hiN

Writes data/jax_<device>[_hiN].json (machine + version stamped).  The plots are
byte-stable from these JSONs.

Extra points at larger N (e.g. on a bigger GPU), merged into the existing JSON
(its other points and env stamps kept; the run is noted in its ``extra_points``):

    python galpy_jax_measure.py gpu hiN --Ns 1e6,3e6
"""

import argparse
import json
import os
import socket
import sys
import time

_ap = argparse.ArgumentParser(description="galpy-jax (diffrax Dopri8) orbit-integration speed vs N")
_ap.add_argument("device", nargs="?", default="gpu", help="'gpu' (default) or 'cpu'")
_ap.add_argument("mode", nargs="?", default="lowN", help="'lowN' (nt=1000; default) or 'hiN' (nt=16)")
_ap.add_argument("--Ns", metavar="N1,N2,...",
                 help="measure ONLY these N (ints or e-notation, e.g. 1e6,3e6) and MERGE them "
                      "into the existing data/jax_<device>[_hiN].json (other points and its env "
                      "stamps kept; this run is recorded in its 'extra_points')")
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

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import galpy  # noqa: E402
from galpy.backend._reference import integrate_orbit  # noqa: E402
from galpy.potential import MWPotential2014  # noqa: E402

assert galpy.__file__.startswith(
    os.environ.get("GALPY_WORKTREE", "/var/tmp/backends-merge")
), galpy.__file__

_DATA = os.path.join(os.path.dirname(__file__), "data")

if mode == "hiN":
    nt, Ns, suffix = 16, [1000, 10000, 100000, 300000], "_hiN"
else:
    # lowN dense (nt=1000). On the CPU the N=3e4 dense steady run is impractically
    # slow (>10 min), and the CPU line only needs the small-N device reference, so
    # cap CPU at 1e4; the GPU run takes the full lowN sweep to 3e4.
    nt, suffix = 1000, ""
    Ns = [1, 10, 100, 1000, 10000] if device == "cpu" else [1, 10, 100, 1000, 10000, 30000]
if _args.Ns is not None:
    Ns = _extra_Ns

ts = jnp.linspace(0.0, 56.2, nt)  # 2 Gyr in galpy internal units
ic0 = jnp.array([1.0, 0.0, 1.0, 0.0125, 0.036, 0.0])  # Sun-like [R,vR,vT,z,vz,phi]
scale = jnp.array([0.05, 0.01, 0.0, 0.01, 0.01, 0.0])


@jax.jit
def batch(ics):
    return jax.vmap(
        lambda v: integrate_orbit(MWPotential2014, v, ts, rtol=1e-8, atol=1e-8)
    )(ics)


plat = jax.devices()[0].platform
out = {
    "backend": "galpy-jax",
    "device": plat,
    "integrator": "diffrax Dopri8 (in-backend)",
    "stepping": "per-orbit adaptive (vmap-of-solve)",
    "nt": nt,
    "tspan_Gyr": 2.0,
    "rtol": 1e-8,
    "atol": 1e-8,
    "potential": "MWPotential2014",
    "host": socket.gethostname(),
    "results": [],
}
for m in ("jax", "galpy", "numpy"):
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
    with open(os.path.join(_DATA, f"jax_{plat}{suffix}.json"), "w") as fh:
        json.dump(out, fh, indent=2)


if _args.Ns is not None:
    _base = extra_points.load(os.path.join(_DATA, f"jax_{plat}{suffix}.json"), required=False)
    for _k in ("device", "gpu", "nt"):
        extra_points.check_same((_base or {}).get(_k), out[_k], _k, f"jax_{plat}{suffix}.json")
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
            f"galpy-jax[{plat},{mode}] N={N:8d}: steady={best * 1e3:9.2f} ms "
            f"({best / N * 1e3:.5f} ms/orbit)  compile+run={compile_run_s:6.2f} s",
            flush=True,
        )
        os.makedirs(_DATA, exist_ok=True)
        _dump(out)
    except Exception as e:  # OOM or step-cap: stop and record where
        print(f"galpy-jax[{plat}] N={N}: STOP {type(e).__name__}: {str(e)[:70]}", flush=True)
        if _args.Ns is not None:
            out["stopped_at"] = {"N": N, "reason": type(e).__name__, "msg": str(e)[:120]}
            os.makedirs(_DATA, exist_ok=True)
            _dump(out)
        break
print("done")
