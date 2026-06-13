"""g12 measure -- galpy's in-backend (diffrax) orbit integrator on JAX.

Integrates MWPotential2014 for 2 Gyr over a batch of N (slightly perturbed) Sun-
like initial conditions, with galpy's OWN differentiable in-backend integrator
(galpy.backend._reference.integrate_orbit -> diffrax Dopri8), jax.vmap-ed over the
batch and jit-fused. Records, per N: the steady-state batch wall time (compile
excluded, median-of-3 best) AND the first-call compile+run time (the amortization
story). Per-orbit adaptive stepping (vmap-of-solve), rtol=atol=1e-8.

Usage (run in the GPU env with the backend-migrated galpy on PYTHONPATH):

    python jax_measure.py gpu lowN   # device=gpu, 1000 dense saves, N up to OOM cap
    python jax_measure.py gpu hiN    # device=gpu, 16 saves,        high N
    python jax_measure.py cpu lowN   # jax forced onto CPU (baseline)

Writes data/jax_<device>[_hiN].json (machine + version stamped). Mirrors g8's
measure scripts; the plot (../make.py) is byte-stable from the JSON.
"""

import json
import os
import socket
import sys
import time

device = sys.argv[1] if len(sys.argv) > 1 else "gpu"  # 'gpu' or 'cpu'
mode = sys.argv[2] if len(sys.argv) > 2 else "lowN"   # 'lowN' or 'hiN'

if device == "cpu":
    os.environ["JAX_PLATFORMS"] = "cpu"  # force CPU before jax import

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from galpy.backend._reference import integrate_orbit  # noqa: E402
from galpy.potential import MWPotential2014  # noqa: E402

_DATA = os.path.join(os.path.dirname(__file__), "..", "data")

# 2 Gyr in galpy internal units (ro=8, vo=220 -> tspan ~ 56.2 t.u.).
# lowN: 1000 dense output points (resolved orbit, but memory-heavy at large N).
# hiN : 16 output points (integration is adaptive-step-bound, not save-bound, so
#       per-orbit cost is the same; lets the batch fit in 12 GiB at large N).
if mode == "hiN":
    nt, Ns, suffix = 16, [1000, 10000, 100000, 300000, 1000000], "_hiN"
else:
    nt, Ns, suffix = 1000, [1, 10, 100, 1000, 10000, 30000, 100000], ""

ts = jnp.linspace(0.0, 56.2, nt)
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
        with open(os.path.join(_DATA, f"jax_{plat}{suffix}.json"), "w") as fh:
            json.dump(out, fh, indent=2)
    except Exception as e:  # OOM or step-cap: stop and record where
        print(f"galpy-jax[{plat}] N={N}: STOP {type(e).__name__}: {str(e)[:70]}", flush=True)
        break
print("done")
