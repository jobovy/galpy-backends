"""g12 -- clean re-measure of the single jax-GPU hiN N=1e5 point.

The original hiN N=1e5 point was timed while a torch-GPU job was co-resident on the
TITAN Xp, inflating it (0.336 ms/orbit vs the uncontended ~0.21-0.28 trend). This
re-runs ONLY N=1e5 with nothing else on the GPU and patches data/jax_gpu_hiN.json
in place (other points untouched). Same config as jax_measure.py hiN.
"""

import json
import os
import time

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from galpy.backend._reference import integrate_orbit  # noqa: E402
from galpy.potential import MWPotential2014  # noqa: E402

_JSON = os.path.join(os.path.dirname(__file__), "..", "data", "jax_gpu_hiN.json")
N = 100000
nt = 16
ts = jnp.linspace(0.0, 56.2, nt)
ic0 = jnp.array([1.0, 0.0, 1.0, 0.0125, 0.036, 0.0])
scale = jnp.array([0.05, 0.01, 0.0, 0.01, 0.01, 0.0])


@jax.jit
def batch(ics):
    return jax.vmap(
        lambda v: integrate_orbit(MWPotential2014, v, ts, rtol=1e-8, atol=1e-8)
    )(ics)


ics = ic0[None] + jax.random.normal(jax.random.PRNGKey(N), (N, 6)) * scale
t0 = time.perf_counter()
batch(ics).block_until_ready()
compile_run_s = time.perf_counter() - t0
best = min(
    (lambda t0: (batch(ics).block_until_ready(), time.perf_counter() - t0)[1])(
        time.perf_counter()
    )
    for _ in range(3)
)
print(f"jax-GPU CLEAN N={N}: steady={best * 1e3:.1f} ms ({best / N * 1e3:.5f} ms/orbit) "
      f"compile+run={compile_run_s:.2f} s", flush=True)

with open(_JSON) as fh:
    d = json.load(fh)
for r in d["results"]:
    if r["N"] == N:
        r["wall_s"] = best
        r["per_orbit_ms"] = best / N * 1e3
        r["compile_run_s"] = compile_run_s
        r["note"] = "re-measured uncontended (GPU-exclusive)"
with open(_JSON, "w") as fh:
    json.dump(d, fh, indent=2)
print("patched jax_gpu_hiN.json N=1e5")
