"""Extend the C-STM scaling line to N=1000, 3000 (it is linear, so just slow)."""
import json, pathlib, numpy, jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from paper.figures._harness.bench import time_call
from galpy.backend._jax.orbit_stm import integrate as cstm_jax
from galpy.potential import MWPotential2014
_IC = numpy.array([1.0, 0.1, 0.9, 0.05, 0.1, 0.2]); _TS = numpy.linspace(0., 10., 100)
ts_j = jnp.asarray(_TS); CPU = jax.devices("cpu")[0]
_F = pathlib.Path(__file__).resolve().parent.parent / "data" / "icderiv.json"
out = json.loads(_F.read_text())
def batch(n): return numpy.stack([_IC * (1 + 0.001 * (k % 97)) for k in range(n)])
for n in (1000, 3000):
    with jax.default_device(CPU):
        g = jax.jit(jax.grad(lambda v: cstm_jax(MWPotential2014, v, ts_j, method="dop853_c")[:, -1, 0].sum()))
        icb = jnp.asarray(batch(n))
        t = time_call(lambda: jax.block_until_ready(g(icb)), sync="jax", repeats=2, warmup=1)["median"]
    out["scaling"].setdefault(str(n), {})["cstm_cpu"] = t
    _F.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"cstm_cpu N={n}: {t*1e3:.0f} ms", flush=True)
print("done", flush=True)
