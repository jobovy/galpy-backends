"""Scale probe: n=600 orbits over span ~126 units (20 orbital periods = realistic
tdisrupt), jax.jit, timing compile vs cached run, on the default device."""
import sys, time; sys.path.insert(0, "/var/tmp/cint-param")
import numpy, galpy
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from galpy.orbit import Orbit
from galpy.potential import LogarithmicHaloPotential
from galpy.util import conversion
print("device:", jax.devices()[0], flush=True)
IC = [1.2, 0.15, 0.85, 0.08, 0.05, 0.0]
TD = 4.5 / conversion.time_in_Gyr(220.0, 8.0)
print("TD (span) =", TD, flush=True)
n = 600
rng = numpy.random.default_rng(1)
ics = jnp.asarray(numpy.tile(IC, (n, 1))) + 0.01 * jnp.asarray(rng.normal(size=(n, 6)))
dt = jnp.asarray(numpy.sort(rng.uniform(size=n)) * TD)


def final_sum(amp):
    pot = LogarithmicHaloPotential(amp=amp, q=0.9)
    o = Orbit(ics); o.turn_physical_off()
    ts = jnp.stack([-dt, jnp.zeros(n)], axis=-1)
    o.integrate(ts, pot, method="diffrax")
    return o.orbit[:, -1, :].sum()


f = jax.jit(jax.value_and_grad(final_sum))
t0 = time.time(); v, g = f(jnp.asarray(1.0)); v.block_until_ready(); g.block_until_ready()
print(f"jit compile+run (value+grad): v={float(v):.4f} g={float(g):.4e}  ({time.time()-t0:.1f}s)", flush=True)
t0 = time.time()
for a in [1.02, 1.04, 1.06, 1.08, 1.10]:
    v, g = f(jnp.asarray(a)); v.block_until_ready(); g.block_until_ready()
print(f"5 cached value+grad steps: {time.time()-t0:.2f}s  ({(time.time()-t0)/5*1000:.0f} ms/step)", flush=True)
