"""Verify Orbit.x(t)/R(t)/... are jittable after the _call_internal fix, both
at off-grid AND exact-grid query times, with theta-grad flowing."""
import sys; sys.path.insert(0, "/var/tmp/cint-param")
import numpy, galpy
assert galpy.__file__.startswith("/var/tmp/cint-param")
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from galpy.orbit import Orbit
from galpy.potential import LogarithmicHaloPotential
from galpy.backend import as_numpy

IC = jnp.asarray([1.2, 0.15, 0.85, 0.08, 0.05, 0.0])
grid = jnp.linspace(0.0, -2.0, 501)
toff = jnp.asarray([-0.313, -0.777, -1.234])   # off-grid


def query(amp, tq):
    pot = LogarithmicHaloPotential(amp=amp, q=0.9)
    o = Orbit(IC); o.turn_physical_off()
    o.integrate(grid, pot, method="diffrax")
    return jnp.stack([o.x(tq), o.y(tq), o.z(tq), o.R(tq), o.vT(tq)], axis=-1)

# reference (numpy) for value check
on = Orbit(numpy.asarray(IC)); on.turn_physical_off()
on.integrate(numpy.linspace(0,-2,501), LogarithmicHaloPotential(amp=1.0, q=0.9))
ref = numpy.column_stack([on.x(numpy.asarray(toff)), on.y(numpy.asarray(toff)),
                          on.z(numpy.asarray(toff)), on.R(numpy.asarray(toff)), on.vT(numpy.asarray(toff))])

print("=== o.x(t) etc. under jax.jit ===")
for label, tq in [("off-grid", toff), ("exact-grid subset", grid[::100])]:
    try:
        f = jax.jit(lambda a: query(a, tq))
        v = as_numpy(f(jnp.asarray(1.0)))
        print(f"  [{label}] jit OK, shape {v.shape}")
        if label == "off-grid":
            print(f"    max|jit - numpy| = {numpy.max(numpy.abs(v - ref)):.2e}")
    except Exception as e:
        print(f"  [{label}] FAILED: {type(e).__name__}: {str(e)[:120]}")

# theta-grad through o.x(t) under jit
try:
    g = jax.jit(jax.grad(lambda a: query(a, toff).sum()))(jnp.asarray(1.0))
    print(f"  jit grad d(sum o.x/y/z/R/vT)/d(amp) = {float(g):.6e}  OK")
except Exception as e:
    print(f"  jit grad FAILED: {type(e).__name__}: {str(e)[:120]}")
