"""Paper snippet: sections/orbits.tex, Sec. 4.2 (An in-backend leapfrog).

Claim: a batch of orbits is integrated inside jax by the fixed-step leapfrog
with a chosen number of steps; the result is a jax array that agrees with the
C integrator to the method's truncation error (second order: 8000 steps over
this time span give 3e-4 in position, 16000 steps a quarter of that).
"""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy

from galpy.orbit import Orbit
from galpy.potential import MWPotential2014

N = 4
rng = numpy.random.default_rng(3)
ics = jnp.asarray(
    numpy.array([1.0, 0.1, 1.0, 0.05, 0.03, 0.0]) + rng.normal(size=(N, 6)) * [0.05, 0.02, 0.02, 0.02, 0.02, 0.0]
)
ts = jnp.linspace(0.0, 56.2, 101)  # 2 Gyr

# --- paper snippet begins ---
o= Orbit(jnp.asarray(ics)) # ics: (N,6)
o.integrate(ts,MWPotential2014,method='leapfrog_jax',
            inbackend_kwargs={'nsteps':8000})
# --- paper snippet ends ---

R = o.R(ts)
assert isinstance(R, jax.Array) and R.shape == (N, len(ts))
ref = Orbit(numpy.asarray(ics))
ref.integrate(numpy.asarray(ts), MWPotential2014, method="dop853_c", rtol=1e-12, atol=1e-12)
for q in ("x", "y", "z"):
    numpy.testing.assert_allclose(numpy.asarray(getattr(o, q)(ts)), getattr(ref, q)(numpy.asarray(ts)), atol=5e-4)
# halving the step divides the error by four (second order)
o2 = Orbit(jnp.asarray(ics))
o2.integrate(ts, MWPotential2014, method="leapfrog_jax", inbackend_kwargs={"nsteps": 16000})
err = lambda ob: numpy.max(numpy.abs(numpy.asarray(ob.x(ts)) - ref.x(numpy.asarray(ts))))
ratio = err(o) / err(o2)
assert 3.5 < ratio < 4.5, ratio
print("OK")
