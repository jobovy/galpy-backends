"""Paper snippet: sections/examples.tex, Sec. 10.1 (A hypervelocity star as a
probe of the potential), #1.

Claim: the derivative of the closest approach of the backward orbit w.r.t.
the star's observed coordinates (ra,dec,d,pmra,pmdec,vlos) comes from the C
variational integrator (C-STM) through an Orbit initialized from observed
coordinates.
"""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy

from galpy.orbit import Orbit
from galpy.potential import LogarithmicHaloPotential

# count the calls into the C state-transition-matrix path
_ncstm = [0]
_orig_cstm = Orbit._integrate_cstm


def _counting_cstm(self, *args, **kwargs):
    _ncstm[0] += 1
    return _orig_cstm(self, *args, **kwargs)


Orbit._integrate_cstm = _counting_cstm

pot = LogarithmicHaloPotential(normalize=1.0, core=0.2, q=0.85)
ts = numpy.linspace(0.0, 0.5, 151)  # ~18 Myr
# an S5-HVS1-like mock: launched from near the Galactic center, observed today
launch = Orbit([0.004, 2.4, 0.03, 0.002, 1.5, 0.6], ro=8.0, vo=220.0)
launch.integrate(ts, pot, method="dop853_c", rtol=1e-13, atol=1e-13)
vxvv_obs = [
    float(getattr(launch, f)(ts[-1]))
    for f in ("ra", "dec", "dist", "pmra", "pmdec", "vlos")
]

# --- paper snippet begins ---
def closest_approach_ic(vxvv):
    # vxvv: (ra,dec,d,pmra,pmdec,vlos)
    o= Orbit(vxvv,radec=True,ro=8.,vo=220.)
    o.integrate(-ts,pot,method='dop853_c') # C + STM
    return jnp.min(o.r(-ts))
J= jax.jacrev(closest_approach_ic)(
    jnp.asarray(vxvv_obs))
# --- paper snippet ends ---

assert isinstance(J, jax.Array) and J.shape == (6,)
assert _ncstm[0] >= 1, "the C variational path was not used"
Orbit._integrate_cstm = _orig_cstm


def closest_np(x):
    o = Orbit(list(x), radec=True, ro=8.0, vo=220.0)
    o.integrate(-ts, pot, method="dop853_c", rtol=1e-13, atol=1e-13)
    return numpy.min(o.r(-ts))


x0 = numpy.array(vxvv_obs)
print("closest approach [kpc]:", float(closest_approach_ic(jnp.asarray(x0))))
# the backward orbit returns close to the center
assert float(closest_approach_ic(jnp.asarray(x0))) < 0.1
steps = numpy.array([1e-5, 1e-5, 1e-4, 1e-4, 1e-4, 1e-2])
fd = numpy.empty(6)
for j in range(6):
    e = numpy.zeros(6)
    e[j] = steps[j]
    fd[j] = (
        -closest_np(x0 + 2 * e) + 8 * closest_np(x0 + e)
        - 8 * closest_np(x0 - e) + closest_np(x0 - 2 * e)
    ) / (12 * steps[j])
print(numpy.asarray(J))
print(fd)
numpy.testing.assert_allclose(numpy.asarray(J), fd, rtol=1e-6)
print("OK")
