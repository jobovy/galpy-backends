"""Paper snippet: sections/orbits.tex, Sec. 5.1 (Three ways to integrate an
orbit), #1.

Claim: an orbit integrated in the backend with diffrax is differentiable
w.r.t. the potential's parameters.
"""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy

from galpy.orbit import Orbit
from galpy.potential import PlummerPotential

ic = [1.0, 0.1, 1.1, 0.1, 0.05, 0.0]
ts = numpy.linspace(0.0, 5.0, 51)

# --- paper snippet begins ---
def final_R(b):
    o= Orbit(jnp.asarray(ic))
    pp= PlummerPotential(amp=1.,b=b)
    o.integrate(ts,pp,method='diffrax')
    return o.R(ts[-1])
dR_db= jax.grad(final_R)(0.6)
# --- paper snippet ends ---

assert isinstance(dR_db, jax.Array)


def final_R_c(b):
    o = Orbit(ic)
    o.integrate(ts, PlummerPotential(amp=1.0, b=b), method="dop853_c")
    return o.R(ts[-1])


numpy.testing.assert_allclose(float(final_R(0.6)), final_R_c(0.6), rtol=1e-9)
h = 1e-3
fd = (
    -final_R_c(0.6 + 2 * h) + 8 * final_R_c(0.6 + h)
    - 8 * final_R_c(0.6 - h) + final_R_c(0.6 - 2 * h)
) / (12 * h)
print(float(dR_db), fd)
numpy.testing.assert_allclose(float(dR_db), fd, rtol=1e-7)
print("OK")
