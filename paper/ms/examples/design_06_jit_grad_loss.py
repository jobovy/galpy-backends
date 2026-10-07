"""Paper snippet: sections/design.tex, Sec. 2.5 (Compilation and
differentiation from the outside).

Claim: the user's loss, which contains a galpy orbit integration, is compiled
and differentiated by jax.jit(jax.grad(.)) without galpy compiling anything.
"""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy

from galpy.orbit import Orbit
from galpy.potential import LogarithmicHaloPotential

ic = [1.0, 0.1, 1.1, 0.1, 0.05, 0.0]
ts = numpy.linspace(0.0, 5.0, 51)
o_obs = Orbit(ic)
o_obs.integrate(ts, LogarithmicHaloPotential(normalize=1.0, q=0.85),
                method="dop853_c")
x_obs = o_obs.x(ts)

# --- paper snippet begins ---
def loss(q):
    pot= LogarithmicHaloPotential(normalize=1.,q=q)
    o= Orbit(jnp.asarray(ic))
    o.integrate(ts,pot,method='diffrax')
    return jnp.sum((o.x(ts)-x_obs)**2)
grad_loss= jax.jit(jax.grad(loss))
# --- paper snippet ends ---

q0 = 0.9
g = grad_loss(q0)
assert isinstance(g, jax.Array)
h = 1e-3  # five-point central difference
fd = (
    -loss(q0 + 2 * h) + 8 * loss(q0 + h) - 8 * loss(q0 - h) + loss(q0 - 2 * h)
) / (12 * h)
print(float(g), float(fd))
numpy.testing.assert_allclose(float(g), float(fd), rtol=1e-7)
# the loss is minimized at the true q
assert abs(float(grad_loss(0.85))) < 1e-8
print("OK")
