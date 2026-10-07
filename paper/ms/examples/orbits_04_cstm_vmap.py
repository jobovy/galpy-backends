"""Paper snippet: sections/orbits.tex, Sec. 5.1 (Three ways to integrate an
orbit), #4.

Claim: a batch of orbits is differentiated with jax.vmap over the C-STM
jacrev of the previous snippet.
"""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy

from galpy.orbit import Orbit
from galpy.potential import MWPotential2014

ts = numpy.linspace(0.0, 10.0, 101)


def end_point(ic):
    o = Orbit(ic)
    o.integrate(ts, MWPotential2014, method="dop853_c")  # C + variational
    return jnp.stack([o.R(ts[-1]), o.z(ts[-1])])


ics = jnp.array(
    [
        [1.0, 0.1, 1.1, 0.1, 0.05, 0.0],
        [0.9, -0.1, 1.0, 0.0, 0.1, 0.5],
        [1.2, 0.2, 0.9, -0.1, 0.0, 1.0],
    ]
)
N = ics.shape[0]

# --- paper snippet begins ---
Ms= jax.vmap(jax.jacrev(end_point))(ics) # ics: (N,6)
# --- paper snippet ends ---

assert isinstance(Ms, jax.Array) and Ms.shape == (N, 2, 6)
for i in range(N):
    Mi = jax.jacrev(end_point)(ics[i])
    numpy.testing.assert_allclose(numpy.asarray(Ms[i]), numpy.asarray(Mi),
                                  rtol=1e-12, atol=1e-12)


def end_point_c(x):
    # finite difference of independently integrated, tightly converged orbits
    o = Orbit(list(x))
    o.integrate(ts, MWPotential2014, method="dop853_c", rtol=1e-13, atol=1e-13)
    return numpy.array([o.R(ts[-1]), o.z(ts[-1])])


h = 1e-4
for i in range(N):
    x0 = numpy.asarray(ics[i])
    fd = numpy.empty((2, 6))
    for j in range(6):
        e = numpy.zeros(6)
        e[j] = h
        fd[:, j] = (
            -end_point_c(x0 + 2 * e) + 8 * end_point_c(x0 + e)
            - 8 * end_point_c(x0 - e) + end_point_c(x0 - 2 * e)
        ) / (12 * h)
    rel = numpy.linalg.norm(numpy.asarray(Ms[i]) - fd) / numpy.linalg.norm(fd)
    print(i, rel)
    assert rel < 3e-9, rel
print("OK")
