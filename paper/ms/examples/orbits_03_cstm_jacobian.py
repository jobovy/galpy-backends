"""Paper snippet: sections/orbits.tex, Sec. 5.1 (Three ways to integrate an
orbit), #3.

Claim: with a C Runge-Kutta method, jax.jacrev of a function of the orbit
w.r.t. the IC uses the C variational equations (state-transition matrix);
the result equals a finite difference of the C orbit.
"""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy

from galpy.orbit import Orbit
from galpy.potential import MWPotential2014

# count the calls into the C state-transition-matrix path
_ncstm = [0]
_orig_cstm = Orbit._integrate_cstm


def _counting_cstm(self, *args, **kwargs):
    _ncstm[0] += 1
    return _orig_cstm(self, *args, **kwargs)


Orbit._integrate_cstm = _counting_cstm

ic = [1.0, 0.1, 1.1, 0.1, 0.05, 0.0]
ts = numpy.linspace(0.0, 10.0, 101)

# --- paper snippet begins ---
def end_point(ic):
    o= Orbit(ic)
    o.integrate(ts,MWPotential2014,
                method='dop853_c') # C + variational
    return jnp.stack([o.R(ts[-1]),o.z(ts[-1])])
M= jax.jacrev(end_point)(jnp.asarray(ic)) # STM rows
# --- paper snippet ends ---

assert isinstance(M, jax.Array) and M.shape == (2, 6)
assert _ncstm[0] == 1, "the C variational path was not used"
Orbit._integrate_cstm = _orig_cstm


def end_point_c(x):
    # finite difference of independently integrated, tightly converged orbits
    o = Orbit(list(x))
    o.integrate(ts, MWPotential2014, method="dop853_c", rtol=1e-13, atol=1e-13)
    return numpy.array([o.R(ts[-1]), o.z(ts[-1])])


ic0 = numpy.array(ic)
h = 1e-4
fd = numpy.empty((2, 6))
for j in range(6):
    e = numpy.zeros(6)
    e[j] = h
    fd[:, j] = (
        -end_point_c(ic0 + 2 * e) + 8 * end_point_c(ic0 + e)
        - 8 * end_point_c(ic0 - e) + end_point_c(ic0 - 2 * e)
    ) / (12 * h)
print(numpy.asarray(M))
print(fd)
rel = numpy.linalg.norm(numpy.asarray(M) - fd) / numpy.linalg.norm(fd)
print("relative Frobenius difference:", rel)
assert rel < 3e-9, rel
print("OK")
