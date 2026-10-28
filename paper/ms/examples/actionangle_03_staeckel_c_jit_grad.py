"""Paper snippet: sections/actionangle.tex, Sec. 6.2 (Staeckel and adiabatic
approximations), #2.

Claim: jax.grad through a c=True Staeckel call uses the Jacobian computed in
C, attached as a custom derivative rule, and runs under jax.jit.
"""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy

from galpy.actionAngle import actionAngleStaeckel
from galpy.potential import MWPotential2014

vR = jnp.array([0.1, -0.05])
vT = jnp.array([1.0, 1.05])
z = jnp.array([0.05, 0.1])
vz = jnp.array([0.05, 0.02])

# --- paper snippet begins ---
aAS= actionAngleStaeckel(pot=MWPotential2014,
                         delta=0.45,c=True)
def total_jr(R):
    return aAS(R,vR,vT,z,vz)[0].sum()
dJ= jax.jit(jax.grad(total_jr))
djr_dR= dJ(jnp.asarray([1.1,0.9]))
# --- paper snippet ends ---

assert isinstance(djr_dR, jax.Array) and djr_dR.shape == (2,)
# the compiled derivative calls out to C (a callback), it does not trace the
# Python action integrals
jaxpr = str(jax.make_jaxpr(jax.grad(total_jr))(jnp.asarray([1.1, 0.9])))
assert "callback" in jaxpr, "c=True derivative did not go through C"
# the same as the eager derivative and as the c=False (AD) derivative
numpy.testing.assert_allclose(
    numpy.asarray(djr_dR),
    numpy.asarray(jax.grad(total_jr)(jnp.asarray([1.1, 0.9]))),
    rtol=1e-12,
)
aASpy = actionAngleStaeckel(pot=MWPotential2014, delta=0.45, c=False)
dpy = jax.grad(lambda R: aASpy(R, vR, vT, z, vz)[0].sum())(jnp.asarray([1.1, 0.9]))
print(numpy.asarray(djr_dR), numpy.asarray(dpy))
numpy.testing.assert_allclose(numpy.asarray(djr_dR), numpy.asarray(dpy), rtol=1e-8)
# and a central finite difference of the C actions
aASn = actionAngleStaeckel(pot=MWPotential2014, delta=0.45, c=True)
h = 1e-5
args = [numpy.asarray(x) for x in (vR, vT, z, vz)]
R0 = numpy.array([1.1, 0.9])
fd = (
    aASn(R0 + h, args[0], args[1], args[2], args[3])[0]
    - aASn(R0 - h, args[0], args[1], args[2], args[3])[0]
) / (2 * h)
print(fd)
numpy.testing.assert_allclose(numpy.asarray(djr_dR), fd, rtol=1e-7)
print("OK")
