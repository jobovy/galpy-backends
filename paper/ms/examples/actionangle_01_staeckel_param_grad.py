"""Paper snippet: sections/actionangle.tex, Sec. 6.2 (Staeckel and adiabatic
approximations), #1.

Claim: the vectorized Python Staeckel path (c=False) is differentiable
w.r.t. the potential's parameters; the derivative agrees with a finite
difference in which the potential and actionAngle object are rebuilt.
"""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy

from galpy.actionAngle import actionAngleStaeckel
from galpy.potential import MiyamotoNagaiPotential

R = jnp.array([1.0, 0.9, 1.1])
vR = jnp.array([0.1, -0.05, 0.02])
vT = jnp.array([0.7, 0.75, 0.65])
z = jnp.array([0.05, 0.1, -0.02])
vz = jnp.array([0.05, 0.02, 0.08])

# --- paper snippet begins ---
def jr_of_a(a):
    mp= MiyamotoNagaiPotential(a=a,b=0.3)
    aAS= actionAngleStaeckel(pot=mp,delta=0.4,c=False)
    return aAS(R,vR,vT,z,vz)[0].sum()
djr_da= jax.grad(jr_of_a)(0.5)
# --- paper snippet ends ---

assert isinstance(djr_da, jax.Array)


def jr_of_a_np(a):  # numpy, rebuilt at each a
    aAS = actionAngleStaeckel(
        pot=MiyamotoNagaiPotential(a=a, b=0.3), delta=0.4, c=False
    )
    return numpy.sum(aAS(*(numpy.asarray(x) for x in (R, vR, vT, z, vz)))[0])


numpy.testing.assert_allclose(float(jr_of_a(0.5)), jr_of_a_np(0.5), rtol=1e-12)
h = 1e-3
fd = (
    -jr_of_a_np(0.5 + 2 * h) + 8 * jr_of_a_np(0.5 + h)
    - 8 * jr_of_a_np(0.5 - h) + jr_of_a_np(0.5 - 2 * h)
) / (12 * h)
print(float(djr_da), fd)
numpy.testing.assert_allclose(float(djr_da), fd, rtol=1e-9)
print("OK")
