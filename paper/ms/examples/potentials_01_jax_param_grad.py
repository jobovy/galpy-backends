"""Paper snippet: sections/potentials.tex, Sec. 3 (Potentials), #1.

Claim: a potential constructed with a (traced) jax parameter is
differentiable with respect to that parameter through its methods.
"""

import jax

jax.config.update("jax_enable_x64", True)
import numpy

# --- paper snippet begins ---
from galpy.potential import MiyamotoNagaiPotential
def Rforce_of_a(a):
    mp= MiyamotoNagaiPotential(amp=1.,a=a,b=0.3)
    return mp.Rforce(1.2,0.1)
dFR_da= jax.grad(Rforce_of_a)(0.5)
# --- paper snippet ends ---

assert isinstance(dFR_da, jax.Array)
# analytic: F_R = -R (R^2+(a+sqrt(z^2+b^2))^2)^-3/2
R, z, a, b = 1.2, 0.1, 0.5, 0.3
s = a + numpy.sqrt(z**2 + b**2)
exact = 3.0 * R * s * (R**2 + s**2) ** -2.5
numpy.testing.assert_allclose(float(dFR_da), exact, rtol=1e-14)
# and a central finite difference of the numpy path
h = 1e-5
fd = (
    MiyamotoNagaiPotential(amp=1.0, a=a + h, b=b).Rforce(R, z)
    - MiyamotoNagaiPotential(amp=1.0, a=a - h, b=b).Rforce(R, z)
) / (2 * h)
numpy.testing.assert_allclose(float(dFR_da), fd, rtol=1e-9)
print("OK")
