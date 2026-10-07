"""Paper listing: sections/potentials.tex, Figure (fig:newpot), the codefig.

Claims: a Potential subclass written with the resolved namespace runs on
every backend and is differentiable w.r.t. its parameters; the last lines
take d z(t_end) / d a of an orbit integrated in it with diffrax.
"""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy
import torch

from galpy.orbit import Orbit
from galpy.potential import MiyamotoNagaiPotential

ts = numpy.linspace(0.0, 5.0, 51)

# --- paper snippet begins ---
from galpy.potential import Potential
from galpy.backend import get_namespace
class MNLikePotential(Potential):
    # a Miyamoto-Nagai disk written for every backend
    def __init__(self,amp=1.,a=1.,b=0.1,
                 normalize=False):
        Potential.__init__(self,amp=amp)
        self._a= a
        self._b2= b**2
        if normalize: self.normalize(normalize)
    def _evaluate(self,R,z,phi=0.,t=0.):
        xp= get_namespace(R,z)
        s= self._a+xp.sqrt(z**2+self._b2)
        return -1./xp.sqrt(R**2+s**2)
    def _Rforce(self,R,z,phi=0.,t=0.):
        xp= get_namespace(R,z)
        s= self._a+xp.sqrt(z**2+self._b2)
        return -R*(R**2+s**2)**-1.5
    def _zforce(self,R,z,phi=0.,t=0.):
        xp= get_namespace(R,z)
        sqb= xp.sqrt(z**2+self._b2)
        s= self._a+sqb
        return -z*s/sqb*(R**2+s**2)**-1.5
# differentiate an orbit in it w.r.t. the scale length a
def final_z(a):
    o= Orbit(jnp.asarray([1.,0.1,1.1,0.1,0.05,0.]))
    mp= MNLikePotential(a=a,b=0.3,normalize=1.)
    o.integrate(ts,mp,method='diffrax')
    return o.z(ts[-1])
jax.grad(final_z)(0.5)
# --- paper snippet ends ---

# the class is a Miyamoto-Nagai disk, on every backend
mnl = MNLikePotential(a=0.5, b=0.3, normalize=1.0)
mn = MiyamotoNagaiPotential(a=0.5, b=0.3, normalize=1.0)
Rs, zs = numpy.array([0.5, 1.0, 2.0]), numpy.array([0.1, -0.3, 1.0])
for f in ("__call__", "Rforce", "zforce"):
    expect = getattr(mn, f)(Rs, zs)
    numpy.testing.assert_allclose(getattr(mnl, f)(Rs, zs), expect, rtol=1e-14)
    out = getattr(mnl, f)(jnp.asarray(Rs), jnp.asarray(zs))
    assert isinstance(out, jax.Array)
    numpy.testing.assert_allclose(numpy.asarray(out), expect, rtol=1e-14)
    out = getattr(mnl, f)(torch.as_tensor(Rs), torch.as_tensor(zs))
    assert isinstance(out, torch.Tensor)
    numpy.testing.assert_allclose(out.numpy(), expect, rtol=1e-14)

# the derivative of the final height matches a central finite difference
g = jax.grad(final_z)(0.5)
assert isinstance(g, jax.Array)


def final_z_c(a):  # the same orbit with galpy's own MN potential, in C
    o = Orbit([1.0, 0.1, 1.1, 0.1, 0.05, 0.0])
    o.integrate(ts, MiyamotoNagaiPotential(a=a, b=0.3, normalize=1.0),
                method="dop853_c")
    return o.z(ts[-1])


numpy.testing.assert_allclose(float(final_z(0.5)), final_z_c(0.5), rtol=1e-9)
h = 1e-3
fd = (
    -final_z_c(0.5 + 2 * h) + 8 * final_z_c(0.5 + h)
    - 8 * final_z_c(0.5 - h) + final_z_c(0.5 - 2 * h)
) / (12 * h)
print(float(g), fd)
numpy.testing.assert_allclose(float(g), fd, rtol=1e-7)
print("OK")
