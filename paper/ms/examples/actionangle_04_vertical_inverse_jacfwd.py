"""Paper snippet: sections/actionangle.tex, Sec. 6.3 (Spherical potentials,
the isochrone approximation, and the inverse transformation).

Claim: the tori of actionAngleVerticalInverse for a family of isothermal
disks are differentiable w.r.t. the disk's velocity dispersion (through the
construction of the torus map); the derivative agrees with a finite
difference in which the family is rebuilt.
"""

import time

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy

from galpy.actionAngle import actionAngleVerticalInverse
from galpy.potential import IsothermalDiskPotential

Es = numpy.linspace(0.0, 1.0, 17)  # energies of the torus family
J = 0.06
angles = jnp.linspace(0.0, 2.0 * numpy.pi, 16, endpoint=False)

t0 = time.time()
# --- paper snippet begins ---
def torus(sigma):
    disk= IsothermalDiskPotential(amp=1.,sigma=sigma)
    aAVI= actionAngleVerticalInverse(
        pot=disk,Es=Es,nta=128,setup_interp=True)
    return aAVI.xvFreqs(J,angles)[:2] # (z,v_z)
z,vz= torus(0.5)
dz_dsigma,dvz_dsigma= jax.jacfwd(torus)(0.5)
# --- paper snippet ends ---
print(f"snippet: {time.time() - t0:.1f} s")

for x in (z, vz, dz_dsigma, dvz_dsigma):
    assert isinstance(x, jax.Array) and x.shape == angles.shape


# the torus is a torus: the action computed forward from (z,vz) is J
from galpy.actionAngle import actionAngleVertical

aAV = actionAngleVertical(pot=IsothermalDiskPotential(amp=1.0, sigma=0.5))
numpy.testing.assert_allclose(
    numpy.asarray(aAV(numpy.asarray(z), numpy.asarray(vz))), J, rtol=1e-7
)


# five-point finite difference, rebuilding the whole family on numpy
def torus_np(sigma):
    disk = IsothermalDiskPotential(amp=1.0, sigma=sigma)
    aA = actionAngleVerticalInverse(pot=disk, Es=Es, nta=128, setup_interp=True)
    return numpy.array(aA.xvFreqs(J, numpy.asarray(angles))[:2])


numpy.testing.assert_allclose(numpy.array([z, vz]), torus_np(0.5), atol=1e-10)
h = 1e-3
fd = (
    -torus_np(0.5 + 2 * h) + 8 * torus_np(0.5 + h)
    - 8 * torus_np(0.5 - h) + torus_np(0.5 - 2 * h)
) / (12 * h)
ad = numpy.array([dz_dsigma, dvz_dsigma])
err = numpy.max(numpy.abs(ad - fd)) / numpy.max(numpy.abs(ad))
print("max |AD - FD| / max |AD|:", err)
assert err < 1e-8, err
print("OK")
