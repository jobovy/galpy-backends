"""Paper snippet: sections/orbits.tex, Sec. 5.2 (Variational equations and
chaos).

Claims: integrate_dxdv integrates an orbit together with a deviation vector,
getOrbit_dxdv returns the deviation along the orbit, and lyapunov returns the
running estimate of the largest Lyapunov exponent (numpy/C features).
"""

import numpy

from galpy.orbit import Orbit
from galpy.potential import MWPotential2014

ts = numpy.linspace(0.0, 100.0, 1001)

# --- paper snippet begins ---
o= Orbit([0.8,0.3,0.75,0.,0.2,0.])
o.integrate_dxdv([1.,0.,0.,0.,0.,0.],ts,
                 MWPotential2014,method='dop853_c')
o.getOrbit_dxdv() # the deviation along the orbit
o.lyapunov(ts,MWPotential2014,method='dop853_c',
           renorm_every=10)
# --- paper snippet ends ---

dxdv = o.getOrbit_dxdv()
assert dxdv.shape == (len(ts), 6), dxdv.shape
numpy.testing.assert_allclose(dxdv[0], [1.0, 0, 0, 0, 0, 0], atol=1e-15)
# the deviation is the derivative of the orbit w.r.t. R0: finite difference
def R_of_R0(R0):
    oo = Orbit([R0, 0.3, 0.75, 0.0, 0.2, 0.0])
    oo.integrate(ts, MWPotential2014, method="dop853_c", rtol=1e-13, atol=1e-13)
    return oo.R(ts)


h = 1e-5  # the R(R0) map is strongly non-linear over 100 time units
fd = (
    -R_of_R0(0.8 + 2 * h) + 8 * R_of_R0(0.8 + h)
    - 8 * R_of_R0(0.8 - h) + R_of_R0(0.8 - 2 * h)
) / (12 * h)
err = numpy.max(numpy.abs(dxdv[:, 0] - fd)) / numpy.max(numpy.abs(fd))
print("max |dxdv - FD| / max |FD|:", err)
assert err < 1e-8, err
lam = o.lyapunov(ts, MWPotential2014, method="dop853_c", renorm_every=10)
lam = numpy.asarray(lam)
assert lam.shape[-1] == len(ts), lam.shape
assert numpy.all(numpy.isfinite(lam[..., 1:]))
# a regular orbit: the running estimate decays (roughly as ln t / t)
print(lam[..., 100], lam[..., -1])
assert abs(lam[..., -1]) < 0.3 * abs(lam[..., 100])
print("OK")
