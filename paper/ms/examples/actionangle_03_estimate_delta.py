"""Paper snippet: sections/actionangle.tex, Sec. 6.2 (Staeckel and adiabatic
approximations), #3.

Claim: estimateDeltaStaeckel runs on the backends and traces, such that a
focal length estimated along an integrated orbit can be part of a compiled
program.
"""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy

from galpy.actionAngle import actionAngleStaeckel, estimateDeltaStaeckel
from galpy.orbit import Orbit
from galpy.potential import MWPotential2014

ts = numpy.linspace(0.0, 10.0, 101)
o = Orbit(jnp.array([1.0, 0.1, 1.1, 0.1, 0.05, 0.0]))
o.integrate(ts, MWPotential2014, method="diffrax")

# --- paper snippet begins ---
delta= estimateDeltaStaeckel(MWPotential2014,
                             o.R(ts),o.z(ts))
aAS= actionAngleStaeckel(pot=MWPotential2014,
                         delta=delta,c=False)
jr,lz,jz= aAS(o.R(ts),o.vR(ts),o.vT(ts),
              o.z(ts),o.vz(ts))
# --- paper snippet ends ---

for x in (delta, jr, lz, jz):
    assert isinstance(x, jax.Array), type(x)
assert jr.shape == (len(ts),)
# the same as numpy
onp = Orbit([1.0, 0.1, 1.1, 0.1, 0.05, 0.0])
onp.integrate(ts, MWPotential2014, method="dop853_c", rtol=1e-12, atol=1e-12)
delta_np = estimateDeltaStaeckel(MWPotential2014, onp.R(ts), onp.z(ts))
numpy.testing.assert_allclose(float(delta), delta_np, rtol=1e-8)
jr_np = actionAngleStaeckel(pot=MWPotential2014, delta=delta_np, c=False)(
    onp.R(ts), onp.vR(ts), onp.vT(ts), onp.z(ts), onp.vz(ts)
)[0]
numpy.testing.assert_allclose(numpy.asarray(jr), jr_np, rtol=1e-7)
# the actions are conserved along the orbit (to the Staeckel approximation)
assert numpy.std(numpy.asarray(jr)) < 0.05 * numpy.mean(numpy.asarray(jr))
numpy.testing.assert_allclose(numpy.asarray(lz), 1.1, rtol=1e-8)


# the whole chain compiles: orbit -> delta -> actions in one jitted function
@jax.jit
def mean_jz(ic):
    o = Orbit(ic)
    o.integrate(ts, MWPotential2014, method="diffrax")
    delta = estimateDeltaStaeckel(MWPotential2014, o.R(ts), o.z(ts))
    aAS = actionAngleStaeckel(pot=MWPotential2014, delta=delta, c=False)
    return jnp.mean(aAS(o.R(ts), o.vR(ts), o.vT(ts), o.z(ts), o.vz(ts))[2])


ic = jnp.array([1.0, 0.1, 1.1, 0.1, 0.05, 0.0])
numpy.testing.assert_allclose(float(mean_jz(ic)), float(jnp.mean(jz)), rtol=1e-10)
print("OK")
