"""Paper snippet: sections/design.tex, Sec. 2.2 (Choosing a backend), #2.

Claim: when there is no array to dispatch on, the default backend set by the
galpy.backend.use context manager is used.

NOTE: the original snippet built an Orbit from Python floats inside the
context and claimed its IC becomes a jax array. Before galpy PR
#1646 that was false (the IC stayed numpy and the integration ran in C);
with it, Orbit([floats]) inside an unforced use('jax') gets a jax IC, as
checked below, so the original snippet can be restored once it is merged
(the default method still integrates in C, returning jax arrays; an
in-backend method= integrates in jax). The snippet here evaluates a
potential at Python floats, which follows the context either way.
"""

import jax

jax.config.update("jax_enable_x64", True)
import numpy

from galpy.potential import MWPotential2014, evaluatePotentials

# --- paper snippet begins ---
import galpy.backend
with galpy.backend.use('jax'):
    Phi= evaluatePotentials(MWPotential2014,1.,0.) # jax
# --- paper snippet ends ---

assert isinstance(Phi, jax.Array), type(Phi)
numpy.testing.assert_allclose(
    float(Phi), evaluatePotentials(MWPotential2014, 1.0, 0.0), rtol=1e-14
)
# outside the context, the default (numpy) is back
assert not isinstance(evaluatePotentials(MWPotential2014, 1.0, 0.0), jax.Array)
# and the claim the original snippet made (galpy PR #1646): an Orbit from
# floats inside the context gets a jax IC, and is integrated in jax with an
# in-backend method
from galpy.orbit import Orbit

ts = numpy.linspace(0.0, 10.0, 101)
with galpy.backend.use("jax"):
    o = Orbit([1.0, 0.1, 1.1, 0.0, 0.05, 0.0])  # IC becomes a jax array
    o.integrate(ts, MWPotential2014, method="diffrax")
assert isinstance(o._ic_backend, jax.Array), type(o._ic_backend)
assert isinstance(o.R(ts), jax.Array)
oc = Orbit([1.0, 0.1, 1.1, 0.0, 0.05, 0.0])
oc.integrate(ts, MWPotential2014, method="dop853_c")
numpy.testing.assert_allclose(numpy.asarray(o.R(ts)), oc.R(ts), rtol=1e-8)
print("OK")
