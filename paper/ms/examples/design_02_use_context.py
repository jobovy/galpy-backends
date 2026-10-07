"""Paper snippet: sections/design.tex, Sec. 2.2 (Choosing a backend), #2.

Claim: when there is no array to dispatch on, the default backend set by the
galpy.backend.use context manager is used.

NOTE: the original snippet built an Orbit from Python floats inside the
context and claimed it was integrated in jax; that is false (an Orbit's
backend follows its initial condition only: the IC stays numpy and the
integration runs in C). The snippet now evaluates a potential at Python
floats, which does follow the context.
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
# and the claim the original snippet made does NOT hold: an Orbit from floats
# stays numpy inside the context
from galpy.orbit import Orbit

with galpy.backend.use("jax"):
    o = Orbit([1.0, 0.1, 1.1, 0.0, 0.05, 0.0])
assert isinstance(o.vxvv, numpy.ndarray) and o._ic_backend is None
print("OK")
