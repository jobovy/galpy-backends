"""Paper snippet: sections/design.tex, Sec. 2.2 (Choosing a backend), #2.

Claim: an object set up with a backend array as one of its parameters
computes with that backend in every call, also at python floats ("backends
are greedy").
"""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy
import torch

from galpy.potential import NFWPotential

# --- paper snippet begins ---
pot= NFWPotential(a=jnp.array(2.))
pot(1.1,0.2) # jax array
# --- paper snippet ends ---

out = pot(1.1, 0.2)
assert isinstance(out, jax.Array), type(out)
ref = NFWPotential(a=2.0)(1.1, 0.2)
assert isinstance(ref, float)
numpy.testing.assert_allclose(float(out), ref, rtol=1e-14)
# the same with a torch parameter, and the parameter is differentiable
a = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
pot_t = NFWPotential(a=a)
out_t = pot_t(1.1, 0.2)
assert isinstance(out_t, torch.Tensor)
numpy.testing.assert_allclose(out_t.item(), ref, rtol=1e-14)
(g,) = torch.autograd.grad(out_t, a)
fd = (NFWPotential(a=2.0 + 1e-6)(1.1, 0.2) - NFWPotential(a=2.0 - 1e-6)(1.1, 0.2)) / 2e-6
numpy.testing.assert_allclose(g.item(), fd, rtol=1e-7)
print("OK")
