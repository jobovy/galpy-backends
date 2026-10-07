"""Paper snippet: sections/design.tex, Sec. 2.2 (Choosing a backend), #1.

Claim: a galpy function called on a jax array computes with jax and returns
a jax array on the input's device, without further instruction.
"""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy

# --- paper snippet begins ---
from galpy.potential import (MWPotential2014,
                             evaluateRforces)
R= jnp.linspace(0.5,2.,1001)
FR= evaluateRforces(MWPotential2014,R,0.) # jax array
# --- paper snippet ends ---

assert isinstance(FR, jax.Array), type(FR)
assert FR.devices() == R.devices()
assert FR.shape == (1001,)
# the same numbers as the numpy path
FR_np = evaluateRforces(MWPotential2014, numpy.linspace(0.5, 2.0, 1001), 0.0)
assert isinstance(FR_np, numpy.ndarray)
numpy.testing.assert_allclose(numpy.asarray(FR), FR_np, rtol=1e-13, atol=0.0)
# "with a torch tensor the same call returns a tensor"
import torch

FR_t = evaluateRforces(
    MWPotential2014, torch.linspace(0.5, 2.0, 1001, dtype=torch.float64), 0.0
)
assert isinstance(FR_t, torch.Tensor)
numpy.testing.assert_allclose(FR_t.numpy(), FR_np, rtol=1e-13, atol=0.0)
print("OK")
