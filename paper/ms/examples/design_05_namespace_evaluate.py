"""Paper snippet: sections/design.tex, Sec. 2.3 (Backend-agnostic code).

Claim: a compute method that resolves the namespace from its coordinates and
uses only that namespace runs on numpy, jax, and torch.
"""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy
import torch

from galpy.backend import get_namespace
from galpy.potential import PlummerPotential, Potential


class PlummerLike(Potential):
    def __init__(self, amp=1.0, b=0.8):
        Potential.__init__(self, amp=amp)
        self._b2 = b**2

    # --- paper snippet begins ---
    def _evaluate(self,R,z,phi=0.,t=0.):
        xp= get_namespace(R,z,phi,t)
        return -self._amp/xp.sqrt(R**2+z**2+self._b2)
    # --- paper snippet ends ---


# (the snippet multiplies by self._amp itself, so evaluate the private method)
pl = PlummerLike(amp=1.0, b=0.8)
ref = PlummerPotential(amp=1.0, b=0.8)
Rs, zs = numpy.array([0.5, 1.0, 2.0]), numpy.array([0.1, -0.3, 1.0])
expect = ref(Rs, zs)
out_np = pl._evaluate(Rs, zs)
out_jax = pl._evaluate(jnp.asarray(Rs), jnp.asarray(zs))
out_torch = pl._evaluate(torch.as_tensor(Rs), torch.as_tensor(zs))
assert isinstance(out_np, numpy.ndarray)
assert isinstance(out_jax, jax.Array)
assert isinstance(out_torch, torch.Tensor)
for out in (out_np, numpy.asarray(out_jax), out_torch.numpy()):
    numpy.testing.assert_allclose(out, expect, rtol=1e-14)
assert get_namespace(Rs, zs, 0.0, 0.0) is numpy
print("OK")
