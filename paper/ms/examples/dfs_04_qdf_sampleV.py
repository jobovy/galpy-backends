"""Paper snippet: sections/dfs.tex, Sec. 7.3 (Disk DFs).

Claims: the quasi-isothermal DF's velocity sampler at a given position draws
by inverse transformation, such that, for a fixed key, a sample is
differentiable in the DF's parameters.

NOTES (both fixed in galpy PR #1646, checked below):
- before the PR, sampleV(...,key=<jax key>) on a qdf built from Python floats
  without a backend context raised a TypeError (numpy and jax arrays mixed in
  the inverse-CDF draw) and needed a galpy.backend.use('jax') block; with
  the PR the draw follows the key, so the snippet no longer has the block.
- before the PR, the derivative of the sample w.r.t. the DF's parameters was
  NaN for MWPotential2014 (far-tail rows of the velocity mesh with total
  probability ~1e-177, whose CDF normalisation's 1/total**2 derivative
  overflowed); it now matches a finite difference as for MWPotential.
"""

import time

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy

import galpy.backend
from galpy.actionAngle import actionAngleStaeckel
from galpy.backend import random as grandom
from galpy.df import quasiisothermaldf
from galpy.potential import MWPotential, MWPotential2014

aAS = actionAngleStaeckel(pot=MWPotential2014, delta=0.45, c=True)

t0 = time.time()
# --- paper snippet begins ---
qdf= quasiisothermaldf(1./3.,0.2,0.1,1.,1.,
                       pot=MWPotential2014,aA=aAS)
vs= qdf.sampleV(0.9,0.05,n=10**4,
                key=grandom.key(0,backend='jax'))
sigma_R= jnp.std(vs[:,0])
# --- paper snippet ends ---
print(f"snippet: {time.time() - t0:.1f} s")

assert isinstance(vs, jax.Array) and vs.shape == (10**4, 3)
# the sampled dispersion matches the DF's moment
sR = numpy.sqrt(float(qdf.sigmaR2(0.9, 0.05)))
print(float(sigma_R), sR)
numpy.testing.assert_allclose(float(sigma_R), sR, rtol=0.02)


# differentiable in the DF's parameters for a fixed key: d sigma_R / d sr
def make_sigR(pot, aA, n):
    def sigR(sr):
        q = quasiisothermaldf(1.0 / 3.0, sr, 0.1, 1.0, 1.0, pot=pot, aA=aA)
        v = q.sampleV(0.9, 0.05, n=n, key=grandom.key(0, backend="jax"))
        return jnp.std(v[:, 0])

    return sigR


sigR = make_sigR(MWPotential, actionAngleStaeckel(pot=MWPotential, delta=0.5, c=True), 1000)
g = jax.grad(sigR)(0.2)
h = 1e-5
fd = (sigR(0.2 + h) - sigR(0.2 - h)) / (2 * h)
print("MWPotential:", float(g), float(fd))
# the inverse CDF is piecewise linear in the tables: FD agrees to ~1e-5
numpy.testing.assert_allclose(float(g), float(fd), rtol=3e-5)
# MWPotential2014 (NaN before galpy PR #1646)
sigR14 = make_sigR(MWPotential2014, aAS, 100)
g14 = jax.grad(sigR14)(0.2)
fd14 = (sigR14(0.2 + h) - sigR14(0.2 - h)) / (2 * h)
print("MWPotential2014:", float(g14), float(fd14))
numpy.testing.assert_allclose(float(g14), float(fd14), rtol=3e-5)
# and inside the backend context (what the snippet needed before galpy PR
# #1646): the same draw, to a few ulp (up to 1.2e-12 relative on galpy
# feat/backends 87a95afc8; bit-identical on the #1646 base)
with galpy.backend.use("jax"):
    vs_ctx = qdf.sampleV(0.9, 0.05, n=10**4, key=grandom.key(0, backend="jax"))
assert isinstance(vs_ctx, jax.Array)
numpy.testing.assert_allclose(numpy.asarray(vs_ctx), numpy.asarray(vs), rtol=1e-11, atol=1e-14)
print("OK")
