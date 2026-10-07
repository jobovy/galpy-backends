"""Paper snippet: sections/dfs.tex, Sec. 7.3 (Disk DFs).

Claims: the quasi-isothermal DF's velocity sampler at a given position draws
by inverse transformation, such that, for a fixed key, a sample is
differentiable in the DF's parameters.

NOTES:
- as originally written (no backend context), sampleV(...,key=<jax key>) on a
  qdf built from Python floats raises a TypeError (numpy and jax arrays mixed
  in the inverse-CDF draw); it runs inside galpy.backend.use('jax').
- the derivative of the sample w.r.t. the DF's parameters is finite and
  matches a finite difference for MWPotential, but is NaN for MWPotential2014
  (the gradient of the DF on the v_T=0 node of the velocity mesh, where L_z=0,
  is NaN; jax_debug_nans points at _cumcdf in _sampleV_icdf). This is a galpy
  defect; the check below flags when it is fixed.
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
with galpy.backend.use('jax'):
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
        with galpy.backend.use("jax"):
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
# KNOWN DEFECT: NaN for MWPotential2014
g14 = jax.grad(make_sigR(MWPotential2014, aAS, 100))(0.2)
print("MWPotential2014:", float(g14))
assert numpy.isnan(float(g14)), (
    "the MWPotential2014 qdf derivative is no longer NaN: galpy was fixed, "
    "check it against a finite difference here instead"
)
print("OK")
