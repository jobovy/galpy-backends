"""Paper snippet: sections/design.tex, Sec. 2.4 (Numerical building blocks).

Claims: key=None (the default) keeps numpy's global generator, such that a
seeded script draws the same samples as before; key=<jax key> draws with jax
and the sample is differentiable (w.r.t. the DF's parameters).
"""

import time

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy

from galpy.df import isotropicHernquistdf
from galpy.potential import HernquistPotential

numpy.random.seed(1)
t0 = time.time()
# --- paper snippet begins ---
from galpy.backend import random as grandom
hp= HernquistPotential(amp=2.,a=1.3)
df= isotropicHernquistdf(pot=hp)
sam= df.sample(n=10**6)          # numpy, as before
key= grandom.key(42,backend='jax')
sam= df.sample(n=10**6,key=key)  # jax, differentiable
# --- paper snippet ends ---
print(f"snippet: {time.time() - t0:.1f} s")

# the jax sample is an Orbit with jax phase-space coordinates
assert isinstance(sam.r(), jax.Array), type(sam.r())
assert sam.r().shape == (10**6,)
# the numpy sample is reproducible from numpy's global seed
numpy.random.seed(1)
s1 = isotropicHernquistdf(pot=HernquistPotential(amp=2.0, a=1.3)).sample(n=1000)
numpy.random.seed(1)
s2 = isotropicHernquistdf(pot=HernquistPotential(amp=2.0, a=1.3)).sample(n=1000)
assert isinstance(s1.r(), numpy.ndarray)
numpy.testing.assert_array_equal(s1.r(), s2.r())
# median radius of a Hernquist sphere is a(1+sqrt(2)) ~ 3.14 (mass-weighted;
# check the jax sample statistically)
numpy.testing.assert_allclose(
    numpy.median(numpy.asarray(sam.r())), 1.3 * (1.0 + numpy.sqrt(2.0)), rtol=0.01
)


# differentiable: for fixed key, the radii scale as r_i ~ a, so dr_i/da = r_i/a
def mean_r(a):
    df = isotropicHernquistdf(pot=HernquistPotential(amp=2.0, a=a))
    return jnp.mean(df.sample(n=1000, key=key).r())


g = jax.grad(mean_r)(1.3)
numpy.testing.assert_allclose(float(g), float(mean_r(1.3)) / 1.3, rtol=1e-8)
print("OK")
