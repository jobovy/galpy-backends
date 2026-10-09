"""Paper snippet: sections/dfs.tex, Sec. 7.2 (Spherical DFs), #1.

Claim: the derivative of a set of sampled radii w.r.t. the scale radius of a
Hernquist potential under torch; for a Hernquist sphere this derivative is
r_i/a, which the sampler reproduces to 1e-13.

NOTE: the original snippet passed rmin=0.1, with which the radii do not scale
with a (the derivative is off from r_i/a by 2-4%), and a float32 a, which
limits the derivative to float32 (5e-8); both changed. Before galpy PR
#1656 the construction and sampling needed a galpy.backend.use('torch')
block (the fE inversion and the sampler ran on numpy); with it, the
differentiated potential parameter is the data, so the block is dropped.
"""

import time

import numpy
import torch

import galpy.backend.random
from galpy.df import constantbetadf
from galpy.potential import HernquistPotential

t0 = time.time()
# --- paper snippet begins ---
a= torch.tensor(1.,dtype=torch.float64,
                requires_grad=True)
hp= HernquistPotential(amp=2.,a=a)
df= constantbetadf(pot=hp,beta=-0.2)
key= galpy.backend.random.key(7,backend='torch')
sam= df.sample(n=1000,key=key)
dr_da= torch.autograd.grad(sam.r().sum(),a)
# --- paper snippet ends ---
print(f"snippet: {time.time() - t0:.1f} s")

r = sam.r()
assert isinstance(r, torch.Tensor) and r.shape == (1000,)
assert isinstance(dr_da, tuple) and len(dr_da) == 1
# for a Hernquist sphere, r_i scales with a at fixed uniforms: dr_i/da = r_i/a,
# so d(sum r_i)/da = sum r_i / a
expect = float(r.detach().sum()) / 1.0
print(float(dr_da[0]), expect, r.dtype)
numpy.testing.assert_allclose(float(dr_da[0]), expect, rtol=1e-13)
# and per particle, dr_i/da = r_i/a to 1e-13
a2 = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
df2 = constantbetadf(pot=HernquistPotential(amp=2.0, a=a2), beta=-0.2)
s2 = df2.sample(n=1000, key=galpy.backend.random.key(7, backend="torch"))
r2 = s2.r()
numpy.testing.assert_array_equal(r2.detach().numpy(), r.detach().numpy())
for i in range(0, 1000, 97):
    gi = torch.autograd.grad(r2[i], a2, retain_graph=True)[0]
    numpy.testing.assert_allclose(float(gi), float(r2[i]) / 1.0, rtol=1e-13)
print("OK")
