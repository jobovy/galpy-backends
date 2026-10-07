"""Paper snippet: sections/orbits.tex, Sec. 5.1 (Three ways to integrate an
orbit), #2.

Claim: the same derivative under torch, with torchode, by backward().
"""

import numpy
import torch

from galpy.orbit import Orbit
from galpy.potential import PlummerPotential

ic = [1.0, 0.1, 1.1, 0.1, 0.05, 0.0]
ts = numpy.linspace(0.0, 5.0, 51)

# --- paper snippet begins ---
b= torch.tensor(0.6,requires_grad=True)
o= Orbit(torch.as_tensor(ic))
pp= PlummerPotential(amp=1.,b=b)
o.integrate(torch.as_tensor(ts),pp,method='torchode')
o.R(ts[-1]).backward() # b.grad = dR/db
# --- paper snippet ends ---

assert isinstance(b.grad, torch.Tensor)


def final_R_c(bb):
    oc = Orbit(ic)
    oc.integrate(ts, PlummerPotential(amp=1.0, b=bb), method="dop853_c")
    return oc.R(ts[-1])


h = 1e-3
fd = (
    -final_R_c(0.6 + 2 * h) + 8 * final_R_c(0.6 + h)
    - 8 * final_R_c(0.6 - h) + final_R_c(0.6 - 2 * h)
) / (12 * h)
print(b.dtype, float(b.grad), fd)
# b is a float32 tensor (torch's default), so float32 accuracy
numpy.testing.assert_allclose(float(b.grad), fd, rtol=1e-5)
print("OK")
