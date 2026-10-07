"""Paper snippet: sections/potentials.tex, Sec. 3 (Potentials), #2.

Claim: the same parameter derivative with torch, by backward().
"""

import numpy
import torch

from galpy.potential import MiyamotoNagaiPotential

# --- paper snippet begins ---
a= torch.tensor(0.5,requires_grad=True)
mp= MiyamotoNagaiPotential(amp=1.,a=a,b=0.3)
mp.Rforce(torch.tensor(1.2),
          torch.tensor(0.1)).backward()
a.grad # dFR/da
# --- paper snippet ends ---

assert isinstance(a.grad, torch.Tensor)
R, z, b = 1.2, 0.1, 0.3
s = 0.5 + numpy.sqrt(z**2 + b**2)
exact = 3.0 * R * s * (R**2 + s**2) ** -2.5
print(a.grad.dtype, float(a.grad), exact)
# the user's tensors are float32, so the derivative is float32-accurate
numpy.testing.assert_allclose(float(a.grad), exact, rtol=1e-6)
print("OK")
