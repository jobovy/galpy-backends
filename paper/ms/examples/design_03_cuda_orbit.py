"""Paper snippet: sections/design.tex, Sec. 2.2 (Choosing a backend), #3.

Claim: an Orbit set up from a tensor on a GPU is integrated there (with
torchode) and its accessors return CUDA tensors. Needs CUDA.

NOTE: the original snippet did not pass a method; galpy's default method,
symplec4_c, does not route to the in-backend solver for a concrete IC, so
the orbit was integrated in C on the host. method='torchode' is required.
"""

import numpy
import torch

assert torch.cuda.is_available(), "this example needs a CUDA GPU"
from galpy.orbit import Orbit
from galpy.potential import MWPotential2014

# the times on the GPU (with galpy PR #1646 numpy times work too, checked
# below; before it torchode raised its same_device assertion)
ts = torch.linspace(0.0, 10.0, 101, dtype=torch.float64, device="cuda")

# --- paper snippet begins ---
o= Orbit(torch.tensor([1.,0.1,1.1,0.,0.05,0.],
                      device='cuda'))
o.integrate(ts,MWPotential2014,method='torchode')
o.R(ts) # a CUDA tensor
# --- paper snippet ends ---

R = o.R(ts)
assert isinstance(R, torch.Tensor) and R.device.type == "cuda", R.device
# agrees with the C integrator (to float32: the IC is a float32 tensor, and a
# user's float32 tensor stays float32)
assert R.dtype == torch.float32
oc = Orbit([1.0, 0.1, 1.1, 0.0, 0.05, 0.0])
tsn = ts.cpu().numpy()
oc.integrate(tsn, MWPotential2014, method="dop853_c")
numpy.testing.assert_allclose(R.cpu().numpy(), oc.R(tsn), rtol=1e-6, atol=1e-6)
# numpy times + a CUDA IC with torchode (fixed in galpy PR #1646)
o2 = Orbit(torch.tensor([1.0, 0.1, 1.1, 0.0, 0.05, 0.0], device="cuda"))
o2.integrate(tsn, MWPotential2014, method="torchode")
R2 = o2.R(tsn)
assert R2.device.type == "cuda"
numpy.testing.assert_allclose(R2.cpu().numpy(), R.cpu().numpy(), rtol=1e-12)
print("OK")
