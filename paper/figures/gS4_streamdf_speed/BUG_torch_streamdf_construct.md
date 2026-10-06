# BUG: streamdf cannot be constructed with a torch progenitor (eager torch)

galpy: /var/tmp/fbp @ 9f6c9d6f7 (feat/backends), torch 2.12.0, CPU.

A streamdf whose progenitor is a torch Orbit (the backend-native, differentiable
construction that works under jax with `Orbit(jnp.asarray(IC))`) fails in
`_offset_setup` -- for a plain-float potential parameter as well as for a torch
(non-grad or requires_grad) one:

```python
import torch
torch.set_default_dtype(torch.float64)
from galpy.df import streamdf
from galpy.actionAngle import actionAngleIsochroneApprox
from galpy.orbit import Orbit
from galpy.potential import LogarithmicHaloPotential
IC = [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596]
lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)          # also fails with q=torch.tensor(0.9)
aA = actionAngleIsochroneApprox(pot=lp, b=0.8, tintJ=20, integrate_method="torchode")
streamdf(0.365 / 220.0, progenitor=Orbit(torch.tensor(IC)), pot=lp, aA=aA,
         leading=True, nTrackChunks=4, tdisrupt=4.0)
```

```
  File "galpy/df/streamdf.py", line 646, in _offset_setup
    * _ns_sqrt(self._sortedSigOEig[2])
  File "galpy/df/streamdf.py", line 151, in _ns_sqrt
    return numpy.sqrt(x)
  File ".../torch/_tensor.py", line 1253, in __array__
RuntimeError: Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.
```

Cause: `_ns_sqrt` only takes the namespace branch when `under_trace(x)`; an eager torch
tensor that carries an autograd graph (here `_sortedSigOEig`; it requires grad even
when no user input does -- presumably from the torch AD `calcaAJac` dO/dJ) is not
"under trace", so it falls to `numpy.sqrt` -> `Tensor.__array__` raises. This is the
"under_trace misses eager torch grad" pattern: the check should be
`differentiating(x)` / `is_backend_array(x)`, not `under_trace(x)`.
The same construction under jax (`Orbit(jnp.asarray(IC))`, diffrax aA) succeeds
(eager jax arrays have no tape, so `numpy.sqrt` works on them).

Effect on gS4: no "torch" series for streamdf construction or for sampling from a
torch-native streamdf (recorded as unsupported in the JSON).
