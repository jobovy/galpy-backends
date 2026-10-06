# BUG: streamdf construction is not traceable (jax.jit / jax.grad w.r.t. a potential parameter)

galpy: /var/tmp/fbp @ 9f6c9d6f7 (feat/backends), jax 0.10.1, CPU.

The backend-native streamdf construction (jax progenitor IC + diffrax
actionAngleIsochroneApprox) runs EAGERLY (it is what gS4 times), but tracing the
constructor -- `jax.jit` or `jax.grad` of any constructed quantity w.r.t. a potential
parameter -- fails in the track setup:

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from galpy.df import streamdf
from galpy.actionAngle import actionAngleIsochroneApprox
from galpy.orbit import Orbit
from galpy.potential import LogarithmicHaloPotential
IC = [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596]

def track(q):
    lp = LogarithmicHaloPotential(normalize=1.0, q=q)
    aA = actionAngleIsochroneApprox(pot=lp, b=0.8, tintJ=20, integrate_method="diffrax")
    sdf = streamdf(0.365 / 220.0, progenitor=Orbit(jnp.asarray(IC)), pot=lp, aA=aA,
                   leading=True, nTrackChunks=4, nTrackIterations=0, tdisrupt=4.0)
    return sdf._ObsTrack.sum()

track(jnp.asarray(0.9))            # eager: 4.97535..., OK (~210 s)
jax.jit(track)(jnp.asarray(0.9))   # TracerArrayConversionError
jax.grad(track)(jnp.asarray(0.9))  # TracerArrayConversionError
```

```
  File "galpy/df/streamdf.py", line 1737, in _determine_stream_track_backend
    prog_offset = _determine_stream_track_single_backend(
  File "galpy/df/streamdf.py", line 5335, in _determine_stream_track_single_backend
    tf = meanOmega(thetasTrack)
  File "galpy/df/streamdf.py", line 3333, in meanOmega
    * xp.sqrt(sig)
jax.errors.TracerArrayConversionError: The numpy.ndarray conversion method __array__()
was called on traced array with shape float64[]
```

Cause: `meanOmega` enters its backend branch because the stored moments are backend
arrays, but then resolves `xp` with `_ns_coerce(dangle, dOmin)` -- from `dangle`, which
is the Python float `0.0` at the prog_offset call -- so `xp` is numpy and
`numpy.sqrt(<tracer sig>)` raises. Eagerly the same line silently computes on numpy
(`numpy.sqrt` of a concrete jax array works), which is why only traced use fails.
The namespace should come from all of (dangle, _meandO, _sortedSigOEig[2]).

Secondary (cosmetic): without an explicit `nTrackIterations`, the intended
`ValueError("nTrackIterations cannot be chosen from the data under a trace ...")` in
`_determine_nTrackIterations` is never reached: `misalignment()` itself does
`if out > numpy.pi / 2.0:` on the tracer first -> `TracerBoolConversionError`.

Effect on gS4: no "jax (jit)" series for construction (eager only). Sampling and DF
evaluation from an eagerly built jax streamdf DO jit (timed in gS4).
