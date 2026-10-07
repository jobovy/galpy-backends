"""Paper snippet: sections/dfs.tex, Sec. 7.2 (Spherical DFs), #2.

Claim: for the King model, the derivative w.r.t. W0 (through the solution of
the Poisson equation) runs under jax.jit, where the ODE is solved by diffrax.

NOTE: as originally written (no backend context), kingdf(W0=<tracer>) fails
both eagerly and under jax.jit: the constructor evaluates its potential at
Python floats, which dispatch to numpy and meet the traced W0. Inside
galpy.backend.use('jax') those floats go to jax and both work; the eager
derivative takes ~2 min on a CPU, so the snippet now shows only the jitted
one (~1 min, nearly all compilation).
"""

import time

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy

import galpy.backend
from galpy.df import kingdf

t0 = time.time()
# --- paper snippet begins ---
def sigmar(W0):
    with galpy.backend.use('jax'):
        kdf= kingdf(W0=W0,M=2.3,rt=1.4)
        return kdf.sigmar(jnp.asarray(0.5))
dsig= jax.jit(jax.grad(sigmar)) # ODE by diffrax
dsig(3.)
# --- paper snippet ends ---
print(f"snippet: {time.time() - t0:.1f} s")

g = dsig(3.0)
assert isinstance(g, jax.Array)


def sigmar_np(W0):  # numpy, rebuilt at each W0
    return float(kingdf(W0=W0, M=2.3, rt=1.4).sigmar(0.5))


h = 1e-3
fd = (
    -sigmar_np(3.0 + 2 * h) + 8 * sigmar_np(3.0 + h)
    - 8 * sigmar_np(3.0 - h) + sigmar_np(3.0 - 2 * h)
) / (12 * h)
print(float(g), fd)
# the jitted (diffrax) solve vs the scipy solve: agree to ~1e-8 (galpy's tests)
numpy.testing.assert_allclose(float(g), fd, rtol=1e-6)
# and without the backend context the derivative does not run (the reason
# for the context in the snippet; flip this when galpy dispatches on W0)
def sigmar_noctx(W0):
    return kingdf(W0=W0, M=2.3, rt=1.4).sigmar(jnp.asarray(0.5))


try:
    jax.grad(sigmar_noctx)(3.0)
except Exception as e:  # noqa: BLE001
    print("without use('jax'):", type(e).__name__)
else:
    raise AssertionError("kingdf(W0=tracer) now works without use('jax')")
print("OK")
