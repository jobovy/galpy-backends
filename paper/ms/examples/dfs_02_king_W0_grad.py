"""Paper snippet: sections/dfs.tex, Sec. 7.2 (Spherical DFs), #2.

Claim: for the King model, the derivative w.r.t. W0 (through the solution of
the Poisson equation) runs under jax.jit, where the ODE is solved by diffrax.

NOTE: before galpy PR #1646, kingdf(W0=<tracer>) without a backend
context failed both eagerly and under jax.jit: the constructor evaluated its
potential at Python floats, which dispatched to numpy and met the traced W0.
With the PR the constructor follows the traced W0, so the snippet no longer
needs a galpy.backend.use('jax') block, and since galpy PR #1656 sigmar takes
a Python-float radius under a traced parameter (it needed jnp.asarray(0.5)).
The eager derivative takes ~2 min on a CPU, so the snippet shows only the
jitted one (~1 min, nearly all compilation).
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
    kdf= kingdf(W0=W0,M=2.3,rt=1.4)
    return kdf.sigmar(0.5)
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
# and the same derivative inside a galpy.backend.use('jax') block (what the
# snippet needed before galpy PR #1646)
def sigmar_ctx(W0):
    with galpy.backend.use("jax"):
        return kingdf(W0=W0, M=2.3, rt=1.4).sigmar(jnp.asarray(0.5))


g_ctx = jax.jit(jax.grad(sigmar_ctx))(3.0)
print("with use('jax'):", float(g_ctx))
numpy.testing.assert_allclose(float(g_ctx), float(g), rtol=1e-12)
print("OK")
