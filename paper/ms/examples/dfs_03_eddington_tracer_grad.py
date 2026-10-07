"""Paper snippet: sections/dfs.tex, Sec. 7.2 (Spherical DFs), #3.

Claim: a DF built from a density that is not the one generating the
potential (a tracer) is differentiable w.r.t. the parameters of both.

NOTE: the snippet (tracer parameter) runs as written. The derivative w.r.t.
the POTENTIAL's parameter fails without a backend context (the constructor
evaluates the potential at Python floats, e.g. r=0 in _handle_rmin, which go
to numpy and meet the traced parameter); it runs inside
galpy.backend.use('jax'), which is how it is checked below.
"""

import time

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy

from galpy.df import eddingtondf
from galpy.potential import HernquistPotential, NFWPotential

t0 = time.time()
# --- paper snippet begins ---
def dispersion(a_tracer):
    hp= HernquistPotential(amp=0.1,a=a_tracer)
    df= eddingtondf(pot=NFWPotential(amp=1.,a=2.),
                    denspot=hp)
    return df.sigmar(jnp.asarray(1.))
jax.grad(dispersion)(0.7)
# --- paper snippet ends ---
print(f"snippet: {time.time() - t0:.1f} s")

g = jax.grad(dispersion)(0.7)
assert isinstance(g, jax.Array)


def dispersion_np(a):
    df = eddingtondf(
        pot=NFWPotential(amp=1.0, a=2.0), denspot=HernquistPotential(amp=0.1, a=a)
    )
    return float(df.sigmar(1.0))


numpy.testing.assert_allclose(float(dispersion(0.7)), dispersion_np(0.7), rtol=1e-7)
h = 1e-3
fd = (
    -dispersion_np(0.7 + 2 * h) + 8 * dispersion_np(0.7 + h)
    - 8 * dispersion_np(0.7 - h) + dispersion_np(0.7 - 2 * h)
) / (12 * h)
print(float(g), fd)
numpy.testing.assert_allclose(float(g), fd, rtol=1e-6)


# and w.r.t. the potential's parameter (needs the backend context)
import galpy.backend


def dispersion_pot(a_pot):
    with galpy.backend.use("jax"):
        df = eddingtondf(
            pot=NFWPotential(amp=1.0, a=a_pot),
            denspot=HernquistPotential(amp=0.1, a=0.7),
        )
        return df.sigmar(jnp.asarray(1.0))


def dispersion_pot_noctx(a_pot):
    df = eddingtondf(
        pot=NFWPotential(amp=1.0, a=a_pot), denspot=HernquistPotential(amp=0.1, a=0.7)
    )
    return df.sigmar(jnp.asarray(1.0))


try:
    jax.grad(dispersion_pot_noctx)(2.0)
except Exception as e:  # noqa: BLE001
    print("w.r.t. the potential without use('jax'):", type(e).__name__)
else:
    raise AssertionError("now works without use('jax'): update the paper note")


def dispersion_pot_np(a_pot):
    df = eddingtondf(
        pot=NFWPotential(amp=1.0, a=a_pot), denspot=HernquistPotential(amp=0.1, a=0.7)
    )
    return float(df.sigmar(1.0))


gp = jax.grad(dispersion_pot)(2.0)
fdp = (
    -dispersion_pot_np(2.0 + 2 * h) + 8 * dispersion_pot_np(2.0 + h)
    - 8 * dispersion_pot_np(2.0 - h) + dispersion_pot_np(2.0 - 2 * h)
) / (12 * h)
print(float(gp), fdp)
numpy.testing.assert_allclose(float(gp), fdp, rtol=1e-6)
print("OK")
