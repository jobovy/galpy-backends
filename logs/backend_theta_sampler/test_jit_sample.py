"""Goal: jax.jit of fardal15spraydf(...).sample() under a backend-theta potential,
with a correct h-converged gradient. THE Option-A target."""
import sys; sys.path.insert(0, "/var/tmp/cint-param")
import numpy
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from galpy.orbit import Orbit
from galpy.potential import LogarithmicHaloPotential
from galpy.df import fardal15spraydf
from galpy.backend import random as grandom

M = 3e-5; TD = 2.0; N = 30
IC = [1.2, 0.15, 0.85, 0.08, 0.05, 0.0]
key = grandom.key(0, backend="jax")


def sample_sum(amp):
    spdf = fardal15spraydf(M, progenitor=Orbit(IC),
                           pot=LogarithmicHaloPotential(amp=amp, q=0.9), tdisrupt=TD)
    xv = spdf.sample(N, return_orbit=False, tail="leading", key=key)
    return xv[0].sum()   # sum of sampled R


print("=== eager (no jit) ===")
try:
    v = sample_sum(jnp.asarray(1.1))
    print("  sample_sum =", float(v))
except Exception as e:
    import traceback; traceback.print_exc()

print("=== jax.jit ===")
try:
    f = jax.jit(sample_sum)
    v = f(jnp.asarray(1.1)); print("  jit sample_sum =", float(v))
except Exception as e:
    import traceback; traceback.print_exc()

print("=== jit grad + h-convergence ===")
try:
    g = float(jax.jit(jax.grad(sample_sum))(jnp.asarray(1.1)))
    print("  AD d(sum R)/d(amp) =", g)
    fj = jax.jit(sample_sum)
    for h in [1e-3, 1e-4, 1e-5]:
        gfd = (float(fj(jnp.asarray(1.1 + h))) - float(fj(jnp.asarray(1.1 - h)))) / (2 * h)
        print(f"    h={h:g}: FD={gfd:.6e} REL={abs(g-gfd)/max(abs(g),1e-9):.2e}")
except Exception as e:
    import traceback; traceback.print_exc()
