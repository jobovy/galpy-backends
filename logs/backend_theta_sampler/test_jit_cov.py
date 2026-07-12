"""jit streamTrack covariance: tr._cov_xyz populated + differentiable (h-converged)."""
import sys; sys.path.insert(0, "/var/tmp/cint-param")
import numpy
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from galpy.orbit import Orbit
from galpy.potential import LogarithmicHaloPotential
from galpy.df import fardal15spraydf
from galpy.backend import random as grandom, is_backend_array

M = 3e-5; TD = 2.0; N = 60
IC = [1.2, 0.15, 0.85, 0.08, 0.05, 0.0]
key = grandom.key(0, backend="jax")


def cov_sum(amp):
    spdf = fardal15spraydf(M, progenitor=Orbit(IC),
                           pot=LogarithmicHaloPotential(amp=amp, q=0.9), tdisrupt=TD)
    tr = spdf.streamTrack(n=N, tail="leading", velocity_weight=1.0, order=2, key=key)  # AUTO extent
    return tr._cov_xyz.sum()   # trace/sum of the (ninterp,6,6) covariance


# shape/populated check
spdf = fardal15spraydf(M, progenitor=Orbit(IC), pot=LogarithmicHaloPotential(amp=jnp.asarray(1.1), q=0.9), tdisrupt=TD)
tr = spdf.streamTrack(n=N, tail="leading", velocity_weight=1.0, order=2, key=key)  # AUTO extent
print("cov_xyz is not None:", tr._cov_xyz is not None, " backend:", is_backend_array(tr._cov_xyz),
      " shape:", tuple(tr._cov_xyz.shape))
# PSD check: eigenvalues >= 0 (a few tp)
import numpy as _np
c = _np.asarray(tr._cov_xyz)
mineig = min(float(_np.linalg.eigvalsh(c[k]).min()) for k in [10, 100, 150])
print("min eigenvalue across sample tp (>= ~0 => PSD):", mineig)

print("=== jit cov gradient ===")
g = float(jax.jit(jax.grad(cov_sum))(jnp.asarray(1.1)))
print("  AD d(cov.sum)/d(amp) =", g)
fj = jax.jit(cov_sum); best = 1e9
for h in [1e-3, 1e-4, 1e-5, 1e-6]:
    gfd = (float(fj(jnp.asarray(1.1 + h))) - float(fj(jnp.asarray(1.1 - h)))) / (2 * h)
    rel = abs(g - gfd) / max(abs(g), 1e-9); best = min(best, rel)
    print(f"    h={h:g}: FD={gfd:.6e} REL={rel:.2e}")
print(f"  BEST REL = {best:.2e}  -> {'VALIDATED' if best < 1e-3 else 'CHECK'}")
