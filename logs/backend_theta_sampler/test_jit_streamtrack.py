"""Next Option-A target: jax.jit of fardal15spraydf(...).streamTrack() (samples
internally + reconstructs). Find the reconstruction barriers."""
import sys; sys.path.insert(0, "/var/tmp/cint-param")
import numpy
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from galpy.orbit import Orbit
from galpy.potential import LogarithmicHaloPotential
from galpy.df import fardal15spraydf
from galpy.backend import random as grandom

M = 3e-5; TD = 2.0; N = 40
IC = [1.2, 0.15, 0.85, 0.08, 0.05, 0.0]
key = grandom.key(0, backend="jax")


def track_x(amp):
    spdf = fardal15spraydf(M, progenitor=Orbit(IC),
                           pot=LogarithmicHaloPotential(amp=amp, q=0.9), tdisrupt=TD)
    tr = spdf.streamTrack(tail="leading", velocity_weight=1.0, n=N, key=key)  # AUTO track_time_range
    return tr._track_xyz.sum()   # robust loss (whole track, no tp-range issue)


print("=== eager streamTrack() ===")
try:
    v = track_x(jnp.asarray(1.1)); print("  track.x(0.3) =", float(v))
except Exception as e:
    import traceback; traceback.print_exc()

print("=== jax.jit streamTrack() ===")
try:
    v = jax.jit(track_x)(jnp.asarray(1.1)); print("  jit track.x =", float(v))
    g = float(jax.jit(jax.grad(track_x))(jnp.asarray(1.1)))
    print("  AD d(track.sum)/d(amp) =", g)
    fj = jax.jit(track_x)
    best = 1e9
    for h in [1e-3, 1e-4, 1e-5, 1e-6]:
        gfd = (float(fj(jnp.asarray(1.1 + h))) - float(fj(jnp.asarray(1.1 - h)))) / (2 * h)
        rel = abs(g - gfd) / max(abs(g), 1e-9); best = min(best, rel)
        print(f"    h={h:g}: FD={gfd:.6e} REL={rel:.2e}")
    print(f"  BEST REL = {best:.2e}  -> {'VALIDATED' if best < 1e-3 else 'CHECK'}")
except Exception as e:
    import traceback
    tb = traceback.format_exc()
    for line in tb.splitlines():
        if "streamTrack.py" in line or "streamspraydf.py" in line or "Error" in line or "__array__" in line:
            print("  " + line.strip())
