"""Paper snippet: sections/streams.tex, Sec. 8.2 (Particle-spray models), #2.

Claim: the fitted track of a particle-spray stream is obtained, and
differentiated w.r.t. the potential, in the same way.

NOTE: streamTrack returns a StreamTrack object, which is not a jax pytree, so
jax.jacfwd cannot differentiate a function returning it; the snippet now
returns the track evaluated at fixed times tps (StreamTrack.x). Since galpy
PR #1646 the progenitor can be set up from the list (see streams_02); the
direct adjoint stays for jax 0.6 (Python 3.10).
The derivative is generative (the particles are drawn and integrated inside
it). Since galpy #1731/#1732 the track fit (method="pspline", the default) is
a smooth function of the particles, and the same eagerly and under jax.jit,
so central finite differences of the track converge to the derivative.
"""

import time

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy

import galpy.backend
import galpy.backend.random
from galpy.df import fardal15spraydf
from galpy.orbit import Orbit
from galpy.potential import LogarithmicHaloPotential
from galpy.util.conversion import mass_in_msol, time_in_Gyr

ic = [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596]
mass = 2.0e4 / mass_in_msol(220.0, 8.0)
td = 4.5 / time_in_Gyr(220.0, 8.0)
key = galpy.backend.random.key(1, backend="jax")
tps = jnp.linspace(0.0, 0.2, 5)  # progenitor times along the leading arm

t0 = time.time()
# --- paper snippet begins ---
def track(q):
    lp= LogarithmicHaloPotential(normalize=1.,q=q)
    spdf= fardal15spraydf(mass,progenitor=Orbit(ic),
        pot=lp,tdisrupt=td,
        integrate_kwargs={'adjoint':'direct'})
    tr= spdf.streamTrack(n=1500,tail='leading',key=key)
    return tr.x(tps)
track(0.9); jax.jacfwd(track)(0.9)
# --- paper snippet ends ---
print(f"snippet: {time.time() - t0:.1f} s")

x = track(0.9)
dx = jax.jacfwd(track)(0.9)
assert isinstance(x, jax.Array) and isinstance(dx, jax.Array)
assert dx.shape == tps.shape
assert numpy.all(numpy.isfinite(numpy.asarray(dx)))
# The eager and jitted tracks are the same function (they differ only through
# the progenitor's integrator: dop853_c eagerly, diffrax under jit).
track_jit = jax.jit(track)
diff = numpy.max(numpy.abs(numpy.asarray(x) - numpy.asarray(track_jit(0.9))))
print("max |eager - jit|:", diff)
assert diff < 1e-8, diff  # measured 5.0e-10
# The derivative vs central finite differences of the eager track, which
# converge to it as h^2 down to the integration noise.
for h, tol in ((1e-4, 1e-3), (1e-5, 1e-5)):
    fd = (numpy.asarray(track(0.9 + h)) - numpy.asarray(track(0.9 - h))) / (2 * h)
    err = numpy.max(numpy.abs(numpy.asarray(dx) - fd)) / numpy.max(numpy.abs(fd))
    print(f"h={h:.0e}: AD:", numpy.asarray(dx), "FD:", fd, "rel:", err)
    assert err < tol, err  # measured 1.2e-4 (1e-4), 1.8e-6 (1e-5)
print("OK")
