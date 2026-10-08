"""Paper snippet: sections/streams.tex, Sec. 8.2 (Particle-spray models), #2.

Claim: the fitted track of a particle-spray stream is obtained, and
differentiated w.r.t. the potential, in the same way.

NOTE: streamTrack returns a StreamTrack object, which is not a jax pytree, so
jax.jacfwd cannot differentiate a function returning it; the snippet now
returns the track evaluated at fixed times tps (StreamTrack.x). Before galpy
PR #PRNUM forward mode (jax.jacfwd) also needed
integrate_kwargs={'adjoint':'direct'} and a jax progenitor Orbit; with it,
neither (see streams_02).
The derivative is that of the fit with its discrete choices frozen; see the
KNOWN LIMITATION below.
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
    prog= Orbit(jnp.asarray(ic))
    spdf= fardal15spraydf(mass,progenitor=prog,
        pot=lp,tdisrupt=td,
        integrate_kwargs={'adjoint':'direct'})
    tr= spdf.streamTrack(n=1500,key=key)
    return tr.x(tps)
track(0.9); jax.jacfwd(track)(0.9)
# --- paper snippet ends ---
print(f"snippet: {time.time() - t0:.1f} s")

x = track(0.9)
dx = jax.jacfwd(track)(0.9)
assert isinstance(x, jax.Array) and isinstance(dx, jax.Array)
assert dx.shape == tps.shape
assert numpy.all(numpy.isfinite(numpy.asarray(dx)))
# KNOWN LIMITATION: the fitted track is not a smooth function of q. Its
# closest-node assignment of particles and its GCV smoothing level are
# discrete choices, frozen in the derivative but re-made at q+-h, so a
# finite difference does not converge onto the derivative (it changes
# erratically with h; measured relative differences 1.0-1.6 for h=1e-3..1e-5,
# also with fixed smoothing/ntp). jacfwd runs and gives the derivative with
# those choices frozen; it cannot be checked against a finite difference.
h = 1e-3
fd = (numpy.asarray(track(0.9 + h)) - numpy.asarray(track(0.9 - h))) / (2 * h)
print("AD:", numpy.asarray(dx), "FD:", fd)
err = numpy.max(numpy.abs(numpy.asarray(dx) - fd)) / numpy.max(numpy.abs(fd))
print("max |AD - FD| / max |FD|:", err)
assert err > 0.1, (
    "the track derivative now agrees with a finite difference: galpy made the "
    "track smooth in q; tighten this check to a convergence test"
)
print("OK")
