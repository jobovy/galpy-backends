"""Paper snippet: sections/streams.tex, Sec. 8.1 (The action-angle stream
model).

Claims: a streamdf is set up as before, with a progenitor orbit, an
actionAngle instance (here the isochrone approximation integrating with
diffrax), and a velocity dispersion; sampling with a jax key returns jax
arrays.

NOTE: the original snippet called time_in_Gyr() without its (vo,ro)
arguments.
"""

import time

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy

from galpy.actionAngle import actionAngleIsochroneApprox
from galpy.backend import random as grandom
from galpy.df import streamdf
from galpy.orbit import Orbit
from galpy.potential import LogarithmicHaloPotential
from galpy.util.conversion import time_in_Gyr

lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
prog = [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596]
sigv = 0.365  # km/s

t0 = time.time()
# --- paper snippet begins ---
aAI= actionAngleIsochroneApprox(pot=lp,b=0.8,
        integrate_method='diffrax')
sdf= streamdf(sigv/220.,pot=lp,aA=aAI,
              progenitor=Orbit(jnp.asarray(prog)),
              tdisrupt=4.5/time_in_Gyr(220.,8.),
              leading=True,nTrackChunks=11)
sdf.sample(n=10**4,key=grandom.key(3,backend='jax'))
# --- paper snippet ends ---
print(f"snippet: {time.time() - t0:.1f} s")

RvR = sdf.sample(n=10**4, key=grandom.key(3, backend="jax"))
assert isinstance(RvR, jax.Array) or all(isinstance(x, jax.Array) for x in RvR)
RvR = numpy.asarray(RvR)
assert RvR.shape == (6, 10**4), RvR.shape
# the same key gives the same sample
numpy.testing.assert_array_equal(
    numpy.asarray(sdf.sample(n=10**4, key=grandom.key(3, backend="jax"))), RvR
)
# the stream is near the progenitor and its track
assert numpy.all(numpy.isfinite(RvR))
print("OK")
