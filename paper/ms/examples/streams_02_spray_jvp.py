"""Paper snippet: sections/streams.tex, Sec. 8.2 (Particle-spray models), #1.

Claims: sample() of a particle-spray model is a differentiable function of
the potential (for a fixed key); the forward-mode derivative gives the change
of every particle's position for a unit change in the halo flattening and
agrees with a finite difference of the same sample (same random numbers).

NOTE: before galpy PR #PRNUM, forward-mode differentiation (jax.jvp)
through diffrax's default (checkpointed) adjoint raised "can't apply
forward-mode autodiff (jvp) to a custom_vjp function", so the snippet passes
integrate_kwargs={'adjoint':'direct'}; and with a numpy progenitor
(Orbit(ic) from a list) the derivative disagreed with a finite difference at
7e-5 -- not because the progenitor escaped the differentiable path, but
because the backend sampler splined the progenitor on a coarser grid than
the numpy sampler the finite difference of plain calls runs (~1e-7 off in
velocity, ~7e-6 in the stream). With the PR galpy picks the direct adjoint
under jvp/jacfwd itself and the two samplers agree to ~1e-8, so both can be
simplified back once it is merged (checked below).
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
q0 = 0.9

t0 = time.time()
# --- paper snippet begins ---
key= galpy.backend.random.key(1,backend='jax')
def leading_arm(q):
    lp= LogarithmicHaloPotential(normalize=1.,q=q)
    prog= Orbit(jnp.asarray(ic))
    spdf= fardal15spraydf(mass,progenitor=prog,
        pot=lp,tdisrupt=td,
        integrate_kwargs={'adjoint':'direct'})
    return spdf.sample(n=1500,return_orbit=False,
                       tail='leading',key=key)
xv,dxv= jax.jvp(leading_arm,(q0,),(1.,))
# --- paper snippet ends ---
print(f"snippet: {time.time() - t0:.1f} s")

xv, dxv = jnp.asarray(xv), jnp.asarray(dxv)
assert xv.shape == (6, 1500) and dxv.shape == (6, 1500)
numpy.testing.assert_allclose(
    numpy.asarray(leading_arm(q0)), numpy.asarray(xv), rtol=1e-8, atol=1e-8
)
# central finite difference of the same sample (common random numbers)
h = 2e-4  # five-point stencil


def arm(q):
    return numpy.asarray(leading_arm(q))


fd = (-arm(q0 + 2 * h) + 8 * arm(q0 + h) - 8 * arm(q0 - h) + arm(q0 - 2 * h)) / (
    12 * h
)
ad = numpy.asarray(dxv)
# phi (row 5) is an angle: compare differences, which do not wrap here
err = numpy.max(numpy.abs(ad - fd)) / numpy.max(numpy.abs(ad))
print("max |AD - FD| / max |AD|:", err)
assert err < 1e-6, err


# galpy PR #PRNUM: a numpy progenitor and no integrate_kwargs give the same
# forward-mode derivative, which matches a finite difference of plain calls
def leading_arm_np(q):
    lp = LogarithmicHaloPotential(normalize=1.0, q=q)
    spdf = fardal15spraydf(mass, progenitor=Orbit(ic), pot=lp, tdisrupt=td)
    return spdf.sample(n=1500, return_orbit=False, tail="leading", key=key)


xv2, dxv2 = jax.jvp(leading_arm_np, (q0,), (1.0,))
err2 = numpy.max(numpy.abs(numpy.asarray(dxv2) - ad)) / numpy.max(numpy.abs(ad))
print("numpy progenitor, default adjoint vs the snippet:", err2)
assert err2 < 1e-10, err2


def arm_np(q):
    return numpy.asarray(leading_arm_np(q))


fd_np = (
    -arm_np(q0 + 2 * h) + 8 * arm_np(q0 + h) - 8 * arm_np(q0 - h) + arm_np(q0 - 2 * h)
) / (12 * h)
err_np = numpy.max(numpy.abs(ad - fd_np)) / numpy.max(numpy.abs(ad))
print("max |AD - FD(numpy progenitor)| / max |AD|:", err_np)
assert err_np < 1e-6, err_np
print("OK")
