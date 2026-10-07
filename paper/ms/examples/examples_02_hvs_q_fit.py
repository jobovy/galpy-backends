"""Paper snippet: sections/examples.tex, Sec. 10.1 (A hypervelocity star as a
probe of the potential), #2.

Claim: the flattening q of the halo is fit by descending the closest
approach with the derivative obtained by reverse-mode differentiation
through the in-backend orbit integration (compiled with jax.jit); the fit
recovers the q from which the mock star was generated.
"""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy
import optax

from galpy.orbit import Orbit
from galpy.potential import LogarithmicHaloPotential

q_true = 0.85
ts = numpy.linspace(0.0, 0.5, 151)  # ~18 Myr
launch = Orbit([0.004, 2.4, 0.03, 0.002, 1.5, 0.6], ro=8.0, vo=220.0)
launch.integrate(
    ts, LogarithmicHaloPotential(normalize=1.0, core=0.2, q=q_true),
    method="dop853_c", rtol=1e-13, atol=1e-13,
)
vxvv = [
    float(getattr(launch, f)(ts[-1]))
    for f in ("ra", "dec", "dist", "pmra", "pmdec", "vlos")
]


def adam_descent(objective, q0, nsteps=60, lr=0.02):
    opt = optax.adam(lr)
    q = jnp.asarray(q0)
    state = opt.init(q)
    for _ in range(nsteps):
        _, g = objective(q)
        updates, state = opt.update(g, state)
        q = optax.apply_updates(q, updates)
    return q


# --- paper snippet begins ---
def closest_approach(q):
    pot= LogarithmicHaloPotential(normalize=1.,
                                  core=0.2,q=q)
    o= Orbit(jnp.asarray(vxvv),radec=True,ro=8.,vo=220.)
    o.integrate(-ts,pot,method='diffrax')
    return jnp.min(o.r(-ts))
objective= jax.jit(jax.value_and_grad(closest_approach))
q= adam_descent(objective,q0=1.1) # a few lines of optax
# --- paper snippet ends ---

# the derivative matches a finite difference of the C orbit
def closest_np(qq):
    o = Orbit(vxvv, radec=True, ro=8.0, vo=220.0)
    o.integrate(-ts, LogarithmicHaloPotential(normalize=1.0, core=0.2, q=qq),
                method="dop853_c", rtol=1e-13, atol=1e-13)
    return numpy.min(o.r(-ts))


v, g = objective(1.0)
numpy.testing.assert_allclose(float(v), closest_np(1.0), rtol=1e-8)
h = 1e-4
fd = (
    -closest_np(1.0 + 2 * h) + 8 * closest_np(1.0 + h)
    - 8 * closest_np(1.0 - h) + closest_np(1.0 - 2 * h)
) / (12 * h)
print(float(g), fd)
numpy.testing.assert_allclose(float(g), fd, rtol=1e-6)
print("recovered q:", float(q), "closest approach:", float(objective(q)[0]))
assert abs(float(q) - q_true) < 0.01, float(q)
print("OK")
