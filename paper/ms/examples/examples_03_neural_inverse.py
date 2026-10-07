"""Paper snippet: sections/examples.tex, Sec. 10.2 (Learning the inverse
action-angle transformation).

Claim: the cycle-consistency loss of a network inverse G(J,theta) -> (z,v_z)
through galpy's forward action-angle map is differentiable w.r.t. the
network's weights, and its gradient is computed (and compiled) by jax.
"""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy

from galpy.actionAngle import actionAngleVertical
from galpy.potential import MWPotential2014, toVerticalPotential

# the vertical motion in MWPotential2014 at the solar radius
aAV = actionAngleVertical(pot=toVerticalPotential(MWPotential2014, 1.0))


class MLP:
    """A small multi-layer perceptron (J,theta) -> (z,v_z)."""

    def init(self, key, width=16):
        k1, k2, k3 = jax.random.split(key, 3)
        return [
            (0.5 * jax.random.normal(k1, (3, width)), jnp.zeros(width)),
            (0.5 * jax.random.normal(k2, (width, width)), jnp.zeros(width)),
            (0.1 * jax.random.normal(k3, (width, 2)), jnp.zeros(2)),
        ]

    def apply(self, params, J, theta):
        sj = jnp.sqrt(J)
        h = jnp.stack([J, sj * jnp.cos(theta), sj * jnp.sin(theta)], axis=-1)
        for W, b in params[:-1]:
            h = jnp.tanh(h @ W + b)
        W, b = params[-1]
        out = 0.5 * jnp.tanh(h @ W + b)  # stay on bound orbits
        return out[..., 0], out[..., 1]


def wrap(dtheta):
    return (dtheta + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


net = MLP()
params = net.init(jax.random.key(0))
rng = numpy.random.default_rng(1)
J = jnp.asarray(rng.uniform(0.01, 0.05, 64))
theta = jnp.asarray(rng.uniform(0.0, 2.0 * numpy.pi, 64))

# --- paper snippet begins ---
def loss(params,J,theta):
    z,vz= net.apply(params,J,theta) # the network
    Jp,_,thetap= aAV.actionsFreqsAngles(z,vz) # galpy
    return (jnp.mean((Jp-J)**2/J**2)
            +jnp.mean(wrap(thetap-theta)**2))
grads= jax.jit(jax.grad(loss))(params,J,theta)
# --- paper snippet ends ---

# the gradient has the structure of the parameters
assert jax.tree.structure(grads) == jax.tree.structure(params)
assert all(isinstance(g, jax.Array) for g in jax.tree.leaves(grads))
# the forward map is galpy's: the actions agree with the numpy path
z, vz = net.apply(params, J, theta)
Jp = aAV.actionsFreqsAngles(z, vz)[0]
Jp_np = actionAngleVertical(pot=toVerticalPotential(MWPotential2014, 1.0))(
    numpy.asarray(z), numpy.asarray(vz)
)
numpy.testing.assert_allclose(numpy.asarray(Jp), Jp_np, rtol=1e-8, atol=2e-11)
# directional derivative vs a central finite difference along a random
# direction in weight space
leaves, tdef = jax.tree.flatten(params)
keys = jax.random.split(jax.random.key(2), len(leaves))
dirs = jax.tree.unflatten(
    tdef, [jax.random.normal(k, x.shape) for k, x in zip(keys, leaves)]
)
ad = sum(
    jnp.sum(g * d) for g, d in zip(jax.tree.leaves(grads), jax.tree.leaves(dirs))
)
h = 1e-4


def shifted(s):
    return jax.tree.map(lambda p, d: p + s * d, params, dirs)


fd = (
    -loss(shifted(2 * h), J, theta) + 8 * loss(shifted(h), J, theta)
    - 8 * loss(shifted(-h), J, theta) + loss(shifted(-2 * h), J, theta)
) / (12 * h)
print(float(ad), float(fd))
# the finite difference bottoms out at ~1e-7 (h=1e-4; 4e-7 at 3e-4, 3e-6 at 3e-5)
numpy.testing.assert_allclose(float(ad), float(fd), rtol=3e-7)
print("OK")
