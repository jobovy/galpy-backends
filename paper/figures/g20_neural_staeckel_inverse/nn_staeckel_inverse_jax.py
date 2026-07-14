"""Physics-informed neural inverse of the Staeckel action-angle map for
MWPotential2014 in the meridional plane at fixed L_z -- a map that has *no
analytic inverse*. Trained by backpropagating through galpy's differentiable
(jittable, c=False) Staeckel forward AA. Validated by round-trip and by
reconstructing integrated orbits.

Forward  F: (R, vR, z, vz) -> (J_R, J_z, theta_R, theta_z)   [vT = L_z/R, phi=0]
Inverse  G_phi: (J_R, J_z, theta_R, theta_z) -> (R, vR, z, vz)
"""

import time

import numpy

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import flax.linen as nn
import optax

from galpy.actionAngle import actionAngleStaeckel, estimateDeltaStaeckel
from galpy.potential import MWPotential2014

LZ = 1.0
DELTA = float(
    estimateDeltaStaeckel(
        MWPotential2014, numpy.linspace(0.5, 1.5, 15), numpy.linspace(0.0, 0.4, 15)
    )
)
_aAS = actionAngleStaeckel(pot=MWPotential2014, delta=DELTA, c=False)

CEN = jnp.array([1.0, 0.0, 0.0, 0.0])
SCL = jnp.array([0.45, 0.32, 0.45, 0.32])
J_LO, J_HI = 1e-3, 0.2


def forward(R, vR, z, vz):
    vT = LZ / R
    phi = jnp.zeros_like(R)
    o = _aAS.actionsFreqsAngles(R, vR, vT, z, vz, phi)
    return o[0], o[2], o[6], o[8]  # JR, Jz, thetaR, thetaz


class InverseMLP(nn.Module):
    width: int = 192

    @nn.compact
    def __call__(self, x):  # feats(JR,Jz,thR,thz) -> (R,vR,z,vz)
        h = x
        for _ in range(4):
            h = nn.tanh(nn.Dense(self.width)(h))
        return CEN + SCL * jnp.tanh(nn.Dense(4)(h))


def feats(JR, Jz, thR, thz):
    # raw actions + bare angle (cos,sin) + harmonic amplitude coords
    # sqrt(J)*(cos,sin) for each DOF (the natural small-action coordinate).
    sJR = jnp.sqrt(jnp.clip(JR, 0.0))
    sJz = jnp.sqrt(jnp.clip(Jz, 0.0))
    cR, sR, cZ, sZ = jnp.cos(thR), jnp.sin(thR), jnp.cos(thz), jnp.sin(thz)
    return jnp.stack(
        [JR, Jz, cR, sR, cZ, sZ, sJR * cR, sJR * sR, sJz * cZ, sJz * sZ], axis=-1
    )


def gen_dataset(n, seed=0):
    rng = numpy.random.RandomState(seed)
    sc = numpy.concatenate(
        [numpy.full(n // 3, s) for s in (1.0, 0.55, 0.28)]
    )[:n]
    sc = numpy.concatenate([sc, numpy.ones(n - sc.shape[0])])
    R = jnp.asarray(1.0 + sc * 0.45 * (2 * rng.rand(n) - 1))
    vR = jnp.asarray(sc * 0.32 * (2 * rng.rand(n) - 1))
    z = jnp.asarray(sc * 0.45 * (2 * rng.rand(n) - 1))
    vz = jnp.asarray(sc * 0.32 * (2 * rng.rand(n) - 1))
    JR, Jz, thR, thz = jax.jit(forward)(R, vR, z, vz)
    JR = numpy.asarray(JR); Jz = numpy.asarray(Jz)
    m = (JR > J_LO) & (JR < J_HI) & (Jz > J_LO) & (Jz < J_HI)
    g = lambda a: jnp.asarray(numpy.asarray(a)[m])
    return g(R), g(vR), g(z), g(vz), g(JR), g(Jz), g(thR), g(thz)


def loss_fn(p, model, R, vR, z, vz, JR, Jz, thR, thz, lam):
    pred = model.apply(p, feats(JR, Jz, thR, thz))
    # action-weight ~1/(JR+Jz) so small (near-circular / near-planar) orbits are
    # fit as precisely as the big ones (unweighted MSE favours the big orbits).
    w = 1.0 / (JR + Jz + 0.02)
    sup = jnp.sum(w * jnp.sum((pred - jnp.stack([R, vR, z, vz], -1)) ** 2, -1)) / jnp.sum(w)
    JRh, Jzh, thRh, thzh = forward(pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3])
    # per-sample cycle residual, masked where the Staeckel forward is non-finite
    # (a degenerate/unbound orbit the net briefly proposed); the mask keeps the
    # loss finite and, with optax.zero_nans as a backstop, the update stable.
    res = ((JRh - JR) ** 2 + (Jzh - Jz) ** 2
           + (1 - jnp.cos(thRh - thR)) + (1 - jnp.cos(thzh - thz)))
    ok = jnp.isfinite(res)
    cyc = jnp.sum(jnp.where(ok, w * res, 0.0)) / (jnp.sum(jnp.where(ok, w, 0.0)) + 1e-6)
    return sup + lam * cyc, (sup, cyc)


def train(n_steps=2600, batch=512, lr=2e-3, warmup=400, seed=0):
    R, vR, z, vz, JR, Jz, thR, thz = gen_dataset(40000, seed=seed)
    nd = int(R.shape[0])
    print(f"dataset {nd}, JR<={float(JR.max()):.3f} Jz<={float(Jz.max()):.3f}")
    model = InverseMLP()
    params = model.init(jax.random.PRNGKey(0), jnp.zeros((1, 10)))
    sched = optax.cosine_decay_schedule(lr, n_steps, alpha=0.04)
    # zero_nans backstop: a NaN Staeckel gradient from a briefly-degenerate
    # proposed orbit is replaced by 0 instead of poisoning the whole update.
    opt = optax.chain(optax.zero_nans(), optax.clip_by_global_norm(1.0),
                      optax.adam(sched))
    st = opt.init(params)

    @jax.jit
    def step(p, st, idx, lam):
        b = [a[idx] for a in (R, vR, z, vz, JR, Jz, thR, thz)]
        (l, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(p, model, *b, lam)
        u, st = opt.update(g, st)
        return optax.apply_updates(p, u), st, l, aux

    key = jax.random.PRNGKey(1)
    hist = []
    t0 = time.time()
    for i in range(n_steps):
        key, k = jax.random.split(key)
        idx = jax.random.randint(k, (batch,), 0, nd)
        # supervised warmup: ramp the physics-cycle weight 0 -> 1 so the net
        # first learns a rough (bounded) inverse before the cycle term engages.
        lam = jnp.asarray(min(1.0, i / float(warmup)))
        params, st, l, (s, c) = step(params, st, idx, lam)
        if i % 150 == 0 or i == n_steps - 1:
            hist.append((i, float(l), float(s), float(c)))
            print(f"step {i:4d} loss={float(l):.3e} sup={float(s):.3e} cyc={float(c):.3e} "
                  f"[{time.time()-t0:.0f}s]")
    return params, model, hist, time.time() - t0
