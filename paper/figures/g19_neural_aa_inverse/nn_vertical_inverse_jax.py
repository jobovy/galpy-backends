"""Physics-informed neural inverse of the 1D (anharmonic) vertical action-angle
map, trained by backpropagating through galpy's *differentiable* forward AA.

Forward (galpy, jittable, autodiff-native):  F: (z, vz) -> (J, theta)
NN inverse we train:                          G_phi: (J, theta) -> (z, vz)

Loss = supervised + physics-informed cycle-consistency:
  L = ||G(J,th) - (z,vz)||^2                         (data term; (z,vz) sampled,
                                                       (J,th)=F(z,vz) exact)
    + lambda * [ (F_J(G(J,th)) - J)^2 + (1-cos(F_th(G(J,th)) - th)) ]
                                                      (cycle term -- gradient
                                                       flows THROUGH galpy's
                                                       differentiable forward AA)

The data term keeps G on the bound tori (stable); the cycle term is the autodiff
showcase and drives F(G) = identity. Reference (numpy, never in the graph):
actionAngleVerticalInverse.
"""

import os

import numpy

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import flax.linen as nn
import optax

from galpy.actionAngle import actionAngleVertical
from galpy.potential import MWPotential2014, toVerticalPotential

R0 = 1.0
_vpot = toVerticalPotential(MWPotential2014, R0)
_aAV = actionAngleVertical(pot=_vpot)


def forward_Jtheta(z, vz):
    """(z,vz) -> (J, theta) via galpy; jittable, differentiable, vectorized."""
    J, _Omega, theta = _aAV._actionsFreqsAngles(z, vz)
    return J, theta


# Output box (keeps G on bound orbits; set from the data below).
Z_BOX, VZ_BOX = 1.6, 0.95


def feats(J, th):
    """Input features: raw action + harmonic amplitude coords sqrt(J)*(cos,sin)
    (the natural small-J coordinate) + the bare angle (cos,sin)."""
    sj = jnp.sqrt(jnp.clip(J, 0.0))
    c, s = jnp.cos(th), jnp.sin(th)
    return jnp.stack([J, sj * c, sj * s, c, s], axis=-1)


class InverseMLP(nn.Module):
    width: int = 128

    @nn.compact
    def __call__(self, x):  # (J, sqrt(J)cos, sqrt(J)sin, cos, sin) -> (z, vz)
        h = x
        for _ in range(3):
            h = nn.tanh(nn.Dense(self.width)(h))
        raw = nn.Dense(2)(h)
        box = jnp.array([Z_BOX, VZ_BOX])
        return box * jnp.tanh(raw)


def gen_dataset(n, Jlo, Jhi, seed=0):
    """Sample (z,vz) in a mix of scales (so small/low-J orbits are covered too),
    forward to (J,theta); keep the target J-range."""
    rng = numpy.random.RandomState(seed)
    # three nested boxes -> roughly uniform coverage in action, not just area
    scales = numpy.concatenate(
        [numpy.full(n // 3, s) for s in (1.0, 0.45, 0.2)]
    )[: n]
    scales = numpy.concatenate([scales, numpy.ones(n - scales.shape[0])])
    z = jnp.asarray(scales * Z_BOX * (2 * rng.rand(n) - 1))
    vz = jnp.asarray(scales * VZ_BOX * (2 * rng.rand(n) - 1))
    J, th = jax.jit(forward_Jtheta)(z, vz)
    J = numpy.asarray(J)
    m = (J > Jlo) & (J < Jhi)
    return (
        jnp.asarray(numpy.asarray(z)[m]),
        jnp.asarray(numpy.asarray(vz)[m]),
        jnp.asarray(J[m]),
        jnp.asarray(numpy.asarray(th)[m]),
    )


def loss_fn(params, model, z, vz, J, th, lam):
    pred = model.apply(params, feats(J, th))
    # weight ~1/J so small (tiny-amplitude) tori are fit as precisely as big
    # ones -- an unweighted MSE is dominated by the large orbits.
    w = 1.0 / (J + 0.015)
    se = (pred[:, 0] - z) ** 2 + (pred[:, 1] - vz) ** 2
    sup = jnp.sum(w * se) / jnp.sum(w)
    Jh, thh = forward_Jtheta(pred[:, 0], pred[:, 1])
    cyc_J = jnp.mean(((Jh - J) / (J + 0.015)) ** 2)
    cyc_a = jnp.mean(1.0 - jnp.cos(thh - th))
    return sup + lam * (cyc_J + cyc_a), (sup, cyc_J, cyc_a)


def main(n_steps=1200, batch=512, lr=2e-3, Jlo=0.008, Jhi=0.14, lam=1.0,
         seed=0, width=128):
    z, vz, J, th = gen_dataset(60000, Jlo, Jhi, seed=seed)
    ndat = int(z.shape[0])
    print(f"dataset: {ndat} bound samples, J in [{float(J.min()):.3f},{float(J.max()):.3f}]")

    key = jax.random.PRNGKey(seed)
    model = InverseMLP(width=width)
    key, k0 = jax.random.split(key)
    params = model.init(k0, jnp.zeros((1, 5)))
    sched = optax.cosine_decay_schedule(lr, n_steps, alpha=0.05)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(sched))
    state = opt.init(params)

    @jax.jit
    def step(params, state, idx):
        b = (z[idx], vz[idx], J[idx], th[idx])
        (l, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(
            params, model, *b, lam
        )
        updates, state = opt.update(g, state)
        params = optax.apply_updates(params, updates)
        return params, state, l, aux

    hist = []
    for i in range(n_steps):
        key, kb = jax.random.split(key)
        idx = jax.random.randint(kb, (batch,), 0, ndat)
        params, state, l, (s, cJ, ca) = step(params, state, idx)
        if i % 250 == 0 or i == n_steps - 1:
            hist.append((i, float(l), float(s), float(cJ), float(ca)))
            print(
                f"step {i:4d}  loss={float(l):.3e}  sup={float(s):.3e}  "
                f"cyc_J={float(cJ):.3e}  cyc_ang={float(ca):.3e}"
            )
    return params, model, hist


def validate_vs_numpy(params, model, Es=(0.02, 0.05, 0.10), n_ang=64):
    from galpy.actionAngle import actionAngleVerticalInverse

    aAVi = actionAngleVerticalInverse(
        pot=_vpot, Es=list(numpy.linspace(0.005, 0.16, 12)), nta=128, setup_interp=True
    )
    th = numpy.linspace(0.0, 2.0 * numpy.pi, n_ang, endpoint=False)
    out = {}
    for E in Es:
        J = float(aAVi.J(E))
        z_ref, vz_ref = aAVi(numpy.array([J] * n_ang), th)
        zv = numpy.asarray(
            model.apply(params, feats(jnp.full((n_ang,), J), jnp.asarray(th)))
        )
        rms = float(
            numpy.sqrt(numpy.mean((zv[:, 0] - z_ref) ** 2 + (zv[:, 1] - vz_ref) ** 2))
        )
        out[E] = (J, rms)
    return out


if __name__ == "__main__":
    params, model, hist = main()
    print("\n=== validation vs numpy actionAngleVerticalInverse (RMS |dz,dvz|) ===")
    for E, (J, rms) in validate_vs_numpy(params, model).items():
        print(f"  E={E:.3f}  J={J:.4f}  RMS(z,vz) inverse error = {rms:.3e}")
