"""Physics-informed neural inverse of the Staeckel action-angle map for
MWPotential2014 in the meridional plane at fixed L_z -- a map that has *no
analytic inverse*. Trained by backpropagating through galpy's differentiable
(jittable, c=False) Staeckel forward AA. Validated by round-trip and by
reconstructing integrated orbits.

Forward  F: (R, vR, z, vz) -> (J_R, J_z, theta_R, theta_z)   [vT = L_z/R, phi=0]
Inverse  G_phi: (J_R, J_z, theta_R, theta_z) -> (R, vR, z, vz)
"""

import os
import pickle
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

# output box sized to contain the FULL high-action tori (J_R,J_z up to ~0.2 at
# L_z=1 reach R~1.9, |vR|~0.6, |z|~0.8, |vz|~0.6); an undersized box clips them
# and biases the recovered action low.
CEN = jnp.array([1.0, 0.0, 0.0, 0.0])
SCL = jnp.array([0.95, 0.72, 0.92, 0.72])
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


CHUNK = 512  # = the training batch
TIMINGS = {}  # compile / phase wall times (s), recorded in the dump
_FWD = {}


def _forward_compiled(chunk):
    if chunk not in _FWD:
        t = time.time()
        a = jax.ShapeDtypeStruct((chunk,), jnp.float64)
        _FWD[chunk] = jax.jit(forward).lower(a, a, a, a).compile()
        TIMINGS[f"compile_forward_{chunk}"] = time.time() - t
    return _FWD[chunk]


def forward_chunked(R, vR, z, vz, chunk=CHUNK):
    """jit(forward) in fixed-size chunks (last one edge-padded), so ONE compiled
    executable serves every input size (each new shape recompiles, ~minutes).
    Returns numpy arrays. Jitted: an unbound input gives the 9999.99 sentinel /
    NaN instead of raising UnboundError as the eager forward does."""
    f = _forward_compiled(chunk)
    xs = [numpy.asarray(a, dtype=numpy.float64) for a in (R, vR, z, vz)]
    n = xs[0].shape[0]
    outs = [[], [], [], []]
    for i0 in range(0, n, chunk):
        c = [x[i0:i0 + chunk] for x in xs]
        m = c[0].shape[0]
        if m < chunk:
            c = [numpy.pad(x, (0, chunk - m), mode="edge") for x in c]
        for o, r in zip(outs, f(*(jnp.asarray(x) for x in c))):
            o.append(numpy.asarray(r)[:m])
    return tuple(numpy.concatenate(o) for o in outs)


def gen_dataset(n, seed=0):
    rng = numpy.random.RandomState(seed)
    sc = numpy.concatenate(
        [numpy.full(n // 3, s) for s in (1.0, 0.55, 0.28)]
    )[:n]
    sc = numpy.concatenate([sc, numpy.ones(n - sc.shape[0])])
    R = jnp.asarray(1.0 + sc * 0.9 * (2 * rng.rand(n) - 1))
    vR = jnp.asarray(sc * 0.7 * (2 * rng.rand(n) - 1))
    z = jnp.asarray(sc * 0.9 * (2 * rng.rand(n) - 1))
    vz = jnp.asarray(sc * 0.7 * (2 * rng.rand(n) - 1))
    JR, Jz, thR, thz = forward_chunked(R, vR, z, vz)
    m = (JR > J_LO) & (JR < J_HI) & (Jz > J_LO) & (Jz < J_HI)
    g = lambda a: jnp.asarray(numpy.asarray(a)[m])
    return g(R), g(vR), g(z), g(vz), g(JR), g(Jz), g(thR), g(thz)


def loss_fn(p, model, R, vR, z, vz, JR, Jz, thR, thz, lam):
    pred = model.apply(p, feats(JR, Jz, thR, thz))
    # action-weight ~1/sqrt(JR+Jz): up-weights small (near-circular / near-planar)
    # orbits without crushing the high-action end (a full 1/(JR+Jz) gives the
    # small tori ~14x the weight and leaves the big tori under-fit / biased).
    w = 1.0 / jnp.sqrt(JR + Jz + 0.02)
    sup = jnp.sum(w * jnp.sum((pred - jnp.stack([R, vR, z, vz], -1)) ** 2, -1)) / jnp.sum(w)
    JRh, Jzh, thRh, thzh = forward(pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3])
    # per-sample cycle residual, masked where the Staeckel forward is non-finite
    # (a degenerate/unbound orbit the net briefly proposed); the mask keeps the
    # loss finite and, with optax.zero_nans as a backstop, the update stable.
    res = ((JRh - JR) ** 2 + (Jzh - Jz) ** 2
           + (1 - jnp.cos(thRh - thR)) + (1 - jnp.cos(thzh - thz)))
    # Mask samples where the net briefly proposed an unbound/degenerate orbit:
    # the Staeckel forward returns a huge finite "9999.99" sentinel (survives an
    # isfinite check) or a NaN angle -- either would blow up / diverge the cycle
    # gradient. The larger output box makes this reachable (the old small box
    # confined G to bound orbits). Real actions are < 0.2 here.
    ok = jnp.isfinite(res) & (JRh < 1.0) & (Jzh < 1.0)
    cyc = jnp.sum(jnp.where(ok, w * res, 0.0)) / (jnp.sum(jnp.where(ok, w, 0.0)) + 1e-6)
    return sup + lam * cyc, (sup, cyc)


def _save_ckpt(path, **c):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(jax.device_get(c), f)
    os.replace(tmp, path)  # atomic: a kill mid-write leaves the previous checkpoint


def train(n_steps=3200, batch=512, lr=2e-3, warmup=400, seed=0, n_data=40000,
          ckpt=None, ckpt_every=200, resume=False, log_every=50, stop_after=None):
    """Train G. With ckpt, save (params, optimizer state, RNG key, step, history)
    every ckpt_every steps; resume=True continues from it (bit-identical to an
    uninterrupted run). stop_after: end this run after that many steps (tests)."""
    hp = dict(n_steps=n_steps, batch=batch, lr=lr, warmup=warmup, seed=seed, n_data=n_data)
    t = time.time()
    data = gen_dataset(n_data, seed=seed)
    TIMINGS["dataset_seconds"] = time.time() - t  # incl. the forward compile
    nd = int(data[0].shape[0])
    print(f"dataset {nd}, JR<={float(data[4].max()):.3f} Jz<={float(data[5].max()):.3f} "
          f"[{TIMINGS['dataset_seconds']:.0f}s]", flush=True)
    model = InverseMLP()
    params = model.init(jax.random.PRNGKey(0), jnp.zeros((1, 10)))
    sched = optax.cosine_decay_schedule(lr, n_steps, alpha=0.04)
    # zero_nans backstop: a NaN Staeckel gradient from a briefly-degenerate
    # proposed orbit is replaced by 0 instead of poisoning the whole update.
    opt = optax.chain(optax.zero_nans(), optax.clip_by_global_norm(1.0),
                      optax.adam(sched))
    st = opt.init(params)
    key = jax.random.PRNGKey(1)
    hist, i0, train_s, n_runs = [], 0, 0.0, 1
    if resume and ckpt and os.path.exists(ckpt):
        with open(ckpt, "rb") as f:
            c = pickle.load(f)
        if c["hp"] != hp:
            raise SystemExit(f"checkpoint {ckpt} has {c['hp']}, this run {hp}")
        params, st, key = (jax.tree.map(jnp.asarray, c[k]) for k in ("params", "opt_state", "key"))
        hist, i0, train_s, n_runs = list(c["hist"]), c["step"], c["train_seconds"], c["n_runs"] + 1
        print(f"resumed from {ckpt} at step {i0} ({train_s:.0f}s trained)", flush=True)
    TIMINGS["resumed_from_step"] = i0

    # data passed as an argument (not closed over: no 40000-row HLO constants)
    def step(p, st, data, idx, lam):
        b = [a[idx] for a in data]
        (l, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(p, model, *b, lam)
        u, st = opt.update(g, st)
        return optax.apply_updates(p, u), st, l, aux

    t = time.time()
    ex_idx = jax.random.randint(jax.random.PRNGKey(0), (batch,), 0, nd)
    step_c = jax.jit(step).lower(params, st, data, ex_idx, jnp.asarray(0.0)).compile()
    TIMINGS["compile_step"] = time.time() - t
    print(f"train step compiled in {TIMINGS['compile_step']:.0f}s", flush=True)

    t0, last = time.time(), i0
    for i in range(i0, n_steps):
        key, k = jax.random.split(key)
        idx = jax.random.randint(k, (batch,), 0, nd)
        # supervised warmup: ramp the physics-cycle weight 0 -> 1 so the net
        # first learns a rough (bounded) inverse before the cycle term engages.
        lam = jnp.asarray(min(1.0, i / float(warmup)))
        params, st, l, (s, c) = step_c(params, st, data, idx, lam)
        if i % 150 == 0 or i == n_steps - 1 or n_steps <= 50:
            hist.append((i, float(l), float(s), float(c)))
        done = i + 1
        if done % log_every == 0 or i == i0 or done == n_steps or n_steps <= 50:
            el = time.time() - t0
            print(f"step {i:4d} loss={float(l):.3e} sup={float(s):.3e} cyc={float(c):.3e} "
                  f"[{train_s + el:.0f}s trained, {el / (done - i0):.3f} s/step this run]", flush=True)
        stop = stop_after is not None and done - i0 >= stop_after
        if ckpt and (done % ckpt_every == 0 or done == n_steps or stop):
            _save_ckpt(ckpt, params=params, opt_state=st, key=key, hist=hist, step=done,
                       train_seconds=train_s + time.time() - t0, hp=hp, n_runs=n_runs)
        if stop:
            raise SystemExit(f"stopped after {done - i0} steps at step {done} (stop_after)")
    train_s += time.time() - t0
    TIMINGS["n_runs"] = n_runs
    return params, model, hist, train_s
