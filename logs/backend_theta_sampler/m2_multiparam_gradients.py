"""Broaden the proof: the GENERATIVE stream track is differentiable w.r.t. ALL
parameters through the jittable sampler -- halo amp, halo flattening q, AND the
progenitor's initial conditions. Each gradient h-converged (streams are sensitive).
Younger coherent stream (fittable regime)."""
import sys; sys.path.insert(0, "/var/tmp/cint-param")
import numpy
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from galpy.orbit import Orbit
from galpy.potential import LogarithmicHaloPotential, evaluateRforces, rtide
from galpy.util import conversion

IC0 = jnp.asarray([1.2, 0.15, 0.85, 0.08, 0.05, 0.0])
Mprog = 3e4 / conversion.mass_in_msol(220.0, 8.0)
TD = 25.0
N, NGRID, NNODE, NINTERP = 300, 2001, 50, 150
LAM = 1.0
rng = numpy.random.default_rng(20260712)
u_dt = jnp.asarray(rng.uniform(size=N)); kvec = jnp.asarray(rng.normal(size=(N, 6)))
MEANK = jnp.asarray([2.0, 0.0, 0.3, 0.0, 0.0, 0.0]); SIGK = jnp.asarray([0.4, 0.0, 0.4, 0.5, 0.5, 0.0])
tgrid = jnp.linspace(0.0, -TD, NGRID); tnodes = jnp.linspace(-TD, 0.0, NNODE)
tp_out = jnp.linspace(-TD * 0.9, -TD * 0.02, NINTERP)
_D = jnp.diff(jnp.eye(NNODE), n=2, axis=0); _DtD = _D.T @ _D


def sample_and_integrate(amp, q, ic):
    pot = LogarithmicHaloPotential(amp=amp, q=q)
    prog = Orbit(ic[None, :]); prog.turn_physical_off()
    prog.integrate(tgrid, pot, method="diffrax")
    cyl = prog.orbit[0]
    pxyz = jnp.stack([cyl[:, 0] * jnp.cos(cyl[:, 5]), cyl[:, 0] * jnp.sin(cyl[:, 5]), cyl[:, 3]], axis=-1)
    dt = u_dt * TD; ti = -dt; ta = tgrid[::-1]
    Rp = jnp.interp(ti, ta, cyl[::-1, 0]); vRp = jnp.interp(ti, ta, cyl[::-1, 1])
    vTp = jnp.interp(ti, ta, cyl[::-1, 2]); zp = jnp.interp(ti, ta, cyl[::-1, 3])
    php = jnp.interp(ti, ta, cyl[::-1, 5])
    rt = rtide(pot, Rp, zp, phi=php, t=ti, M=Mprog, use_physical=False)
    vc = jnp.sqrt(-Rp * evaluateRforces(pot, Rp, zp, phi=php, t=ti, use_physical=False))
    rf = rt / Rp; k = -MEANK + kvec * SIGK
    ics = jnp.stack([Rp + k[:, 0] * rt, vRp * (1 + k[:, 1]), vTp + k[:, 2] * vc * rf,
                     k[:, 3] * rf, k[:, 4] * vc * rf, php + k[:, 5] * rf], axis=-1)
    o = Orbit(ics); o.turn_physical_off()
    o.integrate(jnp.stack([-dt, jnp.zeros(N)], axis=-1), pot, method="diffrax")
    fin = o.orbit[:, -1, :]
    xv = jnp.stack([fin[:, 0] * jnp.cos(fin[:, 5]), fin[:, 0] * jnp.sin(fin[:, 5]), fin[:, 3]], axis=-1)
    return pxyz, xv


def basis(tp_i):
    pos = (tp_i + TD) / TD * (NNODE - 1)
    kk = jnp.clip(jnp.floor(pos), 0, NNODE - 2).astype(int); w = pos - kk
    r = jnp.arange(tp_i.shape[0])
    return jnp.zeros((tp_i.shape[0], NNODE)).at[r, kk].add(1 - w).at[r, kk + 1].add(w)


def structure(amp, q, ic):
    pxyz, xv = sample_and_integrate(amp, q, ic)
    d2 = jnp.where(tgrid[None, :] <= 0.0, jnp.sum((xv[:, None] - pxyz[None]) ** 2, -1), jnp.inf)
    idx = jnp.argmin(d2, axis=1)
    return idx, basis(tgrid[idx])


def track(theta, idx, B):
    amp, q, ic = theta[0], theta[1], theta[2:]
    pxyz, xv = sample_and_integrate(amp, q, ic)
    off = xv - pxyz[idx]
    A = jax.lax.stop_gradient(B.T @ B + LAM * _DtD)
    s = jnp.linalg.solve(A, B.T @ off)
    pat = jnp.stack([jnp.interp(tp_out, tgrid[::-1], pxyz[::-1, j]) for j in range(3)], axis=-1)
    oat = jnp.stack([jnp.interp(tp_out, tnodes, s[:, j]) for j in range(3)], axis=-1)
    return (pat + oat).sum()


theta0 = jnp.concatenate([jnp.asarray([1.0, 0.9]), IC0])
idx0, B0 = jax.jit(structure)(jnp.asarray(1.0), jnp.asarray(0.9), IC0)
idx0 = jax.lax.stop_gradient(idx0); B0 = jax.lax.stop_gradient(B0)
gfull = jax.jit(jax.grad(lambda th: track(th, idx0, B0)))(theta0)
names = ["amp", "q", "R", "vR", "vT", "z", "vz", "phi"]
lj = jax.jit(lambda th: track(th, idx0, B0))
print("generative d(track.sum)/d(theta), h-converged:", flush=True)
for i, nm in enumerate(names):
    g_ad = float(gfull[i]); best = 1e9; bh = None
    for h in [1e-4, 1e-5, 1e-6]:
        tp = theta0.at[i].add(h); tm = theta0.at[i].add(-h)
        g_fd = (float(lj(tp)) - float(lj(tm))) / (2 * h)
        rel = abs(g_ad - g_fd) / max(abs(g_ad), 1e-9)
        if rel < best: best, bh = rel, h
    print(f"  d/d({nm:3s}): AD={g_ad:+.4e}  bestREL={best:.1e} (h={bh:g})  {'OK' if best<1e-2 else 'CHECK'}", flush=True)
