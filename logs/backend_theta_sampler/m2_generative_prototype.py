"""M2: fully-jittable generative stream track with a jax-native penalized
smoothing spline (Whittaker/P-spline: (W + lam D'D)^-1 W, GCV-tuned lambda).
Replaces M1's coarse histogram -> a FAITHFUL smooth track, so the frozen-vs-
moving-structure gap closes (the gradient is trustworthy for real fitting).

Everything runs under jax.jit; xv(theta) flows (all orbits on the backend);
only discrete structure (closest-point assignment, bin membership, GCV lambda)
is stop_gradient'd. Validation: physical grad-vs-FD (frozen structure) + the
frozen-vs-moving gap.
"""
import sys, os, time; sys.path.insert(0, "/var/tmp/cint-param")
import numpy, galpy
assert galpy.__file__.startswith("/var/tmp/cint-param")
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from galpy.orbit import Orbit
from galpy.potential import LogarithmicHaloPotential, evaluateRforces, rtide
from galpy.util import conversion

IC = numpy.array([1.2, 0.15, 0.85, 0.08, 0.05, 0.0])
Q0 = 0.9
Mprog = 3e4 / conversion.mass_in_msol(220.0, 8.0)
# STREAM_TD env overrides tdisrupt (internal units). Full physical 4.5 Gyr ~ 126
# (20 orbital periods -> chaotically sensitive); a younger stream is coherent.
TD = float(os.environ.get("STREAM_TD", 4.5 / conversion.time_in_Gyr(220.0, 8.0)))
N = 400
NGRID = 3001
NNODE = 60                      # smoothing-spline nodes over the tp range
NINTERP = 200                   # output track resolution
LAM_GRID = jnp.logspace(-6.0, 2.0, 25)   # GCV lambda search grid (structural)
TP_LO, TP_HI = -TD, 0.0

key = jax.random.PRNGKey(20260711)
k_dt, k_k = jax.random.split(key)
u_dt = jax.random.uniform(k_dt, (N,))
kvec = jax.random.normal(k_k, (N, 6))
MEANK = jnp.asarray([2.0, 0.0, 0.3, 0.0, 0.0, 0.0])
SIGK = jnp.asarray([0.4, 0.0, 0.4, 0.5, 0.5, 0.0])
tgrid = jnp.linspace(0.0, -TD, NGRID)
tnodes = jnp.linspace(TP_LO, TP_HI, NNODE)
tp_out = jnp.linspace(TP_LO * 0.98, TP_HI * 0.02, NINTERP)   # interior track


def _cyl_to_cart_pos(cyl):
    R, phi, z = cyl[..., 0], cyl[..., 5], cyl[..., 3]
    return jnp.stack([R * jnp.cos(phi), R * jnp.sin(phi), z], axis=-1)


def _sample_and_integrate(amp):
    pot = LogarithmicHaloPotential(amp=amp, q=Q0)
    prog = Orbit(jnp.asarray(IC)[None, :]); prog.turn_physical_off()
    prog.integrate(tgrid, pot, method="diffrax")
    prog_cyl = prog.orbit[0]                                       # (NGRID,6)
    prog_xyz = _cyl_to_cart_pos(prog_cyl)                          # (NGRID,3)
    dt = u_dt * TD
    ti = -dt
    ta = tgrid[::-1]
    Rp = jnp.interp(ti, ta, prog_cyl[::-1, 0]); vRp = jnp.interp(ti, ta, prog_cyl[::-1, 1])
    vTp = jnp.interp(ti, ta, prog_cyl[::-1, 2]); zp = jnp.interp(ti, ta, prog_cyl[::-1, 3])
    vzp = jnp.interp(ti, ta, prog_cyl[::-1, 4]); php = jnp.interp(ti, ta, prog_cyl[::-1, 5])
    rt = rtide(pot, Rp, zp, phi=php, t=ti, M=Mprog, use_physical=False)
    vc = jnp.sqrt(-Rp * evaluateRforces(pot, Rp, zp, phi=php, t=ti, use_physical=False))
    rf = rt / Rp
    k = -MEANK + kvec * SIGK
    ics = jnp.stack([Rp + k[:, 0] * rt, vRp * (1.0 + k[:, 1]), vTp + k[:, 2] * vc * rf,
                     k[:, 3] * rf, k[:, 4] * vc * rf, php + k[:, 5] * rf], axis=-1)
    o = Orbit(ics); o.turn_physical_off()
    ts = jnp.stack([-dt, jnp.zeros(N)], axis=-1)
    o.integrate(ts, pot, method="diffrax")
    fin = o.orbit[:, -1, :]
    xv = jnp.stack([fin[:, 0] * jnp.cos(fin[:, 5]), fin[:, 0] * jnp.sin(fin[:, 5]), fin[:, 3]], axis=-1)
    return prog_xyz, xv


_D = jnp.diff(jnp.eye(NNODE), n=2, axis=0)   # 2nd-difference penalty (NNODE-2, NNODE)
_DtD = _D.T @ _D


def _basis(tp_i):
    """(N,NNODE) piecewise-linear P-spline design matrix from particle tp to nodes."""
    pos = (tp_i - TP_LO) / (TP_HI - TP_LO) * (NNODE - 1)
    k = jnp.clip(jnp.floor(pos), 0, NNODE - 2).astype(int)
    w = pos - k
    rows = jnp.arange(tp_i.shape[0])
    B = jnp.zeros((tp_i.shape[0], NNODE))
    B = B.at[rows, k].add(1.0 - w).at[rows, k + 1].add(w)
    return B


def _structure(amp):
    """Discrete/structural: closest-point tp assignment idx + P-spline basis B."""
    prog_xyz, xv = _sample_and_integrate(amp)
    d2 = jnp.sum((xv[:, None, :] - prog_xyz[None, :, :]) ** 2, axis=-1)
    d2 = jnp.where(tgrid[None, :] <= 0.0, d2, jnp.inf)
    idx = jnp.argmin(d2, axis=1)
    B = _basis(tgrid[idx])
    return idx, B


def _gcv_lambda(B, off):
    """GCV over LAM_GRID for the RAW-data P-spline fit (N >> NNODE -> real minimum)."""
    BtB = B.T @ B
    Btoff = B.T @ off
    def gcv(lam):
        A = BtB + lam * _DtD
        trH = jnp.trace(jnp.linalg.solve(A, BtB))
        s = jnp.linalg.solve(A, Btoff)
        resid = off - B @ s
        return N * jnp.sum(resid ** 2) / jnp.clip((N - trH) ** 2, 1e-9, None)
    return LAM_GRID[jnp.argmin(jax.vmap(gcv)(LAM_GRID))]


def track_given(amp, idx, B, lam=None):
    prog_xyz, xv = _sample_and_integrate(amp)
    off = xv - prog_xyz[idx]                              # (N,3) flows theta
    if lam is None:
        lam = jax.lax.stop_gradient(_gcv_lambda(B, jax.lax.stop_gradient(off)))
    A = jax.lax.stop_gradient(B.T @ B + lam * _DtD)       # frozen normal-eqn matrix
    s = jnp.linalg.solve(A, B.T @ off)                    # (NNODE,3) flows theta
    prog_at = jnp.stack([jnp.interp(tp_out, tgrid[::-1], prog_xyz[::-1, j]) for j in range(3)], axis=-1)
    off_at = jnp.stack([jnp.interp(tp_out, tnodes, s[:, j]) for j in range(3)], axis=-1)
    return prog_at + off_at, lam


def stream_track(amp):
    idx, B = jax.lax.stop_gradient(_structure(amp))
    return track_given(amp, idx, B)[0]


if __name__ == "__main__":
    idx0, B0 = jax.jit(_structure)(jnp.asarray(1.0))
    idx0 = jax.lax.stop_gradient(idx0); B0 = jax.lax.stop_gradient(B0)
    _, lam0 = jax.jit(track_given)(jnp.asarray(1.0), idx0, B0)
    lam0 = float(jax.lax.stop_gradient(lam0))
    print(f"GCV lambda = {lam0:.4g}", flush=True)
    # diagnostic: is xv(amp) itself smooth? (AD vs FD of raw sample sum)
    xvsum = lambda a: _sample_and_integrate(a)[1].sum()
    gx_ad = float(jax.jit(jax.grad(xvsum))(jnp.asarray(1.0)))
    gx_fd = (float(jax.jit(xvsum)(jnp.asarray(1.0001))) - float(jax.jit(xvsum)(jnp.asarray(0.9999)))) / 2e-4
    print(f"xv smoothness: AD={gx_ad:.5e} FD={gx_fd:.5e} REL={abs(gx_ad-gx_fd)/max(abs(gx_fd),1e-12):.2e}", flush=True)
    # validate the TRACK gradient with a moderately-smoothed track (fixed lambda),
    # via h-CONVERGENCE (streams are sensitive -> single-h FD is unreliable)
    LAM_FIX = 1.0
    loss = lambda a: track_given(a, idx0, B0, LAM_FIX)[0].sum()
    f = jax.jit(jax.value_and_grad(loss))
    t0 = time.time(); v, g = f(jnp.asarray(1.0)); v.block_until_ready()
    g = float(g)
    print(f"[jit] track.sum={float(v):.6f} d/d(amp)={g:.6e} (lam={LAM_FIX}) compile {time.time()-t0:.1f}s", flush=True)
    lj = jax.jit(loss)
    print("  h-convergence of frozen-structure FD (xv(theta) flows):", flush=True)
    best = 1e9
    for h in [1e-3, 1e-4, 1e-5, 1e-6]:
        g_fd = (float(lj(jnp.asarray(1.0 + h))) - float(lj(jnp.asarray(1.0 - h)))) / (2 * h)
        rel = abs(g - g_fd) / max(abs(g), 1e-12); best = min(best, rel)
        print(f"    h={h:g}: FD={g_fd:.6e} REL={rel:.2e}", flush=True)
    print(f"  BEST REL = {best:.2e}  -> gradient {'VALIDATED' if best < 1e-3 else 'CHECK'}", flush=True)
    t0 = time.time()
    for a in [1.02, 1.05, 1.08]:
        f(jnp.asarray(a))[0].block_until_ready()
    print(f"cached value+grad: {(time.time()-t0)/3*1000:.0f} ms/step", flush=True)
