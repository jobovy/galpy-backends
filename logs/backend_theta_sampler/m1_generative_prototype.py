"""M1: standalone FULLY-JITTABLE generative stream-track prototype.

Proves the PHYSICAL d(track)/d(theta): the spray sample xv(theta) FLOWS (all
orbits integrated on the backend via diffrax), only the structural indices
(closest-point assignment, tp bins) are stop_gradient'd. Everything runs inside
jax.jit. Validation = grad-vs-FD where xv genuinely varies with theta (common
random numbers: the jax.random key is fixed, so the noise is theta-independent
and the derivative is pathwise/reparameterized).

Reuses galpy's jittable leaves: Orbit.integrate(method='diffrax'),
evaluateRforces, rtide, coords. Reconstruction is jax-native.
"""
import sys, time; sys.path.insert(0, "/var/tmp/cint-param")
import numpy, galpy
assert galpy.__file__.startswith("/var/tmp/cint-param")
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from galpy.orbit import Orbit
from galpy.potential import LogarithmicHaloPotential, evaluateRforces, rtide
from galpy.util import conversion, coords

IC = numpy.array([1.2, 0.15, 0.85, 0.08, 0.05, 0.0])
Q0 = 0.9
Mprog = 3e4 / conversion.mass_in_msol(220.0, 8.0)
TD = 4.5 / conversion.time_in_Gyr(220.0, 8.0)
N = 400
NGRID = 4001           # progenitor curve resolution for closest-point + track
NBIN = 40              # tp bins for the offset mean
SMOOTH = 3.0           # fixed Gaussian smoothing bandwidth in bin units (frozen structure)

# reparameterized noise: fixed key -> theta-independent draws (common random numbers)
key = jax.random.PRNGKey(20260711)
k_dt, k_k = jax.random.split(key)
u_dt = jax.random.uniform(k_dt, (N,))            # stripping-time fractions
kvec = jax.random.normal(k_k, (N, 6))            # fardal15 action-angle offsets
MEANK = jnp.asarray([2.0, 0.0, 0.3, 0.0, 0.0, 0.0])
SIGK = jnp.asarray([0.4, 0.0, 0.4, 0.5, 0.5, 0.0])


def _cyl_to_cart_pos(cyl):
    R, phi, z = cyl[..., 0], cyl[..., 5], cyl[..., 3]
    return jnp.stack([R * jnp.cos(phi), R * jnp.sin(phi), z], axis=-1)


def _prog_curve(amp):
    """Progenitor phase-space curve on a dense grid over [-TD, 0], via diffrax.
    Read o.orbit directly (integrated ON tgrid) -- o.x(t) is NOT jittable."""
    pot = LogarithmicHaloPotential(amp=amp, q=Q0)
    o = Orbit(jnp.asarray(IC)[None, :]); o.turn_physical_off()
    tgrid = jnp.linspace(0.0, -TD, NGRID)
    o.integrate(tgrid, pot, method="diffrax")
    cyl = o.orbit[0]                                                   # (NGRID,6) [R,vR,vT,z,vz,phi]
    xyz = _cyl_to_cart_pos(cyl)                                        # (NGRID,3) positions
    return tgrid, xyz, cyl


def _sample_and_integrate(amp):
    """Sample the fardal15 spray (xv(theta)) and integrate every star to t=0."""
    pot = LogarithmicHaloPotential(amp=amp, q=Q0)
    tgrid, prog_xyz, prog_cyl = _prog_curve(amp)
    dt = u_dt * TD                                     # stripping times (theta-indep noise)
    # progenitor state at stripping times: interpolate the dense curve (jittable)
    ti = -dt
    # linear interp in cyl coords at ti over tgrid (tgrid descending)
    Rp = jnp.interp(ti, tgrid[::-1], prog_cyl[::-1, 0])
    vRp = jnp.interp(ti, tgrid[::-1], prog_cyl[::-1, 1])
    vTp = jnp.interp(ti, tgrid[::-1], prog_cyl[::-1, 2])
    zp = jnp.interp(ti, tgrid[::-1], prog_cyl[::-1, 3])
    vzp = jnp.interp(ti, tgrid[::-1], prog_cyl[::-1, 4])
    php = jnp.interp(ti, tgrid[::-1], prog_cyl[::-1, 5])
    # tidal radius + circular velocity at the progenitor points (theta-dependent)
    rt = rtide(pot, Rp, zp, phi=php, t=ti, M=Mprog, use_physical=False)
    vc = jnp.sqrt(-Rp * evaluateRforces(pot, Rp, zp, phi=php, t=ti, use_physical=False))
    rt_frac = rt / Rp
    # fardal15 offsets in the cyl frame (leading tail)
    k = -MEANK + kvec * SIGK
    Rs = Rp + k[:, 0] * rt
    phs = php + k[:, 5] * rt_frac
    zs = k[:, 3] * rt_frac
    vRs = vRp * (1.0 + k[:, 1])
    vTs = vTp + k[:, 2] * vc * rt_frac
    vzs = k[:, 4] * vc * rt_frac
    ics = jnp.stack([Rs, vRs, vTs, zs, vzs, phs], axis=-1)             # (N,6)
    # integrate every star from -dt to 0 via diffrax (batch, per-particle span)
    o = Orbit(ics); o.turn_physical_off()
    ts = jnp.stack([-dt, jnp.zeros(N)], axis=-1)
    o.integrate(ts, pot, method="diffrax")
    fin = o.orbit[:, -1, :]                                            # (N,6) cyl at t=0
    x = fin[:, 0] * jnp.cos(fin[:, 5]); y = fin[:, 0] * jnp.sin(fin[:, 5]); z = fin[:, 3]
    xv_cart = jnp.stack([x, y, z], axis=-1)                           # (N,3) positions
    return tgrid, prog_xyz, xv_cart, dt


TP_LO, TP_HI = -TD, 0.0


def _structure(amp):
    """Discrete reconstruction structure (closest-point assignment idx, tp bin b).
    argmin/histogram are non-differentiable -> computed here and frozen."""
    tgrid, prog_xyz, xv_cart, dt = _sample_and_integrate(amp)
    curve_xyz = prog_xyz
    d2 = jnp.sum((xv_cart[:, None, :] - curve_xyz[None, :, :]) ** 2, axis=-1)
    d2 = jnp.where(tgrid[None, :] <= 0.0, d2, jnp.inf)                 # leading-arm mask
    idx = jnp.argmin(d2, axis=1)
    tp_i = tgrid[idx]
    b = jnp.clip(((tp_i - TP_LO) / (TP_HI - TP_LO) * NBIN).astype(int), 0, NBIN - 1)
    return idx, b


def track_given(amp, idx, b):
    """Track that is DIFFERENTIABLE in amp with the structure (idx, b) FIXED.
    xv(theta) and curve(theta) both flow; only the discrete indices are frozen."""
    tgrid, prog_xyz, xv_cart, dt = _sample_and_integrate(amp)
    curve_xyz = prog_xyz
    off = xv_cart - curve_xyz[idx]                                     # (N,3) flows theta
    onehot = (b[:, None] == jnp.arange(NBIN)[None, :]).astype(off.dtype)
    cnt = onehot.sum(0)
    bin_mean = (onehot.T @ off) / jnp.clip(cnt, 1.0, None)[:, None]    # (NBIN,3) flows theta
    bc = jnp.arange(NBIN)
    W = jnp.exp(-0.5 * ((bc[:, None] - bc[None, :]) / SMOOTH) ** 2) * (cnt[None, :] > 0)
    W = W / jnp.clip(W.sum(1, keepdims=True), 1e-12, None)            # fixed smoother
    smean = W @ bin_mean
    bin_tp = TP_LO + (bc + 0.5) / NBIN * (TP_HI - TP_LO)
    curve_at_bins = jnp.stack([jnp.interp(bin_tp, tgrid[::-1], curve_xyz[::-1, j]) for j in range(3)], axis=-1)
    return curve_at_bins + smean                                      # (NBIN,3) flows theta


def stream_track(amp):
    """Public: structure frozen from THIS amp via stop_gradient, values flow."""
    idx, b = jax.lax.stop_gradient(_structure(amp))
    return track_given(amp, idx, b)


if __name__ == "__main__":
    # freeze structure at amp=1.0 (concrete) so AD and FD apply the IDENTICAL operator
    idx0, b0 = jax.jit(_structure)(jnp.asarray(1.0))
    idx0 = jax.lax.stop_gradient(idx0); b0 = jax.lax.stop_gradient(b0)
    loss = lambda a: track_given(a, idx0, b0).sum()
    f = jax.jit(jax.value_and_grad(loss))
    t0 = time.time()
    v, g = f(jnp.asarray(1.0)); v.block_until_ready()
    print(f"[jit] track.sum()={float(v):.6f}  d/d(amp)={float(g):.6e}  compile {time.time()-t0:.1f}s", flush=True)
    # PHYSICAL grad-vs-FD with FROZEN structure (idx0,b0) -> identical operator; xv(theta) flows
    h = 1e-4
    lj = jax.jit(loss)
    g_fd = (float(lj(jnp.asarray(1.0 + h))) - float(lj(jnp.asarray(1.0 - h)))) / (2 * h)
    print(f"[FD frozen-struct] d/d(amp)={g_fd:.6e}   REL DIFF={abs(float(g)-g_fd)/max(abs(g_fd),1e-12):.2e}", flush=True)
    # for transparency: moving-structure FD (recomputes idx,b -> includes assignment jumps)
    lm = jax.jit(lambda a: stream_track(a).sum())
    g_mv = (float(lm(jnp.asarray(1.0 + h))) - float(lm(jnp.asarray(1.0 - h)))) / (2 * h)
    print(f"[FD moving-struct] d/d(amp)={g_mv:.6e}  (differs by discrete assignment/bin jumps)", flush=True)
    t0 = time.time()
    for a in [1.02, 1.05, 1.08]:
        f(jnp.asarray(a))[0].block_until_ready()
    print(f"cached value+grad: {(time.time()-t0)/3*1000:.0f} ms/step", flush=True)
