"""G7 -- differentiable orbit integration: sensitivity + a gradient-descent fit.

The showpiece for the in-backend integrator (galpy.backend._reference.integrate_orbit,
diffrax/torchdiffeq through the backend-agnostic forces). Two demonstrations:

(a) *Orbit sensitivity (state-transition matrix).* The derivative of the orbit w.r.t.
    its initial conditions, d x(t)/d x0, by autodiff through the ODE solve -- the
    tangent dynamics. A few STM elements vs time, with finite-difference markers
    overlaid (autodiff == FD).

(b) *Gradient-descent orbit fit.* Recover the launch velocity (vR0, vz0) that lands an
    orbit on a target end-point, by minimizing the miss-distance with gradients
    backpropagated through the integrator. Loss vs iteration; the optimizer path in
    (vR0, vz0) converges to the truth.

Pure-CPU float64 and deterministic (jax pinned to the CPU device), so the PDF is
byte-stable. Needs jax + diffrax.
"""

import matplotlib.pyplot as plt
import numpy

from paper.figures._harness.figure import figure
from paper.figures._harness.style import BACKEND_COLORS


@figure(
    name="g7_orbit_sensitivity",
    track="G",
    seed=0,
    description="in-backend differentiable orbit integration: STM sensitivity vs time + gradient-descent orbit fit",
)
def make():
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from galpy.backend._reference import integrate_orbit
    from galpy.potential import PlummerPotential

    _CPU = jax.devices("cpu")[0]
    p = PlummerPotential(amp=1.0, b=0.6)
    ic0 = numpy.array([1.0, 0.1, 0.9, 0.2, 0.05, 0.3])  # R,vR,vT,z,vz,phi

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 4.0))

    # ---- (a) orbit sensitivity: STM elements d x(t)/d x0 vs time -----------------
    with jax.default_device(_CPU):
        ts = jnp.asarray(numpy.linspace(0.0, 8.0, 60))

        def traj_R(ic):
            return integrate_orbit(p, ic, ts)[:, 0]  # R(t)

        def traj_z(ic):
            return integrate_orbit(p, ic, ts)[:, 3]  # z(t)

        JR = numpy.asarray(jax.jacobian(traj_R)(jnp.asarray(ic0)))  # (nt,6)
        Jz = numpy.asarray(jax.jacobian(traj_z)(jnp.asarray(ic0)))

    tn = numpy.asarray(ts)
    # element: (label, array, color)
    elems = [
        (r"$\partial R(t)/\partial R_0$", JR[:, 0], "#0072B2"),
        (r"$\partial R(t)/\partial v_{R0}$", JR[:, 1], "#56B4E9"),
        (r"$\partial z(t)/\partial z_0$", Jz[:, 3], "#D55E00"),
        (r"$\partial z(t)/\partial v_{z0}$", Jz[:, 4], "#E69F00"),
    ]
    for lab, arr, col in elems:
        ax1.plot(tn, arr, color=col, lw=1.8, label=lab)
    # finite-difference validation at a handful of times
    eps = 1e-6
    tcheck = tn[::12]

    def _fd_col(traj_fn, j):
        icp = ic0.copy(); icp[j] += eps
        icm = ic0.copy(); icm[j] -= eps
        with jax.default_device(_CPU):
            fp = numpy.asarray(traj_fn(jnp.asarray(icp)))
            fm = numpy.asarray(traj_fn(jnp.asarray(icm)))
        return (fp - fm) / (2 * eps)

    fd_map = [(traj_R, 0), (traj_R, 1), (traj_z, 3), (traj_z, 4)]
    for (lab, arr, col), (fn, j) in zip(elems, fd_map):
        fdvals = _fd_col(fn, j)[::12]
        ax1.plot(tcheck, fdvals, ls="none", marker="o", ms=4, mfc="none",
                 mec=col, mew=1.2)
    ax1.axhline(0, color="0.8", lw=0.8, zorder=0)
    ax1.set_xlabel("time")
    ax1.set_ylabel(r"sensitivity  $\partial x(t)/\partial x_0$")
    ax1.set_title("(a) orbit sensitivity (autodiff STM; o = finite-diff)")
    ax1.legend(fontsize=7.5, ncol=2, loc="upper left", framealpha=0.9)

    # ---- (b) gradient-descent orbit fit: recover (vR0, vz0) ----------------------
    with jax.default_device(_CPU):
        tsf = jnp.asarray(numpy.linspace(0.0, 3.0, 60))
        vR_true, vz_true = 0.20, 0.15
        # solver headroom so no descent step ever hits the diffrax max_steps cap
        _kw = dict(rtol=1e-9, atol=1e-9, max_steps=2_000_000)
        ic_true = jnp.array([1.0, vR_true, 0.9, 0.2, vz_true, 0.3])
        target = integrate_orbit(p, ic_true, tsf, **_kw)[-1]
        target_Rz = jnp.array([target[0], target[3]])  # final (R, z)

        def resid(theta):  # theta = [vR0, vz0] -> end-point (R,z) miss vector
            ic = jnp.array([1.0, theta[0], 0.9, 0.2, theta[1], 0.3])
            end = integrate_orbit(p, ic, tsf, **_kw)[-1]
            return jnp.array([end[0], end[3]]) - target_Rz

        # Gauss-Newton using the autodiff Jacobian of the residual w.r.t. the launch
        # velocity (jax.jacobian *through the ODE solve*). 2nd-order -> quadratic,
        # monotonic convergence to machine precision in a few steps; the autodiff
        # gives the full Jacobian for free, not just the gradient.
        theta = jnp.array([0.40, -0.10])  # wrong initial guess
        n_it = 10
        ls, path = [], []
        for _ in range(n_it):
            r = resid(theta)
            ls.append(float(jnp.sum(r**2)))
            path.append((float(theta[0]), float(theta[1])))
            J = jax.jacobian(resid)(theta)  # (2,2), through the solver
            step = jnp.linalg.lstsq(J, r, rcond=None)[0]
            # clip to a sane velocity box (guards a stray step into a stiff plunge)
            theta = jnp.clip(theta - step, -0.8, 0.8)
        path = numpy.array(path)

    it = numpy.arange(n_it)
    ax2.semilogy(it, numpy.array(ls) + 1e-20, marker="o", ms=3,
                 color=BACKEND_COLORS["jax"], lw=2)
    ax2.set_xlabel("Gauss-Newton iteration")
    ax2.set_ylabel(r"end-point miss  $\|(R,z)_f-(R,z)_f^{\rm target}\|^2$")
    ax2.set_title("(b) fit launch velocity: autodiff Jacobian through the solver")
    ax2.set_ylim(1e-18, None)
    # inset: optimizer path in (vR0, vz0)
    axin = ax2.inset_axes([0.50, 0.50, 0.46, 0.44])
    axin.plot(path[:, 0], path[:, 1], "-o", ms=2.5, color=BACKEND_COLORS["jax"], lw=1)
    axin.plot([vR_true], [vz_true], marker="*", ms=11, color="k", zorder=5)
    axin.text(vR_true, vz_true, "  truth", fontsize=6.5, va="center")
    axin.set_xlabel(r"$v_{R0}$", fontsize=7)
    axin.set_ylabel(r"$v_{z0}$", fontsize=7)
    axin.tick_params(labelsize=6)
    axin.set_title("optimizer path", fontsize=7)

    fig.suptitle("Differentiable orbit integration (in-backend diffrax; Plummer, "
                 "CPU float64)", fontsize=11)
    fig.tight_layout()
    return fig
