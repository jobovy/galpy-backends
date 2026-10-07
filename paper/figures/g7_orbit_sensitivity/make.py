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

import os

import matplotlib.pyplot as plt
import numpy

from paper.figures._harness import layout
from paper.figures._harness.figure import figure
from paper.figures._harness.style import BACKEND_COLORS

# journal version (default): no suptitle, short panel titles, no stamps; the
# caption carries the explanations. GALPY_JAX_PAPER=0 restores the annotated figure.
PAPER = os.environ.get("GALPY_JAX_PAPER", "1") == "1"


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

    # print size: 7.0 in wide (figure*), text 8-10 pt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=layout.figsize(PAPER, (7.0, 3.1), (9.8, 4.0)))

    def _f(paper_pt, detailed_pt):
        return paper_pt if PAPER else detailed_pt

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
    ax1.set_title("(a) orbit sensitivity" if PAPER
                  else "(a) orbit sensitivity (autodiff STM; o = finite-diff)")
    if PAPER:  # headroom so the legend does not cover the dR/dR0 peak
        ax1.set_ylim(-1.6, 6.3)
    ax1.legend(fontsize=_f(8.5, 7.5), ncol=2, loc="upper left", framealpha=0.9)

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
    ls = numpy.array(ls)
    # past convergence the miss sits at the ~1e-27 round-off floor, far below the
    # axis: show only the iterations above the floor
    y_floor = 1e-18
    n_show = int(numpy.nonzero(ls >= y_floor)[0][-1]) + 1
    ax2.semilogy(it[:n_show], ls[:n_show], marker="o", ms=3,
                 color=BACKEND_COLORS["jax"], lw=2)
    ax2.set_xlim(-0.5, n_show - 0.5)
    ax2.set_xticks(it[:n_show])
    ax2.set_xlabel("Gauss-Newton iteration")
    ax2.set_ylabel(r"end-point miss  $\|(R,z)_f-(R,z)_f^{\rm target}\|^2$")
    ax2.set_title("(b) launch-velocity fit" if PAPER
                  else "(b) fit launch velocity: autodiff Jacobian through the solver")
    ax2.set_ylim(y_floor, None)
    # inset: optimizer path in (vR0, vz0)
    # lower left: the loss curve runs top-left -> bottom-right
    axin = ax2.inset_axes([0.21, 0.25, 0.33, 0.32])
    axin.plot(path[:, 0], path[:, 1], "-o", ms=2.5, color=BACKEND_COLORS["jax"], lw=1)
    axin.plot([vR_true], [vz_true], marker="*", ms=11, color="k", zorder=5)
    axin.margins(0.15)  # the truth sits at a corner of the path's range
    axin.annotate("truth", (vR_true, vz_true), xytext=(7, 0), textcoords="offset points",
                  fontsize=_f(8, 6.5), va="center")
    axin.set_xlabel(r"$v_{R0}$", fontsize=_f(8, 7))
    axin.set_ylabel(r"$v_{z0}$", fontsize=_f(8, 7))
    axin.tick_params(labelsize=_f(7, 6))
    axin.set_title("optimizer path", fontsize=_f(8, 7))

    if not PAPER:
        fig.suptitle("Differentiable orbit integration (in-backend diffrax; Plummer, "
                     "CPU float64)", fontsize=11)
    fig.tight_layout()
    return fig
