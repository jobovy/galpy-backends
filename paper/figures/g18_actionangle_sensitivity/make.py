"""G18 -- differentiable action-angle: sensitivity of the actions to the initial
conditions and to the potential parameters (the AA analogue of G7).

The action-angle map J(x) = (J_r, L_z, J_z) is now backend-agnostic, so it runs --
and *differentiates* -- under jax/torch. This figure is the "derivatives show-off"
for that stack: it overlays the autodiff sensitivities dJ/d(.) (jax.grad straight
through the action map) on top of the action curves, and validates every one
against a central finite difference (open circles). autodiff == finite-diff.

(a) *Initial-condition sensitivity.* Sweep the radial velocity v_R (at fixed
    R, v_T, z, v_z) and plot the radial and vertical actions J_r(v_R), J_z(v_R)
    (left axis), with the autodiff sensitivities dJ_r/dv_R, dJ_z/dv_R (right axis)
    overlaid -- how the actions respond to a kick in v_R.

(b) *Potential-parameter sensitivity* -- the showpiece for fitting a potential to
    action data. Hold one orbit fixed and sweep the isochrone scale b, plotting
    J_r(b) (left axis) with dJ_r/db (right axis) overlaid: the gradient an inference
    loop backpropagates when it tunes the potential to match measured actions.

We use actionAngleIsochrone: its actions are closed-form (no quadrature), so the
autodiff derivative is machine-exact and the byte-stable panel is a clean
autodiff==FD statement. (actionAngleStaeckel is the branchier flagship -- its
actions flow through turning-point root-finds + Gauss-Legendre quadrature and run
under the backends too -- but on this build a plain-GL *value* quadrature is used
to match the C path bit-for-bit, and jax.grad through that plain-GL integral of a
sqrt-vanishing integrand converges only algebraically to dJ/d(.); the accurate
derivative integrands live on the frequency path, so a <1e-4 autodiff==FD Staeckel
statement waits on the derivative-integrand work. See README.)

Pure-CPU float64 and deterministic (jax pinned to the CPU device, x64 enabled), so
the PDF is byte-stable. Needs jax.
"""

import os

import matplotlib.pyplot as plt
import numpy

from paper.figures._harness.figure import figure

# journal version (default): no suptitle, short panel titles, no stamps; the
# caption carries the explanations. GALPY_JAX_PAPER=0 restores the annotated figure.
PAPER = os.environ.get("GALPY_JAX_PAPER", "1") == "1"
# colour families: an action and its sensitivity share a hue (dark = value,
# light = derivative), so panel (a) reads as two (value, d/dv_R) pairs.
_C_JR, _C_DJR = "#0072B2", "#56B4E9"  # blue      / sky blue   (radial)
_C_JZ, _C_DJZ = "#D55E00", "#E69F00"  # vermillion / orange     (vertical)


@figure(
    name="g18_actionangle_sensitivity",
    track="G",
    seed=0,
    description="differentiable action-angle: IC sensitivity dJ/dv_R + potential-parameter sensitivity dJ_r/db (jax.grad vs finite-diff, CPU float64)",
)
def make():
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from galpy.actionAngle import actionAngleIsochrone
    from galpy.potential import IsochronePotential

    # Pin all jax work to the CPU *device* (not just the platform flag, which is
    # ignored once jax has initialised on a GPU) -> byte-stable, portable.
    _CPU = jax.devices("cpu")[0]

    # Isochrone scaled so the circular velocity at R=1 is 1 (GM = (b+rb)^2 rb).
    B0 = 0.8
    _rb = numpy.sqrt(B0**2 + 1.0)
    GM = (B0 + _rb) ** 2 * _rb
    EPS = 1e-6  # central finite-difference step

    # PAPER: final printed width of a figure* (7.0 in); text sizes are then the
    # printed sizes (9-10 pt labels/ticks, 10 pt titles, 8 pt notes/legends).
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 3.3) if PAPER else (9.8, 4.0))
    _ts = {"fontsize": 10} if PAPER else {}
    _fs_leg = 8 if PAPER else 7.3
    _fs_note = 8 if PAPER else 7.2

    # ---- (a) initial-condition sensitivity: dJ/dv_R -----------------------------
    R, vT, z, vz = 1.0, 0.85, 0.15, 0.10
    vR = numpy.linspace(-0.45, 0.45, 121)

    with jax.default_device(_CPU):
        aA = actionAngleIsochrone(ip=IsochronePotential(amp=GM, b=B0))
        cst = {k: jnp.asarray(v) for k, v in dict(R=R, vT=vT, z=z, vz=vz).items()}

        def _act(vr, idx):  # scalar v_R -> scalar action (idx 0=Jr, 2=Jz)
            return aA(cst["R"], vr, cst["vT"], cst["z"], cst["vz"])[idx]

        vRj = jnp.asarray(vR)
        Jr = numpy.asarray(jax.vmap(lambda v: _act(v, 0))(vRj))
        Jz = numpy.asarray(jax.vmap(lambda v: _act(v, 2))(vRj))
        dJr = numpy.asarray(jax.vmap(jax.grad(lambda v: _act(v, 0)))(vRj))
        dJz = numpy.asarray(jax.vmap(jax.grad(lambda v: _act(v, 2)))(vRj))

        # finite-difference check at a handful of sweep points
        ic = numpy.arange(0, vR.size, 20)  # 7 markers
        vRc = vR[ic]

        def _fd(idx):
            fp = jax.vmap(lambda v: _act(v, idx))(jnp.asarray(vRc + EPS))
            fm = jax.vmap(lambda v: _act(v, idx))(jnp.asarray(vRc - EPS))
            return numpy.asarray((fp - fm) / (2 * EPS))

        fdJr, fdJz = _fd(0), _fd(2)

    res_a = max(
        float(numpy.max(numpy.abs(dJr[ic] - fdJr) / (numpy.abs(fdJr) + 1e-12))),
        float(numpy.max(numpy.abs(dJz[ic] - fdJz) / (numpy.abs(fdJz) + 1e-12))),
    )

    # left axis: the actions themselves
    (l_jr,) = ax1.plot(vR, Jr, color=_C_JR, lw=2.0, label=r"$J_r$")
    (l_jz,) = ax1.plot(vR, Jz, color=_C_JZ, lw=2.0, label=r"$J_z$")
    ax1.set_xlabel(r"radial velocity  $v_R$")
    ax1.set_ylabel(r"action  $J$")
    ax1.set_ylim(-0.01, None)
    # right axis: the autodiff sensitivities + FD markers
    ax1r = ax1.twinx()
    ax1r.grid(False)
    (l_djr,) = ax1r.plot(vR, dJr, color=_C_DJR, lw=1.6, ls="--",
                         label=r"$\partial J_r/\partial v_R$ (autodiff)")
    (l_djz,) = ax1r.plot(vR, dJz, color=_C_DJZ, lw=1.6, ls="--",
                         label=r"$\partial J_z/\partial v_R$ (autodiff)")
    ax1r.plot(vRc, fdJr, ls="none", marker="o", ms=5, mfc="none", mec=_C_DJR, mew=1.3)
    ax1r.plot(vRc, fdJz, ls="none", marker="o", ms=5, mfc="none", mec=_C_DJZ, mew=1.3)
    ax1r.axhline(0, color="0.8", lw=0.8, zorder=0)
    ax1r.set_ylabel(r"sensitivity  $\partial J/\partial v_R$")
    ax1.set_title("(a) initial-condition sensitivity", **_ts)
    ax1.legend(handles=[l_jr, l_jz, l_djr, l_djz], fontsize=_fs_leg, ncol=1,
               loc="upper center", framealpha=0.9)
    ax1.text(0.03, 0.15 if PAPER else 0.05, r"$\circ$ = finite-diff", transform=ax1.transAxes,
             fontsize=8, va="bottom", ha="left")
    # worst autodiff-vs-FD residual across both panels (filled in after panel b)
    _resid_ax = ax1

    # ---- (b) potential-parameter sensitivity: dJ_r/db ---------------------------
    Rb, vRb, vTb, zb, vzb = 1.0, 0.20, 0.85, 0.15, 0.10
    bsw = numpy.linspace(0.4, 1.6, 121)

    with jax.default_device(_CPU):
        cb = {k: jnp.asarray(v) for k, v in
              dict(R=Rb, vR=vRb, vT=vTb, z=zb, vz=vzb).items()}

        def _jr_of_b(b):  # scalar b -> scalar J_r (potential rebuilt inside grad)
            aAb = actionAngleIsochrone(ip=IsochronePotential(amp=GM, b=b))
            return aAb(cb["R"], cb["vR"], cb["vT"], cb["z"], cb["vz"])[0]

        bj = jnp.asarray(bsw)
        Jrb = numpy.asarray(jax.vmap(_jr_of_b)(bj))
        dJrb = numpy.asarray(jax.vmap(jax.grad(_jr_of_b))(bj))

        icb = numpy.arange(0, bsw.size, 20)
        bc = bsw[icb]
        fp = jax.vmap(_jr_of_b)(jnp.asarray(bc + EPS))
        fm = jax.vmap(_jr_of_b)(jnp.asarray(bc - EPS))
        fdJrb = numpy.asarray((fp - fm) / (2 * EPS))

    res_b = float(numpy.max(numpy.abs(dJrb[icb] - fdJrb) / (numpy.abs(fdJrb) + 1e-12)))

    (l_jrb,) = ax2.plot(bsw, Jrb, color=_C_JR, lw=2.0, label=r"$J_r(b)$")
    ax2.set_xlabel(r"isochrone scale  $b$")
    ax2.set_ylabel(r"radial action  $J_r$")
    ax2r = ax2.twinx()
    ax2r.grid(False)
    (l_djrb,) = ax2r.plot(bsw, dJrb, color=_C_DJR, lw=1.6, ls="--",
                          label=r"$\partial J_r/\partial b$ (autodiff)")
    ax2r.plot(bc, fdJrb, ls="none", marker="o", ms=5, mfc="none", mec=_C_DJR, mew=1.3)
    ax2r.set_ylabel(r"sensitivity  $\partial J_r/\partial b$")
    ax2.set_title("(b) potential-parameter sensitivity", **_ts)
    ax2.legend(handles=[l_jrb, l_djrb], fontsize=8, loc="upper left", framealpha=0.9)
    ax2.text(0.03, 0.86, r"$\circ$ = finite-diff", transform=ax2.transAxes,
             fontsize=8, va="bottom", ha="left")
    # annotate the (tiny) worst autodiff-vs-FD residual across both panels, in the
    # clear lower-right corner of panel (a)
    _resid_ax.text(0.97, 0.04,
                   r"$\max\,|{\rm AD}-{\rm FD}|/|{\rm FD}| = %.0e$" % max(res_a, res_b),
                   transform=_resid_ax.transAxes, fontsize=_fs_note, va="bottom",
                   ha="right", color="0.35")

    if not PAPER:
        fig.suptitle("Differentiable action-angle (autodiff vs finite-difference; "
                     "CPU float64)", fontsize=11)
    fig.tight_layout()
    return fig
