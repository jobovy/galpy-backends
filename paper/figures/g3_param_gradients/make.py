"""G3 -- autodiff w.r.t. potential *parameters* (d/dtheta): correctness + a fit.

The complement to G1 (which shows coordinate-gradient autodiff, dPhi/dR == -F_R):
here the differentiation is w.r.t. the potential's *constructor parameters*
(amplitude, scale length), the quantity an inference/optimization problem varies.

(a) *Correctness.* For Plummer and Isochrone, the parameter gradient of Phi and of
    F_R w.r.t. amp and the scale length b, under jax and torch. Two checks per
    quantity: jax-vs-torch (two independent AD engines agree to ~1e-15 -- both
    compute the *exact* derivative), and AD-vs-finite-difference (agree to ~1e-9,
    the central-difference accuracy floor -- AD is exact, FD is the noisy one).

(b) *A worked example.* Recover a Plummer scale length by gradient descent: fit b
    to a target F_R(R) profile, the gradient dLoss/db supplied by autodiff. Both
    backends drive the same descent to b_true to ~1e-8.

Pure-CPU float64 and deterministic, so the PDF is byte-stable. Enabled by the
unit-parser pass-through for backend arrays (galpy.backend.is_backend_array).
"""

import matplotlib.pyplot as plt
import numpy

from paper.figures._harness.figure import figure
from paper.figures._harness.style import BACKEND_COLORS


@figure(
    name="g3_param_gradients",
    track="G",
    seed=0,
    description="autodiff w.r.t. potential parameters (d/dtheta): jax/torch vs FD + a gradient-descent fit",
)
def make():
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import torch

    # Pin all jax work to the CPU *device* (not the process-global platform flag,
    # which is ignored once jax has initialized on a GPU -- e.g. when `verify` runs
    # this right after a GPU figure in the same process). This makes the figure
    # byte-stable and portable regardless of whether a GPU is present/initialized.
    _CPU = jax.devices("cpu")[0]

    torch.set_default_dtype(torch.float64)
    from galpy.potential import (
        IsochronePotential,
        PlummerPotential,
        evaluatePotentials,
        evaluateRforces,
    )

    EPS = numpy.finfo(float).eps
    _R = numpy.linspace(0.3, 3.0, 30)
    _Z = 0.1 * numpy.ones_like(_R)

    # quantity evaluators (public: apply amp, so amp-gradients flow)
    QUANTS = [(r"$\Phi$", evaluatePotentials), (r"$F_R$", evaluateRforces)]
    # (name, constructor, fixed kwargs, differentiated-parameter name, value)
    SPECS = [
        ("Plummer", PlummerPotential, {"amp": 1.0}, "b", 0.7, "#0072B2"),
        ("Plummer", PlummerPotential, {"b": 0.7}, "amp", 1.3, "#0072B2"),
        ("Isochrone", IsochronePotential, {"amp": 1.0}, "b", 1.1, "#D55E00"),
        ("Isochrone", IsochronePotential, {"b": 1.1}, "amp", 2.0, "#D55E00"),
    ]

    def param_jac(fn, ctor, fixed, pname, theta, backend):
        # per-R derivative d(fn(R,z))/d(theta), as a length-len(_R) vector
        if backend == "jax":
            with jax.default_device(_CPU):
                R, z = jnp.asarray(_R), jnp.asarray(_Z)

                def f(th):
                    return fn(ctor(**{**fixed, pname: th}), R, z)

                return numpy.asarray(jax.jacobian(f)(jnp.asarray(theta)))
        if backend == "torch":
            R, z = torch.as_tensor(_R), torch.as_tensor(_Z)
            # per-element d out_i/d theta (theta is scalar; len(_R)=30, cheap loop)
            rows = []
            for i in range(len(_R)):
                thi = torch.tensor(theta, requires_grad=True)
                oi = fn(ctor(**{**fixed, pname: thi}), R, z)[i]
                (gi,) = torch.autograd.grad(oi, thi)
                rows.append(float(gi))
            return numpy.asarray(rows)
        # finite difference (numpy path)
        h = 1e-6
        fp = numpy.asarray(fn(ctor(**{**fixed, pname: theta + h}), _R, _Z))
        fm = numpy.asarray(fn(ctor(**{**fixed, pname: theta - h}), _R, _Z))
        return (fp - fm) / (2 * h)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.9))

    # ---- (a) parameter-gradient correctness ------------------------------------
    xpos, xlab, xcol = [], [], []
    for si, (pn, ctor, fixed, pname, th, col) in enumerate(SPECS):
        for qi, (qlab, fn) in enumerate(QUANTS):
            x = si + (qi - 0.5) * 0.34
            jacj = param_jac(fn, ctor, fixed, pname, th, "jax")
            jact = param_jac(fn, ctor, fixed, pname, th, "torch")
            jfd = param_jac(fn, ctor, fixed, pname, th, "fd")
            d_jt = max(float(numpy.max(numpy.abs(jacj - jact))), 1e-20)  # AD vs AD
            d_fd = float(numpy.max(numpy.abs(jacj - jfd)))  # AD vs FD
            ax1.scatter(x, d_jt, marker="o", s=40, facecolors=col, edgecolors=col,
                        zorder=3)
            ax1.scatter(x, d_fd, marker="^", s=46, facecolors="none", edgecolors=col,
                        linewidths=1.4, zorder=3)
            xpos.append(x)
            xlab.append(qlab)
            xcol.append(col)
    ax1.axhline(EPS, ls="--", lw=1, color="0.4")
    ax1.text(len(SPECS) - 0.5, EPS * 1.3, r"float64 $\epsilon$", fontsize=8,
             color="0.4", ha="right", va="bottom")
    ax1.set_yscale("log")
    ax1.set_ylim(3e-17, 1e-6)
    ax1.set_xticks(xpos)
    ax1.set_xticklabels(xlab, fontsize=8)
    ax1.set_xlim(-0.6, len(SPECS) - 0.4)
    for si, (pn, _, _, pname, _, col) in enumerate(SPECS):
        ax1.text(si, 1.5e-7, f"{pn}\n" + r"$\partial_{%s}$" % pname, fontsize=7.5,
                 color=col, ha="center", va="top")
    ax1.set_ylabel(r"max$_R$ |gradient difference|")
    ax1.set_title("(a) parameter-gradient correctness")
    h1 = ax1.scatter([], [], marker="o", s=40, color="0.3",
                     label=r"jax vs torch (AD vs AD)")
    h2 = ax1.scatter([], [], marker="^", s=46, facecolors="none", edgecolors="0.3",
                     label="AD vs finite-diff")
    ax1.legend(handles=[h1, h2], fontsize=7.5, loc="upper left", framealpha=0.9)

    # ---- (b) gradient-descent recovery of a Plummer scale length ---------------
    # start above b_true so the descent is monotone (clean convergence curve)
    b_true, amp, lr, n_it = 0.85, 1.0, 0.5, 60
    b0 = 1.50
    target_np = numpy.asarray(
        evaluateRforces(PlummerPotential(amp=amp, b=b_true), _R, _Z)
    )

    # jax descent (pinned to CPU device for byte-stability; see _CPU above)
    with jax.default_device(_CPU):
        Rj, zj, tj = jnp.asarray(_R), jnp.asarray(_Z), jnp.asarray(target_np)

        def loss_j(b):
            fr = evaluateRforces(PlummerPotential(amp=amp, b=b), Rj, zj)
            return jnp.mean((fr - tj) ** 2)

        gj = jax.grad(loss_j)
        bj = jnp.asarray(b0)
        lj, bj_hist = [], []
        for _ in range(n_it):
            lj.append(float(loss_j(bj)))
            bj_hist.append(float(bj))
            bj = bj - lr * gj(bj)

    # torch descent
    Rt, zt, tt = torch.as_tensor(_R), torch.as_tensor(_Z), torch.as_tensor(target_np)
    bt = torch.tensor(b0, requires_grad=True)
    lt = []
    for _ in range(n_it):
        loss = torch.mean(
            (evaluateRforces(PlummerPotential(amp=amp, b=bt), Rt, zt) - tt) ** 2
        )
        lt.append(float(loss.detach()))
        (g,) = torch.autograd.grad(loss, bt)
        with torch.no_grad():
            bt = (bt - lr * g).clone().detach().requires_grad_(True)

    it = numpy.arange(n_it)
    ax2.semilogy(it, numpy.array(lj) + 1e-20, color=BACKEND_COLORS["jax"], lw=2,
                 label="jax.grad descent")
    ax2.semilogy(it, numpy.array(lt) + 1e-20, color=BACKEND_COLORS["torch"], lw=1,
                 ls="--", label="torch.autograd descent")
    ax2.set_xlabel("gradient-descent iteration")
    ax2.set_ylabel(r"fit loss  $\langle (F_R(b)-F_R^{\rm target})^2\rangle$")
    ax2.set_title("(b) recover Plummer $b$ by autodiff gradient descent")
    ax2.set_ylim(1e-18, 1.0)
    ax2.legend(fontsize=8, loc="upper right")
    # parameter-recovery inset, bottom-left (the loss curve runs top-left ->
    # bottom-right, leaving the lower-left corner empty)
    axin = ax2.inset_axes([0.13, 0.14, 0.40, 0.38])
    axin.plot(it, bj_hist, color=BACKEND_COLORS["jax"], lw=1.5)
    axin.axhline(b_true, ls=":", color="0.4", lw=1)
    axin.text(n_it * 0.97, b_true, " $b_{\\rm true}$", fontsize=7, color="0.4",
              va="center", ha="left")
    axin.set_xlabel("iter", fontsize=7)
    axin.set_ylabel("$b$", fontsize=7)
    axin.tick_params(labelsize=6)
    axin.text(0.95, 0.9, f"recovered\n$b={float(bj):.6f}$", transform=axin.transAxes,
              fontsize=6.5, ha="right", va="top", color=BACKEND_COLORS["jax"])

    fig.suptitle("Parameter-gradient autodiff: correctness & a worked fit "
                 "(CPU, float64)", fontsize=11)
    fig.tight_layout()
    return fig
