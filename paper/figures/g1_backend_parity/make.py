"""G1 -- backend value parity + autodiff identities (CPU).

Two promises of the multi-backend design, shown on the migrated potentials:

(a) *Same answer, every backend*: max |numpy - backend| for the potential,
    forces, and second derivatives over an (R,z) grid. In float64, jax is
    bit-identical to numpy (exactly 0) and torch agrees to <= machine epsilon.
(b) *And it differentiates correctly*: both jax.grad and torch.autograd
    reproduce galpy's analytic derivative (d Phi/dR == -F_R) to ~1e-15.

Grows automatically as more potentials are migrated (Track A); today it covers
the pilot set (Plummer, Isochrone). Pure-CPU float64; a GPU companion lands as G2.
"""

import matplotlib.pyplot as plt
import numpy

from paper.figures._harness.figure import figure


@figure(
    name="g1_backend_parity",
    track="G",
    seed=0,
    description="numpy<->jax/torch value parity + jax/torch autodiff identities (migrated potentials)",
)
def make():
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import torch

    torch.set_default_dtype(torch.float64)
    from galpy.potential import (
        IsochronePotential,
        PlummerPotential,
        evaluatePotentials,
        evaluateR2derivs,
        evaluateRforces,
        evaluatez2derivs,
        evaluatezforces,
    )

    pots = [
        ("Plummer", PlummerPotential(normalize=1.0), "#0072B2"),
        ("Isochrone", IsochronePotential(normalize=1.0), "#D55E00"),
    ]
    methods = [
        (r"$\Phi$", evaluatePotentials),
        (r"$F_R$", evaluateRforces),
        (r"$F_z$", evaluatezforces),
        (r"$\partial^2_R\Phi$", evaluateR2derivs),
        (r"$\partial^2_z\Phi$", evaluatez2derivs),
    ]
    R = numpy.linspace(0.3, 3.0, 60)
    z = 0.1 * numpy.ones_like(R)
    FLOOR = 3e-18  # where exactly-zero (bit-identical) points are drawn on the log axis
    EPS = numpy.finfo(float).eps

    def parity(fn, p, backend):
        a = numpy.asarray(fn(p, R, z), dtype=float)
        if backend == "jax":
            b = numpy.asarray(fn(p, jnp.asarray(R), jnp.asarray(z)))
        else:
            b = fn(p, torch.as_tensor(R), torch.as_tensor(z)).detach().numpy()
        return float(numpy.max(numpy.abs(a - b)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.8))

    # ---- (a) value parity: markers (not bars) so exact-zero jax points stay visible ----
    nm = len(methods)
    for pi, (pn, p, col) in enumerate(pots):
        for bi, (backend, mk, filled) in enumerate(
            [("jax", "o", True), ("torch", "^", False)]
        ):
            xs = [mi + (bi - 0.5) * 0.18 + (pi - 0.5) * 0.06 for mi in range(nm)]
            ys = [max(parity(fn, p, backend), FLOOR) for _, fn in methods]
            ax1.scatter(
                xs, ys, marker=mk, s=46,
                facecolors=(col if filled else "none"),
                edgecolors=col, linewidths=1.4, zorder=3,
                label=f"{pn} ({backend})",
            )
    ax1.axhline(EPS, ls="--", lw=1, color="0.4")
    ax1.text(nm - 0.55, EPS * 1.4, r"float64 $\epsilon$", fontsize=8, color="0.4",
             ha="right", va="bottom")
    ax1.axhline(FLOOR, ls=":", lw=1, color="0.6")
    ax1.text(-0.45, FLOOR * 1.5, "= 0 (bit-identical to numpy)", fontsize=8,
             color="0.5", ha="left", va="bottom")
    ax1.set_yscale("log")
    ax1.set_ylim(1.5e-18, 1e-12)
    ax1.set_xlim(-0.6, nm - 0.4)
    ax1.set_xticks(range(nm))
    ax1.set_xticklabels([m[0] for m in methods])
    ax1.set_ylabel(r"$\max\,|\,\mathrm{numpy}-\mathrm{backend}\,|$")
    ax1.set_title("(a) value parity over an $(R,z)$ grid")
    ax1.legend(fontsize=7, ncol=2, loc="upper center", framealpha=0.9)

    # ---- (b) autodiff identity d Phi/dR == -F_R for Plummer, jax + torch ----
    p = PlummerPotential(normalize=1.0)
    z0 = 0.1
    Rl = numpy.linspace(0.3, 3.0, 120)
    FR = numpy.asarray(evaluateRforces(p, Rl, z0 * numpy.ones_like(Rl)))  # = -dPhi/dR
    dphidR_j = numpy.asarray(
        jax.vmap(jax.grad(lambda r: evaluatePotentials(p, r, z0)))(jnp.asarray(Rl))
    )
    res_j = numpy.abs(dphidR_j + FR)
    Rg = torch.tensor(Rl, requires_grad=True)
    phi = evaluatePotentials(p, Rg, torch.as_tensor(z0 * numpy.ones_like(Rl)))
    dphidR_t = torch.autograd.grad(phi.sum(), Rg)[0].detach().numpy()
    res_t = numpy.abs(dphidR_t + FR)

    ax2.semilogy(Rl, res_j + 1e-19, color="#0072B2", label="jax.grad")
    ax2.semilogy(Rl, res_t + 1e-19, color="#D55E00", label="torch.autograd")
    ax2.axhline(EPS, ls="--", lw=1, color="0.4")
    ax2.text(3.0, EPS * 1.4, r"float64 $\epsilon$", fontsize=8, color="0.4",
             ha="right", va="bottom")
    ax2.set_xlabel("R")
    ax2.set_ylim(1e-18, 1e-12)
    ax2.set_ylabel(r"$|\,\partial\Phi/\partial R + F_R\,|$")
    ax2.set_title("(b) autodiff vs analytic (Plummer)")
    ax2.legend(fontsize=8)

    fig.suptitle("Backend parity & autodiff consistency (CPU, float64)", fontsize=11)
    fig.tight_layout()
    return fig
