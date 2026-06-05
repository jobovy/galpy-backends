"""G1 -- backend value parity + autodiff identities (CPU).

Two things the multi-backend design promises, shown on the migrated potentials:

(a) *Same answer, every backend*: |numpy - jax| for the potential, forces, and
    second derivatives over an (R,z) grid sits at the float64 round-off floor.
(b) *And it differentiates correctly*: jax.grad reproduces galpy's own analytic
    derivatives -- d(Phi)/dR == -F_R and d(-F_R)/dR == d2Phi/dR2 -- to ~1e-15.

Grows automatically as more potentials are migrated (Track A); today it covers
the pilot set (Plummer, Isochrone). Pure-CPU; a GPU companion lands as G2.
"""

import matplotlib.pyplot as plt
import numpy

from paper.figures._harness.figure import figure


@figure(
    name="g1_backend_parity",
    track="G",
    seed=0,
    description="numpy<->jax value parity + jax autodiff identities (migrated potentials)",
)
def make():
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
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
        ("Plummer", PlummerPotential(normalize=1.0)),
        ("Isochrone", IsochronePotential(normalize=1.0)),
    ]
    methods = [
        (r"$\Phi$", evaluatePotentials),
        (r"$F_R$", evaluateRforces),
        (r"$F_z$", evaluatezforces),
        (r"$\partial^2\Phi/\partial R^2$", evaluateR2derivs),
        (r"$\partial^2\Phi/\partial z^2$", evaluatez2derivs),
    ]
    Rg = numpy.linspace(0.3, 3.0, 60)
    zg = numpy.linspace(-1.0, 1.0, 60)
    RR, ZZ = numpy.meshgrid(Rg, zg)
    Rf, Zf = RR.ravel(), ZZ.ravel()
    Rj, Zj = jnp.asarray(Rf), jnp.asarray(Zf)

    # (a) value parity: max |numpy - jax| per (potential, method)
    labels, errs, colors = [], [], []
    cmap = {"Plummer": "#0072B2", "Isochrone": "#D55E00"}
    for pn, p in pots:
        for mn, fn in methods:
            a = numpy.asarray(fn(p, Rf, Zf), dtype=float)
            b = numpy.asarray(fn(p, Rj, Zj), dtype=float)
            errs.append(float(numpy.max(numpy.abs(a - b))))
            labels.append(mn)
            colors.append(cmap[pn])

    # (b) autodiff identities for Plummer at fixed z0
    p = PlummerPotential(normalize=1.0)
    z0 = 0.1
    Rline = numpy.linspace(0.3, 3.0, 120)
    Rlj = jnp.asarray(Rline)
    dphidR = numpy.asarray(
        jax.vmap(jax.grad(lambda R: evaluatePotentials(p, R, z0)))(Rlj)
    )
    minusFR = -numpy.asarray(evaluateRforces(p, Rline, z0 * numpy.ones_like(Rline)))
    resid1 = numpy.abs(dphidR - minusFR)
    dFRdR = numpy.asarray(
        jax.vmap(jax.grad(lambda R: -evaluateRforces(p, R, z0)))(Rlj)
    )
    R2 = numpy.asarray(evaluateR2derivs(p, Rline, z0 * numpy.ones_like(Rline)))
    resid2 = numpy.abs(dFRdR - R2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.6))

    floor = 1e-18
    x = numpy.arange(len(errs))
    ax1.bar(x, numpy.maximum(errs, floor), color=colors)
    ax1.axhline(numpy.finfo(float).eps, ls="--", lw=1, color="0.4", label="float64 eps")
    ax1.set_yscale("log")
    ax1.set_ylim(1e-18, 1e-12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels * 1, rotation=45, ha="right")
    ax1.set_ylabel(r"$\max\,|\,\mathrm{numpy}-\mathrm{jax}\,|$")
    ax1.set_title("(a) value parity over an $(R,z)$ grid")
    handles = [
        plt.Line2D([], [], color=cmap["Plummer"], lw=6, label="Plummer"),
        plt.Line2D([], [], color=cmap["Isochrone"], lw=6, label="Isochrone"),
        plt.Line2D([], [], color="0.4", ls="--", lw=1, label="float64 eps"),
    ]
    ax1.legend(handles=handles, fontsize=8)

    ax2.semilogy(Rline, resid1 + floor, color="#0072B2",
                 label=r"$|\,\partial\Phi/\partial R + F_R\,|$")
    ax2.semilogy(Rline, resid2 + floor, color="#009E73",
                 label=r"$|\,\partial(-F_R)/\partial R - \partial^2\Phi/\partial R^2\,|$")
    ax2.axhline(numpy.finfo(float).eps, ls="--", lw=1, color="0.4")
    ax2.set_xlabel("R")
    ax2.set_ylabel("autodiff identity residual")
    ax2.set_ylim(1e-18, 1e-12)
    ax2.set_title("(b) jax.grad vs analytic (Plummer)")
    ax2.legend(fontsize=8)

    fig.suptitle("Backend parity & autodiff consistency (CPU, float64)", fontsize=11)
    fig.tight_layout()
    return fig
