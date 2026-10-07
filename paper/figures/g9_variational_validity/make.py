"""G9 -- validity diagnostics of the variational equations (Track B).

Four panels measured against the feat/pvar-lyapunov galpy build (MWPotential2014,
C dop853_c at rtol=atol=1e-12; see measure/variational_validity_measure.py):

 (a) Liouville invariant |det M(t)-1| for the 6x6 (3D) and 4x4 (planar) STM
     built from the canonical rectangular basis deviations of integrate_dxdv;
 (b) symplecticity defect ||M^T Omega M - Omega||_max in canonical rectangular
     coordinates -- a stronger necessary condition than (a);
 (c) the ground-truth test: finite differences of the integrated flow vs the
     STM columns at t=10, as a function of the FD step eps -- the classic FD
     V-shape (truncation arm down, roundoff arm up), with the STM column
     agreeing with the best central difference at its ~1e-8 floor;
 (d) the 2D-reduction bridge: for a planar IC, the (x,y,vx,vy) block of the 3D
     STM vs the independently-implemented planar STM.

Cross-environment pattern (like g8): the measure step runs standalone against
the galpy worktree and writes data/*.json; this make() is plot-only and
byte-stable from that JSON.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy

from paper.figures._harness.figure import figure

_DATA = os.path.join(os.path.dirname(__file__), "data")
# journal version (default): no provenance stamp / explanatory annotations; the
# caption carries them. GALPY_JAX_PAPER=0 restores the annotated figure.
PAPER = os.environ.get("GALPY_JAX_PAPER", "1") == "1"


def _load():
    with open(os.path.join(_DATA, "variational_validity.json")) as fh:
        return json.load(fh)


@figure(
    name="g9_variational_validity",
    track="G",
    seed=0,
    description="Variational-equation validity: Liouville det M, symplecticity, FD-of-the-flow V-curve, 2D-reduction bridge (MWPotential2014, dop853_c) [cross-env measure->json, plot byte-stable]",
)
def make():
    d = _load()
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.8))

    # ---- (a) Liouville: |det M(t) - 1| ----
    ax = axes[0, 0]
    ab = d["panel_ab"]
    t = numpy.array(ab["t"])
    ax.semilogy(
        t[1:], numpy.array(ab["abs_det_err_3d"])[1:],
        color="#0072B2", label=r"3D ($6\times6$ STM)",
    )
    ax.semilogy(
        t[1:], numpy.array(ab["abs_det_err_2d"])[1:],
        color="#D55E00", label=r"planar ($4\times4$ STM)",
    )
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$|\det M(t) - 1|$")
    ax.set_title("(a) Liouville invariant", fontsize=10)
    ax.set_ylim(1e-15, 1e-5)
    ax.legend(fontsize=7.5, loc="lower right")

    # ---- (b) symplecticity ----
    ax = axes[0, 1]
    ax.semilogy(
        t[1:], numpy.array(ab["symp_err_3d"])[1:],
        color="#0072B2", label="3D",
    )
    ax.semilogy(
        t[1:], numpy.array(ab["symp_err_2d"])[1:],
        color="#D55E00", label="planar",
    )
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\|M^{\mathsf{T}}\Omega M - \Omega\|_{\max}$")
    ax.set_title("(b) symplecticity defect", fontsize=10)
    ax.set_ylim(1e-14, 1e-5)
    ax.legend(fontsize=7.5, loc="lower right")

    # ---- (c) FD of the flow: the V-curve ----
    ax = axes[1, 0]
    c = d["panel_c"]
    eps = numpy.array(c["eps"])
    fwd = numpy.array(c["resid_forward"])
    cen = numpy.array(c["resid_central"])
    ax.loglog(eps, fwd, marker="o", ms=3.5, color="#009E73",
              label="forward difference")
    ax.loglog(eps, cen, marker="s", ms=3.5, color="#CC79A7",
              label="central difference")
    # slope guides anchored on the well-resolved arms (two-point segments,
    # clipped so they do not stretch the axes)
    g1 = numpy.array([3e-8, 1e-2])
    ax.loglog(g1, fwd[4] * (g1 / eps[4]), ls=":", lw=0.9, color="0.65")
    g2 = numpy.array([5e-7, 1e-2])
    ax.loglog(g2, cen[4] * (g2 / eps[4]) ** 2, ls=":", lw=0.9, color="0.65")
    g3 = numpy.array([1e-11, 1e-6])
    ax.loglog(g3, cen[-3] * (eps[-3] / g3), ls=":", lw=0.9, color="0.65")
    ax.annotate(r"$\propto\varepsilon$", xy=(8e-4, 4e-2), fontsize=8, color="0.45")
    ax.annotate(r"$\propto\varepsilon^{2}$", xy=(8e-4, 2e-4), fontsize=8,
                color="0.45")
    ax.annotate(r"$\propto\varepsilon^{-1}$", xy=(1e-9, 1.2e-6), fontsize=8,
                color="0.45")
    if not PAPER:
        ax.annotate(
            "STM matches the best FD\n" r"at its floor ($\sim$10$^{-8}$)",
            xy=(1.2e-6, 1.4e-8), xytext=(8e-5, 3.2e-9), fontsize=7.5,
            arrowprops=dict(arrowstyle="->", lw=0.8, color="0.3"), color="0.3",
        )
    ax.set_xlabel(r"FD step $\varepsilon$")
    ax.set_ylabel(
        r"$\max_i\,\|M(t)\,e_i - \Delta x(t)/\varepsilon\|_\infty$"
    )
    ax.set_title(r"(c) finite differences vs STM ($t=10$)" if PAPER
                 else r"(c) FD of the flow vs STM ($t=10$)", fontsize=10)
    ax.set_ylim(1e-9, 3e0)
    ax.legend(fontsize=7.5, loc="upper center")

    # ---- (d) 2D-reduction bridge ----
    ax = axes[1, 1]
    pd_ = d["panel_d"]
    td = numpy.array(pd_["t"])
    ax.semilogy(
        td[1:], numpy.array(pd_["bridge_err"])[1:], color="#0072B2",
        label=r"$\max\,|M_{3\mathrm{D}}[(x,y,v_x,v_y)] - M_{\rm planar}|$",
    )
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"in-plane block difference")
    ax.set_title("(d) planar reduction" if PAPER
                 else "(d) 2D-reduction bridge (planar IC)", fontsize=10)
    ax.set_ylim(1e-14, 1e-5)
    if not PAPER:  # one line; the y-label names it
        ax.legend(fontsize=7.5, loc="lower right")
        ax.text(
            0.03, 0.92,
            f"out-of-plane leakage = {pd_['max_leak_err']:.0f} (exact)",
            transform=ax.transAxes, fontsize=7, color="0.4", ha="left",
        )
        env = d["env"]
        fig.text(
            0.005, 0.005,
            f"galpy {env['galpy_branch']}@{env['galpy_commit']}"
            f"{'+local-fix' if env['galpy_dirty'] else ''}, "
            f"MWPotential2014, dop853_c rtol=atol=1e-12",
            fontsize=6, color="0.55",
        )
    fig.tight_layout()
    return fig
