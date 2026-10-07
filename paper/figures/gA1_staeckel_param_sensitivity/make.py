"""gA1 -- Staeckel actions differentiated w.r.t. a potential parameter: the
halo flattening q of LogarithmicHaloPotential(normalize=1, q), by jax.grad
through galpy's pure-Python Staeckel code (actionAngleStaeckel(c=False)),
validated against central finite differences of the numpy path.

The potential AND the actionAngleStaeckel instance are rebuilt from q inside
the differentiated function, so dJ/dq flows through the normalisation, the
Staeckel setup (E, L_z, I_3), the turning-point root-finds and the action
quadrature. The finite differences rebuild everything from q +/- h as well.

 (a) J_R(q), J_z(q) of one Sun-like orbit (R=1, v_R=0.1, v_T=1.1, z=0.1,
     v_z=0.05) over q in [0.7, 1.1].
 (b) dJ_R/dq, dJ_z/dq: jax.grad (lines) vs numpy central finite differences
     (open circles) at every 5th q.
 (c) |AD - FD|/|FD| at q=0.9 as a function of the FD step h: the h^2 truncation
     branch, the round-off branch rising as h shrinks, and the AD value sitting
     at the floor of the V.

Pattern A: measure/measure.py runs against the galpy worktree and writes
data/gA1_staeckel_param_sensitivity.json (galpy sha + jax version stamped);
this make() is plot-only and byte-stable from that JSON.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy

from paper.figures._harness import layout
from paper.figures._harness.figure import figure

_DATA = os.path.join(os.path.dirname(__file__), "data")
_JSON = os.path.join(_DATA, "gA1_staeckel_param_sensitivity.json")
# journal version (default): no suptitle, short panel titles, no stamps; the
# caption carries the explanations. GALPY_JAX_PAPER=0 restores the annotated figure.
PAPER = os.environ.get("GALPY_JAX_PAPER", "1") == "1"
# one hue per action (Wong palette, as in g18): radial blue, vertical vermillion
_C = {"JR": "#0072B2", "Jz": "#D55E00"}
_LBL = {"JR": r"$J_R$", "Jz": r"$J_z$"}


def _load():
    with open(_JSON) as fh:
        return json.load(fh)


@figure(
    name="gA1_staeckel_param_sensitivity",
    track="A",
    seed=0,
    description="Staeckel actions (c=False) differentiated w.r.t. the halo flattening q: jax.grad dJ/dq vs numpy finite differences along a q sweep + the |AD-FD|/|FD| vs h V-curve [cross-env measure->json, plot byte-stable]",
)
def make():
    d = _load()
    sw, vc = d["sweep"], d["vcurve"]
    q = numpy.asarray(sw["q"])
    im = numpy.asarray(sw["fd_idx"], dtype=int)

    # print size: 7.0 in wide (figure*), text 8-10 pt (layout.FS)
    fig = plt.figure(figsize=layout.figsize(PAPER, (7.0, 3.3), (8.4, 4.2)))
    FS = layout.FS
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0], hspace=0.38, wspace=0.36,
                          left=0.10, right=0.985, bottom=0.13, top=0.93)
    axa = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[1, 0], sharex=axa)
    axc = fig.add_subplot(gs[:, 1])
    for ax in (axa, axb, axc):
        ax.tick_params(labelsize=FS["tick"])

    # ---- (a) the actions along the q sweep ------------------------------------
    for k in ("JR", "Jz"):
        axa.plot(q, numpy.asarray(sw[k]) * 1e3, color=_C[k], lw=1.8)
        axa.text(q[-1] + 0.012, sw[k][-1] * 1e3, _LBL[k], color=_C[k],
                 fontsize=FS["line"], va="center", ha="left")
    axa.set_ylabel(r"$J\ [10^{-3}]$", fontsize=FS["label"])
    axa.set_title("(a) Staeckel actions", loc="left")
    axa.tick_params(labelbottom=False)
    axa.set_xlim(q[0] - 0.01, q[-1] + 0.095)

    # ---- (b) dJ/dq: jax.grad lines vs FD circles -------------------------------
    for k in ("JR", "Jz"):
        ad = numpy.asarray(sw[f"d{k}_dq"]) * 1e3
        fd = numpy.asarray(sw[f"d{k}_dq_fd"]) * 1e3
        axb.plot(q, ad, color=_C[k], lw=1.8)
        axb.plot(q[im], fd, ls="none", marker="o", ms=5.5, mfc="none", mec=_C[k],
                 mew=1.2)
        axb.text(q[-1] + 0.012, ad[-1], r"$\partial %s/\partial q$" % _LBL[k].strip("$"),
                 color=_C[k], fontsize=FS["line"], va="center", ha="left")
    axb.set_xlabel(r"halo flattening $q$", fontsize=FS["label"])
    axb.set_ylabel(r"$\partial J/\partial q\ [10^{-3}]$", fontsize=FS["label"])
    axb.set_title("(b) flattening sensitivity", loc="left")
    axb.plot([], [], color="0.3", lw=1.8, label="jax.grad")
    axb.plot([], [], ls="none", marker="o", ms=5.5, mfc="none", mec="0.3", mew=1.2,
             label="finite diff.")
    axb.legend(fontsize=FS["legend"], loc="lower right", frameon=True,
               framealpha=0.9, handlelength=1.5, borderpad=0.35, labelspacing=0.3)

    # ---- (c) the V: |AD-FD|/|FD| vs the FD step h at q=q0 -----------------------
    h = numpy.asarray(vc["h"])
    for k in ("JR", "Jz"):
        rel = numpy.asarray(vc[f"{k}_relerr"])
        axc.loglog(h, rel, color=_C[k], lw=1.6, marker="o", ms=4.5,
                   label=r"$\partial %s/\partial q$" % _LBL[k].strip("$"))
    # guide slopes, each anchored to a measured J_R point: h^2 truncation (from
    # the h=1e-2 end) and 1/h round-off growth (from the h=1e-7 end)
    rel_r = numpy.asarray(vc["JR_relerr"])
    ht = numpy.logspace(numpy.log10(3e-6), numpy.log10(h[0]), 20)
    axc.loglog(ht, rel_r[0] * (ht / h[0]) ** 2 / 3.0, ls=":", color="0.45", lw=1.0)
    axc.text(3e-4, rel_r[0] * (3e-4 / h[0]) ** 2 / 3.0 / 2.2, r"$\propto h^2$",
             fontsize=FS["note"], color="0.4", ha="left", va="top")
    # (no 1/h round-off guide: with only the h=1e-7 point past the floor, a
    # slope line does not follow the data)
    floor = min(vc["JR_best_relerr"], vc["Jz_best_relerr"])
    axc.set_xlabel(r"finite-difference step $h$", fontsize=FS["label"])
    axc.set_ylabel(r"$|{\rm AD}-{\rm FD}|\,/\,|{\rm FD}|$", fontsize=FS["label"])
    axc.set_title("(c) autodiff vs finite difference", loc="left")
    axc.set_xlim(4e-8, 3e-2)
    axc.set_ylim(2e-11, 3e-3)
    axc.legend(fontsize=FS["legend"], loc="upper left", frameon=True, framealpha=0.9,
               handlelength=1.6, borderpad=0.35, labelspacing=0.3)
    axc.text(0.97, 0.04,
             r"$q=%.1f$; floor $\simeq10^{%d}$"
             % (vc["q0"], int(numpy.round(numpy.log10(floor)))),
             transform=axc.transAxes, fontsize=FS["note"], color="0.35", ha="right",
             va="bottom")

    if not PAPER:
        env = d["env"]
        fig.suptitle("Staeckel action sensitivity to the halo flattening (jax.grad vs "
                     "finite differences; CPU float64)", fontsize=10)
        fig.text(0.005, 0.005,
                 f"galpy {env['galpy_branch']}@{env['galpy_commit']}"
                 f"{'+dirty' if env['galpy_dirty'] else ''}, jax {env['jax']}; "
                 f"{d['config']['actionAngle']}", fontsize=6, color="0.55")
    return fig
