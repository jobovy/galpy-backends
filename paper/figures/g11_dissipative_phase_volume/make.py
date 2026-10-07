"""G11 -- the dissipative phase-volume law (Track B, dissipative variational eqns).

For a velocity-dependent force the variational Jacobian is the general
J = [[0, I], [K + dF/dx, dF/dv]] and Abel-Jacobi-Liouville gives

    det M(t) = exp( int_0^t tr(dF/dv) dt' )

-- the dissipative replacement of the Liouville invariant det M = 1.  Panels:

 (a) det M(t) from the 6x6 STM (integrate_dxdv canonical deviations) for a
     decaying orbit in MWPotential2014 + dynamical friction, overplotted with
     the prediction exp(int tr(dF/dv) dt') computed INDEPENDENTLY from the
     C-extension rectangular force Jacobian along the orbit; Chandrasekhar
     friction (5+ decades of contraction) and FDM friction at identical
     GMs (weaker contraction: quantum pressure suppresses the friction,
     Cfdm/Ccdm ~ 0.5-0.6 along this orbit).
 (b) the relative difference between the two sides of the law.

Cross-environment pattern (like g8): the measure steps run standalone against
the feat/pvar-dissip-struct (Chandrasekhar) and feat/pvar-tail-fdm (FDM) galpy
builds and write data/dissipative_{cdf,fdm}.json; this make() is plot-only and
byte-stable from those JSONs.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy

from paper.figures._harness import layout
from paper.figures._harness.figure import figure

_DATA = os.path.join(os.path.dirname(__file__), "data")
# journal version (default): no suptitle, short panel titles, no stamps; the
# caption carries the explanations. GALPY_JAX_PAPER=0 restores the annotated figure.
PAPER = os.environ.get("GALPY_JAX_PAPER", "1") == "1"

_SERIES = [
    ("dissipative_cdf.json", "Chandrasekhar friction", "#0072B2"),
    ("dissipative_fdm.json", "FDM friction ($m\\hbar^{-1}=10$)", "#D55E00"),
]


def _load(fname):
    with open(os.path.join(_DATA, fname)) as fh:
        return json.load(fh)


@figure(
    name="g11_dissipative_phase_volume",
    track="G",
    seed=0,
    description="Dissipative phase-volume law det M(t)=exp(int tr(dF/dv) dt'): STM determinant vs C-Jacobian trace integral for Chandrasekhar and FDM friction in MWPotential2014 [cross-env measure->json, plot byte-stable]",
)
def make():
    # print size: 7.0 in wide (figure*), text 8-10 pt
    fig, axes = plt.subplots(1, 2, figsize=layout.figsize(PAPER, (7.0, 3.0), (7.6, 3.3)))
    fs_leg = layout.FS["legend"] if PAPER else 7.5

    ax = axes[0]
    envs = []
    for fname, label, col in _SERIES:
        d = _load(fname)
        envs.append(d["env"])
        r = d["result"]
        t = numpy.array(r["t"])
        ax.semilogy(t, numpy.array(r["detM"]), color=col, label=label, zorder=3)
        ax.semilogy(
            t, numpy.array(r["pred"]), ls="--", lw=1.0, color="0.1",
            zorder=4, dashes=(4, 3),
        )
        axes[1].semilogy(
            t[1:], numpy.array(r["relerr"])[1:], color=col, label=label,
        )
    ax.plot([], [], ls="--", lw=1.0, color="0.1", dashes=(4, 3),
            label=r"$\exp\!\int_0^t \mathrm{tr}\,(\partial F/\partial v)\,\mathrm{d}t'$")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\det M(t)$")
    ax.set_title("(a) phase-space volume contraction", fontsize=10)
    ax.legend(fontsize=fs_leg, loc="lower left")

    ax = axes[1]
    ax.set_xlabel(r"$t$")
    ax.set_ylabel("relative difference")
    if PAPER:
        ax.set_title("(b) relative difference", fontsize=10)
    else:
        ax.set_title(
            r"(b) $|\det M - \exp\!\int \mathrm{tr}\,(\partial F/\partial v)\,"
            r"\mathrm{d}t'\,|\;/\;\exp\!\int \mathrm{tr}\,(\partial F/\partial v)\,"
            r"\mathrm{d}t'$",
            fontsize=9,
        )
    ax.set_ylim(1e-11, 1e-4)
    if not PAPER:  # duplicates the legend of (a)
        ax.legend(fontsize=fs_leg, loc="upper left")

    if not PAPER:
        env = envs[0]
        fig.text(
            0.005, 0.005,
            f"galpy {env['galpy_branch']}@{env['galpy_commit']} (Chandrasekhar) / "
            f"{envs[1]['galpy_branch']}@{envs[1]['galpy_commit']} (FDM); "
            "MWPotential2014 + friction (GMs=0.008), dopr54_c rtol=atol=1e-12",
            fontsize=6, color="0.55",
        )
    fig.tight_layout()
    return fig
