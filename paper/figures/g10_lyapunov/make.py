"""G10 -- largest Lyapunov exponents from the variational equations (Track B).

Running Benettin estimates lambda(t) computed by Orbit.lyapunov (renormalized
deviation vectors propagated with the C variational equations; see
measure/lyapunov_measure.py, run against the feat/pvar-lyapunov galpy build):

 (a) Henon-Heiles at E=1/8: the documented chaotic orbit 'F' converges into
     the literature band lambda ~ 0.04-0.08 (Benettin, Galgani & Strelcyn
     1976; Skokos et al. 2002; Skokos 2010), while the regular orbit 'E' at
     the SAME energy decays ~ln(t)/t -- the classic chaos-detection contrast.
 (b) A regular Sun-like 3D orbit in MWPotential2014: lambda(t) decays ~ln(t)/t,
     i.e. the measurement is an upper limit that keeps tightening.

Cross-environment pattern (like g8): the measure step runs standalone against
the galpy worktree and writes data/lyapunov.json; this make() is plot-only and
byte-stable from that JSON.
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


def _load():
    with open(os.path.join(_DATA, "lyapunov.json")) as fh:
        return json.load(fh)


@figure(
    name="g10_lyapunov",
    track="G",
    seed=0,
    description="Largest-Lyapunov-exponent running estimates (Orbit.lyapunov): chaotic vs regular Henon-Heiles at E=1/8 with the literature band, and a regular MWPotential2014 orbit [cross-env measure->json, plot byte-stable]",
)
def make():
    d = _load()
    # print size: 7.0 in wide (figure*), text 8-10 pt
    fig, axes = plt.subplots(1, 2, figsize=layout.figsize(PAPER, (7.0, 3.0), (7.6, 3.3)))
    fs_leg = layout.FS["legend"] if PAPER else 7.5
    fs_note = layout.FS["note"] if PAPER else 7.5

    # ---- (a) Henon-Heiles ----
    ax = axes[0]
    hh = d["henon_heiles"]
    band = d["config"]["henon_heiles"]["literature_band"]
    t = numpy.array(hh["t"])
    lamF = numpy.array(hh["lam_chaotic"])
    lamE = numpy.array(hh["lam_regular"])
    ax.axhspan(band[0], band[1], color="#E69F00", alpha=0.25, lw=0, zorder=0)
    # print version: just above the band at the right (the legend moves to the
    # empty lower left), clear of every line
    ax.text(
        *((3.0e4, 0.095) if PAPER else (3.2e3, 0.062)),
        r"literature $\lambda\simeq0.04$--$0.08$",
        fontsize=fs_note, color="#9c6c00", va="bottom" if PAPER else "center",
        ha="right" if PAPER else "left",
    )
    ax.loglog(t[1:], lamF[1:], color="#D55E00",
              label="orbit F (chaotic)", zorder=3)
    ax.loglog(t[1:], lamE[1:], color="#0072B2",
              label="orbit E (regular)", zorder=3)
    ax.loglog(t[1:], numpy.log(t[1:]) / t[1:], ls="--", lw=1.0, color="0.4",
              label=r"$\ln(t)/t$", zorder=2)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"running estimate $\lambda(t)$")
    ax.set_title("(a) Hénon–Heiles, $E=1/8$", fontsize=10)
    ax.set_xlim(10.0, 4e4)
    ax.set_ylim(1e-4, 1.0)
    ax.legend(fontsize=fs_leg, loc="lower left" if PAPER else "upper right")
    # right-aligned just inside the frame, above the ln(t)/t line (which its left
    # end may touch: inside the frame matters more, JB)
    ax.annotate(
        rf"$\lambda(3\times10^4)={hh['lam_chaotic_final']:.3f}$",
        xy=(t[-1], lamF[-1]), xytext=(3.3e4, 6e-3), fontsize=fs_note, ha="right",
        arrowprops=dict(arrowstyle="->", lw=0.8, color="0.3"), color="0.3",
    )

    # ---- (b) MWPotential2014 regular orbit ----
    ax = axes[1]
    mw = d["mw2014"]
    t3 = numpy.array(mw["t"])
    lam3 = numpy.array(mw["lam"])
    ax.loglog(t3[1:], lam3[1:], color="#009E73",
              label="Sun-like 3D orbit", zorder=3)
    ax.loglog(t3[1:], numpy.log(t3[1:]) / t3[1:], ls="--", lw=1.0, color="0.4",
              label=r"$\ln(t)/t$", zorder=2)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"running estimate $\lambda(t)$")
    ax.set_title("(b) MWPotential2014, regular orbit", fontsize=10)
    ax.set_xlim(2.0, 4e3)
    ax.set_ylim(1e-3, 1.0)
    ax.legend(fontsize=fs_leg, loc="upper right")
    lam_mantissa = mw["lam_final"] / 10.0 ** numpy.floor(
        numpy.log10(mw["lam_final"])
    )
    lam_exp = int(numpy.floor(numpy.log10(mw["lam_final"])))
    ax.text(
        0.03, 0.05,
        rf"$\lambda(3\times10^3)={lam_mantissa:.1f}\times10^{{{lam_exp}}}$"
        + ("" if PAPER else "\n" r"(upper limit, still $\propto \ln(t)/t$)"),
        transform=ax.transAxes, fontsize=fs_note, color="0.3",
    )

    if not PAPER:
        env = d["env"]
        fig.text(
            0.005, 0.005,
            f"galpy {env['galpy_branch']}@{env['galpy_commit']}"
            f"{'+local-fix' if env['galpy_dirty'] else ''}, "
            "Orbit.lyapunov (dop853_c, renorm_every=10)",
            fontsize=6, color="0.55",
        )
    fig.tight_layout()
    return fig
