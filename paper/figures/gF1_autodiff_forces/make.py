"""GF1 -- F_R by autodiff vs galpy's exact analytic force, for complicated potentials.

Every galpy potential ships a hand-coded ``Rforce``. Now that the potentials are
backend-agnostic, the same force can be obtained as ``-d(Phi)/dR`` by autodiff --
no derivative written by hand. This figure asks whether the two agree for the
potentials where the analytic force is genuinely awkward: a basis-function
expansion (SCF), a disc expansion (DiskSCF), a spiral pattern, a rotating bar,
and multi-component composites.

(a) F_R(R) from autodiff (lines) with the exact analytic force overplotted (open
    circles). They are indistinguishable, which is the point.
(b) The residual |AD - exact| / |exact|, log scale, against the float64 epsilon.
    Everything sits within a few ULP: autodiff is not an approximation to the
    analytic force here, it IS the analytic force, evaluated by the chain rule.

Cheap to compute, but cached anyway -- data/gF1_autodiff_forces.json -- so the
plot is byte-stable. Re-measure by deleting the JSON.
"""

import json
import os

import numpy

from paper.figures._harness.figure import figure

_DATA = os.path.join(os.path.dirname(__file__), "data")
_JSON = os.path.join(_DATA, "gF1_autodiff_forces.json")
_Z, _PHI = 0.15, 0.3
_EPS = numpy.finfo(numpy.float64).eps
# journal version (default): no suptitle, short panel titles, no stamps; the
# caption carries the explanations. GALPY_JAX_PAPER=0 restores the annotated figure.
PAPER = os.environ.get("GALPY_JAX_PAPER", "1") == "1"

_STYLE = {  # colour-blind-safe, one per potential
    "MWPotential2014": "#0072B2", "SCF (basis expansion)": "#D55E00",
    "DiskSCF": "#009E73", "SpiralArms": "#CC79A7",
    "DehnenBar (rotating)": "#E69F00", "MN + NFW": "#56B4E9",
}


def _measure():
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from galpy.potential import (
        DehnenBarPotential, DiskSCFPotential, MiyamotoNagaiPotential,
        MWPotential2014, NFWPotential, SCFPotential, SpiralArmsPotential,
        evaluatePotentials, evaluateRforces,
    )

    pots = {
        "MWPotential2014": MWPotential2014,
        "SCF (basis expansion)": SCFPotential(normalize=1.0, a=1.0),
        "DiskSCF": DiskSCFPotential(normalize=1.0),
        "SpiralArms": SpiralArmsPotential(amp=1.0, N=4),
        "DehnenBar (rotating)": DehnenBarPotential(),
        "MN + NFW": [MiyamotoNagaiPotential(normalize=0.6, a=0.5, b=0.05),
                     NFWPotential(normalize=0.4, a=2.0)],
    }
    Rs = numpy.geomspace(0.35, 3.0, 40)
    Rj = jnp.asarray(Rs)
    out = {"R": Rs.tolist(), "pots": {}}
    for name, P in pots.items():
        def phi(R, P=P):
            return evaluatePotentials(P, R, jnp.asarray(_Z), phi=jnp.asarray(_PHI),
                                      t=0.0, use_physical=False)
        ad = numpy.asarray(-jax.vmap(jax.grad(phi))(Rj), dtype=float)
        ex = numpy.asarray(evaluateRforces(P, Rj, jnp.asarray(_Z), phi=jnp.asarray(_PHI),
                                           t=0.0, use_physical=False), dtype=float)
        out["pots"][name] = {"ad": ad.tolist(), "exact": ex.tolist()}
    return out


@figure(
    name="gF1_autodiff_forces",
    track="G",
    seed=0,
    description="F_R by autodiff vs the exact analytic force for complicated potentials (SCF, DiskSCF, spiral, bar, composites): agreement at a few ULP of float64",
)
def make():
    import matplotlib.pyplot as plt

    if os.path.exists(_JSON):
        data = json.load(open(_JSON))
    else:
        data = _measure()
        os.makedirs(_DATA, exist_ok=True)
        with open(_JSON, "w") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
            fh.write("\n")

    R = numpy.array(data["R"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.1))

    for name, d in data["pots"].items():
        c = _STYLE.get(name, "0.4")
        ad = numpy.array(d["ad"]); ex = numpy.array(d["exact"])
        ax1.plot(R, ad, "-", color=c, lw=1.7, label=name, zorder=3)
        ax1.plot(R[::4], ex[::4], "o", mfc="none", mec=c, ms=5.5, mew=1.1, lw=0, zorder=4)
        rel = numpy.abs(ad - ex) / numpy.maximum(numpy.abs(ex), 1e-300)
        ax2.semilogy(R, numpy.maximum(rel, 1e-18), "-", color=c, lw=1.4, label=name)

    ax1.set_xscale("log")
    ax1.set_xlabel(r"$R$"); ax1.set_ylabel(r"$F_R$")
    ax1.set_title("(a) autodiff vs analytic" if PAPER else
                  r"lines: $-\partial\Phi/\partial R$ by autodiff   ·   circles: analytic")
    ax1.legend(fontsize=6.8, framealpha=0.92, loc="best")
    ax1.axhline(0, color="0.8", lw=0.7, zorder=0)

    ax2.axhline(_EPS, color="0.35", ls="--", lw=1.0, zorder=1)
    ax2.text(R[0] * 1.05, _EPS * 1.35, r"float64 $\epsilon$", fontsize=7.5, color="0.35")
    ax2.set_xscale("log"); ax2.set_ylim(1e-18, 1e-12)
    ax2.set_xlabel(r"$R$"); ax2.set_ylabel(r"$|{\rm AD}-{\rm exact}|\,/\,|{\rm exact}|$")
    ax2.set_title("(b) relative residual" if PAPER else "residual against the hand-coded force")
    ax2.grid(True, which="both", ls=":", lw=0.5, alpha=0.5)
    if not PAPER:  # duplicates the legend of (a)
        ax2.legend(fontsize=6.8, framealpha=0.92, loc="upper right", ncol=2)

    fig.tight_layout()
    return fig
