"""G14 -- "The Great Escape": a hypervelocity star constrains the Milky Way, with
BOTH differentiable-orbit mechanisms composed.

S5-HVS1 (Koposov et al. 2019) was ejected by Sgr A*, so integrated backward it must
return to the Galactic Centre. Using the backward orbit's closest approach to the GC
as a differentiable objective:

  (a) C-STM initial-condition fit (potential fixed): recover the present-day phase
      space WITHIN the observational uncertainties that best returns to the GC.
      Gradients dx(t)/dx0 come from galpy's compiled C variational integrator (the
      state-transition matrix) -- fast, the natural tool for an IC fit.
  (b) in-backend AD potential fit: recover the halo flattening q by reverse-mode
      autodiff through the in-backend diffrax solve -- a PARAMETER gradient, which
      C-STM cannot provide. The closest-approach landscape vs q has its minimum at
      the true q; Adam (gradients through the solver) descends to it.

One science example, the two mechanisms composed: C-STM for the IC gradients, in-
backend AD for the parameter gradient. The fitted values + orbits are computed by
measure/measure_fit.py (deterministic; data/fit.json); this plot is byte-stable.
The mock is S5-HVS1-like (ejected from the GC); real S5-HVS1 data is a follow-on.
"""

import json
import pathlib

import matplotlib.pyplot as plt
import numpy

from paper.figures._harness.figure import figure

_DATA = pathlib.Path(__file__).resolve().parent / "data" / "fit.json"


@figure(
    name="g14_hvs_potential_fit",
    track="G",
    seed=0,
    description="HVS 'Great Escape' hybrid fit: C-STM IC fit + in-backend AD halo-flattening fit",
)
def make():
    d = json.loads(_DATA.read_text())
    _RO = 8.0  # kpc: galpy default ro; natural-unit positions -> physical kpc
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.3))

    # ---- (a) C-STM IC fit: backward orbits obs vs fitted, threading to the GC -----
    ob, ft = d["orbit_obs"], d["orbit_fit"]
    obx, oby = _RO * numpy.array(ob["x"]), _RO * numpy.array(ob["y"])
    ftx, fty = _RO * numpy.array(ft["x"]), _RO * numpy.array(ft["y"])
    ax1.plot(obx, oby, color="#bbbbbb", lw=1.3, label="observed IC")
    ax1.plot(ftx, fty, color="#0072B2", lw=1.8, label="C-STM-fitted IC")
    ax1.plot([0], [0], marker="*", ms=15, color="k", zorder=5)
    ax1.text(0.0, 0.0, "  Sgr A*", fontsize=7.5, va="center")
    ax1.set_xlabel("x  [kpc]")
    ax1.set_ylabel("y  [kpc]")
    ax1.set_title("(a) C-STM IC fit: backward orbit -> Galactic Centre")
    ax1.legend(fontsize=7.8, loc="best")
    ax1.set_aspect("equal", adjustable="datalim")
    # inset: zoom on the GC showing the closest-approach improvement (kpc)
    icf = d["ic_fit"]
    axin = ax1.inset_axes([0.58, 0.10, 0.38, 0.38])
    axin.plot(obx, oby, color="#bbbbbb", lw=1.3)
    axin.plot(ftx, fty, color="#0072B2", lw=1.3)
    axin.plot([0], [0], marker="*", ms=9, color="k")
    axin.set_xlim(-0.5, 0.5); axin.set_ylim(-0.5, 0.5)
    axin.set_title(f"closest: {_RO * icf['r_obs']:.2f}->{_RO * icf['r_fit']:.2f} kpc", fontsize=6.6)
    axin.tick_params(labelsize=5.5)

    # ---- (b) in-backend AD potential fit: GC closest-approach vs q + Adam path -----
    # black = the objective landscape (one backward integration per q on a grid);
    # green = the Adam optimizer descending it with AD gradients through the solver.
    qf = d["q_fit"]
    qs = numpy.array(qf["qs"])
    appr = numpy.sqrt(numpy.array(qf["losses"])) * _RO  # closest approach in kpc
    path_appr = numpy.sqrt(numpy.array(qf["path_loss"])) * _RO
    ax2.plot(qs, appr, color="#000000", lw=1.8, label="objective: grid scan (1 integration / $q$)")
    ax2.axvline(d["meta"]["q_true"], color="#D55E00", ls="--", lw=1.2, label=f"true $q$={d['meta']['q_true']}")
    ax2.plot(qf["path_q"], path_appr, "-o", color="#009E73", ms=3, lw=1.2,
             label="Adam path (AD through the solver)")
    ax2.plot([qf["recovered"]], [path_appr[-1]], marker="*", ms=13, color="#009E73",
             zorder=5, label=f"recovered $q$={qf['recovered']:.3f}")
    ax2.set_yscale("log")
    ax2.set_xlabel(r"halo flattening  $q$")
    ax2.set_ylabel("closest approach to the GC  [kpc]")
    ax2.set_title("(b) in-backend AD: recover the halo flattening")
    ax2.legend(fontsize=7.2, loc="best")

    fig.suptitle("Hypervelocity-star constraint on the Milky Way: C-STM IC fit + "
                 "in-backend-AD potential fit", fontsize=10.5)
    fig.tight_layout()
    return fig
