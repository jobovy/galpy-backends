"""G16 -- the IC-Jacobian ∂x(t)/∂x0 computed four ways, and how well they agree.

Trust anchor for the differentiable-orbit machinery: the state-transition matrix
M(t)=∂x(t)/∂x0 of a representative orbit (MWPotential2014, 3 Gyr) computed by

  * C-STM     -- galpy's compiled C variational integrator (the reference),
  * finite-difference of the flow (central, through the same C integrator),
  * jax AD    -- jax.jacrev through the in-backend diffrax solve,
  * torch AD  -- torch.autograd through torchdiffeq,

all agree. (a) a representative element ∂R(t)/∂v_{R0} the four ways — visually
indistinguishable. (b) the per-time relative Frobenius residual of each method vs
C-STM (t>0): FD at 2e-10-6e-10 (same C integrator; the eps=1e-6 central-difference
truncation sets the floor), the AD paths at 2e-11-6e-10 (jax 3e-11-3e-10, torch
2e-11-6e-10; different integrators at rtol=atol=1e-10) -- no worse than FD despite
the different integrator. The autodiff gradients ARE the variational
state-transition matrix, to solver precision.

Reads data/precision.json (machine-stamped); plot byte-stable.
"""

import json
import os
import pathlib

import matplotlib.pyplot as plt
import numpy

from paper.figures._harness import layout
from paper.figures._harness.figure import figure

_DATA = pathlib.Path(__file__).resolve().parent / "data" / "precision.json"
# journal version (default): no suptitle, short panel titles; the caption carries
# the explanations. GALPY_JAX_PAPER=0 restores the annotated figure.
PAPER = os.environ.get("GALPY_JAX_PAPER", "1") == "1"
# galpy C = grey (palette.py): the C-STM reference mid grey, its FD a lighter grey
_C = {"cstm": "#666666", "fd": "#AAAAAA", "jax_ad": "#0072B2", "torch_ad": "#D55E00"}
_LAB = {"cstm": "C-STM (C variational)", "fd": "finite difference (C)",
        "jax_ad": "jax AD (diffrax)", "torch_ad": "torch AD (torchdiffeq)"}


@figure(
    name="g16_icjacobian_precision",
    track="G",
    seed=0,
    description="IC-Jacobian dx(t)/dx0 four ways (C-STM, FD, jax/torch AD) agree to ~1e-11..6e-10",
)
def make():
    d = json.loads(_DATA.read_text())
    ts = numpy.array(d["ts"])
    # print size: 7.0 in wide (figure*), text 8-10 pt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=layout.figsize(PAPER, (7.0, 3.0), (10.2, 4.1)))
    fs_leg = layout.FS["legend"] if PAPER else 7.8

    # ---- (a) one representative element, four ways (they coincide) -------------
    el = d["elem_dR_dvR0"]
    for k in ("cstm", "fd", "jax_ad", "torch_ad"):
        ax1.plot(ts, el[k], color=_C[k], lw=(2.6 if k == "cstm" else 1.3),
                 ls=("-" if k == "cstm" else "--"), label=_LAB[k],
                 alpha=(1.0 if k == "cstm" else 0.9))
    ax1.set_xlabel("time")
    ax1.set_ylabel(r"$\partial R(t)/\partial v_{R0}$")
    ax1.set_title("(a) one matrix element" if PAPER
                  else "(a) a state-transition-matrix element, 4 ways")
    ax1.legend(fontsize=fs_leg, loc="best")

    # ---- (b) relative Frobenius residual vs C-STM, per time --------------------
    rf = d["rel_frob"]
    # skip t=0: there M is exactly the identity for every method, so the residual
    # is machine-zero and would otherwise drag the log axis down to ~1e-16.
    for k in ("fd", "jax_ad", "torch_ad"):
        ax2.semilogy(ts[1:], numpy.array(rf[k])[1:], color=_C[k], lw=1.8,
                     marker="o", ms=3, label=f"{_LAB[k]} vs C-STM")
    ax2.set_xlabel("time")
    ax2.set_ylabel(r"$\|M_x-M_{\rm CSTM}\|_F\,/\,\|M_{\rm CSTM}\|_F$" if PAPER
                   else r"rel. Frobenius residual  $\|M_x-M_{\rm CSTM}\|/\|M_{\rm CSTM}\|$")
    ax2.set_title("(b) residual vs C-STM" if PAPER
                  else "(b) agreement with the C variational STM")
    if PAPER:  # headroom above the FD line for the legend, off the data
        ax2.set_ylim(1.2e-11, 7e-9)
        ax2.legend(fontsize=fs_leg, loc="upper left")
    else:
        ax2.legend(fontsize=fs_leg, loc="best")

    if not PAPER:
        fig.suptitle(r"$\partial x(t)/\partial x_0$: autodiff $=$ the C state-transition "
                     "matrix to solver precision (MWPotential2014)", fontsize=10.5)
    fig.tight_layout()
    return fig
