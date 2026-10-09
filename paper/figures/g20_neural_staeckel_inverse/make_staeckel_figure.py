"""Render the g20 figure from data/results_staeckel_jax.npz (produced by
train_and_dump.py on the H100). Plot-only: fast and byte-stable.

A neural network learns the inverse of the Staeckel action-angle map for
MWPotential2014 (no analytic inverse), trained by backpropagating through
galpy's differentiable + jittable forward action-angle map.
"""

import os
import sys

import numpy
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))  # galpy-jax root
from paper.figures._harness import palette  # noqa: E402

DUMP = os.path.join(HERE, "data", "results_staeckel_jax.npz")
ORB_C = ["#22577a", "#e07a5f", "#3d5a40"]
# journal version (default): no suptitle, short panel titles, no stamps; the
# caption carries the explanations. GALPY_JAX_PAPER=0 restores the annotated figure.
PAPER = os.environ.get("GALPY_JAX_PAPER", "1") == "1"


def main():
    d = numpy.load(DUMP, allow_pickle=True)
    JAX_C = palette.style("jax", str(d["device_kind"]), "jit")["color"]  # the training GPU

    if PAPER:  # final printed width of a figure* (7.0 in): printed text sizes
        plt.rcParams.update({"axes.titlesize": 10, "axes.labelsize": 9,
                             "xtick.labelsize": 9, "ytick.labelsize": 9})
    fig, (axo, axl, axr) = plt.subplots(
        1, 3, figsize=(7.0, 3.3) if PAPER else (15.5, 4.7),
        **({"width_ratios": [1.35, 1, 1]} if PAPER else {}))  # PAPER: room for (a)'s legend

    # (a) meridional-plane orbit reconstruction
    for k, c in enumerate(ORB_C):
        JRm, Jzm = float(d["orbit_JR"][k]), float(d["orbit_Jz"][k])
        axo.plot(d["orbit_R"][k], d["orbit_z"][k], "-", color=c, lw=1.0 if PAPER else 1.4,
                 alpha=0.55, label=f"{JRm:.3f}, {Jzm:.3f}" if PAPER
                 else f"orbit $J_R$={JRm:.3f},$J_z$={Jzm:.3f}")
        axo.plot(d["recon_R"][k][::7], d["recon_z"][k][::7], "o", color=c,
                 ms=2.2 if PAPER else 3.5, mfc="white", mew=0.7 if PAPER else 1.0)
    axo.plot([], [], "o", color="0.35", mfc="white", mew=0.7 if PAPER else 1.0,
             ms=2.2 if PAPER else 3.5, label="NN $G(J,\\theta)$" if PAPER
             else "NN inverse $G(J,\\theta)$")
    axo.set_xlabel(r"$R$  [$R_0$]"); axo.set_ylabel(r"$z$  [$R_0$]")
    if PAPER:
        axo.set_title("(a) orbit reconstruction")
        lo, hi = axo.get_ylim()  # headroom for the legend above the orbits
        axo.set_ylim(lo, lo + (hi - lo) / 0.62)
        axo.legend(fontsize=8, loc="upper center", ncol=2, handlelength=1.0,
                   handletextpad=0.3, columnspacing=0.6, borderaxespad=0.3, borderpad=0.3,
                   title=r"integrated ($J_R$, $J_z$)", title_fontsize=8)
    else:
        axo.set_title("(a) orbit reconstruction from actions+angles\n"
                      "(line=integrated, ○=neural inverse)")
        axo.legend(fontsize=7.5, loc="upper right")
    axo.grid(alpha=0.25)

    # (b) training curve
    h = d["hist"]
    axl.semilogy(h[:, 0], h[:, 1], "-o", color=JAX_C, ms=2.5 if PAPER else 4, label="total")
    axl.semilogy(h[:, 0], h[:, 2], "--", color="#7aa5c2", label="data")
    axl.semilogy(h[:, 0], h[:, 3], ":", color="#e07a5f",
                 label="cycle" if PAPER else "physics cycle")
    axl.set_xlabel("training step")
    axl.set_ylabel("loss")
    axl.set_title("(b) training loss" if PAPER else
                  f"(b) backprop through Staeckel AA (jax)\n"
                  f"trained in {float(d['train_seconds']):.0f}s on {d['device_kind']}")
    axl.legend(fontsize=8 if PAPER else 8.5)
    axl.grid(alpha=0.25)

    # (c) round-trip: recovered vs input action
    ok = d["rt_ok"]
    JR, Jz, JRh, Jzh = (d[k][ok] for k in ("JR_in", "Jz_in", "JR_rt", "Jz_rt"))
    rms = float(d["rms_dJ"])
    axr.plot([0, 0.2], [0, 0.2], "k-", lw=0.8, alpha=0.5)
    axr.plot(JR, JRh, ".", color=JAX_C, ms=1.2 if PAPER else 2, alpha=0.4, label="$J_R$")
    axr.plot(Jz, Jzh, ".", color="#e07a5f", ms=1.2 if PAPER else 2, alpha=0.4, label="$J_z$")
    axr.set_xlabel("input action  $J$")
    axr.set_ylabel(r"recovered  $F_J(G(J,\theta))$" if PAPER
                   else r"recovered action  $F_J(G(J,\theta))$")
    if PAPER:
        axr.set_title("(c) round-trip actions")
        axr.text(0.97, 0.04, f"RMS $\\Delta J$ = {rms:.1e}", transform=axr.transAxes,
                 ha="right", va="bottom", fontsize=9)
        axr.legend(fontsize=8, markerscale=6, loc="upper left")
    else:
        axr.set_title(f"(c) round-trip closure\nRMS $\\Delta J$ = {rms:.2e}")
        axr.legend(fontsize=9, markerscale=4)
    axr.grid(alpha=0.25)

    if not PAPER:
        fig.suptitle(
            "Neural inverse of the Staeckel action-angle map for MWPotential2014 "
            "(no analytic inverse) -- learned by differentiating through galpy",
            fontsize=12.5, y=1.02,
        )
    fig.tight_layout()
    out = os.path.join(HERE, "g20_neural_staeckel_inverse.png")
    fig.savefig(out, dpi=300 if PAPER else 130, bbox_inches="tight")  # >= 300 dpi at 7 in
    print(f"round-trip RMS dJ = {rms:.3e}; wrote {out}")


if __name__ == "__main__":
    main()
