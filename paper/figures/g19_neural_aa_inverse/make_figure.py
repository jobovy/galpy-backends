"""Render the Track G figure from results_{jax,torch}.npz (produced by
train_and_dump.py). Plot-only: fast and byte-stable.

A neural network learns the action-angle INVERSE of galpy's anharmonic 1D
vertical potential, trained by backpropagating through galpy's differentiable +
jittable forward action-angle map -- in both JAX and PyTorch.
"""

import os

import numpy
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
JAX_C, TORCH_C = "#2a6f97", "#e07a5f"
TORI_C = ["#22577a", "#e07a5f", "#3d5a40"]
# journal version (default): no suptitle, short panel titles, no stamps; the
# caption carries the explanations. GALPY_JAX_PAPER=0 restores the annotated figure.
PAPER = os.environ.get("GALPY_JAX_PAPER", "1") == "1"


def load(backend):
    return numpy.load(os.path.join(HERE, f"results_{backend}.npz"), allow_pickle=True)


def main():
    dj, dt = load("jax"), load("torch")

    if PAPER:  # final printed width of a figure* (7.0 in): printed text sizes
        plt.rcParams.update({"axes.titlesize": 10, "axes.labelsize": 10,
                             "xtick.labelsize": 9, "ytick.labelsize": 9})
    fig, (axr, axl, axe) = plt.subplots(1, 3, figsize=(7.1, 3.3) if PAPER else (15.5, 4.7))
    _fl = (lambda d: 8) if PAPER else (lambda d: d)  # legend sizes

    # (a) phase-space tori: numpy inverse (line) vs JAX-NN (circles)
    th = dj["th_a"]
    for k, (Jv, c) in enumerate(zip(dj["J_a"], TORI_C)):
        axr.plot(dj["z_ref_a"][k], dj["vz_ref_a"][k], "-", color=c, lw=2.2,
                 label=f"$J_z$={Jv:.3f}" if PAPER else f"exact inverse, $J_z$={Jv:.3f}")
        p = dj["pred_a"][k]
        axr.plot(p[::10, 0], p[::10, 1], "o", color=c, ms=4.5 if PAPER else 6, mfc="white", mew=1.5)
    axr.plot([], [], "o", color="0.35", mfc="white", mew=1.5, label="neural net $G(J,\\theta)$")
    axr.set_xlabel(r"$z$  [$R_0$]")
    axr.set_ylabel(r"$v_z$  [$v_0$]")
    axr.set_title("(a) learned vs exact tori" if PAPER else "(a) learned AA inverse vs exact tori")
    if PAPER:  # headroom for the legend above the tori
        lo, hi = axr.get_ylim()
        axr.set_ylim(lo, lo + (hi - lo) / 0.55)
    axr.legend(fontsize=_fl(8), loc="upper right",
               **({"title": "exact inverse", "title_fontsize": 8} if PAPER else {}))
    axr.grid(alpha=0.25)

    # (b) training curves, JAX vs Torch (col 1 = step, col 2 = total loss)
    hj, ht = dj["hist"], dt["hist"]
    _sep = "\n" if PAPER else " "  # PAPER: two-line entries fit the narrow panel
    axl.semilogy(hj[:, 0], hj[:, 1], "-o", color=JAX_C, ms=4,
                 label=f"JAX{_sep}(flax+optax), {float(dj['train_seconds']):.0f}s")
    axl.semilogy(ht[:, 0], ht[:, 1], "-s", color=TORCH_C, ms=4,
                 label=f"PyTorch{_sep}(torch.nn), {float(dt['train_seconds']):.0f}s")
    axl.set_xlabel("training step")
    axl.set_ylabel("loss = data +\nphysics-informed cycle" if PAPER
                   else "loss = data + physics-informed cycle")
    axl.set_title("(b) training loss" if PAPER else
                  "(b) same physics, two autodiff backends\nbackprop through galpy's action-angle map")
    if PAPER:  # headroom for the legend above the curves
        lo, hi = axl.get_ylim()
        axl.set_ylim(lo, hi * 1e4)
    axl.legend(fontsize=_fl(8.5), loc="upper right" if PAPER else "best",
               **(dict(handlelength=1.2, handletextpad=0.4, borderaxespad=0.3) if PAPER else {}))
    axl.grid(alpha=0.25)

    # (c) inverse accuracy vs action
    axe.semilogy(dj["J_c"], dj["err_c"], "-o", color=JAX_C, ms=4, label="JAX")
    axe.semilogy(dt["J_c"], dt["err_c"], "-s", color=TORCH_C, ms=4, label="PyTorch")
    axe.set_xlabel(r"action  $J_z$")
    axe.set_ylabel("RMS phase-space error\nvs numpy inverse" if PAPER
                   else "RMS phase-space error vs numpy inverse")
    axe.set_title("(c) inverse accuracy" if PAPER else "(c) accuracy across the action range")
    axe.legend(fontsize=_fl(9))
    axe.grid(alpha=0.25)

    if not PAPER:
        fig.suptitle(
            "Learning the action-angle inverse of the Milky-Way vertical potential by "
            "differentiating through galpy  (JAX & PyTorch)",
            fontsize=12.5, y=1.02,
        )
    fig.tight_layout()
    out = os.path.join(HERE, "g19_neural_aa_inverse.png")
    fig.savefig(out, dpi=300 if PAPER else 130, bbox_inches="tight")  # >= 300 dpi at 7 in
    print("wrote", out)


if __name__ == "__main__":
    main()
