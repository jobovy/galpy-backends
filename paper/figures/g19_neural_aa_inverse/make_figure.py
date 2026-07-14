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


def load(backend):
    return numpy.load(os.path.join(HERE, f"results_{backend}.npz"), allow_pickle=True)


def main():
    dj, dt = load("jax"), load("torch")

    fig, (axr, axl, axe) = plt.subplots(1, 3, figsize=(15.5, 4.7))

    # (a) phase-space tori: numpy inverse (line) vs JAX-NN (circles)
    th = dj["th_a"]
    for k, (Jv, c) in enumerate(zip(dj["J_a"], TORI_C)):
        axr.plot(dj["z_ref_a"][k], dj["vz_ref_a"][k], "-", color=c, lw=2.2,
                 label=f"exact inverse, $J_z$={Jv:.3f}")
        p = dj["pred_a"][k]
        axr.plot(p[::10, 0], p[::10, 1], "o", color=c, ms=6, mfc="white", mew=1.5)
    axr.plot([], [], "o", color="0.35", mfc="white", mew=1.5, label="neural net $G(J,\\theta)$")
    axr.set_xlabel(r"$z$  [$R_0$]")
    axr.set_ylabel(r"$v_z$  [$v_0$]")
    axr.set_title("(a) learned AA inverse vs exact tori")
    axr.legend(fontsize=8, loc="upper right")
    axr.grid(alpha=0.25)

    # (b) training curves, JAX vs Torch (col 1 = step, col 2 = total loss)
    hj, ht = dj["hist"], dt["hist"]
    axl.semilogy(hj[:, 0], hj[:, 1], "-o", color=JAX_C, ms=4,
                 label=f"JAX (flax+optax), {float(dj['train_seconds']):.0f}s")
    axl.semilogy(ht[:, 0], ht[:, 1], "-s", color=TORCH_C, ms=4,
                 label=f"PyTorch (torch.nn), {float(dt['train_seconds']):.0f}s")
    axl.set_xlabel("training step")
    axl.set_ylabel("loss = data + physics-informed cycle")
    axl.set_title("(b) same physics, two autodiff backends\nbackprop through galpy's action-angle map")
    axl.legend(fontsize=8.5)
    axl.grid(alpha=0.25, which="both")

    # (c) inverse accuracy vs action
    axe.semilogy(dj["J_c"], dj["err_c"], "-o", color=JAX_C, ms=4, label="JAX")
    axe.semilogy(dt["J_c"], dt["err_c"], "-s", color=TORCH_C, ms=4, label="PyTorch")
    axe.set_xlabel(r"action  $J_z$")
    axe.set_ylabel("RMS phase-space error vs numpy inverse")
    axe.set_title("(c) accuracy across the action range")
    axe.legend(fontsize=9)
    axe.grid(alpha=0.25, which="both")

    fig.suptitle(
        "Learning the action-angle inverse of the Milky-Way vertical potential by "
        "differentiating through galpy  (JAX & PyTorch)",
        fontsize=12.5, y=1.02,
    )
    fig.tight_layout()
    out = os.path.join(HERE, "g19_neural_aa_inverse.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
