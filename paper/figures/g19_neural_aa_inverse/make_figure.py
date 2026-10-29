"""Render the Track G figure from results_{jax,torch}.npz (produced by
train_and_dump.py). Plot-only: fast and byte-stable.

A neural network learns the action-angle INVERSE of galpy's anharmonic 1D
vertical potential, trained by backpropagating through galpy's differentiable +
jittable forward action-angle map -- in both JAX and PyTorch.
"""

import os
import sys

import numpy
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))  # galpy-jax root
from paper.figures._harness import layout, palette  # noqa: E402
from paper.figures._harness.style import apply_style  # noqa: E402

# paper palette: both trainings ran on the TITAN Xp; jax with a jax.jit training
# step (solid), torch eager (dashed)
_ST = {"jax": palette.style("jax", "titan", "jit"), "torch": palette.style("torch", "titan", "eager")}
_LAB = {"jax": "jax jit", "torch": "torch eager"}
# tori: hues outside the framework colours (Tol muted: indigo, rose, green)
TORI_C = ["#332288", "#CC6677", "#117733"]
# journal version (default): no suptitle, short panel titles, no stamps; the
# caption carries the explanations. GALPY_JAX_PAPER=0 restores the annotated figure.
PAPER = os.environ.get("GALPY_JAX_PAPER", "1") == "1"


def load(backend):
    return numpy.load(os.path.join(HERE, f"results_{backend}.npz"), allow_pickle=True)


def main():
    dj, dt = load("jax"), load("torch")
    apply_style()  # the harness style (fonts, major grid) shared by every paper figure
    if PAPER:  # final printed width of a figure* (7.0 in): printed text sizes
        plt.rcParams.update({"axes.titlesize": layout.FS["title"], "axes.labelsize": layout.FS["label"],
                             "xtick.labelsize": layout.FS["tick"], "ytick.labelsize": layout.FS["tick"],
                             "savefig.bbox": "standard"})
    fig, (axr, axl, axe) = plt.subplots(1, 3, figsize=(7.0, 3.3) if PAPER else (15.5, 4.7))
    _fl = (lambda d: 8) if PAPER else (lambda d: d)  # legend sizes

    # (a) phase-space tori: numpy inverse (line) vs jax-NN (circles)
    for k, (Jv, c) in enumerate(zip(dj["J_a"], TORI_C)):
        axr.plot(dj["z_ref_a"][k], dj["vz_ref_a"][k], "-", color=c, lw=2.0,
                 label=None if PAPER else f"exact inverse, $J_z$={Jv:.3f}")
        p = dj["pred_a"][k]
        axr.plot(p[::10, 0], p[::10, 1], "o", color=c, ms=4.0 if PAPER else 6, mfc="white", mew=1.3)
    if not PAPER:
        axr.plot([], [], "o", color="0.35", mfc="white", mew=1.3, label="neural net $G(J,\\theta)$")
    axr.set_xlabel(r"$z$  [$R_0$]")
    axr.set_ylabel(r"$v_z$  [$v_0$]")
    axr.set_title("(a) learned vs exact tori" if PAPER else "(a) learned AA inverse vs exact tori")
    if PAPER:  # the three tori are the point, not their actions: a key, no legend
        lo, hi = axr.get_ylim()  # a little headroom for the key
        axr.set_ylim(lo, hi + 0.25 * (hi - lo))
        axr.text(0.03, 0.97, "lines: exact\ncircles: network", transform=axr.transAxes,
                 fontsize=7, color="0.35", va="top", ha="left", linespacing=1.1)
    else:
        axr.legend(fontsize=_fl(8), loc="upper center")

    # (b) training curves, jax vs torch (col 1 = step, col 2 = total loss)
    hj, ht = dj["hist"], dt["hist"]
    for fw, h, d in (("jax", hj, dj), ("torch", ht, dt)):
        lib = "flax+optax" if fw == "jax" else "torch.nn"
        lab = (f"{_LAB[fw]}, {float(d['train_seconds']):.0f} s" if PAPER
               else f"{_LAB[fw]} ({lib}), {float(d['train_seconds']):.0f} s")
        axl.semilogy(h[:, 0], h[:, 1], ms=4, label=lab, **_ST[fw])
    axl.set_xlabel("training step")
    axl.set_ylabel("loss = data +\nphysics-informed cycle" if PAPER
                   else "loss = data + physics-informed cycle")
    axl.set_title("(b) training loss" if PAPER else
                  "(b) same physics, two autodiff backends\nbackprop through galpy's action-angle map")
    if PAPER:  # headroom for the legend above the curves
        lo, hi = axl.get_ylim()
        axl.set_ylim(lo, hi * 1e4)
    axl.legend(fontsize=_fl(8.5), loc="upper right" if PAPER else "best",
               **(dict(handlelength=1.6, handletextpad=0.4, borderaxespad=0.3) if PAPER else {}))

    # (c) inverse accuracy vs action
    for fw, d in (("jax", dj), ("torch", dt)):
        axe.semilogy(d["J_c"], d["err_c"], ms=4, label=_LAB[fw], **_ST[fw])
    axe.set_xlabel(r"action  $J_z$")
    axe.set_ylabel("RMS phase-space error\nvs numpy inverse" if PAPER
                   else "RMS phase-space error vs numpy inverse")
    axe.set_title("(c) inverse accuracy" if PAPER else "(c) accuracy across the action range")
    axe.legend(fontsize=_fl(9), loc="upper left",
               **(dict(handlelength=1.6, borderaxespad=0.3) if PAPER else {}))

    if not PAPER:
        fig.suptitle(
            "Learning the action-angle inverse of the Milky-Way vertical potential by "
            "differentiating through galpy  (jax & torch)",
            fontsize=12.5, y=1.02,
        )
    fig.tight_layout()
    out = os.path.join(HERE, "g19_neural_aa_inverse.png")
    # PAPER: the page is exactly 7.0 in wide (no tight-bbox crop), so the 8-10 pt text
    # prints at size when included at the text width; 300 dpi
    fig.savefig(out, dpi=300 if PAPER else 130, **({} if PAPER else {"bbox_inches": "tight"}))
    print("wrote", out)


if __name__ == "__main__":
    main()
