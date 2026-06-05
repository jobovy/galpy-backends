"""G5 -- the cost of differentiability: gradient throughput vs value throughput.

The paper's central performance question for an autodiff library: how much more
expensive is the *gradient* than the *value*? For reverse-mode AD the gradient of
a scalar w.r.t. all N inputs costs a small constant factor over the forward pass
(independent of N) -- so differentiability is cheap. This figure measures it
directly for a vectorised Plummer evaluation on the GPU:

  value : Phi(R, z)            over N points
  grad  : d/dR sum_i Phi(R_i)  -- the per-point dPhi/dR, one reverse-mode pass

(a) throughput (points/s) vs N, value (solid) vs gradient (dashed), jax & torch.
(b) the overhead ratio (grad wall-time / value wall-time) vs N -- a small, roughly
    N-independent constant once past the launch-bound small-N regime.

Reproducibility: timings -> data/g5_autodiff_overhead.json (machine + versions
stamped); the plot is byte-stable from the JSON. Re-measure in the GPU env.
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy

from paper.figures._harness import bench
from paper.figures._harness.figure import figure
from paper.figures._harness.style import BACKEND_COLORS

_DATA = os.path.join(os.path.dirname(__file__), "data")
_JSON = os.path.join(_DATA, "g5_autodiff_overhead.json")

# solid = value, dashed = gradient; colour per backend.
_STYLE = {
    "jax value": (BACKEND_COLORS["jax"], "-"),
    "jax grad": (BACKEND_COLORS["jax"], "--"),
    "torch value": (BACKEND_COLORS["torch"], "-"),
    "torch grad": (BACKEND_COLORS["torch"], "--"),
}


def _grid(N):
    return numpy.linspace(0.3, 3.0, N), 0.1 * numpy.ones(N)


def _measure():
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import torch

    torch.set_default_dtype(torch.float64)
    from galpy.potential import PlummerPotential, evaluatePotentials

    p = PlummerPotential(normalize=1.0)
    Ns = [int(10**k) for k in range(2, 8)]
    series = {}

    def med(fn, **kw):
        return bench.time_call(fn, **kw)["median"]

    if any(d.platform == "gpu" for d in jax.devices()):
        val, grd = [], []
        for N in Ns:
            Rn, zn = _grid(N)
            Rj, zj = jnp.asarray(Rn), jnp.asarray(zn)
            vfun = jax.jit(lambda R, z: evaluatePotentials(p, R, z))
            gfun = jax.jit(jax.grad(lambda R, z: jnp.sum(evaluatePotentials(p, R, z))))
            vfun(Rj, zj).block_until_ready()  # compile
            gfun(Rj, zj).block_until_ready()
            val.append(med(lambda: vfun(Rj, zj), warmup=3, repeats=9, sync="jax"))
            grd.append(med(lambda: gfun(Rj, zj), warmup=3, repeats=9, sync="jax"))
        series["jax value"] = val
        series["jax grad"] = grd
    if torch.cuda.is_available():
        val, grd = [], []
        for N in Ns:
            Rn, zn = _grid(N)
            Rt = torch.as_tensor(Rn, device="cuda")
            zt = torch.as_tensor(zn, device="cuda")

            def grad_call():
                R = Rt.detach().requires_grad_(True)
                evaluatePotentials(p, R, zt).sum().backward()
                return R.grad

            val.append(
                med(lambda: evaluatePotentials(p, Rt, zt), warmup=3, repeats=9, sync="torch")
            )
            grd.append(med(grad_call, warmup=3, repeats=9, sync="torch"))
        series["torch value"] = val
        series["torch grad"] = grd

    env = {"python": sys.version.split()[0]}
    for m in ("numpy", "jax", "torch", "galpy"):
        try:
            env[m] = __import__(m).__version__
        except Exception:
            pass
    try:
        env["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    except Exception:
        env["gpu"] = "?"
    data = {"Ns": Ns, "series": series, "env": env}
    os.makedirs(_DATA, exist_ok=True)
    with open(_JSON, "w") as fh:
        json.dump(data, fh, indent=2)
    return data


@figure(
    name="g5_autodiff_overhead",
    track="G",
    seed=0,
    description="gradient vs value throughput (reverse-mode AD overhead), jax/torch GPU [measure->json, plot byte-stable]",
)
def make():
    if os.path.exists(_JSON):
        with open(_JSON) as fh:
            data = json.load(fh)
    else:
        data = _measure()
    Ns = numpy.asarray(data["Ns"], dtype=float)
    series = data["series"]
    env = data.get("env", {})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 4.0))

    # ---- (a) throughput: value (solid) vs grad (dashed) ------------------------
    for name in [k for k in _STYLE if k in series]:
        col, ls = _STYLE[name]
        thr = Ns / numpy.asarray(series[name], dtype=float)
        ax1.loglog(Ns, thr, marker="o", ms=3.5, color=col, ls=ls, label=name, zorder=3)
    ax1.set_xlabel(r"array length $N$")
    ax1.set_ylabel(r"throughput  (points / second)")
    ax1.set_title(r"(a) value (solid) vs gradient (dashed)")
    ax1.legend(fontsize=7.5, loc="lower right", framealpha=0.92, handlelength=3.0)

    # ---- (b) overhead ratio: grad time / value time ----------------------------
    for backend in ("jax", "torch"):
        v, g = f"{backend} value", f"{backend} grad"
        if v in series and g in series:
            ratio = numpy.asarray(series[g], dtype=float) / numpy.asarray(series[v], dtype=float)
            ax2.semilogx(Ns, ratio, marker="o", ms=4, color=BACKEND_COLORS[backend],
                         label=f"{backend}: grad / value", zorder=3)
    ax2.axhline(1.0, ls=":", lw=1, color="0.6")
    ax2.text(Ns[0], 1.04, "value cost", fontsize=7, color="0.5", va="bottom")
    ax2.set_xlabel(r"array length $N$")
    ax2.set_ylabel(r"gradient wall-time / value wall-time")
    ax2.set_title("(b) reverse-mode autodiff overhead")
    ax2.set_ylim(0, None)
    ax2.legend(fontsize=8, loc="upper right", framealpha=0.92)
    ax2.text(
        0.03, 0.03,
        f"GPU: {env.get('gpu', '?')} · jax {env.get('jax', '?')} · torch {env.get('torch', '?')}",
        transform=ax2.transAxes, fontsize=6.5, color="0.5", va="bottom",
    )

    fig.suptitle(r"Cost of differentiability: Plummer $\Phi$ gradient vs value (GPU, float64)",
                 fontsize=11)
    fig.tight_layout()
    return fig
