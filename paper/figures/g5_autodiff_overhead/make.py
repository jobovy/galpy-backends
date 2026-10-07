"""G5 -- the cost of differentiability: gradient throughput vs value throughput.

The paper's central performance question for an autodiff library: how much more
expensive is the *gradient* than the *value*? For reverse-mode AD the gradient of
a scalar w.r.t. all N inputs costs a small constant factor over the forward pass
(independent of N) -- so differentiability is cheap. This figure measures it
directly for a vectorised Plummer evaluation on the GPU:

  value : Phi(R, z)            over N points
  grad  : d/dR sum_i Phi(R_i)  -- the per-point dPhi/dR, one reverse-mode pass

(a) throughput (points/s) vs N, value (filled markers) vs gradient (hollow), jax &
    torch, on the TITAN Xp and (data/h100/) an H100. Paper-wide line convention
    (_harness/palette.py): colour = framework x device, jit solid / eager dashed
    (jax is jit-compiled here, torch eager).
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

from paper.figures._harness import bench, h100, palette
from paper.figures._harness.figure import figure

_DATA = os.path.join(os.path.dirname(__file__), "data")
_JSON = os.path.join(_DATA, "g5_autodiff_overhead.json")

# colour = framework x device (palette.py); jax jit-compiled (solid), torch eager
# (dashed); value = filled markers, gradient = hollow markers.
_ORDER = ["jax value", "jax grad", "torch value", "torch grad"]
_MODE = {"jax": "jit", "torch": "eager"}


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

    hdata = h100.load(os.path.dirname(__file__), "g5_autodiff_overhead.json")
    runs = [("titan", data)] + ([("h100", hdata)] if hdata else [])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.3))

    # ---- (a) throughput: value (filled) vs grad (hollow) -----------------------
    for dev, d in runs:
        Nd = numpy.asarray(d["Ns"], dtype=float)
        for name in [k for k in _ORDER if k in d["series"]]:
            fw, kind = name.split()
            st = palette.style(fw, dev, _MODE[fw])
            if kind == "grad":
                st["mfc"] = "white"
            thr = Nd / numpy.asarray(d["series"][name], dtype=float)
            ax1.loglog(Nd, thr, ms=3.8, zorder=3, label=palette.label(fw, dev, _MODE[fw], kind), **st)
    ax1.set_xlabel(r"array length $N$")
    ax1.set_ylabel(r"throughput  (points / second)")
    ax1.set_title(r"(a) value (filled) vs gradient (hollow markers)", fontsize=10)

    # ---- (b) overhead ratio: grad time / value time ----------------------------
    for dev, d in runs:
        Nd = numpy.asarray(d["Ns"], dtype=float)
        for fw in ("jax", "torch"):
            v, g = f"{fw} value", f"{fw} grad"
            if v in d["series"] and g in d["series"]:
                ratio = numpy.asarray(d["series"][g], dtype=float) / numpy.asarray(d["series"][v], dtype=float)
                ax2.semilogx(Nd, ratio, ms=4, zorder=3, label=palette.label(fw, dev, _MODE[fw]),
                             **palette.style(fw, dev, _MODE[fw]))
    ax2.axhline(1.0, ls=":", lw=1, color="0.6")
    ax2.text(Ns[0], 0.94, "value cost", fontsize=7, color="0.5", va="top")
    ax2.set_xlabel(r"array length $N$")
    ax2.set_ylabel(r"gradient wall-time / value wall-time")
    ax2.set_title("(b) reverse-mode autodiff overhead", fontsize=10)
    ax2.set_ylim(0, None)
    stamp = "\n".join(f"{d['env'].get('gpu', '?')}: jax {d['env'].get('jax', '?')} · torch {d['env'].get('torch', '?')}"
                      for _, d in runs)
    ax2.text(0.97, 0.03, stamp, transform=ax2.transAxes, fontsize=6.3, color="0.5", va="bottom", ha="right")

    fig.suptitle(r"Cost of differentiability: Plummer $\Phi$ gradient vs value (TITAN Xp and H100 GPUs, float64)",
                 fontsize=11)
    fig.tight_layout(w_pad=11)
    palette.label_lines(ax1, fontsize=6.5)
    palette.label_lines(ax2, fontsize=6.5)
    return fig
