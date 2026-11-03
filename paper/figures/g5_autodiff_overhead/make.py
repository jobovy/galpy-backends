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

from paper.figures._harness import bench, h100, layout, palette
from paper.figures._harness.figure import figure

_DATA = os.path.join(os.path.dirname(__file__), "data")
_JSON = os.path.join(_DATA, "g5_autodiff_overhead.json")
# journal version (default): no suptitle, short panel titles, no stamps; the
# caption carries the explanations. GALPY_JAX_PAPER=0 restores the annotated figure.
PAPER = os.environ.get("GALPY_JAX_PAPER", "1") == "1"

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
    # (2026-10-28, JB's review) one CPU core as well (measure_cpu.py)
    _cpu = os.path.join(_DATA, "cpu", "g5_autodiff_overhead.json")
    cdata = json.load(open(_cpu)) if os.path.exists(_cpu) else None
    runs = ([("cpu", cdata)] if cdata else []) + [("titan", data)] + ([("h100", hdata)] if hdata else [])
    # print size: 7.0 in wide (figure*), text 8-10 pt. In (a) only the value lines
    # carry a label (the gradient line is the hollow-marker twin, said by a note),
    # and the labels are two-line, so the label columns stay narrow.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=layout.figsize(PAPER, (7.0, 3.2), (11.6, 4.3)))
    fs_line = layout.FS["line"] if PAPER else 6.5
    fs_note = layout.FS["note"] if PAPER else 7

    # ---- (a) throughput: value (filled) vs grad (hollow) -----------------------
    for dev, d in runs:
        Nd = numpy.asarray(d["Ns"], dtype=float)
        for name in [k for k in _ORDER if k in d["series"]]:
            fw, kind = name.split()
            st = palette.style(fw, dev, _MODE[fw])
            if kind == "grad":
                st["mfc"] = "white"
            thr = Nd / numpy.asarray(d["series"][name], dtype=float)
            if PAPER:
                lab = layout.label(fw, dev, _MODE[fw]) if kind == "value" else "_" + name
            else:
                lab = palette.label(fw, dev, _MODE[fw], kind)
            ax1.loglog(Nd, thr, ms=3.8, zorder=3, label=lab, **st)
    # constant-wall-time-per-call reference lines (throughput = N / t0, slope 1),
    # decade-spaced to bracket every curve's time per call at the largest N
    ax1.set_ylim(None, 3.0 * ax1.dataLim.y1)  # room above the fastest curve's reference
    # one decade faster than the bracket and two fewer at the slow end (JB,
    # round 3: 10 us in, 1 s and 10 s out -- they bracket only the slowest
    # gradient's end point), the note wrapped at the unit change so its last
    # line stays clear of the curves
    dec = layout.ref_decades(layout.last_costs(ax1, kind="rate"))
    refs = layout.cost_lines(ax1, [dec[0] / 10.0] + dec[:-2], kind="rate",
                             x_range=(Ns.min(), Ns.max()))
    if PAPER:
        ax1.text(0.03, 0.97, "filled: value, hollow: gradient\ngrey dotted, per call:\n"
                 + layout.fmt_seconds(refs, wrap=True), transform=ax1.transAxes,
                 fontsize=fs_note, color="0.5", va="top", ha="left")
        layout.faster_arrow(ax1, "up", xy=(0.9, 0.05), length=0.15)
    ax1.set_xlabel(r"array length $N$")
    ax1.set_ylabel(r"throughput  (points / second)")
    ax1.set_title("(a) throughput" if PAPER
                  else r"(a) value (filled) vs gradient (hollow markers)", fontsize=layout.FS["title"])

    # ---- (b) overhead ratio: grad time / value time ----------------------------
    for dev, d in runs:
        Nd = numpy.asarray(d["Ns"], dtype=float)
        for fw in ("jax", "torch"):
            v, g = f"{fw} value", f"{fw} grad"
            if v in d["series"] and g in d["series"]:
                ratio = numpy.asarray(d["series"][g], dtype=float) / numpy.asarray(d["series"][v], dtype=float)
                ax2.semilogx(Nd, ratio, ms=4, zorder=3,
                             label=(layout.label if PAPER else palette.label)(fw, dev, _MODE[fw]),
                             **palette.style(fw, dev, _MODE[fw]))
    ax2.axhline(1.0, ls=":", lw=1, color="0.6")
    ax2.text(Ns[0], 0.88, "value cost", fontsize=fs_note, color="0.5", va="top")
    ax2.set_xlabel(r"array length $N$")
    ax2.set_ylabel(r"gradient / value wall-time" if PAPER else r"gradient wall-time / value wall-time")
    ax2.set_title("(b) autodiff overhead" if PAPER
                  else "(b) reverse-mode autodiff overhead", fontsize=layout.FS["title"])
    ax2.set_ylim(0, None)
    if not PAPER:
        stamp = "\n".join(f"{d['env'].get('gpu', '?')}: jax {d['env'].get('jax', '?')} · torch {d['env'].get('torch', '?')}"
                          for _, d in runs)
        ax2.text(0.97, 0.03, stamp, transform=ax2.transAxes, fontsize=6.3, color="0.5", va="bottom", ha="right")
        fig.suptitle(r"Cost of differentiability: Plummer $\Phi$ gradient vs value (TITAN Xp and H100 GPUs, float64)",
                     fontsize=11)
    fig.tight_layout(**({} if PAPER else {"w_pad": 11}))
    sep = 1.3 * fs_line if PAPER else None
    palette.label_lines(ax1, fontsize=fs_line, min_sep_pt=sep)
    palette.label_lines(ax2, fontsize=fs_line, min_sep_pt=sep)
    if PAPER:
        layout.fit_right(fig)
    return fig
