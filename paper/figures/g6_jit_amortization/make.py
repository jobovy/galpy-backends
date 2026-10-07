"""G6 -- jit compile-once / run-many amortization (jax, GPU).

A jit-compiled backend trades a one-time XLA compilation for fast steady-state
calls. This figure quantifies the trade for a vectorised Plummer F_R:

(a) per-call wall-time vs call index for a freshly-jitted function at two array
    sizes -- call 0 includes compilation (a spike), every later call runs the
    cached executable at the steady-state cost.
(b) cumulative wall-time vs number of calls, jit vs un-jit (eager): jit starts
    behind by the compile cost but has a shallower slope, so it overtakes eager
    after a break-even number of calls (marked).

Both panels show the dev box's TITAN Xp and (data/h100/) an H100 on Trillium, in the
paper-wide line convention (_harness/palette.py): colour = framework x device (jax
blue; darker = H100), jit solid / eager dashed; in (a) N=1e6 filled, N=1e4 hollow.

Reproducibility: timings -> data/g6_jit_amortization.json (machine + versions
stamped); the plot is byte-stable from the JSON.
"""

import json
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy

from paper.figures._harness.figure import figure
from paper.figures._harness import h100, palette

_DATA = os.path.join(os.path.dirname(__file__), "data")
_JSON = os.path.join(_DATA, "g6_jit_amortization.json")
_NCALLS = 15


def _grid(N):
    return numpy.linspace(0.3, 3.0, N), 0.1 * numpy.ones(N)


def _measure():
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from galpy.potential import PlummerPotential, evaluateRforces

    p = PlummerPotential(normalize=1.0)
    Ns = [int(1e4), int(1e6)]
    per_call = {}
    eager = {}

    for N in Ns:
        Rn, zn = _grid(N)
        Rj, zj = jnp.asarray(Rn), jnp.asarray(zn)
        # fresh jit -> call 0 compiles; subsequent calls hit the cached executable
        f = jax.jit(lambda R, z: evaluateRforces(p, R, z))
        ts = []
        for _ in range(_NCALLS):
            t0 = time.perf_counter()
            out = f(Rj, zj)
            jax.block_until_ready(out)
            ts.append(time.perf_counter() - t0)
        per_call[str(N)] = ts
        # eager (un-jit) steady-state per-call median over many calls
        g = lambda: jax.block_until_ready(evaluateRforces(p, Rj, zj))
        for _ in range(3):
            g()  # warmup
        et = []
        for _ in range(15):
            t0 = time.perf_counter()
            g()
            et.append(time.perf_counter() - t0)
        eager[str(N)] = float(numpy.median(et))

    env = {"python": sys.version.split()[0]}
    for m in ("numpy", "jax", "galpy"):
        try:
            env[m] = __import__(m).__version__
        except Exception:
            pass
    try:
        import torch

        env["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "?"
        env["torch"] = torch.__version__
    except Exception:
        env["gpu"] = "?"
    data = {"Ns": Ns, "per_call": per_call, "eager": eager, "env": env}
    os.makedirs(_DATA, exist_ok=True)
    with open(_JSON, "w") as fh:
        json.dump(data, fh, indent=2)
    return data


@figure(
    name="g6_jit_amortization",
    track="G",
    seed=0,
    description="jax jit compile-once/run-many amortization + jit-vs-eager break-even [measure->json, plot byte-stable]",
)
def make():
    if os.path.exists(_JSON):
        with open(_JSON) as fh:
            data = json.load(fh)
    else:
        data = _measure()
    Ns = data["Ns"]
    per_call = data["per_call"]
    eager = data["eager"]
    env = data.get("env", {})
    hdata = h100.load(os.path.dirname(__file__), "g6_jit_amortization.json")
    runs = [("titan", data)] + ([("h100", hdata)] if hdata else [])

    def _nlab(N):
        return rf"$N=10^{{{int(round(numpy.log10(N)))}}}$"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.3))

    # ---- (a) per-call wall-time vs call index ----------------------------------
    # jit solid, eager (un-jit) per-call dashed; N=1e6 filled, N=1e4 hollow markers
    for dev, d in runs:
        for N in d["Ns"]:
            ts = numpy.asarray(d["per_call"][str(N)], dtype=float)
            idx = numpy.arange(len(ts))
            st = palette.style("jax", dev, "jit")
            if N != max(d["Ns"]):
                st["mfc"] = "white"
            ax1.semilogy(idx, ts * 1e3, ms=4, label=palette.label("jax", dev, "jit", _nlab(N)), **st)
            se = palette.style("jax", dev, "eager")
            se.pop("marker")
            ax1.semilogy([0, len(ts) - 1], [d["eager"][str(N)] * 1e3] * 2, lw=1.1,
                         label=palette.label("jax", dev, "eager", _nlab(N)), **se)
    ax1.annotate("call 0 = compile", xy=(0, data["per_call"][str(Ns[1])][0] * 1e3),
                 xytext=(2.2, data["per_call"][str(Ns[1])][0] * 1e3 * 0.62), fontsize=7.5,
                 color="0.3", arrowprops=dict(arrowstyle="->", color="0.5", lw=0.8))
    ax1.set_xlabel("call index")
    ax1.set_ylabel("wall-time per call  (ms)")
    ax1.set_title("(a) compile-once, run-many (eager per-call dashed)", fontsize=10)

    # ---- (b) cumulative time: jit vs eager, with break-even --------------------
    N = Ns[1]  # the larger size, where the GPU is fully engaged
    stats = []
    for dev, d in runs:
        ts = numpy.asarray(d["per_call"][str(N)], dtype=float)
        compile_plus = ts[0]
        steady = float(numpy.median(ts[1:]))
        eg = d["eager"][str(N)]
        kstar = compile_plus / (eg - steady) if eg > steady else None  # break-even calls
        stats.append((dev, compile_plus, steady, eg, kstar))
    # extend the x-range comfortably past the crossovers so the lines actually meet
    kmax = int(numpy.ceil(max(k for *_, k in stats if k) * 1.5)) if any(k for *_, k in stats) else 60
    k = numpy.arange(0, kmax)
    for dev, compile_plus, steady, eg, kstar in stats:
        ax2.plot(k + 1, (compile_plus + steady * k) * 1e3, lw=2,
                 label=palette.label("jax", dev, "jit", "compile + run"),
                 **{**palette.style("jax", dev, "jit"), "marker": "none"})
        ax2.plot(k + 1, eg * (k + 1) * 1e3, lw=1.5, label=palette.label("jax", dev, "eager"),
                 **{**palette.style("jax", dev, "eager"), "marker": "none"})
        if kstar:
            ystar = (compile_plus + steady * kstar) * 1e3
            col = palette.style("jax", dev)["color"]
            ax2.plot([kstar + 1], [ystar], marker="o", ms=6, color=col, zorder=5, label="_be")
            ax2.text(kstar * 0.97, ystar * 1.04, f"break-even $\\approx${kstar + 1:.0f} calls",
                     fontsize=7.5, va="bottom", ha="right", color=col)
    ax2.set_xlabel("number of calls")
    ax2.set_ylabel("cumulative wall-time  (ms)")
    ax2.set_title(rf"(b) jit vs eager break-even ({_nlab(N)})", fontsize=10)
    stamp = "\n".join(f"{d['env'].get('gpu', '?')}: jax {d['env'].get('jax', '?')}" for _, d in runs)
    ax2.text(0.97, 0.03, stamp, transform=ax2.transAxes, fontsize=6.3, color="0.5",
             ha="right", va="bottom")

    fig.suptitle(r"jit amortization: Plummer $F_R$ (jax; TITAN Xp and H100 GPUs, float64)", fontsize=11)
    fig.tight_layout(w_pad=11)
    palette.label_lines(ax1, fontsize=6.5)
    palette.label_lines(ax2, fontsize=6.5)
    return fig
