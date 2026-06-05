"""G6 -- jit compile-once / run-many amortization (jax, GPU).

A jit-compiled backend trades a one-time XLA compilation for fast steady-state
calls. This figure quantifies the trade for a vectorised Plummer F_R:

(a) per-call wall-time vs call index for a freshly-jitted function at two array
    sizes -- call 0 includes compilation (a spike), every later call runs the
    cached executable at the steady-state cost.
(b) cumulative wall-time vs number of calls, jit vs un-jit (eager): jit starts
    behind by the compile cost but has a shallower slope, so it overtakes eager
    after a break-even number of calls (marked).

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
from paper.figures._harness.style import BACKEND_COLORS

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
    jcol = BACKEND_COLORS["jax"]
    shades = {str(Ns[0]): jcol, str(Ns[1]): "#56B4E9"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 4.0))

    # ---- (a) per-call wall-time vs call index ----------------------------------
    for N in Ns:
        ts = numpy.asarray(per_call[str(N)], dtype=float)
        idx = numpy.arange(len(ts))
        ax1.semilogy(idx, ts * 1e3, marker="o", ms=4, color=shades[str(N)],
                     label=rf"$N=10^{{{int(round(numpy.log10(N)))}}}$")
        ax1.semilogy([1, len(ts) - 1], [eager[str(N)] * 1e3] * 2, ls=":", lw=1.2,
                     color=shades[str(N)])
    ax1.annotate("call 0 = compile", xy=(0, per_call[str(Ns[1])][0] * 1e3),
                 xytext=(2.2, per_call[str(Ns[1])][0] * 1e3 * 0.62), fontsize=7.5,
                 color="0.3", arrowprops=dict(arrowstyle="->", color="0.5", lw=0.8))
    ax1.text(0.97, 0.05, "dotted: eager (un-jit) per-call", transform=ax1.transAxes,
             fontsize=6.8, color="0.5", ha="right", va="bottom")
    ax1.set_xlabel("call index")
    ax1.set_ylabel("wall-time per call  (ms)")
    ax1.set_title("(a) compile-once, run-many")
    ax1.legend(fontsize=8, loc="upper right", framealpha=0.92)

    # ---- (b) cumulative time: jit vs eager, with break-even --------------------
    N = Ns[1]  # the larger size, where the GPU is fully engaged
    ts = numpy.asarray(per_call[str(N)], dtype=float)
    compile_plus = ts[0]
    steady = float(numpy.median(ts[1:]))
    eg = eager[str(N)]
    kstar = compile_plus / (eg - steady) if eg > steady else None  # break-even calls
    # extend the x-range comfortably past the crossover so the lines actually meet
    kmax = int(numpy.ceil(kstar * 1.5)) if kstar else 60
    k = numpy.arange(0, kmax)
    cum_jit = (compile_plus + steady * k) * 1e3  # k calls after the compiling one
    cum_eager = eg * (k + 1) * 1e3
    ax2.plot(k + 1, cum_jit, color=jcol, lw=2, label="jit (compile + run)")
    ax2.plot(k + 1, cum_eager, color="0.45", lw=1.5, ls="--", label="eager (un-jit)")
    if kstar:
        ystar = (compile_plus + steady * kstar) * 1e3
        ax2.axvline(kstar + 1, ls=":", lw=1, color="0.5")
        ax2.plot([kstar + 1], [ystar], marker="o", ms=6, color="k", zorder=5)
        ax2.text(kstar * 0.96, ystar * 1.06, f"break-even $\\approx${kstar + 1:.0f} calls",
                 fontsize=7.5, va="bottom", ha="right", color="0.2")
    ax2.set_xlabel("number of calls")
    ax2.set_ylabel("cumulative wall-time  (ms)")
    ax2.set_title(rf"(b) jit vs eager break-even ($N=10^{{{int(round(numpy.log10(N)))}}}$)")
    ax2.legend(fontsize=8, loc="upper left", framealpha=0.92)
    ax2.text(0.97, 0.03,
             f"GPU: {env.get('gpu', '?')} · jax {env.get('jax', '?')}",
             transform=ax2.transAxes, fontsize=6.5, color="0.5", ha="right", va="bottom")

    fig.suptitle(r"jit amortization: Plummer $F_R$ (GPU, float64)", fontsize=11)
    fig.tight_layout()
    return fig
