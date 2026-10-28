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
from paper.figures._harness import h100, layout, palette

_DATA = os.path.join(os.path.dirname(__file__), "data")
_JSON = os.path.join(_DATA, "g6_jit_amortization.json")
_NCALLS = 15
# journal version (default): no suptitle, short panel titles, no stamps; the
# caption carries the explanations. GALPY_JAX_PAPER=0 restores the annotated figure.
PAPER = os.environ.get("GALPY_JAX_PAPER", "1") == "1"


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


def _load_json(path):
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return None


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
    hdata = h100.load(os.path.dirname(__file__), "g6_jit_amortization.json")
    # (2026-10-28, JB's review) one CPU core (measure_cpu.py) and torch.compile
    # (measure_torch.py; on the TITAN Xp torch.compile's inductor backend refuses
    # the card -- Triton needs compute capability >= 7.0 -- so only its eager
    # cost is recorded, and the compiled line comes from the H100)
    cdata = _load_json(os.path.join(_DATA, "cpu", "g6_jit_amortization.json"))
    tdata = _load_json(os.path.join(_DATA, "torch", "g6_jit_amortization.json"))
    thdata = _load_json(os.path.join(_DATA, "h100", "torch", "g6_jit_amortization.json"))
    runs = [r for r in (("jax", "cpu", cdata), ("jax", "titan", data), ("jax", "h100", hdata),
                        ("torch", "titan", tdata), ("torch", "h100", thdata)) if r[2]]
    N = max(Ns)  # the larger size, where the GPU is fully engaged

    def _nlab(N):
        return rf"$N=10^{{{int(round(numpy.log10(N)))}}}$"

    # print size: 7.0 in wide (figure*), text 8-10 pt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=layout.figsize(PAPER, (7.0, 3.2), (11.6, 4.3)))
    fs_line = layout.FS["line"] if PAPER else 6.5
    fs_note = layout.FS["note"] if PAPER else 7.5

    def _lab(fw, dev, mode, extra=""):
        if PAPER:
            return layout.label(fw, dev, mode, extra)
        return palette.label(fw, dev, mode, extra)

    def _mode(fw, compiled):
        if compiled:
            return "jit" if fw == "jax" else "compiled"
        return "eager"

    # ---- (a) per-call wall-time vs call index ----------------------------------
    # compiled solid, eager (per-call) dashed, at the larger N only (the N=1e4
    # series, measured too, told the same story and cluttered the panel)
    for fw, dev, d in runs:
        if str(N) in d.get("per_call", {}):
            ts = numpy.asarray(d["per_call"][str(N)], dtype=float)
            st = palette.style(fw, dev, "jit")
            ax1.semilogy(numpy.arange(len(ts)), ts * 1e3, ms=4, label=_lab(fw, dev, _mode(fw, True)), **st)
        se = palette.style(fw, dev, "eager")
        se.pop("marker")
        ax1.semilogy([0, _NCALLS - 1], [d["eager"][str(N)] * 1e3] * 2, lw=1.1,
                     label=_lab(fw, dev, "eager"), **se)
    ax1.annotate("call 0 = compile", xy=(0, data["per_call"][str(N)][0] * 1e3),
                 xytext=(2.2, data["per_call"][str(N)][0] * 1e3 * 0.62), fontsize=fs_note,
                 color="0.3", arrowprops=dict(arrowstyle="->", color="0.5", lw=0.8))
    ax1.set_xlabel("call index")
    ax1.set_ylabel("wall-time per call  (ms)")
    ax1.set_title(rf"(a) per-call wall-time, {_nlab(N)}" if PAPER
                  else "(a) compile-once, run-many (eager per-call dashed)", fontsize=layout.FS["title"])

    # ---- (b) cumulative time in units of the device's eager call, so every
    # device shares one linear axis: eager is the diagonal, a compiled line
    # starts at its compile cost (in eager calls) and rises at its steady cost
    # per call; the break-even is where it crosses the diagonal ----------------
    stats = []
    for fw, dev, d in runs:
        if str(N) not in d.get("per_call", {}):
            continue
        ts = numpy.asarray(d["per_call"][str(N)], dtype=float)
        compile_plus = ts[0]
        steady = float(numpy.median(ts[1:]))
        eg = d["eager"][str(N)]
        kstar = compile_plus / (eg - steady) if eg > steady else None  # break-even calls
        stats.append((fw, dev, compile_plus, steady, eg, kstar))
    # extend the x-range comfortably past the crossovers so the lines actually meet
    kmax = int(numpy.ceil(max(k for *_, k in stats if k) * 1.5)) if any(k for *_, k in stats) else 60
    k = numpy.arange(0, kmax)
    ax2.plot(k + 1, k + 1, lw=1.5, ls="--", color="0.45", label="eager (any device)")
    for fw, dev, compile_plus, steady, eg, kstar in stats:
        ax2.plot(k + 1, (compile_plus + steady * k) / eg, lw=2,
                 label=_lab(fw, dev, _mode(fw, True), "compile + run"),
                 **{**palette.style(fw, dev, "jit"), "marker": "none"})
        if kstar:
            ystar = (compile_plus + steady * kstar) / eg
            col = palette.style(fw, dev)["color"]
            ax2.plot([kstar + 1], [ystar], marker="o", ms=6, color=col, zorder=5, label="_be")
            if PAPER:  # two lines, right of the crossing and below the compiled line
                xr = (kstar + 1) + 0.03 * kmax
                ax2.text(xr, ystar * 0.97, f"break-even\n$\\approx${kstar + 1:.0f} calls",
                         fontsize=fs_note, color=col, va="top", ha="left")
            else:
                ax2.text(kstar * 0.97, ystar * 1.04, f"break-even $\\approx${kstar + 1:.0f} calls",
                         fontsize=fs_note, va="bottom", ha="right", color=col)
    ax2.set_xlabel("number of calls")
    ax2.set_ylabel("cumulative time  (eager calls)")
    ax2.set_xlim(0, kmax)
    ax2.set_ylim(0, kmax)
    ax2.set_title(rf"(b) break-even, {_nlab(N)}" if PAPER
                  else rf"(b) compiled vs eager break-even ({_nlab(N)})", fontsize=layout.FS["title"])
    if not PAPER:
        stamp = "\n".join(f"{fw} {d['env'].get('gpu', dev)}: {d['env'].get(fw, '?')}" for fw, dev, d in runs)
        ax2.text(0.97, 0.03, stamp, transform=ax2.transAxes, fontsize=6.3, color="0.5",
                 ha="right", va="bottom")
        fig.suptitle(r"compilation amortization: Plummer $F_R$ (jax and torch; CPU, TITAN Xp and H100, float64)", fontsize=11)
    fig.tight_layout(**({} if PAPER else {"w_pad": 11}))
    palette.label_lines(ax1, fontsize=fs_line, min_sep_pt=1.3 * fs_line)
    palette.label_lines(ax2, fontsize=fs_line, min_sep_pt=1.3 * fs_line if PAPER else None)
    if PAPER:
        layout.fit_right(fig)
    return fig
