"""G2 -- force-evaluation throughput: CPU vs GPU.

The headline performance figure: vectorised potential-force evaluation throughput
(evals/s) vs problem size N, for numpy (CPU) vs jax (GPU) vs torch (GPU). GPU
launch/dispatch overhead dominates at small N (CPU wins) and amortises at large N
(GPU wins) -- the crossover is the point of the multi-backend design.

Reproducibility: timings are machine-dependent, so the *measure* step writes
data/g2_throughput.json (stamped with the GPU + library versions) and the *plot*
step is byte-stable from that JSON. Re-measure by deleting the JSON and re-running
`run` in the GPU env; `verify` only checks the (deterministic) plot.
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy

from paper.figures._harness import bench
from paper.figures._harness.figure import figure

_DATA = os.path.join(os.path.dirname(__file__), "data")
_JSON = os.path.join(_DATA, "g2_throughput.json")
_COLORS = {"numpy (CPU)": "#000000", "jax (GPU)": "#0072B2", "torch (GPU)": "#D55E00",
           "jax (CPU)": "#56B4E9", "torch (CPU)": "#E69F00"}


def _measure():
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import torch

    torch.set_default_dtype(torch.float64)
    from galpy.potential import PlummerPotential, evaluateRforces

    p = PlummerPotential(normalize=1.0)
    Ns = [int(10**k) for k in range(2, 8)]  # 1e2 .. 1e7
    series = {}

    def grid(N, xp=numpy, device=None):
        R = numpy.linspace(0.3, 3.0, N)
        z = 0.1 * numpy.ones(N)
        if xp is numpy:
            return R, z
        if xp == "jax":
            return jnp.asarray(R), jnp.asarray(z)
        return torch.as_tensor(R, device=device), torch.as_tensor(z, device=device)

    # numpy (CPU)
    series["numpy (CPU)"] = [
        bench.time_call(lambda: evaluateRforces(p, *grid(N)), warmup=1, repeats=5)["median"]
        for N in Ns
    ]
    # jax (GPU)
    if any(d.platform == "gpu" for d in jax.devices()):
        res = []
        for N in Ns:
            R, z = grid(N, "jax")
            try:
                f = jax.jit(lambda R, z: evaluateRforces(p, R, z))
                f(R, z).block_until_ready()  # compile
                res.append(bench.time_call(lambda: f(R, z), warmup=3, repeats=9, sync="jax")["median"])
            except Exception:
                res.append(bench.time_call(lambda: evaluateRforces(p, R, z), warmup=3, repeats=9, sync="jax")["median"])
        series["jax (GPU)"] = res
    # torch (GPU)
    if torch.cuda.is_available():
        res = []
        for N in Ns:
            R, z = grid(N, "torch", device="cuda")
            res.append(bench.time_call(lambda: evaluateRforces(p, R, z), warmup=3, repeats=9, sync="torch")["median"])
        series["torch (GPU)"] = res

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
    name="g2_throughput",
    track="G",
    seed=0,
    description="force-eval throughput vs N: numpy(CPU) vs jax/torch(GPU) [measure->json, plot byte-stable]",
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

    fig, ax = plt.subplots(figsize=(5.4, 4.1))
    for name, times in series.items():
        thr = Ns / numpy.asarray(times, dtype=float)
        ax.loglog(Ns, thr, marker="o", ms=4, color=_COLORS.get(name), label=name, zorder=3)
    ax.set_xlabel(r"array length $N$  (number of points per $F_R$ call)")
    ax.set_ylabel(r"throughput  (points evaluated / second)")
    ax.set_title(r"Vectorized Plummer $F_R$: throughput vs array size")
    # Constant-wall-time reference lines: throughput = N / t0 is a slope-1 line in
    # log-log. The GPU curves hug these at small N -- launch/overhead-bound, so the
    # wall-time per call is ~constant -- and peel off toward N/time scaling at large N.
    ax.set_ylim(*ax.get_ylim())  # freeze to the data range so the lines don't rescale it
    Nlo, Nhi = float(Ns.min()), float(Ns.max())
    for t0 in (1e-4, 1e-3, 1e-2, 1e-1):
        ax.plot([Nlo, Nhi], [Nlo / t0, Nhi / t0], ls=":", lw=0.9, color="0.72", zorder=0)
    ax.text(0.025, 0.975, "dotted: constant wall-time / call\n(0.1, 1, 10, 100 ms)",
            transform=ax.transAxes, fontsize=6.5, color="0.5", va="top", ha="left")
    ax.legend(fontsize=8, loc="lower right", framealpha=0.92)
    ax.text(
        0.025, 0.025,
        f"GPU: {env.get('gpu', '?')}\njax {env.get('jax', '?')} · torch {env.get('torch', '?')}",
        transform=ax.transAxes, fontsize=6.5, color="0.5", va="bottom",
    )
    fig.tight_layout()
    return fig
