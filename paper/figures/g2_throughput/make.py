"""G2 -- force-evaluation throughput: CPU vs GPU, compute-only vs end-to-end.

The headline performance figure: vectorised potential-force evaluation throughput
(points/s) vs problem size N, for numpy (CPU) vs jax/torch (GPU). Two regimes are
shown per GPU backend:

  * compute-only (solid): input array already resident on-device, result left
    on-device -- the timed wait (block_until_ready / cuda.synchronize) is for the
    kernel, not a copy. Best-case GPU throughput.
  * end-to-end (dashed): each call starts from a host (numpy) array and returns a
    host array, so it includes host->device and device->host transfer -- the
    realistic cost of calling galpy on the GPU from a numpy workflow.

GPU launch/dispatch overhead dominates at small N (CPU wins); compute amortises at
large N (GPU wins by ~100x compute-only); but the round-trip transfer is PCIe-
bandwidth-bound (cost ~ N) and saturates the end-to-end throughput an order of
magnitude below compute-only -- so the GPU only pays off when data *stays* on
device across a chained pipeline, not for one-shot host calls.

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

# solid = compute-only, dashed = end-to-end (incl. host<->GPU transfer); colour per backend.
_STYLE = {
    "numpy (CPU)": ("#000000", "-"),
    "jax (GPU)": ("#0072B2", "-"),
    "jax (GPU, +transfer)": ("#0072B2", "--"),
    "torch (GPU)": ("#D55E00", "-"),
    "torch (GPU, +transfer)": ("#D55E00", "--"),
}


def _grid(N):
    return numpy.linspace(0.3, 3.0, N), 0.1 * numpy.ones(N)


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

    def med(fn, **kw):
        return bench.time_call(fn, **kw)["median"]

    # numpy (CPU) baseline
    series["numpy (CPU)"] = [
        med(lambda N=N: evaluateRforces(p, *_grid(N)), warmup=1, repeats=5) for N in Ns
    ]
    # jax (GPU): compute-only + end-to-end
    if any(d.platform == "gpu" for d in jax.devices()):
        co, e2e = [], []
        for N in Ns:
            Rn, zn = _grid(N)
            Rj, zj = jnp.asarray(Rn), jnp.asarray(zn)  # resident on device
            f = jax.jit(lambda R, z: evaluateRforces(p, R, z))
            f(Rj, zj).block_until_ready()  # compile
            co.append(med(lambda f=f, Rj=Rj, zj=zj: f(Rj, zj), warmup=3, repeats=9, sync="jax"))
            numpy.asarray(f(jnp.asarray(Rn), jnp.asarray(zn)))  # warm
            e2e.append(
                med(lambda f=f, Rn=Rn, zn=zn: numpy.asarray(f(jnp.asarray(Rn), jnp.asarray(zn))),
                    warmup=3, repeats=9)
            )
        series["jax (GPU)"] = co
        series["jax (GPU, +transfer)"] = e2e
    # torch (GPU): compute-only + end-to-end
    if torch.cuda.is_available():
        co, e2e = [], []
        for N in Ns:
            Rn, zn = _grid(N)
            Rt = torch.as_tensor(Rn, device="cuda")
            zt = torch.as_tensor(zn, device="cuda")
            co.append(
                med(lambda Rt=Rt, zt=zt: evaluateRforces(p, Rt, zt), warmup=3, repeats=9, sync="torch")
            )
            e2e.append(
                med(lambda Rn=Rn, zn=zn: evaluateRforces(
                        p, torch.as_tensor(Rn, device="cuda"), torch.as_tensor(zn, device="cuda")
                    ).cpu().numpy(), warmup=3, repeats=9)
            )
        series["torch (GPU)"] = co
        series["torch (GPU, +transfer)"] = e2e

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
    description="force-eval throughput vs N: numpy(CPU) vs jax/torch(GPU), compute-only vs end-to-end transfer [measure->json, plot byte-stable]",
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

    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    # compute-only first, then +transfer, so the legend groups sensibly
    order = [k for k in _STYLE if k in series]
    for name in order:
        col, ls = _STYLE[name]
        thr = Ns / numpy.asarray(series[name], dtype=float)
        ax.loglog(Ns, thr, marker="o", ms=3.5, color=col, ls=ls, label=name, zorder=3)
    ax.set_xlabel(r"array length $N$  (number of points per $F_R$ call)")
    ax.set_ylabel(r"throughput  (points evaluated / second)")
    ax.set_title(r"Plummer $F_R$: compute-only (solid) vs end-to-end (dashed)")
    # Constant-wall-time reference lines: throughput = N / t0 is a slope-1 line in
    # log-log. The GPU curves hug these at small N -- launch/overhead-bound, so the
    # wall-time per call is ~constant -- and peel off toward N/time scaling at large N.
    ax.set_ylim(*ax.get_ylim())  # freeze to the data range so the lines don't rescale it
    Nlo, Nhi = float(Ns.min()), float(Ns.max())
    for t0 in (1e-4, 1e-3, 1e-2, 1e-1):
        ax.plot([Nlo, Nhi], [Nlo / t0, Nhi / t0], ls=":", lw=0.8, color="0.78", zorder=0)
    ax.text(0.025, 0.975, "dotted: constant wall-time / call\n(0.1, 1, 10, 100 ms)",
            transform=ax.transAxes, fontsize=6.5, color="0.5", va="top", ha="left")
    # handlelength long enough that the dashed (+transfer) handles are distinguishable
    # from the solid (compute-only) ones -- the marker otherwise hides a short dash.
    ax.legend(fontsize=7, loc="lower right", framealpha=0.92, handlelength=3.0)
    ax.text(
        0.30, 0.025,
        f"GPU: {env.get('gpu', '?')}\njax {env.get('jax', '?')} · torch {env.get('torch', '?')}",
        transform=ax.transAxes, fontsize=6.5, color="0.5", va="bottom", ha="left",
    )
    fig.tight_layout()
    return fig
