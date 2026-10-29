"""G2 -- force-evaluation throughput: CPU vs GPU, compute-only vs end-to-end.

The headline performance figure: vectorised potential-force evaluation throughput
(points/s) vs problem size N, for numpy (CPU) vs jax/torch (GPU). Two regimes are
shown per GPU backend:

Line convention (all backend figures, _harness/palette.py): colour = framework x
device (numpy black; jax blue shades, torch orange/red shades, darker = TITAN Xp
then H100), jit solid / eager dashed. Here jax is jit-compiled and torch eager;
end-to-end (+transfer) series are dotted with hollow markers. Panel (a) is the dev
box (TITAN Xp), panel (b) the same measurement on Trillium (H100, with numpy on
Trillium's CPU), from data/h100/g2_throughput.json.

  * compute-only: input array already resident on-device, result left
    on-device -- the timed wait (block_until_ready / cuda.synchronize) is for the
    kernel, not a copy. Best-case GPU throughput.
  * end-to-end (+transfer): each call starts from a host (numpy) array and returns a
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

from paper.figures._harness import bench, h100, layout, palette
from paper.figures._harness.figure import figure

_DATA = os.path.join(os.path.dirname(__file__), "data")
_JSON = os.path.join(_DATA, "g2_throughput.json")
# journal version (default): no suptitle / env stamps / long notes; the captions
# carry the explanations. GALPY_JAX_PAPER=0 restores the annotated figure.
PAPER = os.environ.get("GALPY_JAX_PAPER", "1") == "1"

# colour = framework x device (palette.py); jax is jit-compiled (solid), torch eager
# (dashed); end-to-end (+transfer) series are dotted with hollow markers.
_ORDER = ["numpy (CPU)", "jax (GPU)", "jax (GPU, +transfer)", "torch (GPU)", "torch (GPU, +transfer)"]
_MODE = {"numpy": None, "jax": "jit", "torch": "eager"}


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

    hdata = h100.load(os.path.dirname(__file__), "g2_throughput.json")
    panels = [("titan", data, "(a) TITAN Xp" if PAPER else "(a) dev box: NVIDIA TITAN Xp")]
    if hdata:
        panels.append(("h100", hdata, "(b) H100" if PAPER else "(b) Trillium: NVIDIA H100"))
    # print size: 7.0 in wide (figure*), text 8-10 pt; the two-line labels and the
    # device-free labels (the titles name the device) keep the label columns narrow
    fig, axes = plt.subplots(1, len(panels), sharey=True, squeeze=False,
                             figsize=layout.figsize(PAPER, (7.0, 3.2),
                                                    (5.2 * len(panels) + 1.4, 4.3)))
    fs_line = layout.FS["line"] if PAPER else 6.5
    fs_note = layout.FS["note"] if PAPER else 6.3
    for ax, (dev, d, title) in zip(axes[0], panels):
        Nd = numpy.asarray(d["Ns"], dtype=float)
        for name in [k for k in _ORDER if k in d["series"]]:
            fw = name.split()[0]
            transfer = "+transfer" in name
            st = palette.style(fw, "cpu" if fw == "numpy" else dev, _MODE[fw])
            if transfer:
                st.update(ls=":", mfc="white")
            # numpy's elementwise F_R runs on one core (no BLAS; measured cpu/wall = 1.0)
            extra = "+transfer" if transfer else ("1 core" if fw == "numpy" else "")
            lab = (layout.label(fw, None, _MODE[fw], extra) if PAPER
                   else palette.label(fw, dev, _MODE[fw], extra))
            thr = Nd / numpy.asarray(d["series"][name], dtype=float)
            ax.loglog(Nd, thr, ms=3.5, label=lab, zorder=3, **st)
        ax.set_xlabel(r"array length $N$  (points per $F_R$ call)")
        ax.set_title(title, fontsize=layout.FS["title"])
        e = d.get("env", {})
        if not PAPER:
            ax.text(0.97, 0.03, f"{e.get('gpu', '?')}\njax {e.get('jax', '?')} · torch {e.get('torch', '?')}"
                    "\nnumpy on this machine's CPU",
                    transform=ax.transAxes, fontsize=6.3, color="0.5", va="bottom", ha="right")
    axes[0][0].set_ylabel(r"throughput  (points / second)" if PAPER
                          else r"throughput  (points evaluated / second)")
    thr_all = [numpy.asarray(d["Ns"], float) / numpy.asarray(v, float)
               for _, d, _ in panels for v in d["series"].values()]
    lo = 0.4 * min(t.min() for t in thr_all)
    hi = 3.0 * max(t.max() for t in thr_all)
    for ax in axes[0]:
        ax.set_ylim(lo, hi)
    # constant-wall-time-per-call reference lines (throughput = N / t0, slope 1),
    # decade-spaced to bracket every curve's time per call at EVERY N (the
    # launch-bound small-N end of the H100 lines sits above 100 us per call)
    t_all = [t for _, d, _ in panels for v in d["series"].values()
             for t in numpy.asarray(v, dtype=float) if numpy.isfinite(t) and t > 0]
    for ax, (_, d, _) in zip(axes[0], panels):
        Nd = numpy.asarray(d["Ns"], dtype=float)
        refs = layout.cost_lines(ax, t_all, kind="rate", x_range=(Nd.min(), Nd.max()))
    axes[0][0].text(0.03, 0.97, "grey dotted, per call:\n" + layout.fmt_seconds(refs),
                    transform=axes[0][0].transAxes, fontsize=fs_note, color="0.5", va="top", ha="left")
    for ax in axes[0]:
        layout.faster_arrow(ax, "up", xy=(0.9, 0.05), length=0.16)
    if not PAPER:
        fig.suptitle(r"Plummer $F_R$ throughput: compute-only vs end-to-end (+transfer, dotted)",
                     fontsize=11)
    fig.tight_layout()
    for ax in axes[0]:
        palette.label_lines(ax, fontsize=fs_line, min_sep_pt=1.3 * fs_line if PAPER else None)
    if PAPER:
        layout.fit_right(fig)
    else:
        fig.tight_layout()
    return fig
