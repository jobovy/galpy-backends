"""GD3 -- spherical-DF speed: sampling throughput and DF construction cost,
numpy vs jax vs torch, CPU and GPU.

Track F made galpy's spherical DFs (isotropic analytic families, eddingtondf,
osipkovmerrittdf, constantbetadf, kingdf) run natively on the jax/torch backends:
``sample(key=...)`` with a jax/torch key draws backend uniforms and evaluates the
radial and velocity inverse CDFs in-namespace, and the DF CONSTRUCTION (the f(E)
or f(Q) inversion / King ODE solve and the sampling tables) runs on the backend
when the potential's parameters are backend arrays, including under ``jax.jit``.
This figure measures what that costs and buys.

(a) Sampling throughput. Wall time of ``df.sample(n=N, return_orbit=False)``
    (positions AND velocities, N = 1e3 .. 1e7) for four DFs, each built ONCE
    from a potential with plain-float parameters and with its sampling tables
    already built (one-time setup, excluded here; see (b)):
      * isotropicHernquistdf: closed-form f(E) and inverse CMF;
      * eddingtondf on an NFW potential: numerical Eddington f(E), tabulated CMF;
      * osipkovmerrittdf on a Hernquist potential (r_a = 1.5): anisotropic f(Q);
      * kingdf (W0 = 5, M = 1, r_t = 10): ODE-solution tables.
    Series: numpy (``key=None``: the global numpy RNG and scipy's
    RectBivariateSpline inverse CDF -- galpy's historical, byte-identical path);
    jax eager and torch eager with a backend key (native bilinear inverse CDF);
    jax under the CALLER's ``jax.jit(lambda key: df.sample(n=N, key=key))``
    (compile excluded and recorded separately -- galpy adds no internal jit);
    and the same on the GPU (jax jit/eager with the key and computation on the
    GPU via ``jax.default_device``; torch with ``torch.device('cuda')`` as the
    default device -- a torch key draws on the CPU generator and the draws are
    moved to the default device, so a key gives the same particles on every
    device). On the GPU the DF's numpy tables are uploaded per eager call (tiny:
    ~1 MB) and baked in as constants under jit.

(b) Construction cost. Time to construct the DF from a potential whose
    parameter is given as a backend scalar (numpy: Python float) AND draw its
    first 1000 particles -- which is what builds the CMF, r(Phi) and p(v|r)
    sampling tables (galpy builds them lazily on the first ``sample``). This is
    the per-step cost of any model that rebuilds the DF when a parameter
    changes. Parameter: NFW/Hernquist scale a (eddingtondf, osipkovmerrittdf,
    constantbetadf with beta = -0.2) or King W0 (kingdf). Each (DF, backend)
    runs in a FRESH subprocess: outline = first call (for jax eager: per-op
    compilation; for jax.jit: trace + XLA compile), filled = warm (median of
    re-calls with new parameter values -- same shapes, so jit does not
    recompile). torch is eager (torch.compile is not used; it does not run on the
    sm_61 GPU this paper uses).

Float64 throughout (galpy's policy: ``jax_enable_x64``, and torch computes in
float64 internally). Timings: ``bench.time_call`` medians (warmup 1, 3
repeats; device-synchronised), a series stops once a call exceeds ~60 s. CPU:
pinned to 4 cores (``taskset``) of a SHARED 72-thread box; numpy/scipy are
single-threaded, jax uses XLA's CPU threadpool on those cores, and torch runs
with ONE intra-op thread (``GD3_TORCH_THREADS``, default 1): with 4 threads its
OpenMP workers spin-wait against busy hyperthread siblings and the timings were
5-15x slower and load-dependent (N = 1e5: 0.5 s at 4 threads vs 0.08 s at 1 on
the loaded box; 0.03 s at 4 on a quiet one). Absolute CPU numbers vary with the
box load by up to ~2x; the ratios between backends are the robust content.
Colours/line styles: the shared palette (paper/figures/_harness/palette.py) --
colour = framework x device (numpy black; jax sky blue / blue / navy and torch
orange / vermillion / dark red for CPU / TITAN Xp / H100), jit solid, eager dashed;
in (e) eager bars are white-hatched and the first call is the hollow outline. The
H100 series (NVIDIA H100 80GB, Trillium; jax 0.11.2, torch 2.14.1+cu130) come from
data/h100/gD3_spherical_df_speed.json, measured with the same script.
GPU series are measured only when a GPU is visible: re-measure them with
``python paper/figures/gD3_spherical_df_speed/make.py --gpu`` (merges into the
JSON), then ``python -m paper.figures._harness run gD3_spherical_df_speed``.

Reproducibility: timings are machine-dependent, so measure -> data/gD3_spherical_df_speed.json
(stamped with CPU/GPU model and library versions); the plot is byte-stable from
the JSON. Re-measure by deleting the JSON.
"""

import contextlib
import json
import os
import subprocess
import time
import sys

import numpy

if __name__ == "__main__" and __package__ is None:  # subprocess worker / --gpu
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from paper.figures._harness import bench, palette
from paper.figures._harness.figure import figure

_DATA = os.path.join(os.path.dirname(__file__), "data")
_JSON = os.path.join(_DATA, "gD3_spherical_df_speed.json")

_NS = [1000, 3000, 10000, 30000, 100000, 300000, 1000000, 3000000, 10000000]
_TMAX = 60.0
_SAMPLE_DFS = ["isotropicHernquistdf", "eddingtondf (NFW)", "osipkovmerrittdf", "kingdf"]
_CONSTRUCT_DFS = ["eddingtondf (NFW)", "osipkovmerrittdf", "constantbetadf", "kingdf"]
_CPU_MODES = ["numpy", "jax", "jax-jit", "torch"]
_GPU_MODES = ["jax-gpu", "jax-gpu-jit", "torch-gpu"]
_N_CONSTRUCT = 1000
_P0 = {"eddingtondf (NFW)": 1.0, "osipkovmerrittdf": 1.0, "constantbetadf": 1.0, "kingdf": 5.0}

# mode -> (framework, device, mode); colours/line styles from the shared palette
# (paper/figures/_harness/palette.py). H100 twins of the GPU modes come from
# data/h100/ (same script on an NVIDIA H100, Trillium).
_MODES = {
    "numpy": ("numpy", "cpu", None),
    "jax": ("jax", "cpu", "eager"),
    "jax-jit": ("jax", "cpu", "jit"),
    "torch": ("torch", "cpu", "eager"),
    "jax-gpu": ("jax", "gpu", "eager"),
    "jax-gpu-jit": ("jax", "gpu", "jit"),
    "torch-gpu": ("torch", "gpu", "eager"),
}
_JSON_H100 = os.path.join(_DATA, "h100", "gD3_spherical_df_speed.json")
# journal version (default): no env stamp / long titles or notes; the caption
# carries the explanations. GALPY_JAX_PAPER=0 restores the annotated figure.
PAPER = os.environ.get("GALPY_JAX_PAPER", "1") == "1"


def _runs(mode, data, h100):
    """(record source, device) pairs to draw for ``mode``: TITAN [+ H100]."""
    dev = _MODES[mode][1]
    out = [(data, dev)]
    if dev == "gpu" and h100 is not None:
        out.append((h100, "h100"))
    return out


# ------------------------------------------------------------------ helpers ---
def _setup():
    import jax

    jax.config.update("jax_enable_x64", True)
    import torch

    torch.set_default_dtype(torch.float64)
    # torch CPU intra-op threads: 1 by default (see module docstring)
    torch.set_num_threads(int(os.environ.get("GD3_TORCH_THREADS", "1")))
    import warnings

    warnings.filterwarnings("ignore")


def _devices():
    import jax
    import torch

    gpus = [d for d in jax.devices() if d.platform == "gpu"]
    return {
        "jax_cpu": jax.devices("cpu")[0],
        "jax_gpu": gpus[0] if gpus else None,
        "torch_gpu": torch.cuda.is_available(),
    }


def _env():
    import jax
    import torch

    env = {"python": sys.version.split()[0]}
    for m in ("numpy", "scipy", "jax", "torch", "galpy"):
        try:
            env[m] = __import__(m).__version__
        except Exception:
            pass
    try:
        with open("/proc/cpuinfo") as fh:
            env["cpu"] = next(l.split(":", 1)[1].strip() for l in fh if l.startswith("model name"))
    except Exception:
        env["cpu"] = "?"
    try:
        env["cpu_affinity"] = len(os.sched_getaffinity(0))
    except Exception:
        pass
    env["torch_threads"] = torch.get_num_threads()
    try:
        env["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    except Exception:
        env["gpu"] = "?"
    return env


def _sample_df(name):
    from galpy.df import eddingtondf, isotropicHernquistdf, kingdf, osipkovmerrittdf
    from galpy.potential import HernquistPotential, NFWPotential

    if name == "isotropicHernquistdf":
        return isotropicHernquistdf(pot=HernquistPotential(amp=2.0, a=1.0))
    if name == "eddingtondf (NFW)":
        return eddingtondf(pot=NFWPotential(amp=1.0, a=1.0), rmin=0.0)
    if name == "osipkovmerrittdf":
        return osipkovmerrittdf(pot=HernquistPotential(amp=2.0, a=1.0), ra=1.5, rmin=0.0)
    if name == "kingdf":
        return kingdf(W0=5.0, M=1.0, rt=10.0)
    raise ValueError(name)


def _sync_for(mode):
    return "jax" if mode.startswith("jax") else ("torch" if mode == "torch-gpu" else "none")


@contextlib.contextmanager
def _device_ctx(mode, dev):
    if mode in ("jax", "jax-jit", "jax-gpu", "jax-gpu-jit"):
        import jax

        with jax.default_device(dev["jax_gpu"] if "gpu" in mode else dev["jax_cpu"]):
            yield
    elif mode == "torch-gpu":
        import torch

        with torch.device("cuda"):
            yield
    else:
        yield


def _clear_gpu():
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


# ------------------------------------------------------- (a) sampling sweep ---
def _measure_sampling(modes, dev, errors):
    import jax

    from galpy.backend import random as grandom

    out, comp = {}, {}
    for name in _SAMPLE_DFS:
        df = _sample_df(name)
        numpy.random.seed(1)
        df.sample(n=10, return_orbit=False)  # builds the (shared) sampling tables
        out[name], comp[name] = {}, {}
        for mode in modes:
            times, ctimes = [], []
            stopped = False
            for N in _NS:
                if stopped:
                    times.append(float("nan"))
                    continue
                try:
                    with _device_ctx(mode, dev):
                        if mode == "numpy":
                            fn = lambda N=N: df.sample(n=N, return_orbit=False)
                        elif mode.startswith("jax"):
                            key = grandom.key(1, "jax")
                            if mode.endswith("jit"):
                                f = jax.jit(lambda k, N=N: df.sample(n=N, key=k, return_orbit=False))
                                t0 = time.perf_counter()
                                jax.block_until_ready(f(key))
                                ctimes.append(time.perf_counter() - t0)
                                fn = lambda f=f, key=key: f(key)
                            else:
                                fn = lambda N=N, key=key: df.sample(n=N, key=key, return_orbit=False)
                        else:
                            key = grandom.key(1, "torch")
                            fn = lambda N=N, key=key: df.sample(n=N, key=key, return_orbit=False)
                        t = bench.time_call(fn, warmup=1, repeats=3, sync=_sync_for(mode))["median"]
                except Exception as e:  # e.g. GPU OOM at large N
                    errors.append(f"sample {name} {mode} N={N}: {str(e)[:200]}")
                    print("  skip:", errors[-1], flush=True)
                    _clear_gpu()
                    t = float("nan")
                times.append(t)
                print(f"sample {name:22s} {mode:12s} N={N:>9d}  {t:.4g} s", flush=True)
                if t > _TMAX:
                    stopped = True
            out[name][mode] = times
            if ctimes:
                comp[name][mode] = ctimes
    return out, comp


# --------------------------------------------------- (b) construction cost ---
def _construct_worker(dfname, mode):
    """Run in a FRESH process: time the first and warm construct+sample calls."""
    _setup()


    import jax
    import jax.numpy as jnp
    import torch

    from galpy.backend import random as grandom
    from galpy.backend import use
    from galpy.df import constantbetadf, eddingtondf, kingdf, osipkovmerrittdf
    from galpy.potential import HernquistPotential, NFWPotential

    dev = _devices()
    backend = "numpy" if mode == "numpy" else ("jax" if mode.startswith("jax") else "torch")

    def build_and_sample(p):
        ctx = contextlib.nullcontext() if backend == "numpy" else use(backend, force=True)
        with ctx:
            if dfname == "eddingtondf (NFW)":
                d = eddingtondf(pot=NFWPotential(amp=1.0, a=p), rmin=0.0)
            elif dfname == "osipkovmerrittdf":
                d = osipkovmerrittdf(pot=HernquistPotential(amp=2.0, a=p), ra=1.5, rmin=0.0)
            elif dfname == "constantbetadf":
                d = constantbetadf(pot=HernquistPotential(amp=2.0, a=p), beta=-0.2, rmin=0.0)
            else:
                d = kingdf(W0=p, M=1.0, rt=10.0)
            key = None if backend == "numpy" else grandom.key(1, backend)
            return d.sample(n=_N_CONSTRUCT, key=key, return_orbit=False)

    fn = jax.jit(build_and_sample) if mode.endswith("jit") else build_and_sample

    def param(p):
        if backend == "numpy":
            return float(p)
        if backend == "jax":
            d_ = dev["jax_gpu"] if "gpu" in mode else dev["jax_cpu"]
            return jax.device_put(jnp.asarray(float(p)), d_)
        return torch.tensor(float(p), dtype=torch.float64, device="cuda" if "gpu" in mode else "cpu")

    def sync(out):
        if backend == "jax":
            jax.block_until_ready(out)
        elif mode == "torch-gpu":
            torch.cuda.synchronize()

    p0 = _P0[dfname]
    if backend == "numpy":
        numpy.random.seed(1)
    with _device_ctx(mode, dev):
        t0 = time.perf_counter()
        sync(fn(param(p0)))
        first = time.perf_counter() - t0
        warm = []
        for i in range(1, 4 if first < _TMAX else 2):
            p = param(p0 * (1.0 + 0.01 * i))
            t0 = time.perf_counter()
            sync(fn(p))
            warm.append(time.perf_counter() - t0)
    print("RESULT " + json.dumps({"first": first, "warm": float(numpy.median(warm)), "warm_all": warm}))


def _measure_construction(modes, errors):
    out = {}
    env = dict(os.environ)
    for dfname in _CONSTRUCT_DFS:
        out[dfname] = {}
        for mode in modes:
            try:
                res = subprocess.run(
                    [sys.executable, os.path.abspath(__file__), "--construct", dfname, mode],
                    env=env, capture_output=True, text=True, timeout=3600,
                )
                line = [l for l in res.stdout.splitlines() if l.startswith("RESULT ")]
                if not line:
                    raise RuntimeError(res.stderr.strip().splitlines()[-1] if res.stderr.strip() else "no output")
                r = json.loads(line[-1][len("RESULT "):])
            except Exception as e:
                errors.append(f"construct {dfname} {mode}: {str(e)[:300]}")
                print("  skip:", errors[-1], flush=True)
                r = {"first": float("nan"), "warm": float("nan"), "warm_all": []}
            out[dfname][mode] = r
            print(f"construct {dfname:20s} {mode:12s} first {r['first']:.4g} s  warm {r['warm']:.4g} s", flush=True)
    return out


# ------------------------------------------------------------ measure/merge ---
def _measure(gpu_only=False, only=None):
    """Measure all available modes, or only the GPU ones / the given ``only``
    modes, merging the latter two into an existing JSON."""
    _setup()
    dev = _devices()
    gpu_modes = [m for m in _GPU_MODES
                 if (dev["jax_gpu"] is not None if m.startswith("jax") else dev["torch_gpu"])]
    modes = gpu_modes if gpu_only else _CPU_MODES + gpu_modes
    if only is not None:
        modes = [m for m in only if m in _CPU_MODES or m in gpu_modes]
    errors = []
    sample, comp = _measure_sampling(modes, dev, errors)
    construct = _measure_construction(modes, errors)
    if (gpu_only or only is not None) and os.path.exists(_JSON):
        with open(_JSON) as fh:
            data = json.load(fh)
    else:
        data = {"Ns": _NS, "n_construct": _N_CONSTRUCT, "sample": {}, "compile": {},
                "construct": {}, "errors": []}
    for name in sample:
        data["sample"].setdefault(name, {}).update(sample[name])
        data["compile"].setdefault(name, {}).update(comp[name])
    for name in construct:
        data["construct"].setdefault(name, {}).update(construct[name])
    data["errors"] = [e for e in data.get("errors", []) if not any(m in e for m in modes)] + errors
    env = _env()
    if gpu_only:
        data["env_gpu"] = env
    elif only is None or "env" not in data:
        data["env"] = env
        data.setdefault("env_gpu", {"gpu": env["gpu"]})
    data.setdefault("modes_measured", {})
    for m in modes:
        data["modes_measured"][m] = env
    os.makedirs(_DATA, exist_ok=True)
    with open(_JSON, "w") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return data


# --------------------------------------------------------------------- plot ---
def _fmt(t):
    return f"{t:.0f}" if t >= 10 else f"{t:.2g}"


@figure(
    name="gD3_spherical_df_speed",
    track="F",
    seed=0,
    description="spherical-DF speed: sample(n=N) wall time vs N for four DFs and DF construction (+first 1000 draws) time, numpy vs jax (eager/jit) vs torch, CPU & GPU [measure->json, plot byte-stable]",
)
def make():
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Patch

    if os.path.exists(_JSON):
        with open(_JSON) as fh:
            data = json.load(fh)
    else:
        data = _measure()
    h100 = None
    if os.path.exists(_JSON_H100):
        with open(_JSON_H100) as fh:
            h100 = json.load(fh)
    Ns = numpy.array(data["Ns"], dtype=float)
    order = _CPU_MODES + _GPU_MODES

    if PAPER:  # final printed width of a figure* (7.0 in), printed text sizes: the
        # four sampling panels as a 2x2 grid (a 1x4 row is too narrow for the names)
        fig = plt.figure(figsize=(6.8, 8.0))
        gs = GridSpec(3, 4, figure=fig, height_ratios=[1.0, 1.0, 1.3], hspace=0.5, wspace=0.08,
                      left=0.085, right=0.80, top=0.965, bottom=0.145)
        axs = [fig.add_subplot(gs[i // 2, 2 * (i % 2):2 * (i % 2) + 2]) for i in range(4)]
        _bar_gs, _left = gs[2, :], (0, 2)
    else:
        fig = plt.figure(figsize=(14.6, 8.4))
        gs = GridSpec(2, 4, figure=fig, height_ratios=[1.0, 1.0], hspace=0.42, wspace=0.08,
                      left=0.06, right=0.86, top=0.95, bottom=0.12)
        axs = [fig.add_subplot(gs[0, i]) for i in range(4)]
        _bar_gs, _left = gs[1, :], (0,)
    _fs_note, _fs_bar, _fs_leg, _fs_lab = (8, 8, 8, 8) if PAPER else (6.5, 5.2, 7, 6.5)
    for i, (ax, name) in enumerate(zip(axs, _SAMPLE_DFS)):
        for mode in order:
            fw, _, md = _MODES[mode]
            for d, dev in _runs(mode, data, h100):
                ser = d["sample"].get(name, {})
                if mode not in ser:
                    continue
                ax.loglog(Ns, numpy.array(ser[mode], dtype=float), ms=3.5,
                          label=palette.label(fw, dev, md), **palette.style(fw, dev, md))
        for rate in (1e5, 1e6, 1e7, 1e8):
            ax.plot(Ns, Ns / rate, ":", lw=0.7, color="0.8", zorder=0)
        ax.set_ylim(3e-5, 200)
        ax.set_xlabel(r"$N$ particles")
        ax.set_title(f"({'abcd'[i]}) {name}", fontsize=10)
        if i in _left:
            ax.set_ylabel("wall time of\n$\\tt sample(n=N)$  (s)" if PAPER
                          else r"wall time of $\tt sample(n=N)$  (s)")
        else:
            ax.tick_params(labelleft=False)
    axs[0].text(0.03, 0.97, "dotted: constant particles/s" if PAPER
                else "dotted: $10^5$..$10^8$ particles/s", transform=axs[0].transAxes,
                fontsize=_fs_note, color="0.5", va="top")

    # (e) bars: colour = backend x device; filled = warm (white-hatched = eager, the
    # bar analogue of a dashed line); hollow outline = first call in a fresh process
    ax = fig.add_subplot(_bar_gs)
    bars = []  # (mode, source, device)
    labels = []  # deferred value labels: (x, y, text, va, colour), drawn once ylim is set
    for mode in order:
        for d, dev in _runs(mode, data, h100):
            if any(mode in d["construct"].get(dn, {}) for dn in _CONSTRUCT_DFS):
                bars.append((mode, d, dev))
    nm = len(bars)
    w = 0.86 / nm
    for g, dname in enumerate(_CONSTRUCT_DFS):
        for j, (mode, d, dev) in enumerate(bars):
            rec = d["construct"].get(dname, {})
            if mode not in rec:
                continue
            fw, _, md = _MODES[mode]
            col = palette.style(fw, dev, md)["color"]
            eager = md == "eager"
            x = g + (j - (nm - 1) / 2.0) * w
            first, warm = rec[mode]["first"], rec[mode]["warm"]
            ax.bar(x, first, w * 0.92, color="none", edgecolor=col, lw=1.0)
            ax.bar(x, warm, w * 0.92, color=col, edgecolor="w", hatch="////" if eager else None, lw=0)
            ax.bar(x, warm, w * 0.92, color="none", edgecolor="k", lw=0.4)
            if numpy.isfinite(warm):
                rgb = matplotlib.colors.to_rgb(col)
                dark = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) < 0.5
                labels.append((x, warm, _fmt(warm), "top", "w" if dark else "k"))
            if numpy.isfinite(first):
                labels.append((x, first, _fmt(first), "bottom", "0.35"))
    ax.set_yscale("log")
    ax.set_xticks(range(len(_CONSTRUCT_DFS)))
    ax.set_xticklabels(_CONSTRUCT_DFS)
    ax.set_xlim(-0.5, len(_CONSTRUCT_DFS) - 0.5)
    ax.set_ylabel("construct + first %d draws  (s)" % data.get("n_construct", _N_CONSTRUCT))
    ax.set_title("(e) DF construction" if PAPER else
                 "(e) DF construction (f(E)/f(Q) inversion or King ODE + sampling tables): "
                 "filled = warm (hatched = eager), outline = first call in a fresh process",
                 fontsize=10)
    handles = []
    for mode, d, dev in bars:
        fw, _, md = _MODES[mode]
        handles.append(Patch(facecolor=palette.style(fw, dev, md)["color"], edgecolor="w",
                             hatch="////" if md == "eager" else None, label=palette.label(fw, dev, md)))
    handles.append(Patch(facecolor="none", edgecolor="0.3", label="first call (outline)"))
    ax.legend(handles=handles, fontsize=_fs_leg, ncol=4 if PAPER else 6, loc="upper center",
              bbox_to_anchor=(0.5, -0.12 if PAPER else -0.1), framealpha=0.92)
    recs = [r[m] for src in ([data] + ([h100] if h100 else [])) for r in src["construct"].values()
            for m in r]
    lo = min(r["warm"] for r in recs if numpy.isfinite(r["warm"]))
    hi = max(r["first"] for r in recs if numpy.isfinite(r["first"]))
    ax.set_ylim(lo / 4.0, hi * 12.0)
    # value labels (rotated): a warm label goes inside its bar, hanging from 0.8x
    # the top, and is dropped when the bar is shorter than the text (PAPER sizes)
    ax_h_pt = ax.get_position().height * fig.get_figheight() * 72.0
    dec = numpy.log10(hi * 12.0 / (lo / 4.0))
    for x, y, txt, va, c in ([] if PAPER else labels):  # PAPER: too narrow to read at 7 in
        if va == "top":
            bar_pt = numpy.log10(y * 0.8 / (lo / 4.0)) / dec * ax_h_pt
            if PAPER and bar_pt < 0.8 * _fs_bar * len(txt) + 8:
                continue
            ax.text(x, y * 0.8, txt, ha="center", va="top", fontsize=_fs_bar, rotation=90, color=c)
        else:
            ax.text(x, y * 1.15, txt, ha="center", va="bottom", fontsize=_fs_bar, rotation=90,
                    color=c)
    env = data.get("env", {})
    eg, eh = data.get("env_gpu", {}), (h100 or {}).get("env_gpu", {})
    if not PAPER:
        fig.text(0.995, 0.005,
                 f"CPU: {env.get('cpu', '?')} ({env.get('cpu_affinity', '?')} cores): numpy {env.get('numpy', '?')} · "
                 f"jax {env.get('jax', '?')} · torch {env.get('torch', '?')}  |  "
                 f"{eg.get('gpu', 'no GPU')}: jax {eg.get('jax', '?')} · torch {eg.get('torch', '?')}"
                 + (f"  |  {eh.get('gpu', 'H100')}: jax {eh.get('jax', '?')} · torch {eh.get('torch', '?')}" if eh else "")
                 + "  ·  float64",
                 fontsize=6, color="0.5", ha="right", va="bottom")
    # the four sampling panels share their series: label them once, beside (d)
    palette.label_lines(axs[-1], fontsize=_fs_lab)
    return fig


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--construct":
        _construct_worker(sys.argv[2], sys.argv[3])
    elif len(sys.argv) >= 2 and sys.argv[1] == "--gpu":
        _measure(gpu_only=True)
    elif len(sys.argv) >= 3 and sys.argv[1] == "--modes":
        _measure(only=sys.argv[2].split(","))
    else:
        _measure()
