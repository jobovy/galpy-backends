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
    runs in a FRESH subprocess: hatched = first call (for jax eager: per-op
    compilation; for jax.jit: trace + XLA compile), solid = warm (median of
    re-calls with new parameter values -- same shapes, so jit does not
    recompile). torch is eager (torch.compile is not used; it does not run on the
    sm_61 GPU this paper uses).

Float64 throughout (galpy's policy: ``jax_enable_x64``, and torch computes in
float64 internally). Timings: ``bench.time_call`` medians (warmup 1, 3
repeats; device-synchronised), a series stops once a call exceeds ~60 s. CPU:
pinned to 4 cores (``taskset``); torch uses its default intra-op threads on
those cores, numpy/scipy are single-threaded, jax uses XLA's CPU threadpool.
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

from paper.figures._harness import bench
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

_STYLE = {  # label, colour, linestyle, marker
    "numpy": ("numpy (CPU)", "#000000", "-", "s"),
    "jax": ("jax (CPU, eager)", "#56B4E9", "-", "^"),
    "jax-jit": ("jax (CPU, jit)", "#009E73", "-.", "P"),
    "torch": ("torch (CPU)", "#E69F00", "-", "v"),
    "jax-gpu": ("jax (GPU, eager)", "#0072B2", "-", "o"),
    "jax-gpu-jit": ("jax (GPU, jit)", "#CC79A7", "-.", "D"),
    "torch-gpu": ("torch (GPU)", "#D55E00", "-", "o"),
}


# ------------------------------------------------------------------ helpers ---
def _setup():
    import jax

    jax.config.update("jax_enable_x64", True)
    import torch

    torch.set_default_dtype(torch.float64)
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
def _measure(gpu_only=False):
    _setup()
    dev = _devices()
    gpu_modes = [m for m in _GPU_MODES
                 if (dev["jax_gpu"] is not None if m.startswith("jax") else dev["torch_gpu"])]
    modes = gpu_modes if gpu_only else _CPU_MODES + gpu_modes
    errors = []
    sample, comp = _measure_sampling(modes, dev, errors)
    construct = _measure_construction(modes, errors)
    if gpu_only and os.path.exists(_JSON):
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
    data["env_gpu" if gpu_only else "env"] = env
    if not gpu_only:
        data.setdefault("env_gpu", {"gpu": env["gpu"]})
    os.makedirs(_DATA, exist_ok=True)
    with open(_JSON, "w") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return data


# --------------------------------------------------------------------- plot ---
@figure(
    name="gD3_spherical_df_speed",
    track="F",
    seed=0,
    description="spherical-DF speed: sample(n=N) wall time vs N for four DFs and DF construction (+first 1000 draws) time, numpy vs jax (eager/jit) vs torch, CPU & GPU [measure->json, plot byte-stable]",
)
def make():
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Patch

    if os.path.exists(_JSON):
        with open(_JSON) as fh:
            data = json.load(fh)
    else:
        data = _measure()
    Ns = numpy.array(data["Ns"], dtype=float)
    order = _CPU_MODES + _GPU_MODES

    fig = plt.figure(figsize=(13.6, 8.2))
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1.0, 1.0], hspace=0.42, wspace=0.08)
    axs = [fig.add_subplot(gs[0, i]) for i in range(4)]
    for i, (ax, name) in enumerate(zip(axs, _SAMPLE_DFS)):
        ser = data["sample"].get(name, {})
        for mode in order:
            if mode not in ser:
                continue
            lab, col, ls, mk = _STYLE[mode]
            ax.loglog(Ns, numpy.array(ser[mode], dtype=float), ls=ls, marker=mk, ms=3.5,
                      color=col, label=lab)
        for rate in (1e5, 1e6, 1e7, 1e8):
            ax.plot(Ns, Ns / rate, ":", lw=0.7, color="0.8", zorder=0)
        ax.set_ylim(3e-5, 200)
        ax.set_xlabel(r"$N$ particles")
        ax.set_title(f"({'abcd'[i]}) {name}", fontsize=10)
        if i == 0:
            ax.set_ylabel(r"wall time of $\tt sample(n=N)$  (s)")
        else:
            ax.tick_params(labelleft=False)
    axs[0].text(0.03, 0.97, "dotted: $10^5$..$10^8$ particles/s", transform=axs[0].transAxes,
                fontsize=6.5, color="0.5", va="top")
    axs[-1].legend(fontsize=6.8, loc="lower right", framealpha=0.92)

    ax = fig.add_subplot(gs[1, :])
    modes = [m for m in order if any(m in data["construct"].get(d, {}) for d in _CONSTRUCT_DFS)]
    nm = len(modes)
    w = 0.8 / nm
    for g, dname in enumerate(_CONSTRUCT_DFS):
        rec = data["construct"].get(dname, {})
        for j, mode in enumerate(modes):
            if mode not in rec:
                continue
            x = g + (j - (nm - 1) / 2.0) * w
            first, warm = rec[mode]["first"], rec[mode]["warm"]
            col = _STYLE[mode][1]
            ax.bar(x, first, w * 0.92, color="none", edgecolor=col, hatch="////", lw=0.8)
            ax.bar(x, warm, w * 0.92, color=col, edgecolor="k", lw=0.4)
            if numpy.isfinite(warm):
                ax.text(x, warm * 0.8, f"{warm:.2g}", ha="center", va="top", fontsize=5.6,
                        rotation=90, color="w" if mode in ("numpy", "jax-gpu", "jax-gpu-jit", "torch-gpu", "jax-jit") else "k")
            if numpy.isfinite(first):
                ax.text(x, first * 1.15, f"{first:.2g}", ha="center", va="bottom", fontsize=5.6,
                        rotation=90, color="0.35")
    ax.set_yscale("log")
    ax.set_xticks(range(len(_CONSTRUCT_DFS)))
    ax.set_xticklabels(_CONSTRUCT_DFS)
    ax.set_xlim(-0.5, len(_CONSTRUCT_DFS) - 0.5)
    ax.set_ylabel("construct + first %d draws  (s)" % data.get("n_construct", _N_CONSTRUCT))
    ax.set_title("(e) DF construction (f(E)/f(Q) inversion or King ODE + sampling tables): "
                 "hatched = first call in a fresh process, solid = warm", fontsize=10)
    handles = [Patch(facecolor=_STYLE[m][1], edgecolor="k", label=_STYLE[m][0]) for m in modes]
    handles.append(Patch(facecolor="none", edgecolor="0.3", hatch="////", label="first call"))
    ax.legend(handles=handles, fontsize=7, ncol=len(handles), loc="upper center",
              bbox_to_anchor=(0.5, -0.1), framealpha=0.92)
    lo = min(v[m]["warm"] for v in data["construct"].values() for m in v
             if numpy.isfinite(v[m]["warm"]))
    hi = max(v[m]["first"] for v in data["construct"].values() for m in v
             if numpy.isfinite(v[m]["first"]))
    ax.set_ylim(lo / 4.0, hi * 12.0)
    env = data.get("env", {})
    fig.text(0.995, 0.005,
             f"CPU: {env.get('cpu', '?')} ({env.get('cpu_affinity', '?')} cores) · "
             f"GPU: {data.get('env_gpu', {}).get('gpu', 'none')} · numpy {env.get('numpy', '?')} · "
             f"jax {env.get('jax', '?')} · torch {env.get('torch', '?')} · float64",
             fontsize=6, color="0.5", ha="right", va="bottom")
    return fig


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--construct":
        _construct_worker(sys.argv[2], sys.argv[3])
    elif len(sys.argv) >= 2 and sys.argv[1] == "--gpu":
        _measure(gpu_only=True)
    else:
        _measure()
