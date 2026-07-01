"""G17 -- Staeckel action-angle throughput: C vs numpy vs jax/torch (CPU & GPU).

The action-angle analog of G2. ``actionAngleStaeckel.actionsFreqsAngles`` (the full
jr,Lz,jz,Omega_r,Omega_phi,Omega_z,theta_r,theta_phi,theta_z for N phase-space
points) is now backend-agnostic and *vectorised* in the c=False path, so it runs --
and differentiates -- under numpy/jax/torch in addition to the fast C path. This
figure is the headline performance picture for that stack:

  * C (c=True): the compiled per-object reference loop -- the bar to beat.
  * numpy (CPU): the vectorised pure-Python path (this work). One bisection +
    fixed-order Gauss-Legendre quadrature, fully array-wise over N.
  * numpy (CPU, per-object, OLD): the pre-vectorisation c=False path (git 7a31c196,
    a Python for-loop over actionAngleStaeckelSingle) -- measured in a subprocess
    against that build, capped at small N because it is O(N) Python. The gap to
    "numpy (CPU)" is the vectorisation win.
  * jax / torch (CPU): the same vectorised code under the array backends, eager
    (galpy adds no internal jit -- it is made jit-*compatible* for the caller).
  * jax (CPU/GPU, jit): the caller's jax.jit of actionsFreqsAngles (dash-dot) --
    how a user runs it for speed inside an optimisation/training loop. Compile is
    one-time (~3 s/shape post-#1040: the fixed-iteration bisection rolls into
    lax.fori_loop instead of unrolling; was ~8.6 min before) and EXCLUDED; the
    warm call is timed, matching the jit-then-time convention of g2/g8/g15.
  * jax / torch (GPU): compute-only (solid; arrays resident on device, timed wait
    is the kernel) and end-to-end (dashed; incl. host<->device transfer).

What it shows: C wins at every N for this branchy root-find + quadrature workload
(no surprise -- it is compiled and has no per-op dispatch); among the differentiable
paths numpy is fastest on CPU, EAGER jax/torch are dispatch-bound at small N (eager
jax sits on a ~3-7 s per-op-dispatch floor), JITTING jax collapses the warm
single-orbit call to ~5 ms after a ~3 s/shape compile (post-#1040 fori_loop
rolling; was ~8.6 min), so jit is now worthwhile even for modest reuse. The
backends are not here to beat C on raw throughput -- they are here to make the
action-angle map *differentiable and composable* inside jax/torch models; this figure
quantifies what that costs, eager vs jit.

Reproducibility: timings are machine-dependent, so the *measure* step writes
data/g17_staeckel_actionangle_speed.json (stamped with GPU + library versions) and
the *plot* step is byte-stable from that JSON. Re-measure by deleting the JSON and
re-running ``run`` in the GPU env; ``verify`` only checks the (deterministic) plot.
"""

import json
import os
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy

from paper.figures._harness import bench
from paper.figures._harness.figure import figure

_DATA = os.path.join(os.path.dirname(__file__), "data")
_JSON = os.path.join(_DATA, "g17_staeckel_actionangle_speed.json")

# colour per backend; solid = compute-only / CPU, dashed = +transfer or OLD.
_STYLE = {
    "C (c=True)": ("#000000", "-", "o"),
    "numpy (CPU)": ("#444444", "-", "s"),
    "numpy (CPU, per-object, OLD)": ("#999999", "--", "x"),
    "jax (CPU)": ("#56B4E9", "-", "^"),
    "jax (CPU, jit)": ("#56B4E9", "-.", "P"),
    "jax (GPU)": ("#0072B2", "-", "o"),
    "jax (GPU, jit)": ("#0072B2", "-.", "D"),
    "jax (GPU, +transfer)": ("#0072B2", "--", "o"),
    "torch (CPU)": ("#E69F00", "-", "v"),
    "torch (GPU)": ("#D55E00", "-", "o"),
    "torch (GPU, +transfer)": ("#D55E00", "--", "o"),
}

# A known-bound base grid (the validated _STK_GRID: 768 bound orbits for
# MWPotential2014 + delta=0.45) + per-object azimuths, tiled/sliced to length N so
# every benchmarked batch is bound (no UnboundError) and N-independent in content.
_Rg = numpy.array([0.7, 0.9, 1.1, 1.3])
_vRg = numpy.array([-0.3, -0.1, 0.1, 0.3])
_vTg = numpy.array([-0.8, 0.6, 0.9, 1.2])
_zg = numpy.array([-0.3, -0.1, 0.1, 0.3])
_vzg = numpy.array([-0.2, 0.0, 0.15])
_G = numpy.meshgrid(_Rg, _vRg, _vTg, _zg, _vzg, indexing="ij")
_BASE = [g.ravel() for g in _G]
_BASE.append(numpy.linspace(0.1, 6.2, _BASE[0].size))


def _grid(N):
    reps = (N // _BASE[0].size) + 1
    return [numpy.tile(b, reps)[:N] for b in _BASE]


def _med(fn, **kw):
    return bench.time_call(fn, **kw)["median"]


def _measure():
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import torch

    torch.set_default_dtype(torch.float64)
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.potential import MWPotential2014

    aC = actionAngleStaeckel(pot=MWPotential2014, delta=0.45, c=True)
    aF = actionAngleStaeckel(pot=MWPotential2014, delta=0.45, c=False)

    cpu = jax.devices("cpu")[0]
    gpus = [d for d in jax.devices() if d.platform == "gpu"]
    gpu = gpus[0] if gpus else None
    has_torch_gpu = torch.cuda.is_available()

    # N ranges: C/numpy are cheap (-> 1e6); the eager backends are dispatch-heavy
    # (root-find + quadrature = hundreds of ops/call) so cap CPU eager lower; the
    # old O(N) Python loop is capped hardest.
    Ns = [1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000]
    # extend the rising backend curves until throughput flattens: GPU to a few e6
    # (compute saturates the device), CPU eager as far as is affordable.
    Ns_gpu = Ns + [300000, 1000000]  # 3e6 OOMs the 12 GB GPU (quadrature (N,order) arrays)
    Ns_jaxcpu = [n for n in Ns if n <= 30000]  # jax eager is dispatch-bound + slow
    Ns_torchcpu = [n for n in Ns if n <= 100000]
    Ns_old = [1, 3, 10, 30, 100, 300, 1000, 3000]
    # jitted jax: compile is ~3 s/shape post-#1040 (rolled fori_loop bisection;
    # was ~8.6 min unrolled); keep the reduced N-grid -- the warm curve is smooth
    # and a few points show the floor + the large-N rise.
    Ns_jaxjit_cpu = [1, 100, 3000, 30000]
    Ns_jaxjit_gpu = [1, 100, 3000, 100000]  # cap < 1e6: the huge unrolled graph
    #                                          risks a compile-time OOM on 12 GB
    series = {}

    def _safe(fn):  # large-N GPU calls may OOM the 12 GB device -> record NaN
        try:
            return fn()
        except Exception as e:
            print("  skip (likely OOM):", str(e)[:70])
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            return float("nan")

    # --- C (c=True) ---
    series["C (c=True)"] = []
    for N in Ns:
        g = _grid(N)
        aC.actionsFreqsAngles(*g)
        series["C (c=True)"].append(_med(lambda g=g: aC.actionsFreqsAngles(*g),
                                         warmup=1, repeats=5))
    # --- numpy (CPU), vectorised ---
    series["numpy (CPU)"] = []
    for N in Ns:
        g = _grid(N)
        aF.actionsFreqsAngles(*g)
        series["numpy (CPU)"].append(_med(lambda g=g: aF.actionsFreqsAngles(*g),
                                          warmup=1, repeats=5))

    # --- jax CPU / GPU (explicit device placement; one process) ---
    def _jax_series(dev, Nlist, sync):
        out = []
        for N in Nlist:
            def _t(N=N):
                g = [jax.device_put(jnp.asarray(b), dev) for b in _grid(N)]
                jax.block_until_ready(aF.actionsFreqsAngles(*g))
                return _med(lambda g=g: aF.actionsFreqsAngles(*g),
                            warmup=2, repeats=3, sync=sync)
            out.append(_safe(_t))
        return out

    series["jax (CPU)"] = _jax_series(cpu, Ns_jaxcpu, "jax")

    # jitted jax (how a user actually runs it for speed: compile once at a fixed
    # shape -- e.g. inside an optimization/training loop -- then call repeatedly).
    # Compile is EXCLUDED (one-time, ~minutes here); the WARM call is timed,
    # matching the jit-then-time-warm convention of g2/g8/g15. galpy adds no
    # internal jit -- the CALLER jits, exactly as done here.
    def _jax_jit_series(dev, Nlist):
        out = []
        for N in Nlist:
            def _t(N=N):
                g = [jax.device_put(jnp.asarray(b), dev) for b in _grid(N)]
                f = jax.jit(lambda *a: aF.actionsFreqsAngles(*a))
                jax.block_until_ready(f(*g))  # compile (excluded from timing)
                return _med(lambda f=f, g=g: f(*g), warmup=2, repeats=3, sync="jax")
            out.append(_safe(_t))
        return out

    series["jax (CPU, jit)"] = _jax_jit_series(cpu, Ns_jaxjit_cpu)
    if gpu is not None:
        series["jax (GPU, jit)"] = _jax_jit_series(gpu, Ns_jaxjit_gpu)
        series["jax (GPU)"] = _jax_series(gpu, Ns_gpu, "jax")
        # end-to-end: host numpy -> device -> compute -> host
        e2e = []
        for N in Ns_gpu:
            def _t(N=N):
                gn = _grid(N)
                return _med(
                    lambda gn=gn: numpy.asarray(
                        aF.actionsFreqsAngles(*[jax.device_put(jnp.asarray(b), gpu) for b in gn])[0]
                    ),
                    warmup=2, repeats=3,
                )
            e2e.append(_safe(_t))
        series["jax (GPU, +transfer)"] = e2e

    # --- torch CPU / GPU ---
    def _torch_series(device, Nlist, sync):
        out = []
        for N in Nlist:
            def _t(N=N):
                g = [torch.as_tensor(b, device=device) for b in _grid(N)]
                aF.actionsFreqsAngles(*g)
                if sync == "torch":
                    torch.cuda.synchronize()
                return _med(lambda g=g: aF.actionsFreqsAngles(*g),
                            warmup=2, repeats=3, sync=sync)
            out.append(_safe(_t))
        return out

    series["torch (CPU)"] = _torch_series("cpu", Ns_torchcpu, "none")
    if has_torch_gpu:
        series["torch (GPU)"] = _torch_series("cuda", Ns_gpu, "torch")
        e2e = []
        for N in Ns_gpu:
            def _t(N=N):
                gn = _grid(N)
                return _med(
                    lambda gn=gn: aF.actionsFreqsAngles(
                        *[torch.as_tensor(b, device="cuda") for b in gn]
                    )[0].cpu().numpy(),
                    warmup=2, repeats=3,
                )
            e2e.append(_safe(_t))
        series["torch (GPU, +transfer)"] = e2e

    # --- OLD per-object c=False (subprocess against the pre-vectorisation build) ---
    old_wt = "/var/tmp/staeckel-old"
    if os.path.isdir(old_wt):
        try:
            old = _measure_old(old_wt, Ns_old)
            if old:
                series["numpy (CPU, per-object, OLD)"] = old
        except Exception as e:  # pragma: no cover
            print("OLD measurement skipped:", e)

    env = {"python": sys.version.split()[0]}
    for m in ("numpy", "jax", "torch", "galpy"):
        try:
            env[m] = __import__(m).__version__
        except Exception:
            pass
    try:
        env["gpu"] = torch.cuda.get_device_name(0) if has_torch_gpu else "none"
    except Exception:
        env["gpu"] = "?"
    data = {
        "Ns": Ns, "Ns_jaxcpu": Ns_jaxcpu, "Ns_torchcpu": Ns_torchcpu,
        "Ns_gpu": Ns_gpu, "Ns_old": Ns_old,
        "Ns_jaxjit_cpu": Ns_jaxjit_cpu, "Ns_jaxjit_gpu": Ns_jaxjit_gpu,
        "series": series, "env": env,
    }
    os.makedirs(_DATA, exist_ok=True)
    with open(_JSON, "w") as fh:
        json.dump(data, fh, indent=2)
    return data


def _measure_old(old_wt, Ns_old):
    """Time the OLD per-object c=False actionsFreqsAngles in a subprocess against
    the pre-vectorisation galpy build (different install -> cannot co-import)."""
    script = (
        "import sys,time,numpy,warnings;warnings.filterwarnings('ignore')\n"
        "import galpy;assert galpy.__file__.startswith('%s')\n"
        "from galpy.actionAngle import actionAngleStaeckel\n"
        "from galpy.potential import MWPotential2014\n"
        "aF=actionAngleStaeckel(pot=MWPotential2014,delta=0.45,c=False)\n"
        "Rg=numpy.array([0.7,0.9,1.1,1.3]);vRg=numpy.array([-0.3,-0.1,0.1,0.3]);vTg=numpy.array([-0.8,0.6,0.9,1.2]);zg=numpy.array([-0.3,-0.1,0.1,0.3]);vzg=numpy.array([-0.2,0.0,0.15])\n"
        "G=numpy.meshgrid(Rg,vRg,vTg,zg,vzg,indexing='ij');B=[g.ravel() for g in G];B.append(numpy.linspace(0.1,6.2,B[0].size))\n"
        "def grid(N):\n reps=(N//B[0].size)+1\n return [numpy.tile(b,reps)[:N] for b in B]\n"
        "def med(fn,r):\n ts=[]\n for _ in range(r):\n  t0=time.perf_counter();fn();ts.append(time.perf_counter()-t0)\n return sorted(ts)[len(ts)//2]\n"
        "for N in %r:\n g=grid(N);aF.actionsFreqsAngles(*g)\n print(N,med(lambda: aF.actionsFreqsAngles(*g),3))\n"
    ) % (old_wt, Ns_old)
    env = dict(os.environ, PYTHONPATH=old_wt, CUDA_VISIBLE_DEVICES="")
    out = subprocess.run([sys.executable, "-c", script], cwd=old_wt, env=env,
                         capture_output=True, text=True, timeout=1800)
    res = {}
    for line in out.stdout.splitlines():
        p = line.split()
        if len(p) == 2:
            res[int(p[0])] = float(p[1])
    return [res.get(N, float("nan")) for N in Ns_old]


# 11 curves in one figure is too dense, so split by backend (each its own figure):
# both share the C + numpy (+ old per-object numpy) reference backdrop, then add
# the backend's eager/jit/transfer curves. Plot-only -- one shared measurement.
_REF = ["C (c=True)", "numpy (CPU)", "numpy (CPU, per-object, OLD)"]
_JAX = ["jax (CPU)", "jax (CPU, jit)", "jax (GPU)", "jax (GPU, jit)", "jax (GPU, +transfer)"]
_TORCH = ["torch (CPU)", "torch (GPU)", "torch (GPU, +transfer)"]


def _data():
    if os.path.exists(_JSON):
        with open(_JSON) as fh:
            return json.load(fh)
    return _measure()


def _Nmap(data):
    return {
        "jax (CPU)": data["Ns_jaxcpu"], "torch (CPU)": data["Ns_torchcpu"],
        "jax (GPU)": data["Ns_gpu"], "jax (GPU, +transfer)": data["Ns_gpu"],
        "torch (GPU)": data["Ns_gpu"], "torch (GPU, +transfer)": data["Ns_gpu"],
        "numpy (CPU, per-object, OLD)": data["Ns_old"],
        "jax (CPU, jit)": data.get("Ns_jaxjit_cpu", data["Ns"]),
        "jax (GPU, jit)": data.get("Ns_jaxjit_gpu", data["Ns_gpu"]),
    }


def _make(names, backend_label):
    data = _data()
    series, env, Nmap = data["series"], data.get("env", {}), _Nmap(data)
    plot_names = [n for n in names if n in series]
    fig, (axw, axt) = plt.subplots(1, 2, figsize=(10.4, 4.4))
    _plot_panel(axw, series, Nmap, data, "walltime", plot_names)
    _plot_panel(axt, series, Nmap, data, "throughput", plot_names)
    axt.legend(fontsize=6.8, loc="lower right", framealpha=0.92, handlelength=2.6, ncol=1)
    axt.text(0.97, 0.05,
             f"GPU: {env.get('gpu', '?')}\njax {env.get('jax', '?')} · torch {env.get('torch', '?')}",
             transform=axt.transAxes, fontsize=6.0, color="0.5", va="bottom", ha="right")
    fig.suptitle(r"Staeckel $(\mathbf{J},\mathbf{\Omega},\mathbf{\theta})$ via actionsFreqsAngles "
                 r"vs batch size $N$ -- " + backend_label, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


@figure(
    name="g17_staeckel_actionangle_speed_jax",
    track="G", seed=0,
    description="Staeckel actionsFreqsAngles throughput vs N -- jax (eager vs jit, CPU & GPU) vs C/numpy [measure->json, plot byte-stable]",
)
def make_jax():
    return _make(_REF + _JAX, "jax (eager vs jit)")


@figure(
    name="g17_staeckel_actionangle_speed_torch",
    track="G", seed=0,
    description="Staeckel actionsFreqsAngles throughput vs N -- torch (eager, CPU & GPU) vs C/numpy [measure->json, plot byte-stable]",
)
def make_torch():
    return _make(_REF + _TORCH, "torch (eager)")


def _plot_panel(ax, series, Nmap, data, mode, names):
    """One log-log panel: mode='walltime' (s) [matches the orbit speed figures]
    or mode='throughput' (points/s). NaN points (OOM-skipped) drop out. `names`
    is the ordered subset of series to draw (the backend split)."""
    Nfull = float(max(data["Ns_gpu"] if "Ns_gpu" in data else data["Ns"]))
    for name in names:
        if name not in series:
            continue
        col, ls, mk = _STYLE[name]
        Ns = numpy.asarray(Nmap.get(name, data["Ns"]), dtype=float)
        t = numpy.asarray(series[name], dtype=float)
        y = t if mode == "walltime" else Ns / t
        ax.loglog(Ns, y, marker=mk, ms=4, color=col, ls=ls, label=name, zorder=3)
    ax.set_xlabel(r"number of phase-space points $N$ per call")
    if mode == "walltime":
        ax.set_ylabel(r"wall time for the batch  (s)")
        ax.set_title("(a) wall time")
    else:
        ax.set_ylabel(r"throughput  (points / second)")
        ax.set_title("(b) throughput")
        ax.set_ylim(*ax.get_ylim())
        for t0 in (1e-3, 1e-2, 1e-1, 1e0):  # constant wall-time/call references
            ax.plot([1.0, Nfull], [1.0 / t0, Nfull / t0], ls=":", lw=0.8, color="0.82", zorder=0)
        ax.text(0.025, 0.975, "dotted: constant wall-time / call\n(1, 10, 100, 1000 ms)",
                transform=ax.transAxes, fontsize=6.2, color="0.55", va="top", ha="left")
