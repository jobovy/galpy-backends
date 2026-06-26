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
  * jax / torch (GPU): compute-only (solid; arrays resident on device, timed wait
    is the kernel) and end-to-end (dashed; incl. host<->device transfer).

What it shows: C wins at every N for this branchy root-find + quadrature workload
(no surprise -- it is compiled and has no per-op dispatch); among the differentiable
paths numpy is fastest on CPU, the GPU backends are dispatch-bound at small N and
amortise toward large N, and the vectorised numpy path is orders of magnitude faster
than the old per-object loop. The backends are not here to beat C on raw throughput
-- they are here to make the action-angle map *differentiable and composable* inside
jax/torch models; this figure quantifies what that costs.

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
    "jax (GPU)": ("#0072B2", "-", "o"),
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
    Ns_jaxcpu = [n for n in Ns if n <= 3000]  # jax eager is dispatch-bound + slow
    Ns_torchcpu = [n for n in Ns if n <= 30000]
    Ns_gpu = [n for n in Ns if n <= 100000]
    Ns_old = [1, 3, 10, 30, 100, 300, 1000, 3000]
    series = {}

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
            g = [jax.device_put(jnp.asarray(b), dev) for b in _grid(N)]
            r = aF.actionsFreqsAngles(*g)
            jax.block_until_ready(r)
            out.append(_med(lambda g=g: aF.actionsFreqsAngles(*g),
                            warmup=2, repeats=5, sync=sync))
        return out

    series["jax (CPU)"] = _jax_series(cpu, Ns_jaxcpu, "jax")
    if gpu is not None:
        series["jax (GPU)"] = _jax_series(gpu, Ns_gpu, "jax")
        # end-to-end: host numpy -> device -> compute -> host
        e2e = []
        for N in Ns_gpu:
            gn = _grid(N)
            numpy.asarray(
                jax.block_until_ready(
                    aF.actionsFreqsAngles(*[jax.device_put(jnp.asarray(b), gpu) for b in gn])
                )[0]
            )
            e2e.append(_med(
                lambda gn=gn: numpy.asarray(
                    aF.actionsFreqsAngles(*[jax.device_put(jnp.asarray(b), gpu) for b in gn])[0]
                ),
                warmup=2, repeats=5,
            ))
        series["jax (GPU, +transfer)"] = e2e

    # --- torch CPU / GPU ---
    def _torch_series(device, Nlist, sync):
        out = []
        for N in Nlist:
            g = [torch.as_tensor(b, device=device) for b in _grid(N)]
            r = aF.actionsFreqsAngles(*g)
            if sync == "torch":
                torch.cuda.synchronize()
            out.append(_med(lambda g=g: aF.actionsFreqsAngles(*g),
                            warmup=2, repeats=5, sync=sync))
        return out

    series["torch (CPU)"] = _torch_series("cpu", Ns_torchcpu, "none")
    if has_torch_gpu:
        series["torch (GPU)"] = _torch_series("cuda", Ns_gpu, "torch")
        e2e = []
        for N in Ns_gpu:
            gn = _grid(N)
            aF.actionsFreqsAngles(*[torch.as_tensor(b, device="cuda") for b in gn])[0].cpu().numpy()
            e2e.append(_med(
                lambda gn=gn: aF.actionsFreqsAngles(
                    *[torch.as_tensor(b, device="cuda") for b in gn]
                )[0].cpu().numpy(),
                warmup=2, repeats=5,
            ))
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
        "Ns_gpu": Ns_gpu, "Ns_old": Ns_old, "series": series, "env": env,
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


@figure(
    name="g17_staeckel_actionangle_speed",
    track="G",
    seed=0,
    description="Staeckel actionsFreqsAngles throughput vs N: C vs numpy vs jax/torch CPU&GPU; + old per-object loop [measure->json, plot byte-stable]",
)
def make():
    if os.path.exists(_JSON):
        with open(_JSON) as fh:
            data = json.load(fh)
    else:
        data = _measure()
    series = data["series"]
    env = data.get("env", {})
    Nmap = {
        "jax (CPU)": data["Ns_jaxcpu"], "torch (CPU)": data["Ns_torchcpu"],
        "jax (GPU)": data["Ns_gpu"], "jax (GPU, +transfer)": data["Ns_gpu"],
        "torch (GPU)": data["Ns_gpu"], "torch (GPU, +transfer)": data["Ns_gpu"],
        "numpy (CPU, per-object, OLD)": data["Ns_old"],
    }

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for name in _STYLE:
        if name not in series:
            continue
        col, ls, mk = _STYLE[name]
        Ns = numpy.asarray(Nmap.get(name, data["Ns"]), dtype=float)
        t = numpy.asarray(series[name], dtype=float)
        thr = Ns / t
        ax.loglog(Ns, thr, marker=mk, ms=4, color=col, ls=ls, label=name, zorder=3)
    ax.set_xlabel(r"number of phase-space points $N$ per call")
    ax.set_ylabel(r"throughput  (points / second)")
    ax.set_title(r"Staeckel $(\mathbf{J},\mathbf{\Omega},\mathbf{\theta})$ via actionsFreqsAngles")
    ax.set_ylim(*ax.get_ylim())
    Nlo, Nhi = 1.0, float(max(data["Ns"]))
    for t0 in (1e-3, 1e-2, 1e-1, 1e0):
        ax.plot([Nlo, Nhi], [Nlo / t0, Nhi / t0], ls=":", lw=0.8, color="0.82", zorder=0)
    ax.text(0.025, 0.975, "dotted: constant wall-time / call\n(1, 10, 100, 1000 ms)",
            transform=ax.transAxes, fontsize=6.5, color="0.55", va="top", ha="left")
    ax.legend(fontsize=6.6, loc="lower right", framealpha=0.92, handlelength=2.8, ncol=1)
    ax.text(0.30, 0.025,
            f"GPU: {env.get('gpu', '?')}\njax {env.get('jax', '?')} · torch {env.get('torch', '?')}",
            transform=ax.transAxes, fontsize=6.5, color="0.5", va="bottom", ha="left")
    fig.tight_layout()
    return fig
