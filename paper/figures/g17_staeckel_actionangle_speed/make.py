"""G17 -- Staeckel action-angle throughput: C vs numpy vs jax/torch (CPU & GPU).

The action-angle analog of G2. ``actionAngleStaeckel.actionsFreqsAngles`` (the full
jr,Lz,jz,Omega_r,Omega_phi,Omega_z,theta_r,theta_phi,theta_z for N phase-space
points) is now backend-agnostic and *vectorised* in the c=False path, so it runs --
and differentiates -- under numpy/jax/torch in addition to the fast C path. This
figure is the headline performance picture for that stack:

  * C (c=True): the compiled per-object reference loop -- the bar to beat; on one
    thread (OMP_NUM_THREADS=1) and, as an extra line, on 16 OpenMP threads.
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

What it shows (re-measured 2026-10-08; every CPU series on ONE core, plus C on 16
OpenMP threads from measure/c_omp_threads.py): single-thread C is fastest on one
core at every N (0.31 ms/point at large N; 310 s at N=1e6); vectorised numpy costs
2.5-3x that at large N (761 s at 1e6) and is 5-6x faster than the old per-object
loop for N>~1e3; eager torch CPU ~ numpy at large N; eager jax CPU is
dispatch-bound (~4 s floor); single-core jitted jax is 5 ms at N=1 but slower
than numpy at large N. C on 16 threads: 26 s at 1e6 (12x). GPUs: H100 eager
torch 3.5 s at N=1e6 (89x one C thread, 7.5x 16 threads), H100 jitted jax 0.16 s
at N=1e5; the TITAN Xp beats one C thread 6-8x at large N but not 16 threads, and
runs out of memory for eager jax beyond 1e5 and torch at 1e6. The backends make
the action-angle map *differentiable and composable* inside jax/torch models; on
a modern GPU they are also the fastest way to evaluate it at large N.

Plot: colours from _harness/palette.py (framework x device; jit solid, eager dashed,
+transfer dotted), lines labelled directly; the GPU series of an H100 measurement in
data/h100/ (same JSON name) are drawn as extra lines when present.

Reproducibility: timings are machine-dependent, so the *measure* step writes
data/g17_staeckel_actionangle_speed.json (stamped with GPU, library versions,
thread settings and per-series child stamps) and the *plot* step is byte-stable
from that JSON. Re-measure with ``OMP_NUM_THREADS=1 python make.py`` in the GPU env
(every series in its own child process, CPU series pinned to one core,
checkpointed to <json>.partial after each series and resumed from it; see
``_measure``); ``python make.py --help`` for dry-run options. ``verify`` only
checks the (deterministic) plot.
"""

import json
import os
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy

from paper.figures._harness import bench, layout, palette
from paper.figures._harness.figure import figure

_DATA = os.path.join(os.path.dirname(__file__), "data")
_JSON = os.path.join(_DATA, "g17_staeckel_actionangle_speed.json")
# the same measurement on an H100 (Trillium): GPU series drawn as extra lines
_JSON_H100 = os.path.join(_DATA, "h100", "g17_staeckel_actionangle_speed.json")
# journal version (default): printed size, no suptitle / env stamp / long notes;
# the caption carries them. GALPY_JAX_PAPER=0 restores the annotated figure.
PAPER = os.environ.get("GALPY_JAX_PAPER", "1") == "1"
# C on 16 OpenMP threads (measure/c_omp_threads.py): drawn next to the 1-thread C
_JSON_C16 = os.path.join(_DATA, "g17_c_16threads.json")
# PAPER: shorter label suffixes (the caption explains them)
_SUF_PAPER = {"c=True": "", "vectorised c=False": "vectorised",
              "per-object, old c=False": "old per-object loop"}

# palette.py: colour = framework x device; jit solid, eager dashed. Variants:
# "+transfer" (eager, incl. host<->device copies) dotted; the OLD per-object
# numpy loop dotted black. (framework, mode, line-style override, label suffix)
_SPEC = {
    "C (c=True)": ("c", None, None, "c=True"),
    "numpy (CPU)": ("numpy", None, None, "vectorised c=False"),
    "numpy (CPU, per-object, OLD)": ("numpy", None, ":", "per-object, old c=False"),
    "jax (CPU)": ("jax", "eager", None, ""),
    "jax (CPU, jit)": ("jax", "jit", None, ""),
    "jax (GPU)": ("jax", "eager", None, ""),
    "jax (GPU, jit)": ("jax", "jit", None, ""),
    "jax (GPU, +transfer)": ("jax", "eager", ":", "+transfer"),
    "torch (CPU)": ("torch", "eager", None, ""),
    "torch (GPU)": ("torch", "eager", None, ""),
    "torch (GPU, +transfer)": ("torch", "eager", ":", "+transfer"),
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


def _safe(fn):  # a large-N call may OOM the device -> record NaN
    try:
        return fn()
    except Exception as e:
        print("  skip (likely OOM):", str(e)[:70], flush=True)
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
        return float("nan")


# --- measurement --------------------------------------------------------------
# Every series runs in its OWN child process (_series_child): jax and torch never
# share a process's GPU memory, each GPU series starts from an empty memory pool,
# and a crash costs one series. After every series the results are checkpointed to
# <out>.partial (temp file + rename); a restarted _measure() resumes from it and
# skips the series already there (a crashed series is re-run). The finished
# measurement is renamed to <out>. Delete <out>.partial to start from scratch.
_NS = [1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000, 300000, 1000000]
_OLD_WT = "/var/tmp/staeckel-old"  # pre-vectorisation galpy build (per-object loop)
_OLD = "numpy (CPU, per-object, OLD)"
# eager jax (and its +transfer twin) on a GPU with <= _SMALL_GPU_GB of memory: run
# only N <= _JAX_EAGER_GPU_CAP_SMALL (NaN above; the cap is stamped in the env).
# On the 12 GB TITAN Xp, N=1e6 sends XLA's BFC allocator into its out-of-memory
# retry loop (allocation fails -> 10 s wait -> retry, hundreds of times).
_SMALL_GPU_GB = 16.0
_JAX_EAGER_GPU_CAP_SMALL = 300000
# a child whose output shows an allocator OOM retry is killed at once and its
# point (and every larger N) recorded NaN: a time never includes the retry waits
_OOM_RETRY = ("bfc_allocator", "ran out of memory")
_MARK = "@@g17 "
# name: (framework, device, kind, N-grid key), in measurement order
_RUN = {
    "C (c=True)": ("c", "cpu", None, "Ns"),
    "numpy (CPU)": ("numpy", "cpu", None, "Ns"),
    "jax (CPU)": ("jax", "cpu", "eager", "Ns_jaxcpu"),
    # jitted jax (how a user runs it for speed: compile once at a fixed shape --
    # e.g. inside an optimization/training loop -- then call repeatedly). Compile
    # is EXCLUDED; the WARM call is timed (as in g2/g8/g15). galpy adds no
    # internal jit -- the CALLER jits, exactly as done here.
    "jax (CPU, jit)": ("jax", "cpu", "jit", "Ns_jaxjit_cpu"),
    "torch (CPU)": ("torch", "cpu", "eager", "Ns_torchcpu"),
    _OLD: (None, "cpu", None, "Ns_old"),
    "jax (GPU, jit)": ("jax", "gpu", "jit", "Ns_jaxjit_gpu"),
    "jax (GPU)": ("jax", "gpu", "eager", "Ns_gpu"),
    "jax (GPU, +transfer)": ("jax", "gpu", "transfer", "Ns_gpu"),  # host->device->host
    "torch (GPU)": ("torch", "gpu", "eager", "Ns_gpu"),
    "torch (GPU, +transfer)": ("torch", "gpu", "transfer", "Ns_gpu"),
}


def _grids(Ns=None):
    """The N grid of every series. ``Ns`` (a dry run): every grid is ``Ns``."""
    if Ns is not None:
        Ns = [int(n) for n in Ns]
        return {k: list(Ns) for k in ("Ns", "Ns_jaxcpu", "Ns_torchcpu", "Ns_gpu",
                                      "Ns_old", "Ns_jaxjit_cpu", "Ns_jaxjit_gpu")}
    return {
        # C/numpy are cheap (-> 1e6); the eager backends are dispatch-heavy
        # (root-find + quadrature = hundreds of ops/call), so CPU eager is capped
        # lower; the old O(N) Python loop hardest
        "Ns": _NS,
        "Ns_jaxcpu": [n for n in _NS if n <= 30000],  # jax eager: dispatch-bound
        "Ns_torchcpu": [n for n in _NS if n <= 100000],
        "Ns_gpu": _NS,  # 3e6 OOMs the 12 GB GPU (quadrature (N, order) arrays)
        "Ns_old": [1, 3, 10, 30, 100, 300, 1000, 3000],
        # jitted jax: compile ~3 s/shape (rolled fori_loop bisection); a few points
        # show the floor and the large-N rise
        "Ns_jaxjit_cpu": [1, 100, 3000, 30000],
        "Ns_jaxjit_gpu": [1, 100, 3000, 100000],  # < 1e6: compile-time OOM risk
    }


def _series_child(name, Ns):
    """Child process of _run_series: time one series at ``Ns``; marker lines
    (_MARK) on stdout: "N <n>" before a point, "T <n> <seconds>" after it, "ENV
    <json>" before the first point and at the end, "NODEV" when the series' device
    is absent."""
    import platform

    fw, dev, kind, _ = _RUN[name]
    env = {"python": platform.python_version(), "numpy": numpy.__version__,
           "affinity": len(os.sched_getaffinity(0))}
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "XLA_FLAGS"):
        env[v] = os.environ.get(v, "unset")
    import galpy
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.potential import MWPotential2014

    env["galpy"] = galpy.__version__
    aa = actionAngleStaeckel(pot=MWPotential2014, delta=0.45, c=(fw == "c"))
    cap = None
    if fw in ("c", "numpy"):
        def time_at(N):
            g = _grid(N)
            aa.actionsFreqsAngles(*g)
            return _med(lambda: aa.actionsFreqsAngles(*g), warmup=1,
                        repeats=5 if N <= 100000 else 3)
    elif fw == "jax":
        import jax

        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp

        env["jax"] = jax.__version__
        ds = [d for d in jax.devices() if d.platform == ("gpu" if dev == "gpu" else "cpu")]
        if not ds:
            print(_MARK + "NODEV", flush=True)
            return
        d = ds[0]
        if dev == "gpu":
            env["gpu"] = d.device_kind
            try:  # the allocator limit is a fraction of the device memory
                lim = d.memory_stats()["bytes_limit"]
                frac = float(os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", 0.75))
                env["gpu_mem_gb"] = round(lim / frac / 2**30, 2)
                if kind in ("eager", "transfer") and env["gpu_mem_gb"] <= _SMALL_GPU_GB:
                    cap = _JAX_EAGER_GPU_CAP_SMALL
            except Exception:
                pass

        def put(N):
            return [jax.device_put(jnp.asarray(b), d) for b in _grid(N)]

        def time_at(N):
            if kind == "eager":
                g = put(N)
                jax.block_until_ready(aa.actionsFreqsAngles(*g))
                return _med(lambda: aa.actionsFreqsAngles(*g), warmup=2, repeats=3,
                            sync="jax")
            if kind == "jit":
                g = put(N)
                f = jax.jit(lambda *a: aa.actionsFreqsAngles(*a))
                jax.block_until_ready(f(*g))  # compile (excluded from timing)
                return _med(lambda: f(*g), warmup=2, repeats=3, sync="jax")
            gn = _grid(N)  # +transfer: host numpy -> device -> compute -> host
            return _med(lambda: numpy.asarray(aa.actionsFreqsAngles(
                *[jax.device_put(jnp.asarray(b), d) for b in gn])[0]), warmup=2, repeats=3)
    else:  # torch
        import torch

        torch.set_default_dtype(torch.float64)
        env["torch"] = torch.__version__
        env["torch_threads"] = torch.get_num_threads()
        tdev = "cuda" if dev == "gpu" else "cpu"
        if dev == "gpu":
            if not torch.cuda.is_available():
                print(_MARK + "NODEV", flush=True)
                return
            env["gpu"] = torch.cuda.get_device_name(0)
            env["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
        sync = "torch" if dev == "gpu" else "none"

        def time_at(N):
            if kind == "eager":
                g = [torch.as_tensor(b, device=tdev) for b in _grid(N)]
                aa.actionsFreqsAngles(*g)
                if dev == "gpu":
                    torch.cuda.synchronize()
                return _med(lambda: aa.actionsFreqsAngles(*g), warmup=2, repeats=3, sync=sync)
            gn = _grid(N)  # +transfer
            return _med(lambda: aa.actionsFreqsAngles(
                *[torch.as_tensor(b, device=tdev) for b in gn])[0].cpu().numpy(),
                warmup=2, repeats=3)
    if cap is not None:
        env["eager_cap_N"] = cap
    print(_MARK + "ENV " + json.dumps(env), flush=True)  # kept if the child is killed
    for N in Ns:
        print(f"{_MARK}N {N}", flush=True)
        t = float("nan") if cap is not None and N > cap else _safe(lambda N=N: time_at(N))
        print(f"{_MARK}T {N} {t!r}", flush=True)
    env["loadavg"] = list(os.getloadavg())
    print(_MARK + "ENV " + json.dumps(env), flush=True)


def _run_series(name, Ns, pin=None):
    """Run _series_child(name, Ns) in a child process (CPU series: GPUs hidden;
    ``pin``: the child's one core). (times, status, child env); status "ok",
    "nodev", "oom-retry at N=<n>" or "crashed ...". The child's output is streamed;
    an allocator OOM retry kills it, NaN from that point on."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        p for p in (os.environ.get("PYTHONPATH", ""), root) if p))
    if _RUN[name][1] == "cpu":
        env.update(JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="")
    code = ("from paper.figures.g17_staeckel_actionangle_speed.make import _series_child; "
            f"_series_child({name!r}, {list(Ns)!r})")
    pre = (lambda: os.sched_setaffinity(0, {pin})) if pin is not None else None
    proc = subprocess.Popen([sys.executable, "-c", code], cwd=root, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
                            preexec_fn=pre)
    times, cenv, cur, status = {}, {}, None, None
    for line in proc.stdout:
        if line.startswith(_MARK):
            tag, _, rest = line[len(_MARK):].strip().partition(" ")
            if tag == "N":
                cur = int(rest)
            elif tag == "T":
                n, t = rest.split()
                times[int(n)], cur = float(t), None
            elif tag == "ENV":
                cenv = json.loads(rest)
            elif tag == "NODEV":
                status = "nodev"
            if tag != "ENV":
                print(f"  [{name}] {line.strip()}", flush=True)
            continue
        print(line, end="", flush=True)
        if all(s in line for s in _OOM_RETRY):
            status = f"oom-retry at N={cur}"
            print(f"  [{name}] allocator OOM retry at N={cur}: child killed, NaN from "
                  "this N on", flush=True)
            proc.kill()
            break
    proc.stdout.close()
    rc = proc.wait()
    if status is None:
        status = "ok" if rc == 0 else f"crashed rc={rc} at N={cur}"
    return [times.get(N, float("nan")) for N in Ns], status, cenv


def _checkpoint(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as fh:
        json.dump(data, fh, indent=2)
    os.replace(path + ".tmp", path)


def _stamp(data):
    """Top-level env stamps (read by the plot and torch_gpu_only.py) from the
    parent and the per-series child stamps."""
    env, per = data["env"], data["env"].get("series", {})
    env.update(python=sys.version.split()[0], cpu_affinity=len(os.sched_getaffinity(0)),
               loadavg=list(os.getloadavg()))
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "XLA_FLAGS"):
        env[v] = os.environ.get(v, "unset")
    for name, ce in per.items():
        for k in ("numpy", "jax", "torch", "galpy"):
            if k in ce:
                env[k] = ce[k]
        if _RUN[name][1] == "gpu" and "gpu" in ce:
            env["gpu"] = ce["gpu"]
    if "torch (CPU)" in per:
        env["torch_threads"] = per["torch (CPU)"].get("torch_threads")
    if "jax (CPU)" in per:
        env["jax_cpu_affinity"] = per["jax (CPU)"].get("affinity")
    caps = {n: ce["eager_cap_N"] for n, ce in per.items() if "eager_cap_N" in ce}
    if caps:
        env["jax_gpu_eager_cap_N"] = caps
    env.setdefault("gpu", "none")


def _measure(out=None, Ns=None, gpu=True, cpu=True):
    """Measure every series (each in its own child process; see _run_series) into
    ``out`` (default _JSON), checkpointing to ``out``.partial after each series and
    resuming from it. With OMP_NUM_THREADS=1 every CPU series runs pinned to ONE
    core: XLA:CPU sizes its thread pool from the CPU affinity (jax 0.10.1 ignores
    --xla_cpu_multi_thread_eigen=false; measured cpu/wall ~6-60 with it, 1.0
    pinned). ``Ns``: one N list for every series (dry runs); ``gpu=False``: CPU
    series only; ``cpu=False``: GPU series only (the H100 runs: the paper's CPU
    series come from the dev box)."""
    out = out or _JSON
    partial = out + ".partial"
    grids = _grids(Ns)
    data = {**grids, "series": {}, "status": {}, "env": {}}
    if os.path.exists(partial):
        with open(partial) as fh:
            old = json.load(fh)
        if all(old.get(k) == v for k, v in grids.items()):
            data = old
            data.setdefault("status", {})
            print("resuming from", partial, "with", sorted(data["series"]), flush=True)
        else:
            print("ignoring", partial, "(different N grids)", flush=True)
    pin = max(os.sched_getaffinity(0)) if os.environ.get("OMP_NUM_THREADS") == "1" else None
    for name, (fw, dev, kind, key) in _RUN.items():
        if (dev == "gpu" and not gpu) or (dev == "cpu" and not cpu):
            continue
        if name in data["series"] and not data["status"].get(name, "ok").startswith("crashed"):
            print(f"[{name}] in checkpoint ({data['status'].get(name, 'ok')}), skipped", flush=True)
            continue
        print(f"[{name}] N = {grids[key]}", flush=True)
        if name == _OLD:
            if not os.path.isdir(_OLD_WT):
                continue
            try:
                times, status, cenv = _measure_old(_OLD_WT, grids[key]), "ok", {"galpy_path": _OLD_WT}
            except Exception as e:
                print("OLD measurement skipped:", e)
                continue
        else:
            times, status, cenv = _run_series(name, grids[key], pin if dev == "cpu" else None)
        if status == "nodev":
            continue
        data["series"][name] = times
        data["status"][name] = status
        data["env"].setdefault("series", {})[name] = cenv
        _stamp(data)
        _checkpoint(data, partial)
        print(f"[{name}] {status}; checkpointed", flush=True)
    _stamp(data)
    _checkpoint(data, partial)
    os.replace(partial, out)
    print("written", out, flush=True)
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


def _data_h100():
    if os.path.exists(_JSON_H100):
        with open(_JSON_H100) as fh:
            return json.load(fh)
    return None


def _Nmap(data):
    return {
        "jax (CPU)": data["Ns_jaxcpu"], "torch (CPU)": data["Ns_torchcpu"],
        "jax (GPU)": data["Ns_gpu"], "jax (GPU, +transfer)": data["Ns_gpu"],
        "torch (GPU)": data["Ns_gpu"], "torch (GPU, +transfer)": data["Ns_gpu"],
        "numpy (CPU, per-object, OLD)": data["Ns_old"],
        "jax (CPU, jit)": data.get("Ns_jaxjit_cpu", data["Ns"]),
        "jax (GPU, jit)": data.get("Ns_jaxjit_gpu", data["Ns_gpu"]),
    }


def _cpu_threads(data):
    """{series: "1 core" / "N threads"} for the CPU series, from the measurement's
    env stamps ({} for a JSON without them). numpy and the old loop are
    single-threaded (OPENBLAS_NUM_THREADS=1); C threads over points with OpenMP
    (OMP_NUM_THREADS); torch uses torch.get_num_threads(); the jax CPU series run on
    jax_cpu_affinity cores (1: pinned by _measure under OMP_NUM_THREADS=1; otherwise XLA's
    pool spans the whole affinity, used only by its larger kernels: "up to")."""
    env = data.get("env", {})
    if "cpu_affinity" not in env:
        return {}
    one = "1 core"
    out = {"numpy (CPU)": one, "numpy (CPU, per-object, OLD)": one}
    omp = env.get("OMP_NUM_THREADS", "unset")
    if omp != "unset":
        out["C (c=True)"] = "1 thread" if omp == "1" else f"{omp} threads"
    if "torch_threads" in env:
        n = int(env["torch_threads"])
        out["torch (CPU)"] = one if n == 1 else f"{n} threads"
    n = int(env.get("jax_cpu_affinity", env["cpu_affinity"]))
    out["jax (CPU)"] = out["jax (CPU, jit)"] = one if n == 1 else f"up to {n} threads"
    return out


def _curves(names):
    """[(name, Ns, times, plot kwargs, label)] for the listed series: the TITAN Xp
    measurement, plus each GPU series' H100 twin when data/h100/ has one."""
    out = []
    srcs = [(_data(), None)]
    h = _data_h100()
    if h is not None:
        srcs.append((h, "h100"))
    for data, dev_override in srcs:
        series, Nmap = data["series"], _Nmap(data)
        for name in names:
            if name not in series:
                continue
            fw, mode, ls, suf = _SPEC[name]
            dev = palette.parse(name)[1]
            if dev_override:
                if dev == "cpu":
                    continue  # only the GPU series are re-measured on the H100
                dev = dev_override
            st = palette.style(fw, dev, mode)
            if ls:
                st["ls"] = ls
            if name == "numpy (CPU, per-object, OLD)":
                st["marker"] = "x"
            Ns = numpy.asarray(Nmap.get(name, data["Ns"]), dtype=float)
            if PAPER:
                suf = _SUF_PAPER.get(suf, suf)
                thr = _cpu_threads(data).get(name) if dev == "cpu" else None
                if thr:
                    suf = f"{suf}, {thr}" if suf else thr
            lab = palette.label(fw, dev, mode, suf)
            out.append((name, Ns, numpy.asarray(series[name], dtype=float), st, lab))
        if dev_override is None and "C (c=True)" in names and os.path.exists(_JSON_C16):
            with open(_JSON_C16) as fh:
                c16 = json.load(fh)
            key = next(k for k in c16 if k.startswith("C (c=True"))
            st = dict(palette.style("c", "cpu"), ls="-.")
            out.append((key, numpy.asarray(c16["Ns"], dtype=float),
                        numpy.asarray(c16[key], dtype=float), st,
                        "galpy C (%s threads)" % c16.get("env", {}).get("omp_num_threads", 16)
                        if PAPER else "galpy C (c=True, 16 threads)"))
    return out


def _make(names, backend_label):
    data = _data()
    env = data.get("env", {})
    h = _data_h100()
    curves = _curves(names)
    Nfull = float(max(data["Ns_gpu"] if "Ns_gpu" in data else data["Ns"]))
    if PAPER:  # printed width of a figure* (7.0 in); stacked, labels at the right
        fig, (axw, axt) = plt.subplots(2, 1, figsize=layout.figsize(PAPER, (7.0, 6.2), None))
    else:
        fig, (axw, axt) = plt.subplots(1, 2, figsize=(13.0, 4.8))
    _plot_panel(axw, curves, Nfull, "walltime")
    _plot_panel(axt, curves, Nfull, "throughput")
    if PAPER:
        fig.tight_layout()
        fs = layout.FS["line"]
        for ax in (axw, axt):
            palette.label_lines(ax, fontsize=fs, min_sep_pt=1.3 * fs)
        layout.fit_right(fig)
        return fig
    stamp = f"{env.get('gpu', '?')}: jax {env.get('jax', '?')} · torch {env.get('torch', '?')}"
    if h is not None:
        he = h.get("env", {})
        stamp += f"\n{he.get('gpu', 'H100')}: jax {he.get('jax', '?')} · torch {he.get('torch', '?')}"
    axt.text(0.97, 0.03, stamp, transform=axt.transAxes, fontsize=6.0, color="0.5",
             va="bottom", ha="right")
    fig.suptitle(r"Staeckel $(\mathbf{J},\mathbf{\Omega},\mathbf{\theta})$ via actionsFreqsAngles "
                 r"vs batch size $N$ -- " + backend_label, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97], w_pad=11.0)
    for ax in (axw, axt):
        palette.label_lines(ax, fontsize=6.3)
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


def _plot_panel(ax, curves, Nfull, mode):
    """One log-log panel: mode='walltime' (s) [matches the orbit speed figures]
    or mode='throughput' (points/s). NaN points (OOM-skipped) drop out."""
    for name, Ns, t, st, lab in curves:
        y = t if mode == "walltime" else Ns / t
        ax.loglog(Ns, y, ms=3.5, label=lab, zorder=3, **st)
    ax.set_xlabel(r"number of phase-space points $N$ per call")
    if mode == "walltime":
        ax.set_ylabel(r"wall time for the batch  (s)")
        ax.set_title("(a) wall time")
    else:
        ax.set_ylabel(r"throughput  (points / second)")
        ax.set_title("(b) throughput")
        ax.set_ylim(*ax.get_ylim())
        for t0 in (1e-3, 1e-2, 1e-1, 1e0):  # constant wall-time/call references
            ax.plot([1.0, Nfull], [1.0 / t0, Nfull / t0], ls=":", lw=0.8, color="0.82",
                    zorder=0, label="_ref")
        ax.text(0.025, 0.975, "dotted: 1 ms–1 s per call" if PAPER else
                "dotted grey: constant wall-time / call\n(1, 10, 100, 1000 ms)",
                transform=ax.transAxes, fontsize=layout.FS["note"] if PAPER else 6.2,
                color="0.55", va="top", ha="left")


if __name__ == "__main__":  # measure (see _measure); the plot is ``run``
    import argparse

    ap = argparse.ArgumentParser(description="Measure g17 (each series in its own "
                                 "child process; checkpointed to <out>.partial).")
    ap.add_argument("--out", default=_JSON, help="output JSON (default: %(default)s)")
    ap.add_argument("--Ns", help="comma list: one N grid for every series (dry runs)")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--cpu-only", action="store_true", help="skip the GPU series")
    grp.add_argument("--gpu-only", action="store_true",
                     help="skip the CPU series (C, numpy, old loop, jax/torch CPU)")
    args = ap.parse_args()
    _measure(out=args.out, Ns=None if args.Ns is None else [int(float(n)) for n in args.Ns.split(",")],
             gpu=not args.cpu_only, cpu=not args.gpu_only)
