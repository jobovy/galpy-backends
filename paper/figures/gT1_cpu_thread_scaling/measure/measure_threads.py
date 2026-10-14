"""gT1 measure -- CPU thread scaling of orbit integration (one framework, one core count).

The g12 high-N experiment (MWPotential2014, Sun-like ICs perturbed per-N-seeded,
2 Gyr = linspace(0, 56.2, 16), rtol = atol = 1e-8; the same ICs as
../../_shared/orbit_speed/galpy_{c,jax}_measure.py) run on K CPU cores:

  * c     -- galpy's compiled dop853_c; its orbit loop is OpenMP-parallel, so
             OMP_NUM_THREADS=K (set here, before galpy loads) + the affinity.
  * jax   -- galpy's in-backend diffrax Dopri8, jax.vmap over the batch, jit
             (compiled ahead of time with .lower().compile(), so the compile is
             timed separately and no un-timed full-size warm-up run is needed).
             jaxlib 0.10.1 sizes the XLA:CPU thread pool from the CPU affinity and
             ignores XLA_FLAGS thread settings: the affinity is the only limit.
  * torch -- galpy's in-backend torch path (integrate_orbit's default engine,
             torchdiffeq dopri5, eager; the batch is one (N, 6) state with a shared
             adaptive controller -- the g12 torch line), torch.set_num_threads(K).

The core set is the caller's ``taskset`` (asserted to have K cores). Each timed
call records wall time and process CPU time (all threads; time.process_time), so
cpu/wall = the number of cores the framework actually kept busy.

    taskset -c <list> python measure_threads.py {c,jax,torch} K --Ns 1e4,1e5

Merges its points into data/<fw>.json (keyed by (K, N); other points kept).
"""

import argparse
import json
import os
import socket
import statistics
import subprocess
import sys
import time

_ap = argparse.ArgumentParser()
_ap.add_argument("fw", choices=("c", "jax", "torch"))
_ap.add_argument("k", type=int, help="cores (the taskset size) and thread count")
_ap.add_argument("--Ns", default="1e4,1e5")
_ap.add_argument("--max-repeats", type=int, default=3)
_ap.add_argument("--single-above", type=float, default=60.0,
                 help="time ONE call when the first timed call exceeds this many s")
_ap.add_argument("--out", default=None, help="output JSON (default data/<fw>.json)")
_args = _ap.parse_args()
FW, K = _args.fw, _args.k
Ns = [int(float(s)) for s in _args.Ns.split(",")]

aff = sorted(os.sched_getaffinity(0))
assert len(aff) == K, f"affinity has {len(aff)} cores, expected K={K}: {aff}"
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = str(K)  # before any framework loads
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np  # noqa: E402

import galpy  # noqa: E402

_WT = os.environ.get("GALPY_WORKTREE", "/var/tmp/fbfinal")
assert galpy.__file__.startswith(_WT), galpy.__file__
from galpy.potential import MWPotential2014  # noqa: E402

nt = 16
T_END = 56.2  # 2 Gyr in galpy internal units (ro=8, vo=220)
ic0 = np.array([1.0, 0.0, 1.0, 0.0125, 0.036, 0.0])  # Sun-like [R,vR,vT,z,vz,phi]
scale = np.array([0.05, 0.01, 0.0, 0.01, 0.01, 0.0])
RTOL = ATOL = 1e-8

versions = {"galpy": galpy.__version__, "numpy": np.__version__}
try:
    versions["galpy_git"] = subprocess.check_output(
        ["git", "-C", _WT, "rev-parse", "--short=9", "HEAD"], text=True).strip()
except Exception:
    versions["galpy_git"] = "?"

if FW == "c":
    from galpy.orbit import Orbit

    ts = np.linspace(0.0, T_END, nt)
    integrator = "dop853_c (compiled C, OpenMP over orbits)"

    def make_ics(N):  # same ICs as _shared/orbit_speed/galpy_c_measure.py
        rng = np.random.default_rng(N)
        base = np.tile(ic0, (N, 1))
        base[:, 0] += rng.normal(0, 0.05, N)
        base[:, 1] += rng.normal(0, 0.01, N)
        base[:, 3] += rng.normal(0, 0.01, N)
        base[:, 4] += rng.normal(0, 0.01, N)
        return base

    def prepare(N):
        ics = make_ics(N)
        w = Orbit(ics[:100])  # warm-up: library + potential caches
        w.integrate(ts, MWPotential2014, method="dop853_c", rtol=RTOL, atol=ATOL)

        def run():
            o = Orbit(ics)
            o.integrate(ts, MWPotential2014, method="dop853_c", rtol=RTOL, atol=ATOL)
            return o

        def final(o):
            return np.stack([o.x(ts[-1]), o.y(ts[-1]), o.z(ts[-1])], axis=-1)

        return run, final, {}

elif FW == "jax":
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    from galpy.backend._reference import integrate_orbit

    import diffrax

    versions.update(jax=jax.__version__, jaxlib=jax.lib.__version__ if hasattr(jax.lib, "__version__") else "?",
                    diffrax=diffrax.__version__)
    integrator = "diffrax Dopri8 (in-backend), jax.vmap over orbits, jit (AOT-compiled)"
    ts = jnp.linspace(0.0, T_END, nt)

    def _batch(ics):
        return jax.vmap(lambda v: integrate_orbit(MWPotential2014, v, ts, rtol=RTOL, atol=ATOL))(ics)

    def prepare(N):  # same ICs as _shared/orbit_speed/galpy_jax_measure.py
        ics = jnp.array(ic0)[None] + jax.random.normal(jax.random.PRNGKey(N), (N, 6)) * jnp.array(scale)
        t0 = time.perf_counter()
        exe = jax.jit(_batch).lower(ics).compile()
        compile_s = time.perf_counter() - t0

        def run():
            return exe(ics).block_until_ready()

        def final(ys):  # (nt, N, 6) Orbit order [R,vR,vT,z,vz,phi]
            y = np.asarray(ys[-1])
            return np.stack([y[:, 0] * np.cos(y[:, 5]), y[:, 0] * np.sin(y[:, 5]), y[:, 3]], axis=-1)

        return run, final, {"compile_s": compile_s}

else:
    import torch
    import torchdiffeq

    torch.set_num_threads(K)
    torch.set_default_dtype(torch.float64)
    from galpy.backend._reference import integrate_orbit

    versions.update(torch=torch.__version__, torchdiffeq=torchdiffeq.__version__)
    integrator = ("torchdiffeq dopri5 (in-backend, eager; integrate_orbit default engine), "
                  "batched (N,6) state, shared adaptive controller")
    ts = torch.linspace(0.0, T_END, nt)

    def make_ics(N):
        g = torch.Generator().manual_seed(N)
        return torch.tensor(ic0)[None] + torch.randn(N, 6, generator=g) * torch.tensor(scale)

    def prepare(N):
        ics = make_ics(N)
        with torch.no_grad():
            integrate_orbit(MWPotential2014, ics[:100], ts, rtol=RTOL, atol=ATOL)  # warm-up

        def run():
            with torch.no_grad():
                return integrate_orbit(MWPotential2014, ics, ts, rtol=RTOL, atol=ATOL)

        def final(ys):
            y = ys[-1].numpy()
            return np.stack([y[:, 0] * np.cos(y[:, 5]), y[:, 0] * np.sin(y[:, 5]), y[:, 3]], axis=-1)

        return run, final, {"torch_num_threads": torch.get_num_threads(),
                            "torch_num_interop_threads": torch.get_num_interop_threads()}


def _jiffies():  # per-CPU (busy, total) from /proc/stat; idle = idle + iowait
    out = {}
    with open("/proc/stat") as fh:
        for line in fh:
            f = line.split()
            if f[0].startswith("cpu") and f[0] != "cpu":
                v = [int(x) for x in f[1:]]
                out[int(f[0][3:])] = (sum(v) - v[3] - v[4], sum(v))
    return out


_NCPU = os.cpu_count()
# hyper-thread twins of the pinned CPUs (CPU n <-> n+36 here) that lie OUTSIDE the
# affinity: another process there shares the pinned core's execution units
_TWINS = sorted({(c + _NCPU // 2) % _NCPU for c in aff} - set(aff))


def _busy(j0, j1, cpus):  # busy cores over the interval, summed over cpus
    return sum((j1[c][0] - j0[c][0]) / max(1, j1[c][1] - j0[c][1]) for c in cpus)


def _timed(run):
    l0 = os.getloadavg()[0]
    j0 = _jiffies()
    c0, t0 = time.process_time(), time.perf_counter()
    out = run()
    wall = time.perf_counter() - t0
    cpu = time.process_time() - c0
    j1 = _jiffies()
    others = sorted(set(j1) - set(aff))
    return out, {"wall_s": wall, "cpu_s": cpu, "util": cpu / wall, "load1_before": l0,
                 "load1_after": os.getloadavg()[0],
                 # contamination monitors over the call (busy cores, /proc/stat):
                 "busy_own_cpus": _busy(j0, j1, aff),
                 "busy_twin_cpus": _busy(j0, j1, _TWINS),
                 "busy_other_cpus": _busy(j0, j1, others)}


env = {
    "host": socket.gethostname(),
    "cpu_affinity": aff,
    "n_cores": K,
    "twin_cpus_outside_affinity": _TWINS,
    "thread_env": {v: os.environ.get(v, "unset") for v in
                   ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "XLA_FLAGS",
                    "NUMPY_MADVISE_HUGEPAGE", "GT1_NUMA",
                    "JAX_PLATFORMS", "CUDA_VISIBLE_DEVICES")},
    "versions": versions,
    "galpy_file": galpy.__file__,
    "python": sys.version.split()[0],
    "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    "loadavg_start": list(os.getloadavg()),
    # busy cores on the whole box just before this run (/proc/stat over 10 s, iowait
    # counted idle; the chain's idle gate is < 3): load averages are inflated by other
    # users' I/O-bound jobs and are kept only for the record
    "busy_cores_at_start": os.environ.get("GT1_BUSY_CORES"),
}
points = []
for N in Ns:
    run, final, extra = prepare(N)
    calls = []
    out, rec = _timed(run)
    calls.append(rec)
    nrep = _args.max_repeats if rec["wall_s"] < _args.single_above else 1
    for _ in range(nrep - 1):
        out, rec = _timed(run)
        calls.append(rec)
    xyz = final(out)
    del out
    wall = statistics.median(c["wall_s"] for c in calls)
    cpu = statistics.median(c["cpu_s"] for c in calls)
    p = {"fw": FW, "k": K, "N": N, "wall_s": wall, "cpu_s": cpu, "util": cpu / wall,
         "per_orbit_ms": wall / N * 1e3, "n_calls": len(calls),
         "stat": "median" if len(calls) > 1 else "single", "calls": calls,
         "final_xyz_mean": xyz.mean(axis=0).tolist(), "final_xyz_orbit0": xyz[0].tolist(),
         **extra}
    points.append(p)
    print(f"[{FW} k={K:2d}] N={N:7d}: wall={wall:9.3f} s ({wall / N * 1e3:.4f} ms/orbit) "
          f"cpu/wall={cpu / wall:5.2f}  calls={len(calls)}  {extra}", flush=True)

env["loadavg_end"] = list(os.getloadavg())
_OUT = _args.out or os.path.join(os.path.dirname(__file__), "..", "data", f"{FW}.json")
doc = {"fw": FW, "integrator": integrator, "nt": nt, "tspan_Gyr": 2.0, "rtol": RTOL, "atol": ATOL,
       "potential": "MWPotential2014", "runs": {}, "points": []}
if os.path.exists(_OUT):
    with open(_OUT) as fh:
        doc = json.load(fh)
doc["integrator"] = integrator
doc["runs"][str(K)] = env
keys = {(p["k"], p["N"]) for p in points}
doc["points"] = sorted([p for p in doc["points"] if (p["k"], p["N"]) not in keys] + points,
                       key=lambda p: (p["N"], p["k"]))
os.makedirs(os.path.dirname(os.path.abspath(_OUT)), exist_ok=True)
with open(_OUT, "w") as fh:
    json.dump(doc, fh, indent=1)
print("wrote", _OUT)
