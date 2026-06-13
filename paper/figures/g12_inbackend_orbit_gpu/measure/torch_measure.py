"""g12 measure -- galpy's in-backend (torchdiffeq) orbit integrator on torch/GPU.

Same MWPotential2014 / 2 Gyr / Sun-like-batch experiment as the shared jax measure,
but through galpy's torch in-backend path (torchdiffeq dopri5). Batched by carrying
the state as shape (6, N): galpy's backend-agnostic _eom_rhs indexes y[0]..y[5]
componentwise, so a (6, N) state vectorises all N orbits through one odeint call --
the same EOM and force layer the public per-orbit integrate_orbit uses.

KEY STORY (the figure's torch line): unlike jax/diffrax, torchdiffeq runs *eager*
-- there is no jit-fuse of the adaptive-step loop, so every one of the ~thousands of
Dopri5 steps over 2 Gyr pays full Python dispatch overhead. At small N the batch is
*overhead-bound*: the wall time is a near-constant ~12 s set by the Python step loop,
the GPU is starved, and per-orbit cost falls as ~1/N. We push N out to large batches
to capture the transition to the *compute-bound* regime, where the per-step GPU
kernels finally dominate the Python overhead: wall time starts rising ~linearly with
N and ms/orbit flattens to torch's true per-orbit asymptote. We stop at the first
OOM or when a single steady run exceeds the wall-time budget (recorded in the JSON).
Correct and fully differentiable, but ~40x slower than the jit-fused JAX-GPU line --
which is exactly why g8 omits a torch line; g12 includes it to SHOW the gap and the
eager overhead->compute transition.

Uses the hiN (nt=16 sparse-save) convention -- torch is step-bound not save-bound, and
few saves keep the (nt, 6, N) output small so memory is the (6,N) state, letting the
sweep reach large N.

Run STRICTLY SERIALLY on the GPU (one device-timed process at a time):
    PYTHONPATH=/var/tmp/backends-merge python torch_measure.py

Writes data/torch_<device>.json (machine + version stamped).
"""

import json
import os
import socket
import time

import torch

torch.set_default_dtype(torch.float64)
from torchdiffeq import odeint  # noqa: E402

import galpy  # noqa: E402
from galpy.backend._reference.inbackend_ode import _eom_rhs, _to_eom  # noqa: E402
from galpy.potential import MWPotential2014  # noqa: E402

assert galpy.__file__.startswith("/var/tmp/backends-merge"), galpy.__file__

_DATA = os.path.join(os.path.dirname(__file__), "..", "data")
dev = "cuda" if torch.cuda.is_available() else "cpu"

nt = 16  # hiN convention: torch is step-bound, saves are irrelevant; keeps output small
# Push N from the overhead-bound floor (~1..1e4) into the compute-bound regime
# (where wall time rises ~linearly and ms/orbit flattens). Intermediate points
# (5e5, 7e5) resolve the rise; the 1e6 point earlier OOMed by only ~322 MiB, so this
# process imports NO jax (XLA does not preallocate the GPU here) and frees each N's
# tensors + empties the cache before the next, reclaiming that memory to attempt 1e6.
# Stop on OOM or budget (recorded in stopped_at).
Ns = [1, 10, 100, 1000, 10000, 30000, 100000, 300000, 500000, 700000, 1000000]
WALL_BUDGET_S = 200.0  # stop once a single steady run exceeds this
ts = torch.linspace(0.0, 56.2, nt, device=dev)
ic0 = torch.tensor([1.0, 0.0, 1.0, 0.0125, 0.036, 0.0], device=dev)
scale = torch.tensor([0.05, 0.01, 0.0, 0.01, 0.01, 0.0], device=dev)


def _y0(N):
    g = torch.Generator(device=dev).manual_seed(N)
    ics = ic0[None] + torch.randn(N, 6, generator=g, device=dev) * scale  # (N,6) Orbit
    return _to_eom(torch, ics.T)  # -> (6,N) rectangular EOM state


def _field(t, y):  # y shape (6,N); _eom_rhs is componentwise -> vectorises over N
    return torch.stack(_eom_rhs(y, MWPotential2014, t, torch))


def _sync():
    if dev == "cuda":
        torch.cuda.synchronize()


out = {
    "backend": "galpy-torch",
    "device": dev,
    "integrator": "torchdiffeq dopri5 (in-backend, eager)",
    "stepping": "batched state (6,N), shared adaptive controller",
    "nt": nt,
    "tspan_Gyr": 2.0,
    "rtol": 1e-8,
    "atol": 1e-8,
    "potential": "MWPotential2014",
    "host": socket.gethostname(),
    "wall_budget_s": WALL_BUDGET_S,
    "results": [],
}
for m in ("torch", "galpy", "numpy"):
    try:
        out[m + "_version"] = __import__(m).__version__
    except Exception:
        pass
try:
    out["gpu"] = torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"
except Exception:
    out["gpu"] = dev

for N in Ns:
    try:
        y0 = _y0(N)
        t0 = time.perf_counter()
        odeint(_field, y0, ts, method="dopri5", rtol=1e-8, atol=1e-8)
        _sync()
        first_s = time.perf_counter() - t0
        # large N is expensive: time fewer repeats once a single run is slow
        repeats = 3 if first_s < 30.0 else 1
        best = 1e18
        for _ in range(repeats):
            t0 = time.perf_counter()
            odeint(_field, y0, ts, method="dopri5", rtol=1e-8, atol=1e-8)
            _sync()
            best = min(best, time.perf_counter() - t0)
        out["results"].append(
            {"N": N, "wall_s": best, "per_orbit_ms": best / N * 1e3, "first_s": first_s}
        )
        print(
            f"galpy-torch[{dev}] N={N:8d}: steady={best * 1e3:10.1f} ms "
            f"({best / N * 1e3:.4f} ms/orbit)  first={first_s:7.2f} s",
            flush=True,
        )
        os.makedirs(_DATA, exist_ok=True)
        with open(os.path.join(_DATA, f"torch_{dev}.json"), "w") as fh:
            json.dump(out, fh, indent=2)
        del y0
        if dev == "cuda":
            torch.cuda.empty_cache()
        if best > WALL_BUDGET_S:
            print(
                f"galpy-torch[{dev}] N={N}: STOP wall {best:.1f}s > budget "
                f"{WALL_BUDGET_S:.0f}s",
                flush=True,
            )
            out["stopped_at"] = {"N": N, "reason": "wall_budget", "wall_s": best}
            with open(os.path.join(_DATA, f"torch_{dev}.json"), "w") as fh:
                json.dump(out, fh, indent=2)
            break
    except RuntimeError as e:  # CUDA OOM or step-cap: stop and record where
        print(f"galpy-torch[{dev}] N={N}: STOP {type(e).__name__}: {str(e)[:70]}", flush=True)
        out["stopped_at"] = {"N": N, "reason": type(e).__name__, "msg": str(e)[:120]}
        with open(os.path.join(_DATA, f"torch_{dev}.json"), "w") as fh:
            json.dump(out, fh, indent=2)
        if dev == "cuda":
            torch.cuda.empty_cache()
        break
print("done")
