"""g12 measure -- galpy's in-backend (torchdiffeq) orbit integrator on torch/GPU.

Same MWPotential2014 / 2 Gyr / Sun-like-batch experiment as jax_measure.py, but
through galpy's torch in-backend path (torchdiffeq dopri5). Batched by carrying the
state as shape (6, N): galpy's backend-agnostic _eom_rhs indexes y[0]..y[5]
componentwise, so a (6, N) state vectorises all N orbits through one odeint call --
the same EOM and force layer the public per-orbit integrate_orbit uses.

KEY LIMITATION (the figure's torch story): unlike jax/diffrax, torchdiffeq runs
*eager* -- there is no jit-fuse of the adaptive-step loop, so every one of the
~thousands of Dopri5 steps over 2 Gyr pays full Python dispatch overhead and the
GPU is starved. It is correct and differentiable but impractically slow for large
batches, so this script only sweeps small N (and stops on the first failure/cap).
This is why g8 omits a torch line; g12 includes a few points to SHOW the gap.

Usage (GPU env, backend-migrated galpy on PYTHONPATH):
    python torch_measure.py            # device=cuda if available, else cpu

Writes data/torch_<device>.json (machine + version stamped).
"""

import json
import os
import socket
import time

import torch

torch.set_default_dtype(torch.float64)
from torchdiffeq import odeint  # noqa: E402

from galpy.backend._reference.inbackend_ode import _eom_rhs, _to_eom  # noqa: E402
from galpy.potential import MWPotential2014  # noqa: E402

_DATA = os.path.join(os.path.dirname(__file__), "..", "data")
dev = "cuda" if torch.cuda.is_available() else "cpu"

nt = 16  # torch is step-bound; saves are irrelevant. Few saves, small N.
Ns = [1, 10, 100, 1000, 10000]  # eager torchdiffeq is impractical beyond this
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
    y0 = _y0(N)
    try:
        t0 = time.perf_counter()
        odeint(_field, y0, ts, method="dopri5", rtol=1e-8, atol=1e-8)
        _sync()
        first_s = time.perf_counter() - t0
        best = 1e18
        for _ in range(3):
            t0 = time.perf_counter()
            odeint(_field, y0, ts, method="dopri5", rtol=1e-8, atol=1e-8)
            _sync()
            best = min(best, time.perf_counter() - t0)
        out["results"].append(
            {"N": N, "wall_s": best, "per_orbit_ms": best / N * 1e3, "first_s": first_s}
        )
        print(
            f"galpy-torch[{dev}] N={N:6d}: steady={best * 1e3:10.1f} ms "
            f"({best / N * 1e3:.4f} ms/orbit)  first={first_s:7.2f} s",
            flush=True,
        )
        os.makedirs(_DATA, exist_ok=True)
        with open(os.path.join(_DATA, f"torch_{dev}.json"), "w") as fh:
            json.dump(out, fh, indent=2)
    except Exception as e:
        print(f"galpy-torch[{dev}] N={N}: STOP {type(e).__name__}: {str(e)[:70]}", flush=True)
        break
print("done")
