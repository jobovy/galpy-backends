"""Measure (large N): extend the flat IC-gradient lines until they bend upward.

The flat lines in g15(b) -- diffrax-GPU, torchdiffeq-CPU/GPU -- are overhead-bound
(fixed launch / eager-step-loop cost, batch nearly free). Push N until the per-step
GPU/CPU compute finally dominates the overhead and wall time rises ~linearly (the
compute-bound regime, cf. g12). Adds/updates large-N points in the existing
data/icderiv.json "scaling" block; per-series stop on OOM or wall-budget; incremental
save so the bend is captured wherever it lands.

Single process: jax set to NOT preallocate the GPU so torch can coexist; each cell
frees its tensors + empties the cache before the next.
Run:  PYTHONPATH=<#972-worktree> python measure/measure_largeN.py
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import json
import pathlib
import time

import numpy

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import torch

torch.set_default_dtype(torch.float64)
from torchdiffeq import odeint

from galpy.backend._reference.inbackend_ode import _eom_rhs, _to_eom, integrate_orbit
from galpy.potential import MWPotential2014

_CPU = jax.devices("cpu")[0]
_GPU_ONLY = os.environ.get("G15_GPU_ONLY") == "1"  # H100 runs: skip the CPU series
_GPU = next((d for d in jax.devices() if d.platform == "gpu"), None)
_IC = numpy.array([1.0, 0.1, 0.9, 0.05, 0.1, 0.2])
_TS = numpy.linspace(0.0, 10.0, 100)
ts_j = jnp.asarray(_TS)
_F = pathlib.Path(__file__).resolve().parent.parent / "data" / "icderiv.json"
out = json.loads(_F.read_text())

N_EXT = [4096, 16384, 65536, 262144, 1048576]
PER_RUN_BUDGET_S = 120.0  # stop a series once one timed run exceeds this


def _batch(n):
    return numpy.stack([_IC * (1 + 0.001 * (k % 97)) for k in range(n)])


def _set(n, key, val):
    out["scaling"].setdefault(str(n), {})[key] = val
    _F.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")


def _time(fn, sync, repeats=2):
    fn()  # warmup / compile
    sync()
    best = 1e18
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        sync()
        best = min(best, time.perf_counter() - t0)
    return best


def _jsync(x):
    jax.block_until_ready(x)


def _tsync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _torch_field(t, y):  # y (N,6), batch-first
    return torch.stack(_eom_rhs(y, MWPotential2014, t, torch), dim=-1)


# series: (key, kind, device)
series = [("diffrax_cpu", "jax", _CPU), ("diffrax_gpu", "jax", _GPU),
          ("torchdiffeq_cpu", "torch", "cpu"), ("torchdiffeq_gpu", "torch", "cuda")]
if _GPU_ONLY:
    series = [s_ for s_ in series if s_[0].endswith("_gpu")]
alive = {k: True for k, *_ in series}

for n in N_EXT:
    icn = _batch(n)
    for key, kind, dev in series:
        if dev is None or not alive[key]:
            continue
        if kind == "torch" and dev == "cuda" and not torch.cuda.is_available():
            continue
        try:
            if kind == "jax":
                with jax.default_device(dev):
                    gd = jax.jit(jax.grad(lambda v: jax.vmap(
                        lambda ic: integrate_orbit(MWPotential2014, ic, ts_j)[-1, 0])(v).sum()))
                    icb = jnp.asarray(icn)
                    t = _time(lambda: gd(icb), lambda: _jsync(gd(icb)), repeats=2)
            else:
                ts_t = torch.as_tensor(_TS, device=dev)
                def _tg():
                    ics = torch.tensor(icn, device=dev, requires_grad=True)
                    y0 = _to_eom(torch, ics)
                    ys = odeint(_torch_field, y0, ts_t, method="dopri5", rtol=1e-8, atol=1e-8)
                    final = ys[-1]
                    (torch.sqrt(final[:, 0] ** 2 + final[:, 2] ** 2)).sum().backward()
                    g = ics.grad
                    del ics, y0, ys, final
                    return g
                t = _time(_tg, _tsync, repeats=1)
                if dev == "cuda":
                    torch.cuda.empty_cache()
            _set(n, key, t)
            print(f"N={n} {key}: {t * 1e3:.1f} ms", flush=True)
            if t > PER_RUN_BUDGET_S:
                print(f"  {key}: over budget ({t:.1f}s) -> stop series", flush=True)
                alive[key] = False
        except Exception as e:  # noqa: BLE001  -- OOM / step cap: stop this series
            print(f"N={n} {key}: STOP {type(e).__name__}: {str(e)[:70]}", flush=True)
            alive[key] = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if not any(alive.values()):
        break
print("done", flush=True)
