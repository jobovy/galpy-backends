"""Measure (scaling, large N): IC-gradient wall-clock vs number of orbits N.

Updates the "scaling" block of data/icderiv.json (preserves "single"). Pushes to
large batches and includes a *batched* torchdiffeq path (the g12 trick: carry the
state as (6,N) through one odeint with a shared adaptive controller; _eom_rhs is
componentwise so it vectorises over N), then backprops for the batched gradient.

Mechanisms (gradient of the summed final R w.r.t. all ICs):
  cstm_cpu        C-STM, Python loop over orbits (linear; capped at moderate N)
  diffrax_{cpu,gpu}   jax diffrax, jax.vmap over orbits (per-orbit controller)
  torchdiffeq_{cpu,gpu}  torchdiffeq, batched (6,N) shared-controller solve

Incremental save + per-cell OOM/error guard so partial large-N data survives.
Run:  PYTHONPATH=<#972-worktree> python measure/measure_scaling.py
"""

import json
import pathlib

import numpy

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import torch

torch.set_default_dtype(torch.float64)
from torchdiffeq import odeint

from paper.figures._harness.bench import time_call

from galpy.backend._jax.orbit_stm import integrate as cstm_jax
from galpy.backend._reference.inbackend_ode import _eom_rhs, _to_eom, integrate_orbit
from galpy.potential import MWPotential2014

_CPU = jax.devices("cpu")[0]
_GPU = next((d for d in jax.devices() if d.platform == "gpu"), None)
_IC = numpy.array([1.0, 0.1, 0.9, 0.05, 0.1, 0.2])
_TS = numpy.linspace(0.0, 10.0, 100)
ts_j = jnp.asarray(_TS)
_DPATH = pathlib.Path(__file__).resolve().parent.parent / "data"
_F = _DPATH / "icderiv.json"

out = json.loads(_F.read_text())
out["scaling"] = {}

NS = [1, 4, 16, 64, 256, 1024, 4096]
CSTM_MAX_N = 256  # C-STM loops over orbits (linear) -> cap so it stays tractable


def _batch(n):
    return numpy.stack([_IC * (1 + 0.001 * (k % 97)) for k in range(n)])


def _safe(fn, **kw):
    try:
        return time_call(fn, **kw)["median"]
    except Exception as e:  # noqa: BLE001
        print("  skip:", repr(e)[:80], flush=True)
        return None


def _torch_field(t, y):  # y: (6,N); _eom_rhs componentwise -> vectorises over N
    return torch.stack(_eom_rhs(y, MWPotential2014, t, torch))


def _set(n, key, val):
    out["scaling"].setdefault(str(n), {})[key] = val
    _F.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")


for n in NS:
    icn = _batch(n)
    # --- C-STM (CPU, loop over orbits) ---
    if n <= CSTM_MAX_N:
        with jax.default_device(_CPU):
            g = jax.jit(jax.grad(lambda v: cstm_jax(MWPotential2014, v, ts_j, method="dop853_c")[:, -1, 0].sum()))
            icb = jnp.asarray(icn)
            _set(n, "cstm_cpu", _safe(lambda: jax.block_until_ready(g(icb)), sync="jax", repeats=3))
    # --- diffrax (jax.vmap) on CPU and GPU ---
    for dname, dev in ([("cpu", _CPU)] + ([("gpu", _GPU)] if _GPU else [])):
        with jax.default_device(dev):
            gd = jax.jit(jax.grad(lambda v: jax.vmap(lambda ic: integrate_orbit(MWPotential2014, ic, ts_j)[-1, 0])(v).sum()))
            _set(n, f"diffrax_{dname}", _safe(lambda: jax.block_until_ready(gd(jnp.asarray(icn))), sync="jax", repeats=3))
    # --- torchdiffeq batched (6,N), CPU and GPU ---
    for dname, tdev in ([("cpu", "cpu")] + ([("gpu", "cuda")] if torch.cuda.is_available() else [])):
        ts_t = torch.as_tensor(_TS, device=tdev)
        def _tg(icn=icn, tdev=tdev, ts_t=ts_t):
            ics = torch.tensor(icn, device=tdev, requires_grad=True)  # (N,6)
            y0 = _to_eom(torch, ics.T)                                # (6,N) rect EOM
            ys = odeint(_torch_field, y0, ts_t, method="dopri5", rtol=1e-8, atol=1e-8)
            final = ys[-1]                                            # (6,N): x,vx,y,vy,z,vz
            R = torch.sqrt(final[0] ** 2 + final[2] ** 2)             # (N,) final R
            R.sum().backward()
            return ics.grad
        # torch CPU is very slow at large N; cap CPU at 256, allow GPU to push out
        if dname == "cpu" and n > 256:
            continue
        _set(n, f"torchdiffeq_{dname}", _safe(_tg, repeats=(3 if n <= 64 else 1), sync="torch"))
    print(f"N={n}: {out['scaling'][str(n)]}", flush=True)

print("wrote", _F, flush=True)
