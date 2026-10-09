"""Measure: cost of the IC derivative d(orbit)/d(x0), C-STM vs in-backend AD.

Mechanisms for the initial-condition gradient (grad of the summed final R):
  * C-STM        -- galpy's compiled C variational integrator; the state-
                    transition matrix M=dx(t)/dx0 is a byproduct of the forward
                    solve (galpy.backend._jax.orbit_stm). The forward runs in C
                    (on the CPU) regardless of the jax device.
  * diffrax (AD) -- reverse-mode autodiff through the in-backend diffrax solve
                    (jax); batches over orbits with jax.vmap. CPU and GPU.
  * torchdiffeq  -- reverse-mode autodiff through the torchdiffeq solve. CPU/GPU.

Writes data/icderiv.json incrementally (machine-stamped; NOT byte-checked).
Run:  PYTHONPATH=<#972-worktree> python measure/measure_icderiv.py
"""

import json
import os
import pathlib

import numpy

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import torch

torch.set_default_dtype(torch.float64)

from paper.figures._harness.bench import time_call

from galpy.backend._jax.orbit_stm import integrate as cstm_jax
from galpy.backend._reference.inbackend_ode import integrate_orbit
from galpy.potential import MWPotential2014

_CPU = jax.devices("cpu")[0]
_GPU_ONLY = os.environ.get("G15_GPU_ONLY") == "1"  # H100 runs: skip the CPU series
_GPU = next((d for d in jax.devices() if d.platform == "gpu"), None)
_HAS_GPU = _GPU is not None
_IC = numpy.array([1.0, 0.1, 0.9, 0.05, 0.1, 0.2])
_TS = numpy.linspace(0.0, 10.0, 100)
ts_j = jnp.asarray(_TS)
_DPATH = pathlib.Path(__file__).resolve().parent.parent / "data"
_DPATH.mkdir(exist_ok=True)

out = {"meta": {"nt": len(_TS), "potential": "MWPotential2014", "has_gpu": _HAS_GPU,
                "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)},
       "single": {}, "scaling": {}}


def _save():
    (_DPATH / "icderiv.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")


def _safe(fn, **kw):
    try:
        return time_call(fn, **kw)["median"]
    except Exception as e:  # noqa: BLE001 -- record a skip, keep going
        print("  skip:", repr(e)[:90], flush=True)
        return None


def _batch(n):
    return numpy.stack([_IC * (1 + 0.001 * k) for k in range(n)])


# ----------------------------- single-orbit forward + IC-gradient -------------
icj = jnp.asarray(_IC)
for dname, dev in (([] if _GPU_ONLY else [("cpu", _CPU)]) + ([("gpu", _GPU)] if _HAS_GPU else [])):
    with jax.default_device(dev):
        # C-STM only meaningful on CPU (the variational forward runs in C via a
        # pure_callback regardless of jax device); record it on cpu.
        if dname == "cpu":
            cg = jax.jit(jax.grad(lambda ic: cstm_jax(MWPotential2014, ic, ts_j, method="dop853_c")[-1, 0]))
            out["single"].setdefault("cstm", {})["cpu"] = {
                "grad": _safe(lambda: jax.block_until_ready(cg(icj)), sync="jax")}
        dg = jax.jit(jax.grad(lambda ic: integrate_orbit(MWPotential2014, ic, ts_j)[-1, 0]))
        out["single"].setdefault("diffrax", {})[dname] = {
            "grad": _safe(lambda: jax.block_until_ready(dg(icj)), sync="jax")}
    _save()
    print(f"single jax {dname} done", flush=True)

for dname, tdev in (([] if _GPU_ONLY else [("cpu", "cpu")]) + ([("gpu", "cuda")] if torch.cuda.is_available() else [])):
    ict = torch.tensor(_IC, requires_grad=True, device=tdev)
    ts_t = torch.as_tensor(_TS, device=tdev)
    def _tg(ict=ict, ts_t=ts_t):
        if ict.grad is not None:
            ict.grad = None
        integrate_orbit(MWPotential2014, ict, ts_t)[-1, 0].backward()
        return ict.grad
    out["single"].setdefault("torchdiffeq", {})[dname] = {
        "grad": _safe(_tg, repeats=3, sync="torch")}
    _save()
    print(f"single torch {dname} done", flush=True)

# ----------------------------- batch scaling: grad of summed final R vs N -----
for dname, dev in (([] if _GPU_ONLY else [("cpu", _CPU)]) + ([("gpu", _GPU)] if _HAS_GPU else [])):
    for n in (1, 2, 4, 8, 16, 32, 64):
        rec = out["scaling"].setdefault(str(n), {})
        # C-STM (CPU only; C-bound forward)
        if dname == "cpu":
            with jax.default_device(_CPU):
                g = jax.jit(jax.grad(lambda v: cstm_jax(MWPotential2014, v, ts_j, method="dop853_c")[:, -1, 0].sum()))
                icb = jnp.asarray(_batch(n))
                rec["cstm_cpu"] = _safe(lambda: jax.block_until_ready(g(icb)), sync="jax", repeats=5)
        # diffrax (jax.vmap) on this device
        with jax.default_device(dev):
            gd = jax.jit(jax.grad(lambda v: jax.vmap(lambda ic: integrate_orbit(MWPotential2014, ic, ts_j)[-1, 0])(v).sum()))
            rec[f"diffrax_{dname}"] = _safe(lambda: jax.block_until_ready(gd(jnp.asarray(_batch(n)))), sync="jax", repeats=3)
        _save()
        print(f"scaling {dname} N={n}: {rec}", flush=True)

# torchdiffeq scaling: loop over orbits (no clean batch in the single-orbit field),
# GPU only and small N (slow on CPU); shows it does not auto-vectorise.
if torch.cuda.is_available():
    ts_t = torch.as_tensor(_TS, device="cuda")
    for n in (1, 4, 16):
        ics = torch.tensor(_batch(n), device="cuda", requires_grad=True)
        def _loop(ics=ics):
            if ics.grad is not None:
                ics.grad = None
            s = sum(integrate_orbit(MWPotential2014, ics[k], ts_t)[-1, 0] for k in range(ics.shape[0]))
            s.backward()
            return ics.grad
        out["scaling"].setdefault(str(n), {})["torchdiffeq_gpu"] = _safe(_loop, repeats=2, sync="torch")
        _save()
        print(f"torch scaling gpu N={n} done", flush=True)

_save()
print("wrote", _DPATH / "icderiv.json", flush=True)
