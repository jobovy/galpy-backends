"""Measure (torch large-N): push the batched torchdiffeq IC-gradient past the
nt=100 backward-graph OOM by using nt=16 sparse saves, to capture the eager
overhead->compute-bound turnover (the torch lines were flat-then-OOM in g15 v4).

The gradient is step-bound (the backward graph tracks the adaptive Dopri5 steps,
not the save count), so nt=16 vs nt=100 is ~immaterial to wall-time but frees the
memory that was OOMing at N=65536 -- letting the line reach the compute-bound rise
(same lowN-dense / hiN-sparse merge convention as g12). Appends large-N points to
the existing torchdiffeq_{cpu,gpu} scaling series in data/icderiv.json.
Run:  PYTHONPATH=<#972-worktree> python measure/measure_torch_hiN.py
"""

import json
import pathlib

import numpy
import torch

torch.set_default_dtype(torch.float64)
from torchdiffeq import odeint

from paper.figures._harness.bench import time_call

from galpy.backend._reference.inbackend_ode import _eom_rhs, _to_eom
from galpy.potential import MWPotential2014

_IC = numpy.array([1.0, 0.1, 0.9, 0.05, 0.1, 0.2])
_TS = numpy.linspace(0.0, 10.0, 16)  # hiN sparse saves (gradient is step-bound)
_F = pathlib.Path(__file__).resolve().parent.parent / "data" / "icderiv.json"
out = json.loads(_F.read_text())


def _field(t, y):  # y (6,N); _eom_rhs componentwise -> vectorises over N
    return torch.stack(_eom_rhs(y, MWPotential2014, t, torch))


def _save():
    _F.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")


for dname, dev in ([("gpu", "cuda")] if torch.cuda.is_available() else []):  # GPU only
    ts_t = torch.as_tensor(_TS, device=dev)
    alive = True
    for n in (16384, 65536, 131072, 262144, 524288):
        if not alive:
            break
        icn = numpy.stack([_IC * (1 + 0.001 * (k % 97)) for k in range(n)])
        def _g(icn=icn, dev=dev):
            ics = torch.tensor(icn, device=dev, requires_grad=True)
            y0 = _to_eom(torch, ics.T)
            ys = odeint(_field, y0, ts_t, method="dopri5", rtol=1e-8, atol=1e-8)
            (torch.sqrt(ys[-1][0] ** 2 + ys[-1][2] ** 2)).sum().backward()
            g = ics.grad
            del ics, y0, ys
            return g
        try:
            t = time_call(_g, repeats=1, warmup=1, sync="torch")["median"]
            out["scaling"].setdefault(str(n), {})[f"torchdiffeq_{dname}"] = t
            _save()
            print(f"torch {dname} N={n}: {t * 1e3:.1f} ms", flush=True)
            if dev == "cuda":
                torch.cuda.empty_cache()
            if t > 180.0:
                print(f"  torch {dname}: over budget -> stop", flush=True)
                alive = False
        except Exception as e:  # noqa: BLE001
            print(f"torch {dname} N={n}: STOP {type(e).__name__}: {str(e)[:70]}", flush=True)
            alive = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
print("done", flush=True)
