"""Measure: do the IC-Jacobian dx(t)/dx0 mechanisms agree?

For one representative orbit (MWPotential2014, 3 Gyr), compute the full 6x6
state-transition matrix M(t)=dx(t)/dx0 four ways and compare to the C-STM as the
reference:
  * C-STM      -- galpy's compiled C variational integrator (the M it produces).
  * FD         -- finite-difference of the flow (central, eps=1e-6) through the
                  SAME C integrator -> agrees with C-STM to ~FD truncation.
  * jax AD     -- jax.jacrev through the in-backend diffrax solve (a DIFFERENT
                  integrator) -> agrees to ~orbit accuracy at matched tolerance.
  * torch AD   -- torch.autograd.functional.jacobian through torchdiffeq.

Writes data/precision.json: per-time relative Frobenius residual ||M_x - M_cstm||/
||M_cstm|| for x in {FD, jax-AD, torch-AD}, plus a couple of representative element
tracks. make.py plots it (byte-stable).
"""

import json
import pathlib

import numpy

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import torch

torch.set_default_dtype(torch.float64)

from galpy.backend._reference.inbackend_ode import integrate_orbit
from galpy.backend._reference.inbackend_stm import c_stm_forward
from galpy.orbit import Orbit
from galpy.potential import MWPotential2014

_IC = numpy.array([1.0, 0.1, 0.9, 0.05, 0.1, 0.2])  # R,vR,vT,z,vz,phi
_TS = numpy.linspace(0.0, 3.0, 30)
_RTOL = _ATOL = 1e-10


def _wrap(M):  # phi (index 5) rows/cols of M are fine; phi values wrap but dphi/dx0 don't
    return numpy.asarray(M)


# reference: C-STM M(t) (nt,6,6)
_, M_cstm = c_stm_forward(MWPotential2014, _IC, _TS, "dop853_c", _RTOL, _ATOL)
M_cstm = numpy.asarray(M_cstm)

# FD of the flow through the C integrator (central, per IC component)
eps = 1e-6
M_fd = numpy.zeros_like(M_cstm)
for i in range(6):
    cols = []
    for s in (+1, -1):
        ic = _IC.copy(); ic[i] += s * eps
        o = Orbit(list(ic)); o.integrate(_TS, MWPotential2014, method="dop853_c")
        cols.append(numpy.asarray(o.getOrbit()))  # (nt,6)
    M_fd[:, :, i] = (cols[0] - cols[1]) / (2 * eps)

# jax AD through the in-backend diffrax solve: jacrev -> (nt,6,6)
M_jax = numpy.asarray(
    jax.jacrev(lambda ic: integrate_orbit(MWPotential2014, ic, jnp.asarray(_TS)))(jnp.asarray(_IC))
)

# torch AD through torchdiffeq: jacobian -> (nt,6,6)
ts_t = torch.as_tensor(_TS)
M_torch = numpy.asarray(
    torch.autograd.functional.jacobian(
        lambda ic: integrate_orbit(MWPotential2014, ic, ts_t), torch.as_tensor(_IC)
    ).detach()
)


def _rel_frob(M):
    num = numpy.linalg.norm((M - M_cstm).reshape(len(_TS), -1), axis=1)
    den = numpy.linalg.norm(M_cstm.reshape(len(_TS), -1), axis=1) + 1e-30
    return (num / den).tolist()


out = {
    "meta": {"potential": "MWPotential2014", "tspan": 3.0, "nt": len(_TS),
             "rtol": _RTOL, "atol": _ATOL, "fd_eps": eps,
             "note": "M_cstm = C variational integrator; FD via same C integrator; "
                     "jax/torch AD via diffrax/torchdiffeq (different integrators)"},
    "ts": _TS.tolist(),
    "rel_frob": {"fd": _rel_frob(M_fd), "jax_ad": _rel_frob(M_jax), "torch_ad": _rel_frob(M_torch)},
    # a representative element: dR(t)/dvR0 = M[:,0,1], all four ways
    "elem_dR_dvR0": {
        "cstm": M_cstm[:, 0, 1].tolist(), "fd": M_fd[:, 0, 1].tolist(),
        "jax_ad": M_jax[:, 0, 1].tolist(), "torch_ad": M_torch[:, 0, 1].tolist(),
    },
}
dpath = pathlib.Path(__file__).resolve().parent.parent / "data"
dpath.mkdir(exist_ok=True)
(dpath / "precision.json").write_text(json.dumps(out, indent=2) + "\n")
print("max rel-Frobenius residual vs C-STM:  FD=%.2e  jax-AD=%.2e  torch-AD=%.2e"
      % (max(out["rel_frob"]["fd"]), max(out["rel_frob"]["jax_ad"]), max(out["rel_frob"]["torch_ad"])))
print("wrote", dpath / "precision.json")
