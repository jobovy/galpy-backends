"""H100 diagnostic: actionAngleIsochroneApprox's angle fit is singular on CUDA
(torch 2.14 + cu130) in test_backend_streamdf.py::test_streamdf_progenitor_grad_with_c_aA_is_loud,
but not on the CPU (torch 2.14) or on a TITAN Xp (torch 2.12).

Run from the galpy checkout:  PYTHONPATH=. python h100_isoapprox_diag.py [cpu] [cuda]
For each device it builds the test's streamdf exactly as the test does and, at
every angle fit, prints the point, the diagonal of A^T A (a zero entry is an
all-zero column of the design matrix, i.e. sin(n . angle) == 0 at every time)
and the range of the orbit angles that enter the fit."""

import sys

import numpy
import torch

sys.path.insert(0, "tests")
import importlib
from galpy.actionAngle import actionAngleIsochroneApprox
from galpy.backend import use
from galpy.df import streamdf
from galpy.orbit import Orbit
from galpy.potential import LogarithmicHaloPotential
from test_backend_streamdf import _STREAM_IC, _tdisrupt

AIA = importlib.import_module("galpy.actionAngle.actionAngleIsochroneApprox")
_orig = AIA.actionAngleIsochroneApprox._actionsFreqsAngles_backend
from galpy.backend._namespaces import namespace_for_name
_XP = namespace_for_name("torch")
_inv = _XP.linalg.inv


def _spy(self, *args, **kwargs):
    print("  fit at", [round(float(torch.as_tensor(a).detach().cpu().reshape(-1)[0]), 6)
                       for a in args[:6] if hasattr(a, "shape") or isinstance(a, float)])
    def inv(M):
        d = torch.diagonal(M, dim1=-2, dim2=-1).detach().cpu().numpy()
        print("    AtA diag min/argmin:", float(d.min()), int(d.argmin()), "finite:", bool(torch.isfinite(M).all()))
        return _inv(M)
    _sin = _XP.sin
    def sin(x):
        xc = x.detach().cpu()
        print("    sin args: shape", tuple(xc.shape), "min/max", float(xc.min()), float(xc.max()),
              "exact zeros", int((xc == 0).sum()), "nan", int(torch.isnan(xc).sum()))
        return _sin(x)
    _XP.linalg.inv, _XP.sin = inv, sin
    try:
        return _orig(self, *args, **kwargs)
    finally:
        _XP.linalg.inv, _XP.sin = _inv, _sin


AIA.actionAngleIsochroneApprox._actionsFreqsAngles_backend = _spy
for dev in sys.argv[1:] or ["cpu", "cuda"]:
    print(f"== {dev}")
    torch.set_default_device(dev)
    lp = LogarithmicHaloPotential(normalize=1.0, q=0.9)
    aA = actionAngleIsochroneApprox(pot=lp, b=0.8, tintJ=5.0, ntintJ=250)
    pR = torch.tensor(_STREAM_IC[0], dtype=torch.float64, requires_grad=True)
    prog = Orbit(torch.stack([pR] + [torch.tensor(c, dtype=torch.float64) for c in _STREAM_IC[1:]]))
    try:
        with use("torch", force=True):
            sdf = streamdf(0.365 / 220.0, progenitor=prog, pot=lp, aA=aA, leading=True,
                           nTrackChunks=4, tdisrupt=_tdisrupt(), nTrackIterations=0)
        print("  constructed")
    except Exception as e:  # noqa: BLE001
        print("  FAILED:", type(e).__name__, str(e)[:150])
print("torch", torch.__version__, "cuda", torch.version.cuda,
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
