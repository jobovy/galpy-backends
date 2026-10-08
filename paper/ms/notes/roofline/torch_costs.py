"""Kernel / byte census of galpy's EAGER torch orbit integration (torchdiffeq dopri5).

Eager torch launches one kernel per aten op and every op streams its operands
from and its result to device memory (no fusion), so the traffic of an eager
run is the sum over dispatched aten ops of (input + output tensor bytes). This
script records every aten op (TorchDispatchMode) of

  (a) one RHS evaluation, batched state (N, 6) -- what torchdiffeq calls;
  (b) a complete ``integrate_orbit`` solve (torchdiffeq dopri5, rtol=atol=1e-8,
      nt=16, one shared controller over the (N, 6) batch) at N=1024,

on the CPU (op counts and tensor sizes are device-independent), and reports per
RHS call and per orbit: aten ops (= CUDA kernel launches), bytes moved, and
elementwise "flops" (XLA convention: one per output element of an arithmetic /
compare / select op; transcendental ops counted separately).

Run (CPU only):
  taskset -c 24-35 nice -n 19 env CUDA_VISIBLE_DEVICES= \
    PYTHONPATH=/var/tmp/fbp python torch_costs.py
Writes torch_costs.json next to this script.
"""

import collections
import json
import os

import torch
from torch.utils._python_dispatch import TorchDispatchMode

torch.set_default_dtype(torch.float64)
torch.set_num_threads(4)
HERE = os.path.dirname(os.path.abspath(__file__))

from galpy.backend._reference import inbackend_ode  # noqa: E402
from galpy.backend._reference.inbackend_ode import _eom_rhs, _to_eom  # noqa: E402
from galpy.potential import MWPotential2014  # noqa: E402

_TRANSC = {"exp", "log", "log1p", "expm1", "pow", "sqrt", "rsqrt", "atan2", "sin", "cos",
           "tanh", "special_gammainc", "igamma", "igammac", "lgamma", "special_gammaincc"}
_FREE = {"view", "_unsafe_view", "expand", "slice", "select", "unsqueeze", "squeeze",
         "t", "transpose", "permute", "alias", "detach", "as_strided", "empty", "empty_like",
         "_to_copy", "lift_fresh", "unbind", "split", "split_with_sizes", "clone",
         "copy_", "stack", "cat", "zeros", "zeros_like", "ones_like", "full", "full_like",
         "scalar_tensor", "_local_scalar_dense", "new_empty", "new_zeros", "item",
         "is_nonzero", "index", "index_put_", "fill_", "_reshape_alias", "reshape"}
_VIEW = {"view", "_unsafe_view", "expand", "slice", "select", "unsqueeze", "squeeze", "t",
         "transpose", "permute", "alias", "detach", "as_strided", "unbind", "split",
         "split_with_sizes", "_reshape_alias", "lift_fresh", "_local_scalar_dense", "item",
         "is_nonzero", "empty", "empty_like", "new_empty"}


def _tensors(x):
    if isinstance(x, torch.Tensor):
        yield x
    elif isinstance(x, (list, tuple)):
        for y in x:
            yield from _tensors(y)
    elif isinstance(x, dict):
        for y in x.values():
            yield from _tensors(y)


class Count(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        self.ops = collections.Counter()
        self.kernels = 0
        self.bytes = 0
        self.flops = 0
        self.transc = collections.Counter()
        self.small_kernels = 0  # ops on tensors with < 64 elements (controller scalars)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        name = func.__name__.split(".")[0]
        self.ops[name] += 1
        if name in _VIEW:
            return out
        tin = list(_tensors(args)) + list(_tensors(kwargs or {}))
        tout = list(_tensors(out))
        b = sum(t.numel() * t.element_size() for t in tin + tout)
        nmax = max([t.numel() for t in tin + tout] or [0])
        self.kernels += 1
        if nmax < 64:
            self.small_kernels += 1
        self.bytes += b
        nout = sum(t.numel() for t in tout)
        if name in _TRANSC:
            self.transc[name] += nout
        elif name not in _FREE:
            self.flops += nout
        return out


res = {}
# (a) one RHS evaluation on an (N, 6) batch
N = 4096
y = _to_eom(torch, torch.tensor([1.0, 0.0, 1.0, 0.0125, 0.036, 0.0]).repeat(N, 1))
c = Count()
with c:
    torch.stack(_eom_rhs(y, MWPotential2014, 0.0, torch), dim=-1)
res["rhs"] = dict(N=N, kernels=c.kernels, ops=dict(c.ops), bytes_per_orbit=c.bytes / N,
                  flops_per_orbit=c.flops / N,
                  transc_per_orbit={k: v / N for k, v in c.transc.items()})
print("rhs", json.dumps(res["rhs"], default=float)[:800], flush=True)

# (b) a full solve, counting RHS calls by wrapping the shared EOM
N = 1024
ncall = {"n": 0}
_orig = inbackend_ode._eom_rhs


def counted(*a, **k):
    ncall["n"] += 1
    return _orig(*a, **k)


inbackend_ode._eom_rhs = counted
from galpy.backend._reference import integrate_orbit  # noqa: E402

g = torch.Generator().manual_seed(N)
ics = torch.tensor([1.0, 0.0, 1.0, 0.0125, 0.036, 0.0])[None] + torch.randn(
    N, 6, generator=g) * torch.tensor([0.05, 0.01, 0.0, 0.01, 0.01, 0.0])
ts = torch.linspace(0.0, 56.2, 16)
c = Count()
with c:
    integrate_orbit(MWPotential2014, ics, ts, rtol=1e-8, atol=1e-8)
inbackend_ode._eom_rhs = _orig
nf = ncall["n"]
res["solve"] = dict(N=N, nfev=nf, kernels=c.kernels, small_kernels=c.small_kernels,
                    kernels_per_rhs=c.kernels / nf, bytes_per_orbit=c.bytes / N,
                    bytes_per_orbit_per_rhs=c.bytes / N / nf, flops_per_orbit=c.flops / N,
                    flops_per_orbit_per_rhs=c.flops / N / nf,
                    transc_per_orbit_per_rhs={k: v / N / nf for k, v in c.transc.items()},
                    top_ops=dict(c.ops.most_common(25)))
print("solve", json.dumps(res["solve"], default=float)[:1500], flush=True)
with open(os.path.join(HERE, "torch_costs.json"), "w") as fh:
    json.dump(res, fh, indent=1, default=float)
