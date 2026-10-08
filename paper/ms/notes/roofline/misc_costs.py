"""FLOP / byte census of the g2 (Plummer F_R), gD3 (spherical-DF sampling) and gV2
(VerticalInverse xvFreqs) kernels, exactly as their make.py time them under
``jax.jit``, compiled on the CPU backend at a moderate N and scaled per point.

Every while-loop in these modules is listed with its trip count: XLA's
known_trip_count where the loop is a fixed-length fori_loop (else the trip count
passed below, with its provenance). The eager-torch g2 call is censused per aten
op (TorchDispatchMode): one kernel per op, each streaming its operands and result.

Run (CPU only):
  taskset -c 24-35 nice -n 19 env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu \
    PYTHONPATH=/var/tmp/fbp:/home/bovy/Repos/galpy-jax python misc_costs.py
Writes misc_costs.json next to this script.
"""

import json
import os
import sys
import warnings

import numpy

warnings.filterwarnings("ignore")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hlo_cost as hc  # noqa: E402

out = {}


def census(name, f, args, N, trips=None):
    comp = jax.jit(f).lower(*args).compile()
    ca = comp.cost_analysis()
    txt = comp.as_text()
    loops = []

    def tr(info):
        n = info["known"]
        if n is None and trips is not None:
            n = trips(info)
        loops.append(dict(name=info["name"], depth=len(info["path"]), nleaves=info["nleaves"],
                          lane=info["lane"], known=info["known"], used=n if n is not None else 1))
        return n if n is not None else 1

    acc = hc.Census(txt, trips=tr).total()
    s = hc.summary(acc, per=N)
    s["bytes_ideal"] = hc.Census(txt, trips=lambda i: [l for l in loops if l["name"] == i["name"]][0]["used"],
                                 ideal=True).total()["bytes"] / N
    s["xla_once"] = {k: ca.get(k, 0.0) / N for k in ("flops", "transcendentals", "bytes accessed")}
    s["loops"] = loops
    s["N"] = N
    out[name] = s
    print(name, json.dumps({k: v for k, v in s.items() if k != "loops"}, default=float), flush=True)
    print("   loops:", loops, flush=True)
    return s


# ---------------------------------------------------------------- g2 ----------
from galpy.potential import PlummerPotential, evaluateRforces  # noqa: E402

p = PlummerPotential(normalize=1.0)
N = 1_000_000
R = jnp.asarray(numpy.linspace(0.3, 3.0, N))
z = jnp.asarray(0.1 * numpy.ones(N))
census("g2_plummer_FR_jax", lambda R, z: evaluateRforces(p, R, z), (R, z), N)

# eager torch: per-op census (also the op sequence numpy runs: same galpy code)
import torch  # noqa: E402
from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402

torch.set_default_dtype(torch.float64)


class Ops(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.ops, self.bytes = [], 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        r = func(*args, **(kwargs or {}))
        nm = func.__name__.split(".")[0]
        if nm in ("view", "detach", "lift_fresh", "alias", "_local_scalar_dense"):
            return r
        ts = [a for a in list(args) + [r] if isinstance(a, torch.Tensor)]
        self.ops.append(nm)
        self.bytes += sum(t.numel() * t.element_size() for t in ts)
        return r


Rt, zt = torch.as_tensor(numpy.asarray(R)), torch.as_tensor(numpy.asarray(z))
evaluateRforces(p, Rt, zt)
m = Ops()
with m:
    evaluateRforces(p, Rt, zt)
out["g2_plummer_FR_torch_eager"] = dict(kernels=len(m.ops), ops=m.ops, bytes_per_point=m.bytes / N)
print("g2 torch", out["g2_plummer_FR_torch_eager"], flush=True)

# ---------------------------------------------------------------- gD3 ---------
from galpy.backend import random as grandom  # noqa: E402
from galpy.df import eddingtondf, isotropicHernquistdf, kingdf, osipkovmerrittdf  # noqa: E402
from galpy.potential import HernquistPotential, NFWPotential  # noqa: E402

dfs = {
    "isotropicHernquistdf": lambda: isotropicHernquistdf(pot=HernquistPotential(amp=2.0, a=1.0)),
    "eddingtondf (NFW)": lambda: eddingtondf(pot=NFWPotential(amp=1.0, a=1.0), rmin=0.0),
    "osipkovmerrittdf": lambda: osipkovmerrittdf(pot=HernquistPotential(amp=2.0, a=1.0), ra=1.5,
                                                 rmin=0.0),
    "kingdf": lambda: kingdf(W0=5.0, M=1.0, rt=10.0),
}
N = 100_000
key = grandom.key(1, "jax")
for name, mk in dfs.items():
    df = mk()
    numpy.random.seed(1)
    df.sample(n=10, return_orbit=False)  # builds the sampling tables (as make.py)
    census(f"gD3_{name}", lambda k, df=df: df.sample(n=N, key=k, return_orbit=False), (key,), N)

# ---------------------------------------------------------------- gV2 ---------
from galpy.actionAngle import actionAngleVerticalInverse  # noqa: E402
from galpy.potential import IsothermalDiskPotential  # noqa: E402

pot = IsothermalDiskPotential(amp=jnp.asarray(1.0), sigma=jnp.asarray(0.5))
aA = actionAngleVerticalInverse(pot=pot, Es=numpy.linspace(0.05, 2.0, 31), setup_interp=True)
N = 100_000
ang = jnp.asarray(numpy.linspace(0.0, 2.0 * numpy.pi, N, endpoint=False))
census("gV2_xvFreqs_one_torus", lambda j, a: aA.xvFreqs(j, a), (jnp.asarray(0.5), ang), N)
a16 = jnp.asarray(numpy.linspace(0.0, 2.0 * numpy.pi, 16, endpoint=False))
M = N // 16
Js = jnp.asarray(numpy.linspace(0.05, 0.6, M))
census("gV2_xvFreqs_vmap_tori", jax.vmap(lambda j: aA.xvFreqs(j, a16)), (Js,), M * 16)

with open(os.path.join(HERE, "misc_costs.json"), "w") as fh:
    json.dump(out, fh, indent=1, default=float)
