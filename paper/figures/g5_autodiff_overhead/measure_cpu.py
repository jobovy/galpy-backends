"""g5 measure, CPU variant (2026-10-28, JB's review): the cost of a derivative
on ONE CPU core, next to the two GPUs of ``make._measure``: the Plummer
potential at N points and its gradient w.r.t. the coordinates, under compiled
jax and under eager torch (one thread). Writes data/cpu/g5_autodiff_overhead.json.

    cd paper/figures/g5_autodiff_overhead
    CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 PYTHONPATH=<galpy worktree> \\
        taskset -c 6 python measure_cpu.py
"""

import json
import os
import sys

import numpy

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))
from paper.figures._harness import bench  # noqa: E402

_DATA = os.path.join(_HERE, "data", "cpu")
_JSON = os.path.join(_DATA, "g5_autodiff_overhead.json")


def _grid(N):
    return numpy.linspace(0.3, 3.0, N), 0.1 * numpy.ones(N)


def main():
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import torch

    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    import galpy
    from galpy.potential import PlummerPotential, evaluatePotentials

    assert all(d.platform == "cpu" for d in jax.devices()), jax.devices()
    p = PlummerPotential(normalize=1.0)
    Ns = [int(10**k) for k in range(2, 8)]
    series = {}

    def med(fn, **kw):
        return bench.time_call(fn, **kw)["median"]

    val, grd = [], []
    for N in Ns:
        Rn, zn = _grid(N)
        Rj, zj = jnp.asarray(Rn), jnp.asarray(zn)
        vfun = jax.jit(lambda R, z: evaluatePotentials(p, R, z))
        gfun = jax.jit(jax.grad(lambda R, z: jnp.sum(evaluatePotentials(p, R, z))))
        vfun(Rj, zj).block_until_ready()  # compile
        gfun(Rj, zj).block_until_ready()
        val.append(med(lambda: vfun(Rj, zj), warmup=3, repeats=9, sync="jax"))
        grd.append(med(lambda: gfun(Rj, zj), warmup=3, repeats=9, sync="jax"))
        print(f"jax N={N}: value {val[-1]*1e3:.3f} ms grad {grd[-1]*1e3:.3f} ms", flush=True)
    series["jax value"], series["jax grad"] = val, grd
    val, grd = [], []
    for N in Ns:
        Rn, zn = _grid(N)
        Rt, zt = torch.as_tensor(Rn), torch.as_tensor(zn)

        def grad_call():
            R = Rt.detach().requires_grad_(True)
            evaluatePotentials(p, R, zt).sum().backward()
            return R.grad

        val.append(med(lambda: evaluatePotentials(p, Rt, zt), warmup=3, repeats=9, sync="torch"))
        grd.append(med(grad_call, warmup=3, repeats=9, sync="torch"))
        print(f"torch N={N}: value {val[-1]*1e3:.3f} ms grad {grd[-1]*1e3:.3f} ms", flush=True)
    series["torch value"], series["torch grad"] = val, grd
    env = {"python": sys.version.split()[0], "device": "cpu",
           "cpu_affinity": len(os.sched_getaffinity(0)), "torch_threads": torch.get_num_threads(),
           "galpy_file": galpy.__file__}
    for m in ("numpy", "jax", "torch", "galpy"):
        env[m] = __import__(m).__version__
    os.makedirs(_DATA, exist_ok=True)
    with open(_JSON, "w") as fh:
        json.dump({"Ns": Ns, "series": series, "env": env}, fh, indent=2)
    print("->", _JSON)


if __name__ == "__main__":
    main()
