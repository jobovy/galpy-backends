"""g6 measure, CPU variant (2026-10-28, JB's review): the same compile-once /
run-many measurement as ``make._measure`` (jax jit per-call wall times of
evaluateRforces for a Plummer potential at N=1e6 points, call 0 compiling, and
the eager per-call cost) on ONE CPU core, so the figure can show what
compilation buys on the CPU next to the two GPUs. Writes
data/cpu/g6_jit_amortization.json. Run pinned, with jax on the CPU:

    cd paper/figures/g6_jit_amortization
    CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 PYTHONPATH=<galpy worktree> \\
        taskset -c 4 python measure_cpu.py
"""

import json
import os
import sys
import time

import numpy

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cpu")
_JSON = os.path.join(_DATA, "g6_jit_amortization.json")
_NCALLS = 15


def _grid(N):
    return numpy.linspace(0.3, 3.0, N), 0.1 * numpy.ones(N)


def main():
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    import galpy
    from galpy.potential import PlummerPotential, evaluateRforces

    assert all(d.platform == "cpu" for d in jax.devices()), jax.devices()
    p = PlummerPotential(normalize=1.0)
    Ns = [int(1e4), int(1e6)]
    per_call, eager = {}, {}
    for N in Ns:
        Rn, zn = _grid(N)
        Rj, zj = jnp.asarray(Rn), jnp.asarray(zn)
        f = jax.jit(lambda R, z: evaluateRforces(p, R, z))
        ts = []
        for _ in range(_NCALLS):
            t0 = time.perf_counter()
            out = f(Rj, zj)
            jax.block_until_ready(out)
            ts.append(time.perf_counter() - t0)
        per_call[str(N)] = ts
        g = lambda: jax.block_until_ready(evaluateRforces(p, Rj, zj))
        for _ in range(3):
            g()
        et = []
        for _ in range(15):
            t0 = time.perf_counter()
            g()
            et.append(time.perf_counter() - t0)
        eager[str(N)] = float(numpy.median(et))
        print(f"N={N}: compile {ts[0]*1e3:.1f} ms, jit steady {numpy.median(ts[1:])*1e3:.3f} ms, "
              f"eager {eager[str(N)]*1e3:.3f} ms", flush=True)
    env = {"python": sys.version.split()[0], "device": "cpu",
           "cpu_affinity": len(os.sched_getaffinity(0)), "galpy_file": galpy.__file__}
    for m in ("numpy", "jax", "galpy"):
        env[m] = __import__(m).__version__
    os.makedirs(_DATA, exist_ok=True)
    with open(_JSON, "w") as fh:
        json.dump({"Ns": Ns, "per_call": per_call, "eager": eager, "env": env}, fh, indent=2)
    print("->", _JSON)


if __name__ == "__main__":
    main()
