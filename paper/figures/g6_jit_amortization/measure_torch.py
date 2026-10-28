"""g6 measure, torch variant (2026-10-28, JB's review: "why no torch.compile?"):
the same compile-once / run-many measurement as ``make._measure`` for the
torch backend on the GPU: per-call wall times of evaluateRforces for a Plummer
potential at N=1e6 points under torch.compile (call 0 compiles) and the eager
per-call cost. Writes data/torch/g6_jit_amortization.json on the dev box
(TITAN Xp) and data/h100/torch/g6_jit_amortization.json on the H100 (pass
``--out``). Run with the GPU otherwise idle:

    cd paper/figures/g6_jit_amortization
    PYTHONPATH=<galpy worktree> python measure_torch.py [--out data/h100/torch/g6_jit_amortization.json]
"""

import argparse
import json
import os
import sys
import time

import numpy

_HERE = os.path.dirname(os.path.abspath(__file__))
_NCALLS = 15


def _grid(N):
    return numpy.linspace(0.3, 3.0, N), 0.1 * numpy.ones(N)


def main(out):
    import torch

    torch.set_default_dtype(torch.float64)
    import galpy
    from galpy.potential import PlummerPotential, evaluateRforces

    assert torch.cuda.is_available(), "this measures the GPU; the CPU variant is measure_cpu.py"
    dev = torch.device("cuda")
    p = PlummerPotential(normalize=1.0)
    Ns = [int(1e4), int(1e6)]
    per_call, eager, compile_error = {}, {}, {}
    for N in Ns:
        Rn, zn = _grid(N)
        Rt, zt = torch.as_tensor(Rn, device=dev), torch.as_tensor(zn, device=dev)
        # a fresh compile per N: torch.compile's cache is per process and per
        # graph, so reset it, as a fresh jax.jit does
        torch._dynamo.reset()
        f = torch.compile(lambda R, z: evaluateRforces(p, R, z))
        ts = []
        try:
            for _ in range(_NCALLS):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                out_ = f(Rt, zt)
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
            per_call[str(N)] = ts
        except Exception as e:  # e.g. the TITAN Xp (sm_61) is too old for Triton
            compile_error[str(N)] = f"{type(e).__name__}: {str(e)[:300]}"
            print(f"N={N}: torch.compile failed: {compile_error[str(N)][:120]}", flush=True)

        def g():
            evaluateRforces(p, Rt, zt)
            torch.cuda.synchronize()

        for _ in range(3):
            g()
        et = []
        for _ in range(15):
            t0 = time.perf_counter()
            g()
            et.append(time.perf_counter() - t0)
        eager[str(N)] = float(numpy.median(et))
        if ts:
            print(f"N={N}: compile {ts[0]*1e3:.1f} ms, compiled steady {numpy.median(ts[1:])*1e3:.3f} ms, "
                  f"eager {eager[str(N)]*1e3:.3f} ms", flush=True)
        else:
            print(f"N={N}: eager {eager[str(N)]*1e3:.3f} ms", flush=True)
    env = {"python": sys.version.split()[0], "gpu": torch.cuda.get_device_name(0),
           "galpy_file": galpy.__file__}
    for m in ("numpy", "torch", "galpy"):
        env[m] = __import__(m).__version__
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"Ns": Ns, "per_call": per_call, "eager": eager, "compile_error": compile_error,
                   "env": env}, fh, indent=2)
    print("->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(_HERE, "data", "torch", "g6_jit_amortization.json"))
    main(ap.parse_args().out)
