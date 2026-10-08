"""One kernel launch for a profiler (ncu/nvprof):
    python profile_run.py N [faithful|lazy] [variant] [refill 0|1]"""
import sys

import numpy as np

import cuda_orbits as co

N = int(float(sys.argv[1]))
lazy = (sys.argv[2] if len(sys.argv) > 2 else "faithful") == "lazy"
variant = sys.argv[3] if len(sys.argv) > 3 else ""
refill = bool(int(sys.argv[4])) if len(sys.argv) > 4 else False
lib = co.load(variant)
co.set_potential(lib, *co.load_mw2014())
ref = np.load("ref_N300.npz")
ics = np.tile(ref["ics"], (N // 300 + 1, 1))[:N]
r, st, tm = co.integrate(lib, ics, ref["ts"], lazy=lazy, stats=True, refill=refill)
print("nfev_total", int(st[:, 0].astype(np.int64).sum()), "kernel_ms", tm[1])
