"""One symplectic kernel launch for a profiler (ncu):
    python profile_symp.py N scheme n [variant] [minb]"""
import sys

import numpy as np

import cuda_orbits as co

N = int(float(sys.argv[1]))
scheme, n = sys.argv[2], int(sys.argv[3])
variant = sys.argv[4] if len(sys.argv) > 4 else "_noinl"
minb = int(sys.argv[5]) if len(sys.argv) > 5 else 4
lib = co.load(variant)
co.set_potential(lib, *co.load_mw2014())
ref = np.load("ref_N300.npz")
ics = np.tile(ref["ics"], (N // 300 + 1, 1))[:N]
ts = ref["ts"]
r, st, tm = co.integrate_symp(lib, ics, ts, scheme, dt=(ts[1] - ts[0]) / n, stats=True, minb=minb)
print("nfev_total", int(st[:, 0].astype(np.int64).sum()), "kernel_ms", tm[1])
