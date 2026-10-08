"""Small run for nvprof: N orbits, lazy dense, default build."""
import sys
import numpy as np
import cuda_orbits as co
N = int(sys.argv[1]); lazy = sys.argv[2] == "lazy" if len(sys.argv) > 2 else True
variant = sys.argv[3] if len(sys.argv) > 3 else ""
lib = co.load(variant); co.set_potential(lib, *co.load_mw2014())
ref = np.load("ref_N300.npz")
ics = np.tile(ref["ics"], (N // 300 + 1, 1))[:N]
r, st, tm = co.integrate(lib, ics, ref["ts"], lazy=lazy, stats=True)
print("nfev_total", int(st[:, 0].astype(np.int64).sum()), "kernel_ms", tm[1])
