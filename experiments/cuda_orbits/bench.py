"""Timing of the one-thread-per-orbit CUDA dop853 (galpy dop853_c port) in
MWPotential2014, hiN convention of paper/figures/_shared/orbit_speed (nt=16 saves
over 2 Gyr, rtol=atol=1e-8, ICs from make_ics with rng=default_rng(N)).

wall_s = alloc + H2D(ICs) + kernel + D2H(all N x 16 x 6 outputs) + free, best of
REPEATS, measured around the ctypes call; one-time costs (CUDA context, module
load) are excluded by a warm-up call.

    python bench.py [--Ns 1e3,1e4,...] [--variant _nofma] [--out data/cuda_gpu_hiN.json]
"""
import argparse
import json
import os
import socket
import time

import numpy as np

import cuda_orbits as co

p = argparse.ArgumentParser()
p.add_argument("--Ns", default="1e3,1e4,1e5,1e6,3e6")
p.add_argument("--variant", default="")
p.add_argument("--modes", default="lazy,faithful")
p.add_argument("--block", type=int, default=128)
p.add_argument("--repeats", type=int, default=3)
p.add_argument("--out", default=None)
p.add_argument("--tol", type=float, default=1e-8, help="rtol=atol (1e-8 = paper convention; float32 needs ~1e-5)")
args = p.parse_args()

lib = co.load(args.variant)
co.set_potential(lib, *co.load_mw2014())
dev = co.device_name(lib)
nt = 16
ts = np.linspace(0.0, 56.2, nt)
ic0 = np.array([1.0, 0.0, 1.0, 0.0125, 0.036, 0.0])


def make_ics(N, rng):  # verbatim from galpy_c_measure.py
    base = np.tile(ic0, (N, 1))
    base[:, 0] += rng.normal(0, 0.05, N)
    base[:, 1] += rng.normal(0, 0.01, N)
    base[:, 3] += rng.normal(0, 0.01, N)
    base[:, 4] += rng.normal(0, 0.01, N)
    return base


co.integrate(lib, make_ics(1000, np.random.default_rng(0)), ts)  # warm-up
out = {"backend": "cuda (galpy dop853_c port, one thread per orbit)",
       "device": dev, "variant": args.variant or "fmad", "block": args.block,
       "nt": nt, "tspan_Gyr": 2.0, "rtol": args.tol, "atol": args.tol,
       "potential": "MWPotential2014", "host": socket.gethostname(), "results": []}
for mode in args.modes.split(","):
    lazy = mode == "lazy"
    for N in [int(float(s)) for s in args.Ns.split(",")]:
        ics = make_ics(N, np.random.default_rng(N))
        _, st, _ = co.integrate(lib, ics, ts, rtol=args.tol, atol=args.tol, lazy=lazy, block=args.block, stats=True)
        best, best_tm = 1e18, None
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            res, _, tm = co.integrate(lib, ics, ts, rtol=args.tol, atol=args.tol, lazy=lazy, block=args.block)
            w = time.perf_counter() - t0
            if w < best:
                best, best_tm = w, tm
        del res
        r = {"N": N, "mode": mode, "wall_s": best, "per_orbit_ms": best / N * 1e3,
             "alloc_h2d_ms": best_tm[0], "kernel_ms": best_tm[1], "d2h_ms": best_tm[2],
             "nfev_mean": float(st[:, 0].mean()), "nfev_max": int(st[:, 0].max()),
             "naccept_mean": float(st[:, 1].mean()), "nreject_mean": float(st[:, 2].mean()),
             "nfev_total": int(st[:, 0].astype(np.int64).sum())}
        r["fevals_per_s_kernel"] = r["nfev_total"] / (best_tm[1] * 1e-3)
        out["results"].append(r)
        print(f"[{dev} {out['variant']} {mode}] N={N:8d}: wall {best:8.3f}s "
              f"({r['per_orbit_ms']*1e3:8.3f} us/orbit; kernel {best_tm[1]:.1f} ms, "
              f"d2h {best_tm[2]:.1f} ms); nfev/orbit {r['nfev_mean']:.0f} (max {r['nfev_max']})",
              flush=True)
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "w") as fh:
                json.dump(out, fh, indent=1)
