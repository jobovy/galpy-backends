"""Timing of galpy's symplectic integrators on the GPU (one orbit per thread, refill),
MWPotential2014, 2 Gyr, nt=16 saves, ICs as bench.py (make_ics, rng=default_rng(N)).

    python bench_symp.py --scheme symplec4 --n 64 [--Ns ...] [--variant _fast_noinl] [--minb 4]
    python bench_symp.py --scheme symplec4 --tol 1e-8      # galpy per-orbit estimate_step

--n: common dt = output spacing / n (all orbits in lockstep);  --tol: per-orbit dt.
wall_s = alloc + H2D + kernel + D2H (pageable, or --pinned reused buffer), best of --repeats.
"""
import argparse
import json
import os
import socket
import time

import numpy as np

import cuda_orbits as co

p = argparse.ArgumentParser()
p.add_argument("--scheme", default="symplec4", choices=list(co.SCHEMES))
p.add_argument("--n", type=int, default=0)
p.add_argument("--tol", type=float, default=0.0)
p.add_argument("--Ns", default="1e4,1e5,1e6")
p.add_argument("--variant", default="_noinl")
p.add_argument("--minb", type=int, default=4)
p.add_argument("--block", type=int, default=128)
p.add_argument("--repeats", type=int, default=3)
p.add_argument("--per-thread", type=int, default=1, choices=[1, 2], help="orbits per thread (2: common dt only)")
p.add_argument("--pinned", action="store_true", help="page-locked result buffer (allocated once per N, untimed)")
p.add_argument("--out", default=None)
args = p.parse_args()
assert (args.n > 0) != (args.tol > 0), "give exactly one of --n / --tol"

lib = co.load(args.variant)
co.set_potential(lib, *co.load_mw2014())
dev = co.device_name(lib)
nt = 16
ts = np.linspace(0.0, 56.2, nt)
dt = (ts[1] - ts[0]) / args.n if args.n else 0.0
tol = args.tol if args.tol else 1e-8
ic0 = np.array([1.0, 0.0, 1.0, 0.0125, 0.036, 0.0])


def make_ics(N, rng):  # verbatim from galpy_c_measure.py
    base = np.tile(ic0, (N, 1))
    base[:, 0] += rng.normal(0, 0.05, N)
    base[:, 1] += rng.normal(0, 0.01, N)
    base[:, 3] += rng.normal(0, 0.01, N)
    base[:, 4] += rng.normal(0, 0.01, N)
    return base


kw = dict(scheme=args.scheme, rtol=tol, atol=tol, dt=dt, block=args.block, minb=args.minb,
          per_thread=args.per_thread)
co.integrate_symp(lib, make_ics(1000, np.random.default_rng(0)), ts, **kw)  # warm-up
info = co.kernel_info(lib, "symp2" if args.per_thread == 2 else "symp", (args.scheme, args.minb), args.block)
out = {"backend": "cuda symplectic (galpy bovy_symplecticode.c port, one thread per orbit, refill)",
       "device": dev, "variant": args.variant, "scheme": args.scheme,
       "dt_mode": "common" if args.n else "per_orbit_estimate_step",
       "n_per_output": args.n, "dt": dt, "tol": args.tol, "minb": args.minb, "block": args.block,
       "orbits_per_thread": args.per_thread,
       "kernel_info": info, "pinned": args.pinned, "nt": nt, "tspan_Gyr": 2.0, "potential": "MWPotential2014",
       "host": socket.gethostname(), "results": []}
print(f"[{dev} {args.variant} {args.scheme} {'n=%d' % args.n if args.n else 'tol=%g' % args.tol}"
      f" minb={args.minb}{' 2/thread' if args.per_thread == 2 else ''}] regs {info['regs']} local {info['local_bytes']} B,"
      f" occupancy {info['occupancy']*100:.1f}%", flush=True)
for N in [int(float(s)) for s in args.Ns.split(",")]:
    ics = make_ics(N, np.random.default_rng(N))
    buf = co.pinned_empty(lib, (N, nt, 6)) if args.pinned else None
    _, st, _ = co.integrate_symp(lib, ics, ts, stats=True, out=buf, **kw)
    apps0 = co.gpu_apps()
    best, best_tm = 1e18, None
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        res, _, tm = co.integrate_symp(lib, ics, ts, out=buf, **kw)
        w = time.perf_counter() - t0
        if w < best:
            best, best_tm = w, tm
    apps1 = co.gpu_apps()
    r = {"N": N, "gpu_apps_start": apps0, "gpu_apps_end": apps1,
         "gpu_exclusive": (apps0 == [] and apps1 == []) if apps0 is not None else None,
         "wall_s": best, "per_orbit_ms": best / N * 1e3, "alloc_h2d_ms": best_tm[0],
         "kernel_ms": best_tm[1], "d2h_ms": best_tm[2], "nfev_mean": float(st[:, 0].mean()),
         "nfev_max": int(st[:, 0].max()), "n_per_output_range": [int(st[:, 1].min()), int(st[:, 1].max())]}
    r["fevals_per_s_kernel"] = float(st[:, 0].astype(np.int64).sum()) / (best_tm[1] * 1e-3)
    out["results"].append(r)
    print(f"   N={N:8d}: wall {best:8.3f}s ({r['per_orbit_ms']*1e3:8.3f} us/orbit; kernel"
          f" {best_tm[1]/N*1e3:.3f} us/orbit); nfev/orbit {r['nfev_mean']:.0f};"
          f" {r['fevals_per_s_kernel']:.3g} force evals/s"
          f"{'' if r['gpu_exclusive'] else '  [GPU SHARED: %s]' % r['gpu_apps_start']}", flush=True)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=1)
