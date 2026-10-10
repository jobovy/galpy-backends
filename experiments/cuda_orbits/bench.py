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
p.add_argument("--refill", action="store_true", help="persistent-thread refill kernel")
p.add_argument("--sort", action="store_true", help="schedule by cost proxy (longest first)")
p.add_argument("--pinned", action="store_true", help="page-locked result buffer (allocated once per N, untimed)")
p.add_argument("--fixed-nstep", type=int, default=0, help="diagnostic: constant step, this many steps")
p.add_argument("--nsh", default=None, help="NSH,MINB: dop853 stage slots in shared memory + register cap (refill, faithful)")
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
if args.nsh:
    _kinfo = co.kernel_info(lib, "smem", tuple(int(x) for x in args.nsh.split(",")), args.block)
elif args.refill:
    _kinfo = co.kernel_info(lib, "refill", 0, args.block)
else:
    _kinfo = co.kernel_info(lib, "v1", 0, args.block)
out = {"backend": "cuda (galpy dop853_c port, one thread per orbit)",
       "device": dev, "variant": args.variant or "fmad", "block": args.block,
       "refill": args.refill, "sort": args.sort, "pinned": args.pinned,
       "fixed_nstep": args.fixed_nstep, "nsh": args.nsh,
       "nt": nt, "tspan_Gyr": 2.0, "rtol": args.tol, "atol": args.tol,
       "potential": "MWPotential2014", "host": socket.gethostname(),
       "kernel_info_faithful": _kinfo, "results": []}
for mode in args.modes.split(","):
    lazy = mode == "lazy"
    for N in [int(float(s)) for s in args.Ns.split(",")]:
        ics = make_ics(N, np.random.default_rng(N))
        fixed_h = 56.2 / args.fixed_nstep if args.fixed_nstep else 0.0
        nsh = tuple(int(x) for x in args.nsh.split(",")) if args.nsh else None
        kw = dict(rtol=args.tol, atol=args.tol, lazy=lazy, block=args.block,
                  refill=args.refill, fixed_h=fixed_h, nsh=nsh)
        out_buf = co.pinned_empty(lib, (N, nt, 6)) if args.pinned else None
        _, st, _ = co.integrate(lib, ics, ts, stats=True, out=out_buf,
                                order=co.cost_order(ics) if args.sort else None, **kw)
        apps0 = co.gpu_apps()
        best, best_tm = 1e18, None
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            order = co.cost_order(ics) if args.sort else None  # timed
            res, _, tm = co.integrate(lib, ics, ts, out=out_buf, order=order, **kw)
            w = time.perf_counter() - t0
            if w < best:
                best, best_tm = w, tm
        del res, out_buf
        apps1 = co.gpu_apps()
        r = {"N": N, "mode": mode, "wall_s": best, "per_orbit_ms": best / N * 1e3,
             "alloc_h2d_ms": best_tm[0], "kernel_ms": best_tm[1], "d2h_ms": best_tm[2],
             "nfev_mean": float(st[:, 0].mean()), "nfev_max": int(st[:, 0].max()),
             "naccept_mean": float(st[:, 1].mean()), "nreject_mean": float(st[:, 2].mean()),
             "nfev_total": int(st[:, 0].astype(np.int64).sum()),
             "gpu_apps_start": apps0, "gpu_apps_end": apps1,
             "gpu_exclusive": (apps0 == [] and apps1 == []) if apps0 is not None else None}
        r["fevals_per_s_kernel"] = r["nfev_total"] / (best_tm[1] * 1e-3)
        # lane utilisation a one-orbit-per-thread kernel would get with this schedule:
        # sum(nfev) / sum over warps of 32*max(nfev in warp)
        sched = co.cost_order(ics) if args.sort else np.arange(N)
        w = st[sched, 0][: N // 32 * 32].reshape(-1, 32).astype(float)
        r["warp_lane_eff_static"] = float(w.sum() / (32 * w.max(1)).sum()) if len(w) else None
        out["results"].append(r)
        tag = "".join(x for x, on in [(f"+smem{args.nsh}", args.nsh), ("+refill", args.refill), ("+sort", args.sort),
                                      ("+pinned", args.pinned),
                                      (f"+fixed{args.fixed_nstep}", args.fixed_nstep)] if on)
        print(f"[{dev} {out['variant']}{tag} {mode}] N={N:8d}: wall {best:8.3f}s "
              f"({r['per_orbit_ms']*1e3:8.3f} us/orbit; kernel {best_tm[1]:.1f} ms, "
              f"d2h {best_tm[2]:.1f} ms); nfev/orbit {r['nfev_mean']:.0f} (max {r['nfev_max']})"
              f"{'' if r['gpu_exclusive'] else '  [GPU SHARED: %s]' % r['gpu_apps_start']}",
              flush=True)
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "w") as fh:
                json.dump(out, fh, indent=1)
