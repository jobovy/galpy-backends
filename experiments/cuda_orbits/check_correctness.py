"""Compare the CUDA port (host build and/or GPU) with galpy's dop853_c reference.

    python check_correctness.py ref_N300.npz [--gpu] [--variant _nofma]
"""
import sys
import time

import numpy as np

import cuda_orbits as co

ref = np.load(sys.argv[1])
variant = sys.argv[sys.argv.index("--variant") + 1] if "--variant" in sys.argv else ""
lib = co.load(variant)
co.set_potential(lib, *co.load_mw2014())
ics, ts, gorb = ref["ics"], ref["ts"], ref["orbit"]
# galpy dop853_c at rtol=atol=1e-13: the "truth" against which the 1e-8 runs'
# own truncation error is measured (context for the port-vs-galpy difference)
_tight = sys.argv[1].replace(".npz", "_tol1e-13.npz")
try:
    tight = np.load(_tight)["orbit"]
except FileNotFoundError:
    tight = None


def xyz(o):  # rectangular positions at the final time
    R, z, phi = o[:, -1, 0], o[:, -1, 3], o[:, -1, 5]
    return np.stack([R * np.cos(phi), R * np.sin(phi), z], axis=1)


if tight is not None:
    tr = np.linalg.norm(xyz(gorb) - xyz(tight), axis=1) / np.linalg.norm(xyz(tight), axis=1)
    print(f"[galpy C 1e-8 vs galpy C 1e-13] truncation error |dx|/|x| at t_end: "
          f"max {tr.max():.3e}, median {np.median(tr):.3e}")


def report(tag, res, st, st_ref=None):
    res = res.astype(np.float64)
    pg, pc = xyz(gorb), xyz(res)
    drel = np.linalg.norm(pc - pg, axis=1) / np.linalg.norm(pg, axis=1)
    alld = np.max(np.abs(res - gorb) / np.maximum(np.abs(gorb), 1e-300))
    bit = np.mean(np.all(res == gorb, axis=(1, 2)))
    print(f"[{tag}] N={len(ics)}: max |dx|/|x| at t_end = {drel.max():.3e} "
          f"(median {np.median(drel):.3e}); orbits bit-identical to galpy C over all "
          f"{len(ts)} saves: {bit*100:.1f}%; max rel diff any coord/time {alld:.2e}")
    if tight is not None:
        tc = np.linalg.norm(pc - xyz(tight), axis=1) / np.linalg.norm(xyz(tight), axis=1)
        print(f"    vs galpy C 1e-13 (truth): max {tc.max():.3e}, median {np.median(tc):.3e}")
    if st_ref is not None:
        same = np.all(st[:, 1:] == st_ref[:, 1:], axis=1)
        print(f"    same accepted/rejected step counts as galpy C: {same.mean()*100:.1f}% of orbits; "
              f"max |dx|/|x| among those {drel[same].max() if same.any() else np.nan:.3e}, "
              f"among the rest {drel[~same].max() if (~same).any() else np.nan:.3e}")
    if st is not None:
        print(f"    nfev/orbit mean {st[:,0].mean():.1f} (min {st[:,0].min()}, max {st[:,0].max()}), "
              f"accepted {st[:,1].mean():.1f}, rejected {st[:,2].mean():.2f}")


t0 = time.perf_counter()
res, st, _ = co.integrate(lib, ics, ts, gpu=False, lazy=False, stats=True)
print(f"host build (faithful, 1 thread): {(time.perf_counter()-t0)/len(ics)*1e3:.4f} ms/orbit")
report("host, faithful dense", res, st)
res2, st2, _ = co.integrate(lib, ics, ts, gpu=False, lazy=True, stats=True)
report("host, lazy dense", res2, st2)
print(f"    lazy vs faithful outputs bit-identical: {np.array_equal(res, res2)}")
if "--gpu" in sys.argv:
    for lazy in (False, True):
        r, s, tm = co.integrate(lib, ics, ts, gpu=True, lazy=lazy, stats=True)
        report(f"GPU{variant}, {'lazy' if lazy else 'faithful'} dense", r, s, st)
