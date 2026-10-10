"""Port check: the symplectic kernels' host build (and optionally the GPU) vs galpy's
leapfrog_c / symplec4_c / symplec6_c on the 300 reference orbits.

    python check_symplectic.py [--gpu] [--variant _noinl] [--minb 4]
"""
import glob
import sys

import numpy as np

import os

if "--gpu" not in sys.argv:
    os.environ["CO_HOST_ONLY"] = "1"  # host-only run: no CUDA context
import cuda_orbits as co

variant = sys.argv[sys.argv.index("--variant") + 1] if "--variant" in sys.argv else ""
minb = int(sys.argv[sys.argv.index("--minb") + 1]) if "--minb" in sys.argv else 4
lib = co.load(variant)
co.set_potential(lib, *co.load_mw2014())


def xyz(o):
    R, z, phi = o[:, -1, 0], o[:, -1, 3], o[:, -1, 5]
    return np.stack([R * np.cos(phi), R * np.sin(phi), z], axis=1)


for f in sorted(glob.glob("ref_N300_*_c_tol*.npz")):
    ref = np.load(f)
    method = str(ref["method"]).replace("_c", "")
    if method not in co.SCHEMES:
        continue
    tol, dtfix = float(ref["tol"]), float(ref["dtfix"])
    ics, ts, gorb = ref["ics"], ref["ts"], ref["orbit"]
    for dev in ["host"] + (["gpu"] if "--gpu" in sys.argv else []):
        kw = dict(minb=minb) if dev == "gpu" else {}
        if dev == "gpu" and "--per-thread2" in sys.argv:
            if not dtfix:
                continue
            kw = dict(minb=min(minb, 2), per_thread=2)
        res, st, _ = co.integrate_symp(lib, ics, ts, method, rtol=tol, atol=tol, dt=dtfix,
                                       gpu=dev == "gpu", stats=True, **kw)
        res = res.astype(np.float64)
        d = np.linalg.norm(xyz(res) - xyz(gorb), axis=1) / np.linalg.norm(xyz(gorb), axis=1)
        bit = np.mean(np.all(res == gorb, axis=(1, 2)))
        print(f"[{dev}{variant}] {method} tol={tol:g} dt={'per-orbit' if not dtfix else f'{dtfix:.4g}'}:"
              f" bit-identical {bit*100:.1f}%, max |dx|/|x| at t_end {d.max():.2e};"
              f" substeps/output {st[:,1].min()}-{st[:,1].max()}, nfev/orbit {st[:,0].mean():.0f}")
