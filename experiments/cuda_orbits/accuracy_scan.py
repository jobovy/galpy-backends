"""Matched-accuracy scan (host build = galpy's arithmetic, bit-identical to galpy C):
error vs cost for dop853 (rtol=atol scan) and leapfrog/symplec4/symplec6 (common dt =
output spacing / n, and galpy's per-orbit estimate_step at rtol=atol), on the 300
reference orbits (MWPotential2014, 2 Gyr, nt=16).

Error metrics, against galpy dop853_c at rtol=atol=1e-13 (ref_N300_tol1e-13.npz):
  pos_err  = max over orbits of |x(t_end) - x_true(t_end)| / |x_true(t_end)|
  energy_err = max over orbits and the 16 saves of |E(t) - E(0)| / |E(0)|
nfev = force evaluations per orbit (mean), the hardware-independent cost.

    PYTHONPATH=/var/tmp/fbp python accuracy_scan.py <scheme|dop853> [out.json]
"""
import json
import os
import sys

os.environ["CO_HOST_ONLY"] = "1"  # host build only: never create a CUDA context
import time

import numpy as np
from galpy.potential import MWPotential2014, evaluatePotentials

import cuda_orbits as co

which = sys.argv[1]
# optional: --ns 12,14,20  (common-dt only, extra n values to bracket target accuracies)
TOLS = [float(x) for x in sys.argv[sys.argv.index("--tols") + 1].split(",")] if "--tols" in sys.argv else None
NS = [int(x) for x in sys.argv[sys.argv.index("--ns") + 1].split(",")] if "--ns" in sys.argv else None
_pos = [a for i, a in enumerate(sys.argv[2:], 2)
        if not a.startswith("--") and sys.argv[i - 1] not in ("--ns", "--tols")]
out = _pos[0] if _pos else f"data/accuracy_{which}{'_fine' if (NS or TOLS) else ''}.json"
lib = co.load()
co.set_potential(lib, *co.load_mw2014())
ref = np.load("ref_N300.npz")
truth = np.load("ref_N300_tol1e-13.npz")["orbit"]
ics, ts = ref["ics"], ref["ts"]


def xyz(o):
    R, z, phi = o[:, -1, 0], o[:, -1, 3], o[:, -1, 5]
    return np.stack([R * np.cos(phi), R * np.sin(phi), z], axis=1)


def energy(o):  # o: (N, nt, 6) [R, vR, vT, z, vz, phi]
    R, vR, vT, z, vz = (o[..., i] for i in range(5))
    pot = evaluatePotentials(MWPotential2014, R.ravel(), z.ravel()).reshape(R.shape)
    return 0.5 * (vR**2 + vT**2 + vz**2) + pot


E0 = energy(truth[:, :1])


def metrics(res):
    pe = np.linalg.norm(xyz(res) - xyz(truth), axis=1) / np.linalg.norm(xyz(truth), axis=1)
    E = energy(res)
    ee = np.abs(E - E0) / np.abs(E0)
    return float(pe.max()), float(np.median(pe)), float(ee.max())


rows = []
spacing = ts[1] - ts[0]
if which == "dop853":
    for tol in (TOLS or [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]):
        t0 = time.perf_counter()
        res, st, _ = co.integrate(lib, ics, ts, rtol=tol, atol=tol, gpu=False, lazy=False, stats=True)
        pe, pm, ee = metrics(res)
        rows.append(dict(scheme="dop853", mode="adaptive", tol=tol, pos_err=pe, pos_err_median=pm,
                         energy_err=ee, nfev=float(st[:, 0].mean()), host_s=time.perf_counter() - t0))
        print(rows[-1], flush=True)
else:
    kmax = {"leapfrog": 15, "symplec4": 10, "symplec6": 8}[which]
    for n in (NS or [2**k for k in range(0, kmax + 1)]):
        t0 = time.perf_counter()
        res, st, _ = co.integrate_symp(lib, ics, ts, which, dt=spacing / n, gpu=False, stats=True)
        pe, pm, ee = metrics(res)
        rows.append(dict(scheme=which, mode="common_dt", n_per_output=n, dt=spacing / n,
                         pos_err=pe, pos_err_median=pm, energy_err=ee,
                         nfev=float(st[:, 0].mean()), host_s=time.perf_counter() - t0))
        print(rows[-1], flush=True)
    for tol in ([] if NS else [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]):
        t0 = time.perf_counter()
        res, st, _ = co.integrate_symp(lib, ics, ts, which, rtol=tol, atol=tol, gpu=False, stats=True)
        pe, pm, ee = metrics(res)
        rows.append(dict(scheme=which, mode="per_orbit_dt", tol=tol,
                         n_per_output_range=[int(st[:, 1].min()), int(st[:, 1].max())],
                         pos_err=pe, pos_err_median=pm, energy_err=ee,
                         nfev=float(st[:, 0].mean()), host_s=time.perf_counter() - t0))
        print(rows[-1], flush=True)
import os
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
with open(out, "w") as fh:
    json.dump({"reference": "galpy dop853_c rtol=atol=1e-13", "N": len(ics), "nt": len(ts),
               "rows": rows}, fh, indent=1)
