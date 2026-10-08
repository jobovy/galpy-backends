"""Validate the device port of GSL's gamma_inc_P: vs libgsl (bitwise) and vs scipy.

    python check_gammainc.py [--gpu]
"""
import ctypes
import sys

import numpy as np
from scipy.special import gammainc, gammaln

import cuda_orbits as co

variant = sys.argv[sys.argv.index("--variant") + 1] if "--variant" in sys.argv else ""
lib = co.load(variant)
print(f"variant {variant!r}")
gsl = ctypes.CDLL("/home/bovy/miniforge3/envs/py313/lib/libgsl.so.28")
gsl.gsl_sf_gamma_inc_P.restype = ctypes.c_double
gsl.gsl_sf_gamma_inc_P.argtypes = [ctypes.c_double, ctypes.c_double]
gsl.gsl_sf_lngamma.restype = ctypes.c_double
gsl.gsl_sf_lngamma.argtypes = [ctypes.c_double]

rng = np.random.default_rng(1)
cases = {
    "a=0.6 (MWPotential2014), x in [1e-8,1e3] log": (np.full(20000, 0.6), 10 ** rng.uniform(-8, 3, 20000)),
    "a=0.6, x in [10,30] (series/asymptotic switch at 20)": (np.full(20000, 0.6), rng.uniform(10, 30, 20000)),
    "a in (0.05,9.9), x in [1e-6,200] log": (rng.uniform(0.05, 9.9, 20000), 10 ** rng.uniform(-6, np.log10(200), 20000)),
}
for name, (a, x) in cases.items():
    lg1 = np.array([gsl.gsl_sf_lngamma(ai + 1.0) for ai in a])
    ref_gsl = np.array([gsl.gsl_sf_gamma_inc_P(ai, xi) for ai, xi in zip(a, x)])
    ref_sp = gammainc(a, x)
    for dev in ["host"] + (["gpu"] if "--gpu" in sys.argv else []):
        out = np.empty_like(a)
        assert lib.co_gammaP(len(a), a, x, lg1, out, int(dev == "gpu")) == 0
        bit = np.mean(out == ref_gsl)
        rel_gsl = np.max(np.abs(out / ref_gsl - 1))
        rel_sp = np.max(np.abs(out / ref_sp - 1))
        print(f"[{dev}] {name}: bit-identical to libgsl {bit*100:.2f}%, "
              f"max rel vs libgsl {rel_gsl:.2e}, max rel vs scipy {rel_sp:.2e} "
              f"(libgsl vs scipy {np.max(np.abs(ref_gsl / ref_sp - 1)):.2e})")
