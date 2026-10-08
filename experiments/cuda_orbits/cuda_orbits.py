"""ctypes wrapper for libcudaorbits.so (one-thread-per-orbit galpy dop853 on CUDA).

Only numpy is needed.  The MWPotential2014 parameters come from
mw2014_params.json (written by make_params.py from galpy's own _parse_pot and
GSL's gamma/lngamma), so the H100 run needs neither galpy nor GSL.
"""

import ctypes
import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_dp = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
_ip = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")


def load(variant=""):
    """variant: "" (device FMA contraction), "_nofma", or "_f32" (float32 build)."""
    lib = ctypes.CDLL(os.path.join(_HERE, f"libcudaorbits{variant}.so"))
    lib.co_real_size.restype = ctypes.c_int
    lib.dtype = np.float32 if lib.co_real_size() == 4 else np.float64
    rp = np.ctypeslib.ndpointer(dtype=lib.dtype, flags="C_CONTIGUOUS")
    lib.co_set_potential.argtypes = [ctypes.c_int, _ip, _ip, ctypes.c_int, rp]
    sig = [ctypes.c_int, rp, ctypes.c_int, rp, ctypes.c_double, ctypes.c_double, rp,
           ctypes.c_void_p, ctypes.c_int]
    lib.co_integrate_host.argtypes = sig
    lib.co_integrate_gpu.argtypes = sig + [ctypes.c_int, ctypes.c_void_p]
    lib.co_gammaP.argtypes = [ctypes.c_int, rp, rp, rp, rp, ctypes.c_int]
    lib.co_device_name.argtypes = [ctypes.c_char_p, ctypes.c_int]
    return lib


def device_name(lib):
    buf = ctypes.create_string_buffer(256)
    lib.co_device_name(buf, 256)
    return buf.value.decode()


def repack(pot_type, pot_args, gamma_a, lngamma_a1):
    """galpy (pot_type, pot_args) -> (types, offsets, args) for the CUDA PotSet.

    PowerSphericalPotentialwCutoff's per-call constants are precomputed exactly
    as galpy computes them each call: 2.*M_PI*pow(rc,3.-alpha) (libm pow, as
    Python's math.pow), gsl_sf_gamma(a), gsl_sf_lngamma(a+1).
    """
    import math

    types, offs, args = [], [], []
    k = 0
    for ty in pot_type:
        types.append(ty)
        offs.append(len(args))
        if ty == 15:  # amp, alpha, rc, cache, cache
            amp, alpha, rc = pot_args[k : k + 3]
            a = 1.5 - 0.5 * alpha
            args += [amp, alpha, rc, 2.0 * math.pi * math.pow(rc, 3.0 - alpha),
                     gamma_a[a], lngamma_a1[a], a]
            k += 5
        elif ty == 5:  # MiyamotoNagai: amp, a, b
            args += list(pot_args[k : k + 3])
            k += 3
        elif ty == 9:  # NFW: amp, a
            args += list(pot_args[k : k + 2])
            k += 2
        else:
            raise NotImplementedError(f"galpy pot_type {ty} not ported")
    return (np.array(types, dtype=np.int32), np.array(offs, dtype=np.int32),
            np.array(args, dtype=np.float64))


def load_mw2014(path=os.path.join(_HERE, "mw2014_params.json")):
    with open(path) as fh:
        d = json.load(fh)
    pot_args = [float.fromhex(h) for h in d["pot_args_hex"]]
    gam = {float.fromhex(k): float.fromhex(v) for k, v in d["gamma_a_hex"].items()}
    lg1 = {float.fromhex(k): float.fromhex(v) for k, v in d["lngamma_a1_hex"].items()}
    return repack(d["pot_type"], pot_args, gam, lg1)


def set_potential(lib, types, offs, args):
    args = np.ascontiguousarray(args, dtype=lib.dtype)
    rc = lib.co_set_potential(len(types), types, offs, len(args), args)
    if rc != 0:
        raise RuntimeError(f"co_set_potential failed ({rc})")


def integrate(lib, yo, ts, rtol=1e-8, atol=1e-8, gpu=True, lazy=True, block=128,
              stats=False):
    """yo: (N,6) galpy [R,vR,vT,z,vz,phi]; returns (N,nt,6) like Orbit.getOrbit().

    Returns (result, stats(N,3)=[nfev,naccept,nreject] or None, timings_ms or None).
    """
    yo = np.ascontiguousarray(yo, dtype=lib.dtype)
    ts = np.ascontiguousarray(ts, dtype=lib.dtype)
    N, nt = yo.shape[0], len(ts)
    res = np.empty((N, nt, 6), dtype=lib.dtype)
    st = np.empty((N, 3), dtype=np.int32) if stats else None
    stp = st.ctypes.data_as(ctypes.c_void_p) if stats else None
    lr, la = np.log(rtol), np.log(atol)  # galpy passes log(tol); C does exp()
    if gpu:
        tm = (ctypes.c_float * 3)()
        rc = lib.co_integrate_gpu(N, yo, nt, ts, lr, la, res, stp, int(lazy), block, tm)
        tms = list(tm)
    else:
        rc = lib.co_integrate_host(N, yo, nt, ts, lr, la, res, stp, int(lazy))
        tms = None
    if rc != 0:
        raise RuntimeError(f"integration failed ({rc})")
    return res, st, tms
