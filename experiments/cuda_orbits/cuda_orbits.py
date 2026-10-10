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
CO_NINV = 160  # must match galpy_cuda.cuh
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
    lib.co_integrate_gpu.argtypes = sig + [ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
                                           ctypes.c_double, ctypes.c_void_p, ctypes.c_int]
    ssig = [ctypes.c_int, rp, ctypes.c_int, rp, ctypes.c_double, ctypes.c_double, ctypes.c_int,
            ctypes.c_double, rp, ctypes.c_void_p]
    lib.co_symp_host.argtypes = ssig
    lib.co_symp_gpu.argtypes = ssig + [ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.co_kernel_info.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, _ip]
    lib.co_host_alloc.restype = ctypes.c_void_p
    lib.co_host_alloc.argtypes = [ctypes.c_size_t]
    lib.co_host_free.argtypes = [ctypes.c_void_p]
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
            # CO_FASTGAM builds: 1/(a+n), n < CO_NINV (unused by the GSL-exact path)
            args += [0.0] + [1.0 / (a + n) for n in range(1, CO_NINV)]
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


class _Pinned:
    def __init__(self, lib, nbytes):
        self.lib, self.ptr = lib, lib.co_host_alloc(max(int(nbytes), 1))
        if not self.ptr:
            raise MemoryError("cudaHostAlloc failed")

    def __del__(self):
        self.lib.co_host_free(self.ptr)


def pinned_empty(lib, shape, dtype=None):
    """numpy array in page-locked (pinned) host memory: steady, fast D2H copies."""
    dtype = np.dtype(dtype or lib.dtype)
    n = int(np.prod(shape))
    buf = _Pinned(lib, n * dtype.itemsize)
    arr = np.ctypeslib.as_array((ctypes.c_char * (n * dtype.itemsize)).from_address(buf.ptr))
    arr = arr.view(dtype).reshape(shape).view(_PinnedArray)
    arr._pinned = buf  # keeps the allocation alive (views hold arr via .base)
    return arr


class _PinnedArray(np.ndarray):
    pass


def integrate(lib, yo, ts, rtol=1e-8, atol=1e-8, gpu=True, lazy=True, block=128,
              stats=False, refill=False, fixed_h=0.0, out=None, order=None, nsh=None):
    """yo: (N,6) galpy [R,vR,vT,z,vz,phi]; returns (N,nt,6) like Orbit.getOrbit().

    out: optional preallocated (N,nt,6) result array (e.g. pinned_empty).
    order: optional int32 (N,) schedule (orbit order[k] is started k-th); results
    stay in input order.  refill: persistent-thread kernel.  fixed_h>0: diagnostic.
    nsh = (NSH, MINB), NSH in {10, 14, 18} dop853 stage slots in shared memory, MINB
    (2/3/4 -> register cap 255/170/128): refill kernel, faithful, block <= 128.
    Available: (10,2), (10,3), (14,3), (14,4), (18,3), (18,4).
    Returns (result, stats(N,3)=[nfev,naccept,nreject] or None, timings_ms or None).
    """
    yo = np.ascontiguousarray(yo, dtype=lib.dtype)
    ts = np.ascontiguousarray(ts, dtype=lib.dtype)
    N, nt = yo.shape[0], len(ts)
    res = np.empty((N, nt, 6), dtype=lib.dtype) if out is None else out
    st = np.empty((N, 3), dtype=np.int32) if stats else None
    stp = st.ctypes.data_as(ctypes.c_void_p) if stats else None
    lr, la = np.log(rtol), np.log(atol)  # galpy passes log(tol); C does exp()
    if order is not None:
        order = np.ascontiguousarray(order, dtype=np.int32)  # keep alive over the call
    if gpu:
        tm = (ctypes.c_float * 3)()
        rc = lib.co_integrate_gpu(N, yo, nt, ts, lr, la, res, stp, int(lazy), block, tm,
                                  int(refill), float(fixed_h),
                                  None if order is None else order.ctypes.data,
                                  10 * nsh[0] + nsh[1] if nsh else 0)
        tms = list(tm)
    else:
        rc = lib.co_integrate_host(N, yo, nt, ts, lr, la, res, stp, int(lazy))
        tms = None
    if rc != 0:
        raise RuntimeError(f"integration failed ({rc})")
    return res, st, tms


def cost_order(yo, nu2=9.0):
    """Schedule most-expensive-first by a vertical-energy proxy vz^2 + nu^2 z^2
    (correlates r=0.76 with nfev for the paper's ICs; problem-specific)."""
    key = yo[:, 4] ** 2 + nu2 * yo[:, 3] ** 2
    return np.argsort(-key, kind="stable").astype(np.int32)


SCHEMES = {"leapfrog": 0, "symplec4": 4, "symplec6": 6}


def integrate_symp(lib, yo, ts, scheme="symplec4", rtol=1e-8, atol=1e-8, dt=0.0, gpu=True,
                   block=128, stats=False, out=None, minb=4, per_thread=1):
    """galpy's leapfrog/symplec4/symplec6. dt>0: common step (must divide the output
    spacing); else galpy's per-orbit *_estimate_step with rtol/atol.  minb in {2,4,8}:
    register cap 255/128/64 per thread (GPU; block <= 128).  per_thread=2 (common dt only):
    two orbits per thread with one paired force call (minb in {1, 2}).
    Returns (result (N,nt,6), stats (N,3)=[nfev, substeps per output, 0] or None, timings)."""
    yo = np.ascontiguousarray(yo, dtype=lib.dtype)
    ts = np.ascontiguousarray(ts, dtype=lib.dtype)
    N, nt = yo.shape[0], len(ts)
    res = np.empty((N, nt, 6), dtype=lib.dtype) if out is None else out
    st = np.empty((N, 3), dtype=np.int32) if stats else None
    stp = st.ctypes.data_as(ctypes.c_void_p) if stats else None
    args = (N, yo, nt, ts, np.log(rtol), np.log(atol), SCHEMES[scheme], float(dt), res, stp)
    if gpu:
        tm = (ctypes.c_float * 3)()
        rc = lib.co_symp_gpu(*args, block, tm, int(minb), int(per_thread))
        tms = list(tm)
    else:
        rc = lib.co_symp_host(*args)
        tms = None
    if rc != 0:
        raise RuntimeError(f"symplectic integration failed ({rc})")
    return res, st, tms


def kernel_info(lib, kind, param=0, block=128):
    """kind: 'v1', 'refill', 'smem' (param=(NSH, MINB)), 'symp' (param=(scheme, MINB)).
    Returns dict(regs, local_bytes, blocks_per_sm, threads_per_sm, max_threads_per_sm,
    smem_per_block, occupancy)."""
    k = {"v1": 0, "refill": 1, "smem": 2, "symp": 3, "symp2": 4}[kind]
    if kind in ("symp", "symp2"):  # param = (scheme name, MINB)
        param = 10 * SCHEMES[param[0]] + param[1]
    elif kind == "smem":  # param = (NSH, MINB)
        param = 10 * param[0] + param[1]
    out = np.zeros(7, dtype=np.int32)
    rc = lib.co_kernel_info(k, int(param), int(block), out)
    if rc != 0:
        raise RuntimeError(f"co_kernel_info failed ({rc})")
    d = dict(zip(["regs", "local_bytes", "blocks_per_sm", "threads_per_sm",
                  "max_threads_per_sm", "smem_per_block", "block"], map(int, out)))
    d["occupancy"] = d["threads_per_sm"] / d["max_threads_per_sm"]
    return d


def gpu_apps():
    """Compute processes on the GPU other than this one: [(pid, name, MiB)], or None if
    nvidia-smi is unavailable.  Recorded with every timing to show whether the GPU was
    exclusive."""
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             timeout=20).stdout
    except Exception:
        return None
    apps = []
    for line in out.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 3 and parts[0].isdigit() and int(parts[0]) != os.getpid():
            apps.append((int(parts[0]), parts[1], parts[2]))
    return apps
