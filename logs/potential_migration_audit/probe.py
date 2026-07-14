#!/usr/bin/env python3
"""TEST-BASED backend-migration audit of galpy Potential/Force classes.

Instantiates every concrete Potential/Force subclass exported from
galpy.potential and evaluates evaluateRforces under (1) jax eager,
(2) jax traced (jacfwd), (3) torch, classifying each as
FULLY-BACKEND / EAGER-ONLY / NUMPY-FALLTHROUGH / CRASH / OUT-OF-SCOPE /
NOT-INSTANTIATED.

Run: CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu PYTHONPATH=/var/tmp/perf python3 probe.py
"""
import sys, inspect, traceback, math

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy
import torch
torch.set_default_dtype(torch.float64)

import galpy
assert "/var/tmp/perf" in galpy.__file__, galpy.__file__
import galpy.potential as P
from galpy.backend import is_backend_array
from galpy.potential import evaluateRforces
from galpy.orbit import Orbit

# ---------------------------------------------------------------- enumerate
names = [n for n in dir(P)
         if isinstance(getattr(P, n), type)
         and issubclass(getattr(P, n), (P.Potential, P.Force))
         and not inspect.isabstract(getattr(P, n))]
# Drop the abstract base classes themselves
for base in ("Potential", "Force", "planarPotential", "linearPotential"):
    if base in names:
        names.remove(base)

# migrated children for wrappers / composites / friction dens
def hern():   return P.HernquistPotential(amp=0.5, a=0.3)
def loghalo(): return P.LogarithmicHaloPotential(amp=1.0, q=0.9)
def nfw():    return P.NFWPotential(amp=1.0, a=2.0)

OUT_OF_SCOPE = {"SnapshotRZPotential", "InterpSnapshotRZPotential"}

# ---------------------------------------------------------------- builders
def build(n):
    """Return (instance, is_velocity_dependent) or raise."""
    cls = getattr(P, n)
    veldep = False
    if n in OUT_OF_SCOPE:
        raise RuntimeError("OUT-OF-SCOPE: needs pynbody snapshot")

    # wrappers taking pot=
    if n.endswith("WrapperPotential"):
        if n == "AdiabaticContractionWrapperPotential":
            return cls(pot=nfw(), baryonpot=hern()), veldep
        if n == "RotateAndTiltWrapperPotential":
            return cls(pot=hern(), inclination=0.3, galaxy_pa=0.2, sky_pa=0.1), veldep
        if n == "KuzminLikeWrapperPotential":
            return cls(pot=hern(), a=1.1, b=0.2), veldep
        # DehnenSmooth/GaussianAmplitude/SolidBody/CorotatingRotation etc.
        return cls(pot=hern()), veldep

    # velocity-dependent friction forces
    if n == "ChandrasekharDynamicalFrictionForce":
        return cls(GMs=0.05, rhm=0.1, dens=hern()), True
    if n == "FDMDynamicalFrictionForce":
        return cls(GMs=0.05, rhm=0.1, m=1e-22, dens=hern()), True

    if n == "MovingObjectPotential":
        # keep perturber orbit away from the (R=1.1,phi=0.2) eval point
        o = Orbit([0.5, 0.1, 1.0, 0.0, 0.1, 2.0])
        import numpy as _np
        ts = _np.linspace(0.0, 2.0, 51)
        o.integrate(ts, loghalo())
        return cls(o, pot=hern()), veldep

    if n == "NonInertialFrameForce":
        return cls(Omega=1.0), True  # dissipative -> needs v=

    if n == "CompositePotential":
        return cls(hern(), loghalo()), veldep

    if n == "NullPotential":
        return cls(), veldep

    if n == "interpRZPotential":
        import numpy as _np
        return cls(RZPot=hern(), interpRforce=True,
                   rgrid=(0.01, 2.0, 51), zgrid=(0.0, 1.0, 51), logR=False,
                   enable_c=False, use_c=False), veldep

    if n == "interpSphericalPotential":
        # build from a migrated spherical pot via from_pot classmethod if present
        try:
            return cls.from_pot(hern()), veldep
        except Exception:
            import numpy as _np
            rg = _np.linspace(0.01, 20.0, 101)
            hp = hern()
            rf = _np.array([hp.rforce(r, 0.0) for r in rg])
            return cls(rforce=lambda r: -0.5 / r**2, rgrid=rg), veldep

    if n == "KingPotential":
        return cls(W0=2.0, M=3.0, rt=1.5), veldep

    if n == "SCFPotential":
        return cls(), veldep  # default Acos=[[[1]]] => Hernquist

    if n in ("DiskSCFPotential", "DiskMultipoleExpansionPotential",
             "MultipoleExpansionPotential"):
        if n == "MultipoleExpansionPotential":
            # build from a simple spherical density via from-density-like path
            try:
                from galpy.potential import scf_compute_coeffs_spherical
                # not applicable; use default-ish: build multipole from Hernquist
                hp = hern()
                return P.MultipoleExpansionPotential(
                    dens=lambda R, z, phi=0.: hp.dens(R, z, phi),
                    rgrid=numpy.geomspace(1e-2, 20.0, 201), L=4), veldep
            except Exception:
                return cls(), veldep
        return cls(), veldep  # defaults => exponential disk

    # everything else: try defaults, then amp=1.0
    try:
        return cls(), veldep
    except TypeError:
        return cls(amp=1.0), veldep

# ---------------------------------------------------------------- evaluators
def ev_jax_eager(pot, veldep):
    R = jnp.asarray(1.1); z = jnp.asarray(0.1); phi = jnp.asarray(0.2)
    t = jnp.asarray(0.0)
    kw = dict(phi=phi, t=t)
    if veldep:
        kw["v"] = jnp.asarray([0.1, 0.2, 0.05])
    out = evaluateRforces(pot, R, z, **kw)
    return out

def ev_jax_traced(pot, veldep):
    z = jnp.asarray(0.1); phi = jnp.asarray(0.2); t = jnp.asarray(0.0)
    v = jnp.asarray([0.1, 0.2, 0.05]) if veldep else None
    def f(R):
        kw = dict(phi=phi, t=t)
        if v is not None:
            kw["v"] = v
        return jnp.sum(evaluateRforces(pot, R, z, **kw))
    g = jax.jacfwd(f)(jnp.asarray(1.1))
    return g

def ev_torch(pot, veldep):
    R = torch.tensor(1.1); z = torch.tensor(0.1); phi = torch.tensor(0.2)
    t = torch.tensor(0.0)
    kw = dict(phi=phi, t=t)
    if veldep:
        kw["v"] = torch.tensor([0.1, 0.2, 0.05])
    out = evaluateRforces(pot, R, z, **kw)
    return out

def classify_out(out):
    """Return (ok, is_backend, finite, detail)."""
    try:
        isb = bool(is_backend_array(out))
    except Exception:
        isb = False
    try:
        arr = numpy.asarray(out.detach() if hasattr(out, "detach") else out,
                            dtype=float)
        finite = bool(numpy.all(numpy.isfinite(arr)))
    except Exception as e:
        return False, isb, False, f"finite-check-err {e}"
    return True, isb, finite, ""

def short_exc():
    et, ev, _ = sys.exc_info()
    first = str(ev).strip().splitlines()
    first = first[0] if first else ""
    return f"{et.__name__}: {first[:160]}"

# ---------------------------------------------------------------- run
results = {}
for n in sorted(names):
    rec = {"instantiated": False, "veldep": False,
           "jax_eager": "", "jax_traced": "", "torch": "",
           "classification": "", "notes": ""}
    # reset to numpy backend between classes
    try:
        galpy.backend.use("numpy", force=False)
    except Exception:
        pass
    try:
        pot, veldep = build(n)
        rec["instantiated"] = True
        rec["veldep"] = veldep
    except Exception:
        msg = short_exc()
        if "OUT-OF-SCOPE" in msg:
            rec["classification"] = "OUT-OF-SCOPE"
            rec["notes"] = msg
        else:
            rec["classification"] = "NOT-INSTANTIATED"
            rec["notes"] = "ctor: " + msg
        results[n] = rec
        continue

    # jax eager
    eager_backend = eager_finite = False
    try:
        out = ev_jax_eager(pot, veldep)
        ok, isb, fin, det = classify_out(out)
        eager_backend, eager_finite = isb, fin
        rec["jax_eager"] = f"backend={isb} finite={fin}" + (" " + det if det else "")
    except Exception:
        rec["jax_eager"] = "CRASH " + short_exc()

    # jax traced
    traced_ok = False
    try:
        g = ev_jax_traced(pot, veldep)
        ok, isb, fin, det = classify_out(g)
        traced_ok = isb and fin
        rec["jax_traced"] = f"backend={isb} finite={fin}" + (" " + det if det else "")
    except Exception:
        rec["jax_traced"] = "CRASH " + short_exc()

    # torch
    torch_backend = torch_finite = False
    try:
        galpy.backend.use("numpy", force=False)
        out = ev_torch(pot, veldep)
        ok, isb, fin, det = classify_out(out)
        torch_backend, torch_finite = isb, fin
        rec["torch"] = f"backend={isb} finite={fin}" + (" " + det if det else "")
    except Exception:
        rec["torch"] = "CRASH " + short_exc()

    # classify
    jax_eager_ok = eager_backend and eager_finite
    torch_ok = torch_backend and torch_finite
    jax_crash = rec["jax_eager"].startswith("CRASH")
    torch_crash = rec["torch"].startswith("CRASH")
    traced_crash = rec["jax_traced"].startswith("CRASH")

    if jax_eager_ok and traced_ok and torch_ok:
        rec["classification"] = "FULLY-BACKEND"
    elif jax_crash or torch_crash:
        rec["classification"] = "CRASH"
    elif (rec["jax_eager"].startswith("backend=False") or
          rec["torch"].startswith("backend=False")):
        rec["classification"] = "NUMPY-FALLTHROUGH"
    elif jax_eager_ok and torch_ok and (traced_crash or not traced_ok):
        rec["classification"] = "EAGER-ONLY"
    else:
        rec["classification"] = "PARTIAL/CHECK"
    results[n] = rec

# ---------------------------------------------------------------- dump
import json
with open("/home/bovy/Repos/galpy-jax/logs/potential_migration_audit/raw_results.json", "w") as fh:
    json.dump(results, fh, indent=2)

# print a compact table to stdout
order = {"CRASH":0,"NUMPY-FALLTHROUGH":1,"EAGER-ONLY":2,"PARTIAL/CHECK":3,
         "NOT-INSTANTIATED":4,"OUT-OF-SCOPE":5,"FULLY-BACKEND":6}
for n in sorted(results, key=lambda k: (order.get(results[k]["classification"],9), k)):
    r = results[n]
    print(f"{n:42s} | {r['classification']:17s} | E:{r['jax_eager'][:34]:34s} | T:{r['jax_traced'][:30]:30s} | Tor:{r['torch'][:30]:30s} | {r['notes'][:60]}")

from collections import Counter
c = Counter(r["classification"] for r in results.values())
print("\nSUMMARY", dict(c), "TOTAL", len(results))
