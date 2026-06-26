"""Verify EVERY backend-migrated action-angle method on torch CUDA (the backend
that hard-crashes on the device-preservation bug; jax silently round-trips).
For each method: run on a CUDA input, assert (a) no crash, (b) result is on CUDA,
(c) value parity vs the numpy path. Run in the worktree with the device fixes."""
import warnings, numpy
warnings.filterwarnings("ignore")
import torch
torch.set_default_dtype(torch.float64)
assert torch.cuda.is_available(), "no CUDA"
import galpy
assert galpy.__file__.startswith("/var/tmp/staeckel-freqs"), galpy.__file__
from galpy.potential import (MWPotential2014, IsochronePotential, toVerticalPotential)

DEV = "cuda"
def t(a):  # numpy -> cuda tensor
    return torch.as_tensor(numpy.asarray(a, dtype=float), device=DEV)
def npy(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else numpy.asarray(x)
def oncuda(x):
    return torch.is_tensor(x) and x.is_cuda

results = []
def check(name, fn_np, fn_cuda, ncomp):
    try:
        rn = fn_np(); rc = fn_cuda()
        rn = rn if isinstance(rn, tuple) else (rn,)
        rc = rc if isinstance(rc, tuple) else (rc,)
        dev_ok = all(oncuda(c) for c in rc[:ncomp])
        maxd = max(float(numpy.nanmax(numpy.abs(npy(c) - numpy.asarray(n))))
                   for n, c in zip(rn[:ncomp], rc[:ncomp]))
        ok = dev_ok and maxd < 1e-6
        results.append((name, "OK" if ok else "MISMATCH", f"dev_cuda={dev_ok} maxdiff={maxd:.1e}"))
    except Exception as e:
        results.append((name, "CRASH", f"{type(e).__name__}: {str(e)[:80]}"))

# ---- Staeckel (actions, freqs, angles) ----
from galpy.actionAngle import actionAngleStaeckel
aSt = actionAngleStaeckel(pot=MWPotential2014, delta=0.45, c=False)
R, vR, vT, z, vz = [numpy.array([1.0, 1.1, 0.9]), numpy.array([0.1, -0.2, 0.15]),
                    numpy.array([0.9, 1.0, 0.85]), numpy.array([0.1, -0.1, 0.2]),
                    numpy.array([0.05, 0.1, -0.08])]
phi = numpy.array([0.4, 2.1, 5.0])
check("Staeckel.__call__", lambda: aSt(R, vR, vT, z, vz),
      lambda: aSt(t(R), t(vR), t(vT), t(z), t(vz)), 3)
check("Staeckel.actionsFreqs", lambda: aSt.actionsFreqs(R, vR, vT, z, vz),
      lambda: aSt.actionsFreqs(t(R), t(vR), t(vT), t(z), t(vz)), 6)
check("Staeckel.actionsFreqsAngles", lambda: aSt.actionsFreqsAngles(R, vR, vT, z, vz, phi),
      lambda: aSt.actionsFreqsAngles(t(R), t(vR), t(vT), t(z), t(vz), t(phi)), 9)
check("Staeckel.EccZmaxRperiRap", lambda: aSt.EccZmaxRperiRap(R, vR, vT, z, vz),
      lambda: aSt.EccZmaxRperiRap(t(R), t(vR), t(vT), t(z), t(vz)), 4)

# ---- Vertical (J, freqs, angles) ----
from galpy.actionAngle import actionAngleVertical
vp = toVerticalPotential(MWPotential2014, 1.1)
aV = actionAngleVertical(pot=vp)
x, vx = numpy.array([0.1, 0.2, -0.15]), numpy.array([0.05, -0.1, 0.2])
check("Vertical.__call__", lambda: aV(x, vx), lambda: aV(t(x), t(vx)), 1)
check("Vertical.actionsFreqs", lambda: aV.actionsFreqs(x, vx),
      lambda: aV.actionsFreqs(t(x), t(vx)), 2)
check("Vertical.actionsFreqsAngles", lambda: aV.actionsFreqsAngles(x, vx),
      lambda: aV.actionsFreqsAngles(t(x), t(vx)), 3)

# ---- Spherical (Jr, freqs, angles) ----
from galpy.actionAngle import actionAngleSpherical
ip = IsochronePotential(normalize=1.0)
aSp = actionAngleSpherical(pot=ip)
Rs, vRs, vTs, zs, vzs = (numpy.array([1.0, 1.1]), numpy.array([0.1, -0.1]),
                         numpy.array([0.9, 1.0]), numpy.array([0.05, 0.0]),
                         numpy.array([0.05, 0.1]))
phis = numpy.array([0.4, 2.1])
check("Spherical.__call__", lambda: aSp(Rs, vRs, vTs, zs, vzs),
      lambda: aSp(t(Rs), t(vRs), t(vTs), t(zs), t(vzs)), 3)
check("Spherical.actionsFreqs", lambda: aSp.actionsFreqs(Rs, vRs, vTs, zs, vzs),
      lambda: aSp.actionsFreqs(t(Rs), t(vRs), t(vTs), t(zs), t(vzs)), 6)
check("Spherical.actionsFreqsAngles", lambda: aSp.actionsFreqsAngles(Rs, vRs, vTs, zs, vzs, phis),
      lambda: aSp.actionsFreqsAngles(t(Rs), t(vRs), t(vTs), t(zs), t(vzs), t(phis)), 9)

# ---- Harmonic (alleged false positive -- verify) ----
from galpy.actionAngle import actionAngleHarmonic, actionAngleHarmonicInverse
aH = actionAngleHarmonic(omega=1.3)
hx, hvx = numpy.array([0.5, -1.0, 0.3]), numpy.array([0.2, 0.4, -0.6])
check("Harmonic._evaluate", lambda: aH._evaluate(hx, hvx), lambda: aH._evaluate(t(hx), t(hvx)), 1)
check("Harmonic.actionsFreqs", lambda: aH._actionsFreqs(hx, hvx),
      lambda: aH._actionsFreqs(t(hx), t(hvx)), 2)
check("Harmonic.actionsFreqsAngles", lambda: aH._actionsFreqsAngles(hx, hvx),
      lambda: aH._actionsFreqsAngles(t(hx), t(hvx)), 3)
aHi = actionAngleHarmonicInverse(omega=1.3)
j, ang = numpy.array([0.1, 0.3, 0.2]), numpy.array([0.4, 1.2, 2.5])
check("HarmonicInverse._evaluate", lambda: aHi._evaluate(j, ang), lambda: aHi._evaluate(t(j), t(ang)), 2)

# ---- Isochrone (closed-form) ----
from galpy.actionAngle import actionAngleIsochrone, actionAngleIsochroneInverse
aI = actionAngleIsochrone(b=1.0)
check("Isochrone._evaluate", lambda: aI._evaluate(R, vR, vT, z, vz),
      lambda: aI._evaluate(t(R), t(vR), t(vT), t(z), t(vz)), 3)
check("Isochrone.actionsFreqs", lambda: aI._actionsFreqs(R, vR, vT, z, vz),
      lambda: aI._actionsFreqs(t(R), t(vR), t(vT), t(z), t(vz)), 6)
check("Isochrone.actionsFreqsAngles", lambda: aI._actionsFreqsAngles(R, vR, vT, z, vz, phi),
      lambda: aI._actionsFreqsAngles(t(R), t(vR), t(vT), t(z), t(vz), t(phi)), 9)

print("\n=== torch CUDA verification of all action-angle methods ===")
nbad = 0
for name, status, detail in results:
    flag = "" if status == "OK" else "  <<<<<"
    if status != "OK":
        nbad += 1
    print(f"  {status:9s} {name:34s} {detail}{flag}")
print(f"\n{len(results)-nbad}/{len(results)} methods OK on torch CUDA")
