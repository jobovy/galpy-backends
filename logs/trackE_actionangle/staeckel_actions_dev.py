"""Develop + validate vectorized Staeckel turning-points + actions vs per-object Single."""
import numpy
import galpy
assert "/var/tmp/staeckel-vec" in galpy.__file__, galpy.__file__
from galpy.actionAngle import actionAngleStaeckel
from galpy.actionAngle.actionAngleStaeckel import (
    actionAngleStaeckelSingle,
    calcELStaeckel,
    potentialStaeckel,
    _JRStaeckelIntegrandSquared,
    _JzStaeckelIntegrandSquared,
)
from galpy.potential import MWPotential2014
from galpy.util import coords
from galpy.backend import get_namespace
from galpy.backend.optimize import bisect_root
from galpy.backend.quadrature import fixed_quad

DELTA = 0.45
xp = numpy

# ---- batch of test orbits (varied regimes) ----
R  = numpy.array([0.9, 1.1, 0.7, 1.3, 1.0, 0.8, 1.2, 0.95])
vR = numpy.array([0.1, -0.2, 0.05, 0.0, 0.3, -0.1, 0.15, 0.0])
vT = numpy.array([1.05, 0.9, 1.2, 0.8, 1.0, 1.1, 0.85, 1.0])
z  = numpy.array([0.2, -0.1, 0.3, 0.05, 0.0, 0.15, -0.25, 0.1])
vz = numpy.array([0.08, 0.2, -0.1, 0.15, 0.1, -0.05, 0.2, 0.0])

# ---- vectorized setup ----
ux, vx = coords.Rz_to_uv(R, z, delta=DELTA)
sinvx, cosvx = numpy.sin(vx), numpy.cos(vx)
coshux, sinhux = numpy.cosh(ux), numpy.sinh(ux)
pux = DELTA * (vR * coshux * sinvx + vz * sinhux * cosvx)
pvx = DELTA * (vR * sinhux * cosvx - vz * coshux * sinvx)
E, Lz = calcELStaeckel(R, vR, vT, z, vz, MWPotential2014)
u0 = ux
v0u = numpy.pi / 2.0
sinhu0 = numpy.sinh(u0)
sinv0u = numpy.sin(v0u)
potu0v0 = potentialStaeckel(u0, v0u, MWPotential2014, DELTA)
I3U = (
    E * sinhux**2.0
    - pux**2.0 / 2.0 / DELTA**2.0
    - Lz**2.0 / 2.0 / DELTA**2.0 / sinhux**2.0
    - (sinhux**2.0 + sinv0u**2.0) * potentialStaeckel(ux, v0u, MWPotential2014, DELTA)
    + (sinhu0**2.0 + sinv0u**2.0) * potu0v0
)
sin2v0u = sinv0u**2.0
sinh2u0 = sinhu0**2.0
jr_args = (E, Lz, I3U, DELTA, u0, sinh2u0, v0u, sin2v0u, potu0v0, MWPotential2014)

def fR(u):
    return _JRStaeckelIntegrandSquared(u, *jr_args)

# ---- vectorized calcUminUmax (compute-both + where) ----
eps = 1e-8
fux = fR(ux)
peps = fR(ux + eps)
meps = fR(ux - eps)
at_turn = (numpy.fabs(pux) < 1e-7) | (numpy.fabs(fux) < 1e-10)
at_umin = at_turn & (peps > 0.0) & (meps < 0.0)
at_umax = at_turn & (peps < 0.0) & (meps > 0.0)
circular = at_turn & ~at_umin & ~at_umax

# expanding brackets: lo below ux (f<0), hi above ux (f<0)
lo = ux * 0.9
for _ in range(80):
    lo = numpy.where((fR(lo) >= 0.0) & (lo > 1e-9), lo * 0.9, lo)
hi = ux * 1.1
for _ in range(80):
    hi = numpy.where(fR(hi) >= 0.0, hi * 1.1, hi)

umin_b = bisect_root(fR, lo, ux, numpy, xtol=1e-13, maxiter=200)
umax_b = bisect_root(fR, ux, hi, numpy, xtol=1e-13, maxiter=200)
umin = numpy.where(at_umin | circular, ux, umin_b)
umax = numpy.where(at_umax | circular, ux, umax_b)

# ---- compare vs Single ----
print("=== turning points vs Single ===")
maxdu = 0.0
for ii in range(len(R)):
    s = actionAngleStaeckelSingle(R[ii], vR[ii], vT[ii], z[ii], vz[ii], pot=MWPotential2014, delta=DELTA)
    sumin, sumax = s.calcUminUmax()
    du = abs(float(numpy.atleast_1d(sumin)[0]) - umin[ii]) + abs(float(numpy.atleast_1d(sumax)[0]) - umax[ii])
    maxdu = max(maxdu, du)
    print(f"  obj{ii}: umin {umin[ii]:.10f} vs {float(numpy.atleast_1d(sumin)[0]):.10f}  umax {umax[ii]:.10f} vs {float(numpy.atleast_1d(sumax)[0]):.10f}  d={du:.2e}")
print("max |dumin|+|dumax|:", maxdu)

# ---- vectorized JR / Jz actions (GL; integrand vanishes at turning pts) ----
from galpy.actionAngle.actionAngleStaeckel import _JzStaeckelIntegrandSquared
import math
GL = 10

# vmin (v turning point) for Jz: brentq on _JzStaeckelIntegrandSquared in [vx? , pi/2]
# setup v-side quantities
u0v = u0
cosh2u0v = numpy.cosh(u0v)**2.0
sinh2u0v = numpy.sinh(u0v)**2.0
potupi2 = potentialStaeckel(u0v, numpy.pi/2.0, MWPotential2014, DELTA)
dV = cosh2u0v*potupi2 - (sinh2u0v + sinvx**2.0)*potentialStaeckel(u0v, vx, MWPotential2014, DELTA)
I3V = -E*sinvx**2.0 + pvx**2.0/2.0/DELTA**2.0 + Lz**2.0/2.0/DELTA**2.0/sinvx**2.0 - dV
jz_args = (E, Lz, I3V, DELTA, u0v, cosh2u0v, sinh2u0v, potupi2, MWPotential2014)
def fZ(v):
    return _JzStaeckelIntegrandSquared(v, *jz_args)

# vmin: root of fZ in (something, pi/2). orbit at vx, fZ(vx)>=0; vmin<vx<=pi/2.
# bracket: lo below vx where fZ<0 (expanding down), hi=pi/2 (fZ(pi/2)=0 boundary).
fvx = fZ(vx)
veps = 1e-8
pveps = fZ(vx+veps); mveps = fZ(vx-veps)
at_vturn = (numpy.fabs(pvx) < 1e-7) | (numpy.fabs(fvx) < 1e-10)
# vmin is the lower v turning point; vmax effectively pi/2 (midplane has v=pi/2)
vlo = vx*0.9
for _ in range(80):
    vlo = numpy.where((fZ(vlo) >= 0.0) & (vlo > 1e-9), vlo*0.9, vlo)
vmin_b = bisect_root(fZ, vlo, vx, numpy, xtol=1e-13, maxiter=200)
# at lower v turning point already (pvx~0 and below midplane): vmin=vx
at_vmin = at_vturn & (pveps > 0.0) & (mveps < 0.0)
vmin = numpy.where(at_vmin, vx, vmin_b)

print("=== vmin vs Single ===")
for ii in range(len(R)):
    s = actionAngleStaeckelSingle(R[ii], vR[ii], vT[ii], z[ii], vz[ii], pot=MWPotential2014, delta=DELTA)
    svmin = s.calcVmin()
    print(f"  obj{ii}: vmin {vmin[ii]:.10f} vs {float(numpy.atleast_1d(svmin)[0]):.10f}  d={abs(vmin[ii]-float(numpy.atleast_1d(svmin)[0])):.2e}")

def _r2(a):  # reshape [N] -> [N,1]
    return a[:, None]

def vec_action(xp, lo, hi, args, prefac):
    span = hi - lo
    a2 = tuple(_r2(x) if hasattr(x, "shape") and x.ndim == 1 else x for x in args)
    def integrand(s):
        u = lo[:, None] + span[:, None] * s[None, :]
        sq = _JRStaeckelIntegrandSquared(u, *a2) if args is jr_args else _JzStaeckelIntegrandSquared(u, *a2)
        sq = xp.where(sq > 0.0, sq, xp.zeros_like(sq))
        return xp.sqrt(sq) * span[:, None]
    return prefac * fixed_quad(xp, integrand, 0.0, 1.0, n=GL)

JR = vec_action(numpy, umin, umax, jr_args, 1.0/math.pi*math.sqrt(2.0)*DELTA)
Jz = vec_action(numpy, vmin, numpy.pi/2.0*numpy.ones_like(vmin), jz_args, 2.0/math.pi*math.sqrt(2.0)*DELTA)
# circular -> JR=0; vmin~pi/2 -> Jz=0
JR = numpy.where((umax-umin)/umax < 1e-6, 0.0, JR)
Jz = numpy.where((numpy.pi/2.0 - vmin) < 1e-7, 0.0, Jz)

# c=True reference
aAS = actionAngleStaeckel(pot=MWPotential2014, delta=DELTA, c=True)
jr_c, lz_c, jz_c = aAS(R, vR, vT, z, vz)
print("=== JR / Jz vs c=True ===")
print("JR maxdiff:", float(numpy.amax(numpy.fabs(JR - jr_c))))
print("Jz maxdiff:", float(numpy.amax(numpy.fabs(Jz - jz_c))))
for ii in range(len(R)):
    print(f"  obj{ii}: JR {JR[ii]:.10f} vs {jr_c[ii]:.10f} (d={abs(JR[ii]-jr_c[ii]):.1e})  Jz {Jz[ii]:.10f} vs {jz_c[ii]:.10f} (d={abs(Jz[ii]-jz_c[ii]):.1e})")

# ---- substituted actions: midpoint split + t^2 to regularize sqrt endpoints ----
def vec_action_subst(xp, lo, hi, sqfunc, args, prefac, order=GL):
    mid = 0.5 * (lo + hi)
    a2 = tuple(_r2(x) if hasattr(x, "shape") and getattr(x, "ndim", 0) == 1 else x for x in args)
    limL = xp.sqrt(mid - lo)
    limH = xp.sqrt(hi - mid)
    def integ(lim, base, sign):
        def f(s):
            t = lim[:, None] * s[None, :]
            u = base[:, None] + sign * t**2.0
            sq = sqfunc(u, *a2)
            sq = xp.where(sq > 0.0, sq, xp.zeros_like(sq))
            return xp.sqrt(sq) * 2.0 * t * lim[:, None]
        return f
    lowp = fixed_quad(xp, integ(limL, lo, 1.0), 0.0, 1.0, n=order)
    highp = fixed_quad(xp, integ(limH, hi, -1.0), 0.0, 1.0, n=order)
    return prefac * (lowp + highp)

JR2 = vec_action_subst(numpy, umin, umax, _JRStaeckelIntegrandSquared, jr_args, 1.0/math.pi*math.sqrt(2.0)*DELTA)
Jz2 = vec_action_subst(numpy, vmin, numpy.pi/2.0*numpy.ones_like(vmin), _JzStaeckelIntegrandSquared, jz_args, 2.0/math.pi*math.sqrt(2.0)*DELTA)
JR2 = numpy.where((umax-umin)/umax < 1e-6, 0.0, JR2)
Jz2 = numpy.where((numpy.pi/2.0 - vmin) < 1e-7, 0.0, Jz2)
print("=== substituted JR/Jz vs c=True ===")
print("JR2 maxdiff:", float(numpy.amax(numpy.fabs(JR2 - jr_c))))
print("Jz2 maxdiff:", float(numpy.amax(numpy.fabs(Jz2 - jz_c))))
