"""Develop + validate vectorized Staeckel frequencies vs c=True (actionsFreqs)."""
import math
import numpy
import galpy
assert "/var/tmp/staeckel-vec" in galpy.__file__
from galpy.actionAngle import actionAngleStaeckel
from galpy.actionAngle.actionAngleStaeckel import (
    _staeckel_setup, _staeckel_uminumax, _staeckel_vmin,
    _JRStaeckelIntegrandSquared, _JzStaeckelIntegrandSquared,
)
from galpy.potential import MWPotential2014
from galpy.backend.quadrature import fixed_quad

DELTA = 0.45
xp = numpy
R = numpy.array([0.9, 1.1, 0.7, 1.3, 1.0, 0.8])
vR = numpy.array([0.1, -0.2, 0.05, 0.0, 0.3, -0.1])
vT = numpy.array([1.05, 0.9, 1.2, 0.8, 1.0, 1.1])
z = numpy.array([0.2, -0.1, 0.3, 0.05, 0.0, 0.15])
vz = numpy.array([0.08, 0.2, -0.1, 0.15, 0.1, -0.05])
GL = 10

s = _staeckel_setup(xp, R, vR, vT, z, vz, MWPotential2014, DELTA)
umin, umax, unbound = _staeckel_uminumax(xp, s, MWPotential2014, DELTA)
vmin = _staeckel_vmin(xp, s, MWPotential2014, DELTA)


def _r2(a):
    return a[..., None] if getattr(a, "ndim", 0) >= 1 else a


def _deriv_panels(xp, Sfunc, sq_args, factor_fn, lo, hi, order):
    """Low(lo)+High(hi) t^2-substituted panels of factor/sqrt(S), mid=sqrt(.5(hi-lo))."""
    a2 = tuple(_r2(x) for x in sq_args)
    mid = xp.sqrt(0.5 * (hi - lo))

    def panel(base, sign):
        def integ(ss):
            t = mid[..., None] * ss
            u = base[..., None] + sign * t**2.0
            S = Sfunc(u, *a2)
            Ssafe = xp.where(S > 0.0, S, xp.ones_like(S))
            g = xp.where(S > 0.0, factor_fn(xp, u) / xp.sqrt(Ssafe), xp.zeros_like(S))
            return 2.0 * t * g * mid[..., None]
        return fixed_quad(xp, integ, 0.0, 1.0, n=order)

    return panel(lo, 1.0) + panel(hi, -1.0)


jr_args = (s["E"], s["Lz"], s["I3U"], DELTA, s["u0"], s["sinh2u0"],
           s["v0u"], s["sin2v0u"], s["potu0v0"], MWPotential2014)
jz_args = (s["E"], s["Lz"], s["I3V"], DELTA, s["u0"], s["cosh2u0v"],
           s["sinh2u0v"], s["potupi2"], MWPotential2014)
pi2 = numpy.pi / 2.0 * xp.ones_like(vmin)
pref = DELTA / numpy.pi / math.sqrt(2.0)

# 6 derivative integrals (factor functions per the C dJ?d? integrands)
djrdE = _deriv_panels(xp, _JRStaeckelIntegrandSquared, jr_args, lambda xp, u: xp.sinh(u) ** 2.0, umin, umax, GL) * pref
djrdLz = _deriv_panels(xp, _JRStaeckelIntegrandSquared, jr_args, lambda xp, u: 1.0 / xp.sinh(u) ** 2.0, umin, umax, GL) * (-s["Lz"] / numpy.pi / math.sqrt(2.0) / DELTA)
djrdI3 = _deriv_panels(xp, _JRStaeckelIntegrandSquared, jr_args, lambda xp, u: xp.ones_like(u), umin, umax, GL) * (-pref)
prefz = math.sqrt(2.0) * DELTA / numpy.pi  # NB: djz prefactors are 2x djr's, +I3
djzdE = _deriv_panels(xp, _JzStaeckelIntegrandSquared, jz_args, lambda xp, v: xp.sin(v) ** 2.0, vmin, pi2, GL) * prefz
djzdLz = _deriv_panels(xp, _JzStaeckelIntegrandSquared, jz_args, lambda xp, v: 1.0 / xp.sin(v) ** 2.0, vmin, pi2, GL) * (-s["Lz"] * math.sqrt(2.0) / numpy.pi / DELTA)
djzdI3 = _deriv_panels(xp, _JzStaeckelIntegrandSquared, jz_args, lambda xp, v: xp.ones_like(v), vmin, pi2, GL) * prefz

detA = djrdE * djzdI3 - djzdE * djrdI3
Omegar = djzdI3 / detA
Omegaz = -djrdI3 / detA
Omegaphi = (djrdI3 * djzdLz - djzdI3 * djrdLz) / detA

aT = actionAngleStaeckel(pot=MWPotential2014, delta=DELTA, c=True)
jr_c, lz_c, jz_c, Or_c, Op_c, Oz_c = aT.actionsFreqs(R, vR, vT, z, vz)
print("Omegar maxdiff:", float(numpy.amax(numpy.fabs(Omegar - Or_c))))
print("Omegaphi maxdiff:", float(numpy.amax(numpy.fabs(Omegaphi - Op_c))))
print("Omegaz maxdiff:", float(numpy.amax(numpy.fabs(Omegaz - Oz_c))))
for i in range(len(R)):
    print(f"  obj{i}: Or {Omegar[i]:.8f} vs {Or_c[i]:.8f}  Oz {Omegaz[i]:.8f} vs {Oz_c[i]:.8f}  Op {Omegaphi[i]:.8f} vs {Op_c[i]:.8f}")
print("MY djr (obj0):", float(djrdE[0]), float(djrdLz[0]), float(djrdI3[0]))
print("MY djz (obj0):", float(djzdE[0]), float(djzdLz[0]), float(djzdI3[0]))
