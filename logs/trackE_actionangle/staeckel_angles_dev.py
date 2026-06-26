"""Scratch: validate the vectorized Staeckel angle quadrant tree vs c=True."""
import numpy as np, warnings
warnings.filterwarnings("ignore")
import galpy
assert galpy.__file__.startswith("/var/tmp/staeckel-freqs"), galpy.__file__
from galpy.actionAngle import actionAngleStaeckel
from galpy.potential import MWPotential2014
from galpy.actionAngle.actionAngleStaeckel import (
    _staeckel_setup, _staeckel_uminumax, _staeckel_vmin,
    _JRStaeckelIntegrandSquared, _JzStaeckelIntegrandSquared, _backend_fixed_quad,
)

xp = np
sqrt2 = np.sqrt(2.0); pi = np.pi


def angle_partial(Sfunc, sq_args, factor_fn, base, sign, mid, order):
    """t^2-substituted partial integral of factor_fn(u)/sqrt(S) from turning point
    `base` to base+sign*mid^2 (sign=+1 low, -1 high), dead-branch-guarded."""
    a2 = tuple(x[..., None] if getattr(x, "ndim", 0) >= 1 else x for x in sq_args)

    def integ(t01):
        t = mid[..., None] * t01
        u = base[..., None] + sign[..., None] * t ** 2.0
        S = Sfunc(u, *a2)
        Ssafe = xp.where(S > 0.0, S, xp.ones_like(S))
        g = xp.where(S > 0.0, factor_fn(xp, u) / xp.sqrt(Ssafe), xp.zeros_like(S))
        return 2.0 * t * g * mid[..., None]

    return _backend_fixed_quad(xp, integ, 0.0, 1.0, n=order)


def vec_angles(R, vR, vT, z, vz, phi, pot, delta, order=10):
    s = _staeckel_setup(xp, R, vR, vT, z, vz, pot, delta)
    umin, umax, _ = _staeckel_uminumax(xp, s, pot, delta)
    vmin = _staeckel_vmin(xp, s, pot, delta)
    vmin = xp.where((pi / 2 - vmin) < 1e-7, pi / 2 * xp.ones_like(vmin), vmin)
    Lz = s["Lz"]
    jr_args = (s["E"], s["Lz"], s["I3U"], delta, s["u0"], s["sinh2u0"],
               s["v0u"], s["sin2v0u"], s["potu0v0"], pot)
    jz_args = (s["E"], s["Lz"], s["I3V"], delta, s["u0"], s["cosh2u0v"],
               s["sinh2u0v"], s["potupi2"], pot)
    JRsq, JZsq = _JRStaeckelIntegrandSquared, _JzStaeckelIntegrandSquared
    # ---- full-range derivatives (same as _staeckel_freqs) ----
    from galpy.actionAngle.actionAngleStaeckel import _staeckel_deriv_panels as dP
    prefr = delta / pi / sqrt2; prefz = sqrt2 * delta / pi
    pi2 = pi / 2 * xp.ones_like(vmin)
    djrdE = dP(xp, JRsq, jr_args, lambda xp, u: xp.sinh(u) ** 2, umin, umax, order) * prefr
    djrdLz = dP(xp, JRsq, jr_args, lambda xp, u: 1.0 / xp.sinh(u) ** 2, umin, umax, order) * (-Lz / pi / sqrt2 / delta)
    djrdI3 = dP(xp, JRsq, jr_args, lambda xp, u: xp.ones_like(u), umin, umax, order) * (-prefr)
    djzdE = dP(xp, JZsq, jz_args, lambda xp, v: xp.sin(v) ** 2, vmin, pi2, order) * prefz
    djzdLz = dP(xp, JZsq, jz_args, lambda xp, v: 1.0 / xp.sin(v) ** 2, vmin, pi2, order) * (-Lz * sqrt2 / pi / delta)
    djzdI3 = dP(xp, JZsq, jz_args, lambda xp, v: xp.ones_like(v), vmin, pi2, order) * prefz
    detA = djrdE * djzdI3 - djzdE * djrdI3
    Omegar = djzdI3 / detA; Omegaz = -djrdI3 / detA
    Omegaphi = (djrdI3 * djzdLz - djzdI3 * djrdLz) / detA
    dI3dJR = -djzdE / detA; dI3dJz = djrdE / detA
    dI3dLz = -(djrdE * djzdLz - djzdE * djrdLz) / detA
    ux, vx, pux, pvx = s["ux"], s["vx"], s["pux"], s["pvx"]
    # ---- u-branch (4 leaves) ----
    high_u = ux > umin + 0.5 * (umax - umin)
    base_u = xp.where(high_u, umax, umin)
    sign_u = xp.where(high_u, -1.0, 1.0)
    mid_u = xp.sqrt(xp.where(high_u, umax - ux, ux - umin))
    PE = angle_partial(JRsq, jr_args, lambda xp, u: xp.sinh(u) ** 2, base_u, sign_u, mid_u, order)
    PI = angle_partial(JRsq, jr_args, lambda xp, u: xp.ones_like(u), base_u, sign_u, mid_u, order)
    PL = angle_partial(JRsq, jr_args, lambda xp, u: 1.0 / xp.sinh(u) ** 2, base_u, sign_u, mid_u, order)
    pos_u = pux > 0.0
    K_u = xp.where(high_u, pi, xp.where(pos_u, 0.0, 2 * pi))
    s_u = xp.where(high_u, xp.where(pos_u, -1.0, 1.0), xp.where(pos_u, 1.0, -1.0))
    Or1 = K_u * djrdE + s_u * (delta / sqrt2) * PE
    I3r1 = K_u * djrdI3 - s_u * (delta / sqrt2) * PI
    aphi_u = K_u * djrdLz - s_u * (Lz / delta / sqrt2) * PL
    # ---- v-branch (8 leaves) ----
    mid_v_pt = vmin + 0.5 * (pi / 2 - vmin)
    low_v = (vx < mid_v_pt) | (vx > (pi - mid_v_pt))
    above = vx > pi / 2
    base_v = xp.where(low_v, vmin, pi / 2)
    sign_v = xp.where(low_v, 1.0, -1.0)
    mid_v = xp.where(low_v,
                     xp.where(above, xp.sqrt(xp.abs(pi - vx - vmin)), xp.sqrt(xp.abs(vx - vmin))),
                     xp.sqrt(xp.abs(pi / 2 - vx)))
    QE = angle_partial(JZsq, jz_args, lambda xp, v: xp.sin(v) ** 2, base_v, sign_v, mid_v, order)
    QI = angle_partial(JZsq, jz_args, lambda xp, v: xp.ones_like(v), base_v, sign_v, mid_v, order)
    QL = angle_partial(JZsq, jz_args, lambda xp, v: 1.0 / xp.sin(v) ** 2, base_v, sign_v, mid_v, order)
    pos_v = pvx > 0.0
    K_v = xp.where(low_v,
                   xp.where(pos_v, xp.where(above, pi, 0.0), xp.where(above, pi, 2 * pi)),
                   xp.where(pos_v, pi / 2, 1.5 * pi))
    s_v = xp.where(low_v,
                   xp.where(pos_v, xp.where(above, -1.0, 1.0), xp.where(above, 1.0, -1.0)),
                   xp.where(pos_v, xp.where(above, 1.0, -1.0), xp.where(above, -1.0, 1.0)))
    Or2 = K_v * djzdE + s_v * (delta / sqrt2) * QE
    I3r2 = K_v * djzdI3 + s_v * (delta / sqrt2) * QI
    phitmp = K_v * djzdLz - s_v * (Lz / delta / sqrt2) * QL
    # ---- assembly ----
    Or_sum = Or1 + Or2; I3_sum = I3r1 + I3r2
    angler = Omegar * Or_sum + dI3dJR * I3_sum
    anglez = Omegaz * Or_sum + dI3dJz * I3_sum + pi / 2
    anglephi = aphi_u + phitmp + Omegaphi * Or_sum + dI3dLz * I3_sum
    angler = xp.remainder(angler, 2 * pi)
    anglez = xp.remainder(anglez, 2 * pi)
    anglephi = xp.remainder(anglephi + phi, 2 * pi)  # fold in the azimuth (C wrapper)
    return angler, anglephi, anglez


# ---- validate vs c=True ----
aC = actionAngleStaeckel(pot=MWPotential2014, delta=0.45, c=True)
delta = np.array([0.45])
# generic 3D orbits (z!=0, vz!=0) away from degeneracies
cases = [(0.9, 0.1, 0.95, 0.2, 0.08, 1.3), (1.1, -0.2, 0.9, -0.1, 0.2, 0.4), (0.8, 0.25, 0.7, 0.15, 0.12, 2.1),
         (1.3, 0.05, 1.0, 0.3, -0.1, 5.0), (1.0, -0.15, 0.8, -0.25, 0.18, 0.7), (0.7, 0.2, 0.6, 0.1, 0.25, 3.5)]


def wrapdiff(a, b):
    d = (a - b) % (2 * pi)
    return min(d, 2 * pi - d)


import itertools
from galpy.actionAngle.actionAngleStaeckel import _staeckel_setup, _staeckel_uminumax, _staeckel_vmin
# grid spanning pux/pvx signs (vR,vz both +/-), vx </> pi/2 (z +/-), panel hi/lo
grid = list(itertools.product([0.7, 1.0, 1.3], [-0.25, 0.1, 0.3], [0.6, 0.9, 1.1],
                              [-0.3, -0.1, 0.1, 0.3], [-0.2, 0.1, 0.25], [0.4, 3.5]))
maxd = 0.0; n = 0; worst = None; leaves = {}
for c in grid:
    arr = [np.array([v]) for v in c]
    # classify leaf
    s = _staeckel_setup(np, *arr[:5], MWPotential2014, delta)
    umin, umax, _ = _staeckel_uminumax(np, s, MWPotential2014, delta)
    vmin = _staeckel_vmin(np, s, MWPotential2014, delta)
    ux, vx, pux, pvx = float(s["ux"][0]), float(s["vx"][0]), float(s["pux"][0]), float(s["pvx"][0])
    if (umax[0] - umin[0]) / umax[0] < 1e-6 or (pi / 2 - vmin[0]) < 1e-7:
        continue  # skip degenerate
    hu = ux > umin[0] + 0.5 * (umax[0] - umin[0])
    mvp = vmin[0] + 0.5 * (pi / 2 - vmin[0]); lv = (vx < mvp) or (vx > pi - mvp)
    leafu = ("u+" if pux > 0 else "u-") + ("H" if hu else "L")
    leafv = ("v+" if pvx > 0 else "v-") + ("L" if lv else "H") + ("A" if vx > pi / 2 else "B")
    ar, aphi, az = vec_angles(*arr, MWPotential2014, delta)
    rc = aC.actionsFreqsAngles(*c)
    if not np.isfinite(float(rc[6][0])):
        continue
    dr, dphi, dz = wrapdiff(float(ar[0]), float(rc[6][0])), wrapdiff(float(aphi[0]), float(rc[7][0])), wrapdiff(float(az[0]), float(rc[8][0]))
    dd = max(dr, dphi, dz); n += 1
    leaves.setdefault(leafu, []).append(dd); leaves.setdefault(leafv, []).append(dd)
    if dd > maxd: maxd = dd; worst = (c, leafu, leafv, dr, dphi, dz)
print(f"grid n={n}, MAX angle diff vs c=True: {maxd:.2e}")
print("worst:", worst)
print("\nper-leaf max diff (all 4 u-leaves + 8 v-leaves should appear):")
for k in sorted(leaves):
    print(f"  {k}: n={len(leaves[k]):3d}  max={max(leaves[k]):.2e}")
