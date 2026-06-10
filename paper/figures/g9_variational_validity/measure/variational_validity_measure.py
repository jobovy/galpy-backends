"""G9 measure: validity diagnostics of the 3D/planar variational equations.

Standalone measure script (cross-environment pattern, like g8): run it against
the feat/pvar-lyapunov galpy worktree, NOT inside the figure harness:

    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
    PYTHONPATH=/var/tmp/pvar-lyap \
    python variational_validity_measure.py

Writes ../data/variational_validity.json (machine/worktree-stamped); the
plot-only make.py is byte-stable from that JSON.

What is measured (MWPotential2014, C dop853_c at rtol=atol=1e-12 throughout):

 (a/b) The 6x6 (3D) and 4x4 (planar) state-transition matrices M(t) built by
       integrating the canonical rectangular basis deviations with
       Orbit.integrate_dxdv(rectIn=rectOut=True); the Liouville invariant
       |det M(t) - 1| and the symplecticity defect ||M^T Omega M - Omega||_max
       in canonical rectangular coordinates, as functions of time.
 (c)   Finite-difference-of-the-flow ground truth at t=10: max over the 6
       canonical directions e_i of ||M(t) e_i - [x(t;x0+eps e_i)-x(t;x0)]/eps||_inf
       (forward) and the central-difference analog, as a function of eps.
 (d)   2D-reduction bridge: for a planar IC, the (x,y,vx,vy) block of the 3D
       6x6 STM vs the independently-implemented planar 4x4 STM, plus the
       out-of-plane leakage |dz|,|dvz|, as functions of time.
"""

import json
import os
import subprocess
import sys

import numpy
import scipy

import galpy

EXPECTED_WORKTREE = os.environ.get("GALPY_WORKTREE", "/var/tmp/pvar-lyap")
assert galpy.__file__.startswith(EXPECTED_WORKTREE), (
    f"galpy imported from {galpy.__file__}, not from {EXPECTED_WORKTREE}; "
    "set PYTHONPATH to the worktree (editable-install shadowing?)"
)

from galpy.orbit import Orbit  # noqa: E402
from galpy.potential import MWPotential2014  # noqa: E402
from galpy.util import coords  # noqa: E402

METHOD = "dop853_c"
RTOL = ATOL = 1e-12
IC3 = [1.0, 0.1, 1.1, 0.05, 0.08, 0.2]  # generic 3D (R,vR,vT,z,vz,phi)
R0, VR0, VT0, PHI0 = 1.0, 0.1, 1.1, 0.2  # planar IC (same in-plane numbers)
T1, NT = 100.0, 1001
T_FD, NT_FD = 10.0, 101


def env_stamp():
    wt = os.path.dirname(os.path.dirname(os.path.abspath(galpy.__file__)))

    def git(*args):
        return (
            subprocess.check_output(["git", "-C", wt] + list(args))
            .decode()
            .strip()
        )

    return {
        "galpy_worktree": wt,
        "galpy_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "galpy_commit": git("rev-parse", "--short", "HEAD"),
        "galpy_dirty": bool(git("status", "--porcelain")),
        "galpy_version": galpy.__version__,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "python": sys.version.split()[0],
    }


def stm_series(ic, times, pot):
    """(nt, pd, pd) STM time series from the pd canonical rect deviations."""
    pd = len(ic) if len(ic) in (4, 6) else None
    assert pd is not None
    cols = []
    for ii in range(pd):
        o = Orbit(ic)
        o.integrate_dxdv(
            numpy.eye(pd)[ii],
            times,
            pot,
            method=METHOD,
            rectIn=True,
            rectOut=True,
            rtol=RTOL,
            atol=ATOL,
            progressbar=False,
        )
        cols.append(o.getOrbit_dxdv())
    return numpy.array(cols).transpose(1, 2, 0)  # columns = propagated e_i


def omega(ndim):
    Om = numpy.zeros((2 * ndim, 2 * ndim))
    Om[:ndim, ndim:] = numpy.eye(ndim)
    Om[ndim:, :ndim] = -numpy.eye(ndim)
    return Om


def det_symp(Mt, ndim):
    Om = omega(ndim)
    det_err = numpy.array(
        [numpy.fabs(numpy.linalg.det(Mt[k]) - 1.0) for k in range(len(Mt))]
    )
    symp = numpy.array(
        [
            numpy.amax(numpy.fabs(Mt[k].T @ Om @ Mt[k] - Om))
            for k in range(len(Mt))
        ]
    )
    return det_err, symp


def orbit_rect_3d(o, ts):
    return numpy.array(
        [o.x(ts), o.y(ts), o.z(ts), o.vx(ts), o.vy(ts), o.vz(ts)]
    ).T


def integrate_rect_perturbed(rect_ic, times, pot):
    """Integrate a 3D orbit from a rectangular IC; return rect phase points."""
    Rp, phip, Zp = coords.rect_to_cyl(rect_ic[0], rect_ic[1], rect_ic[2])
    vRp, vTp, vzp = coords.rect_to_cyl_vec(
        rect_ic[3], rect_ic[4], rect_ic[5], rect_ic[0], rect_ic[1], rect_ic[2]
    )
    o = Orbit([Rp, vRp, vTp, Zp, vzp, phip])
    o.integrate(times, pot, method=METHOD, progressbar=False)
    return orbit_rect_3d(o, times)


def main():
    out = {"env": env_stamp()}
    out["config"] = {
        "potential": "MWPotential2014",
        "method": METHOD,
        "rtol": RTOL,
        "atol": ATOL,
        "ic_3d": IC3,
        "ic_planar": [R0, VR0, VT0, PHI0],
        "ic_3d_planar": [R0, VR0, VT0, 0.0, 0.0, PHI0],
        "times": {"t0": 0.0, "t1": T1, "nt": NT},
        "t_fd": T_FD,
        "nt_fd": NT_FD,
    }
    times = numpy.linspace(0.0, T1, NT)

    # ---- panels (a)+(b): det/symplecticity vs t, 3D and planar ----
    print("[a/b] 3D STM series ...", flush=True)
    M3 = stm_series(IC3, times, MWPotential2014)
    det3, symp3 = det_symp(M3, 3)
    print("[a/b] planar STM series ...", flush=True)
    M2 = stm_series([R0, VR0, VT0, PHI0], times, MWPotential2014)
    det2, symp2 = det_symp(M2, 2)
    out["panel_ab"] = {
        "t": times.tolist(),
        "abs_det_err_3d": det3.tolist(),
        "symp_err_3d": symp3.tolist(),
        "abs_det_err_2d": det2.tolist(),
        "symp_err_2d": symp2.tolist(),
        "max_abs_det_err_3d": float(numpy.amax(det3)),
        "max_symp_err_3d": float(numpy.amax(symp3)),
        "max_abs_det_err_2d": float(numpy.amax(det2)),
        "max_symp_err_2d": float(numpy.amax(symp2)),
    }
    print(
        f"      max |det M-1|: 3D {numpy.amax(det3):.3e}, 2D {numpy.amax(det2):.3e}; "
        f"max symp err: 3D {numpy.amax(symp3):.3e}, 2D {numpy.amax(symp2):.3e}",
        flush=True,
    )

    # ---- panel (c): FD of the flow vs eps at t=T_FD ----
    print("[c]   FD-of-the-flow vs eps ...", flush=True)
    times_fd = numpy.linspace(0.0, T_FD, NT_FD)
    Mfd = stm_series(IC3, times_fd, MWPotential2014)[-1]  # 6x6 at t=T_FD
    obase = Orbit(IC3)
    obase.integrate(times_fd, MWPotential2014, method=METHOD, progressbar=False)
    base_rect = orbit_rect_3d(obase, times_fd)
    eps_grid = 10.0 ** numpy.arange(-2.0, -11.01, -0.5)
    resid_fwd, resid_cen = [], []
    for eps in eps_grid:
        rf, rc = 0.0, 0.0
        for ii in range(6):
            icp = base_rect[0].copy()
            icp[ii] += eps
            icm = base_rect[0].copy()
            icm[ii] -= eps
            rp = integrate_rect_perturbed(icp, times_fd, MWPotential2014)[-1]
            rm = integrate_rect_perturbed(icm, times_fd, MWPotential2014)[-1]
            fd_f = (rp - base_rect[-1]) / eps
            fd_c = (rp - rm) / 2.0 / eps
            rf = max(rf, float(numpy.amax(numpy.fabs(Mfd[:, ii] - fd_f))))
            rc = max(rc, float(numpy.amax(numpy.fabs(Mfd[:, ii] - fd_c))))
        resid_fwd.append(rf)
        resid_cen.append(rc)
        print(f"      eps={eps:.1e}: fwd {rf:.3e} cen {rc:.3e}", flush=True)
    out["panel_c"] = {
        "eps": eps_grid.tolist(),
        "resid_forward": resid_fwd,
        "resid_central": resid_cen,
        "min_resid_forward": float(numpy.amin(resid_fwd)),
        "min_resid_central": float(numpy.amin(resid_cen)),
        "stm_norm_max": float(numpy.amax(numpy.fabs(Mfd))),
    }

    # ---- panel (d): 2D-reduction bridge vs t ----
    print("[d]   2D-reduction bridge ...", flush=True)
    M3p = stm_series([R0, VR0, VT0, 0.0, 0.0, PHI0], times, MWPotential2014)
    sub = numpy.ix_([0, 1, 3, 4], [0, 1, 3, 4])
    bridge = numpy.array(
        [numpy.amax(numpy.fabs(M3p[k][sub] - M2[k])) for k in range(len(times))]
    )
    # in-plane deviations must not leak out of the plane (z-symmetric pot)
    leak = numpy.array(
        [
            numpy.amax(numpy.fabs(M3p[k][numpy.ix_([2, 5], [0, 1, 3, 4])]))
            for k in range(len(times))
        ]
    )
    out["panel_d"] = {
        "t": times.tolist(),
        "bridge_err": bridge.tolist(),
        "leak_err": leak.tolist(),
        "max_bridge_err": float(numpy.amax(bridge)),
        "max_leak_err": float(numpy.amax(leak)),
    }
    print(
        f"      max bridge err {numpy.amax(bridge):.3e}, max leak {numpy.amax(leak):.3e}",
        flush=True,
    )

    dest = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "data",
        "variational_validity.json",
    )
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {os.path.normpath(dest)}")


if __name__ == "__main__":
    main()
