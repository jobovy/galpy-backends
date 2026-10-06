"""G11 measure: the dissipative phase-volume law det M(t) = exp(int tr(dF/dv) dt').

Standalone measure script (cross-environment pattern, like g8): each friction
force is measured against the galpy worktree whose C extension implements its
rectangular Jacobian, NOT inside the figure harness:

    # Chandrasekhar dynamical friction (feat/pvar-dissip-struct build)
    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" GALPY_WORKTREE=/var/tmp/pvar-dissip \
    PYTHONPATH=/var/tmp/pvar-dissip python dissipative_measure.py cdf

    # FDM dynamical friction (feat/pvar-tail-fdm build)
    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" GALPY_WORKTREE=/var/tmp/pvt-fdm \
    PYTHONPATH=/var/tmp/pvt-fdm python dissipative_measure.py fdm

Writes ../data/dissipative_{cdf,fdm}.json (machine/worktree-stamped); the
plot-only make.py is byte-stable from those JSONs.

What is measured: a decaying orbit in MWPotential2014 + dynamical friction
(GMs=0.008, rhm=0, identical for both forces; the FDM force additionally has
mhbar=10, putting it in the FDM dispersion regime where quantum pressure
suppresses the friction).  The 6x6 STM M(t) is built from the 6 canonical
rectangular deviations via Orbit.integrate_dxdv (dopr54_c, rtol=atol=1e-12);
det M(t) is compared against the Abel-Jacobi-Liouville prediction
exp(int_0^t tr(dF/dv) dt'), with tr(dF/dv) evaluated along the orbit from the
velocity Jacobian of the friction force by central FD of the Python forces (the
Jacobian the C variational integrator uses) and integrated with the
trapezoidal rule on the fine output grid.
"""

import ctypes
import json
import os
import subprocess
import sys

import numpy
import scipy
import scipy.special
from numpy.ctypeslib import ndpointer

import galpy

MODE = sys.argv[1] if len(sys.argv) > 1 else "cdf"
assert MODE in ("cdf", "fdm"), f"unknown mode {MODE!r} (use 'cdf' or 'fdm')"
EXPECTED_WORKTREE = os.environ.get(
    "GALPY_WORKTREE", "/var/tmp/pvar-dissip" if MODE == "cdf" else "/var/tmp/pvt-fdm"
)
assert galpy.__file__.startswith(EXPECTED_WORKTREE), (
    f"galpy imported from {galpy.__file__}, not from {EXPECTED_WORKTREE}; "
    "set PYTHONPATH to the worktree (editable-install shadowing?)"
)

from galpy.orbit import Orbit  # noqa: E402
from galpy.orbit.integrateFullOrbit import (  # noqa: E402
    _lib,
    _parse_pot,
    _prep_tfuncs,
)
from galpy.potential import MWPotential2014  # noqa: E402

GMS = 0.008
IC = [1.0, 0.1, 1.1, 0.05, 0.08, 0.2]
T1, NT, THIN = 10.0, 4001, 4
METHOD = "dopr54_c"
RTOL = ATOL = 1e-12
MHBAR = 10.0  # FDM only


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


def fd_velocity_jacobian(pot, q, t=0.0, eps=1e-6):
    """(dF/dx, dF/dv) (3x3 each) of the summed rectangular force by CENTRAL
    FINITE DIFFERENCES of the PYTHON forces (evaluateRforces/zforces/
    phitorques with v=..., assembled to Cartesian). Independent of the C
    variational code, so comparing det M(t) from the C STM against
    exp(int tr(dF/dv) dt) built from this trace cross-validates the two
    implementations. (The C single-point evaluator this script originally
    used was removed from galpy per maintainer review -- C internals are not
    exposed directly.)"""
    from galpy.potential import (
        evaluatephitorques,
        evaluateRforces,
        evaluatezforces,
    )

    def cart_force(qq):
        x, y, z, vx, vy, vz = qq
        R = numpy.sqrt(x * x + y * y)
        phi = numpy.arctan2(y, x)
        cp, sp_ = numpy.cos(phi), numpy.sin(phi)
        vR = vx * cp + vy * sp_
        vT = -vx * sp_ + vy * cp
        kw = dict(phi=phi, t=t, v=[vR, vT, vz], use_physical=False)
        FR = evaluateRforces(pot, R, z, **kw)
        Fp = evaluatephitorques(pot, R, z, **kw)
        Fz = evaluatezforces(pot, R, z, **kw)
        return numpy.array(
            [cp * FR - sp_ / R * Fp, sp_ * FR + cp / R * Fp, Fz]
        )

    jac_x = numpy.zeros((3, 3))
    jac_v = numpy.zeros((3, 3))
    for j in range(3):
        dq = numpy.zeros(6)
        dq[j] = eps
        jac_x[:, j] = (cart_force(q + dq) - cart_force(q - dq)) / (2.0 * eps)
        dq = numpy.zeros(6)
        dq[3 + j] = eps
        jac_v[:, j] = (cart_force(q + dq) - cart_force(q - dq)) / (2.0 * eps)
    return jac_x, jac_v


def fdm_regime(fdf, q):
    """Replicate the C FDM friction-coefficient branch logic (cf. the
    pvar-tail-fdm test suite) at the rectangular point q, so we can assert
    that the orbit genuinely exercises the FDM dispersion regime."""
    sp = scipy.special
    r = numpy.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2)
    v = numpy.sqrt(q[3] ** 2 + q[4] ** 2 + q[5] ** 2)
    krC = 2.0 * fdf._mhbar * v * r
    rcl = numpy.clip(r, fdf._sigmar_rs_4interp[0], fdf._sigmar_rs_4interp[-1])
    sr = fdf.sigmar(rcl)
    M = v / sr
    X = v / numpy.sqrt(2.0) / sr
    Xf = sp.erf(X) - 2.0 * X / numpy.sqrt(numpy.pi) * numpy.exp(-(X**2))
    if krC < M:
        Cfdm = (
            numpy.euler_gamma
            + numpy.log(krC)
            - sp.sici(krC)[1]
            + numpy.sin(krC) / krC
            - 1.0
        )
        regime = "zerovel"
    elif krC > 4.0 * M:
        Cfdm = numpy.log(krC / M) * Xf
        regime = "dispersion"
    else:
        Czv = (
            numpy.euler_gamma + numpy.log(M) - sp.sici(M)[1] + numpy.sin(M) / M - 1.0
        )
        mu = (2.0 * M - 0.5 * krC) / (1.5 * M)
        Cfdm = mu * Czv + (1.0 - mu) * numpy.log(4.0) * Xf
        regime = "intermediate"
    suppression = float(Cfdm / (fdf.lnLambda(r, v) * Xf))
    if suppression >= 1.0:
        regime += "+cdmcutoff"
    return regime, suppression


def main():
    if MODE == "cdf":
        from galpy.potential import ChandrasekharDynamicalFrictionForce

        frict = ChandrasekharDynamicalFrictionForce(
            GMs=GMS, rhm=0.0, dens=MWPotential2014, maxr=10.0
        )
        label = "ChandrasekharDynamicalFrictionForce"
    else:
        from galpy.potential import FDMDynamicalFrictionForce

        frict = FDMDynamicalFrictionForce(
            GMs=GMS, rhm=0.0, dens=MWPotential2014, maxr=10.0
        )
        frict._mhbar = MHBAR  # the exact quantity the C parser ships
        label = "FDMDynamicalFrictionForce"
    assert frict.hasC_dxdv3d, f"{label} must have its C rectangular Jacobian"
    pot = MWPotential2014 + frict

    out = {"env": env_stamp()}
    out["config"] = {
        "force": label,
        "GMs": GMS,
        "rhm": 0.0,
        "maxr": 10.0,
        "dens": "MWPotential2014",
        "mhbar": MHBAR if MODE == "fdm" else None,
        "potential": "MWPotential2014 + friction",
        "ic": IC,
        "times": {"t0": 0.0, "t1": T1, "nt": NT},
        "thin": THIN,
        "method": METHOD,
        "rtol": RTOL,
        "atol": ATOL,
    }

    times = numpy.linspace(0.0, T1, NT)
    print(f"[{MODE}] base orbit ...", flush=True)
    ob = Orbit(IC)
    ob.integrate(times, pot, method=METHOD, progressbar=False)
    rect = numpy.array(
        [ob.x(times), ob.y(times), ob.z(times), ob.vx(times), ob.vy(times), ob.vz(times)]
    ).T
    r_t = numpy.sqrt(numpy.sum(rect[:, :3] ** 2, axis=1))
    print(f"      r: {r_t[0]:.3f} -> {r_t[-1]:.3f}", flush=True)

    if MODE == "fdm":
        # the orbit must stay on the FDM dispersion branch with active
        # quantum-pressure suppression for the comparison to mean anything
        regs = [fdm_regime(frict, rect[k]) for k in range(0, NT, THIN)]
        regimes = sorted({r for r, _ in regs})
        supp = [s for _, s in regs]
        assert regimes == ["dispersion"], f"FDM orbit off dispersion branch: {regimes}"
        out["fdm_regime"] = {
            "regimes": regimes,
            "suppression_min": float(numpy.amin(supp)),
            "suppression_max": float(numpy.amax(supp)),
        }
        print(
            f"      FDM dispersion regime, Cfdm/Ccdm in "
            f"[{numpy.amin(supp):.2f}, {numpy.amax(supp):.2f}]",
            flush=True,
        )

    print(f"[{MODE}] 6 canonical deviation integrations ...", flush=True)
    cols = []
    for ii in range(6):
        o = Orbit(IC)
        o.integrate_dxdv(
            numpy.eye(6)[ii],
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
    Mt = numpy.array(cols).transpose(1, 2, 0)
    detM = numpy.array([numpy.linalg.det(Mt[k]) for k in range(NT)])

    print(f"[{MODE}] tr(dF/dv) along the orbit from the Python forces (central FD) ...", flush=True)
    trJv = numpy.array(
        [
            numpy.trace(fd_velocity_jacobian(frict, rect[k], t=times[k])[1])
            for k in range(NT)
        ]
    )
    integral = numpy.concatenate(
        ([0.0], numpy.cumsum(0.5 * (trJv[1:] + trJv[:-1]) * numpy.diff(times)))
    )
    pred = numpy.exp(integral)
    relerr = numpy.fabs(detM - pred) / pred
    print(
        f"      det M(t1)={detM[-1]:.6e} pred={pred[-1]:.6e} "
        f"max relerr={numpy.amax(relerr):.3e}",
        flush=True,
    )

    out["result"] = {
        "t": times[::THIN].tolist(),
        "detM": detM[::THIN].tolist(),
        "pred": pred[::THIN].tolist(),
        "relerr": relerr[::THIN].tolist(),
        "r": r_t[::THIN].tolist(),
        "detM_final": float(detM[-1]),
        "pred_final": float(pred[-1]),
        "max_relerr": float(numpy.amax(relerr)),  # on the full (unthinned) grid
        "r_final": float(r_t[-1]),
    }

    dest = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "data",
        f"dissipative_{MODE}.json",
    )
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {os.path.normpath(dest)}")


if __name__ == "__main__":
    main()
