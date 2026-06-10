"""G10 measure: largest-Lyapunov-exponent running estimates (Orbit.lyapunov).

Standalone measure script (cross-environment pattern, like g8): run it against
the feat/pvar-lyapunov galpy worktree, NOT inside the figure harness:

    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
    PYTHONPATH=/var/tmp/pvar-lyap \
    python lyapunov_measure.py

Writes ../data/lyapunov.json (machine/worktree-stamped); the plot-only make.py
is byte-stable from that JSON.

What is measured (Benettin et al. 1980 renormalization on top of the C
variational equations, Orbit.lyapunov defaults: dop853_c, renorm_every=10):

 - Henon-Heiles at E=1/8 (amp=1), t to 3e4 (output dt=2 -> renormalization
   every 20 t.u., the validated regression-test configuration): the documented
   CHAOTIC orbit 'F' and REGULAR orbit 'E' of Skokos et al. (2002,
   arXiv:nlin/0210053, Sect 3.2). Literature value for the chaotic sea at
   E=1/8: lambda ~ 0.04-0.08 (Benettin, Galgani & Strelcyn 1976, Phys. Rev. A
   14, 2338, Fig. 4; Skokos 2010, Lect. Notes Phys. 790, 63, Fig. 2).
 - A regular 3D Sun-like orbit in MWPotential2014, t to 3e3 (output dt=1 ->
   renormalization every 10 t.u.): lambda(t) must decay ~ln(t)/t.

The lambda(t) arrays are thinned (every 5th / 2nd output time) for storage;
the thinned arrays are exactly what make.py plots.
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
from galpy.potential import HenonHeilesPotential, MWPotential2014  # noqa: E402

# Henon-Heiles, galpy planar coords (R,vR,vT,phi):
#  orbit F (chaotic): Cartesian (x,y,px,py)=(0,-0.016,0.49974,0)
#  orbit E (regular): Cartesian (x,y,px,py)=(0,0.55,0.2417,0)
IC_HH_CHAOTIC = [0.016, 0.0, 0.49974, -numpy.pi / 2.0]
IC_HH_REGULAR = [0.55, 0.0, -0.2417, numpy.pi / 2.0]
T1_HH, NT_HH, THIN_HH = 30000.0, 15001, 5
IC_MW = [1.0, 0.0, 1.0, 0.0125, 0.036, 0.0]  # Sun-like (R,vR,vT,z,vz,phi)
T1_MW, NT_MW, THIN_MW = 3000.0, 3001, 2
METHOD = "dop853_c"
RENORM_EVERY = 10


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


def main():
    out = {"env": env_stamp()}
    out["config"] = {
        "method": METHOD,
        "renorm_every": RENORM_EVERY,
        "rtol": "default",
        "atol": "default",
        "henon_heiles": {
            "amp": 1.0,
            "ic_chaotic": IC_HH_CHAOTIC,
            "ic_regular": IC_HH_REGULAR,
            "times": {"t0": 0.0, "t1": T1_HH, "nt": NT_HH},
            "thin": THIN_HH,
            "literature_band": [0.04, 0.08],
        },
        "mw2014": {
            "ic": IC_MW,
            "times": {"t0": 0.0, "t1": T1_MW, "nt": NT_MW},
            "thin": THIN_MW,
        },
    }

    hh = HenonHeilesPotential(amp=1.0)
    ts = numpy.linspace(0.0, T1_HH, NT_HH)
    oF = Orbit(IC_HH_CHAOTIC)
    oE = Orbit(IC_HH_REGULAR)
    EF, EE = float(oF.E(pot=hh)), float(oE.E(pot=hh))
    assert abs(EF - 0.125) < 1e-4 and abs(EE - 0.125) < 1e-4, (
        f"Henon-Heiles test orbits not at E=1/8: {EF}, {EE}"
    )
    print("[HH] chaotic orbit F ...", flush=True)
    lamF = oF.lyapunov(ts, pot=hh, method=METHOD, renorm_every=RENORM_EVERY)
    print(f"     lambda(t=1.5e4)={lamF[NT_HH // 2]:.5f} "
          f"lambda(t=3e4)={lamF[-1]:.5f}", flush=True)
    print("[HH] regular orbit E ...", flush=True)
    lamE = oE.lyapunov(ts, pot=hh, method=METHOD, renorm_every=RENORM_EVERY)
    print(f"     lambda(t=3e4)={lamE[-1]:.3e} "
          f"[ln(t)/t={numpy.log(T1_HH) / T1_HH:.3e}]", flush=True)
    out["henon_heiles"] = {
        "E_chaotic": EF,
        "E_regular": EE,
        "t": ts[::THIN_HH].tolist(),
        "lam_chaotic": lamF[::THIN_HH].tolist(),
        "lam_regular": lamE[::THIN_HH].tolist(),
        "lam_chaotic_half": float(lamF[NT_HH // 2]),
        "lam_chaotic_final": float(lamF[-1]),
        "lam_regular_final": float(lamE[-1]),
    }

    print("[MW] Sun-like 3D orbit in MWPotential2014 ...", flush=True)
    ts3 = numpy.linspace(0.0, T1_MW, NT_MW)
    o3 = Orbit(IC_MW)
    lam3 = o3.lyapunov(ts3, pot=MWPotential2014, method=METHOD,
                       renorm_every=RENORM_EVERY)
    print(f"     lambda(t=3e3)={lam3[-1]:.3e} "
          f"[ln(t)/t={numpy.log(T1_MW) / T1_MW:.3e}]", flush=True)
    out["mw2014"] = {
        "t": ts3[::THIN_MW].tolist(),
        "lam": lam3[::THIN_MW].tolist(),
        "lam_final": float(lam3[-1]),
    }

    dest = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data", "lyapunov.json"
    )
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {os.path.normpath(dest)}")


if __name__ == "__main__":
    main()
