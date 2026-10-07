"""gA1 measure: d(J_R, J_z)/dq of the Staeckel actions w.r.t. the halo
flattening q, by jax.grad through galpy's pure-Python Staeckel code
(actionAngleStaeckel(..., c=False)), validated against central finite
differences of the numpy path.

Standalone measure script (pattern A, like g10): run it against the galpy
worktree, NOT inside the figure harness:

    CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu PYTHONPATH=/var/tmp/fbp \
    systemd-run --user --scope -p MemoryMax=32G \
    python measure.py

Writes ../data/gA1_staeckel_param_sensitivity.json (worktree-stamped: galpy git
sha, jax version); the plot-only make.py is byte-stable from that JSON.

What is measured, all in float64 on the CPU:

 - sweep: LogarithmicHaloPotential(normalize=1, q) for q in [0.7, 1.1] (30
   points), actionAngleStaeckel(pot, delta=0.45, c=False) rebuilt from each q;
   J_R, J_z of one Sun-like orbit (R=1, vR=0.1, vT=1.1, z=0.1, vz=0.05) and
   dJ/dq by jax.value_and_grad (the potential AND the actionAngle instance are
   constructed inside the differentiated function, so q flows through the
   normalisation, the Staeckel setup, the turning-point roots and the action
   quadrature). Central finite differences of the numpy path at every 5th q
   (the potential and aA rebuilt from q+/-h) are the markers.
 - vcurve: at q=0.9, |AD-FD|/|FD| for dJ_R/dq and dJ_z/dq as a function of the
   FD step h = 1e-2 ... 1e-7 (numpy FD, everything rebuilt from q+/-h).
 - checks: numpy-vs-jax forward parity along the sweep; jax.jit of the
   value_and_grad vs eager at one q; the frequency gradients dOmega/dq at q=0.9
   vs FD (not plotted: the numpy frequency values carry ~1e-11 noise that FD
   amplifies to ~1e-6, so they cannot be validated as tightly as the actions).
"""

import json
import os
import subprocess
import sys
import time

import numpy

import galpy

EXPECTED_WORKTREE = os.environ.get("GALPY_WORKTREE", "/var/tmp/fbp")
assert galpy.__file__.startswith(EXPECTED_WORKTREE), (
    f"galpy imported from {galpy.__file__}, not from {EXPECTED_WORKTREE}; "
    "set PYTHONPATH to the worktree (editable-install shadowing?)"
)

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from galpy.actionAngle import actionAngleStaeckel  # noqa: E402
from galpy.potential import LogarithmicHaloPotential  # noqa: E402

_CPU = jax.devices("cpu")[0]
OUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data",
    "gA1_staeckel_param_sensitivity.json",
)

DELTA, ORDER = 0.45, 10
ORBIT = dict(R=1.0, vR=0.1, vT=1.1, z=0.1, vz=0.05)  # phi=0, natural units
Q = numpy.linspace(0.7, 1.1, 30)
FD_EVERY = 5  # FD markers at every 5th sweep point
H_MARK = 1e-5  # FD step for the sweep markers (near the V floor)
Q0 = 0.9
H_V = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
IDX = {"JR": 0, "Jz": 2}  # (jr, Lz, jz) slots
FIDX = {"OR": 3, "Oz": 5}  # actionsFreqs (jr, Lz, jz, Or, Op, Oz) slots


def env_stamp():
    wt = os.path.dirname(os.path.dirname(os.path.abspath(galpy.__file__)))

    def git(*args):
        return subprocess.check_output(["git", "-C", wt] + list(args)).decode().strip()

    return {
        "galpy_worktree": wt,
        "galpy_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "galpy_commit": git("rev-parse", "--short", "HEAD"),
        "galpy_dirty": bool(git("status", "--porcelain")),
        "galpy_version": galpy.__version__,
        "jax": jax.__version__,
        "jax_x64": bool(jax.config.jax_enable_x64),
        "jax_devices": [str(d) for d in jax.devices()],
        "numpy": numpy.__version__,
        "python": sys.version.split()[0],
    }


def make_aA(q):
    # EVERYTHING rebuilt from q: the (normalised) potential and the aA instance
    return actionAngleStaeckel(
        pot=LogarithmicHaloPotential(normalize=1.0, q=q), delta=DELTA, c=False,
        order=ORDER,
    )


def J_np(q, idx):
    out = make_aA(q)(*[numpy.array([v]) for v in ORBIT.values()])
    return float(out[idx][0])


def Jf_np(q, idx):
    out = make_aA(q).actionsFreqs(*[numpy.array([v]) for v in ORBIT.values()])
    return float(out[idx][0])


def J_jax(q, idx):
    return make_aA(q)(*[jnp.asarray(v) for v in ORBIT.values()])[idx]


def Jf_jax(q, idx):
    return make_aA(q).actionsFreqs(*[jnp.asarray(v) for v in ORBIT.values()])[idx]


def fd_np(fn, q, h):
    return (fn(q + h) - fn(q - h)) / (2.0 * h)


def main():
    t_start = time.time()
    with jax.default_device(_CPU):
        vg = {k: jax.value_and_grad(lambda q, i=i: J_jax(q, i)) for k, i in IDX.items()}

        # ---- sweep ---------------------------------------------------------
        sweep = {"q": Q.tolist()}
        for k, i in IDX.items():
            vals, grads, vnp = [], [], []
            for q in Q:
                v, g = vg[k](jnp.asarray(q))
                vals.append(float(v))
                grads.append(float(g))
                vnp.append(J_np(float(q), i))
                print(f"[sweep] {k} q={q:.4f} J={vals[-1]:.10g} dJ/dq={grads[-1]:.10g}",
                      flush=True)
            sweep[k] = vals
            sweep[f"{k}_numpy"] = vnp
            sweep[f"d{k}_dq"] = grads
        imark = list(range(0, len(Q), FD_EVERY))
        sweep["fd_idx"] = imark
        sweep["fd_h"] = H_MARK
        for k, i in IDX.items():
            fds = [fd_np(lambda q, i=i: J_np(q, i), float(Q[j]), H_MARK) for j in imark]
            sweep[f"d{k}_dq_fd"] = fds
            ad = numpy.asarray(sweep[f"d{k}_dq"])[imark]
            sweep[f"d{k}_dq_fd_maxrel"] = float(
                numpy.max(numpy.abs(ad - numpy.asarray(fds)) / numpy.abs(fds))
            )
        sweep["forward_parity_maxrel"] = float(max(
            numpy.max(numpy.abs(numpy.asarray(sweep[k]) - numpy.asarray(sweep[f"{k}_numpy"]))
                      / numpy.abs(numpy.asarray(sweep[f"{k}_numpy"])))
            for k in IDX
        ))

        # ---- V curve at q=Q0 -------------------------------------------------
        vcurve = {"q0": Q0, "h": H_V}
        for k, i in IDX.items():
            v, g = vg[k](jnp.asarray(Q0))
            vcurve[f"{k}_ad"] = float(g)
            vcurve[f"{k}_value"] = float(v)
            fds = [fd_np(lambda q, i=i: J_np(q, i), Q0, h) for h in H_V]
            vcurve[f"{k}_fd"] = fds
            vcurve[f"{k}_relerr"] = [abs(float(g) - f) / abs(f) for f in fds]
            jbest = int(numpy.argmin(vcurve[f"{k}_relerr"]))
            vcurve[f"{k}_best_h"] = H_V[jbest]
            vcurve[f"{k}_best_relerr"] = vcurve[f"{k}_relerr"][jbest]
            print(f"[vcurve] {k}: AD={float(g):.12g} relerr(h)={vcurve[f'{k}_relerr']}",
                  flush=True)

        # ---- checks ----------------------------------------------------------
        checks = {}
        # the gradient survives a user-level jax.jit unchanged (galpy adds no jit)
        for k in IDX:
            ve, ge = vg[k](jnp.asarray(Q0))
            vj, gj = jax.jit(vg[k])(jnp.asarray(Q0))
            checks[f"jit_vs_eager_{k}"] = {
                "dvalue": float(vj - ve), "dgrad": float(gj - ge),
            }
        # frequency gradients (diagnostic only)
        for k, i in FIDX.items():
            v, g = jax.value_and_grad(lambda q, i=i: Jf_jax(q, i))(jnp.asarray(Q0))
            fds = [fd_np(lambda q, i=i: Jf_np(q, i), Q0, h) for h in H_V[1:5]]
            checks[f"freq_{k}"] = {
                "value_jax": float(v), "value_numpy": Jf_np(Q0, i), "ad": float(g),
                "h": H_V[1:5], "fd": fds,
                "relerr": [abs(float(g) - f) / abs(f) for f in fds],
            }
            print(f"[freq] {k}: AD={float(g):.10g} relerr={checks[f'freq_{k}']['relerr']}",
                  flush=True)

    out = {
        "env": env_stamp(),
        "config": {
            "potential": "LogarithmicHaloPotential(normalize=1., q=q)",
            "actionAngle": f"actionAngleStaeckel(pot, delta={DELTA}, c=False, order={ORDER})",
            "orbit": ORBIT, "phi": 0.0,
            "q_range": [float(Q[0]), float(Q[-1])], "n_q": int(len(Q)),
            "fd_every": FD_EVERY, "fd_h_markers": H_MARK, "q0": Q0, "h_vcurve": H_V,
            "fd_reference": "numpy path (c=False), potential + aA rebuilt from q+/-h",
            "ad": "jax.value_and_grad, eager, float64, CPU",
        },
        "sweep": sweep,
        "vcurve": vcurve,
        "checks": checks,
        "wall_s": time.time() - t_start,
    }
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print("written", os.path.abspath(OUT), f"({out['wall_s']:.0f}s)")


if __name__ == "__main__":
    main()
