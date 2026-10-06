"""g12 work-count measure -- quantify the SOLVER WORK behind the g12 wall times.

The g12 jax line ships diffrax **Dopri8** (8th order); the torch line ships
torchdiffeq **dopri5** (5th order -- torchdiffeq's dopri8 *backward* is buggy, see
galpy.backend._torch.orbit_ode). Different methods do different work per orbit, so a
jax-Dopri8-vs-torch-dopri5 wall-time crossover is NOT same-work. This script counts,
for the SAME orbit (MWPotential2014, Sun-like IC, 2 Gyr, rtol=atol=1e-8), the
(a) accepted steps and (b) RHS/function evaluations (nfev) of:

    * jax  diffrax Dopri8   (what galpy ships on jax)
    * jax  diffrax Dopri5   (the same-work match to torch's dopri5)
    * torch torchdiffeq dopri5 (what galpy ships on torch)

at a representative single orbit (N=1) and a batched N. Counts are device-
independent (same arithmetic on CPU/GPU), so this runs on the CPU to avoid any GPU
contention -- it produces no wall times, only work counts.

  * diffrax nfev: counted at runtime with jax.debug.callback firing once per RHS
    call (verified: matches num_steps*(nstages-1)+1 with FSAL; Dopri8=14 stages,
    Dopri5=7 stages). accepted/rejected/total steps come from sol.stats.
  * torchdiffeq nfev: the field is eager, so a plain Python counter on the wrapped
    func is exact. torchdiffeq exposes no public step count, so accepted steps are
    inferred as nfev//6 (dopri5 = 6 evals/accepted step + a 7th FSAL on step 1;
    rejected steps also cost 6 evals, so this is steps_accepted+steps_rejected).

Run (CPU, no GPU needed; serial-safe):
    PYTHONPATH=/var/tmp/backends-merge python work_counts.py

Writes data/work_counts.json (machine + version stamped).
"""

import json
import os
import socket

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # counts are device-independent

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import diffrax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import galpy  # noqa: E402
from galpy.backend._reference.inbackend_ode import (  # noqa: E402
    _eom_rhs,
    _from_eom,
    _to_eom,
)
from galpy.potential import MWPotential2014  # noqa: E402

assert galpy.__file__.startswith(
    os.environ.get("GALPY_WORKTREE", "/var/tmp/backends-merge")
), galpy.__file__

_DATA = os.path.join(os.path.dirname(__file__), "..", "data")

# Same orbit/tol/nt as the shared jax measure (lowN convention; counts are
# save-independent because integration is adaptive-step-bound).
nt = 1000
ts_np = [float(x) for x in jnp.linspace(0.0, 56.2, nt)]
ic0 = jnp.array([1.0, 0.0, 1.0, 0.0125, 0.036, 0.0])  # Sun-like [R,vR,vT,z,vz,phi]
scale = jnp.array([0.05, 0.01, 0.0, 0.01, 0.01, 0.0])
RTOL = ATOL = 1e-8
N_BATCH = 1000


def _jax_ics(N):
    return ic0[None] + jax.random.normal(jax.random.PRNGKey(N), (N, 6)) * scale


def _count_diffrax(solver_name, vxvv):
    """nfev (runtime callback) + step stats for one diffrax solve of `vxvv` (shape (6,))."""
    solver = {"Dopri8": diffrax.Dopri8, "Dopri5": diffrax.Dopri5}[solver_name]()
    nstages = len(solver.tableau.c) + 1
    calls = {"n": 0}

    def _bump(_):
        calls["n"] += 1

    def field(t, y, args):
        jax.debug.callback(_bump, t)
        return jnp.stack(_eom_rhs(y, MWPotential2014, t, jnp))

    ts = jnp.linspace(0.0, 56.2, nt)
    y0 = _to_eom(jnp, vxvv)
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(field),
        solver,
        t0=ts[0],
        t1=ts[-1],
        dt0=None,
        y0=y0,
        saveat=diffrax.SaveAt(ts=ts),
        stepsize_controller=diffrax.PIDController(rtol=RTOL, atol=ATOL),
        max_steps=100000,
    )
    jax.block_until_ready(sol.ys)
    st = sol.stats
    return {
        "method": f"diffrax {solver_name}",
        "order": {"Dopri8": 8, "Dopri5": 5}[solver_name],
        "nstages": int(nstages),
        "num_accepted_steps": int(st["num_accepted_steps"]),
        "num_rejected_steps": int(st["num_rejected_steps"]),
        "num_steps": int(st["num_steps"]),
        "nfev": int(calls["n"]),
        "nfev_per_accepted_step": calls["n"] / int(st["num_accepted_steps"]),
    }


def _count_torch():
    import torch
    from torchdiffeq import odeint

    torch.set_default_dtype(torch.float64)

    def _torch_one(N):
        g = torch.Generator().manual_seed(N)
        ic0_t = torch.tensor([1.0, 0.0, 1.0, 0.0125, 0.036, 0.0])
        scale_t = torch.tensor([0.05, 0.01, 0.0, 0.01, 0.01, 0.0])
        ics = ic0_t[None] + torch.randn(N, 6, generator=g) * scale_t  # (N,6)
        y0 = _to_eom(torch, ics)  # (N,6): galpy's EOM helpers are batch-first
        calls = {"n": 0}

        def field(t, y):
            calls["n"] += 1
            return torch.stack(_eom_rhs(y, MWPotential2014, t, torch), dim=-1)

        ts = torch.linspace(0.0, 56.2, nt)
        odeint(field, y0, ts, method="dopri5", rtol=RTOL, atol=ATOL)
        # dopri5: 6 evals per (accepted or rejected) step, +1 FSAL on the first step.
        nstep = (calls["n"] - 1) // 6
        return {
            "N": N,
            "method": "torchdiffeq dopri5",
            "order": 5,
            "nstages": 7,
            "nfev": calls["n"],
            "num_total_steps_est": int(nstep),
            "nfev_per_step": calls["n"] / max(nstep, 1),
        }

    return {"single": _torch_one(1), "batch": _torch_one(N_BATCH)}


out = {
    "what": "solver work counts (accepted steps + nfev) for the g12 orbit",
    "orbit": "MWPotential2014, Sun-like IC, 2 Gyr (linspace(0,56.2,nt))",
    "nt": nt,
    "rtol": RTOL,
    "atol": ATOL,
    "N_batch": N_BATCH,
    "host": socket.gethostname(),
    "note": (
        "diffrax counts are per single vmapped solve (per-orbit work); torch carries "
        "a (6,N) batched state through ONE shared adaptive controller, so its nfev is "
        "the number of shared RHS calls (each evaluating all N orbits at once)."
    ),
}
for m in ("jax", "diffrax", "torch", "torchdiffeq", "galpy", "numpy"):
    try:
        out[m + "_version"] = __import__(m).__version__
    except Exception:
        pass

# Single representative orbit (N=1): the literal Sun-like IC (no perturbation).
out["diffrax_single"] = {
    "Dopri8": _count_diffrax("Dopri8", ic0),
    "Dopri5": _count_diffrax("Dopri5", ic0),
}
# One representative orbit drawn from the batch distribution (per-orbit work in the
# batched regime is identical -- vmap runs each orbit's own adaptive controller).
sample = _jax_ics(N_BATCH)[0]
out["diffrax_batch_sample"] = {
    "Dopri8": _count_diffrax("Dopri8", sample),
    "Dopri5": _count_diffrax("Dopri5", sample),
}
out["torch"] = _count_torch()

os.makedirs(_DATA, exist_ok=True)
with open(os.path.join(_DATA, "work_counts.json"), "w") as fh:
    json.dump(out, fh, indent=2)

print("== diffrax, single Sun-like orbit ==")
for k, v in out["diffrax_single"].items():
    print(
        f"  {v['method']:16s} accepted={v['num_accepted_steps']:4d} "
        f"rejected={v['num_rejected_steps']:3d} total={v['num_steps']:4d} "
        f"nfev={v['nfev']:5d}  ({v['nfev_per_accepted_step']:.2f}/acc-step)"
    )
print("== torch, dopri5 (shared (6,N) controller) ==")
for key in ("single", "batch"):
    v = out["torch"][key]
    print(
        f"  N={v['N']:5d}  nfev={v['nfev']:6d}  total_steps(est)={v['num_total_steps_est']:5d} "
        f"({v['nfev_per_step']:.2f}/step)"
    )
print("done -> work_counts.json")
