"""FLOP / byte census of the g12/g8 jax orbit integration (diffrax, vmap-of-solve).

Compiles, on the CPU backend, exactly the function the measure scripts time
(``jax.jit(jax.vmap(lambda v: integrate_orbit(MWPotential2014, v, ts, rtol=atol=1e-8,
solver=...)))`` with nt=16 saves) for a batch of B lanes, and runs the loop-aware
HLO census (hlo_cost.py) with these trip counts per loop:

  depth 0  diffrax step loop            S  (num_steps incl. rejected)
  depth 1  stage loop (k-buffer carry)  13 (Dopri8) / 6 (Dopri5) RHS per step
           (= work_counts.json: nfev = 13*steps+1 / 6*steps+1)
  depth 1  SaveAt interpolation loop    1 per step (est.; with many lanes some
                                        lane crosses a save time at ~every step)
  depth 2  igamma continued fraction    n_cf (jax native igamma, VALUE mode)
  depth 2  igamma ascending series      0 (x=(r/rc)^2 > max(1, a) for every lane)

S and n_cf come from orbit_stats.json and the CF emulation below. The per-RHS
cost is also censused on its own (vmap of the per-orbit RHS, as the solve sees it:
inside vmap-of-solve the igamma argument has per-lane size 1, so galpy's gammainc
router takes jax's NATIVE igamma -- two while-loops -- not the fixed-length
series/CF fallback, which is only used for staged arrays of >= 4096 elements).

Run (CPU only):
  taskset -c 24-35 nice -n 19 env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu \
    PYTHONPATH=/var/tmp/fbp:/home/bovy/Repos/galpy-jax python orbit_costs.py
Writes orbit_costs.json next to this script.
"""

import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy

jax.config.update("jax_enable_x64", True)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hlo_cost as hc  # noqa: E402

from galpy.backend._reference import integrate_orbit  # noqa: E402
from galpy.backend._reference.inbackend_ode import _eom_rhs  # noqa: E402
from galpy.potential import MWPotential2014  # noqa: E402

B = 1024
stats = json.load(open(os.path.join(HERE, "orbit_stats.json")))
psc = MWPotential2014[0]
A_IG = 1.5 - 0.5 * psc.alpha
RC = psc.rc


def n_cf(r):
    """Iterations of jax's native igamma continued fraction (VALUE mode, float64)
    to convergence at x=(r/rc)^2 -- a transcription of
    jax/_src/lax/special.py::_igammac_continued_fraction's stopping rule."""
    a, x = A_IG, (r / RC) ** 2
    eps = numpy.finfo(float).eps
    y, z, c = 1 - a, x + (1 - a) + 1, 0.0
    pkm2, qkm2, pkm1, qkm1 = 1.0, x, x + 1, (x + (1 - a) + 1) * x
    ans, t, n = pkm1 / qkm1, 1.0, 0
    while t > eps and c < 2000:
        c += 1
        y += 1
        z += 2
        yc = y * c
        pk, qk = pkm1 * z - pkm2 * yc, qkm1 * z - qkm2 * yc
        if qk != 0:
            r_ = pk / qk
            t, ans = abs((ans - r_) / r_), r_
        else:
            t = 1.0
        pkm2, pkm1, qkm2, qkm1 = pkm1, pk, qkm1, qk
        if abs(pk) > 1 / eps:
            pkm2, pkm1, qkm2, qkm1 = pkm2 * eps, pkm1 * eps, qkm2 * eps, qkm1 * eps
        n += 1
    return n


def classify(info, nstage_leaves_lane):
    d = len(info["path"])
    if d == 0:
        return "step"
    if info["nleaves"] == 13 and info["lane"] == B:
        return "cf"
    if info["nleaves"] == 5 and info["lane"] == B:
        return "series"
    if d == 1 and info["lane"] == nstage_leaves_lane:
        return "stage"
    if d == 1:
        return "save"
    return "unknown"


def census_solve(text, solver, S, ncf, save_trips=1, ideal=False, all_once=False):
    stage_lane = B * (14 if solver == "dopri8" else 7) * 6
    nst = 13 if solver == "dopri8" else 6
    kinds = []

    def trips(info):
        k = classify(info, stage_lane)
        kinds.append((k, info["name"], info["nleaves"], info["lane"]))
        if all_once:
            return 1
        return {"step": S, "stage": nst, "save": save_trips, "cf": ncf, "series": 0}.get(k, 1)

    cen = hc.Census(text, trips=trips, ideal=ideal)
    return cen.total(), kinds


out = {"B": B, "igamma_a": A_IG, "rc": RC,
       "n_cf_vs_r": {f"{r:.2f}": n_cf(r) for r in (0.6, 0.7, 0.713, 0.75, 0.8, 0.9, 1.0, 1.1, 1.2)}}

# ---- one RHS evaluation (vmap of the per-orbit RHS) --------------------------
y = jnp.tile(jnp.array([1.0, 0.0, 0.0, 1.0, 0.0125, 0.036]), (B, 1))
rhs = jax.jit(jax.vmap(lambda v: jnp.stack(_eom_rhs(v, MWPotential2014, 0.0, jnp), axis=-1)))
c = rhs.lower(y).compile()
ca = c.cost_analysis()
out["rhs_xla_once"] = {k: ca.get(k, 0.0) / B for k in ("flops", "transcendentals", "bytes accessed")}
txt = c.as_text()
out["rhs"] = {}
for label, ncf in (("loops_once", None), ("n_cf=10 (r=1)", 10), ("n_cf=15 (batch r_min)", 15)):
    def tr(info, ncf=ncf):
        if ncf is None:
            return 1
        return ncf if info["nleaves"] == 13 else 0
    out["rhs"][label] = hc.summary(hc.Census(txt, trips=tr).total(), per=B)
    out["rhs"][label]["bytes_ideal"] = hc.Census(txt, trips=tr, ideal=True).total()["bytes"] / B
# isolate one CF iteration and the loop-free remainder
r10, r15 = out["rhs"]["n_cf=10 (r=1)"], out["rhs"]["n_cf=15 (batch r_min)"]
out["cf_iteration"] = {k: (r15[k] - r10[k]) / 5 for k in ("flops_xla", "transcendentals", "bytes")}

# ---- the full solve ----------------------------------------------------------
ts = jnp.linspace(0.0, 56.2, 16)
ics = jnp.ones((B, 6)) * jnp.array([1.0, 0, 1, 0.0125, 0.036, 0])
for solver, sname in (("dopri8", "Dopri8"), ("dopri5", "Dopri5")):
    f = jax.jit(jax.vmap(lambda v: integrate_orbit(MWPotential2014, v, ts, rtol=1e-8,
                                                   atol=1e-8, solver=solver)))
    comp = f.lower(ics).compile()
    ca = comp.cost_analysis()
    text = comp.as_text()
    o = {"xla_once_per_lane": {k: ca.get(k, 0.0) / B for k in ("flops", "transcendentals",
                                                                 "bytes accessed")}}
    once, kinds = census_solve(text, solver, 1, 1, all_once=True)
    o["census_once_per_lane"] = hc.summary(once, per=B)  # validates the parser vs XLA
    o["loops"] = sorted(set(kinds))
    # per step (difference S=2 minus S=1), at the batch-max and typical CF counts
    for lab, ncf in (("ncf15", 15), ("ncf10", 10)):
        a1, _ = census_solve(text, solver, 1, ncf)
        a2, _ = census_solve(text, solver, 2, ncf)
        s1, s2 = hc.summary(a1, per=B), hc.summary(a2, per=B)
        o[f"per_step_{lab}"] = {k: s2[k] - s1[k] for k in ("flops_xla", "transcendentals",
                                                           "flop_equiv", "bytes", "kernels",
                                                           "arith", "div", "cmp_select_logic",
                                                           "integer")}
        i1, _ = census_solve(text, solver, 1, ncf, ideal=True)
        i2, _ = census_solve(text, solver, 2, ncf, ideal=True)
        o[f"per_step_{lab}"]["bytes_ideal"] = (i2["bytes"] - i1["bytes"]) / B
        o[f"prologue_{lab}"] = {k: s1[k] - o[f"per_step_{lab}"][k]
                                for k in ("flops_xla", "transcendentals", "flop_equiv", "bytes")}
    for N in ("300000", "3000000"):
        st = stats[N][sname]
        o[N] = {"S_max": st["S_max_batch_est"], "S_mean": st["S_mean_est"],
                "r_min": st["r_min_batch_est"], "n_cf_rmin": n_cf(st["r_min_batch_est"])}
        ps, ps10 = o["per_step_ncf15"], o["per_step_ncf10"]
        o[N]["performed_per_orbit"] = {k: st["S_max_batch_est"] * ps[k] for k in
                                      ("flops_xla", "transcendentals", "flop_equiv", "bytes",
                                       "bytes_ideal")}
        o[N]["useful_per_orbit"] = {k: st["S_mean_est"] * ps10[k] for k in
                                   ("flops_xla", "transcendentals", "flop_equiv", "bytes",
                                    "bytes_ideal")}
    out[sname] = o
    print(sname, json.dumps({k: v for k, v in o.items() if k != "loops"}, default=float)[:1500],
          flush=True)

with open(os.path.join(HERE, "orbit_costs.json"), "w") as fh:
    json.dump(out, fh, indent=1, default=float)
print(json.dumps(out["rhs"], indent=1, default=float))
print("cf_iteration", out["cf_iteration"], "n_cf_vs_r", out["n_cf_vs_r"])
