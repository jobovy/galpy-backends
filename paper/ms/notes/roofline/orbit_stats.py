"""Per-orbit solver statistics for the g12/g8 orbit benchmark (CPU, no timing).

The jax series integrate N orbits as ``jax.vmap`` of one adaptive diffrax solve.
A vmapped while-loop runs until its LAST lane is done, and every lane executes
every iteration (finished lanes are masked with select). So the work actually
performed per orbit is set by the batch maximum, not the mean:

  * diffrax step loop: S_max = max over the batch of num_steps (accepted+rejected);
  * jax's native igamma continued fraction (PowerSphericalPotentialwCutoff in
    MWPotential2014): its while-loop runs until the slowest lane converges, i.e.
    the lane at the smallest radius (fewest CF iterations at large x=(r/rc)^2).

This script integrates (a) a random subset of the measured initial conditions and
(b) the ICs with the most extreme R (both tails) out of the FULL N-orbit IC set
the measure scripts draw (PRNGKey(N)), with the same diffrax configuration galpy
uses (PIDController(rtol=atol=1e-8), dt0=None, max_steps=1e5), and records
num_steps per orbit and each orbit's minimum r (dense 1000-point save grid).

Run (CPU only):
  taskset -c 24-35 nice -n 19 env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu \
    PYTHONPATH=/var/tmp/fbp:/home/bovy/Repos/galpy-jax python orbit_stats.py
Writes orbit_stats.json next to this script.
"""

import json
import os
import time

import jax
import jax.numpy as jnp
import numpy

jax.config.update("jax_enable_x64", True)
import diffrax  # noqa: E402

from galpy.backend._reference.inbackend_ode import _eom_rhs, _to_eom  # noqa: E402
from galpy.potential import MWPotential2014  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ic0 = jnp.array([1.0, 0.0, 1.0, 0.0125, 0.036, 0.0])
scale = jnp.array([0.05, 0.01, 0.0, 0.01, 0.01, 0.0])
ts = jnp.linspace(0.0, 56.2, 1000)  # dense, for r_min (step counts are save-independent)
term = diffrax.ODETerm(lambda t, y, args: jnp.stack(_eom_rhs(y, MWPotential2014, t, jnp), axis=-1))


def make_solve(solver):
    def one(v):
        sol = diffrax.diffeqsolve(
            term, solver, t0=ts[0], t1=ts[-1], dt0=None, y0=_to_eom(jnp, v),
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-8),
            max_steps=100000,
        )
        ys = sol.ys
        r = jnp.sqrt(ys[:, 0] ** 2 + ys[:, 2] ** 2 + ys[:, 4] ** 2)
        return sol.stats["num_steps"], sol.stats["num_accepted_steps"], jnp.min(r)

    return jax.jit(jax.vmap(one))


def run(solver_name, ics, chunk=1024):
    f = make_solve({"Dopri8": diffrax.Dopri8, "Dopri5": diffrax.Dopri5}[solver_name]())
    out = [[], [], []]
    for i in range(0, ics.shape[0], chunk):
        a, b, c = f(ics[i : i + chunk])
        for lst, v in zip(out, (a, b, c)):
            lst.append(numpy.asarray(v))
    return [numpy.concatenate(o) for o in out]


res = {}
for N in (300000, 3000000):
    ics_all = ic0[None] + jax.random.normal(jax.random.PRNGKey(N), (N, 6)) * scale
    R = numpy.asarray(ics_all[:, 0])
    order = numpy.argsort(R)
    rng = numpy.random.default_rng(0)
    rand = rng.choice(N, 4096, replace=False)
    sel = {"random": rand, "lowR": order[:1024], "highR": order[-1024:]}
    res[N] = {"R0_min": float(R.min()), "R0_max": float(R.max())}
    for solver in ("Dopri8", "Dopri5"):
        t0 = time.time()
        d = {}
        for k, idx in sel.items():
            ns, na, rmin = run(solver, ics_all[numpy.sort(idx)])
            d[k] = dict(
                num_steps_mean=float(ns.mean()), num_steps_median=float(numpy.median(ns)),
                num_steps_p99=float(numpy.percentile(ns, 99)), num_steps_max=int(ns.max()),
                num_steps_min=int(ns.min()), accepted_mean=float(na.mean()),
                rmin_min=float(rmin.min()), rmin_median=float(numpy.median(rmin)),
            )
        d["S_max_batch_est"] = max(v["num_steps_max"] for v in d.values() if isinstance(v, dict))
        d["S_mean_est"] = d["random"]["num_steps_mean"]
        d["r_min_batch_est"] = min(v["rmin_min"] for v in d.values() if isinstance(v, dict))
        d["wall_s"] = time.time() - t0
        res[N][solver] = d
        print(N, solver, json.dumps(d), flush=True)

with open(os.path.join(HERE, "orbit_stats.json"), "w") as fh:
    json.dump({str(k): v for k, v in res.items()}, fh, indent=1)
