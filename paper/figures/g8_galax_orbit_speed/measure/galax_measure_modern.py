import time, json, sys
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import unxt as u
import galax.potential as gp, galax.dynamics as gd, galax.coordinates as gc

mw = gp.BovyMWPotential2014()
field = gd.fields.HamiltonianField(mw)
solver = gd.OrbitSolver()           # diffrax Dopri8
ts = u.Quantity(jnp.linspace(0.0, 2.0, 1000), "Gyr")
q0 = jnp.array([8.0, 0.0, 0.1]); p0 = jnp.array([10.0, 220.0, 8.0])

@jax.jit
def integrate(q, p):
    w0 = gc.PhaseSpaceCoordinate(q=u.Quantity(q, "kpc"),
                                 p=u.Quantity(p, "km/s"), t=u.Quantity(0.0, "Gyr"))
    return solver.solve(field, w0, ts[-1], saveat=ts).ys[0]  # (nt, N, 3)

dev = jax.devices()[0].platform
Ns = [1, 10, 100, 1000, 10000, 30000, 100000]
out = {"backend": "galax", "device": dev, "solver": "Dopri8",
       "nt": 1000, "tspan_Gyr": 2.0, "results": []}
for N in Ns:
    key = jax.random.PRNGKey(N)
    q = q0[None] + jax.random.normal(key, (N, 3)) * 0.05
    p = p0[None] + jax.random.normal(jax.random.fold_in(key, 1), (N, 3)) * 2.0
    try:
        r = integrate(q, p); r.block_until_ready()            # compile (excluded)
        best = min(( (lambda t0: (integrate(q, p).block_until_ready(), time.perf_counter()-t0)[1])(time.perf_counter()) ) for _ in range(3))
        out["results"].append({"N": N, "wall_s": best, "per_orbit_ms": best/N*1e3})
        print(f"galax[{dev}] N={N:7d}: {best*1e3:9.2f} ms  ({best/N*1e3:.5f} ms/orbit)", flush=True)
        json.dump(out, open(f"/var/tmp/galax-bench/galax_{dev}.json", "w"), indent=2)
    except Exception as e:
        print(f"galax[{dev}] N={N}: SKIP ({type(e).__name__}: {str(e)[:60]})", flush=True)
        break
print("done ->", f"galax_{dev}.json")
