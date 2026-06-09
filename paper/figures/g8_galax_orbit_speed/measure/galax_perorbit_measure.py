"""galax GPU BovyMWPotential2014 with PER-ORBIT adaptive stepping (vmap of a
single-orbit OrbitSolver.solve) -- the apples-to-apples control for galpy's
vmap-of-solve. NOT galax's native batched mode (its default shares one
PIDController across the batch); tolerances are the OrbitSolver defaults
(rtol=atol=1e-8), same as the native-mode data. Grids mirror the existing data:
  lowN: 1000 saves, N=1..1e4   -> galax_gpu_perorbit.json
  hiN : 16 saves,   N=1e3..3e5 -> galax_gpu_perorbit_hiN.json
"""
import sys, time, json
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import unxt as u
import galax.potential as gp, galax.dynamics as gd, galax.coordinates as gc

mode = sys.argv[1]
nt, Ns, fname = ((1000, [1, 10, 100, 1000, 10000], "galax_gpu_perorbit.json")
                 if mode == "lowN" else
                 (16, [1000, 10000, 100000, 300000], "galax_gpu_perorbit_hiN.json"))
mw = gp.BovyMWPotential2014()
field = gd.fields.HamiltonianField(mw)
solver = gd.OrbitSolver()
ts = u.Quantity(jnp.linspace(0.0, 2.0, nt), "Gyr")
q0 = jnp.array([8.0, 0.0, 0.1]); p0 = jnp.array([10.0, 220.0, 8.0])

def one(qi, pi):
    w0 = gc.PhaseSpaceCoordinate(q=u.Quantity(qi, "kpc"), p=u.Quantity(pi, "km/s"),
                                 t=u.Quantity(0.0, "Gyr"))
    return solver.solve(field, w0, ts[-1], saveat=ts).ys[0]

integ = jax.jit(jax.vmap(one))
out = {"backend": "galax", "device": "gpu", "nt": nt, "tspan_Gyr": 2.0,
       "rtol": 1e-8, "atol": 1e-8, "solver": "Dopri8",
       "stepping": "per-orbit (vmap of single-orbit solve; NOT galax's native batched mode)",
       "results": []}
for N in Ns:
    key = jax.random.PRNGKey(N)
    q = q0[None] + jax.random.normal(key, (N, 3)) * 0.05
    p = p0[None] + jax.random.normal(jax.random.fold_in(key, 1), (N, 3)) * 2.0
    try:
        integ(q, p).block_until_ready()
        best = min(((lambda t0: (integ(q, p).block_until_ready(), time.perf_counter()-t0)[1])(time.perf_counter())) for _ in range(3))
        out["results"].append({"N": N, "wall_s": best, "per_orbit_ms": best/N*1e3})
        print(f"galax-perorbit[gpu,{mode}] N={N:7d}: {best*1e3:9.2f} ms ({best/N*1e3:.5f} ms/orbit)", flush=True)
        json.dump(out, open(f"/var/tmp/galax-bench/{fname}", "w"), indent=2)
    except Exception as e:
        print(f"N={N}: SKIP {type(e).__name__}: {str(e)[:60]}", flush=True); break
print("done")
