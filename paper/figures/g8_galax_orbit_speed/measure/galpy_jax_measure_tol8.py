"""galpy-jax GPU MWPotential2014 measurement at rtol=atol=1e-8 (matched to galax's
OrbitSolver default), per-orbit adaptive stepping (galpy's as-is vmap-of-solve).
Replaces the 1e-10 (galpy-default) g8 data. Two grids, mirroring the originals:
  lowN: 1000 dense saves, N=1..1e5   -> galpy-jax_gpu.json
  hiN : 16 saves,         N=1e3..3e5 -> galpy-jax_gpu_hiN.json
"""
import sys, time, json
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from galpy.potential import MWPotential2014
from galpy.backend._reference import integrate_orbit

mode = sys.argv[1]  # 'lowN' or 'hiN'
nt, Ns, fname = ((1000, [1, 10, 100, 1000, 10000, 30000, 100000], "galpy-jax_gpu.json")
                 if mode == "lowN" else
                 (16, [1000, 10000, 100000, 300000], "galpy-jax_gpu_hiN.json"))
ts = jnp.linspace(0.0, 56.2, nt)
ic0 = jnp.array([1.0, 0.0, 1.0, 0.0125, 0.036, 0.0])
scale = jnp.array([0.05, 0.01, 0.0, 0.01, 0.01, 0.0])

@jax.jit
def batch(ics):
    return jax.vmap(lambda v: integrate_orbit(MWPotential2014, v, ts, rtol=1e-8, atol=1e-8))(ics)

out = {"backend": "galpy-jax", "device": "gpu", "nt": nt, "tspan_Gyr": 2.0,
       "rtol": 1e-8, "atol": 1e-8, "stepping": "per-orbit (vmap-of-solve)",
       "note": f"{mode}; tolerances matched to galax OrbitSolver default", "results": []}
for N in Ns:
    ics = ic0[None] + jax.random.normal(jax.random.PRNGKey(N), (N, 6)) * scale
    try:
        batch(ics).block_until_ready()
        best = min(((lambda t0: (batch(ics).block_until_ready(), time.perf_counter()-t0)[1])(time.perf_counter())) for _ in range(3))
        out["results"].append({"N": N, "wall_s": best, "per_orbit_ms": best/N*1e3})
        print(f"galpy-jax[gpu,{mode}] N={N:7d}: {best*1e3:9.2f} ms ({best/N*1e3:.5f} ms/orbit)", flush=True)
        json.dump(out, open(f"/var/tmp/galax-bench/tol8_{fname}", "w"), indent=2)
    except Exception as e:
        print(f"N={N}: SKIP {type(e).__name__}: {str(e)[:60]}", flush=True); break
print("done")
