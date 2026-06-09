import time, json
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from galpy.potential import MWPotential2014
from galpy.backend._reference import integrate_orbit
ts = jnp.linspace(0.0, 56.2, 16)   # 16 saves -> memory allows N up to ~1e6
ic0 = jnp.array([1.0,0.0,1.0,0.0125,0.036,0.0]); scale=jnp.array([0.05,0.01,0.,0.01,0.01,0.])
@jax.jit
def batch(ics): return jax.vmap(lambda v: integrate_orbit(MWPotential2014, v, ts))(ics)
out={"backend":"galpy-jax","device":"gpu","nt":16,"tspan_Gyr":2.0,"note":"high-N: 16 saves","results":[]}
for N in [1000,10000,100000,300000,1000000]:
    ics=ic0[None]+jax.random.normal(jax.random.PRNGKey(N),(N,6))*scale
    try:
        r=batch(ics); r.block_until_ready()
        best=min(((lambda t0:(batch(ics).block_until_ready(),time.perf_counter()-t0)[1])(time.perf_counter())) for _ in range(3))
        out["results"].append({"N":N,"wall_s":best,"per_orbit_ms":best/N*1e3})
        print(f"galpy-jax[gpu,16sv] N={N:8d}: {best*1e3:9.1f} ms ({best/N*1e3:.5f} ms/orbit)",flush=True)
        json.dump(out,open("/var/tmp/galax-bench/galpy-jax_gpu_hiN.json","w"),indent=2)
    except Exception as e:
        print(f"galpy-jax[gpu] N={N}: SKIP {str(e)[:40]}",flush=True); break
print("done")
