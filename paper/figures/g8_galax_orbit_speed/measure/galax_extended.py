import time, json
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import unxt as u
import galax.potential as gp, galax.dynamics as gd, galax.coordinates as gc
mw=gp.BovyMWPotential2014(); field=gd.fields.HamiltonianField(mw); solver=gd.OrbitSolver()
ts=u.Quantity(jnp.linspace(0.,2.,16),"Gyr")   # 16 saves -> high N
q0=jnp.array([8.,0.,0.1]); p0=jnp.array([10.,220.,8.])
@jax.jit
def integ(q,p):
    w0=gc.PhaseSpaceCoordinate(q=u.Quantity(q,"kpc"),p=u.Quantity(p,"km/s"),t=u.Quantity(0.,"Gyr"))
    return solver.solve(field,w0,ts[-1],saveat=ts).ys[0]
out={"backend":"galax","device":"gpu","nt":16,"tspan_Gyr":2.0,"note":"high-N: 16 saves","results":[]}
for N in [1000,10000,100000,300000]:
    key=jax.random.PRNGKey(N)
    q=q0[None]+jax.random.normal(key,(N,3))*0.05; p=p0[None]+jax.random.normal(jax.random.fold_in(key,1),(N,3))*2.
    try:
        r=integ(q,p); r.block_until_ready()
        best=min(((lambda t0:(integ(q,p).block_until_ready(),time.perf_counter()-t0)[1])(time.perf_counter())) for _ in range(3))
        out["results"].append({"N":N,"wall_s":best,"per_orbit_ms":best/N*1e3})
        print(f"galax[gpu,16sv] N={N:8d}: {best*1e3:9.1f} ms ({best/N*1e3:.5f} ms/orbit)",flush=True)
        json.dump(out,open("/var/tmp/galax-bench/galax_gpu_hiN.json","w"),indent=2)
    except Exception as e:
        print(f"galax[gpu] N={N}: SKIP {str(e)[:40]}",flush=True); break
print("done")
