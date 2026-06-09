"""galax-GPU per-orbit cost across the same potential ladder, to pair with
galpy_jax_ladder.py. Standalone potentials with MW-ish parameters + a Sun-like
2-Gyr orbit, so per-orbit cost reflects force-eval cost at comparable step counts.
Writes /var/tmp/galax-bench/galax_ladder.json
"""
import time, json
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import unxt as u
import galax.potential as gp, galax.dynamics as gd, galax.coordinates as gc

U = "galactic"
def H(): return gp.HernquistPotential(m_tot=u.Quantity(1.2e12, "Msun"), r_s=u.Quantity(15., "kpc"), units=U)
def MN(): return gp.MiyamotoNagaiPotential(m_tot=u.Quantity(7e10, "Msun"), a=u.Quantity(3., "kpc"), b=u.Quantity(0.28, "kpc"), units=U)
def NFW(): return gp.NFWPotential(m=u.Quantity(1e12, "Msun"), r_s=u.Quantity(16., "kpc"), units=U)
def KEP(): return gp.KeplerPotential(m_tot=u.Quantity(1e11, "Msun"), units=U)
def PLM(): return gp.PlummerPotential(m_tot=u.Quantity(1.2e12, "Msun"), b=u.Quantity(15., "kpc"), units=U)
def MW_H():  # 3-component MW with a Hernquist bulge
    return gp.CompositePotential(bulge=gp.HernquistPotential(m_tot=u.Quantity(5e9,"Msun"), r_s=u.Quantity(0.5,"kpc"), units=U),
                                 disk=MN(),
                                 halo=gp.NFWPotential(m=u.Quantity(5e11,"Msun"), r_s=u.Quantity(16.,"kpc"), units=U))

POTS = {"Kepler": KEP, "Plummer": PLM, "Hernquist": H, "MiyamotoNagai": MN, "NFW": NFW,
        "MW_Hernquist": MW_H, "BovyMW2014": gp.BovyMWPotential2014}

ts = u.Quantity(jnp.linspace(0., 2., 16), "Gyr")
q0 = jnp.array([8., 0., 0.1]); p0 = jnp.array([10., 220., 8.])
Ns = [10000, 100000]
out = {"backend": "galax", "device": "gpu", "nt": 16, "results": {}}
for name, ctor in POTS.items():
    try:
        pot = ctor() if callable(ctor) else ctor
        field = gd.fields.HamiltonianField(pot); solver = gd.OrbitSolver()
        @jax.jit
        def integ(q, p, field=field, solver=solver):
            w0 = gc.PhaseSpaceCoordinate(q=u.Quantity(q, "kpc"), p=u.Quantity(p, "km/s"), t=u.Quantity(0., "Gyr"))
            return solver.solve(field, w0, ts[-1], saveat=ts).ys[0]
        row = {}
        for N in Ns:
            key = jax.random.PRNGKey(N)
            q = q0[None] + jax.random.normal(key, (N, 3)) * 0.05
            p = p0[None] + jax.random.normal(jax.random.fold_in(key, 1), (N, 3)) * 2.
            integ(q, p).block_until_ready()
            best = min((lambda t0: (integ(q, p).block_until_ready(), time.perf_counter() - t0)[1])(time.perf_counter()) for _ in range(5))
            row[N] = best / N * 1e3
            print(f"{name:16s} N={N:7d}: {best/N*1e3:.5f} ms/orbit", flush=True)
        out["results"][name] = row
        json.dump(out, open("/var/tmp/galax-bench/galax_ladder.json", "w"), indent=2)
    except Exception as e:
        print(f"{name:16s}: SKIP {str(e)[:70]}", flush=True)
print("done")
