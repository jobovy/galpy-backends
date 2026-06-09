"""galpy-jax-GPU per-orbit cost across a ladder of potentials of increasing
force-evaluation cost. Isolates how much the (expensive) PowerSphericalwCutoff /
MWPotential2014 force cost contributes to the galpy-vs-galax gap, vs fixed
framework overhead (backend dispatch + cyl<->rect round-trip + vmap).

Usage: python galpy_jax_ladder.py <which>   where <which> selects a pot subset.
Writes /var/tmp/galax-bench/galpy_jax_ladder.json
"""
import sys, time, json
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from galpy.potential import (
    KeplerPotential, PlummerPotential, HernquistPotential,
    MiyamotoNagaiPotential, NFWPotential,
)
from galpy.backend._reference import integrate_orbit

# build the requested potentials (some need the migrated PSCw -> guarded import)
POTS = {
    "Kepler": KeplerPotential(normalize=1.0),
    "Plummer": PlummerPotential(normalize=1.0, b=0.5),
    "Hernquist": HernquistPotential(normalize=1.0, a=1.0),
    "MiyamotoNagai": MiyamotoNagaiPotential(normalize=1.0, a=0.5, b=0.1),
    "NFW": NFWPotential(normalize=1.0, a=2.0),
}
try:
    from galpy.potential import PowerSphericalPotentialwCutoff, MWPotential2014
    POTS["PSCwCutoff"] = PowerSphericalPotentialwCutoff(normalize=1.0, alpha=1.8, rc=1.9)
    POTS["MWPotential2014"] = MWPotential2014
    # MW with a Hernquist bulge instead of PSCw (recreatable in galax)
    from galpy.potential import MiyamotoNagaiPotential as _MN
    mw_h = [HernquistPotential(amp=0.6, a=0.3),
            _MN(amp=0.8, a=3.0, b=0.28),
            NFWPotential(amp=4.85, a=2.0)]
    POTS["MW_Hernquist"] = mw_h
except Exception as e:
    print("PSCw/MW unavailable on this checkout:", str(e)[:60], flush=True)

ts = jnp.linspace(0.0, 56.2, 16)
ic0 = jnp.array([1.0, 0.0, 1.0, 0.0125, 0.036, 0.0])
scale = jnp.array([0.05, 0.01, 0.0, 0.01, 0.01, 0.0])
Ns = [10000, 100000]

out = {"backend": "galpy-jax", "device": "gpu", "nt": 16, "results": {}}
which = sys.argv[1].split(",") if len(sys.argv) > 1 else list(POTS)
for name in which:
    if name not in POTS:
        continue
    pot = POTS[name]
    @jax.jit
    def batch(ics, pot=pot):
        return jax.vmap(lambda v: integrate_orbit(pot, v, ts))(ics)
    row = {}
    for N in Ns:
        ics = ic0[None] + jax.random.normal(jax.random.PRNGKey(N), (N, 6)) * scale
        try:
            batch(ics).block_until_ready()  # compile
            best = min(
                (lambda t0: (batch(ics).block_until_ready(), time.perf_counter() - t0)[1])(time.perf_counter())
                for _ in range(5)
            )
            row[N] = best / N * 1e3
            print(f"{name:16s} N={N:7d}: {best/N*1e3:.5f} ms/orbit", flush=True)
        except Exception as e:
            print(f"{name:16s} N={N:7d}: SKIP {str(e)[:50]}", flush=True)
    out["results"][name] = row
    json.dump(out, open("/var/tmp/galax-bench/galpy_jax_ladder.json", "w"), indent=2)
print("done")
