"""A/B of the incomplete-gamma switch on galpy's compiled in-backend orbit integration
(the g12/g8 measurement: MWPotential2014, Sun-like ICs, 2 Gyr, diffrax Dopri8 or
symplec6 on one batched Orbit, rtol=atol=1e-8).

    python orbit_bench.py <variant> <integrator> N [N ...]
variant: old (the switch before backend/special-gpu-gate: native below 4096
elements, which under vmap is every batch) or new (the branch's GPU rule);
integrator: dopri8 or symplec6. Prints compile+run and the best of 3 warm calls.
"""
import sys, time, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
variant, integ, Ns = sys.argv[1], sys.argv[2], [int(float(v)) for v in sys.argv[3:]]
from galpy.backend.special._fallback import gammainc as gm
if variant == "old":
    gm._jax_on_accelerator = lambda: False
elif variant != "new":
    raise SystemExit("variant?")
from galpy.backend._reference import integrate_orbit
from galpy.potential import MWPotential2014
ts = jnp.linspace(0.0, 56.2, 16)
ic0 = jnp.array([1.0, 0.0, 1.0, 0.0125, 0.036, 0.0])
scale = jnp.array([0.05, 0.01, 0.0, 0.01, 0.01, 0.0])
if integ == "dopri8":  # galpy_jax_measure.py: vmap of the per-orbit solve

    @jax.jit
    def batch(ics):
        return jax.vmap(lambda v: integrate_orbit(MWPotential2014, v, ts, rtol=1e-8, atol=1e-8))(ics)

else:  # galpy_symplec_measure.py: one batched Orbit, 450 steps

    from galpy.orbit import Orbit

    @jax.jit
    def batch(ics):
        o = Orbit(ics)
        o.integrate(ts, MWPotential2014, method="symplec6_jax", inbackend_kwargs={"nsteps": 450})
        return o.getOrbit()


print(f"# {variant} {integ} on {jax.devices()[0].device_kind} jax {jax.__version__}", flush=True)
for N in Ns:
    ics = ic0[None] + jax.random.normal(jax.random.PRNGKey(N), (N, 6)) * scale
    t0 = time.perf_counter(); batch(ics).block_until_ready(); tc = time.perf_counter() - t0
    best = min((lambda t0: (batch(ics).block_until_ready(), time.perf_counter() - t0)[1])(time.perf_counter()) for _ in range(3))
    print(f"{variant:4s} {integ} N={N:8d} compile+run {tc:7.2f} s  warm {best*1e3:10.2f} ms", flush=True)
