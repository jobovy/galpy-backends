"""Self-contained pure-jax orbit benchmark — NO galpy, NO galax imports.

MWPotential2014-like 3-component potential (PSCw bulge + MN disk + NFW halo) in
galpy natural units, Cartesian fused-gradient RHS, diffrax Dopri8 + PIDController
(rtol=atol=1e-8), vmap over N=1e5 perturbed Sun-like ICs. Identical code run in
the galpy env (jax/jaxlib 0.10.1) and the galax env (jax/jaxlib 0.8.0) isolates
the jaxlib/XLA-version (+ diffrax-version) factor on this GPU (TITAN Xp, sm_61).

Also: igamma kernel micro-bench at the orbit-regime arguments (a=0.6, x~(r/rc)^2).
"""
import time
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import diffrax
from jax.scipy.special import gammainc, gammaln

print("jax", jax.__version__, "| diffrax", diffrax.__version__,
      "| device", jax.devices()[0].platform, flush=True)

# --- MWPotential2014-like, galpy natural units ---
AMP_B, ALPHA, RC = 0.05/0.10750509802874262, 1.8, 1.9/8.0   # bulge (galpy's internal amp ~0.0300 effective)
AMP_B = 0.0299946864  # effective amp (matches galpy MWPotential2014[0]._amp)
A_MN, B_MN, AMP_MN = 3.0/8.0, 0.28/8.0, 0.7574802019371595
A_NFW, AMP_NFW = 2.0, 4.852230533528037
GAMMA_15A = jnp.exp(gammaln(1.5 - ALPHA/2.0))   # constant
GAMMA_10A = jnp.exp(gammaln(1.0 - ALPHA/2.0))

def Phi(xyz):
    x, y, z = xyz
    R2 = x*x + y*y
    r = jnp.sqrt(R2 + z*z)
    # PSCw potential (same form as galpy _evaluate)
    xx = (r/RC)**2
    psc = AMP_B * 2.0*jnp.pi * RC**(3.0-ALPHA) * (
        (r/RC) * GAMMA_10A * gammainc(1.0-ALPHA/2.0, xx)
        - GAMMA_15A * gammainc(1.5-ALPHA/2.0, xx)
    ) / r
    # MN
    mn = -AMP_MN / jnp.sqrt(R2 + (A_MN + jnp.sqrt(z*z + B_MN*B_MN))**2)
    # NFW
    nfw = -AMP_NFW * jnp.log(1.0 + r/A_NFW) / r
    return psc + mn + nfw

gradPhi = jax.grad(Phi)

def rhs(t, s, args):
    x, vx, y, vy, z, vz = s
    g = gradPhi(jnp.array([x, y, z]))
    return jnp.array([vx, -g[0], vy, -g[1], vz, -g[2]])

ts = jnp.linspace(0.0, 56.2, 16)

def one(s0):
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(rhs), diffrax.Dopri8(), ts[0], ts[-1], None, s0,
        saveat=diffrax.SaveAt(ts=ts),
        stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-8),
        max_steps=100000)
    return sol.ys

# Sun-like ICs in rectangular natural units
ic0 = jnp.array([1.0, 0.0, 0.0, 1.0, 0.0125, 0.036])  # x,vx,y,vy,z,vz
key = jax.random.PRNGKey(7)
N = 100000
ics = ic0[None] + jax.random.normal(key, (N, 6)) * jnp.array([0.05,0.01,0.0,0.01,0.005,0.01])
batch = jax.jit(jax.vmap(one))
r = batch(ics); r.block_until_ready()
best = min((lambda t0: (batch(ics).block_until_ready(), time.perf_counter()-t0)[1])(time.perf_counter()) for _ in range(5))
print(f"pure-jax MW-like, N={N}: {best/N*1e3:.5f} ms/orbit", flush=True)

# single-orbit step count
sol = diffrax.diffeqsolve(diffrax.ODETerm(rhs), diffrax.Dopri8(), ts[0], ts[-1], None, ic0,
    saveat=diffrax.SaveAt(ts=ts), stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-8),
    max_steps=100000)
print("steps:", int(sol.stats["num_steps"]), flush=True)

# igamma kernel micro-bench at orbit-regime args
for nm, xs in [("x~0.3 (galpy-units regime, r~1,rc=0.24 -> x~17)", None)]:
    pass
x = jnp.abs(jax.random.normal(key, (1_000_000,))) * 10.0 + 5.0   # x ~ 5-35 (continued-fraction branch)
f = jax.jit(lambda x: gammainc(0.6, x))
f(x).block_until_ready()
best = min((lambda t0: (f(x).block_until_ready(), time.perf_counter()-t0)[1])(time.perf_counter()) for _ in range(7))
print(f"igamma kernel (a=0.6, x~5-35, n=1e6): {best*1e3:.3f} ms", flush=True)
x2 = jnp.abs(jax.random.normal(key, (1_000_000,))) * 0.5 + 0.05  # x ~ 0.05-1.5 (series branch)
best = min((lambda t0: (f(x2).block_until_ready(), time.perf_counter()-t0)[1])(time.perf_counter()) for _ in range(7))
print(f"igamma kernel (a=0.6, x~0.05-1.5, n=1e6): {best*1e3:.3f} ms", flush=True)
print("done", flush=True)
