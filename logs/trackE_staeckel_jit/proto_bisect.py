"""Prototype: rolled (lax.fori_loop) vs unrolled (Python for) Staeckel bisection.

Faithful to galpy.backend.optimize.bisect_root over the REAL J_R integrand
(MWPotential2014), so per-iteration cost is representative. Measures jit COMPILE
time + warm runtime + numerical agreement. CPU-only (GPU is busy re-measuring g17).
"""
import os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, "/var/tmp/staeckel-freqs")
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp, numpy
from jax import lax
import galpy; assert "/var/tmp/staeckel-freqs" in galpy.__file__
from galpy.actionAngle.actionAngleStaeckel import (
    _staeckel_setup, _staeckel_uminumax, _JRStaeckelIntegrandSquared,
)
from galpy.potential import MWPotential2014
from galpy.backend.optimize import n_bisect_steps

pot, delta = MWPotential2014, 0.45


def grid(N):
    Rg=numpy.array([0.7,0.9,1.1,1.3]);vRg=numpy.array([-0.3,-0.1,0.1,0.3])
    vTg=numpy.array([-0.8,0.6,0.9,1.2]);zg=numpy.array([-0.3,-0.1,0.1,0.3]);vzg=numpy.array([-0.2,0.0,0.15])
    G=numpy.meshgrid(Rg,vRg,vTg,zg,vzg,indexing="ij");B=[g.ravel() for g in G]
    reps=(N//B[0].size)+1
    return [jnp.asarray(numpy.tile(b,reps)[:N]) for b in B]


def ct(fn, *a):
    t0=time.perf_counter(); jax.block_until_ready(fn(*a)); tc=time.perf_counter()-t0
    fn(*a); fn(*a)
    t0=time.perf_counter()
    for _ in range(3): jax.block_until_ready(fn(*a))
    return tc, (time.perf_counter()-t0)/3


N = 1000
R,vR,vT,z,vz = grid(N)
s = _staeckel_setup(jnp, R, vR, vT, z, vz, pot, delta)
args = (s["E"], s["Lz"], s["I3U"], delta, s["u0"], s["sinh2u0"], s["v0u"], s["sin2v0u"], s["potu0v0"], pot)
f = lambda u: _JRStaeckelIntegrandSquared(u, *args)
ux = s["ux"]
lo = ux*0.5
for _ in range(60):
    lo = jnp.where((f(lo)>=0.0)&(lo>1e-10), lo*0.5, lo)
a_br, b_br = lo, ux
n = n_bisect_steps(numpy.asarray(a_br), numpy.asarray(b_br), 1e-13, 200)
print(f"N={N}  bisection halvings n={n}  (real MWPotential2014 J_R integrand)", flush=True)


def bisect_unrolled(a, b):
    lo=a*1.0; hi=b*1.0; slo=jnp.sign(f(lo))
    for _ in range(n):
        mid=0.5*(lo+hi); same=jnp.sign(f(mid))==slo
        lo=jnp.where(same,mid,lo); hi=jnp.where(same,hi,mid)
    return 0.5*(lo+hi)


def bisect_rolled(a, b):
    lo=a*1.0; hi=b*1.0; slo=jnp.sign(f(lo))
    def body(i, c):
        lo,hi=c; mid=0.5*(lo+hi); same=jnp.sign(f(mid))==slo
        return (jnp.where(same,mid,lo), jnp.where(same,hi,mid))
    lo,hi = lax.fori_loop(0, n, body, (lo,hi))
    return 0.5*(lo+hi)


ju, jr = jax.jit(bisect_unrolled), jax.jit(bisect_rolled)
tcu, twu = ct(ju, a_br, b_br)
print(f"UNROLLED bisect : compile={tcu:6.1f}s  warm={twu*1e3:7.2f}ms", flush=True)
tcr, twr = ct(jr, a_br, b_br)
print(f"ROLLED   bisect : compile={tcr:6.1f}s  warm={twr*1e3:7.2f}ms", flush=True)
md = float(jnp.max(jnp.abs(ju(a_br,b_br) - jr(a_br,b_br))))
print(f"  => compile speedup {tcu/tcr:5.0f}x | warm ratio {twu/twr:.2f}x | max|unrolled-rolled|={md:.2e}", flush=True)

# the full status-quo _staeckel_uminumax (all 60+80+45+45 loops unrolled)
def full(R,vR,vT,z,vz):
    ss=_staeckel_setup(jnp,R,vR,vT,z,vz,pot,delta)
    return _staeckel_uminumax(jnp, ss, pot, delta)
tcf, twf = ct(jax.jit(full), R,vR,vT,z,vz)
print(f"FULL _staeckel_uminumax (status quo, fully unrolled): compile={tcf:.1f}s warm={twf*1e3:.1f}ms", flush=True)
print("DONE", flush=True)
