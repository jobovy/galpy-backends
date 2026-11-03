"""Benchmark the jax incomplete-gamma paths on a GPU, compiled.

    python gate_bench.py <variant> <what> N [N ...]
variant: native (jax's gammainc at every size), rolled (galpy's series/CF as
lax.fori_loop, unroll=10; rolled<K> for unroll=K), unrolled (galpy's series/CF, Python-unrolled); what: fn (the
function alone, a = 0.6 as in MWPotential2014's bulge) or aa (the Staeckel
actionsFreqsAngles of MWPotential2014). Prints compile and warm (median of 5) times.
"""
import sys, time, numpy, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
variant, what, Ns = sys.argv[1], sys.argv[2], [int(v) for v in sys.argv[3:]]
from galpy.backend.special._fallback import gammainc as gm
if variant == "native":
    gm._jax_on_accelerator = lambda: False; gm._JAX_MIN_SIZE = 10**12
elif variant == "unrolled":
    gm._jax_on_accelerator = lambda: False; gm._JAX_MIN_SIZE = 0
elif variant.startswith("rolled"):
    if variant != "rolled":  # rolled<K>: unroll factor K of the fori_loops
        gm._GPU_UNROLL = int(variant[len("rolled"):])
else:
    raise SystemExit("variant?")
dev = jax.devices()[0]
print(f"# {variant} {what} on {dev.device_kind} jax {jax.__version__}", flush=True)
if what == "fn":
    from galpy.backend import special as gsp
    f = jax.jit(lambda x: gsp.gammainc(0.6, x))
    def args(N):
        return (jnp.asarray(numpy.geomspace(1e-3, 50.0, N)),)
else:
    from galpy.actionAngle import actionAngleStaeckel
    from galpy.potential import MWPotential2014
    aa = actionAngleStaeckel(pot=MWPotential2014, delta=0.45, c=False)
    Rg=numpy.array([0.7,0.9,1.1,1.3]);vRg=numpy.array([-0.3,-0.1,0.1,0.3]);vTg=numpy.array([-0.8,0.6,0.9,1.2]);zg=numpy.array([-0.3,-0.1,0.1,0.3]);vzg=numpy.array([-0.2,0.0,0.15])
    G=numpy.meshgrid(Rg,vRg,vTg,zg,vzg,indexing='ij');B=[g.ravel() for g in G];B.append(numpy.linspace(0.1,6.2,B[0].size))
    f = jax.jit(lambda *a: aa.actionsFreqsAngles(*a))
    def args(N):
        reps = (N // B[0].size) + 1
        return tuple(jnp.asarray(numpy.tile(b, reps)[:N]) for b in B)
for N in Ns:
    g = args(N)
    t0 = time.perf_counter(); jax.block_until_ready(f(*g)); tc = time.perf_counter() - t0
    ts = []
    for _ in range(5):
        t0 = time.perf_counter(); jax.block_until_ready(f(*g)); ts.append(time.perf_counter() - t0)
    print(f"{variant:9s} {what} N={N:7d} compile {tc:8.2f} s  warm {sorted(ts)[2]*1e3:10.2f} ms", flush=True)
