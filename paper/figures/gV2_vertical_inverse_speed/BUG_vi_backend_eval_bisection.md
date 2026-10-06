# BUG (performance): actionAngleVerticalInverse backend evaluation does 43-100 full kernel passes per call

Found while producing gV2 (the "jit N=1000 as slow as eager" anomaly). Not a
correctness bug -- values match numpy to 1.5e-14 -- but the jitted backend
evaluation is ~10x slower per angle than the numpy path, for two reasons.

## What dominates

`_mm_xvFreqs_backend` inverts the angle relation with the generic backend
`brentq` (galpy/backend/optimize.py), i.e. a FIXED schedule of bisection halvings
on [0, 2 pi] followed by one Newton polish. Each halving evaluates the full
`_mm_kernel` on all N angles (three (N, mm_npt=40) trig matrices + matmuls).
Measured (IsothermalDisk amp=1 sigma=0.5, Es=linspace(0.05,2,31),
setup_interp=True, J=0.5, jax 0.10.1 CPU, 4 cores of a Xeon E5-2697 v4):

| N | one jitted `_mm_kernel` | jitted fori_loop of 100 kernels | jitted `xvFreqs` |
|---|---|---|---|
| 10 | 9.6e-5 s | 3.4e-3 s | 4.7e-3 s |
| 100 | 5.7e-4 s | 4.8e-2 s | 5.3e-2 s |
| 1000 | 3.2e-3 s | 0.16 s | 0.13 s |
| 1e4 | 1.2e-2 s | 1.1 s | 1.0 s |
| 1e5 | 0.10 s | 8.4 s | 9.0 s |

So `xvFreqs` under jit == 100 x kernel at every N: no retracing, no Python
loop; the cost is the bisection count times the kernel. (The per-angle cost
is not constant in N because XLA:CPU only goes multi-threaded for the larger
fusions -- with all 72 hardware threads available the N>=1e4 calls are several
times faster, which is why an unpinned run sees "N=1e5 at 17 us/angle" but
"N=1e3 at 0.27 ms/angle". That part is XLA, not galpy.)

Single-threaded (OPENBLAS/OMP/torch/XLA = 1 thread; the gV2 measurement),
`xvFreqs(J=0.5, N angles)` is linear in N for every backend, and jit is SLOWER
than eager beyond N~1e3 (100 vs 43 kernel passes):

| N | numpy | torch (eager) | jax eager | jax jit |
|---|---|---|---|---|
| 10 | 1.5e-3 s | 3.9e-2 s | 0.25 s | 6.5e-3 s |
| 1e3 | 1.7e-2 s | 8.2e-2 s | 0.43 s | 0.43 s |
| 1e5 | 2.1 s | 5.1 s | 16.5 s | 30 s |

## Defect 1: traced bisection count = maxiter (100) instead of 43

`galpy.backend.optimize.n_bisect_steps` computes the halving count from the
CONCRETE bracket width; under jit the bracket `lo = zeros_like(target)` is a
tracer, so it falls back to `maxiter = 100`. Eagerly the same call does
ceil(log2(2 pi / 2e-12)) = 43. The bracket here is structurally [0, 2 pi], so
the 57 extra halvings are pure waste. A/B (in-process monkeypatch of
`n_bisect_steps` to 42, nothing else changed):

| N | jit, traced count (100) | jit, 42 halvings | max |diff| vs numpy (both arms) |
|---|---|---|---|
| 100 | 5.3e-2 s | 2.2e-2 s | 3.0e-15 |
| 1000 | 0.131 s | 0.057 s | 1.5e-14 |
| 1e4 | 1.005 s | 0.428 s | 1.5e-14 |

=> 2.35x speed-up under jit with identical accuracy. Generic fix: let callers
pass the (static) bracket width / halving count, or have brentq compute it from
a concrete width even when the bracket array is traced (here the width is a
Python constant 2 pi). This affects every backend `brentq` caller with a
traced bracket, not only this class.

## Defect 2: bisection instead of the numpy path's safeguarded Newton

The numpy evaluation (`_mm_tau_of_angle`) uses bracketed Newton on the
closed-form d(angle)/d(tau), started at tau = angle (within a few percent of
the root), converging in a handful of iterations. The backend path does 43
(eager) / 100 (jit) kernel passes. Measured: numpy N=1e3 1.1e-2 s, N=1e4
0.16 s vs jax jit 0.13 s / 1.0 s and jax eager 0.27 s / 1.5 s. A fixed
schedule of ~6-8 safeguarded Newton steps (branch-free, `where`-guarded,
jit/vmap compatible), followed by the existing one-step implicit-function
polish for the gradient, would cut the backend evaluation by another ~5-7x.

## Reproducer

```python
# CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONPATH=/var/tmp/fbp python repro.py
import time, numpy, jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import galpy.backend.optimize as gopt
from galpy.actionAngle import actionAngleVerticalInverse
from galpy.potential import IsothermalDiskPotential

aA = actionAngleVerticalInverse(pot=IsothermalDiskPotential(amp=1., sigma=0.5),
                                Es=numpy.linspace(0.05, 2, 31), setup_interp=True)
def tm(fn, *a):
    jax.block_until_ready(fn(*a)); ts = []
    for _ in range(3):
        t0 = time.perf_counter(); jax.block_until_ready(fn(*a))
        ts.append(time.perf_counter() - t0)
    return min(ts)
a = jnp.asarray(numpy.linspace(0, 2 * numpy.pi, 1000, endpoint=False))
print("eager count:", gopt.n_bisect_steps(0.0, 2 * numpy.pi, 2e-12, 100))  # 43
f = jax.jit(lambda j, a: aA.xvFreqs(j, a))
print("jit, traced count :", tm(f, jnp.asarray(0.5), a))          # ~0.13 s
gopt.n_bisect_steps = lambda a, b, xtol, maxiter: 42
f = jax.jit(lambda j, a: aA.xvFreqs(j, a))
print("jit, 42 halvings  :", tm(f, jnp.asarray(0.5), a))          # ~0.057 s
print("numpy             :", tm(lambda: aA.xvFreqs(0.5, numpy.asarray(a))))  # ~0.011 s
```
