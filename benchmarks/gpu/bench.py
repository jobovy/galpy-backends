"""GPU vs CPU benchmark for galpy's backends.

One process per (backend, device):  python bench.py <numpy|jax|torch> <cpu|gpu> <out.json>
Each case: one warm-up call (compilation, caches), then the median of 3 timed
calls, synchronized (jax: block_until_ready; torch: cuda.synchronize).
jax cases run under a user-level jax.jit (galpy does not jit internally);
torch cases run eagerly (torch.compile cannot target this GPU, sm_61).
"""

import json
import os
import statistics
import sys
import time

import numpy

backend, device, out = sys.argv[1], sys.argv[2], sys.argv[3]
SIZES = {
    "orbits": [100, 1000, 10000, 100000],
    "forces": [10**4, 10**5, 10**6, 10**7],
    "sample": [10**4, 10**5, 10**6],
    "forcegrad": [10**4, 10**5, 10**6, 10**7],
}
if backend == "jax":
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    assert jax.default_backend() == ("gpu" if device == "gpu" else "cpu")
elif backend == "torch":
    import torch

    torch.set_default_dtype(torch.float64)
    if device == "gpu":
        torch.set_default_device("cuda")

from galpy import backend as gb  # noqa: E402
from galpy.df import isotropicHernquistdf  # noqa: E402
from galpy.orbit import Orbit  # noqa: E402
from galpy.potential import (  # noqa: E402
    HernquistPotential,
    MWPotential2014,
    MiyamotoNagaiPotential,
    NFWPotential,
    PowerSphericalPotentialwCutoff,
    evaluateRforces,
)


def sync(x):
    if backend == "jax":
        jax.block_until_ready(x)
    elif backend == "torch" and device == "gpu":
        torch.cuda.synchronize()
    return x


def timeit(fn):
    sync(fn())  # warm-up (compile)
    ts = []
    for _ in range(3):
        t0 = time.perf_counter()
        sync(fn())
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def arr(x):
    if backend == "jax":
        return jnp.asarray(x)
    if backend == "torch":
        return torch.as_tensor(x)
    return x


rng = numpy.random.default_rng(42)
results = {"backend": backend, "device": device, "cases": {}}


def record(name, n, fn, budget=600.0):
    try:
        t = timeit(fn)
    except Exception as e:  # report, keep going
        t = f"ERR {type(e).__name__}: {str(e)[:120]}"
    results["cases"].setdefault(name, {})[n] = t
    print(f"{backend:5s} {device} {name:9s} n={n:>9d} {t if isinstance(t, str) else f'{t:.4g} s'}", flush=True)
    json.dump(results, open(out, "w"), indent=1)
    return t


# --- orbit integration: N orbits, 201 output times, MWPotential2014
ts = numpy.linspace(0.0, 10.0, 201)
for n in SIZES["orbits"]:
    ic = numpy.column_stack(
        [
            rng.uniform(0.8, 1.2, n),
            rng.normal(0.0, 0.1, n),
            rng.uniform(0.9, 1.1, n),
            rng.normal(0.0, 0.05, n),
            rng.normal(0.0, 0.05, n),
            rng.uniform(0.0, 6.0, n),
        ]
    )
    if backend == "numpy":

        def f(ic=ic):
            o = Orbit(ic)
            o.integrate(ts, MWPotential2014, method="dop853_c", progressbar=False)
            return o.getOrbit()

    elif backend == "jax":
        _jf = jax.jit(
            lambda x: (lambda o: (o.integrate(jnp.asarray(ts), MWPotential2014, method="diffrax"), o.getOrbit())[1])(Orbit(x))
        )

        def f(ic=ic, _jf=_jf):
            return _jf(jnp.asarray(ic))

    else:

        def f(ic=ic):
            o = Orbit(torch.as_tensor(ic))
            o.integrate(torch.as_tensor(ts), MWPotential2014, method="torchode")
            return o.getOrbit()

    t = record("orbits", n, f)
    if not isinstance(t, str) and t > 300.0:
        break

# --- force evaluation: MWPotential2014 R-force at N points
for n in SIZES["forces"]:
    R = rng.uniform(0.1, 20.0, n)
    z = rng.uniform(-5.0, 5.0, n)
    if backend == "jax":
        _jf = jax.jit(lambda R, z: evaluateRforces(MWPotential2014, R, z))
        f = lambda R=R, z=z, _jf=_jf: _jf(jnp.asarray(R), jnp.asarray(z))  # noqa: E731
    else:
        Ra, za = arr(R), arr(z)
        f = lambda Ra=Ra, za=za: evaluateRforces(MWPotential2014, Ra, za)  # noqa: E731
    record("forces", n, f)

# --- gradient of the summed force w.r.t. a potential parameter (NFW amp, MN a)
for n in SIZES["forcegrad"]:
    if backend == "numpy":
        break
    R = rng.uniform(0.1, 20.0, n)
    z = rng.uniform(-5.0, 5.0, n)

    def tot(amp, a, xp_R, xp_z):
        pots = [
            PowerSphericalPotentialwCutoff(alpha=1.8, rc=1.9 / 8.0, normalize=0.05),
            MiyamotoNagaiPotential(a=a, b=0.28 / 8.0, normalize=0.6),
            NFWPotential(a=16 / 8.0, amp=amp),
        ]
        return evaluateRforces(pots, xp_R, xp_z).sum()

    if backend == "jax":
        _jg = jax.jit(jax.grad(tot, argnums=(0, 1)))
        f = lambda R=R, z=z, _jg=_jg: _jg(jnp.asarray(4.0), jnp.asarray(3.0 / 8.0), jnp.asarray(R), jnp.asarray(z))  # noqa: E731
    else:
        Ra, za = torch.as_tensor(R), torch.as_tensor(z)

        def f(Ra=Ra, za=za):
            amp = torch.tensor(4.0, requires_grad=True)
            a = torch.tensor(3.0 / 8.0, requires_grad=True)
            return torch.autograd.grad(tot(amp, a, Ra, za), (amp, a))

    record("forcegrad", n, f)

# --- DF sampling: isotropic Hernquist, n positions + velocities
for n in SIZES["sample"]:
    pot = HernquistPotential(amp=2.0, a=1.3)
    if backend == "numpy":
        df = isotropicHernquistdf(pot=pot)
        f = lambda n=n, df=df: df.sample(n=n).r()  # noqa: E731
    else:
        with gb.use(backend, force=True):
            df = isotropicHernquistdf(pot=pot)

        def f(n=n, df=df):
            with gb.use(backend, force=True):
                return df.sample(n=n, key=gb.random.key(1)).r()

    record("sample", n, f)

print("done", out)
