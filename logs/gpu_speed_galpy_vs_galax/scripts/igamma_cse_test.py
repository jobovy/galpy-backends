"""Does XLA CSE duplicate identical gammainc (igamma) calls under jit on GPU?

Three probes:
  1. wall time: single vs duplicate-identical vs duplicate-different-a vs
     duplicate-different-x. If CSE fires, dup-identical ~= single.
  2. StableHLO text (pre-XLA): is igamma one op (chlo/stablehlo) or already
     expanded into while loops at lowering?
  3. optimized HLO text (post-XLA): how many igamma/while computations survive.
"""
import time, re
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from jax.scipy.special import gammainc

key = jax.random.PRNGKey(0)
a = 0.6  # = 1.5 - alpha/2 for MWPotential2014's bulge (alpha=1.8)

def bench(f, x, reps=7):
    f(x).block_until_ready()
    return min(
        (lambda t0: (f(x).block_until_ready(), time.perf_counter() - t0)[1])(time.perf_counter())
        for _ in range(reps)
    )

def counts(f, x):
    low = f.lower(x)
    shlo = low.as_text()
    opt = low.compile().as_text()
    return {
        "stablehlo_igamma": len(re.findall(r"igamma", shlo)),
        "stablehlo_while": len(re.findall(r"while", shlo)),
        "opt_igamma": len(re.findall(r"igamma", opt)),
        "opt_while_comps": len(re.findall(r"^%?while", opt, re.M)) + opt.count("kWhile"),
        "opt_while_str": opt.count("while"),
    }

for n in (100_000, 10_000_000):
    x = jnp.abs(jax.random.normal(key, (n,))) * 2.0 + 0.05
    fs = {
        "single":        jax.jit(lambda x: gammainc(a, x)),
        "dup-identical": jax.jit(lambda x: gammainc(a, x) + gammainc(a, x)),
        "dup-diff-a":    jax.jit(lambda x: gammainc(a, x) + gammainc(a + 0.25, x)),
        "dup-diff-x":    jax.jit(lambda x: gammainc(a, x) + gammainc(a, x * 1.000001)),
    }
    print(f"--- n={n} ---", flush=True)
    for nm, f in fs.items():
        t = bench(f, x)
        c = counts(f, x)
        print(f"{nm:14s}: {t*1e3:8.3f} ms   "
              f"stablehlo[igamma={c['stablehlo_igamma']},while={c['stablehlo_while']}]  "
              f"opt[igamma={c['opt_igamma']},while_str={c['opt_while_str']}]", flush=True)
print("done")
