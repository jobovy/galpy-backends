"""Timing helpers for benchmark figures.

Benchmarks separate *measure* (machine-dependent timings, cached to data/*.json)
from *plot* (byte-stable, reads the JSON). These helpers do the measuring with
proper warmup and device synchronisation so GPU async execution is not mistimed.
"""

import statistics
import time


def _sync_jax(x):
    try:
        import jax

        jax.block_until_ready(x)
    except Exception:
        pass


def _sync_torch():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def time_call(fn, *, warmup=2, repeats=7, sync="none"):
    """Median wall-clock seconds of ``fn()`` over ``repeats`` runs after ``warmup``.

    Parameters
    ----------
    fn : callable
        Zero-argument callable to time; its return value is used for jax sync.
    warmup : int
        Untimed calls first (JIT compile / allocator warmup).
    repeats : int
        Timed calls; the median is reported (robust to outliers).
    sync : {'none', 'jax', 'torch'}
        Force completion of async device work before stopping the clock:
        'jax' blocks on the return value, 'torch' calls cuda.synchronize().

    Returns
    -------
    dict with keys: median, min, mean, std, n, times (list).
    """
    for _ in range(max(0, warmup)):
        out = fn()
        if sync == "jax":
            _sync_jax(out)
        elif sync == "torch":
            _sync_torch()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        if sync == "jax":
            _sync_jax(out)
        elif sync == "torch":
            _sync_torch()
        times.append(time.perf_counter() - t0)
    return {
        "median": statistics.median(times),
        "min": min(times),
        "mean": statistics.fmean(times),
        "std": statistics.pstdev(times) if len(times) > 1 else 0.0,
        "n": len(times),
        "times": times,
    }
