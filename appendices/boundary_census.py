#!/usr/bin/env python3
"""Count ``@backend_input`` boundary crossings per workload, and per caller.

Why this exists
---------------
A ``@backend_input``-decorated leaf costs ~178 us per call under a forced
backend (measured 2026-08-07; 1.3 us undecorated, 2.8 us decorated-on-numpy).
That is fine at normal call counts and catastrophic in a loop: ``diskdf``'s
scipy-quad path crossed the boundary **133,332 times per ``oortA()``**, which
was ~24 s of pure boundary and slowed the routine eager ``backend jax`` shard
2.8x while staying green (galpy #1261, fixed in #1268).

Nothing in the test suite notices that class of regression -- the values stay
correct, so no assertion fires, and the only symptom is time. This script is
the detector: it counts the crossings directly.

Reading the output
------------------
The signature of the bug is a crossing count in the **tens of thousands**. A
healthy call site is single digits. After #1268 an empirical sweep over five
workloads found nothing above 4.

**The wall-clock column is not the finding** and is printed only for context. A
sub-millisecond numpy call against a first-call jax trace/compile shows ratios
in the thousands and means nothing; the flat per-call backend overhead is
tracked separately. Judge by ``crossings``.

Usage
-----
    python boundary_census.py                 # built-in workloads
    python boundary_census.py --backend torch
    python boundary_census.py --top 20        # per-caller ranking depth

To census your own code, import ``census`` and wrap a callable::

    from boundary_census import census
    n, secs, callers = census(lambda: my_df.some_moment(1.0), backend="jax")

Caveats, stated
---------------
* Only ``coerce_coords`` crossings are visible. A path that reaches a backend
  without going through it is invisible here.
* Built-in workloads are a sample, not a proof of absence.
* Run it with the C extension built, or orbit-integrating workloads fall back to
  Python and the timings (not the counts) become meaningless.
"""

from __future__ import annotations

import argparse
import collections
import sys
import time


def _install_spy(counter):
    """Patch ``coerce_coords`` to count non-numpy crossings by caller."""
    import numpy

    import galpy.backend._input as _bi

    real = _bi.coerce_coords

    # Frames to walk past when attributing a crossing. galpy/backend/ is the
    # boundary itself; util/conversion.py is the units decorator that wraps
    # nearly every public method, so blaming it says only "a decorated method
    # was called" -- true of everything, and useless for finding the caller
    # that actually loops.
    _SKIP = ("galpy/backend/", "galpy/util/conversion.py")

    def spy(xp, *coords):
        if xp is not numpy:
            f = sys._getframe(1)
            while f is not None and any(s in f.f_code.co_filename for s in _SKIP):
                f = f.f_back
            where = "?"
            if f is not None:
                name = f.f_code.co_name
                # A decorator's inner function is usually called "wrapper"; if we
                # landed on one anyway, keep the line number so it is locatable.
                if name in ("wrapper", "<lambda>"):
                    name = f"{name}:{f.f_lineno}"
                where = f"{f.f_code.co_filename.split('galpy/')[-1]}:{name}"
            counter[where] += 1
        return real(xp, *coords)

    _bi.coerce_coords = spy
    return lambda: setattr(_bi, "coerce_coords", real)


def census(fn, backend="jax"):
    """Run ``fn`` under a forced backend; return (crossings, seconds, callers)."""
    import galpy.backend as gb

    counter = collections.Counter()
    restore = _install_spy(counter)
    try:
        t0 = time.perf_counter()
        with gb.use(backend, force=True):
            fn()
        secs = time.perf_counter() - t0
    finally:
        restore()
    return sum(counter.values()), secs, counter


def _workloads():
    from galpy.df import dehnendf, isotropicHernquistdf
    from galpy.potential import (
        HernquistPotential,
        MiyamotoNagaiPotential,
        NFWPotential,
    )

    mp = MiyamotoNagaiPotential(a=0.5, b=0.05, normalize=1.0)
    nfw = NFWPotential(a=2.0, normalize=1.0)
    hp = HernquistPotential(a=1.0, normalize=1.0)
    ddf = dehnendf(beta=0.0, profileParams=(1.0 / 3.0, 1.0, 0.1))
    return [
        ("diskdf.oortA", lambda: ddf.oortA(1.0)),
        ("diskdf.sigmaR2", lambda: ddf.sigmaR2(1.0)),
        ("diskdf.surfacemass", lambda: ddf.surfacemass(1.0)),
        ("Potential.mass", lambda: nfw.mass(1.5)),
        ("Potential.surfdens", lambda: mp.surfdens(1.0, 0.5)),
        ("Potential.vesc", lambda: mp.vesc(1.0)),
        (
            "isoHernquistdf.vmomentdensity",
            lambda: isotropicHernquistdf(pot=hp).vmomentdensity(1.0, 0, 0),
        ),
        ("isoHernquistdf.sigmar", lambda: isotropicHernquistdf(pot=hp).sigmar(1.0)),
    ]


# A crossing count at or above this is the #1261 signature, not noise.
HOT = 1000


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", default="jax", help="forced backend (jax/torch)")
    ap.add_argument("--top", type=int, default=10, help="per-caller ranking depth")
    args = ap.parse_args(argv)

    import galpy

    print(f"PROVENANCE: {galpy.__file__}")
    try:
        from galpy.orbit.integrateFullOrbit import _ext_loaded

        print(f"C-ext loaded: {_ext_loaded}")
        if not _ext_loaded:
            print("  WARNING: no C extension -- timings are not comparable")
    except Exception:  # pragma: no cover
        pass

    total = collections.Counter()
    hot = []
    print(f"\n  {'workload':32s} {'crossings':>10s} {'seconds':>9s}")
    for name, fn in _workloads():
        try:
            n, secs, callers = census(fn, backend=args.backend)
        except Exception as e:
            print(f"  {name:32s} SKIP ({type(e).__name__}: {str(e)[:34]})")
            continue
        flag = "  <-- HOT" if n >= HOT else ""
        print(f"  {name:32s} {n:10d} {secs:8.2f}s{flag}")
        if n >= HOT:
            hot.append((name, n))
        total.update(callers)

    if total:
        print(f"\n  top {args.top} callers by crossings:")
        for site, n in total.most_common(args.top):
            print(f"    {n:9d}  {site[:70]}")
    else:
        print("\n  no crossings recorded at all")

    print()
    if hot:
        print(f"  {len(hot)} workload(s) at or above {HOT} crossings -- investigate:")
        for name, n in hot:
            print(f"    {name}: {n}")
        return 1
    print(f"  no workload reached {HOT} crossings (max "
          f"{max(total.values()) if total else 0}). Sampled, not exhaustive.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
