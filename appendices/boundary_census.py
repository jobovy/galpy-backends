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
healthy call site is single digits to low tens.

Sweep of the built-in workloads, 2026-08-08, C extension loaded, x64 on, with
the planar (galpy #1271), vertical (#1272) and actionAngleVertical-closure
(#1273) fixes applied:

    jax    aAVertical 0/0 · aAAdiabatic 9 · aASpherical 5/23 · aAStaeckel 0
           every other df/potential/orbit/qdf workload <= 112
    torch  same shape, max 10
    residual: interpRZPotential only (task #211), 3840 here / 30603 default grid

``actionAngleSpherical`` was the original outlier at 989/1023: it runs a scipy
quadrature loop that evaluates the potential ~1000 times per call, one scalar
crossing each. It went to **5/23** when #1271 stopped the planar adapters
forwarding through the decorated 3D API, and its ``KNOWN_HOT`` entry was deleted
-- that deletion is how the fix is locked in. The comparison that localised it
originally: ``actionAngleStaeckel``, which does the same job vectorized, scored
**0** throughout.

How big is the rest of the problem? (sweep, 2026-08-08)
-------------------------------------------------------
A static scan found 96 internal call sites reaching a decorated entry point, 34
of them inside a loop -- a tempting library-wide sweep. Measuring 27 previously
uncensused workloads (potential root-finders, orbit derived quantities, two more
action-angle families, five more DFs, streams) instead showed the cost is
**concentrated, not diffuse**: with the planar (galpy #1271) and vertical
(#1272) adapter fixes in, every one of the 27 scored <= 153 except
``actionAngleAdiabatic``, and its residual was the *already-identified*
``actionAngleVertical`` closure (#210).

Measured through the three fixes, C extension loaded, x64 on::

    workload                pre-#1271   +#1271/#1272   +#210
    aAAdiabatic.actions          4166            251       9
    aAVertical.Jz                1708            244       0

``actionAngleAdiabatic`` was never a target -- it is a 460x improvement that
falls out of fixing the two adapters plus the closure. That is the argument
against the sweep: the candidates are mostly cold, and the hot ones share a
handful of root causes.

**But a workload sweep only covers sites its workloads reach.** A static AST
scan (``candidates.py``) found **99** loop- or closure-resident call sites, and
the first sweep had touched none in ``interpRZPotential``, ``SCFPotential``,
``EllipsoidalPotential``, ``CompositePotential`` or
``ChandrasekharDynamicalFrictionForce``. Censusing those families found 12 of 13
cold (<= 19) -- and one that is not:

    interpRZPotential(interpPot=True), default 101x101 grid   30603 crossings
    ... with interpDens=True as well                         ~102000

``__init__`` builds its grids with nested ``for ii: for jj:`` loops calling the
DECORATED ``evaluatePotentials`` once per point. Crossings are **exactly linear
in grid points** (10.00/point, measured over 4 grid sizes), which is why a toy
6x4 grid scores a harmless 240 and hides it completely. Sized by galpy's own
default, this is the second-largest count measured anywhere, behind only the
``diskdf`` case that started all of this. Tracked as galpy task #211.

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

    # Decorator boilerplate. galpy's decorators all name their inner function
    # "wrapper", and blaming one says only "a decorated function was called" --
    # true of nearly every call, and useless for finding the caller that loops.
    # Measured 2026-08-08: without this, 244 of the residual vertical-adapter
    # crossings were attributed to Potential.py:wrapper:235, which is just
    # potential_list_of_potentials_input.
    _SKIP_NAMES = ("wrapper", "wrapped")

    def spy(xp, *coords):
        if xp is not numpy:
            f = sys._getframe(1)
            while f is not None and (
                any(s in f.f_code.co_filename for s in _SKIP)
                or f.f_code.co_name in _SKIP_NAMES
            ):
                f = f.f_back
            where = "?"
            if f is not None:
                name = f.f_code.co_name
                if name == "<lambda>":
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
    from galpy.actionAngle import (
        actionAngleAdiabatic,
        actionAngleSpherical,
        actionAngleStaeckel,
        actionAngleVertical,
    )
    from galpy.df import dehnendf, isotropicHernquistdf, quasiisothermaldf
    from galpy.orbit import Orbit
    from galpy.potential import (
        HernquistPotential,
        interpRZPotential,
        MiyamotoNagaiPotential,
        MWPotential2014,
        NFWPotential,
    )

    mp = MiyamotoNagaiPotential(a=0.5, b=0.05, normalize=1.0)
    nfw = NFWPotential(a=2.0, normalize=1.0)
    hp = HernquistPotential(a=1.0, normalize=1.0)
    ddf = dehnendf(beta=0.0, profileParams=(1.0 / 3.0, 1.0, 0.1))
    aAS = actionAngleStaeckel(pot=MWPotential2014, delta=0.4)
    aASph = actionAngleSpherical(pot=hp)
    from galpy.potential import toVerticalPotential

    aAVert = actionAngleVertical(pot=toVerticalPotential(MWPotential2014, 1.0))
    aAAd = actionAngleAdiabatic(pot=MWPotential2014, c=False)
    qdf = quasiisothermaldf(
        1.0 / 3.0, 0.2, 0.1, 1.0, 1.0, pot=MWPotential2014, aA=aAS, cutcounter=True
    )
    o = Orbit([1.0, 0.1, 1.1, 0.0, 0.1, 0.0])
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
        # actionAngle / qdf / orbit: added 2026-08-08. Their absence is why the
        # aASpherical site below went unmeasured until then.
        ("aAStaeckel.actionsFreqs", lambda: aAS.actionsFreqs(1.0, 0.1, 1.1, 0.0, 0.1)),
        ("aASpherical.actions", lambda: aASph(1.0, 0.1, 1.1, 0.0, 0.1)),
        (
            "aASpherical.actionsFreqs",
            lambda: aASph.actionsFreqs(1.0, 0.1, 1.1, 0.0, 0.1),
        ),
        ("aAVertical.Jz", lambda: aAVert(0.1, 0.05)),
        ("aAVertical.actionsFreqs", lambda: aAVert.actionsFreqs(0.1, 0.05)),
        ("qdf.density", lambda: qdf.density(1.0, 0.1)),
        ("qdf.sigmaR2", lambda: qdf.sigmaR2(1.0, 0.1)),
        ("Orbit.E", lambda: o.E(pot=MWPotential2014)),
        ("Orbit.L", lambda: o.L()),
        # Added 2026-08-08 after an exploratory sweep of 27 previously-uncensused
        # workloads. Only these three earned a permanent slot; the other 24 all
        # scored <= 153 and would just pad the run. See the module docstring.
        ("aAAdiabatic.actions", lambda: aAAd(1.0, 0.1, 1.1, 0.0, 0.1)),
        ("Potential.rguiding", lambda: o.rguiding(pot=MWPotential2014)),
        ("Potential.rE", lambda: nfw.rE(-1.0)),
        # Added 2026-08-08: grid construction is itself the loop. Crossings are
        # exactly linear in grid points (measured 10/point with
        # interpPot+interpDens), so a toy grid HIDES this -- galpy's default
        # 101x101 grid crosses 30603 times for interpPot alone.
        ("interpRZPot.build(24x16)", lambda: interpRZPotential(
            RZPot=mp, rgrid=(0.5, 1.5, 24), zgrid=(0.0, 0.2, 16),
            interpPot=True, interpDens=True, use_c=False)),
    ]


# A crossing count at or above this is the #1261 signature, not noise.
HOT = 1000

# Sites already known to be hot, with the tracking item. These still print and
# still get flagged in the output, but they do NOT set a failing exit code --
# otherwise the detector is permanently red and stops being a regression signal.
# Removing an entry here is how a fix gets locked in.
KNOWN_HOT = {
    # interpRZPotential builds its grids with nested `for ii: for jj:` loops
    # calling the DECORATED evaluatePotentials once per grid point (L270-272 and
    # the same shape at 295/320/348/370/392/414, plus per-r comprehensions at
    # 449/470/493/525). Crossings are exactly linear in grid points, so the
    # default 101x101 grid pays 30603 (interpPot) to ~102k (interpPot+interpDens).
    # The census uses a small grid, which under-reports by design -- keep the
    # default-grid number in mind when judging the fix.
    # Measured 2026-08-08: the DEFAULT case needs no new API. interpPot and
    # interpRforce go 1152 -> 0 (24x16) and 30603 -> 0 (default 101x101) by
    # calling the already-existing _evaluatePotentials / _evaluateRforces --
    # three call sites, 9.69 s -> 4.92 s, grid sha256 unchanged. Only
    # interpDens / interpR2deriv / interpvcirc need the four undecorated inners
    # that do not exist yet, so do NOT start by writing those.
    "interpRZPot.build(24x16)": "galpy task #211: 3 sites use existing inners; dens/2nd-deriv need new ones",

    # (aAVertical entries deleted 2026-08-08 when galpy #1272 merged: the vertical
    # adapter fix took them 1708/1722 -> 244/246, below the HOT threshold.
    # Deleting the entry is how the fix gets locked in -- if they ever climb back
    # over 1000 the detector fails, which is the point.)
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", default="jax", help="forced backend (jax/torch)")
    ap.add_argument("--top", type=int, default=10, help="per-caller ranking depth")
    args = ap.parse_args(argv)

    # Match the test suite's dtype, or jax silently truncates to float32 and the
    # run measures a configuration nobody ships.
    if args.backend == "jax":
        try:
            import jax

            jax.config.update("jax_enable_x64", True)
        except ImportError:  # pragma: no cover
            pass

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
    counts = []
    print(f"\n  {'workload':32s} {'crossings':>10s} {'seconds':>9s}")
    for name, fn in _workloads():
        try:
            n, secs, callers = census(fn, backend=args.backend)
        except Exception as e:
            print(f"  {name:32s} SKIP ({type(e).__name__}: {str(e)[:34]})")
            continue
        flag = ""
        if n >= HOT:
            flag = "  <-- HOT (known)" if name in KNOWN_HOT else "  <-- HOT"
            hot.append((name, n))
        print(f"  {name:32s} {n:10d} {secs:8.2f}s{flag}")
        if name not in KNOWN_HOT:
            counts.append(n)
        total.update(callers)

    if total:
        print(f"\n  top {args.top} callers by crossings:")
        for site, n in total.most_common(args.top):
            print(f"    {n:9d}  {site[:70]}")
    else:
        print("\n  no crossings recorded at all")

    print()
    new_hot = [(name, n) for name, n in hot if name not in KNOWN_HOT]
    for name, n in hot:
        if name in KNOWN_HOT:
            print(f"  known hot: {name}: {n} crossings -- {KNOWN_HOT[name]}")
    if new_hot:
        print(f"  {len(new_hot)} NEW workload(s) at or above {HOT} crossings:")
        for name, n in new_hot:
            print(f"    {name}: {n}")
        return 1
    # Compare like with like: the threshold is per workload, so report the
    # largest *workload* count, not the largest per-caller total.
    print(f"  no new workload reached {HOT} crossings (max unknown-hot workload "
          f"{max(counts, default=0)}). Sampled, not exhaustive.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
