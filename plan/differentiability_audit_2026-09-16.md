# Differentiability audit — 2026-09-16

Method: for each entry point, `jax.grad` w.r.t. a parameter one would actually
FIT, compared against a central finite difference that **rebuilds from that
parameter** (not an in-object perturbation -- that distinction is what hid the
70% `real_eig` error for months). Verdicts are measured, not inferred.

## A. Verified differentiable (AD matches a rebuild FD)

    MiyamotoNagai Rforce      d/d(a)        rel 3.8e-11
    NFW dens                  d/d(a)        rel 8.0e-11
    MiyamotoNagai R2deriv     d/d(amp)      rel 3.9e-12
    LogHalo vcirc             d/d(q)        rel 0.0
    Orbit integrate + R       d/d(pot a)    rel 1.3e-06  (Hernquist)
                                            rel 8.9e-06  (LogHalo q)
                                            rel 2.2e-08  (Plummer b)
    actionAngleStaeckel(c=F)  d/d(pot a)    rel 1.7e-10
    actionAngleAdiabatic      d/d(pot a)    rel 1.9e-10
    actionAngleSpherical      d/d(pot a)    rel 2.7e-11
    dehnendf surfacemass      d/d(hr)       rel 6.7e-10
    streamdf __init__         d/d(q)        rel ~1e-2 vs FD -5.008 (after the
                                            real_eig unfreeze; was 70% off)
    streamspraydf                           d/dq 5.7e-8, d/dmass 7.0e-10

So potentials, orbits and action-angle are in good shape, and the ledger agrees
(3 xfail / 5 slow-skip).

## B. NOT differentiable: DF CONSTRUCTION

The consistent pattern is that DF **evaluation** is differentiable -- that is
what the existing 93 grad-vs-FD assertions cover -- and DF **construction** is
not. streamdf is the only DF whose constructor was migrated (Phase C).

    streamgapdf.__init__           TypeError: in-place item assignment
                                   [_gap_progenitor_setup]
    isotropicHernquistdf.dMdE      TracerArrayConversionError
    constantbetaHernquistdf.dMdE   TracerArrayConversionError
    osipkovmerrittHernquistdf.dMdE TracerArrayConversionError
    isotropicNFWdf.dMdE            TracerArrayConversionError
    quasiisothermaldf.__init__     TracerArrayConversionError
    kingdf.__init__                TracerArrayConversionError

This matters because the fit parameters enter the CONSTRUCTOR, so inference
w.r.t. them needs a traceable `__init__`.

streamgapdf scope (73 in-place assignments, 259 numpy calls, 40 backend sites):

    _determine_impact_coordtransform   33
    _interpolate_stream_track_kick     12
    _cart                              12
    _interpolate_stream_track_kick_aA   3
    _gap_progenitor_setup               3

## C. By design, worth documenting rather than fixing

    actionAngleStaeckel(c=True) d/d(pot a)   TracerArrayConversionError @ as_numpy

The C path differentiates w.r.t. the INITIAL CONDITIONS (the C-STM), not w.r.t.
potential parameters -- those need the Python/diffrax path. Not a defect; it
should just be stated in the docs so nobody reads the failure as a bug.

## D. A jit-only blocker CI cannot see

`MiyamotoNagaiPotential._zforce` (and `_z2deriv`, `_Rzderiv`, `_dens`) branch on
`if self._a == 0.0`. Under plain `jax.grad` this is FINE -- grad linearizes with
concrete primals, so the comparison evaluates (measured: zforce d/d(a) rel
8.8e-10). Inside an ODE solve it is compiled, there is no concrete value, and it
raises TracerBoolConversionError -- so a differentiable orbit integration in
MiyamotoNagai w.r.t. `a` fails while Hernquist/LogHalo/Plummer succeed.

CI never runs --jit, so this class is invisible to it. Parameter branches found:

    MiyamotoNagaiPotential                  4   (self._a == 0.0)
    ChandrasekharDynamicalFrictionForce     1   (self._rhm == 0.0)
    MN3ExponentialDiskPotential             1   (self._brd < 0.0)

The rest of the 9 hits are structural integers (_ntab, _M) or a cache key, which
are fine. Fix is the guarded `xp.where` double-trick -- the branch exists to
avoid a 0/0, so a naive where would NaN-poison the gradient.

## E. Deliberate gradient freezes still outstanding

    psd_project eigenvectors (linalg.py)  same class as the real_eig one just
                                          fixed; cost UNMEASURED
    streamTrack frozen assignment         known; floor measured at 3-5e-5
    actionAngleStaeckel u0                cost UNMEASURED

`_jax/staeckel_c.py` and `_jax/adiabatic_c.py`'s stop_gradients are NOT in this
class -- they are the `pure_callback` pattern, where the derivative comes back
from the C-computed Jacobian.

## Suggested order

1. streamgapdf `__init__` (bounded, 5 methods, mirrors the streamdf work).
2. The MiyamotoNagai/ChandrasekharDynamicalFriction parameter branches (small,
   and MN is in MWPotential2014, so it blocks the most common orbit fit).
3. Measure the two unmeasured freezes; decide as we did for real_eig.
4. The other DF constructors (spherical family, qdf, kingdf).
