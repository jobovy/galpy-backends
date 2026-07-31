# Vacation status snapshot — 2026-07-31

**This file is OVERWRITTEN, not appended.** It is the "where are things now" answer.
`log.md` is the trace; read this first and go there only for the why.

---

## Ledger (the burndown metric)

```
156 -> 150 -> 130 -> 128 -> 122 -> 120 -> 109 -> 107 -> 101   merged
```

**101 on `feat/backends-develop`**, split `jax 43 / jax-jit 5 / torch 52 / torch-jit 1`.

By file (post-#1231 numbers):

```
61  test_actionAngle.py     <- 48 are actionAngleVerticalInverse
10  test_quantity.py        <- mostly the tolerance cluster
 5  test_streamTrack.py   4  test_orbit.py   4  test_interp_potential.py
 3 each: streamdf, sphericaldf, potential, diskdf
 2  test_qdf.py   2  test_noninertial.py   1  test_SpiralArmsPotential.py
```

**Verified composition** (enumerated from the tree, not inferred by subtraction):

| bucket | n | gate |
|---|---|---|
| `actionAngleVerticalInverse` | 48 | scope decision |
| tolerance cluster | 21 | bounds decision |
| RazorThin `-jit` poisson_surfdens | 2 | #181 decision |
| **workable now** | **30** | — |

So **71 of 101 are decisions, not work.** The 30 that are work:

```
 5  streamTrack (torch)      cov_basis, cov_per_call_unit_overrides,
                             out_of_range_returns_nan, plot_smoke,
                             plot_spread_non_cartesian
 4  interp_potential (jax)   force_c, secondderivs, +2 jax-jit use_c   -> #112
 4  orbit                    analytic_ecc_rperi_rap x2, analytic_zmax,
                             jax-jit eccentricity
 3  diskdf (jax)             dehnendf_sample_* x3
 3  streamdf (torch)         fardalwmwpot_trackaa, plotting,
                             streamTrack_multi_parallel  (-> #178 fork deadlock)
 3  sphericaldf              anisotropic_hernquist_negdf (jax),
                             constantbeta_interpolatedpotentials x2 (torch)
 2  actionAngle              AdiabaticGrid_outsidegrid_c (jax+torch) -> #103
 2  qdf                      call_diffinoutputs (jax+torch)
 2  noninertial (jax)        arbitraryaxisrotation, _omegafunc      -> #61
 1  potential (jax-jit)      2ndDeriv RazorThin
 1  SpiralArms (jax)         test_Rzderiv
```

**The 8 torch stream entries are NOT one cluster** (measured 2026-07-31, not
assumed): 4 are a deliberate `NotImplementedError` — backend `cov()` supports
only `basis='galcenrect'`, the sky/cyl-frame Jacobian follow-up parked in #861;
1 is a real numpy/Tensor boundary bug (`fromnumeric` `.max(out=, axis=)`); 3
streamdf are mixed, one of them the `parallel_map` fork deadlock (#178).
All 5 (not 4) are gated on it: the boundary-cast one hits the same guard four
lines later. Sized by running the code, not grepping it: every one of the six
`coords.*_jac` helpers is pure-numpy (~433 lines, 49 numpy refs, zero xp), and
`_analytical_jacobian` adds 13 `float()` casts + 6 in-place row-scalings. Splits
as **PR-A** (coords `*_jac` keystone, numpy byte-identical) then **PR-B**
(streamTrack sky-basis cov, batched `xp.where`) — the same keystone-then-consumer
shape as Streams PR-1. Retires 5 entries and makes sky-frame stream covariance
differentiable, which Track G's showpiece needs.

---

## Traced (`--jit`) coverage — the mission's headline

| file | jax | torch |
|---|---|---|
| `test_potential.py`   | 1326 / 0 (338s) | 1326 / 0 (1792s) |
| `test_orbit.py`       | 581 / 3 | 581 / 0 |
| `test_actionAngle.py` | 209 / 0 (166s) | 209 / 0 (168s) |
| `test_sphericaldf.py` | 223 / 0 | 223 / 0 |
| `test_diskdf.py`      | 129 / 0 (678s) | 129 / 0 (685s) |
| `test_qdf.py`         | 0 fail | 56 / 0 (232s) |

Every major suite runs traced on both backends with nothing failing, except
`test_orbit` on jax: 2 are 1800s performance timeouts, 1 is a local-only signal
artifact (CI-confirmed passing). Six `-jit` ledger entries still suppress the
RazorThin pair and the interpRZ `use_c` pair.

Cost note: torch traced is at **parity** with jax except on `test_potential`
(1792s vs 338s), where ~1300 small distinct traces make front-end compile cost
dominate. Do not generalise the 5x to "torch tracing is slow".

---

## In flight

| item | state |
|---|---|
| **#1232** scipy 1.18 spline pickling | CI |
| **RazorThin array inputs** | branch uncommitted, blocked on a decision |

## Waiting on you (roughly two thirds of the remaining ledger)

1. **RazorThin traced `surfdens`** (#181). With the array-input flag on, traced
   `surfdens` returns a meaningless ~0 instead of today's loud `TypeError`, because
   a razor-thin disk's density is a delta function that fixed-order Gauss-Legendre
   cannot see (measured: -4.16e-06 vs a true 0.2967679). Options: ship the flag,
   keep the error, or **return NaN** mirroring your earlier `R2deriv` off-plane
   call. Recommend NaN. My part is verified correct and value-neutral: all six GL
   integrands vectorise bit-identically, eager and numpy unaffected.
2. **Tolerance batch** (#171, ~24 entries). One call on torch-vs-numpy bounds;
   magnitudes all measured and recorded.
3. **`actionAngleVerticalInverse` scope** (48 entries). Refuses backends by
   construction; moves only if that class is migrated.

## Notable finding for when you're back

**scipy 1.18 breaks pickling of `PPoly`/`BPoly`/`CubicSpline`** (they cache the
array namespace — a *module* — as `_xp`). Radius measured, not inferred: exactly
2 galpy classes (`MultipoleExpansionPotential`, `DiskMultipoleExpansionPotential`).
interpRZ, interpSpherical, the AA grids, qdf/diskdf/sphericaldf and an integrated
Orbit are all fine. Fix built and verified bit-identical on both scipy versions.
Worth reporting upstream — a class that pickled for years no longer does.
