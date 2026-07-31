# Vacation status snapshot — 2026-07-31

**This file is OVERWRITTEN, not appended.** It is the "where are things now" answer.
`log.md` is the trace; read this first and go there only for the why.

---

## Ledger (the burndown metric)

```
156 -> 150 -> 130 -> 128 -> 122 -> 120 -> 109 -> 107 -> 101   merged
```

**101 on `feat/backends-develop`**, split `jax 43 / jax-jit 5 / torch 58 / torch-jit 1`.

By file (post-#1231 numbers):

```
61  test_actionAngle.py     <- 48 are actionAngleVerticalInverse
12  test_quantity.py        <- mostly the tolerance cluster
 5  test_streamTrack.py   5  test_qdf.py   4  test_orbit.py
 4  test_interp_potential.py    3 each: streamdf, sphericaldf, potential, diskdf
 2  test_noninertial.py    1 each: SpiralArms, galpypaper
```

**Two thirds of what remains is decisions, not work** — see "Waiting on you".

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
