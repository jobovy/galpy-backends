# Appendix E — Coverage burndown & the tolerance-relaxation ledger

## Why / what
Two running records that together answer "how complete is the backend migration, and where
did we have to give up bit-for-bit agreement and why":

1. **Coverage burndown** — the `tests/backend_xfail.txt` ledger size *is* the migration's
   progress metric ("where we stand"). A test is in the ledger iff it does not yet pass under
   `xfail(strict=False)` on a backend; as a subsystem is ported its entries flip to pass and
   are pruned at milestones (never per-integration — flaky pass↔timeout luck would bake in
   un-ledgered reds). `backend_slow_skip.txt` is a *separate*, held list of deferred-slow tests
   (skipped in both modes, not regenerated until a vectorization milestone).

2. **Tolerance-relaxation ledger** — every place a backend test compares to the numpy/C
   reference at a *looser* tolerance than the default, with the justification. Almost all
   backend code is **numpy byte-identical**; the loosened cases are a short, deliberately
   documented list (quadrature-method differences, adaptive-vs-fixed solvers, FD-limited
   gradient checks) — not silent regressions. Recording them here keeps "loosened on purpose"
   distinct from "drifted".

---

## 1. xfail-ledger burndown trajectory

Counts are `grep -c '^<backend> ' tests/backend_xfail.txt` (xfail entries) plus the held
`backend_slow_skip.txt` list. "Total ledger" = xfail + slow-skip.

| date | point | jax xfail | jax slow-skip | **jax total** | torch xfail | torch slow-skip | **torch total** |
|------|-------|-----------|---------------|---------------|-------------|-----------------|-----------------|
| 2026-06-16 | project-start baseline (master plan) | 296 | 28 | **324** | 1670 | — | **1670** |
| 2026-06-17 | #969 milestone-1 prune (before → after) | 296 → 253 | 39 | 335 → **292** | 1683 → 1484 | 1 | 1684 → **1485** |
| 2026-06-18 | `feat/backends` HEAD (pre #974/#975) | **254** | 39 | **293** | **1494** | 1 | **1495** |
| 2026-06-19 | post #974/#975 merge + rebase onto #977 (coarse) | **298** | 39 | **337** | **1589** | 1 | **1590** |
| 2026-06-19 | **⟂ granularity refined → per-potential** (#978) | **284** | 39 | **323** | **2072** | 1 | **2073** |

Verification (pre-#974/#975): `git show <ref>:tests/backend_xfail.txt | grep -c '^torch '`
→ 1494; `'^jax '` → 254. `backend_slow_skip.txt`: jax 39, torch 1.

**⟂ GRANULARITY DISCONTINUITY (2026-06-19, #978).** #977 parametrized the 12
`test_potential.py` loop-tests; #978 then regenerated the `test_potential` portion of the
ledger at full per-`(potential × test)` granularity (`GALPY_BACKEND_XFAIL_REGEN`):
`test_potential` jax **19→5**, torch **135→618**. The torch **+483** is a *measurement-
resolution change, NOT a regression* — the old coarse entries prefix-matched every potential,
so one entry hid 618 real failures (+598 passes). jax **−14** because it was *over*-ledgered
(coarse entries marked mostly-passing tests xfail).
**Handling of prior measurements:** the pre-#977 rows above are kept as-is but are a **coarse
undercount** for `test_potential` (cannot be recomputed at fine granularity — the parametrized
tests did not exist at those commits). The burndown therefore has a labeled step here; the
per-potential count is the true metric from this point forward. Only `test_potential` changes
granularity — the rest of the ledger (test_orbit/test_actionAngle/test_sphericaldf/…) is
continuous across the discontinuity. The g13 burndown figure should plot the pre-#977 series
dashed (coarse) up to this marker and the per-potential series solid after.

**Reading the trajectory.**
- The project-start baseline (324 jax / 1670 torch) and #969's "before" numbers (335 / 1684)
  differ because the baseline is the 2026-06-16 master-plan snapshot, whereas #969's "before"
  is the live pre-prune ledger at milestone-1 time (a few triage adds later, and slow-skip had
  grown 28→39 on jax). Both are real; they are different timestamps, not a contradiction.
- The **−199 torch drop** at #969 is dominated by **#960** (central coordinate coercion — the
  systemic torch scalar-input rejection fix), plus the landed P2.x potential migrations and
  Pspecial. torch ≫ jax throughout precisely because that scalar-rejection bug was torch-only.
- The **−43 jax drop** at #969 is the SCF/Multipole vectorization (#959 un-deferred 31 jax SCF
  tests) and the potential tail.
- Branch-local triage on the open PRs adds ledger entries that land on merge: **#974 +16**
  (orbit accessors), **#975 +50** (potential convenience/functional interface). These are *not*
  yet in the `feat/backends` HEAD counts above.

**Milestone policy** (see lab-notebook memory `ledger-milestone-burndown`): regen + review the
add/drop diff only at points where the ledger has *permanently* moved (a subsystem ported), not
per-#892-integration and not on the weekly maintenance cron. #969 was the first such milestone
("milestone-1"); the next anticipated ones are after Track D orbit (test_orbit/test_orbits
un-defer from slow-skip), then action-angle (Track E), then df per family (Track F).

---

## 2. Tolerance-relaxation ledger

### 2a. In-backend ODE integration (#973, `tests/test_backend_orbit_integrate.py`)

`Orbit.integrate(..., method='diffrax')` (jax) / `method='torchdiffeq'` (torch) route through
`Orbits._integrate_inbackend`, which sets the solver tolerances to **`rtol=atol=1e-12`** —
**matching galpy's C integrators** (`_parse_tol`'s default is `-12·ln 10`, i.e. 1e-12), so the
in-backend solve integrates to the same accuracy as `dop853_c`. (Note: the briefing for this
appendix said "1e-8"; the actual merged code uses 1e-12 — the C-integrator `_parse_tol` default.
1e-8 is the default of the *legacy pure-python* `scipy.integrate` path, not the in-backend path.)

Because the in-backend solver now matches the C tolerance, the parity tests against the trusted
`dop853_c` reference are **tight**:

| test (parity vs `dop853_c`) | rtol | atol |
|---|---|---|
| 3D `method='diffrax'` (jax) | 1e-8 | 1e-9 |
| 3D `method='torchdiffeq'` (torch) | 1e-8 | 1e-9 |
| planar/2D diffrax & torchdiffeq | 1e-8 | 1e-9 |
| numpy-array `ts` into `method='diffrax'` | 1e-8 | 1e-9 |

This is the deliberate tightening: the functional **pilot** (#972,
`tests/test_backend_inbackend_ode.py`) compared at the loose `rtol=1e-6, atol=1e-7` (jax) /
`rtol=1e-5, atol=1e-6` (torch) because that pilot's parity test did not yet pin the solver to
the C tolerance. With #973's in-backend `Orbit.integrate` setting 1e-12, the parity assertions
were tightened to 1e-8/1e-9. (Gradient-vs-FD and jax↔torch cross-checks stay at FD-limited
`rtol≈1e-5..1e-6` — that is an FD accuracy floor, not a solver tolerance.)

**The one documented exception — Quantity-input time.**
`test_..._quantity_time` passes `ts = _TS * units.Gyr`. `_TS = linspace(0, 6, 120)` in
*natural* units integrates ~6 natural time units, but `6 Gyr` parses to **~28× more** natural
time units (1 natural time ≈ 0.036 Gyr), so this is a much longer integration than the other
parity tests. It therefore stays at the looser **`rtol=1e-6, atol=1e-7`**, with the in-test
justification recorded verbatim:

> `_TS` Gyr parses to ~28x more natural time units (1 natural time ~ 0.036 Gyr), so this is a
> much longer integration than the other parity tests; the looser tolerance reflects the larger
> accumulated Dopri8-vs-dop853 difference (the point of this test is the Quantity parsing, not
> tight parity).

i.e. the loosening is the larger accumulated `Dopri8`-vs-`dop853` difference over the longer
arc; the test exists to check Quantity parsing, not tight integrator parity.

### 2b. tdyn eager-value tolerance (#975, `tests/test_backend_potential_convenience.py`)

Once `mass()` dispatches through `galpy.backend.quadrature` (fixed-order Gauss–Legendre) for
potentials **without** an analytic `_mass`, the convenience method `tdyn = 2π·R·sqrt(R/M)` (which
closes over `mass()`) can no longer be byte-identical to the numpy path, which uses scipy's
**adaptive** quad. The eager-value tolerance is therefore loosened for `tdyn` only:

```
_DEFAULT_VALUE_RTOL = 1e-10        # every other convenience method (byte/round-trip identical)
_VALUE_RTOL = {"tdyn": 1e-6}       # GL-vs-adaptive quadrature difference (~5e-8 on MiyamotoNagai)
```

The in-test note records this as a **quadrature-method difference, not a detachment/regression**:
the numpy path uses scipy's adaptive quad, the backend path uses fixed-order GL, so the two agree
only to the GL-vs-adaptive quadrature error (~5e-8 on the MiyamotoNagai test case). The
discriminating eager-array assertions (that jax/torch return a backend array) stay strict for
every method; `mass`'s own dedicated tests use the same loosened tolerance for the same reason.
(`tdyn` and `rtide` are also excluded from the AD-vs-FD checks in this module — their gradients
are verified in their own focused modules, mass in `test_backend_potential_rootfind.py` and
rforce in `test_backend_force.py` — but that is a coverage placement, not a tolerance relaxation.)

---

## Source PRs

- **#969** (merged 2026-06-17) — milestone-1 ledger regen prune (jax 335→292, torch 1684→1485).
- **#972** (merged 2026-06-17) — Track D functional STM interface; the loose-tolerance pilot.
- **#973** (merged 2026-06-18) — in-backend `Orbit.integrate` diffrax/torchdiffeq; tightened
  parity to 1e-8/1e-9, with the Quantity-time exception.
- **#974** (OPEN → feat/backends) — orbit accessors; ledger +16 (branch-local).
- **#975** (OPEN → feat/backends) — potential convenience/functional interface; tdyn
  GL-vs-adaptive loosening; ledger +50 (branch-local).
