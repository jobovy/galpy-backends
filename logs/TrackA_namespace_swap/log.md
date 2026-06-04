# Track A — potential namespace-swap fan-out (5 parallel agents, worktrees)

Orchestrated via the Workflow tool: 5 worktree-isolated agents, one per family, each
migrating its family to backend-agnostic compute, writing its own test module,
validating numpy byte-identity + jax/torch parity + grad-vs-FD, committing to its own
`feat/P2.x-*` branch. ~700k subagent tokens, ~34 min wall-clock. All coordinated
correctly (each rebased/reset onto feat/backends for get_namespace; none touched the
shared conventions allowlist).

## PRs (drafts into feat/backends / #892)
| PR | Family | Migrated | Tests | Notes |
|----|--------|----------|-------|-------|
| #893 | spherical | SphericalPotential base + 12 | 95 pass/15 skip | caught torch-`power`/`fabs` bugs |
| #894 | disk | MN, Kuzmin, FlattenedPower, IsothermalDisk, KG, RazorThin(surfdens) | 138 pass | MN isinstance fast-path gated on `xp is numpy` |
| #895 | ellipsoidal | EllipsoidalPotential base + GL quad + 7 triaxials | 258 pass | cache gated numpy-only; traced path pure |
| #896 | halobar | all 10 (DehnenBar 30+ mutations, SpiralArms, …) | 60 pass | nothing deferred; flagged pre-existing CosmphiDisk bug |
| #897 | dissipative | 0 (all need scipy.special) | 6 pass/2 xfail | scaffold + convention; deferred to Pspecial |

## Deferred to Pspecial (scipy.special/interpolate) — the P2.5 batch
TwoPower/PowerSpherical/NFW `_evaluate` (hyp2f1/gamma/xlogy), Einasto, interpSpherical,
Ring (ellipk/e), DoubleExp, RazorThin forces (Bessel), various `_surfdens` (complex-sqrt
branch), Chandrasekhar/FDM dyn-fric (erf/sici), KuzminKutuzov (coords transform).

## CI wiring
- feat/backends backend job TEST_FILES set to glob `tests/test_backend_*.py` (commit acfa0e64),
  so each merged family's test module auto-runs under jax+torch. Family PR CI uses the
  merge with feat/backends (glob), so their family tests run in their PR CI.

## Findings worth carrying to the paper
- torch's array-api-compat namespace lacks `power`/`fabs` → use `**`/`xp.abs` (portable).
- The safe-denominator `xp.where` idiom is essential: a dead `0**negative`/`0/0` branch
  poisons reverse-mode grads otherwise.
- Mutable per-instance caches (Ellipsoidal, SoftenedNeedleBar, SpiralArms self-state) must be
  gated on `xp is numpy`; traced path computes pure locals.
- Pre-existing bug surfaced: CosmphiDiskPotential._Rphideriv (R<rb branch) off by factor m.

## TODO (follow-ups)
- Extend tests/test_backend_conventions.py MIGRATED allowlist to the migrated families
  (after each merges into feat/backends — can't do before, the unmigrated source would fail the guard).
- P2.5 (scipy.special-backed) after Pspecial.
- P2.6 wrappers last.
- Separate fix PR for the CosmphiDisk _Rphideriv bug.
