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
| 2026-06-19 | **⟂ granularity refined → per-potential** (#978) | **293** | 39 | **332** | **2072** | 1 | **2073** |
| 2026-06-19 | #978 onto #980 + new deepcopy test + 9 jax-slow→slow-skip | **284** | 49 | **333** | **2073** | 1 | **2074** |

Verification (pre-#974/#975): `git show <ref>:tests/backend_xfail.txt | grep -c '^torch '`
→ 1494; `'^jax '` → 254. `backend_slow_skip.txt`: jax 39, torch 1.

**⟂ GRANULARITY DISCONTINUITY (2026-06-19, #978).** #977 parametrized the 12
`test_potential.py` loop-tests; #978 then regenerated the `test_potential` portion of the
ledger at full per-`(potential × test)` granularity (`GALPY_BACKEND_XFAIL_REGEN`):
`test_potential` jax **19→14**, torch **135→618**. The torch **+483** is a *measurement-
resolution change, NOT a regression* — the old coarse entries prefix-matched every potential,
so one entry hid 618 real failures (+598 passes). jax **−5** net (was *over*-ledgered: coarse
entries marked mostly-passing tests xfail; the 14 are real per-potential failures).
**Regen pitfall:** the first regen ran `-n auto` (72 jax workers) and OOM-dropped workers →
under-counted jax as 5; a stable `-n 4` re-run (1308 collected, consistent with torch) gave
the true 14. Use a memory-safe worker count for jax regen/validation (see also the CI shard
split + xdist for the post-#977 jax `test_potential` 75-min timeout).
**Handling of prior measurements:** the pre-#977 rows above are kept as-is but are a **coarse
undercount** for `test_potential` (cannot be recomputed at fine granularity — the parametrized
tests did not exist at those commits). The burndown therefore has a labeled step here; the
per-potential count is the true metric from this point forward. Only `test_potential` changes
granularity — the rest of the ledger (test_orbit/test_actionAngle/test_sphericaldf/…) is
continuous across the discontinuity. The g13 burndown figure should plot the pre-#977 series
dashed (coarse) up to this marker and the per-potential series solid after.

**Rebase onto #980 + the deepcopy regression test (2026-06-19).** When `feat/backends` (and
#978 on top of it) was rebased onto main after **#980** merged, two `test_potential` ledger
moves followed — both for the single new test **`test_MultipoleExpansion_deepcopy`** that #980
added:
- **torch xfail +1** (618→619): the new test constructs Multipole/DiskMultipole potentials, and
  torch fails at *construction* (`MultipoleExpansionPotential.py:650` `TypeError`, ~20 s) — the
  same genuine torch-backend SCF/Multipole coefficient-setup failure already behind the 59 torch
  `test_amp_mult_divide[*Multipole*/*SCF*/*Disk*]` entries.
- **jax slow-skip +1** (39→40): under jax the same test *constructs and evaluates correctly* and
  **passes**, but the full deepcopy-and-re-evaluate over 4 potentials × a grid × times × 10
  methods takes **~185 s on the dev box** (jax-eager Multipole eval) → it risks crossing the
  300 s per-test timeout on the slower CI runners (a flaky-timeout red), so it is `slow_skip`,
  not xfail. The numpy run fully covers the actual scipy ≥ 1.18 deepcopy-of-module regression.
  (Consistent with jax having **0** `test_amp_mult_divide` ledger entries: jax handles Multipole
  construction, torch does not.)

  *Measurement caveat (important):* the dev box's py3.13 jax is **CUDA-enabled and GPU-default**;
  a first `-n 4` validation OOM'd the GPU (`CUDA_ERROR_OUT_OF_MEMORY`) and reported ~90 % spurious
  `F`, and the same OOM made this test "time out" at 300 s. CI runs jax on **CPU** — so all
  ledger validation/regen must force CPU (`CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu`); on CPU the
  test is a clean 185 s pass. (Lab-notebook memory `jax-defaults-to-gpu-force-cpu`.)

**Plus a categorization fix surfaced by the same CPU validation: 9 jax `test_potential` tests
moved xfail → slow-skip (xfail −9 / slow-skip +9, jax total unchanged).** The CI-parity full jax
`test_potential.py` validation (CPU, no `--timeout`, so tests run to completion) flagged **9
XPASS** — ledgered-xfail tests that actually *pass* under jax: `test_DehnenBinney98`,
`test_Cautun20`, `test_McMillan17`, `test_diskscf_overflow`, and 5 rotated/tilted-wrapper
`test_poisson_surfdens_potential[…]` cases. They are not stale: each takes **580–2641 s** under
jax-eager (DehnenBinney98 ≈ 44 min; junit-measured), far over the 300 s per-test CI timeout, so
as xfail entries they timed out (→ xfail) on every CI run — but they compute *correct* values
given the time. That is exactly the **slow-skip** category ("correct but pathologically slow until
vectorized"), not xfail. Moving them: (a) is the honest label; (b) reclaims **~190 min/run** of
pure timeout-burn on the jax `test_potential` shards (those 9 had been eating 9 × 300 s every
run) — bringing the 5-way `pytest-split` groups from ~55 min to ~46 min each (75 min ceiling), a
healthier session-timeout margin; (c) torch fails these **fast at construction** (0.0 s), so they
stay in the *torch* xfail-ledger; numpy exercises all of them. Net jax `test_potential`: **5 xfail
+ 15 slow-skip** (was 14 xfail + 6).

**Important — why #980 does NOT drop any existing entries (a misdiagnosis corrected).** It was
initially expected that the 59 torch `amp_mult_divide[*Multipole*/*SCF*]` entries were the
`copy.deepcopy`-of-a-stored-numpy-module failure that #980 fixes, and would therefore *drop*
once #980 was in the base. They do **not**: the dev box is **py3.13, which tolerates deepcopy of
a module**, so the local regen that produced the ledger never saw the deepcopy bug — those 59
entries are *genuine torch construction/eval failures*, unrelated to #980, and re-running the
regen on the #980 base reproduces the identical set (verified: the entries still `xfailed`, not
`xpassed`). The deepcopy-of-module bug is a **py3.14-CI-only** failure mode (scipy ≥ 1.18 stores
the numpy module on its PPoly/BPoly interpolants; py3.14's deepcopy raises `cannot pickle
'module' object`). #980's value for the all-backend suite is therefore:
1. it fixes the **numpy `build.yml` red** on py3.14 (`test_amp_mult_divide[*Multipole*]`), which
   was the actual blocker for #978 going green; and
2. it prevents **un-ledgered py3.14 jax/torch deepcopy reds** for the table-backed potentials
   that the py3.13 local regen could not capture.
Net for #980's new test: no drops, +1 torch xfail, +1 jax slow-skip. (Separately, the CPU
validation recategorized 9 already-ledgered jax tests xfail → slow-skip — see the paragraph
above; that is a label fix, not a burndown change: jax total stays 333.)

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
- **#974** (merged) — orbit accessors; ledger +16 (branch-local).
- **#975** (merged) — potential convenience/functional interface; tdyn
  GL-vs-adaptive loosening; ledger +50 (branch-local).

### Potential closeout wave (#978–#993) — torch `test_potential` burndown

The migrated potentials were already backend-clean; the remaining torch reds were
test-infrastructure / boundary-coercion gaps, not potential-code bugs. Fix PRs carry
**no ledger prunes** (the flipped entries become XPASS under `strict=False`); the prune
happens once in the milestone regen that follows #993.

- **#978** (merged 2026-06-20) — all-backend ledger regen at **per-potential** test_potential
  granularity (this is why torch xfail grew 1485→~2079: each loop-test became many
  per-potential nodeids; ledger size = progress metric, so finer granularity is intended).
- **#981** (merged 2026-06-20) — test_potential legacy asserts backend-tolerant (`numpy.asarray`
  routing of forced-torch outputs; no tolerance change).
- **#982** (merged 2026-06-20) — test_orbit energy/Jacobi conservation backend-tolerant.
- **#983** (merged 2026-06-20) — centralize data-coercion helpers into `galpy.backend._coerce`
  (`coerce_coords` / `promote_scalars` / `as_backend_constant` / `zeros_like_backend`).
- **#984** (merged 2026-06-20) — backend-correct `ttensor` eigenvalues + `zvc_range` no-solution path.
- **#985** (merged 2026-06-20) — `Orbit.L()` cross-product namespace-consistent.
- **#986** (merged 2026-06-20) — pin SCF/Multipole **construction-time** coefficient setup to numpy
  (the #75 coefficient computation stays numpy-pinned; lifting it is a tracked follow-up).
- **#987** (merged 2026-06-20) — centrally coerce forced-backend scalar leaks.
- **#988** (merged 2026-06-20) — pin the 3D variational dxdv driver's host-side coords to numpy.
- **#989** (merged 2026-06-20) — anchor construction-time scalar phi to float64 in EllipsoidalPotential
  (fixes a float32-amp bug exposed while stabilizing the GL sum order).
- **#990** (merged 2026-06-21) — coerce scalar coords for backend-compatible scalar-method potentials;
  **gated on `_check_backend_compatible`** (an unconditional decorator coercion broke unmigrated
  AnyAxiRazorThin — scipy rejects tensors — caught in adversarial review).
- **#991** (merged 2026-06-21) — coerce all-numpy operands in `promote_scalars` under a forced backend;
  restored the `asarray_on_device` device-reject fallback dropped in #987; ledgered 6 streamdf torch
  tests the coercion correctly breaks (unmigrated streamdf numpy track-build).
- **#992** (merged 2026-06-21) — anchor scalar coordinates in migrated analytic/disk potentials
  (SpiralArms, EllipticalDisk, KuijkenDubinski, PowerSpherical[wCutoff]).
- **#993** (merged 2026-06-22) — **last potential-reds PR**: `_check_backend_compatible` now recurses
  into `CompositePotential._potlist` (the dominant fix — composites of migrated members were wrongly
  gated incompatible, so the boundary never coerced); CylindricallySeparable construction-time coerce;
  CorotatingRotation flag parity; test harness made backend-aware (`_tonumpy` helper, derived mock
  flags). 33 stash-rerun-verified flips. Slow-skipped 3 composite-gate eager-jax timeouts
  (`test_streamspraydf::test_center`, `test_interp_potential::_force`, `test_actionAngleAdiabatic_…_cylsep`
  + 3 at-risk interp `_c` tests). **Closes the potential reds.**
- **#994** (merged 2026-06-22, → **main**) — **numpy 2.5.0 compat** (released that day, broke galpy
  repo-wide): `numpy.linalg.eig/eigvals` now return complex for real-symmetric input → `numpy.real(...)`
  in `ttensor` + a `_real_eig` helper at streamdf's 6 eig sites; `numpy.char.array` deprecated →
  `Orbits._name` uses `numpy.array(..., ndmin=1)` (the `ndmin=1` preserves char.array's scalar→1-d
  promotion — caught while verifying older-numpy parity). Byte-identical on numpy<2.5. feat/backends
  rebased onto this (ttensor unified to `xp.real(xp.linalg.eigvals(tij))`), then #993 rebased on top.

### Milestone-2 ledger regen (#996) — **POTENTIAL CLOSEOUT**

Regen from run 27949897402 on the merged feat/backends (`b6f1a072`). **Reviewed, not blind**:
**0 ADDs (no regressions), 1017 DROPs**, all in the composite-gate/coercion families (test_potential
525, test_orbit 256, then sphericaldf/scf/Multipole/quantity/interp/actionAngle). Legit-ledgered
numpy-pinned/unmigrated sets retained (SCF/Multipole #75, AdiabaticContraction, interpSpherical,
TimeDependentAmplitude, streamdf-torch, unmigrated action-angle). All 37 shards completed (no
truncation/carry-forward, unlike milestone-1).

| backend | milestone-1 (2026-06-16) | pre-regen (#978 per-potential granularity) | milestone-2 (#996) |
|---|---|---|---|
| jax xfail | 292 | 284 | **271** |
| torch xfail | 1485 | 2079 | **1075** |

(torch *grew* 1485→2079 via #978's per-potential test parametrization — finer granularity, more
nodeids — then dropped to 1075 here; the count is a granularity-dependent progress proxy, not
monotonic.) **Potential track: done.** Next: orbit (Track D burndown).

**Regen baseline (origin/feat/backends, pre-regen):** jax 284 xfail + 49 slow-skip; torch 2079 xfail
+ 1 slow-skip. The #989–#993 wave's flips become XPASS and get pruned in that regen.

### AA-wave-downstream ledger regen (#1045) — 2026-07-01

Regen from full-matrix run 28516574227 on feat/backends post-#1043 (`2794c74a`). All 45 jax+torch
shards ran to completion (no session-timeout truncation). **Reviewed, not blind: 0 ADDs, 70 DROPs.**
The AA backend-native wave (#1038 real-backend dispatch + #1039/#1040/#1043) makes the df tests that
COMPUTE actions run the real jax/torch action-angle path — streamdf setup/track/`trackaa`/`calcaAJac`,
diskdf, qdf, and their `test_quantity` unit checks. (test_actionAngle itself was pruned earlier at the
AA milestone; this is the downstream df impact.)

| backend | milestone-2 (#996) | pre-AA-wave (post orbit+AA milestones) | AA-wave (#1045) |
|---|---|---|---|
| jax xfail | 271 | 273 | **233** |
| torch xfail | 1075 | 916 | **886** |

**Review discipline:** all 71 regen drops classified before commit — 70 deterministic (seeded samples
`numpy.random.seed(1)`, setup/config, deprecation, unit/returntype) → dropped; the ONE unseeded-RNG
drop `torch test_streamgapdf_impulse::test_impulse_deltav_general_curved` (flaky by construction, #66)
**RETAINED xfail** (a premature drop of a flaky test would red the next run; a retained xfail is
harmless). **AA-perf + AA-wave complete.**
