# Vacation status snapshot — 2026-08-04

**This file is OVERWRITTEN, not appended.** It is the "where are things now" answer.
`log.md` is the trace; read this first and go there only for the why.

---

## PR pipeline

| state | branch / PR | effect |
|---|---|---|
| **merged** | #1246, #1247, #1248 → `730657c3`, #1249 → `9695ae4a`, **#1250 → `1adbe26e`** | jit helper; scipy-1.18 spline pickling; planar adapters derive `_backend_compatible`; **−8 ledger entries (42 → 34)** |
| **open** | **#1251** `feat/hoist-gl-order` (1 commit) | de-duplicates `_BACKEND_GL_ORDER`; corrects its documented floor |
| ready, not opened | `feat/undefer-sphericaldf-jax` @ `535d6659` (1 commit) | **slow-skip 230 → 202**; being *measured* locally rather than left to CI (below) |

**#1251 status:** 133 green, 16 in progress, 2 red — `codecov/project` (the usual
partial-upload artifact) and the `test_orbit.py` shard, whose job log shows
`DALServiceError` on `test_from_name_values`/`test_from_name_errors`: the same
VizieR outage that hit #1250, and unreachable from the three `actionAngle` files
this PR touches. Re-run is queued behind the parent run — `gh run rerun --job`
refuses while the run is `in_progress`, which reads like a permissions error and
is purely timing.

All branches are clean, disjoint in files, and need no rebase — verified against
the **merge-base**, not `HEAD..develop`, which cannot distinguish "they changed
it" from "I changed it".

**Gate note earned on #1249:** a failed `all-clear` can be a stale poll rather
than a real blocker. Its log listed all 151 dependencies as `success` and still
reported `1 failing check run(s)` — it had evaluated while codecov was still red.
Re-running *just the gate* took it to CLEAN. Read the gate's log before deciding
whether to re-run or to hunt a failure.

## Ledger

```
156 → … → 48 → 45 → 44 → 42 → 34
```

**34 today** on `feat/backends-develop`, `jax 15 / jax-jit 3 / torch 15 /
torch-jit 1`. By file: `test_actionAngle` 12 (9 torch + 3 jax), `test_streamdf`
3, `test_diskdf` 3, `test_orbit` 4, `test_potential` 3, `test_noninertial` 2,
`test_interp_potential` 2, `test_sphericaldf` 2, `test_qdf` 2,
`test_SpiralArmsPotential` 1.

**Verified to contain zero stale entries** (2026-08-04), by intersecting the
ledger with the junit artifacts of #1248's completed backend run: torch **19/19
matched, 0 passing**; jax 10/19 observable, 0 passing. So burndown must come from
fixes or decisions — there is nothing left to prune. *(The first version of that
sweep silently compared against nothing; see log. Validate the matcher.)*

### What the remaining entries actually are

**`test_quantity` (8) — SOLVED, in the queued tolerance PR.** One root cause: an
absolute `1e-8` bound on values of ~1.6e10 (= 6e-19 relative, ~5e-6 of one ULP);
the measured difference is exactly 1 ULP. Three of the four tests were never
independently broken — a failure between `_APY_UNITS = False` and its restore
leaked the flag for the rest of the session.

**`test_actionAngle` (12) — TRIAGED, none is a defect:**

| kind | n | disposition |
|---|---|---|
| GL-vs-adaptive quadrature floor (~2e-6 for angles) | **8** | **Jo's decision.** Spherical 4 + Adiabatic 2 (delegates to Spherical) + Vertical 2. Verified *not* fixable by raising the order: n=50/100/200/400 → 3.3e-6 / 2.1e-6 / 1.8e-6 / **3.5e-6** (does not converge; round-off wins at 400). |
| sub-epsilon bound | 2 | **Jo's decision.** StaeckelGrid Jr=0: bound `1e-16` < eps `2.22e-16`; numpy returns *exactly* 0.0, so only bit-exact zero passes. |
| documented design | 2 | AdiabaticGrid off-grid: backend AA grids have no off-grid fallback (docstring says so). Clears only via feature #103. **Keep out of the tolerance batch.** |

## Waiting on Jo

1. **GL quadrature floor** (8 entries): backend-specific bound at the measured
   floor, *or* re-base the test on a gold reference instead of treating scipy's
   adaptive quad as truth. The second is the better test, the larger change.
2. **Sub-epsilon bounds** (2 entries): the `1e-16` Jr=0 style assertions.
3. AnyAxisym small-|z| threshold (#189, pre-existing).

Group A/B tolerance approval (2026-08-01) covered `test_quantity`; these are a
**new** batch. `actionAngleVerticalInverse` and `actionAngleTorus` remain
permanently out of scope.

## The bigger debt: deferrals, not the ledger

`backend_slow_skip.txt` = **230 entries (jax 215 / torch 15)**; the census counts
334 jax *instances*. Decomposed by the file's own documented causes:

| n | cause | actionable by me? |
|---|---|---|
| **66** | jax orbit heavy eager >45 s — per-CI-group session-cap fit | **only via speed work** |
| 39 | jax single-orbit variational/dxdv/lyapunov (Track D) | yes |
| 28 | jax spherical-DF sampling/quadrature (Track F) | yes |
| 17 | NonInertialFrameForce eager-XLA timeout | yes |
| 13 | torch orbit dissipative/variational dxdv | yes |
| 12 | jax multi-orbit battery (Track D/E) | yes |
| 55 | composite-MW Poisson 10, misc single-orbit 10, qdf 7, diskdf 6, dissipative 6, interpRZ 5, streamdf 6, FDM 3, AA 2 | mixed |

**Do not read this as "one fix clears 127 orbit tests"** — that was an earlier
error of mine from grouping by *filename*. The largest block (66) is a CI-budget
fit whose knobs (`--timeout 300`, `timeout-minutes: 180`, pytest-split groups)
live in `.github/workflows/backend-tests.yml`, which **I cannot edit — no
`workflow` scope**. For those, faster code is my only route.

Largest genuine capability gap I *can* close: **Track D variational
vectorization (39 + 12 = 51)**, then Track F spherical-DF (28).

## Jit progress

`test_sphericaldf` under `--backend jax --jit`: **99.8 % of boundary calls traced**
(152 293 traced / 375 not), **0 C-library entries** → nothing not-traceable by
construction. Caveat: that is over the **87 %** of the suite that runs; the 28
skips are the jax slow-skips, excluded by construction rather than passing.

## Operating notes earned this fortnight

* codecov `project` red on a fresh PR is a **partial-upload artifact** — check
  "HEAD has N uploads less than BASE" and wait; #1248 walked 40.86 → 85.43 →
  99.95 % with no intervention. The note vanishing from the comment is *not* a
  completion signal.
* A worktree without `libgalpy.so` **flips ledgered tests to PASS**. Build it
  before attributing any local result.
* Hand-picked nodeids are not a context: two `test_quantity` tests passed in
  isolation and failed in the full file.
* Read the docstring before classifying a numeric difference. Three times this
  session a "bug" turned out to be documented intended behaviour.
