# Vacation status snapshot — 2026-08-17 (gh#1331 merged; gh#1316 diagnosis)

**This file is OVERWRITTEN, not appended.** It is the "where are things now"
answer. `log.md` is the trace; read this first and go there only for the why.

---

## Nothing is blocked on Jo

## Branch state

    main                   342a2dac   (gh#1323 merged 2026-08-17)
    feat/backends-develop  091053a9   (gh#1327/1328/1330/1331 merged)

## Open PRs

**gh#1332 → `feat/backends-develop`** ledger prune (2 stale `torch-jit
test_eccentricity` lines). Head `bb9a17e3`. **All 4 build reds are `Set up job`**
— the jobs never ran a single test, on both ubuntu and macos, including a
same-SHA push/pull_request twin pair that BOTH failed. GitHub was returning
HTTP 503 on the API at the same time; this is platform infra, not the PR.
Re-run is blocked until the run's remaining jobs finish (`rerun-failed-jobs`
is refused while any job is pending).

**gh#1316 → `feat/backends-develop`** interpRZPotential grid spot-check guard,
head **`033684e1`** (pushed 2026-08-17, CI running). **DIAGNOSED AND FIXED — the
tolerance was never the problem.**

The failing `test_..._falls_back_for_nonbroadcasting_potential` asserted
`numpy.array_equal` against the cell-by-cell path, i.e. exactly the
bit-exactness this PR deliberately stops requiring. No value of `rtol` could
have satisfied it, and I spent two rounds (1e-12 -> 1e-14) tightening against
it. It was stable on develop only because the guard ALSO compared with
`array_equal` — guard-criterion and test-criterion were the same predicate, so
any ULP difference forced a fallback and a fallback is cell-by-cell by
construction. The test could not fail, and could not measure the guard either.

    AnySphericalPotential, whole-mesh vs cell-by-cell, on the test's own grid
      numpy   76 of 77 cells differ, worst 9.4e-01   <- non-broadcasting
      jax      0 of 77 cells differ                  <- broadcasts CORRECTLY

Fixed by giving the spot-check fallback a synthetic-evaluator test (matching its
two siblings in the same file) that injects 1e-13 relative — 10x above the
allowance, ~60x above the 1.6e-15 noise it exists to tolerate — in two shapes,
one of them a single cell at a location `_spot_check_cells` is known to visit.
The real-potential test now asserts the contract that holds: bit for bit on
numpy, within 1e-14 on a backend.

    A/B          numpy   jax    torch
    rtol 1e-14   pass    pass   pass
    rtol 1e-12   pass    FAIL   FAIL

numpy passing both arms is the point — it takes `rtol=0` regardless, confirming
the test aims at the backend arm. Local gate: 45 numpy / 40+6skip jax / 45 torch,
0 failures (the one `_secondderivs_c` red is this worktree's missing C
extension, verified via `load_libgalpy() -> False`, and it predates the change).

**gh#1324 → `main`** — **MERGED by Jo 2026-08-17 16:22Z, merge commit `de8e63c0`.**
Array-input `if -> xp.where` for SphericalShell + HomogeneousSphere, plus the
`numpy.infty` -> `numpy.inf` fixes, are now on main.
**Jo's to merge.** SphericalShell + HomogeneousSphere accept array input
(if→where, scalar path bit-identical behind `numpy.ndim`); `numpy.infty` →
`numpy.inf` at two live-`AttributeError` sites. Two things happened to it today,
both resolved:

* It went **DIRTY** when gh#1323 squash-merged (stacked PR: same content on main
  under a different commit). Resolved by merging `main` in — one conflict, in the
  `rmpots` list both PRs edit. Diff vs main is now scoped to its own 4 files.
* It genuinely **cost 2 lines of coverage** (26 vs gh#1323's 24). Fixed — see
  "the coverage lesson" below.

## Merged since the last snapshot

* **gh#1327** → `24500321` — the develop coverage regression, 20 → 15 misses.
* **gh#1328** → `da11a1a8` — AnyAxisym surfdens fixtures dispatch on
  `is_backend_array`, not `get_namespace`. Coverage 15 vs base 15.
* **gh#1330** → **`acfc7d88`** — the `finite_part_quad` log subtraction.
  152 SUCCESS, coverage 15 vs 15 with head on the *more* complete side.
* **gh#1331** → **`091053a9`** — the zforce relative floor. 152 SUCCESS,
  coverage 15 vs base 15, both sides at 28 sessions. Traced AnyAxisym is now
  at ZERO failures on develop.
* **gh#1323** → main `342a2dac` (Jo).

## Traced AnyAxisym: unledgered failures CLOSED, 4 ledgered entries REMAIN

Every step measured on `-k anyaxisym --backend jax --jit`, not projected:

    baseline                       3 failed   (1 accuracy + 2 tracer errors)
    + gh#1328 (fixture)            2 failed
    + gh#1330 (log subtraction)    1 failed
    + gh#1331 (zforce floor)       0 failed   <- MERGED, on develop

Last row verified on the synced branch: **11 passed, 1 skipped, 2 xfailed, 0
FAILED**. The xfails report `x`, not `X` — no hidden XPASS, no ledger delta.

**That "0 FAILED" is about UNLEDGERED failures only, and I have been saying
"closed" too loosely.** Four `-jit` ledger entries for AnyAxisym survive, and
still fail on re-measurement against develop `091053a9`:

    jax-jit / torch-jit  test_poisson_surfdens_potential[AnyAxisymmetricRazorThinDiskPotential]
    jax-jit / torch-jit  test_forceAsDeriv_potential[AnyAxisymmetricRazorThinDiskPotential]

They are PRE-EXISTING (added in gh#1281/gh#1284, well before gh#1331), so not a
regression from the zforce floor — checked with `git log -S` on the ledger files
rather than assumed. `forceAsDeriv` fails at exactly **(R,Z) = (0.500, 0.000)**
with `diff = 7.35e+03, rel. diff = inf`, i.e. against a true value of zero: the
vertical force as a vertical DERIVATIVE of Phi, evaluated at the one point where
a razor-thin disk's `Fz` is genuinely discontinuous. Eager passes and only
`-jit` is ledgered, so there is a real traced-vs-eager difference to explain
before this can be called by-design.

**Out of scope, and NOT a backend gap:** the large-R arm of the same family.
`Sigma(10) = 7.8e-13` sits ~4 orders below the quadrature's own noise and it
fails **identically on numpy** — both paths return 1.206e+04 relative at R=10,
z/R=1e-6. Two independent implementations giving the same wrong number is the
evidence that it is the formulation, not the migration.

Residue from the original dispatched-jit 7: **3 torch traced timeouts**,
slow-skip material, wanting a fresh dispatched run. Deliberately not dispatched
while PR runs hold the runners.

## The coverage lesson of the week

gh#1324 lost 2 lines in **`galpy/orbit/Orbits.py`** — a file it does not touch.
I twice argued it therefore could not be responsible. **That was wrong.** Proven
by tracing lines 2991-2992 on both trees:

    potential              base (gh#1323)   gh#1324
    SphericalShell              hits         NO LONGER hits
    HomogeneousSphere           hits         NO LONGER hits
    AnyAxisymRazorThin          hits         hits

Those two potentials were the only live coverers of the
`except (ValueError, TypeError, IndexError)` per-element fallback in
`Orbits.E()`; making them accept arrays stops the vectorised call raising, so
nothing enters the branch. Fixed with a forcing test using
`AnyAxisymmetricRazorThinDiskPotential`, which still rejects arrays — and which
checks the fallback's VALUES against a per-element reference, not merely that
the line executes.

**File locality is not causal locality.** The right question is never "does the
PR touch that file" but "could the PR change whether that branch is ENTERED".

## Standing corrections to how I work

* **Never decrease coverage, not even one line** (Jo). The loss is always a
  defensive branch going unreachable, never new untested code — so a fallback
  needs its forcing test in the PR that makes it unreachable.
* **codecov: discriminate before believing.** `sessions` base-vs-head separates
  mid-upload phantoms from real deltas — but only when both sides are the SAME
  event type. `main` is a path-filtered *push* run; a PR is a *pull_request*
  run; their session counts are not comparable. Compare against the predecessor
  PR's own run.
* **Structural set-diffs with `ast.parse`, never grep.** `grep -oP '^def \K\w+'`
  silently returned 278 names where `grep -cP '^def '` counted 279, producing a
  false "function deleted" alarm. `git diff --stat`'s deletion count was
  authoritative and disagreed from the start.
* **An A/B needs BOTH arms actually run.** I "could not reproduce" the Orbits.E
  mechanism because I only ever traced the *after* tree.
* **Removing an exception ≠ fixing N tests.** It may unmask a later failure.
* **Run the forced-torch gate before pushing.** jax accepts Python floats where
  torch rejects them, so a jax-only gate is structurally blind to that class.
* **GitHub re-run rules:** `rerun-failed-jobs` is refused while ANY job in the
  run is still pending (`403 already running`), and a re-run of a re-run is
  refused outright (`cannot be retried`).

## Cron

`351c0c5a`, hourly at :13, session-only, created 2026-08-13 → **expires
~2026-08-20**. Verified alive this wake.
