# Vacation status snapshot — 2026-08-17

**This file is OVERWRITTEN, not appended.** It is the "where are things now"
answer. `log.md` is the trace; read this first and go there only for the why.

---

## Nothing is blocked on Jo

## Branch state

    main                   342a2dac   (gh#1323 merged 2026-08-17)
    feat/backends-develop  acfc7d88   (gh#1327, gh#1328, gh#1330 merged)

## Open PRs

**gh#1324 → `main`** `fix/potential-array-input-where`, head `651e75ed`.
**Jo's to merge.** SphericalShell + HomogeneousSphere accept array input
(if→where, scalar path bit-identical behind `numpy.ndim`); `numpy.infty` →
`numpy.inf` at two live-`AttributeError` sites. Two things happened to it today,
both resolved:

* It went **DIRTY** when gh#1323 squash-merged (stacked PR: same content on main
  under a different commit). Resolved by merging `main` in — one conflict, in the
  `rmpots` list both PRs edit. Diff vs main is now scoped to its own 4 files.
* It genuinely **cost 2 lines of coverage** (26 vs gh#1323's 24). Fixed — see
  "the coverage lesson" below.

**gh#1331 → `feat/backends-develop`** `backend-jit/zforce-relative-floor`, head
`557499f8`. **Mine to merge.** Floors `|z|` relative to R in the traced zforce.
CI mid-run; codecov red is the usual mid-upload phantom (21 sessions vs base 27,
85.98% against a true 99.96%).

## Merged since the last snapshot

* **gh#1327** → `24500321` — the develop coverage regression, 20 → 15 misses.
* **gh#1328** → `da11a1a8` — AnyAxisym surfdens fixtures dispatch on
  `is_backend_array`, not `get_namespace`. Coverage 15 vs base 15.
* **gh#1330** → **`acfc7d88`** — the `finite_part_quad` log subtraction.
  152 SUCCESS, coverage 15 vs 15 with head on the *more* complete side.
* **gh#1323** → main `342a2dac` (Jo).

## Traced AnyAxisym is CLOSED

Every step measured on `-k anyaxisym --backend jax --jit`, not projected:

    baseline                       3 failed   (1 accuracy + 2 tracer errors)
    + gh#1328 (fixture)            2 failed
    + gh#1330 (log subtraction)    1 failed
    + gh#1331 (zforce floor)       0 failed

Last row verified on the synced branch: **11 passed, 1 skipped, 2 xfailed, 0
FAILED**. The xfails report `x`, not `X` — no hidden XPASS, no ledger delta.

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
