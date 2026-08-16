# Vacation status snapshot — 2026-08-16, three PRs open

**This file is OVERWRITTEN, not appended.** It is the "where are things now"
answer. `log.md` is the trace; read this first and go there only for the why.

---

## Nothing is blocked on Jo

Two of the three PRs are his to merge (they target `main`), which is the normal
end state, not a blocker.

## Branch state

    main                   54c0324a   (gh#1320 landed since my branch point)
    feat/backends-develop  ccfe583b   (gh#1322 merged)

## The ledger, verified on develop

    xfail        15   real defects, individually
    slow-skip    78   performance-gated
    skip         68   PERMANENT by design (backend_skip.txt says so)
    ---------------
    TOTAL       161      IN SCOPE = 93  (skip is not pending work)

The (ledgered ∧ passed) sweep over CI artifacts finds **0 stale entries** among
everything a normal PR run can evaluate. The `-jit` half was measured once, via
a dispatched traced run — see below.

## Open PRs

**gh#1323 → `main`** `fix/anyspherical-array-mass`, head `95cb7127`.
**149 SUCCESS, 0 failures.** The AnySpherical array bug (forces silently wrong by
87.7 % away from `r[0]`; `_revaluate` raised outright on arrays), plus the
`_grid_eval` fallback test that keeps repo coverage at 24 misses rather than 25.
Two flakes (SIMBAD outage, 1-ULP symplectic) cleared on re-run. Ready.

**gh#1324 → stacked on gh#1323** `fix/potential-array-input-where`, head
`69db2249`. **CLEAN, 105 SUCCESS.** SphericalShell + HomogeneousSphere accept
array input (if→where, scalar path bit-identical behind `numpy.ndim`);
`numpy.infty` → `numpy.inf` at two live-`AttributeError` sites. Retargets to
`main` automatically when gh#1323 lands.

**gh#1327 → `feat/backends-develop`** `backend-jit/coverage-fallbacks`, head
`59a60f9a`. **Mine to merge.** The develop coverage regression, all five lines,
20 → 15 misses. codecov red with ~86 checks still queued, i.e. not yet judged.

## Verified and queued, deliberately NOT a 4th PR

`surfdens=lambda R: get_namespace(R).exp(-R / 0.3)` in
`test_potential.py:14297` fixes **2 of the 7 unledgered traced failures**.
Measured: numpy value identical (`5.588879573` vs gold `5.5888795714e+00`), and
`jax.grad` / `jax.jit` both succeed. Goes in after gh#1327.

## Traced state (from the one dispatched jit run, 31914531334)

7 failures, none of them new library defects:

* **2 test-side** — the numpy-only `surfdens` fixture above. I first reported
  these as genuine tracer leaks; that was wrong, corrected in the log.
* **2 known AnyAxisym singularity accuracy** — task #218.
* **3 torch traced timeouts** — slow-skip material.

## Standing corrections to how I work

* **Never decrease coverage, not even one line** (Jo, 08-16). The loss is always
  a *defensive branch going unreachable*, never new untested code — so a
  fallback needs its forcing test in the PR that introduces it.
* **codecov red ≠ phantom.** Discriminate: could this PR move coverage at all?
  gh#1322 changed 0 source lines (phantom); gh#1327 changes source (judge it).
* **Read the traceback to the innermost frame** before naming a culprit —
  exception *type* is not a diagnosis.
* **The same-SHA twin is not a clean A/B once `main` moves**: the PR arm tests
  head *merged with* new main, the push arm tests head alone.
* **Flake rates sampled from run conclusions are undercounts** — a re-run
  overwrites the failure. Walk `attempts` instead.
* **Check `git branch --show-current` before committing** in a multi-branch
  worktree; `git push` answering "Everything up-to-date" is not success.

## Cron

`351c0c5a`, hourly at :13, session-only, created 2026-08-13 → **expires
~2026-08-20**. Verified alive this wake.
