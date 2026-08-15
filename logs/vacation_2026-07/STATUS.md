# Vacation status snapshot — 2026-08-15, two PRs open

**This file is OVERWRITTEN, not appended.** It is the "where are things now"
answer. `log.md` is the trace; read this first and go there only for the why.

---

## Nothing is blocked on Jo

## Branch state

`feat/backends-develop` = `07e569e8`. Merged 08-14/15: gh#1310, #1311, #1312,
#1319.

## The ledger, honestly

    xfail        15   real defects, individually
    slow-skip    84   performance-gated       -> 78 with gh#1322
    skip         68   PERMANENT by design (backend_skip.txt says so)
    ---------------
    TOTAL       167      IN SCOPE = 99  (skip is not pending work)

With gh#1322 merged: **161 total / 93 in scope.**

Note: STATUS previously said 165/100. Both were wrong — I recounted from the
files themselves (`grep -vc '^#\|^$'` over the three ledgers) against
`origin/feat/backends-develop`. Trust the recount, not the old numbers.

## Open PRs — both mine, both awaiting CI

**gh#1322 → `feat/backends-develop`** `backend-jit/orbit-cluster-burndown`,
head `90303ab0`. Un-defers 6 measured jax `test_orbit.py` entries, 167 → 161.
Also updates `tests/.test_orbit_durations` (four of the six were recorded at
`0.05` — pytest-split's value for a *skipped* test — so un-deferring without
that edit could pack them all into one group). CI queued.

**gh#1323 → `main`** `fix/anyspherical-array-mass`, head `2cecf431`. The
upstream numpy bug: `quad` takes scalar limits, `AnySphericalPotential` passed
`atleast_1d(r).flatten()[0]` at two sites → **forces silently wrong by 87.7 %**
away from `r[0]`, and `_revaluate` raised on arrays outright. Fixed via new
`galpy.util.quadpack.quad_over_limits`. Scalar path bit-identical.

**gh#1316** stays a parked draft — see next.

## The next move: rework gh#1316 as the cherry-pick

Jo: *"you'll need a pr to main to fix the numpy bug and cherry pick it here (so
it can easily be dropped in a subsequent rebase)"*. That cherry-pick and the
`_grid_eval` guard removal belong in **one** PR, and gh#1316 already is that PR
— so rework it rather than opening a third:

1. cherry-pick `2cecf431` (drops cleanly on a later rebase onto main);
2. delete `_grid_eval`'s bit-for-bit guard entirely. Its job was catching
   mis-broadcast potentials; `test_potential_array_input` now does that
   properly, and under backends too. The guard cannot survive on jax anyway —
   whole-mesh vs scalar reassociation differs at ~1e-15, so it falls back for
   *every* potential, which is the 1643x;
3. drop gh#1316's current tolerance commits — they were the wrong fix and CI
   refuted them.

## What I learned this session (don't repeat)

* **The all-potentials array test already existed.** `test_potential_array_input`
  covers all 8 methods at 1e-10 for every discovered potential;
  `AnySphericalPotential` was on its *exclusion list*. Search before building.
* **Never mutate the worktree while a run is in flight.** I did the A/B
  file-swap during a background gate and it reported a false failure. Re-ran
  clean: 1280 passed.
* **`git checkout <ref> -- <file>` also stages.** Copy the file aside first.
* **Ledger notes decay.** All 12 orbit entries said `# 300s (failed)`; two of
  them actually *pass* (one at 1535s, recorded as 620s).
* **Un-deferring is a two-file edit** — the ledger *and*
  `.test_orbit_durations`.
* **`test_quantity` must run serial**; under `-n 16` it reports ~106 phantom
  failures across the whole library.

## Cron

`351c0c5a`, hourly at :13, session-only, created 2026-08-13 → **expires
~2026-08-20**. Verified alive this wake.
