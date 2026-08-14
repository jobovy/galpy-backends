# Vacation status snapshot — 2026-08-14 late (gh#1316 in CI)

**This file is OVERWRITTEN, not appended.** It is the "where are things now"
answer. `log.md` is the trace; read this first and go there only for the why.

---

## Nothing is blocked on Jo

## Branch state

`feat/backends-develop` = `c54fcb8f`. Merged since 08-13: gh#1303, #1305, #1306,
#1307, #1308, #1310, #1311, #1312 (8 PRs in ~36h).

## The ledger, honestly

    xfail        15   real defects, individually
    slow-skip    85   performance-gated
    skip         68   PERMANENT exclusion by design (backend_skip.txt says so)
    ---------------
    TOTAL       168      IN SCOPE = 100  (skip is not pending work)

Of the 68 skips: 48 are VerticalInverse/Torus (`useTM` IS actionAngleTorus), 12
are `Orbit.from_name` SIMBAD network lookups, 8 other. All excluded BY DESIGN —
the file's own header states the contract. I narrowed this denominator by regex
across three wakes before reading that header; do not repeat that.

## Open PR: gh#1316 (the only one)

`backend-jit/gridwave-tolerance`, head `dc1a4cfc`, 3 commits, **ledger delta
168 -> 167**:

1. `galpy/potential/interpRZPotential.py` — `_grid_eval`'s bit-for-bit guard made
   EVERY backend grid build fall back to a cell-by-cell loop, because jax
   reassociates a whole-mesh reduction vs a scalar one (2/9 spot cells differ by
   1.6e-15). Now backend-aware: numpy keeps exact via `allclose(rtol=0, atol=0)`
   (verified identical to `array_equal` over nan/+-inf/-0.0/denormals, so no
   backend-only branch for codecov to flag); backends get rtol 1e-12, ~600x the
   observed difference and ~13 orders below AnySphericalPotential's 95 %-of-cells
   mis-broadcast, which still correctly falls back (2/2).
2. `tests/test_interp_potential.py` — extends the EXISTING `_grid_tol` helper
   (already 30x for traced mode) to backend-built grids at 5x. numpy keeps its
   original 7.0-ULP bound exactly.
3. `tests/backend_slow_skip.txt` — un-defers
   `test_2ndDeriv_potential[mockInterpRZPotentialwSecondDerivs]`.

Measured: jax constructor **136.7 s -> 2.34 s (58x)** on a 201x201 MWPotential
build; **409 s -> 4.54 s (90x)** on the 151x151 6-quantity mock; the un-deferred
entry **PASSES in 3.4 s** (whole file, `--backend jax`, 1277 passed).

Three-arm gate on the head: numpy 45 / jax 39+6skip / torch 79. CI queued (149).

## In flight

Whole-cluster measurement running in `/var/tmp/fpwire`: ALL remaining jax
`test_potential.py` slow-skips un-skipped, one `--backend jax` pass with
junitxml, to get per-entry pass/fail AND runtime in a single ~45 min run instead
of five. Output `scratchpad/cluster.{log,xml}`.

**The worktree's `tests/backend_slow_skip.txt` is stripped for that measurement.**
Backup at `scratchpad/ss.bak`. RESTORE IT before pushing anything from that
branch — this bit me once already and was caught only by `git status`.

## What the remaining 79 slow-skips actually are (measured so far)

Not one problem. The 12-entry `test_potential.py` cluster splits three ways:

* 1 interpRZPotential — **fixed**, gh#1316
* 4 MW mass models (Cautun20/McMillan17/DehnenBinney98/diskscf_overflow) — heavy
  DiskSCF/SCF setup, task #75 territory
* 5 wrapper `poisson_surfdens` — `surfdens` calls `integrate.quad`; measured
  37 ms numpy vs 189 ms jax, i.e. **5x genuine eager overhead, not a fallback
  bug**. Expect these to stay deferred.
* 2 unprobed (`MultipoleExpansion_deepcopy`, `MovingObject_2ndderivs_fd`)

So do NOT expect many more 90x wins: interpRZ was a real defect, not a
representative sample.

## Standing corrections to how I work

* **The hourly loop is a KEEP-ALIVE, not a checkpoint.** Work until a PR is open;
  do not end a turn on a finished measurement or a refuted hypothesis.
* **Assert `is_backend_array` at the call site before believing any timing.** A
  probe that ran on numpy cost ~4 wakes by "eliminating" the exact place the bug
  was.
* **Never project scope from one instance without the grep that would refute it.**
  Three times this trip (67 jit sites, 5 AnyAxisym entries, "_grid_eval is
  everywhere" — it is in ONE file).
* **Verify content, not command success** — `gh pr ready` succeeds even when the
  fix was never pushed.

## Cron

`351c0c5a`, hourly at :13, session-only, created 2026-08-13 -> **expires
~2026-08-20**. Verified alive.
