# Vacation status snapshot — 2026-08-15 late, gh#1322 MERGED

**This file is OVERWRITTEN, not appended.** It is the "where are things now"
answer. `log.md` is the trace; read this first and go there only for the why.

---

## Nothing is blocked on Jo

(Except that main PRs are his to merge — that is the expected end state, not a
blocker.)

## Branch state

`feat/backends-develop` = **`ccfe583b`** (gh#1322 squash-merged 2026-08-15).
Merged 08-14/15: gh#1310, #1311, #1312, #1319, **#1322**.

## The ledger, verified on the merged develop

    xfail        15   real defects, individually
    slow-skip    78   performance-gated
    skip         68   PERMANENT by design (backend_skip.txt says so)
    ---------------
    TOTAL       161      IN SCOPE = 93  (skip is not pending work)

Counted with `grep -vc '^#\|^$'` over the three files at
`origin/feat/backends-develop`. Earlier STATUS numbers (165/100, 167/99) were
recounts along the way — this is the current one.

**The eager ledger is honest.** The (ledgered ∧ passed) sweep over gh#1322's own
`backend-junit-*` artifacts found **0 stale entries** among everything a PR run
can evaluate. 40 of the entries are `-jit` and are only exercised by
`workflow_dispatch(jit=true)` — they remain unmeasured.

## Open PRs

**gh#1323 → `main`** `fix/anyspherical-array-mass`, head `2cecf431`. The upstream
numpy bug: `quad` takes scalar limits, `AnySphericalPotential` passed
`atleast_1d(r).flatten()[0]` at two sites → forces silently wrong by **87.7 %**
away from `r[0]`, and `_revaluate` raised outright on arrays. Fixed via new
`galpy.util.quadpack.quad_over_limits`; scalar path bit-identical.
CI: one red, `test_symplectic_dxdv_base_bit_identity` at **max|diff|=5.55112e-16**
— the known intermittent. Proven a flake by the same-SHA twin (push run
succeeded, PR run failed, identical job) *and* by ruling out the merge-vs-head
alternative (`origin/main` is still my branch point `3762e73e`). **Re-run
triggered.**

**gh#1324 → stacked on gh#1323** `fix/potential-array-input-where`, head
`a07ce379`. `SphericalShellPotential` + `HomogeneousSpherePotential` accept array
input (if→where, scalar path left literally unchanged behind `numpy.ndim`);
`numpy.infty` → `numpy.inf` at two live-`AttributeError` sites. Test goes 43 → 45
potentials. Retarget to `main` when gh#1323 lands.

**gh#1316** parked draft — reworks into "cherry-pick gh#1323 + delete the
`_grid_eval` guard" once gh#1323 is on main. Its own comment records the plan.

## Tools built this trip

`scratchpad/ledger_sweep.py` — (ledgered ∧ passed) over `backend-junit-*`.
Belongs in-repo next to `tests/backend_status_report.py`; not yet PR'd.

## Standing corrections to how I work

* **The CI/local ratio is 0.38–0.71, not a flat 0.71.** Measured across the 6
  un-deferred orbit entries. Do not project a marginal call from one number.
* **Verify a sweep's own output before believing it.** Its first run reported 12
  stale entries; all 12 were bugs in the sweep (bare-name collision across
  `test_orbit`/`test_orbits`; `-jit` entries scored against eager artifacts).
* **A codecov red with checks still pending is an upload phantom.** Discriminate
  by `totals.sessions` base-vs-head; identical `lines` + fewer sessions = phantom.
  It self-heals. Do not "fix" it.
* **Never mutate a worktree while a run is in flight** — a mid-run file swap
  produced a false test failure.
* **ruff-format rejects the first commit** and leaves HEAD unmoved; always
  re-`git add` and re-commit, then check the SHA moved.

## Cron

`351c0c5a`, hourly at :13, session-only, created 2026-08-13 → **expires
~2026-08-20**. Verified alive this wake.
