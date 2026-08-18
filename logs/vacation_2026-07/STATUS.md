# Vacation status snapshot — 2026-08-18 (gh#1332 merged; gh#1336 sole open PR)

**This file is OVERWRITTEN, not appended.** It is the "where are things now"
answer. `log.md` is the trace; read this first and go there only for the why.

---

## Nothing is blocked on Jo

## Branch state

    main                   c7891fca   (gh#1323, gh#1324, gh#1333 merged)
    feat/backends-develop  2eb1ac0e   (gh#1316, gh#1334, gh#1332 merged)

## Open PRs — ONE

**gh#1336 → develop** `backend/anyspherical-array-mass`, head `cf794ba5`.
Three commits:

  * AnySpherical: array input silently returned the mass inside `r[0]` at every
    element. `_rawmass` now goes through `quad_over_limits` (which was missing
    from develop entirely — cherry-picked from main's gh#1323) and `_revaluate`
    splits scalar/array. **76 of 77 grid cells wrong → 0 of 77.**
  * AnyAxisym: port main's numpy Phi midplane-cusp fix + centred midplane FD.
  * **Ledger delta: −2 xfail** (jax-jit + torch-jit
    `test_forceAsDeriv_potential[AnyAxisym]`), measured 11 passed/2 xfailed →
    12 passed/1 xfailed.

Status: one shard re-run in flight. All three reds trace to ONE flake — see
below. `_force_accepts_arrays = False` is KEPT (Jo's design call); its comment
was corrected because the old numerical justification was stale.

## The gh#1336 reds are one flake, not three problems

    head cf794ba5   sessions=27      base 700f8694   sessions=28

A failed shard never uploads coverage, so the flaky shard IS the missing
session → that is the `codecov/project −0.01%` → which held `all-clear`.
**A shard flake also presents as a codecov drop; don't diagnose it separately.**

Flake verdict is twin-proven AND tree-checked: same SHA, push run's
`test_backend*.py` shard succeeded while the PR run's failed, and gh#1336 is
`ahead=3 behind=0` vs develop's current head, so the PR merge is a
fast-forward and both twins test identical trees.

## Live investigation: streamdf AD flake — cause still OPEN

`test_determine_stream_track_differentiable_{potential,progenitor}` compare
`jax.grad` to a central FD of the same backend track.

**Two mechanisms proposed and BOTH killed this wake:**
* *Adaptive-step FD noise* — dead: `inbackend_ode.py:149` already defaults to
  `rtol=atol=1e-12`, and tol/2h is ~1e-7 relative against an observed 20%.
  Six orders short.
* *AD arm truncating at its 10x-smaller `max_steps`* — dead: no `throw=False`,
  so diffrax raises rather than truncating, and the failure was a plain
  assertion with a finite AD.

**Live lead:** the test's own docstring warns "a vmap chunk loop instead leaks
~20%", and the q failure is 8.24e-01/4.131630 = **19.95%**. Points at the
chunk-assembly path, not solver noise. Does not explain the progenitor failure
in the same run (95%), so either two mechanisms or a shared upstream cause.

The FLAKE VERDICT is unaffected — it rests on the twin split plus `behind=0`.

Probe running: asserted quantity across h at default vs rtol=atol=1e-12.
**Caveat: that worktree has no built C extension**, so the numpy sdf falls back
to leapfrog — mechanism unaffected (both arms share the frozen sdf, both run
diffrax), but local magnitudes are NOT CI magnitudes.

## Ledger

Eager sweep at 700f8694: **0 stale**, 48 correctly ledgered, 2 file-level
not-present. No free eager burndown left at this head — further reduction must
come from the `-jit` half or from real fixes.

## Standing corrections to how I work

* **Never decrease coverage, not even one line** (Jo).
* **Check the same-SHA twin FIRST** on any shard red — but a twin split only
  proves nondeterminism if `behind=0`; PR-event runs test head merged into base.
* **Compare against the branch ref, not `base.sha`** (which goes stale).
* **File presence is not behaviour.** Run the measurement the fix changes.
* **A test whose predicate duplicates the guard it tests measures nothing.**
* **Probe inside `jax.jit`.** Eager hands concrete scalars to scipy.
* **`ps` on a shell pipeline measures the WRAPPER**, and cwd resets between
  calls — use absolute paths.

## Cron

`351c0c5a`, hourly at :13, session-only, created 2026-08-13 → **expires
~2026-08-20 (2 days)**. Verified alive this wake.
