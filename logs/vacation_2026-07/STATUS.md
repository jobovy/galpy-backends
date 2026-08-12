# Vacation status snapshot — 2026-08-12

**This file is OVERWRITTEN, not appended.** It is the "where are things now"
answer. `log.md` is the trace; read this first and go there only for the why.

---

## THE ONE THING BLOCKED ON JO

`rebase/develop-onto-backends-20260811` (**817e1333**) is the rebased
`feat/backends-develop`. Jo has authorised force-pushing it, **but I cannot**:
`feat/backends-develop` carries a `non_fast_forward` rule and I have `push` only
(no `admin`/`maintain`, so no bypass). Confirmed non-FF: `f306b389` is not an
ancestor of `817e1333`.

Jo asked to be **pinged on the phone** when it is green and verified. So:
drive CI green -> finish the checks below -> ATTEMPT the push anyway (the rule
reading could be wrong, it costs nothing to find out) -> then `PushNotification`
either way.

The command for Jo:

```bash
cd /home/bovy/Repos/galpy && git fetch origin
git push --force-with-lease=feat/backends-develop:f306b389 \
    origin rebase/develop-onto-backends-20260811:feat/backends-develop
```

## Branch state (both rebases DONE)

    main                    87256016
      feat/backends         982b1781   225 commits onto main   (Jo force-pushed, LANDED)
        rebase/develop-...  817e1333    79 commits onto that   (I pushed; awaiting Jo's force-push)

`feat/backends-develop` itself is still at the OLD `f306b389` until Jo pushes.

## Verification of 817e1333

  * numpy `test_potential.py` **1390 tests, 0 failures, 0 errors** (102 skipped)
  * `test_interp_potential.py` 45 passed
  * C extension builds and LOADS (`dop853_c`, R(1)=8.1111274292)
  * ledger `xfail` 16 / `slow_skip` 85 — matches develop exactly
  * RazorThin resolution agrees with numpy on every path:
    numpy bit-identical 12/12, jax 1.95e-16, torch(float64) 2.68e-16,
    vectorised call shape (5,) correct on both
  * 0 merge commits; main AND feat/backends both ancestors; no conflict markers
  * **STILL RUNNING:** forced-jax + forced-torch `test_potential.py`, and the
    branch's own CI (queued at 02:40 UTC)

## The 4 conflicts I resolved (all "main's numerics vs develop's vectorisation")

All resolved by keeping BOTH sides. The one worth re-reading if something looks
wrong later:

  * **RazorThinExpDisk** — develop's `xp.where` + trailing-node-axis
    vectorisation, with main's k=R substituted nodes restored through
    `_inner_nodes`/`_outer_nodes`. Safe because at R >= 10 `_outer_nodes` gives
    EXACTLY 0.0 weights, preserving develop's empty-panel trick.
  * `Potential.py` x2 — develop's dual numpy/backend path and later
    `_surfdens_poisson` split, with main's `epsabs=0.0` re-applied each time.
  * `test_potential.py` — both sides added different tests; kept both.

**One commit correctly dropped** (79, not 80): `#1275` was a pre-rebase
cherry-pick of main fixes, verified 0 diff lines against the rebased tree.

## Deliberate non-changes (do NOT "fix" these silently)

  * develop's `-inf, inf` quad in `surfdens` has **no `epsabs`**. Main never
    touched it, so adding it inside a rebase would be scope creep. Plausibly the
    same latent bug — worth a follow-up PR, not a rebase edit.

## Live traps

  * `rerere` will replay my resolutions, including ones recorded during an
    ABANDONED cherry-pick. It nearly gutted the ledger commit. Verify anything it
    auto-stages.
  * The 2 stream-track gradient tests (`test_determine_stream_track_
    differentiable_{potential,progenitor}`) are **FLAKY**, proven by a same-SHA
    re-run: attempt 1 = 2 failed, attempt 2 = 6302 passed. Re-run before
    investigating. They compare AD against an ADAPTIVE-solver FD at a 0.3 % bar.
  * Cron `6a357c84` created 2026-08-11 -> **expires ~2026-08-18**. Re-create
    before then; it has already died twice.

## After the push

Resume autonomous development on the new `feat/backends-develop`. Queue:
the units/`--jit` harness-skip decision (33 jax traced failures = ONE astropy
boundary, needs Jo's call), the torch dynamo 5, and the backlog in the task list.
