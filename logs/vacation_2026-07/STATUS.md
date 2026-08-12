# Vacation status snapshot — 2026-08-12 (gh#1300 in CI)

**This file is OVERWRITTEN, not appended.** It is the "where are things now"
answer. `log.md` is the trace; read this first and go there only for the why.

---

## Nothing is blocked on Jo

## Active PR

**gh#1300** `backend-jit/coerce-device-anchor` -> feat/backends-develop.

`@backend_input` coerced each declared coordinate separately, so each derived
its own device anchor. A numpy sibling anchored to `None` -> the backend default
device (CPU) while a CUDA coordinate stayed on the GPU, and the evaluator got a
**split-device** coordinate set — a mixed-device error in some potentials, a
silent GPU→CPU transfer in others (MiyamotoNagai returned a CPU result for a
CUDA input). Fix: compute the anchor ONCE over all declared coordinates and
thread it into `coerce_coords(..., device=)`.

Numpy path unreachable from the change (`device_of` only runs inside the
`xp is not numpy` branch), so byte-identical by construction.

Gates, all from finished runs: numpy 1324 passed, forced torch 1288, forced jax
1276 (57 min), CUDA test on the TITAN Xp, 138 on the rebased tree. Both new
tests verified to FAIL without the fix. Mine to merge when green.

## Branch state

    main                    87256016
      feat/backends         982b1781   225 commits onto main
        feat/backends-develop d76d683b  81 commits (gh#1298, gh#1299)
          gh#1300             62ab8ed3  device anchor (CI running)

## Live traps

  * `rerere` replays my resolutions, including ones recorded during an
    ABANDONED cherry-pick. Verify anything it auto-stages.
  * The 2 stream-track gradient tests are **FLAKY**, proven by a same-SHA re-run
    (2 failed -> 6302 passed). Re-run before investigating. The
    adaptive-solver-tolerance mechanism I proposed is **REFUTED** (defaults
    already 1e-12; FD noise ~5e-9 vs a 3e-3 bar) — do not re-derive it.
  * codecov `sessions` lives at **`totals.sessions`**, not `report.sessions`.
    Reading the wrong key returns 0 for both sides, which looks exactly like
    "the phantom discriminator doesn't apply here". Base-vs-head session count
    with identical `lines` is the proof.
  * A local `tests/test_backend*.py` run is **numpy**; the CI shards of that
    name run `--backend jax/torch`. Not the same context — don't run it locally
    expecting to mirror them.
  * **`galpy-jax` cannot push.** `origin` answers `Repository not found`;
    `origin/main` last moved 2026-07-25, ~977 unpushed commits. Logs are intact
    in the working tree but exist only on this box. Worth Jo re-granting access;
    not a work blocker.
  * Cron `6a357c84` created 2026-08-11 -> **expires ~2026-08-18**. Re-create
    before then; it has already died twice.

## Queue after gh#1300

  * **Needs Jo**: the units/`--jit` harness-skip policy (33 jax traced failures
    = ONE astropy boundary). Proposed a harness-level skip over 33 ledger
    entries; not yet ruled on.
  * task #215 — widen `COMPUTE_METHODS` (100 namespace-aware defs outside it,
    measured 0 unguarded sites; needs a curated criterion since the obvious one
    pulls in `__init__`).
  * task #66 (RNG-fragile `test_impulse_deltav_general_curved`), #130, #150.
  * the torch dynamo 5.
