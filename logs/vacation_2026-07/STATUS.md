# Vacation status snapshot — 2026-08-12 (gh#1299 in CI)

**This file is OVERWRITTEN, not appended.** It is the "where are things now"
answer. `log.md` is the trace; read this first and go there only for the why.

---

## Nothing is blocked on Jo

Both rebases LANDED (2026-08-12). Work happens on `feat/backends-develop`
again, with PRs targeting it — those are mine to squash-merge.

## Active PR

**gh#1299** `backend-jit/conventions-registry` -> feat/backends-develop.
Test-side only; **no `galpy/` source changes**, so the numpy path is untouched
by construction.

The bare-numpy conventions checker was gated on a hand-maintained allowlist that
had drifted to 15 of 48 backend-compatible potential modules. All 64 sites it
flagged in the unregistered 33 were **false positives** (31 scalar constants,
33 numpy-guarded dual-path branches) — zero real defects — so the fix is the
checker: a self-policing constant allowlist, a backend-reachability-aware
visitor, and a **derived** module registry. Coverage 15 -> 50 modules.

Verified by injecting a bare `numpy.*` on the BACKEND side of each of the four
dual-path modules — all four caught. 95 passed on numpy, forced jax, forced
torch.

## Branch state

    main                    87256016
      feat/backends         982b1781   225 commits onto main   (LANDED)
        feat/backends-develop bf520538  79 commits + gh#1298   (LANDED)
          gh#1299             d6898535  conventions registry   (CI running)

## Live traps

  * `rerere` replays my resolutions, including ones recorded during an
    ABANDONED cherry-pick. Verify anything it auto-stages.
  * The 2 stream-track gradient tests (`test_determine_stream_track_
    differentiable_{potential,progenitor}`) are **FLAKY**, proven by a same-SHA
    re-run (2 failed -> 6302 passed). Re-run before investigating. The
    adaptive-solver-tolerance mechanism I proposed is **REFUTED** (defaults are
    already 1e-12; FD noise ~5e-9 vs a 3e-3 bar) — do not re-derive it.
  * Cron `6a357c84` created 2026-08-11 -> **expires ~2026-08-18**. Re-create
    before then; it has already died twice.
  * develop's `-inf, inf` quad in `surfdens` has **no `epsabs`** — plausibly the
    same latent bug main fixed elsewhere. A follow-up PR, not a rebase edit.

## Queue after gh#1299

  * **Needs Jo**: the units/`--jit` harness-skip policy (33 jax traced failures
    = ONE astropy boundary). I proposed a harness-level skip over 33 ledger
    entries; not yet ruled on.
  * task #215 — widen `COMPUTE_METHODS` (100 namespace-aware defs outside it,
    measured 0 unguarded sites, needs a curated criterion).
  * task #66 (RNG-fragile `test_impulse_deltav_general_curved`), #49
    (`coerce_coords` device across positional + kwargs), #130, #150.
  * the torch dynamo 5.
