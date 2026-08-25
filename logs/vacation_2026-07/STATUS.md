# Vacation status snapshot — 2026-08-25 (late)

**This file is OVERWRITTEN, not appended.** "Where are things now"; `log.md` is
the trace.

---

## Open PRs (2 = my cap)

* **gh#1382** `fix/quadrature-batching-anyaxisym-units` — findings 2 + 10 + the
  torch `.to()` construction bug. **CI 154/154, mergeStateStatus CLEAN.**
  NOT merged yet: the pre-merge duplication audit found that the `on_nodes`
  closure I added to `finite_part_quad` duplicates the module-level, documented
  `node_axis()` helper five lines up the same file. Fix is committed in the
  worktree `/var/tmp/quadfix` (closure deleted, 5 call sites rewired). Pushing
  it costs one full CI cycle; merging without it ships a duplicate. Decision:
  push the fix, then merge — the audit gate exists for exactly this.
  Its earlier ubuntu-qdf red was **wget** failing SSL on the `lcov` download
  during setup (exit 4 is wget's, not pytest's); re-run passed.
* **gh#1381** `fix/gammainc-endpoint-grad` — finding 1 + the int-dtype
  follow-up. **The `pull_request` arm is fully green** (Linux/Mac, notebooks,
  All-backend, Windows). The `push` arm carried the SIMBAD-outage reds plus one
  job cancelled at GitHub's 6h cap; both re-run dispatched now that the runs
  finally completed. `all-clear` re-runs last.

## Committed locally, PR HELD behind the cap

* **`fix/coords-inverse-transforms`** @ 17802043 (worktree `/var/tmp/rev1`) —
  findings 3 + 4 + 6 (**finding 6 complete, 7 of 7**), plus:
  - the f64-constant/f32-data promotion fix, hoisted to `galpy.backend` as
    `promote_common_dtype` (helper-placement audit);
  - `scalarDecorator` no longer lets a plain-Python-scalar call fall to torch
    float32 (user-facing only — conftest sets float64, so no test outcome
    changes; regression test controls the dtype itself and is verified to fail
    without the fix);
  - a torch `.T`-on-1-D deprecation guard.
  Gate: `tests/test_coords.py` 98/98 under numpy, torch AND jax; numpy
  byte-identity verified with a 1-ulp negative control; gradients grad-vs-FD
  1e-8..1e-11 where non-vacuous. **Ledger delta: 0.**

## FOR JO — a pre-existing red on develop, not mine

`tests/test_galpypaper.py::test_orbmethods` fails on `feat/backends-develop`
under **plain numpy** (and identically under torch): `o.vR(5.0, vo=220.0)`
gives 45.201138694253274 against a hard-coded 45.202530965094553, i.e. 1.392e-3
against a 1e-3 tolerance. `feat/backends` gives 45.202993419433426 and passes.
Established by reverting only my stack's five source files to the develop base
INSIDE the same worktree (so the built C extension is held constant) — the
value does not move, so my commits are not the cause. Something merged between
feat/backends and eabc3a5a shifted the orbit result. Worth a look: it is a
numpy-side red sitting on the integration branch.

## Open review findings not yet done

* **finding 5** — decided on my side: `set_jit` only sets a ContextVar and has a
  paired `jit(mode)` CONTEXT MANAGER, so having it turn the backend on would
  give `with jit("jax"):` two exit obligations and silently swap the user's
  backend. Going with "require an active backend, raise otherwise". Not started
  (cap).
* **findings 8 & 9** — traced constructor params + `hyp2f1` scalar params; all
  four finding-8 sites reproduce `TracerBoolConversionError`.
* **finding 11 (mine, reported not fixed)** — no torch gradient through surfdens.

## Housekeeping

* Cron `5451d57e` (hourly at :13) verified alive 2026-08-25; created 2026-08-24,
  expires ~2026-08-31.
* Root filesystem hit 100% on 2026-08-25 and wedged every command with ENOSPC.
  Freed ~10G from never-GC'd `torchinductor_*` caches; expect recurrence.
