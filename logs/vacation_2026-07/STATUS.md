# Vacation status snapshot — 2026-09-21

**This file is OVERWRITTEN, not appended.** "Where are things now";
`log.md` is the trace. Refresh on every merge / branch change.

---

## Now

    feat/backends   1a1e2211d   (= #1534)
    merged since the last snapshot:
      #1530 dynfric v coercion
      #1531 osipkovmerritt dM/dE (panel split + support edge + cosine map)
      #1532 Orbit accessors (+ _pot_grad_namespace)
      #1533 Spline1D vector-valued + bc default (not-a-knot for n>=4)
      #1534 Spline1D consumers (streamTrack dual repr, SCF eval_ppoly clone)

## Open PRs

| PR | what | state |
|----|------|-------|
| #1535 | interpRZPotential -> Spline2D, Orbits one-spline-per-orbit | green; codecov phantom resolved (18 sessions, 99.97%), all-clear re-run |
| #1536 | jit as a CI matrix axis + traced-share reporting | 104 jax-jit test_quantity failures FIXED (see below); re-running |
| #1537 | qdf moments w.r.t. potential params + `_rg` array branch | **green 106/106** |
| #1538 | actionAngle grid classes delegate under a potential gradient | building |
| #1539 | evolveddiskdf: torch `xp.where` guard on odd moments | building |

## The differentiability audit sweep is COMPLETE

Write-up: `plan/differentiability_audit_followups_2026-09-19.md`.

Areas swept this round and what they produced:

* **actionAngle grid variants** (T4g) — both raised in `__init__` for ANY
  gradient-carrying potential. Fixed by delegating to the exact class (#1538).
  `stop_gradient` on the nodes was rejected: it returns a silently ZERO
  gradient, worse than the error.
* **evolveddiskdf** (T4i, T4i-b) — two findings. T4i-b (torch `xp.where` on a
  numpy-bool condition, odd moments only) fixed in #1539. T4i (the gate asks
  `is_backend_array(R)`, so a potential-parameter gradient with a float R is
  silently lost) is DEFERRED with its reasoning recorded: the guard's intent is
  sound and documented (task #117), only the predicate is too narrow, and
  widening it may route onto genuinely unmigrated paths.
* **DF `sample()`** (T4j) — raises under a gradient-carrying potential. A crash,
  not a wrong number; differentiable sampling is already a roadmap item.
* **streamspraydf** — needs nothing. It already probes for a backend
  theta/mass/stripping_pdf and discriminates a merely-forced context.

## The one that nearly went wrong

#1536's jax-jit `test_quantity` shard failed 104 tests. The shape matched a
remembered "Quantity is not traceable" boundary and my first instinct was to
ledger them. That would have repeated a mistake this project has already made
and retracted once.

It was not a galpy defect at all: **this PR's own traced-share counter** imports
galpy inside `pytest_configure`, which runs before test modules are imported,
locking in the default `astropy-units=False`. `conversion.py` binds `_APY_UNITS`
as an import-time copy, so `test_quantity.py`'s module-level config set never
reaches it, and every `physical_conversion` returned a bare array with the
CORRECT value and no units. Moved the install to `pytest_collection_finish`:
104 failed -> 344 passed, and the counter still reports 92.7% traced.

Two discriminators did the work and are worth reusing: the pass/fail ribbon in
file order (215 passes AFTER the first failure => not a leaked global), and
checking that today's symptom actually matched the remembered one (it did not —
units silently absent, not a tracer error).

## Standing constraints

* Jo merges the main PRs. I can merge develop, not main.
* I **cannot** edit `.github/workflows/**` — provide patches; Jo applies.
* Never name a working branch `feat/...` — use `backend/...`.
* numpy must stay byte-identical; coverage never decreases, not one line.
* Rebase only, no merge commits. Cancel the PRIOR sha's CI on push.
