# Vacation status snapshot — 2026-08-24 (gh#1374 MERGED; gh#1380 Group-1 ledger OPEN; PAUSED for Jo's #1204 review)

**This file is OVERWRITTEN, not appended.** It is the "where are things now"
answer. `log.md` is the trace; read this first and go there only for the why.

---

## Nothing is blocked on Jo except the main->develop rebase (parked, Jo applies)

## Branch state

    main                   5d7b7e76
    feat/backends          982b1781
    feat/backends-develop  856ceccc   (ledger xfail 8 / slow_skip 61 / skip 68)

## Open PR — gh#1380 ledger/group1-measured @ c2744b79 (CI running)

Group 1 done: all 10 previously-unmeasured jax deferrals measured. 3 un-deferred
(25.2s / 83.4s / 139.6s), 7 stay with measured notes. jax slow_skip 33 -> 30,
UNMEASURED 10 -> 0.

## PAUSED for Jo's review of #1204 (as agreed)

Remaining 30 jax deferrals are ONE root cause: jax eager per-call overhead
(~5 ms flat) -- Python-loop orbit integration and per-pixel grid loops. That is
the vectorization project (tasks #74/#79/#130), weeks not days, deliberately NOT
started before the review.


## JO'S PLAN: finish gh#1374 -> Group 1 measurement sweep -> PAUSE for #1204 review

Group 1 = the 10 jax slow_skip entries carrying NO measured time. Sweep running
in /var/tmp/g1sweep (whole files under --backend jax, per-test cap 900s via
signal so it MEASURES rather than kills). Remaining 23 are one root cause: jax
eager per-call overhead (~5 ms flat) -- a weeks-long vectorization project, NOT
before the review.


## Live findings from gh#1368

* **`torch.special.gammaincc` is ~6 digits worse than scipy for a >~ 21.**
  |dQ| <= 2.2e-16 for a<=12, but 2.4e-10..4.0e-10 for a>=21 (jax <= 9e-15) — a
  discrete `igammac` algorithm switch. Diagnose this class by evaluating the
  residual of the NUMPY root under each library BEFORE suspecting the solver.
  Blast radius in galpy: EinastoPotential only (PowerSphericalPotentialwCutoff
  uses a<1.5). Recorded in memory `torch-op-gotchas-vs-numpy-jax`.
* **torch has no `igammac` derivative w.r.t. the order** — torch grads through
  any gammaincc are blocked upstream. A `gammaincc` fallback in
  `galpy.backend.special` (`_no_fallback` today) closes BOTH gaps. Follow-up.
* **Two pre-existing numpy Einasto defects, reported not fixed:** n=0.05 ->
  fsolve returns a NEGATIVE d_n (-1.720765); n>=60 -> Gamma(3n)=inf.

## Deferred / next up

* **main -> develop rebase** — PARKED, Jo applies. MEASURED (merge-base
  87256016): main +13, feat/backends +225, develop +107; 16 overlapping files;
  `git merge-tree` = **7 conflicting**, a LOWER BOUND. I prepare on a scratch
  branch, Jo applies. Unblocks the 2 `AdiabaticGrid_outsidegrid_c` xfails +
  task #103.
* **Differentiable `gammaincc` in galpy.backend.special** — see above; would
  serve every consumer, not just Einasto.
* **AdiabaticContractionWrapperPotential** — NOT small: two
  `scipy.optimize.fixed_point` calls, Blumenthal1986 iterates on an ARRAY with
  `interp1d` inside. Needs a fixed-point helper + backend interpolation.
* Off-grid AA fallback — gated on the rebase.

## Standing corrections to how I work

* **Never decrease coverage, not even one line** (Jo).
* **A value-only assertion cannot see a numpy fallback** — scipy happily takes a
  backend array and returns the RIGHT np.float64. Assert `is_backend_array`
  FIRST, then compare. Cost 3 of 4 tests silently passing on base this PR.
* **Match the test module's dtype idiom** — `test_backend_spherical.py` passes
  `dtype=torch.float64` per construction; it does NOT set a global default.
* **Read the source in the tree you MEASURED in**, not the main repo checkout.
* **`--author @me` is not "mine"** — check the `Claude-Session` trailer.
* **Check the same-SHA twin FIRST** on any shard red.
* **A failed shard uploads no coverage**, so a flake presents AS a codecov drop.

## Cron

`e935a60c`, hourly at :13, session-only, **created 2026-08-19 -> expires
~2026-08-26**. Verified alive this wake via CronList (3 days of headroom).
