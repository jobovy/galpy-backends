# Vacation status snapshot — 2026-08-22 (gh#1368 OPEN, awaiting CI)

**This file is OVERWRITTEN, not appended.** It is the "where are things now"
answer. `log.md` is the trace; read this first and go there only for the why.

---

## Nothing is blocked on Jo except the main->develop rebase (parked, Jo applies)

## Branch state

    main                   5d7b7e76
    feat/backends          982b1781
    feat/backends-develop  08db5d90   (ledger xfail 8 / slow_skip 61 / skip 68)

## Open PR

**gh#1368 `1ae24301` — EinastoPotential differentiable d_n solve (jax/torch).**
Branch `backend/einasto-dn-differentiable`, based exactly on develop 08db5d90.
Files: `galpy/potential/EinastoPotential.py`, `tests/test_backend_spherical.py`.
Ledger delta: NONE. Local gate: 4/4 arms green (231/231/231 + 1288 numpy).
A/B: 9 of 12 new tests fail on base; the other 3 are byte-identity guards that
must pass on both.

Merged earlier today: gh#1363 19f31a54, gh#1364 a4193e84, gh#1365 46ea6ade,
gh#1366 47875227, gh#1367 08db5d90.

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
