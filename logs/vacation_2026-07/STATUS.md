# Vacation status snapshot — 2026-10-03

**This file is OVERWRITTEN, not appended.** "Where are things now";
`log.md` is the trace. Refresh on every merge / branch change.

---

## Now

    feat/backends   1cffb1ca8   ... + #1610-#1620 all merged (sky ICs, TwoPower,
    SCF/Multipole from_density, simdlen, gammainc, SIMBAD unskip, qdf _log_const,
    dynfric, CPU header). main: + #1611.

## Open PRs

    #1613 backend/f32-sweep 7bd7305a8 -- float64-by-default policy (scoped torch
          default dtype). +7db0d5954: py3.14 torch.compile fix (sys.modules closure)
          + sphericaldf._vmax_at_r numpy-2.5 __array_wrap__ deprecation; +7bd7305a8
          no-torch scope test (codecov patch miss). CI running.
    #1621 backend/qdf-samplev-fixed-shape bd0cd43c3 (stacked on #1613) -- qdf
          sampler pow-4 buckets. Fixed shape (1st commit) regressed torch (CI 300s
          timeouts); buckets: torch 21s, jax 217s (was 2837s) -- jax/jax-jit
          slow-skips RESTORED (CI not faster than this box here). CI running.
    #1623 backend/staeckel-defaultdelta-test 4e34284b4 -- test fix: 1-ulp
          jit-vs-eager delta difference amplified to 4.7e-11 vs a 1e-16 bar. CI running.
    AVX-512 (#1614 retirement): still NO kingdf_W0 run on an avx512f runner; the
          4 post-merge All-backend runs sat queued. Watching each test_backend shard.
    Eager-floor jax skips (4): measured NOT compile-bound (1-20%); they stay.
    On hold per Jo: main merge prep, docs.
    Remote scratch branches (ruleset blocks my delete): Jo.

## What is differentiable now (this round)

* constantbetadf: fE and construction w.r.t. potential params, torch AND eager
  jax (#1543, #1546). NOT under jax.jit (calibration loop).
* spherical sampling with a backend key (constantbetadf, eddingtondf): r and v
  vs FD to 4.3e-6 / 7.7e-5; constantbeta Hernquist d r_i/da = r_i/a to 1e-13
  (#1547). Four defects: r=0 mass knot NaN, detached CMF radii (double
  count), frozen f(E) table (11% off, silent), frozen p(v|r) extent (0.7%).
* kingdf W0 (#1551): all tables vs tight FD to <1e-7.
* dynamical friction w.r.t. host (#1549): sigma_r 9e-9, Rforce 1.4e-7.
* psd_project (#1550): exact; the old frozen gradient was 77-210% off
  (wrong sign) whenever it clipped.

## Next (2026-09-28, agreed with Jo)

1. Jo lands stack #1577 (#1574, #1575, #1584). NOT #1578/#1579.
2. Rebase feat/backends onto main (19 main commits incl. #1585/#1586; 32 files
   changed on both sides, 194 of 467 commits touch them) on a scratch branch
   backend/feat-backends-onto-main; #1585/#1586 files resolved to the
   feat/backends side; verify (per-file diff vs main, full numpy + backend
   suites, CI on the branch); then force-push with lease.
3. Rebuild #1578/#1579 on the rebased branch as the backend-only delta (the
   numpy/C parts are on main); land right after step 2.
4. IMMEDIATELY after: replace the torch.compile opt-outs -- streamTrack fit
   (#1575: run the jit-native recipe on torch) and the qdf constructor (#1574:
   dynamo-safe setup). Tests differentiating the fit/constructor inputs.
5. Open for Jo: TwoPower forces/second derivatives near beta=3 and at large r
   (also on main).

## Recurring traps this round

* **Anomaly mode flags inner torch.func.grad backwards in the FORWARD pass.**
  The "DivBackward0 in the startt calibration" chased for days was that; the
  real NaN was a PowBackward0 in the outer backward (Hernquist mass at r=0).
  Use check_nan=False forward, True only for the outer backward.
* **A table built at a detached parameter + an attached conversion counts the
  gradient twice.** Pick one parametrisation and attach consistently.
* **Frozen knots are not free when the values are differentiated**: the knot
  motion is an O(interpolation) derivative error that a small derivative
  amplifies (7% on p(v|r)).
* **FD on a noisy plain path** (scipy-adaptive fE) reads as several % at
  h=1e-4; FD on the gradient-carrying path instead.
* **Concurrent local suites cause 300 s cap timeouts** that look like
  regressions; re-time the test alone on both branches before believing it.

## Standing constraints

* Jo merges the main PRs. I can merge develop, not main.
* I **cannot** edit `.github/workflows/**` — provide patches; Jo applies.
* Never name a working branch `feat/...` — use `backend/...`.
* numpy must stay byte-identical (flag any exception: the King tolerance is
  one, approved); coverage never decreases, not one line.
* Rebase only, no merge commits. Real `gh stack`s for dependent PRs. Cancel
  the PRIOR sha's CI on push.
