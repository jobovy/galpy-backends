# Vacation status snapshot — 2026-10-03

**This file is OVERWRITTEN, not appended.** "Where are things now";
`log.md` is the trace. Refresh on every merge / branch change.

---

## Now

    feat/backends   767193159   rebased onto main 7c675eb63 (2026-10-04, Jo applied #1626).
    actionAngleVerticalInverse ON THE BACKENDS: stack #1629 = #1627 -> #1628 -> #1630.

## Open PRs

    #1627 backend/vi-mm-eval 6e858e7af -- VI (1/3): momentum-matched map evaluates natively
          (eval_ppoly tables, brentq angle inversion w/ IFT, regular sin(eta)/sin(tau));
          construction TEMPORARILY numpy-pinned; 17x2 skips removed. GREEN.
    #1628 backend/vi-mm-construct 02489d2cc (stacked on #1627, stack #1629) -- VI (2/3):
          native construction, differentiable in potential params (batched GN + 1 IFT
          step, cubic_hermite_coeffs), jit-able end to end. GREEN (b108c23e3).
    #1630 backend/vi-legacy-eval 9bb4d1d36 (stacked on #1628) -- VI (3/3): legacy modes evaluate
          natively (brentq angle, _SplineXP, backend map_coordinates order 5 + mirror-fold fix);
          construction numpy (no backend-param pots). All 49 VI tests in 4 backend modes. CI running.
    AVX-512: #1614's simdlen=256 workaround CONFIRMED on an avx512f runner (EPYC 9V74,
          #1613 run) -- inductor-kingdf_W0 PASSED. Keep it (removing re-exposes the GCC ICE).
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
