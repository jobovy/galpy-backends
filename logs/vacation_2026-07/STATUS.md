# Vacation status snapshot — 2026-09-29

**This file is OVERWRITTEN, not appended.** "Where are things now";
`log.md` is the trace. Refresh on every merge / branch change.

---

## Now

    feat/backends   60310a179   (OLD tip; to be force-pushed, lease on this sha)
    stack #1577 (#1574/#1575/#1584) merged.

    REBASE IN CI: draft #1592 (head backend/eddington-fE-near-Emin-rebuilt,
    base backend/feat-backends-onto-main). Stack of scratch branches:
      backend/feat-backends-onto-main   ffd411667  (= new feat/backends)
        470 replayed commits + port commit 778d8ed54 + fixes:
        a6772dee9 _PVRInterpolator grid=True default (main #1568 test)
        5c067f9cf backend n_new_pvr=1000 (#1568), x2deriv @backend_input
                  (#1519), _force2deriv->_x2deriv in tests
        2f7c66ac2 clamp backend inverse-CDF weight (XLA reciprocal ulp)
        f8d27a601 test_potential: main's tests as_numpy under forced
        98719c7c4 backend_skip: main's 12 new VerticalInverse tests
        ffd411667 spherical: backend epicycle branch (#1527); 8 main
                  near-circular/small-orbit tests LEDGERED (relative
                  problem not ported -> NEXT PR)
      backend/potential-edge-limits-rebuilt  ca3cd41ef  (-> #1578)
      backend/eddington-fE-near-Emin-rebuilt f9c566d11  (-> #1579; also
        main's Emin divergence criterion ported to the backend fE)
    CI round 3 running (runs 36524300846/867/872/889). Rounds 1-2 failed on
    the fixes above. All of main's new tests pass locally under jax+torch.

    Prepared (local, not pushed): backend/replace-compile-optouts in
    /var/tmp/stc2, on ffd411667: streamTrack fit compiles (traced recipe;
    inductor argmin/where-inf bug worked around), qdf ctor compiles (only
    the constant rg table opt-out). Push after feat/backends is updated.

## Open PRs

#1578, #1579 (to be force-updated to the rebuilt branches), #1592 (draft,
close after the force-push).

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
