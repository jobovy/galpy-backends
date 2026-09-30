# Vacation status snapshot — 2026-09-30

**This file is OVERWRITTEN, not appended.** "Where are things now";
`log.md` is the trace. Refresh on every merge / branch change.

---

## Now

    feat/backends   4599bdcba   rebased onto main (ffd411667) + #1578 + #1594 +
    #1593 (streamTrack fit / qdf ctor compile) + #1600 (kingdf compile) +
    #1598 (OM sample under jit; D-form constantbeta fE).

## Open PRs (all CI green unless noted)

    #1596 (main) backend/twopower-forces-main c07c3295c -- TwoPower forces/2nd
          derivs/mass from the incomplete beta (numpy+C); twin of #1597
    #1597 backend/twopower-forces a8eaee5e7 -- same + backends; util/C files
          byte-identical to #1596's (drop on next rebase)
    #1599 backend/orbit-analytic-turning-points 62c241b59 -- Orbit.rperi/rap/
          zmax/e(analytic=True) backend-native
    #1602 backend/streamdf-vmap-track 822a21b51 -- stream track chunks under
          jax.vmap; shared constant steps keep d/dparam exact; grad 1.2-1.8x
    #1603 backend/gpu-validation 6ec1706e8 (rebased onto 4599bdcba; CI running)
          -- GPU (CUDA): pytest --device cuda; to_host (strict host move, keeps
          autograd) at ~20 implicit numpy conversions; device-independent torch
          draws; Spline1D.derivative in-backend for backend queries; Orbit.E
          kinetic terms on the forced namespace; SCF alpha device anchor.
          torch AND jax backend_* sweeps green except: hardware (no Triton on
          sm_61), upstream torch (dynamo numpy under set_default_device, torch
          2.12+2.14; torchode CUDA event), fork+CUDA. OPEN for Jo: python
          integrators under forced torch on CUDA; GPU tolerances (eddington
          near-Emin 2.9e-8 vs 2e-8; sphericaldf jax-GPU 1-ulp). rperi analytic
          on CUDA after #1599. Non-backend suites --device cuda: running.
    Known pre-existing: test_sphericaldf dehnencore_in_nfw_*_directint fail out
    of file order (uses dfh._fE_interp built by an earlier test).
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
