# Vacation status snapshot — 2026-09-30

**This file is OVERWRITTEN, not appended.** "Where are things now";
`log.md` is the trace. Refresh on every merge / branch change.

---

## Now

    feat/backends   8d5769a41   rebased onto main (ffd411667) + #1578 + #1594 +
    #1593 + #1600 + #1598 + #1599 (analytic turning points backend-native) +
    #1602 (vmapped stream track).

## Open PRs

    #1596 (main) backend/twopower-forces-main 51ad82df4 -- TwoPower forces/2nd
          derivs/mass via incomplete beta; Codex rounds 1-4 addressed (alpha>beta
          hang, bounded caches, r=inf limits incl. beta=1 and beta=0, beta<0
          NaN at r=inf only (round 5), amp=0, alpha>=beta+2 split, direct mass for alpha>=1.5)
    #1597 backend/twopower-forces e44caac8b -- same + backends (beta<2 'hi'
          series sign fix; jit-vs-eager 3e-15 and alpha~3 forced-backend 1e-12
          bars approved by Jo)
    #1603 backend/gpu-validation 4b01c6fc3 -- GPU (CUDA) validation (rebased
          onto 8d5769a41). Decided: numpy integrators on CUDA -> error pointing to
          torchode/torchdiffeq; GPU-only tolerances OK; upstream issues NOT filed.
    Background agent: CUDA numpy-island removal from the main-suite GPU sweep
          (~90 library sites / 35 files), PRs grouped by module, stacked on #1603.
    Parked until that agent finishes (file overlap): SCF from_density; FDM
          dynfric torch slow-skip (564 s: python dop853 x 474k EOMs, ~40% of it
          coerce_coords 2.3M / asarray_on_device 6.2M calls in the Chandrasekhar/
          FDM force; torchode is SLOWER, >10 min) -> cut per-call coercion.
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
