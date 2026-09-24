# Vacation status snapshot — 2026-09-24

**This file is OVERWRITTEN, not appended.** "Where are things now";
`log.md` is the trace. Refresh on every merge / branch change.

---

## Now

    feat/backends   87f4ec20b   (= #1546)
    merged since the last snapshot:
      #1541 sphericaldf.sample() under a gradient-carrying potential
      #1542 hierarchical velocity grid on the backend
      #1543 constantbetadf fE w.r.t. the potential (torch)
      #1544 namespace_from_arrays for value-namespace lifts (alias dropped)
      #1546 constantbetadf under jax too (eager grad; as_numpy_constant)

## Open PRs

| PR | what | base | state |
|----|------|------|-------|
| #1547 | differentiable backend-key sampling (constantbeta + eddington) | stack #1548 (bottom) | green 150/150 |
| #1551 | kingdf d/dW0 (forward sensitivities + graft) | stack on #1547 | green 150/150 |
| #1552 | King ODE at rtol=1e-10 (approved) + monotone-repair coverage test | stack on #1551 | CI |
| #1556 | osipkovmerrittdf differentiable velocity sampling (2-26% silent error; jax crashed) | stack on #1552 | CI |
| #1557 | constantbetadf construction ~4-6x faster (fixed-shape calibration; PCR tridiagonal spline solve) | stack on #1556 | CI |
| #1558 | constantbetadf under jax.jit (construction, fE, moments, sampling) | stack on #1557 | CI |
| #1554 | potentials: finite parameter gradients at r=0/inf (radial_limits) | feat/backends | CI |
| #1555 | backend hyp2f1 accurate at large negative z (TwoPower small r) | feat/backends | CI |

Merged since last snapshot: #1549 (dynfric host), #1550 (psd_project, diffrax list).

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

## Next

1. Watch the 7-PR stack + #1554/#1555 through CI.
2. Deferred for Jo's decision (listed in #1554/#1555): edge values NaN on numpy
   too; TwoPower Phi cancellation below r/a~1e-5 (formula rewrite changes numpy).
3. Clean up superseded local worktrees (/var/tmp/omdf, cbfast, cbjit, pedge,
   misc, dynfric2, audit3 once the stack merges).

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
