# Vacation status snapshot — 2026-09-25

**This file is OVERWRITTEN, not appended.** "Where are things now";
`log.md` is the trace. Refresh on every merge / branch change.

---

## Now

    feat/backends   ee3eff6c6   (= #1573)
    merged: stack #1570 -- #1567 (compiled C-integrator orbits, 2.14
    singledispatchmethod fix), #1569 (method="torchode" + CI install + extra),
    #1572 (kingdf second W0-derivatives), #1573 (compile-workflow tests).
    closed: #1571 (premise was stale bisect state).

## Open PRs

none.

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

1. Jo's decision: eddingtondf numpy fE wrong near Emin (blocks OM jit sampling).
2. Jo's decision: numpy NaN edge values; TwoPower Phi cancellation r/a < 1e-5.
3. Item 4: stream code (streamspraydf/streamgapdf/streamdf) on torchode for
   compiled torch streams.
4. The three strict xfails in test_backend_compile_workflows.py: torch spray ->
   streamTrack (numpy fit path), torch qdf density (fake-tensor 0-d indexing in
   interpolate.py), jax Orbit.E at a traced time.
5. eager jax.grad of kingdf.fE takes ~2 min (pre-existing).

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
