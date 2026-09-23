# Vacation status snapshot — 2026-09-23

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
| #1547 | differentiable backend-key sampling w.r.t. the potential (constantbetadf + eddingtondf) | stack #1548 (bottom) | CI re-running on the rebased head a8836f09d (= d9faa198a) |
| #1551 | kingdf differentiable w.r.t. W0 (forward sensitivities + graft) | stack on #1547 | CI running |
| — | `backend/kingdf-tight-tol`: King ODE at rtol=1e-10 (Jo approved 2026-09-23); drops the tight-solve test fixture | stack on #1551, LOCAL, not yet submitted | tests running |
| #1549 | dynamical friction w.r.t. the host potential; backend jeans.sigmar n=100->200 | feat/backends | one macOS network flake (SIMBAD/pypi); auto-rerun queued |
| #1550 | psd_project exact gradient (Daleckii-Krein); diffrax accepts a potential list; Staeckel u0 measured | feat/backends | CI running |

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

## Next (Jo: "the rest once CI clears")

1. mass(r=0) / Phi(inf) NaN gradients in the potentials themselves (Jaffe,
   Plummer, Dehnen, DehnenCore, PowerSphCut at 0; Hernquist Phi(inf)).
2. Osipkov-Merritt general DF: p(v|r) via scipy -> velocity samples carry no
   gradient (same class as the frozen f(E) table).
3. eager-jax constantbetadf construction ~45 s (torch ~5 s): vectorize.
4. constantbetadf under jax.jit: traceable calibration.
5. Check CI durations of test_dynamfric_c[McMillan17] and
   test_constantbetadf_against_hernquist (~255-280 s locally, near the cap).

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
