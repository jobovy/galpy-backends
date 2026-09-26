# Vacation status snapshot — 2026-09-25

**This file is OVERWRITTEN, not appended.** "Where are things now";
`log.md` is the trace. Refresh on every merge / branch change.

---

## Now

    feat/backends   7236d252c   (= #1566)
    merged since the last snapshot: stack #1548 -- #1556, #1557, #1558,
    #1559, #1566 (all squash-merged by Jo)

## Open PRs -- stack #1570 (merge with `gh stack merge`)

| PR | what | CI |
|----|------|----|
| #1567 | torch.compile of C-integrator orbits (routing + no silent detach) + torch 2.14 singledispatchmethod recursion fix | running |
| #1569 | `method="torchode"`: compilable torch in-backend integrator; inductor tensor-alpha rtol workaround; get_namespace torch fast path | running |

#1569 NEEDS JO: workflow edit adding `torchode` to the torch install lines
(build.yml:350, backend-tests.yml:290) -- patch in the PR body. Until then its
tests self-skip on CI and the torchode lines show as uncovered.

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

1. Jo's decision: eddingtondf numpy fE wrong within ~1e-3 of Emin (quad over
   [0, 0.5/rphi]; NFW gives large NEGATIVE values at 63/202 knots). Blocks
   OM jit sampling. Fixing changes numpy output there.
2. Jo's decision (from #1554/#1555): edge values NaN on numpy too; TwoPower
   Phi cancellation below r/a ~ 1e-5.
3. Jo's decision (#1569): torchode in the galpy[torch] extra? (pins typeguard 2.x)
4. Parked: kingdf torch path via torchode ("revisit kingdf later").
5. RETRACTED: "torch 2.14 cannot compile linalg.solve" was stale compiler-bisector
   state in my local inductor cache (clean cache: compiles fine). #1571 built on
   it -> recommend closing. Compiled torch stream tracks DO fail, for an older
   reason (streamTrack's numpy fit path).

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
