# Vacation status snapshot — 2026-10-06 (03:45)

**This file is OVERWRITTEN, not appended.** "Where are things now";
`log.md` is the trace. Refresh on every merge / branch change.

---

## Now

    feat/backends   9f6c9d6f7 (#1631). Figure re-run + 6 new figures in progress
    (logs/figure_rerun_2026-10-06/log.md). CPU phase DONE + committed.
    Timing chain 1 (/var/tmp/figs = feat/backends + #1634/#1635/#1636 commits):
        g2 g5 g6 done; g17 running (since 01:10); then shared orbit_speed, g12, g15,
        benchmarks/gpu, re-render g8/g12/g15, gV2 cpu.
    Chain 2 (after chain 1): figs += #1637, #1636 torch-cuda re-verify, GPU series
        gV2 gD3 gS5 gS4, render.

## Open PRs

    #1633 (main) Chandrasekhar sigma_r log grid + natural spline. Pyodide wheel jobs
          fail on the ftp.gnu.org outage; prwatch.sh reruns when reachable.
    #1634 backend side of #1633. GREEN.
    #1635 brentq bracket_width= (VI jit 2.5-3.7x). GREEN.
    #1636 VI tables follow the potential's CUDA device (+ test as_numpy). GREEN.
          torch-cuda re-verify of the test commit queued in chain 2.
    #1637 streamdf torch progenitor constructs; misalignment traceable; jax.jit of
          construction works (723 s; 1.1e-5 rel from eager -- follow-up); jax.grad of
          the constructor fails in diffrax adaptive backward (follow-up).
          1 CI job timed out (kuzminkutuzov leapfrog, load) -> rerun requested.
    Decisions: constantbetadf numpy fE stays (outside jit is faster; Jo).
    On hold per Jo: main merge prep, docs.

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
