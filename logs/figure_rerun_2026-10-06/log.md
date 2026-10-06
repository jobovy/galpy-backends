# Figure re-run + new figures (2026-10-06)

Jo: "make sure the notes in galpy-jax are up to date and re-run all of the paper
figures with the latest code, including all of the speed tests ... more examples
(spherical DFs, 1D inverse action-angle, streamdf, streamspraydf) incl. GPU
timings. Autonomous." Then, on the g11 finding: "If there's an issue, fix it!!"

Code under test: galpy feat/backends 9f6c9d6f7 (worktree /var/tmp/fbp).
Procedure: tools/rerun_figures_cpu.sh (4 pinned groups), tools/rerun_figures_gpu.sh
(serial, waits for an idle GPU). Harness now stamps galpy's commit (env.galpy_git)
and writes the manifest under a lock (parallel runs).

## CPU phase (21:37-22:04)
- byte-identical: g0, g3, g18.
- round-off / MC-level data changes only: g1 (jax now bit-identical to numpy --
  what the caption already claimed; the committed PDF had drifted), g7, gF1,
  gD1, gD2 (MC 2e-3), gS1 (track 1e-5), gS2, gS3, g9 (noise-level), g10, g14, g16.
- g12 work_counts.py crashed: galpy's EOM helpers (_to_eom/_eom_rhs) are
  batch-first now; fixed the script; counts reproduce EXACTLY.
- g13: data had 4 points (2026-06-16); now 146. Script ignored the jax-jit /
  torch-jit ledger rows and used COMMITTER dates (all 2026-10-04 after the main
  rebase); now author dates + jit series + symlog + numbered milestones.

## g11 -> a real galpy defect (Chandrasekhar sigma_r)
g11's committed data (2026-06-10) used the C single-point Jacobian for the trace;
the script switched to FD of the PYTHON forces the same day (88a9e53) but was never
re-measured. Re-measured: max relerr 3.7e-3 (was 5.7e-6), rising once r < 0.06.
- Not a regression: identical on pre-#1619, main, #933 and even 43451e7a
  (bisected the WRONG thing first -- the variable was the script, not galpy).
- Cause: sigma_r(r) tabulated on linspace(minr, maxr, nr) (spacing 0.05 for the
  default maxr=25) and interpolated by DIFFERENT splines: C GSL cspline (natural)
  vs Python InterpolatedUnivariateSpline (not-a-knot). 2.6e-3 apart at r=0.05;
  both 20-60% off the exact Jeans sigma_r at r < 0.03.
- Proof: Python force on the natural spline -> 5.729e-6 exactly the old number.
- Fix (#1633 main, #1634 feat/backends): geomspace grid + natural spline on every
  path. sigma_r ~1e-9 at small r; sinking orbit C vs Python 2e-10 (was 20% in r).
  g11 re-measured on the fix: 5.7e-6. Final r of the test orbit 0.0017 -> 0.0031
  (old C used a sigma_r ~50% wrong there).

## Agent findings (new figures)
- gS4: streamdf with a torch progenitor cannot be constructed (eager torch grad
  tensors hit numpy: under_trace gates miss requires_grad). Fixing.
- gV2: backend brentq under jit ran 100 halvings (traced bracket has no width)
  instead of 43 -> bracket_width= hint; VI jit 2.5-3.7x faster (N=1e4 6.0->1.65 s).
  Remaining (not fixed): bisection vs numpy's Newton (~5x more kernel passes).
- Machine: load avg up to 160 (another user at ~64 cores) during the agents'
  CPU timing runs; they ran single-threaded and pinned.

## PRs from this round (feat/backends unless noted)
- #1633 (main) + #1634: Chandrasekhar sigma_r log-r grid + natural spline everywhere.
- #1635: backend brentq bracket_width= (VI jit 2.5-3.7x).
- #1636: VI tables follow the potential's device. Found by gV2's GPU run: CUDA
  potential params + torch default device CPU -> device mismatch in the GN step
  and the legacy row interp. `--device cuda` sets torch's DEFAULT device to cuda,
  which HID it from the suite. + test-side as_numpy for --backend torch --device cuda.
- #1637: streamdf with a torch progenitor constructs (requires_backend_grad at 5
  under_trace gates + _ns_coerce namespace from any operand + Orbit time grid).
  _span_grid silently DROPPED d(grid)/d(extent) for an eager torch extent.
  Full torch build: 1346 s (eager reverse-mode Jacobians through torchode).

## Decisions
- constantbetadf numpy fE (2-10 min): NO internal jit (Jo). Outside jit 5.8 s /
  0.13 s warm beats an internal jit (17.4 s, recompiles per DF) -- the backend fE
  is one vectorized GL integral vs ~7000 sequential adaptive quadratures.

## Ops notes
- GNU mirror (ftp.gnu.org) outage 2026-10-06: Pyodide wheel builds fail; rerun
  when reachable (watcher does it).
- VI CPU gate's "9 numpy failures" = the gate started before build_ext finished
  in the fresh worktree (.so at 23:53, gate 23:52); rerun: 101 passed.
- Timing chain on /var/tmp/figs = feat/backends 9f6c9d6f7 + #1634 + #1635 + #1636
  commits (cherry-picked), so jit timings include the brentq fix.
