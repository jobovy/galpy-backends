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

## 2026-10-06 (evening): H100 fold-in, one palette, direct labels

- H100 (Trillium) data from branch `gpu-h100` copied into `<fig>/data/h100/` and
  `_shared/orbit_speed/data/h100/`; `benchmarks/gpu/results_h100/`. Every GPU series
  with an H100 twin is drawn as an extra line (g2 g5 g6 g8 g12 gD3 gS4 gS5 gV2).
  g15/g17 pick `data/h100/` up automatically when those stages run.
- `_harness/palette.py`: one colour per framework x device (numpy black, C grey,
  jax sky/blue/navy, torch orange/vermillion/dark red for CPU/TITAN Xp/H100, galax
  green); jit solid, eager dashed; `label_lines` = direct labels in a right-hand
  column. `style.BACKEND_COLORS` now derives from it. All 12 speed figures converted.
- H100 vs TITAN Xp speed-ups: 8-12x for ODE stepping and eager torch (launch/
  dispatch floors identical on both; g12's H100 line is not yet saturated at N=3e5),
  30x for the bandwidth-bound Plummer force (g2: 1.7 TB/s = half the H100 peak),
  30-40x for arithmetic-dense kernels (gV2 root-finding, gS4 DF evaluation).
- **Load-corrupted this morning** (box at load ~190 during the CPU phase): g17 (all
  series) and gV2's CPU series must be re-measured on an idle box. g17's torch GPU
  series (NaN this morning: the GPU run lacked galpy #1639) were re-measured but at
  load ~9; kept only as a placeholder.
- **galpy regression found via g17**: numpy Staeckel `c=False` actionsFreqsAngles at
  N=1e4 takes 26.4 s on an idle box vs 2.07 s in August (9ec5bd2); C unchanged.
  ~13x, in the Python path. Bisect pending.

## 2026-10-07: H100 stages 2 folded in; Staeckel regression fixed (PR #1642)

- From `gpu-h100`: g15, g17 and the galax g8 series measured on the H100 copied into
  `data/h100/` (GPU series only are drawn; the branch's CPU series are from Trillium's
  EPYC and are not used); `logs/h100/` (report.md, stage logs) copied. g8 now has galax
  H100 lines (galax main 16a7e636b): galax's batched mode on the H100 ~ galpy's
  per-orbit mode on the H100.
- g17 H100: torch is nan at N>=1e5 and eager jax at N=1e6 (OOM: jax's retained pool
  starves torch in the same process; torch alone completes, see logs/h100/diag on the
  branch). The TITAN CPU series of g17 remain load-corrupted; re-measure after #1642.
- Staeckel c=False regression (commit dd0aba7ca, 2026-09-02; 200 nodes/integral):
  fixed in galpy PR #1642 (mesh matches C's 2x10 on the u side; 3.8x faster, 0.63 s vs
  2.35 s for 1000 orbits; the remaining gap to the 0.275 s of August is the fixed-count
  turning-point bisection).
- H100 torch CUDA suite: 47 failures, mostly torch 2.14 dynamo (fake-tensor ndarray,
  stream/graph-input mutation) and 7 eager float32/CUDA->numpy conversions; jax 2
  (a 5.6e-4 vs 2e-4 grad-vs-FD tolerance miss in evolveddiskdf). Not yet triaged.
- Paper draft (paper/ms): all sections drafted, 26 pages; see paper/ms/README.md.
- 2026-10-07 (later): main's per-orbit c=False path has the same 20-panel chi rule
  (#1357) but was NOT slowed by it (setup and turning points dominate there); matching
  C's mesh still gives 1.6x -> galpy PR #1644 (main). H100 follow-up instructions in
  tools/H100_FOLLOWUP.md; g17 torch-alone measure script added.
