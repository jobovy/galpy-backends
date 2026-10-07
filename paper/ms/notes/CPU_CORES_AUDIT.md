# CPU cores/threads behind every CPU timing series (audit 2026-10-07)

Dev box `geir`: 2 x Xeon E5-2697 v4 = 36 physical cores / **72 hardware threads**.
Who threads what (measured, not assumed):

* **galpy C** (orbit loop, action-angle loops): OpenMP, team size
  `min(n_objects, omp_get_max_threads())`; `omp_get_max_threads()` = `OMP_NUM_THREADS`,
  else the CPU affinity. `numcores=` does not reach it. dop853_c, N=1e4, hiN:
  2.22 ms/orbit on 1 thread, 0.174 on 16, 0.102 unpinned (72). Staeckel c=True,
  N=1e5: 30.7 s on 1 thread, 2.74 s on 16. A C call on ONE object runs one thread.
* **jax 0.10.1 XLA:CPU**: thread pool sized from the CPU affinity at client
  creation; `XLA_FLAGS=--xla_cpu_multi_thread_eigen=false
  intra_op_parallelism_threads=1` does NOT limit it (confirmed here: cpu/wall
  unchanged with the flags). Only `taskset` limits jax.
* **torch** CPU: `torch.get_num_threads()` defaults to 36 (physical cores) when
  unpinned, to the affinity when pinned, 1 under `OMP_NUM_THREADS=1`.
* **numpy/scipy**: single-threaded except BLAS (`OPENBLAS_NUM_THREADS`); none of
  the audited numpy series is BLAS-bound (cpu/wall = 1.00 measured).

"cpu/wall" below = process CPU time / wall time of one timed call, from quick
probes (`coreaudit/p_*.py` in the session scratchpad; pinned `taskset -c 24-35`
(12 cores) or `32-35` (4) vs one core, `nice 19`, memcap 16 GB; the box was
loaded by a timing chain on cores 8-31, so the absolute times are rough).

## Audit table

| Fig | series | cores/threads available (effective) | evidence | label/caption before | now |
|---|---|---|---|---|---|
| g2 (a,b) | numpy (CPU) | **1 core** (cpu/wall 1.00 with 12 available) | ran unpinned via rerun_figures_gpu.sh/trillium g2g5g6; elementwise Plummer F_R, no BLAS; probe `p_g2.py` | "numpy (CPU)" — no count; caption "faster than its CPU" | label "numpy (1 core)"; caption says one core |
| g5, g6 | — | no CPU timing series (GPU only; g6's compile is host-side but not a CPU series) | make.py | n/a | unchanged |
| g8, g12 | galpy C, `galpy_c_cpu[_hiN].json` | **72 threads** (OpenMP, unpinned) | rerun_gpu/chain3.sh + chain.sh s1 (no taskset, no OMP_NUM_THREADS); integrator stamp says "single-thread" (wrong); c3_omp_orbit_unset 0.102 ms/orbit vs 2.22 on 1 | **dishonest**: "single thread / sequential / numcores=1 / one core" (figure, caption.md, docstrings, ms captions + orbits.tex) | label "galpy C (72 threads)", light grey dash-dot |
| g8, g12 | galpy C, `galpy_c_cpu_1cores[_hiN].json` | **1 core** (OMP_NUM_THREADS=1, `taskset -c 16`) | chain3 c3_orbit1_*; JSON omp_num_threads=1, cpu_affinity=1 | new | label "galpy C (1 core)", grey solid; N=1..1e6 |
| g12 | galpy C, `galpy_c_cpu_16cores[_hiN].json` | **16 threads** (`taskset -c 16-31`, --numcores 16) | chain3 c3_orbit16_*; JSON omp_num_threads=16, cpu_affinity=16 | new | label "galpy C (16 threads)", mid grey dash-dot-dot; N=1..1e6 (0.17-0.2 ms/orbit); toggle `_SHOW_C16` |
| g8, g12 | galpy jax CPU (Dopri8), `jax_cpu[_hiN].json` | **up to 72 threads** (effective cpu/wall 1.3, and 2.4x SLOWER than pinned to 1 core: N=1e3 2.2 s on 1 core vs 5.2 s on 12 / 5.5 s plotted) | rerun_gpu chain.sh "shared" step: `CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu python galpy_jax_measure.py cpu ...`, unpinned, no XLA_FLAGS; probe `p_jaxorbit.py` (1, 2, 4, 12 cores: 2.2, 2.4, 3.5, 5.2 s) | no count; implied per-core | label "≤72 threads"; caption flags the pinned speed |
| g8 | galax CPU, `galax_cpu.json` | **up to 72 threads** (cpu/wall 1.1; 2x slower than 1 core: N=1e3 1.56 s pinned vs 3.05 s on 12 / 3.94 s plotted) | rerun_gpu/chain4.sh `CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu galax_measure_modern.py`, unpinned; probe `p_galax.py` | "galax CPU" — no count | label "galax CPU (≤72 threads)"; caption |
| g15 | C-STM (cstm_cpu), single + scaling N=1..1e5 | **1 thread** (72 available; one C call per orbit = OpenMP team of 1; cpu/wall 1.00 at N=64 with 12 available) | measure_* run unpinned (rerun_gpu chain/chain3, chain.sh s4); `inbackend_stm._c_stm_forward_augmented` loops orbits in Python | "galpy C (C-STM)" — ok-ish, no count | "CPU (1)" bar, "galpy C (C-STM, 1 core)" |
| g15 | diffrax CPU (diffrax_cpu) single + N=1..16384 | **up to 72 threads** (cpu/wall 1.2 at N=256; 1 core 1.83 s vs 2.15 s on 12) | unpinned, no XLA_FLAGS; probe `p_g15.py diffrax 256` | no count | "CPU (≤72)" bar, "jax jit, ≤72 threads" |
| g15 | torchdiffeq CPU single + N=1..65536 | **36 torch threads** (default; truly parallel at large N: cpu/wall 9.7 of 12 at N=4096) | unpinned, no OMP_NUM_THREADS -> torch.get_num_threads()=36; probe `p_g15.py torch 4096` (1 thread 13 s; 12 threads on a contended box 83 s) | no count | "CPU (36)" bar, "torch eager, 36 threads" |
| g17 (report only) | C (c=True) in the current JSON | **72 threads** (2.18 s at N=1e5 vs 30.7 s on 1 thread) | rerun_gpu chain.sh step g17, unpinned, no OMP | caption.md table says "C (c=True, 1 thread)" — **dishonest** | other agent; chain3 step 6 re-measures with OMP_NUM_THREADS=1 |
| g17 (report only) | numpy (CPU), numpy per-object OLD | **1 core** (cpu/wall 1.00 at N=1e4, OPENBLAS unset or 1) | probe `p_g17.py numpy 1e4` | no count | other agent |
| g17 (report only) | torch (CPU) | **36 threads** (unpinned, no OMP) | rerun_gpu g17 step | no count | chain3 c3_g17_measure has OMP_NUM_THREADS=1 -> 1 thread |
| g17 (report only) | jax (CPU) eager / jit | **up to 72 threads** | unpinned | no count | other agent's make.py pins jax CPU to one core in a child when OMP_NUM_THREADS=1 (chain3 step 6) |
| gD3 | numpy (sampling + construction) | **1 core** of 4 pinned (cpu/wall 1.00) | env cpu_affinity=4 (taskset); probe `p_gD3.py numpy 1e6` | caption "4 pinned cores… torch 1 thread" — numpy count implicit | "numpy, 1 core" |
| gD3 | jax eager / jax jit CPU | **up to 4 cores** (jit sample N=1e6: cpu/wall 1.8; 0.39 s vs 0.55 s on 1 core) | env cpu_affinity=4; probe `p_gD3.py jax 1e6` | not stated per series | "jax eager/jit, ≤4 cores" (from env stamps) |
| gD3 | torch CPU | **1 thread** (affinity 4) | modes_measured["torch"]: cpu_affinity 4, torch_threads 1 (the top-level env's torch_threads=4 is from the superseded first run) | caption "torch 1 thread" ok | "torch eager, 1 core" |
| gS4 | numpy (construction incl. C orbits; sample; eval) | **1 core** of 4 pinned (construction cpu/wall 0.9: one orbit per C call) | env_cpu cpu_cores_used=4; streamdf track loop calls aA on one Orbit at a time; probe `gS4_build.py numpy 4` (26 s wall, 23.6 s CPU incl. import) | caption "4 cores" — implied all used 4 | "numpy, 1 core" |
| gS4 | jax CPU eager/jit (+cold) | **up to 4 cores** | env_cpu cpu_cores_used=4; construct subprocesses inherit affinity | "4 cores" | "jax …, ≤4 cores" |
| gS4 | torch CPU eager | **1 thread** | torch_cpu_threads=1 | not in caption | "torch eager, 1 core" |
| gS5 | numpy (CPU, C) N=100..1e5 (+3e5, 1e6 queued, chain3, 4 cores) | **4 threads** (OpenMP over 2000-particle chunks; cpu/wall 2.5, 1 core 2x slower) | env_cpu cpu_cores_used=4; numpy_largeN_extended cores=4; probe `p_gS5.py 1000` (1.39 s vs 2.84 s on 1 core) | "4 cores" | "numpy/C, 4 threads" |
| gS5 | jax CPU eager/jit | **up to 4 cores** | env_cpu | "4 cores" | "jax …, ≤4 cores" |
| gS5 | torch CPU eager | **1 thread** | torch_cpu_threads=1 | in docstring only | "torch eager, 1 core" |
| gV2 | numpy, numpy loop over tori | **1 core** | env cpu_run: OPENBLAS=1, OMP=1, affinity 72 | ms caption "single CPU core" ok for numpy | "numpy, 1 core" |
| gV2 | torch CPU | **1 thread** | env torch_threads=1 | ok | "torch eager, 1 core" |
| gV2 | jax CPU eager / jit / jit vmap (eval + construction) | **up to 72 threads** (XLA_FLAGS set but ignored; cpu/wall ~6 for a large elementwise jit, ~55 for a matmul — coordinator's check) | env cpu_run: XLA_FLAGS single-thread, cpu_affinity 72 | **dishonest** before today ("single-threaded"); caption.md already corrected by the coordinator; ms caption still says "single CPU core" | "jax …, ≤72 threads" (from env stamps) |

H100/Trillium CPU series: only g2's numpy (1 core, elementwise) is plotted; the
other h100/ JSONs contribute GPU series only.

## Series that need re-measurement for a 1-core comparison

Pin to ONE core with `taskset -c <core>` (jax honours nothing else) plus
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` (C and torch), e.g. core 16; prefix
`CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu`, `PYTHONPATH=$WT:$REPO GALPY_WORKTREE=$WT`
and the memcap wrapper.

| Fig | series | N | how | est. time on 1 core |
|---|---|---|---|---|
| g8, g12 | galpy jax CPU Dopri8 | lowN N=1,10,100,1e3,1e4 (nt=1000); hiN N=1e3,1e4,1e5,3e5 (nt=16) | `galpy_jax_measure.py cpu lowN` and `... cpu hiN` under `taskset -c 16`. The script overwrites `jax_cpu[_hiN].json`: first copy the 72-thread files aside (e.g. `jax_cpu_72threads[_hiN].json`), then rename the outputs to `jax_cpu_1cores[_hiN].json` and restore; switch g8/g12 `_SERIES` to the 1-core files (label "1 core") | lowN ~5 min, hiN ~1 h (3e5: ~11 min per call x 4) |
| g8 | galax CPU | N=1,10,100,1e3,1e4,3e4,1e5 | `taskset -c 16 /var/tmp/galax-main/.venv/bin/python paper/figures/g8_galax_orbit_speed/measure/galax_measure_modern.py` (writes /var/tmp/galax-bench/galax_cpu.json); copy to `g8…/data/galax_cpu_1cores.json`; switch `_SERIES` | ~15 min |
| g15 | diffrax_cpu (single + N=1..16384), torchdiffeq_cpu (single + N=1..65536) | as listed | the measure scripts rewrite `icderiv.json` (measure_scaling resets `scaling`), so: back up `icderiv.json`; run measure_icderiv.py, measure_scaling.py, measure_largeN.py with the GPU hidden under `taskset -c 16 OMP_NUM_THREADS=1`; merge only `single.{diffrax,torchdiffeq}.cpu` and `scaling.*.{diffrax,torchdiffeq}_cpu` into the backup; set `_CPU_THREADS` in g15 make.py to "1" | ~30-40 min |
| g17 | C, torch CPU, jax CPU (eager, jit) | full grid | **already queued**: chain3 step 6 (`OMP_NUM_THREADS=1`; the other agent's make.py pins jax CPU) | hours (chain3) |
| gV2 | jax CPU eager, jit, jit vmap (eval N=10..1e6; construction nE=8..256) | as listed | `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 taskset -c 16 python paper/figures/gV2_vertical_inverse_speed/make.py --devices cpu` (re-measures all CPU series, merges; labels follow the new env stamps) | ~15-25 min (was 12 min unpinned) |
| gD3 | jax eager, jax jit CPU (sampling N=1e3..1e7; construction, 4 DFs) | as listed | `taskset -c 16 python paper/figures/gD3_spherical_df_speed/make.py --modes jax,jax-jit` (merges; per-mode env stamp -> label becomes "1 core") | ~30-45 min |
| gS5 | jax eager, jax jit CPU N=100..1e5; numpy/C N=100..1e5 (4 threads) | as listed | `taskset -c 16 OMP_NUM_THREADS=1 python -m paper.figures.gS5_streamspray_speed.make --measure cpu` — re-measure ALL CPU series together (labels read the single `env_cpu` stamp); numpy/C on one thread is ~2x slower | hours (jit compile 36-50 s per N; 1e5 points) |
| gS4 | jax CPU eager/jit (construction cold/warm, nch 4..32; sample/eval N=100..1e5) | as listed | `taskset -c 16 OMP_NUM_THREADS=1 python -m paper.figures.gS4_streamdf_speed.make --measure cpu` — all CPU series together (single `env_cpu` stamp) | many hours (jit construction 15-24 min compile x 4) — optional |

Not needed: g2 numpy, C-STM (g15), gD3/gS4 numpy and torch, gV2 numpy and torch
(all already one core); g8/g12 galpy C (1-core files done by chain3).

## Manuscript (paper/ms) text to change

* orbits.tex fig:galax caption: "Grey: \galpy's \cc\ integrator on one core." ->
  "Grey: \galpy's \cc\ integrator on one core (solid) and on all 72 hardware
  threads of the CPU (dash-dot)." ; "Blue: … on the CPU" -> add "(the CPU lines
  could use all 72 threads)".
* orbits.tex fig:orbitspeed caption: "The grey line is \galpy's \cc\ integrator
  \texttt{dop853\_c}, one orbit at a time on a single core." -> "The grey lines
  are \galpy's \cc\ integrator \texttt{dop853\_c}, which parallelizes over orbits
  with OpenMP, on one core (solid), 16 threads, and all 72 hardware threads
  (dash-dot)." ; "under \jax\ on a CPU" -> "on a CPU (up to 72 threads)".
* orbits.tex §Speed (lines ~205-217): "with the \cc\ integrator, one orbit at a
  time on one core" / "0.08 to 0.1 ms per orbit" / "pass single-threaded \cc\ near
  3e4" are the 72-thread numbers; one core is 2.2-2.4 ms/orbit and every GPU line
  passes it near N=1e3 (H100: ~100-280x cheaper per orbit at the largest N). The
  "also runs in parallel over cores" sentence becomes the explanation of the two
  lines.
* orbits.tex fig:icderivcost caption: add "On the CPU the \cc\ variational
  integrator runs on one core, the \diffrax\ solve could use up to 72 threads and
  \texttt{torchdiffeq} 36."
* design.tex fig:throughput caption: "with \numpy\ on the CPU" -> "with \numpy\ on
  one CPU core"; body "the gain over the CPU" -> "over one CPU core".
* actionangle.tex fig:vinvspeed caption: "on the backends on a single CPU core" is
  wrong for jax -> "\numpy\ and \torch\ on one CPU core, \jax\ on up to 72 CPU
  threads, the TITAN~Xp, and the H100".
* actionangle.tex §Speed line ~199 ("as fast as the single-threaded \cc\ path"):
  the current g17 C series is 72 threads; fix after chain3's 1-thread g17.
* dfs.tex fig:dfspeed caption: add "CPU series on 4 pinned cores: \numpy\ and
  \torch\ on one, \jax\ on up to four."
* streams.tex fig:streamdfspeed and fig:spraysp captions: add "CPU series on 4
  pinned cores: \numpy\ (gS5: its \cc\ integrator on 4 OpenMP threads; gS4: one
  core) and \torch\ on one thread, \jax\ on up to four."
