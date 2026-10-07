# H100 re-measurement of the galpy-jax speed figures (Trillium)

Branch `gpu-h100`, measured 2026-10-06/07 by a Claude session on the Trillium
GPU login node, every stage as a one-GPU Slurm batch job (`def-bovy`), one job
at a time. Logs of every stage are in this directory (`<stage>.log` = the
stage's command output, `<stage>.slurm.out` = the job's stdout).

## Follow-up run (2026-10-07)

`gpu-h100` was rebased onto `origin/main` (the pre-rebase tip is kept locally as tag
`gpu-h100-run1`). Every conflict was a data file, PDF or the manifest, and was
resolved to this branch's version; main's copies of the run-1 data live in
`data/h100/`. New stages in `tools/trillium_stage.sh`: `g17torch`, `gV2`,
`triage`. Everything below ran on galpy "follow-up" (table above).

**g17 (stage `g17`, run 2; 46 min instead of run 1's 4 h 31 min, because of #1642).**
The run-1 numbers further down are stale for g17.

| series | N | run 1 | run 2 | |
|---|---|---|---|---|
| jax GPU jit | 1e5 | 0.325 s | 0.161 s | 2.0x |
| jax GPU eager | 3e5 | 13.3 s | 6.63 s | 2.0x |
| jax GPU eager | 1e6 | OOM | 12.5 s | now fits |
| torch GPU, in-process | 1e5 / 3e5 / 1e6 | OOM / OOM / OOM | 0.786 / 1.38 s / OOM | |
| **torch GPU, alone** (`g17torch`, `torch_gpu_only.py`) | 1e5 / 3e5 / 1e6 | – | 0.778 / 1.37 / **3.50 s** | TITAN 1e6 (pre-#1642): 8.36 s |
| numpy CPU (EPYC node) | 1e5 | 424 s | 46.5 s | 9.1x (the #1642 mesh) |
| C (c=True) | 1e5 | 1.04 s | 1.03 s | unchanged |

After `g17torch`, the torch GPU series in the g17 JSON are the torch-alone
ones, complete to N=1e6. `nvidia-smi` sampled every 5 s
(`g17torch_mem.csv`) peaked at 43.9 of 81.6 GB. This much lower peak (run 1's
diagnostic saw 65 GB at 1e6) is #1642's doing. The `g17` job's output lacks
the final `>>> done` line, because I edited `tools/trillium_stage.sh` while that
job was running (reverted within a minute). The measurement itself completed
and exited 0.

**gV2 after #1641 (stage `gV2`, gV2 only, 2 min).** GPU series only; CPU
series untouched.

| eval, N=1e6 | before (#1635 bisection) | after #1641 (Newton) | |
|---|---|---|---|
| jax GPU jit | 30.0 ms | 10.9 ms | 2.8x |
| jax GPU jit, vmap over tori | 39.3 ms | 11.3 ms | 3.5x |
| torch GPU | 99.9 ms | 69.8 ms | 1.4x |
| jax GPU jit, small N (floor) | 0.6 ms | 0.3 ms | |

Construction times are unchanged (warm jit at n_E=256: 5.8 ms).

**Test triage:** see `tests_triage.md`. In short, **there is no torch-2.14
regression**: all 42 torch failures also fail on torch 2.12. They break down
as:
- 28 galpy CUDA bugs: 26 are a numpy operand reaching dynamo next to a CUDA
  tensor; one is `parallel_map` forking after CUDA init; one is the dop853_c
  handoff calling `.numpy()` on a grad tensor.
- 7 test-suite `.numpy()` calls on CUDA results.
- 7 torchode + `torch.compile` on CUDA (third-party).

The 2 jax evolveddiskdf gradient tests are **nondeterministic** (the autodiff
value takes at least three values across 9 runs; the FD value is stable), as
is the atol=0 `finite_part_quad` test. 2 tests (`test_misalignment_traces`)
are fixed upstream.

**Larger N (stage `hiN2`, 2026-10-07, job 1054212, 12 min 40 s).** The
branch was first rebased onto main ab37edd, which includes 8c72ba6 (`--Ns` option + `hiN2` stage);
the `trillium_stage.sh` conflict was resolved by keeping both sides. galpy is
`feat/backends` @ 9a877712 (#1641 and #1642 merged, no cherry-picks), rebuilt
with `build_ext --force`.

Each N ran in its own process and was merged into the H100 JSONs with `--Ns`.
**Every requested N ran: no OOM, no `stopped_at`, no NaN, empty gD3 `errors`,**
and no OOM, `RESOURCE_EXHAUSTED` or traceback anywhere in `hiN2.log`.
Every `extra_points` record has `replaced: []`, so no existing point changed.

| series (JSON) | previous largest N | largest N that ran | wall | per item |
|---|---|---|---|---|
| g12/g8 jax Dopri8 hiN (`_shared/orbit_speed/data/jax_gpu_hiN.json`) | 3e5 (7.40 s) | **3e6** (1e6: 16.4 s) | 46.5 s | 0.0155 ms/orbit (1e6: 0.0164) |
| g12 jax Dopri5 hiN (`g12/data/jax_dopri5_gpu_hiN.json`) | 3e5 (5.47 s) | **3e6** (1e6: 13.8 s) | 37.4 s | 0.0125 ms/orbit (1e6: 0.0138) |
| g12 torch dopri5 (`g12/data/torch_cuda.json`) | 1e6 (9.22 s) | **3e6** | 25.3 s | 0.00844 ms/orbit |
| gV2 eval, jax GPU jit | 1e6 (10.9 ms) | **1e7** | 102 ms | |
| gV2 eval, jax GPU jit, vmap over tori | 1e6 (11.3 ms) | **1e7** | 107 ms | |
| gV2 eval, torch GPU | 1e6 (69.8 ms) | **1e7** | 659 ms | |
| gD3 sample, jax GPU jit (eddington NFW / Hernquist / King / OM) | 1e7 | **1e8** | 62.4 / 38.5 / 72.0 / 63.8 ms | |
| gD3 sample, jax GPU eager (same order) | 1e7 | **1e8** | 167 / 118 / 174 / 171 ms | |
| gD3 sample, torch GPU (same order) | 1e7 | **1e8** | 4.93 / 4.91 / 4.90 / 5.56 s | |

**Saturation.** Per-orbit cost is nearly flat from 1e6 to 3e6 (Dopri8 0.0164 →
0.0155, Dopri5 0.0138 → 0.0125, torch 0.0092 → 0.0084 ms/orbit), so the orbit
integrators saturate the H100 at about 1e6 orbits. gV2 and gD3 are already
linear in N from 1e6–1e7 on (10× N gives about 10× wall time). The 80 GB card
had room at every requested N, so none of these series hits a memory limit up
to 3e6 orbits / 1e7 angles / 1e8 samples. I didn't log `nvidia-smi` memory in
this stage (not part of `hiN2`).

**gS4 GPU re-measured (stage `gS4`, job 1055078, 22 min 50 s, galpy 9a877712).**
The earlier GPU series were taken on 70165d10 + #1639, before #1640, which
changed streamdf's track solves. `--measure gpu` re-measured only the GPU series
and merged them; the CPU series are untouched.

| gS4 GPU series | before (70165d10 + #1639) | after (9a877712) |
|---|---|---|
| construct, jax eager, warm (nch 4/8/16/32) | 53.0 / 53.6 / 53.8 / 54.4 s | **104.8 / 107.5 / 105.6 / 106.3 s** |
| construct, jax eager, cold (nch 4/8/16/32) | 109.4 / 110.5 / 111.1 / 112.4 s | **165.2 / 161.6 / 163.9 / 165.9 s** |
| build for panels (b)/(c), cold | 109.5 s | 159.7 s |
| eval / sample, jax jit, N=1e6 | 1.66 / 4.06 ms | 1.65 / 4.06 ms |
| eval / sample, jax eager and torch eager | – | unchanged within about 10% at every N (largest: jax eager sample at 1e5, 36.1 → 33.0 ms) |
| compile (eval, sample) | – | unchanged |

- **Eager jax construction on the GPU got about 2× slower.** The node load was
  the same (loadavg ~8 on 24 cores) and nothing else changed for the
  eval/sample series, so this is the galpy update. The likely cause is #1640:
  `streamdf.py` now solves the tracks with `DiffraxDirectAdjoint`. Not confirmed
  by an A/B against 1bfce100. The CPU construction series on `main` predate
  #1640 too, so they would presumably show the same shift if re-measured.
- **The N=1e6 eager GPU points are still OOM** (`nan`): jax eager sample and
  eval need single 16.3 / 22.4 GiB allocations, and torch eager needs a
  17.5 GB one. `nvidia-smi` (`gS4_mem.csv`, 5 s sampling) peaked at 67.2 of
  81.6 GB. The jit series complete at 1e6.
- The JSON still notes `construct_jit: not measured` (gS4's script). #1640
  makes a jitted construction run; measuring it would need a change to gS4's
  measure code.

## Machine and software

| | |
|---|---|
| GPU | NVIDIA H100 80GB HBM3 (sm_90), 1 of 4 on the node; driver 580.173.02 (CUDA 13.0) |
| CPU | AMD EPYC 9654 96-core; a 1-GPU job gets 24 cores and 187.5 GB host RAM |
| galpy, run 1 | `feat/backends` @ 70165d10 (#1637 merge) + #1639 cherry-picked (`1478b68f`, then still open) |
| galpy, hiN2 | `feat/backends` @ 9a877712 (#1641, #1642 merged; no cherry-picks) |
| galpy, follow-up | `feat/backends` @ 3d9ac516 (#1639, #1640, #1641 merged) + #1642 cherry-picked (`0465ad32`; its second commit, the misalignment atol, was already on the tip and is empty) |
| jax | 0.11.2, jaxlib 0.11.2, jax-cuda12-plugin/pjrt 0.11.2 (nvidia cu12 12.9 wheels) |
| torch | 2.14.1+cu130, triton 3.8.0; torchdiffeq 0.2.5, torchode 1.0.1 |
| diffrax / equinox | 0.7.2 / 0.13.8 |
| numpy / scipy / python | 2.5.3 / 1.18.1 / 3.13.16 (conda-forge env `galpy-h100`, gsl from conda-forge) |
| galax (g8) | main @ 16a7e636b, its own `uv.lock` (jax/jaxlib 0.8.0) + jax-cuda12-plugin 0.8.0 `--no-deps` |

Torch-2.12 triage env `galpy-h100-t212`: a clone of `galpy-h100` with torch 2.12.0+cu126.

TITAN Xp reference (on `main`): jax 0.10.1, torch 2.12.0+cu126, numpy 2.4.3,
same galpy commits.

## Machine-side fixes (no figure code touched)

1. **Alliance pip config leaks into conda.** `PIP_CONFIG_FILE` points at the
   cvmfs wheelhouse with constraints. The first `pip install` mixed wheelhouse
   builds in, and numpy failed on import (`libcpupower.so.0`). Fix: everything
   reinstalled from PyPI with `PIP_CONFIG_FILE=/dev/null` and `PYTHONPATH`
   unset. `conda create` used `--override-channels -c conda-forge`, because the
   `defaults` channels require accepting Anaconda's terms of service.
2. **`~/.triton` is read-only on compute nodes.** Every torch inductor compile
   failed with `PermissionError`. Fixed in `tools/trillium_stage.sh` (commit
   4ee9309 after the rebase: `TRITON_HOME`, `TRITON_CACHE_DIR`, `CUDA_CACHE_PATH` moved to
   scratch). All measurement stages ran after this fix.
3. **galax writes to node-local `/var/tmp`.** Its job script
   (`galax_job.sbatch`, here) copies the JSONs back to scratch.

## Tests (`tests` stage)

Run 1 hit the 2 h limit: jax finished, torch was cut off, and 13 of the 17 jax
failures came from `~/.triton`. Run 2 (6 h limit, Triton fixed) is the
reference (`tests_run2.log`):

| suite | passed | failed | skipped | time |
|---|---|---|---|---|
| `--backend jax --device cuda tests/test_backend_*.py` | 8417 | 2 | 106 | 1:26:33 |
| `--backend torch --device cuda tests/test_backend_*.py` | 8371 | 47 | 107 | 1:29:20 |
| `test_backend_special.py test_backend_actionAngleVerticalInverse.py -k cuda` | 4 | 0 | – | 0:27 |

**jax-suite failures (2):** `test_backend_evolveddiskdf.py::test_moment_grad_through_evolution_vs_fd[meanvT-jax]`
and `[vertexdev-jax]`. The gradient differs from the finite difference by a
relative 5.6e-4, against rtol 2e-4: a tolerance miss, not a wrong gradient.

**torch-suite failures (47), by cause:**
- ~25: `torch.compile` / dynamo, `TorchRuntimeError: RuntimeError when making fake tensor call`.
  Dynamo runs `sub`/`mul` on CUDA FakeTensors and hits `AttributeError: 'ndarray' object ...`,
  so a numpy array reaches the graph. Affected tests:
  `test_backend_potential_jit.py::test_torch_compile_traces_public_entry_point[...]`
  (TwoPowerSpherical, TwoPowerTriaxial, DehnenBar, Einasto, ExpTruncNFW, Ferrers,
  PowerSphericalwCutoff), `test_workflow_compiled_matches_eager[torch(-inductor)-kingdf_W0/qdf_*]`,
  `test_get_namespace_all_torch_fast_path`, `test_kingdf_W0_under_torch_compile_torchode`,
  `test_traced_call_traces_exactly_the_matching_namespace[torch]`.
- 7: `InternalTorchDynamoError: An event was recorded on a stream where a graph input was previously mutated`
  (`test_workflow_compiled_matches_eager[torch(-inductor)-spray_sample/spray_track]`,
  `test_torch_compile_torchode_orbit_matches_eager`).
- 7: `TypeError: can't convert cuda:0 device type tensor to numpy`. These are
  eager float32/device paths: `test_float32_ic_integrates_in_float64[torchode/torchdiffeq]`,
  `test_float32_default_internals_run_in_float64`, `test_float32_default_orbit_and_df_match_float64_default`,
  `test_approxaAInv_float32_query_points`, `test_set_at_casts_values_to_the_destination_dtype`,
  `test_map_coordinates_quintic_and_narrow_grids_vs_scipy[torch]`.
- 1 each: `Can't call numpy() on Tensor that requires grad`
  (`test_workflow_compiled_matches_eager[torch-orbit_c_integrator]`);
  `Cannot re-initialize CUDA in forked subprocess` (`test_parallel_map_survives_torch_work_in_the_child`);
  `torch._dynamo.exc.Unsupported: Observed exception` (`test_forced_backend_survives_a_fullgraph_compile`).
- 3 round-off misses (also in the jax run): `test_finite_part_quad_batches_over_its_limits[jax]`
  (1.8e-15 vs atol 0), `test_misalignment_traces[False/True]` (1.2-1.3e-15 vs atol 1e-15).

Not verified: whether the dynamo failures come from torch 2.14 (the TITAN
environment had 2.12) or from galpy on CUDA. The 7 numpy-conversion failures
look like genuine galpy device bugs on the float32 paths. Full tracebacks are
in `tests_run2.log`. Nothing was worked around.

## Per figure

| figure | stage | status |
|---|---|---|
| g2, g5, g6 | g2g5g6 | measured OK. **All series re-measured**, including numpy CPU, which is now on the EPYC host |
| g8 (galpy-jax side) | orbits | measured OK (`_shared/orbit_speed/data/jax_gpu*.json`) |
| g8 (galax side) | galax | measured OK; the 4 `galax_gpu*.json` replaced (they carry no GPU stamp; all four are from the H100) |
| g12 | orbits | measured OK (jax Dopri8, jax Dopri5, torch CUDA) |
| benchmarks/gpu | bench | measured OK, `benchmarks/gpu/results_h100_2026-10-06/` |
| gV2, gD3, gS5, gS4 | newfigs | measured OK (GPU series only; CPU series untouched). gS4: eager jax/torch GPU at N=1e6 OOM → `nan` |
| g15 | g15 | measured, partial at the top end: `measure_largeN.py` was **killed by the host OOM killer** (187.5 GB job limit) at N=1048576, after the GPU series had stopped on GPU OOM. Every point up to N=262144 was saved incrementally; only torchdiffeq_cpu at N=1048576 is lost. CPU series are re-measured, now on the EPYC host |
| g17 | g17 | measured; large-N GPU points `nan` (see "Surprises"). CPU series re-measured, now on the EPYC host |

**For folding in:** the CPU series of g2/g5/g6/g15/g17 on this branch are from
a Trillium compute node, not the TITAN box. In g17 they are much slower (see
below), so take only the GPU series from this branch.

## Headline GPU numbers: H100 vs TITAN Xp (`main`)

| quantity | TITAN Xp | H100 | speed-up |
|---|---|---|---|
| g2 forces, jax GPU, N=1e7 | 5.52 ms | 0.146 ms | 38x |
| g2 forces, torch GPU, N=1e7 | 7.30 ms | 0.587 ms | 12x |
| g2 forces, jax GPU incl. transfer, N=1e7 | 158 ms | 24.6 ms | 6.4x |
| g5 jax grad, N=1e7 | 1.88 ms | 0.145 ms | 13x |
| g5 torch grad, N=1e7 | 7.77 ms | 1.22 ms | 6.4x |
| g6 jit per call, N=1e6 | 0.899 ms | 0.0736 ms | 12x |
| g12 jax Dopri8 GPU, ms/orbit, N=3e5 | 0.212 | 0.0247 | 8.6x |
| g12 jax Dopri5 GPU, ms/orbit, N=3e5 | 0.126 | 0.0182 | 6.9x |
| g12 torch dopri5 CUDA, ms/orbit, N=1e6 | 0.0673 | 0.00922 | 7.3x |
| g8 galax GPU (16 saves), ms/orbit, N=3e5 | 0.0681 | 0.0130 | 5.2x |
| g8 galax GPU (1000 saves), ms/orbit, N=3e4 | 0.0916 | 0.0494 | 1.9x (H100 now reaches N=1e5: 0.0181) |
| g15 diffrax GPU grad, N=16384 | 7.84 s | 5.28 s | 1.5x |
| g15 torchdiffeq GPU grad, N=4096 | 10.8 s | 3.64 s | 3.0x |
| g17 jax GPU jit, N=1e5 (run 1; run 2: 0.161 s) | 4.45 s | 0.325 s | 14x |
| g17 jax GPU eager, N=3e5 | 17.9 s | 13.3 s | 1.3x |
| gS5 jax GPU jit, ms/particle, N=1e6 | – (no TITAN series on `main`) | 0.00719 | |
| gV2 jax GPU jit eval, N=1e6 (run 1; after #1641: 10.9 ms) | – (no TITAN series on `main`) | 30.0 ms (torch GPU 99.9 ms; numpy CPU on the TITAN box, untouched: 16.5 s) | |
| gD3 eddingtondf (NFW) sample, jax GPU jit, N=1e7 | – (no TITAN series on `main`) | 6.24 ms (torch GPU 0.50 s) | |

The GPU series of gV2, gD3, gS4 and gS5 were never measured on the TITAN
(gS4's caption still says "[to be added]"), so these H100 values are the only
GPU series for those figures.

`benchmarks/gpu` (largest N per case, seconds):

| backend | case | N | TITAN Xp | H100 | speed-up |
|---|---|---|---|---|---|
| jax | orbits | 1e5 | 11.25 | 0.694 | 16x |
| jax | forces | 1e7 | 0.485 | 0.0136 | 36x |
| jax | forcegrad | 1e7 | 0.494 | 0.0139 | 36x |
| jax | sample | 1e6 | 0.145 | 0.0921 | 1.6x |
| torch | orbits | 1e5 | 29.3 | 10.3 | 2.9x |
| torch | forces | 1e7 | 0.816 | 0.0725 | 11x |
| torch | forcegrad | 1e7 | 0.818 | 0.0742 | 11x |
| torch | sample | 1e6 | 0.119 | 0.119 | 1.0x |

As expected, the large-N, float64-throughput-bound points gain 10-40x, while
the launch- and dispatch-bound ones (small N, sampling, eager torch, g15's
per-call floor) change little.

## Surprises

1. **Some GPU timings are slower on the H100 than on the TITAN Xp.**
   - g15 single-orbit gradients: diffrax 2.13 → 3.23 s, torchdiffeq 13.4 → 18.4 s.
   - g15 diffrax-GPU scaling at small N: about 2.0 → 2.5 s.
   - g17 torch GPU, run alone (diagnostic below): 1e5 2.00 s, 3e5 5.00 s,
     1e6 15.6 s, against the TITAN's 1.33, 2.57, 8.36 s.

   These points are dispatch- or launch-bound, so this is plausibly the newer
   jax/torch versions (0.11.2 / 2.14.1 vs 0.10.1 / 2.12.0). Not isolated.
   *Superseded for g17 by the follow-up run:* after #1642, torch alone takes
   0.778 / 1.37 / 3.50 s at 1e5 / 3e5 / 1e6, faster than the TITAN. The g15
   points still stand.
2. **g17 GPU OOMs on 80 GB where the 12 GB TITAN did not.** Recorded as `nan`:
   torch GPU at N ≥ 1e5 and jax GPU eager at N = 1e6. Diagnostic
   (`diag/g17_gpu_oom.py`, output in `diag/g17_gpu_oom.out`; not figure code):
   - torch alone completes 1e5, 3e5 and 1e6, peaking at 6.5 / 19.5 / **65 GB**.
   - torch at 1e5 still runs after the jax jit 1e5 and eager 3e5 series, with
     jax holding a 32 GB pool.
   - jax eager at 1e6 fails even alone: a 10.45 GiB allocation fails with
     46.6 GB in use.

   So in the g17 run, jax's BFC pool (never returned to the device) grew during
   the failing jax 1e6 and +transfer points, and starved torch from N=1e5 on.
   The underlying question is open: these paths use far more memory under the
   H100's versions than the TITAN's 12 GB could have held. g17's `nan`s from
   1e5 up are an artefact of running both backends in one process, not of the
   card.
   *Resolved in the follow-up run:* with #1642, jax eager 1e6 fits in-process
   (12.5 s), and the torch series measured alone (`g17torch`) is complete to
   1e6, with a 43.9 GB peak instead of 65 GB.
3. **gS4 eager GPU at N=1e6 OOMs** (single 16.3 / 22.4 GiB allocations,
   `nan`). The jit series completed.
4. **g15 GPU series now reach larger N.** torchdiffeq GPU runs to 131072 and
   OOMs at 262144; diffrax GPU runs to 262144 and OOMs at 1048576. The host
   OOM kill described above came next.
5. **CPU series on the Trillium node are much slower than the TITAN box's** in
   g17: numpy 21 → 424 s at N=1e5, torch CPU 11 → 89 s, jax CPU 60 → 179 s at
   3e4, all at full contention with 24 cores. In g15 they are mostly faster (e.g. cstm_cpu about 8x). Don't fold
   these in as CPU references.
6. Run 1 of the test suite took 1:24 for jax alone, so `tests` needs about 3 h
   (the 2 h estimate was too short).
