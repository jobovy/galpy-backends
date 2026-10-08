# CUDA one-thread-per-orbit port of galpy's `dop853_c` (MWPotential2014)

This is an experiment. It asks how fast galpy's orbit integration gets on a GPU when the C potentials and the
integrator are compiled as CUDA device code, with one thread integrating one orbit from start to finish, instead
of jax/diffrax's batched array execution. galpy itself is not modified.

## What was ported (≈1 260 lines)

| file | lines | content |
|---|---|---|
| `galpy_cuda.cuh` | 326 | `PowerSphericalPotentialwCutoff`, `MiyamotoNagaiPotential`, `NFWPotential` (forces, verbatim arithmetic); a line-by-line port of the GSL `gamma_inc_P` branches galpy reaches (series, asymptotic, CF); `calcRforce/zforce` aggregation; `evalRectDeriv`; cyl↔rect. Options: `CO_FASTGAM`, `CO_FASTPOW` (see below). |
| `dop853_cuda.cuh` + `dop853_coeffs.inc` | 342 + 159 | galpy's `util/leung_dop853.c`: same initial-step estimate, error norm, step control, and dense output at the requested times. It is split into `dop853_init` / `dop853_step` so that two kernels share it. |
| `cuda_orbits.cu` | 276 | kernels (`k_integrate`: one orbit per thread; `k_integrate_refill`: persistent threads), host path (the same code on the CPU), pinned host buffers, C ABI |
| `cuda_orbits.py` | 157 | ctypes wrapper (numpy only) |

The potential is a flat `PotSet` in `__constant__` memory: a type code and an offset into one args table per
component, with a `switch`. There are no function pointers and no heap. The host passes the per-call constants
(`2π rc^(3-α)`, `Γ(a)`, `lnΓ(a+1)`, and for `CO_FASTGAM` a table of `1/(a+n)`) as the same doubles galpy computes.

## Correctness

The tests use 300 orbits drawn like `galpy_c_measure.py` (hiN: nt=16 over 2 Gyr, rtol=atol=1e-8). They are compared
with galpy's `Orbit.integrate(..., method='dop853_c')` (`ref_N300.npz`) and with galpy at 1e-13 as the truth
(`ref_N300_tol1e-13.npz`). Run with `python check_correctness.py ref_N300.npz [--gpu] [--variant ...]`.

- The **host build of the port is bit-identical to galpy C**: 100% of orbits match at all 16 saves, with the same
  step counts. The GSL incomplete-gamma port is bit-identical to libgsl on the host (`check_gammainc.py`).
- **GPU**: the TITAN Xp and the H100 give identical numbers. The H100 v1 run checked default and `_nofma`, and v2
  checked default, `_noinl` and `_fast_noinl`, each with the v1, refill and sorted schedules:

  | build | max \|dx\|/\|x\| at t_end vs galpy C | median | orbits with galpy's exact accept/reject sequence |
  |---|---|---|---|
  | default (device FMA) | 1.55e-6 | 3.7e-13 | 80.3% (max 3.6e-7 among those) |
  | `_nofma` | 4.85e-6 | 3.4e-13 | 80.3% |
  | `_noinl` (out-of-line force) | 1.55e-6 | 3.7e-13 | 80.3% |
  | `_fast`, `_fast_noinl` (`CO_FASTGAM`+`CO_FASTPOW`) | 5.3e-6 | 3.9e-13 | 78.0% |

  The rest of the difference comes from libdevice `pow`/`log`/`exp` and FMA rounding being ~1 ulp off glibc. That
  flips an accept/reject decision on ~20% of orbits, and those orbits then differ at the 1e-6 level. All of it lies
  within galpy's own truncation error: galpy at 1e-8 vs galpy at 1e-13 differs by up to **1.4e-5** (median 8e-7).
  Against the 1e-13 truth, the GPU and galpy C have the same error distribution (max 1.49e-5 vs 1.40e-5).
- Refill and sorted schedules give bit-identical results to the one-orbit-per-thread kernel.
- The fast incomplete gamma agrees with GSL to ≤1.7e-15 (`check_gammainc.py --variant _fastgam`).

## Results at a glance

MWPotential2014, 2 Gyr, rtol=atol=1e-8, nt=16 saves (the paper's hiN convention), ICs from `galpy_c_measure.py`.
Each cell is wall time per orbit. For the CUDA kernels that is H2D + kernel + D2H into a reused pinned buffer (+ host
argsort for "sort"). One-time compilation and the pinned allocation are excluded. All rows are measured; nothing is
projected.

| code (hardware) | N=1e4 | N=1e5 | N=1e6 | N=1e7 | source |
|---|---|---|---|---|---|
| galpy C `dop853_c`, 1 thread (Xeon E5-2697 v4) | 2.16 ms | 2.26 ms | 2.34 ms | – | `galpy_c_cpu_1cores_hiN.json` |
| galpy C, 16 OpenMP threads (same box) | 175 µs | 172 µs | 178 µs | – | `galpy_c_cpu_16cores_hiN.json` |
| galpy C, 72 OpenMP threads (full 36-core box) | 100 µs | 81 µs | 79 µs | – | `galpy_c_cpu_hiN.json`* |
| jax/diffrax Dopri8, TITAN Xp | 274 µs | 246 µs | (3e5: 215 µs) | – | `jax_gpu_hiN.json` |
| galax Dopri8, TITAN Xp | 129 µs | 80 µs | (3e5: 68 µs) | – | `galax_gpu_hiN.json` |
| **CUDA v1** (one orbit per thread, galpy arithmetic), TITAN Xp | 106 µs | 72 µs | 70 µs | – | `data/titan/scan_v1.json` |
| **CUDA `_noinl`+refill** (galpy arithmetic), TITAN Xp | 82 µs | 51 µs | 49 µs | – | `data/titan/scan_noinl_refill_pinned.json` |
| **CUDA `_fast_noinl`+refill+sort**, TITAN Xp | 21 µs | 16.0 µs | 15.6 µs | (3e6: 15.6 µs) | `data/titan/scan_fast_noinl_refill_sort_pinned.json` |
| jax/diffrax Dopri8, H100 | 186 µs | 34 µs | 16.4 µs | (3e6: 15.5 µs) | `h100/jax_gpu_hiN.json` |
| jax/diffrax Dopri5, H100 | 159 µs | 25 µs | 13.8 µs | (3e6: 12.5 µs) | `g12…/h100/jax_dopri5_gpu_hiN.json` |
| torch/torchdiffeq dopri5, H100 (shared controller)† | 400 µs | 41 µs | 9.2 µs | (3e6: 8.4 µs) | `g12…/h100/torch_cuda.json` |
| galax Dopri8, H100 | 132 µs | 17.5 µs | (3e5: 13.0 µs) | – | `g8…/h100/galax_gpu_hiN.json` |
| **CUDA v1**, H100 (pinned) | – | 6.7 µs | 7.6 µs | 8.3 µs | `data/h100/v2/scan_v1_pinned.json` |
| **CUDA `_noinl`+refill** (galpy arithmetic), H100 | 12.0 µs | 2.9 µs | **2.36 µs** | **2.30 µs** | `data/h100/v2/scan_noinl_refill_pinned.json` |
| **CUDA `_fast_noinl`+refill+sort**, H100 | 5.3 µs | 1.11 µs | **1.04 µs** | **1.10 µs** | `data/h100/v2/scan_fast_noinl_refill_sort_pinned.json` |

\* The 72-thread galpy C series was labelled "single-thread" in the JSON. galpy's C orbit loop uses
`omp_get_max_threads()` and ignores `numcores`.
† torch dopri5 advances all orbits with one shared adaptive step controller. It does different work from the
per-orbit-adaptive Dopri8 rows: a 5th-order method, with the step set by the worst orbit.

At N=1e6 on the H100, the galpy-arithmetic CUDA kernel (`_noinl`+refill) is 7× faster than jax Dopri8 on the same
GPU, 4× faster than torch dopri5, 34× faster than galpy C on the full 72-thread box, and ~1 000× faster than one
core. The fast-arithmetic kernel is about twice as fast again: 16× jax Dopri8, 9× torch, 76× the 72-thread box,
2 250× one core. Both agree with galpy C to within galpy's own truncation error (Correctness, above). Below N≈1e5 the
H100 is not full: 132 SMs × 256 resident threads is 34 k orbits at a time, and a launch costs as long as its slowest
orbit. Per-orbit cost there is latency.

The same headers compiled for the host (`co_integrate_host`, one thread, Xeon E5-2697 v4) take 1.5-2.2 ms/orbit
with galpy's arithmetic. That matches galpy C, which is the same arithmetic plus function-pointer dispatch. With
the fast arithmetic the host build takes 0.96 ms/orbit, so the arithmetic change alone is worth 2.3× on a CPU too.

## Diagnosis

### v1 → v2: what each fix bought (kernel µs/orbit, faithful dense output)

The rows are controlled one-variable experiments (`run_diag.sh`): TITAN at N=3e4 (`data/titan/diag/`), H100 at
N=1e6 (`data/h100/v2/diag/`), and Nsight Compute on the H100 at N=2e5 (`data/h100/v2/ncu_*.txt`). nvprof fails on
the TITAN (driver 580 + Pascal, "Internal profiling error 4211:27"), and ncu does not support sm_61. The TITAN
evidence is therefore operation counts (`flopcount.cpp`), SASS (`cuobjdump`) and the controlled runs.

| configuration | TITAN Xp | H100 | what it isolates |
|---|---|---|---|
| v1: one orbit per thread, galpy arithmetic, force inlined 17× (718 kB of SASS) | 73.2 | 7.58 | baseline |
| v1, lazy dense output | 66.2 | 10.90 | lazy is *slower* on the H100 |
| v1 in lockstep (fixed 250 steps, 3752 nfev) | 51.5 | 2.38 | no divergence: warps stay at the same code |
| v1 + cost-sorted schedule | 57.2 | 4.40 | |
| v1 + refill | 59.3 | 8.48 | |
| v1 + refill + sort | 52.7 | 7.97 | |
| one out-of-line force (`_noinl`, 107 kB) | 71.5 | **3.27** | instruction cache |
| `_noinl` + refill | 58.0 | **2.33** | |
| 128-register cap + refill (`_r128`) | 62.1 | 8.91 | occupancy vs spills |
| division-free Γ (`_fastgam`) | 38.4 | 2.18 | |
| `sqrt` instead of `pow` (`_fastpow`) | 58.9 | 4.85 | |
| `_fast` (both, force still inlined) | 23.5 | 1.47 | |
| `_fast_noinl` | 22.8 | 1.30 | |
| `_fast_noinl` + refill | 18.4 | 0.90 | |
| `_fast_noinl` + refill + sort | **16.4** | **0.88** | |
| `_fast_noinl` in lockstep (3752 nfev → 3466 scaled) | 15.7 → 14.5 | 0.89 → 0.82 | the no-divergence bound |
| float32 vs float64, tol 1e-5, `_noinl`+refill | 1.87 vs 27.6 (15×) | 0.27 vs 1.14 (4.2×) | the FP64 penalty (hardware: 32× / 2×) |

### The four inefficiencies, in order of size

1. **Instruction cache (H100: 2.3×; TITAN: none).** With the force inlined at all 17 call sites the kernel is 45 k
   SASS instructions (718 kB). Nsight Compute for v1 shows each warp waiting **8.7 of 15.1 cycles per issued
   instruction** to fetch an instruction or on an instruction-cache miss ("No Instruction" stall), and 18.0 of 24.3
   cycles in lazy mode. One out-of-line force cuts the kernel to 107 kB and v1 from 7.58 to 3.27 µs. The same
   mechanism explains the anomalies:
   - **lazy slower than faithful in v1**: lazy's divergent dense-output branch puts the warps of an SM at more
     distinct code addresses. With `_noinl`, lazy is faster again (2.96 vs 3.27; 0.85 vs 0.90 for fast+refill).
   - **lockstep 3× faster than adaptive on the H100**, against 1.4× on the TITAN: lockstep warps share i-cache lines.
   - **"sort" helping v1 (7.58 → 4.40) while "refill" hurts it (8.48)**: sorting makes warps more coherent, while
     refill's flattened loop scatters them across the code.

   I did not predict this from the TITAN, where `_noinl` is neutral (71.5 vs 73.2). Pascal has the same code but
   ~10× lower FP64 throughput, so instruction fetch is never the bottleneck there.
2. **Arithmetic form: galpy's incomplete gamma and `pow` (TITAN 3.1×, H100 2.5×).** Per force evaluation, galpy's
   arithmetic is 141 add, 96 mul, **121 div**, 8 pow, 3 log, 1 exp and 1 sqrt (`flopcount`). The divisions come from
   GSL's series for P(a, r²/rc²) (~60 terms, `x/(a+n)` plus a `|term/sum|` test). A DP division is
   MUFU.RCP64H + 7 dependent DFMA, and `pow` is ~85 DP instructions: ≈7.6 M DP instructions per orbit for 1.7 M
   algorithmic operations. `CO_FASTGAM` (table of `1/(a+n)`, `|term| < eps·|sum|`, `·1/x`; ≤1.7e-15 vs GSL) plus
   `CO_FASTPOW` (`sqrt` forms, fused R/z forces) bring it to 2.4 M. The TITAN time falls in proportion (73.2 → 23.5),
   which is what an FP64-throughput-bound kernel should do.
3. **Warp imbalance and divergence (1.3-1.5×).** nfev per orbit ranges from 1 283 to 8 109. In v1 a warp runs until
   its slowest lane finishes: lane utilisation Σnfev / Σ(32·max per warp) = **0.69** (`warp_lane_eff_static` in the
   JSONs). The persistent-thread **refill** kernel (one flat step loop; a finished lane takes the next orbit from an
   atomic counter) removes this. Combined with a small kernel, it brings both GPUs within 7-13% of the lockstep
   bound. Sorting adds only 2% on top of refill, and at N=1e7 its *host* `np.argsort` costs **≈2 s** (2.2 s for
   1e7 keys on one core here). That is the "other" time in the 1e7 fast scan (1.96 s, vs 6 ms without sort). Drop
   the sort with refill, or sort on the device (cub radix sort, milliseconds).
4. **Latency at 8 warps per SM (the remaining gap on the H100).** For the best kernel (`_fast_noinl`+refill, 2e5
   orbits), ncu reports 255 registers per thread, so theoretical occupancy is 12.5% (achieved 12.2%, 1.98 warps per
   scheduler). On average 0.30 warps are eligible per cycle, and **72.5% of cycles issue nothing**. The top stall is
   waiting on a fixed-latency dependency, 3.9 of 7.2 cycles per instruction: dependent DFMA chains in divisions,
   series recurrences and RK stage sums. Issue slots are 27% busy. Of the issued instructions only 33% are FP64
   (18.1 G of 55.1 G warp instructions); the rest are loop control, constant-table and local (spill) loads, and
   branches. On average 21.2 of 32 lanes are active, because of accept/reject, dense output and the x<20 / x≥20 Γ
   branches with variable trip counts. ncu puts the result at **8% of the H100's FP64 peak**. By my instruction count
   (1.9-2.4 M DP instructions per orbit at 0.88 µs) it is 13-16%. On the TITAN the same kernel issues DP
   instructions at **76% of its FP64 peak** (`fp64_efficiency.py`). The TITAN is throughput-bound, the H100
   latency-bound, which is why TITAN→H100 is 18× rather than the 88× peak ratio.

   The capped-register build confirms that occupancy alone is no lever: at 128 registers the spills reach
   3.7 kB/thread and it gets slower (8.9 µs). Closing the remaining ~6-8× to the FP64 roofline (0.11-0.14 µs/orbit)
   needs less live state per thread. The 12 stage vectors are 66 doubles, which could live in shared memory, with
   the dense-output coefficients computed on demand. It also needs more independent work per thread (two
   interleaved orbits for ILP), and a branch-free Γ: fixed-length Chebyshev or a minimax fit in r for a given
   (α, rc), evaluated in a fixed order.

## Fixes implemented (all optional; all verified on both GPUs)

- `--pinned`: page-locked, reused result buffer (`co_host_alloc`). On the H100 the D2H copy of 1e7 × 16 × 6
  doubles takes a steady 272 ms (28 GB/s). Before, v1's fresh `np.empty` page-faulted on first touch and
  took 1-100 s.
- `--refill`: persistent-thread kernel with atomic orbit refill (`k_integrate_refill`).
- `--sort`: most-expensive-first schedule by a vertical-energy proxy `vz²+9z²` (correlation 0.76 with nfev). This
  is problem-specific and only worth it without refill.
- `_noinl`: one out-of-line force (kernel 718 → 107 kB).
- `_fastgam`, `_fastpow`, `_fast`, `_fast_noinl`: division- and pow-free force arithmetic. This is not
  bit-identical to galpy, but agrees with GSL to ≤1.7e-15 in Γ and with galpy C within the GPU's ulp-level spread
  on orbits.

## What a full implementation would entail

Effort estimate: **6-10 weeks** for all conservative potentials plus dop853/RK/symplectic integrators, with
correctness tests against the C. Dissipative forces, `NonInertialFrameForce` and the variational equations add
**3-4 weeks**.

- **~40 potentials (12.7 k lines of C in `potential_c_ext/`).** Most are short pure functions, and the arithmetic
  ports verbatim: here only `pow(x,0.5)` and friends needed attention, and only for speed. What helps: every
  potential's state is already a flat `double *args` plus an integer type code parsed in one place
  (`parse_leapFuncArgs_Full`). That maps directly onto the `PotSet` constant table, and a `switch` replaces the
  function pointers, which CUDA could also use but which block inlining and constant propagation.
- **What hurts:**
  - **Function-pointer dispatch per potential and per component** (`Rforce`, `zforce`, `phitorque`, … set in a
    1 000-line parser): every potential needs a switch entry, or one device function-pointer table built on the
    device.
  - **Caching in `args`.** `PowerSphericalPotentialwCutoff`, `SoftenedNeedleBar`, `Ellipsoidal`, `Spiral`,
    `RotateAndTilt` and others write cache values into `args`. On the GPU that state must become per-thread
    (registers), because `args` is shared and read-only.
  - **Heap and `malloc` at evaluation time.** `SCFPotential` and `MultipoleExpansionPotential` `malloc` work arrays
    per call (6 and 35 `malloc` sites). These must become fixed-size per-thread scratch, or the coefficient
    contraction must be restructured.
  - **Large coefficient tables** (SCF `Acos/Asin`, multipole radial splines, `interpRZPotential` 2D splines,
    `DiskSCF`). They are too big for 64 kB of constant memory and belong in global memory through the read-only
    cache. Their access is data-dependent per orbit, so it becomes memory-latency-bound. Expect these to be the
    slow potentials.
  - **GSL.** The incomplete gamma (3 potentials), `gsl_sf_gamma` (11 sites, but constants), Legendre and
    Gegenbauer polynomials (SCF, multipole), `expint`, `lnbeta`, and **cubic splines** (`gsl_spline`, 32 sites:
    interp potentials, MovingObject, Chandrasekhar dynamical friction, NonInertialFrame) all need device versions.
    The splines are the bulk: a small device cubic-spline evaluator plus host-side coefficient precomputation with
    GSL's exact boundary conditions, if bit-for-bit agreement with C matters.
  - **Wrappers** (12, e.g. `DehnenSmooth`, `SolidBodyRotation`, `RotateAndTilt`, `KuzminLike`) recurse into
    `wrappedPotentialArg` arrays. They flatten into the `PotSet` as sub-ranges, which needs depth-limited
    recursion or an explicit stack on the device.
  - **`tfuncs`** (`TimeDependentAmplitudeWrapper`, `NonInertialFrameForce`) are *Python callables* (numba cfunc
    or ctypes). They cannot run on the device. Options: tabulate them on a time grid plus a spline (an
    approximation), or JIT them with numba.cuda.
  - **Dissipative forces** (Chandrasekhar, FDM) need velocities, which is fine, and splines of σ(r), which is a
    table.
  - **Variational equations** (`evalRectDeriv_dxdv` / STM): every potential's Hessian (`R2deriv`, `z2deriv`,
    `Rzderiv`, …) must be ported as well, roughly doubling the per-potential work. The state grows to 42
    (6 + 36) doubles, which with dop853's 12 stages no longer fits in registers.
- **Performance engineering is not optional.** A verbatim port (v1) was only 2× faster than jax on the H100. Three
  structural choices bought 3.3× more, all applicable to every potential:
  - one out-of-line force, so the force code is not inlined at all 17 RK call sites;
  - the persistent-thread refill loop;
  - pinned output buffers.

  Replacing GSL-style divisions and `pow` bought another 2.6×, but that has to be done per potential. Expect to keep
  two arithmetic paths: galpy-exact for validation and a GPU-tuned path for speed. With ~40 potentials in one
  `switch`, the force function grows again, so keep it out of line and benchmark the instruction-cache footprint.
- **No gradients.** These kernels are not differentiable. Sensitivities need either a hand-written variational /
  adjoint kernel (dxdv: the per-potential Hessians above) or a framework-level custom-call that supplies one. The
  jax/torch paths keep their place for anything that needs gradients.
- **Integration with galpy:** `Orbit.integrate(..., method='dop853_cuda')` would reuse `_parse_pot` unchanged and
  call this library through ctypes, as the C path does. Packaging needs an nvcc build (or a PTX/fatbin shipped
  for sm_70-sm_90) and a CUDA runtime dependency.

## v1 → v2 on the H100 (measured, N=1e6, kernel)

The v1 README projected ≈2 µs/orbit from the TITAN diagnosis. Measured: 2.33 µs with galpy's arithmetic
(`_noinl`+refill) and 0.88 µs with the fast arithmetic. The projection got there for the wrong reason: it credited
refill (1.4×) and `_fast` (≥3×). On the H100, refill without `_noinl` is *slower*, and the biggest single factor is
the instruction cache (`_noinl`, 2.3×), which the TITAN cannot show.

## Running

```
./build.sh                      # all variants, sm_61 + sm_90 (ARCHS="90" for H100 only)
python check_correctness.py ref_N300.npz --gpu [--variant _fast_noinl]
./run_diag.sh <outdir> <N>      # the controlled-experiment matrix (GATE=./gpu_idle.sh on the shared box)
python bench.py --Ns 1e3,...,1e7 --variant _fast_noinl --refill --sort --pinned --modes faithful --out ...
python fp64_efficiency.py data/titan/diag/*.json
```

**H100 (Trillium):** `module load StdEnv/2023 gcc/12.3 cuda/12.9`. A bare `module load cuda` loads nothing, and
cuda/13.x is too new for driver 580. Then run `./run_h100.sh` (v1) or `./run_h100_v2.sh`. v2 takes ≈1 h and writes
`data/h100/v2/`: correctness, the diagnosis matrix at N=1e6, N scans with pinned buffers for the galpy-exact and
fast kernels, and Nsight Compute reports if `ncu` is available. The runs need nvcc ≥ 11.8, gcc, and python3 with
numpy. No galpy, GSL or jax is needed: `mw2014_params.json` and the reference orbits ship here.

`make_params.py` / `make_reference.py` regenerate those inputs from galpy (`PYTHONPATH=<galpy>`).
`gpu_idle.sh` is the shared dev box's GPU gate. `run_titan.sh` runs the TITAN N scans.
