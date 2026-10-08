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
- **GPU** (TITAN Xp; the H100 v1 run reproduced the default and `_nofma` rows exactly):

  | build | max \|dx\|/\|x\| at t_end vs galpy C | median | orbits with galpy's exact accept/reject sequence |
  |---|---|---|---|
  | default (device FMA) | 1.55e-6 | 3.7e-13 | 80.3% (max 3.6e-7 among those) |
  | `_nofma` | 4.85e-6 | 3.4e-13 | 80.3% |
  | `_fast` (`CO_FASTGAM`+`CO_FASTPOW`) | 5.3e-6 | 3.9e-13 | 78.0% |

  The rest of the difference comes from libdevice `pow`/`log`/`exp` and FMA rounding being ~1 ulp off glibc. That
  flips an accept/reject decision on ~20% of orbits, and those orbits then differ at the 1e-6 level. All of it lies
  within galpy's own truncation error: galpy at 1e-8 vs galpy at 1e-13 differs by up to **1.4e-5** (median 8e-7).
  Against the 1e-13 truth, the GPU and galpy C have the same error distribution (max 1.49e-5 vs 1.40e-5).
- Refill and sorted schedules give bit-identical results to the one-orbit-per-thread kernel.
- The fast incomplete gamma agrees with GSL to ≤1.7e-15 (`check_gammainc.py --variant _fastgam`).

## Diagnosis: why v1 reached only ~1.5-2× jax on the H100

The v1 H100 run (`data/h100/cuda_gpu_hiN*.json`) measured 6.7-8.4 µs/orbit (faithful) and 10.8-11 µs/orbit (lazy),
kernel only. For comparison, jax Dopri8 is 16.4 µs and torch dopri5 is 9.2 µs per orbit at 1e6.

Profiler status: on the TITAN, `nvprof --metrics` fails ("Internal profiling error 4211:27", driver 580 + Pascal),
and Nsight Compute does not support sm_61. The diagnosis therefore uses (a) exact operation counts (`flopcount.cpp`:
the same headers on the host with a counting `double`), (b) SASS inspection (`cuobjdump`, `sass_ops.cu`), and (c)
controlled one-variable experiments (`run_diag.sh`, TITAN results in `data/titan/diag/`, N=3e4).

Kernel µs/orbit on the TITAN Xp (galpy-faithful dense output; lazy is ~10% faster on Pascal):

| configuration | µs/orbit | what it isolates |
|---|---|---|
| v1 (one orbit per thread, exact galpy arithmetic) | **73.2** | baseline |
| v1 with fixed steps, all lanes in lockstep (250 steps, 3752 nfev) | 51.5 → 47.6 at equal nfev | no divergence or imbalance: the bound for this arithmetic |
| + cost-sorted schedule | 57.2 | warp imbalance |
| + persistent-thread refill | 59.3 | warp imbalance |
| + refill + sort | 52.7 | imbalance removed (within 10% of lockstep) |
| one out-of-line force (`_noinl`; kernel 718 kB → 107 kB of SASS) | 71.5 | instruction cache: no effect on Pascal |
| 128-register cap (`_r128`, 2× resident warps, 3.7 kB spills) | 62.1 (vs refill 59.3) | occupancy: no gain |
| division-free incomplete gamma (`_fastgam`) | 38.4 | the gamma series' divisions are ~half the time |
| `sqrt` instead of `pow` (`_fastpow`) | 58.9 | 8 libdevice `pow` per force evaluation |
| `_fast` (both) | 23.5 | |
| `_fast_noinl` + refill + sort | **16.4** | everything (4.5× faster than v1) |
| `_fast_noinl` in lockstep | 15.7 at 3752 nfev | the bound |
| float64 vs float32 at tol 1e-5 (refill, `_noinl`) | 27.6 vs 1.9 | the FP64 penalty: 15× (hardware ratio 32×) |

Findings, with evidence:

1. **The arithmetic is dominated by libdevice division and `pow`, almost all of it in the incomplete gamma.** Per
   force evaluation, galpy's form costs 141 add, 96 mul, **121 div**, 8 pow, 3 log, 1 exp and 1 sqrt (`flopcount`).
   The 121 divisions come from GSL's series (`term *= x/(a+n)` plus a `|term/sum|` test, ~60 terms for x = r²/rc² ≲ 20).
   On the GPU a DP division is a MUFU.RCP64H followed by 7 dependent DFMAs (plus a slow-path branch), and `pow` is
   ~85 DP instructions. That is ≈7.6 M DP-pipe instructions per orbit, against 1.7 M algorithmic operations.
   `CO_FASTGAM` uses a precomputed `1/(a+n)` table, compares `|term|` with `eps·|sum|`, and multiplies by `1/x`;
   it agrees with GSL to ≤1.7e-15. `CO_FASTPOW` uses `sqrt` forms and fuses the R/z forces. Together they cut the
   count to 2.4 M DP instructions per orbit, and the TITAN time drops 3.1×, in proportion.
2. **Warp imbalance and divergence: 1.4-1.5×.** nfev per orbit ranges from 1 283 to 6 852. A warp runs until its
   slowest lane finishes, so lane utilisation is Σnfev / Σ(32·max per warp) = **0.69** (`warp_lane_eff_static` in
   the JSONs). The fixed-step lockstep runs confirm the cost: per force evaluation, v1 is 1.54× slower than
   lockstep. The fix is persistent-thread **refill**: one flat step loop, and a lane whose orbit finishes takes the
   next one from an atomic counter (`k_integrate_refill`). Scheduling the most expensive orbits first, by
   `vz²+9z²` (correlation 0.76 with nfev), helps further. The two together come within 10% of lockstep.
3. **Faithful faster than lazy on the H100.** "Lazy" skips the 3 dense-output evaluations on accepted steps without a
   requested time. In SIMT, though, a warp pays for them whenever *any* lane needs them. With ~15 of ~188 accepted
   steps per lane needing dense output, P(no lane of 32 needs it) ≈ 0.92³² ≈ 7%, so lazy saves almost nothing. Its
   only gain is the 15% lower nfev in the counters, while it adds a divergent region around the largest code block.
   On the TITAN it is still ~10% faster. On the H100 it is 30-40% slower. The likely reason is the
   instruction-cache/reconvergence behaviour of a 718 kB kernel under Volta+ independent thread scheduling: the
   force is inlined 17 times, so the kernel is 45 k SASS instructions. `_noinl` (one out-of-line force, 107 kB)
   tests this on the H100 in v2. It is untestable on Pascal, where `_noinl` is neutral.
4. **On the TITAN the kernel is FP64-throughput-bound. On the H100 it is latency-bound.** With the DP instruction
   counts above, v1 issues 1.04e11 DP instr/s on the TITAN, **55% of its FP64 peak** (30 SM × 4 FP64/clk ×
   1.58 GHz = 1.9e11). Refill + sort raises that to **76%** for both the exact and the fast arithmetic. On the H100
   the same v1 kernel issues 0.9-1.1e12 DP instr/s, **5-7% of FP64 peak** (1.67e13), and 3.5% in lazy mode. That is
   why the TITAN→H100 speedup is ~10× rather than the 88× peak ratio. The kernel uses 255 registers and 100-170 B of
   spills, so only 8 warps are resident per SM, 2 per scheduler. The work consists of long *dependent* DFMA chains
   (Newton steps of every division, the series recurrence, `pow`). With 2 warps per scheduler and a multi-cycle DFMA
   latency, a scheduler cannot keep its 16 FP64 lanes busy. Capping registers at 128 doubles the warps but spills
   3.7 kB per thread, which made things worse on the H100 (11.9 vs 10.8 µs) and on the TITAN. Removing the
   divisions shortens the dependency chains, so `_fast` should help the H100 *more* than its instruction-count
   reduction alone. v2 measures this; it can't be shown on the TITAN.
5. **The D2H copy is erratic because the host buffers are pageable.** v1 results go into a fresh `np.empty`, so every
   page faults on first touch during `cudaMemcpy` (25 s for 2.3 GB at 3e6 lazy vs 1.2 s faithful). `--pinned`
   allocates a page-locked result buffer once (`cudaHostAlloc`) and reuses it. On the TITAN at 3e4 the copy drops
   from 11 to 3.6 ms, i.e. 6.4 GB/s.

Roofline: the fast arithmetic is 2.4 M DP instructions per orbit, which is 0.14 µs/orbit at the H100's FP64 peak and
12.6 µs/orbit at the TITAN's. On the TITAN the best kernel is at 16.4 µs, 76% of that bound. On the H100 the
remaining gap is latency at 2 warps per scheduler. To close it, the kernel's register footprint has to shrink: the
12 RK stages are 66 live doubles. Options: keep the stage vectors in shared memory, or recompute instead of storing
the dense-output coefficients. Alternatively, give each thread more independent work, e.g. 2 orbits interleaved
per thread for ILP, which is only possible once registers are freed.

## Fixes implemented (all optional, all verified on the TITAN)

- `--pinned`: page-locked, reused result buffer (`co_host_alloc`), so the D2H copy is steady.
- `--refill`: persistent-thread kernel with atomic orbit refill.
- `--sort`: most-expensive-first schedule by a vertical-energy proxy (problem-specific).
- `_noinl`: one out-of-line force copy (kernel 7× smaller).
- `_fastgam`, `_fastpow`, `_fast`, `_fast_noinl`: division- and pow-free force arithmetic. This is not bit-identical
  to galpy, but agrees to ≤1.7e-15 in Γ and within the GPU's ulp-level differences on orbits.

## Timing

### v1 (H100, kernel only, µs/orbit; `data/h100/cuda_gpu_hiN.json`)

| N | 1e3 | 1e4 | 1e5 | 1e6 | 3e6 | 1e7 |
|---|---|---|---|---|---|---|
| lazy | 109 | 13.2 | 11.0 | 10.8 | 10.9 | 10.9 |
| faithful | 116 | 12.4 | 6.7 | 7.6 | 8.2 | 8.4 |

Comparison at 1e6 (µs/orbit):

| H100 CUDA v1 | jax Dopri8 H100 | torch dopri5 H100 | galpy C 1 thread | galpy C 16 threads | galpy C 72 threads (36-core dev box) |
|---|---|---|---|---|---|
| 7.6 | 16.4 | 9.2 | 2 220-2 340 | 178 | 79 |

The **galpy C series in `galpy_c_cpu_hiN.json` ran on all 72 OpenMP threads**: galpy's C orbit loop uses
`omp_get_max_threads()` and ignores `numcores`. True single-thread `dop853_c` costs 2.2-2.35 ms/orbit (chain-3
measurement `galpy_c_cpu_1cores_hiN.json`, and 2.5-2.6 ms measured here). The host build of this port costs
1.5-2.2 ms/orbit single-threaded on the same box. That is the same arithmetic without function-pointer dispatch or
the per-call `Γ(a)`; with `_fast` arithmetic it is 0.96 ms.

### v2: TITAN N scan and H100 matrix

Pending. The TITAN timing waits for the shared box's measurement chain. The H100 numbers come from `run_h100_v2.sh`.
Compile with `make_table.py` and `fp64_efficiency.py data/.../*.json`.

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
- **Performance engineering is not optional.** v1 shows that a verbatim port of galpy's arithmetic spends half its
  time in GSL-style divisions, and that one-thread-per-orbit adaptive stepping loses 1.4-1.5× to warp imbalance
  unless the kernel refills lanes. Expect to keep two arithmetic paths: galpy-exact for validation and a
  GPU-tuned path for speed.
- **Integration with galpy:** `Orbit.integrate(..., method='dop853_cuda')` would reuse `_parse_pot` unchanged and
  call this library through ctypes, as the C path does. Packaging needs an nvcc build (or a PTX/fatbin shipped
  for sm_70-sm_90) and a CUDA runtime dependency.

## Projected H100 speed

v1 measured 7.6 µs/orbit (faithful, 1e6). The diagnosis predicts, conservatively, an additional 1.4× from refill
and sort (imbalance is hardware-independent) and ≥3× from `_fast` (fewer and shorter dependent chains), giving
**≈2 µs/orbit**. This is ≈8× faster than jax Dopri8 on the same H100 and ≈1 100× faster than single-thread galpy C.
It is still ~15× from the FP64 roofline because of latency at 255 registers per thread. v2 measures this.

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
`gpu_idle.sh` is the shared dev box's GPU gate.
