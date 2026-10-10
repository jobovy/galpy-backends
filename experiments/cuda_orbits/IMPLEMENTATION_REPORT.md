# GPU-native orbit kernels in galpy: what it would take

*Report for Jo Bovy, 2026-10-08. It is based on the `experiments/cuda_orbits` prototype (MWPotential2014 + `dop853`,
one CUDA thread per orbit) and on reading galpy's C code at `feat/backends` 7236d252c: `galpy/orbit/orbit_c_ext/`,
`galpy/potential/potential_c_ext/`, `galpy/util/*.c`, `galpy/orbit/integrateFullOrbit.py:_parse_pot`, `setup.py`
and `.github/workflows/`. The effort figures are estimates for one experienced developer; the uncertainties are
stated where they matter.*

## 1. Summary

The prototype shows that a GPU-native kernel compiled from galpy's own C is fast and correct. On an H100 at
N = 10⁶ orbits (MWPotential2014, 2 Gyr, rtol = atol = 1e-8) it costs 2.4 µs/orbit with galpy's arithmetic and
1.0 µs/orbit with GPU-tuned special functions. On the same GPU, galpy's jax Dopri8 costs 16 µs, torch dopri5
9 µs, and `dop853_c` on a 72-thread node 79 µs. Results agree with `dop853_c` to within its own truncation error.
On a consumer GPU with crippled FP64 (TITAN Xp: 49 and 16 µs/orbit) the gain over a multi-core CPU node is only
1.6–5×.

Bringing this into galpy is a medium-to-large project. My estimate is **≈ 6–8 weeks for an opt-in minimum
viable version**: the analytic potentials of MWPotential2014's kind, `dop853` + RK4 + leapfrog, runtime
compilation, and CPU-proxy tests. **Full parity with the C integrators** (all potentials, wrappers, dissipative
forces, planar/1D orbits, variational equations, binary wheels) takes **≈ 5–8 months**. The heaviest single items
are the table-based potentials (SCF, multipole, interpolated) and the variational/Hessian kernels.

The ongoing cost is real: every potential becomes a fourth implementation unless the C is made single-source.
CUDA versions also churn, and CUDA 13 already drops the dev box's Pascal GPU.

**Recommendation:** proceed only with a single-source design (the existing C compiled for both host and device),
opt-in and runtime-compiled. Start with the minimum viable version and decide on the rest from its uptake.
Section 9 lists what would make it not worth doing.

## 2. What the prototype established

| finding | evidence |
|---|---|
| galpy's potential and integrator C ports to CUDA nearly verbatim. A host build of the same source is **bit-identical** to `dop853_c`. | 300 orbits, all 16 saves, identical step sequences (`check_correctness.py`) |
| On the GPU, results differ from C only through libdevice vs glibc rounding (~1 ulp). That flips an accept/reject decision on ~20% of orbits. The resulting differences (≤ 1.6e-6, or 5.3e-6 with fast arithmetic) are inside galpy's own 1e-8 truncation error (1.4e-5). | same, TITAN Xp and H100 |
| A verbatim port is only ~2× faster than jax on the H100 (7.6 µs/orbit). | `data/h100/v2/scan_v1_pinned.json` |
| Three *structural* changes give 3.3×, and none of them touches potential code: one out-of-line force function (the inlined kernel was 718 kB and thrashed the instruction cache), a persistent-thread "refill" loop (warp imbalance), and pinned output buffers. | Nsight Compute: 8.7 of 15.1 cycles per instruction stalled on instruction fetch in v1 |
| Rewriting the special functions (GSL's incomplete-gamma series is ~121 divisions per force evaluation) gives another 2.6×, but that is per-potential work. | `_fastgam`, `_fastpow` variants |
| The best kernel reaches ~8–16% of the H100's FP64 peak. It is latency-bound: 255 registers per thread leave 8 warps per SM. | ncu: 12.5% occupancy, 72.5% of cycles with no eligible warp |

The two lessons that shape any real implementation:

1. **Kernel structure matters more than arithmetic.** One out-of-line force and a refill loop carry over to every
   potential.
2. **galpy's special functions were written for CPUs.** GSL's series and continued fractions are division-heavy
   and branchy. A GPU path needs its own versions. This is optional per potential, but it is where the second
   factor of ~2.5 comes from.

## 3. Architecture options

### (a) Single-source: compile the existing C as CUDA device code

Mark every potential function and the integrator core with a macro (`GALPY_HD` → `__host__ __device__` under
nvcc, empty under a C compiler). Compile the same files twice, as now with gcc for `libgalpy` and with nvcc (or
NVRTC) for the GPU. The prototype did exactly this for three potentials and `dop853`, and its host build is
bit-identical to galpy C.

- **For it:** one implementation per potential. A new potential is written once. The CPU build *is* the
  correctness reference for the GPU code, and ordinary CI can test it (section 6).
- **Against it:** the C has to change in ways that also touch the CPU path:
  - dispatch through a type `switch` instead of function pointers, at least on the device;
  - no writes into `args` (caching);
  - no `malloc` in evaluation paths;
  - no GSL in device-reachable code.

  Each change must keep the CPU path bit-identical, or be accepted as a documented ≤ 1-ulp change. The prototype
  shows bit-identity is achievable; GSL's `gamma_inc_P` was ported and matched libgsl on every input tested.
  Some C idioms also need care under nvcc's C++ front end: designated initialisers, `bool` from `stdbool.h`,
  implicit `void*` casts from `malloc`.

### (b) A separate CUDA tree

Write the device code as its own sources (`galpy/cuda/*.cu`), sharing only the parameter packing with C.

- **For it:** no risk to the C path. The code is free to be GPU-shaped: fused R/z forces, division-free special
  functions, structure-of-arrays state.
- **Against it:** galpy would then have *four* implementations of every potential: C, the Python/array-API backend
  code, the C Hessians for the variational equations, and CUDA. They drift. This option makes sense only if the
  GPU path stays small (a handful of potentials). It is not a good basis for parity.

### (c) Runtime compilation (JIT) instead of shipping binaries

Ship the (single-source) C/CUDA files as package data and compile at first use with NVRTC, through `cuda-python`
or CuPy's `RawModule`. Cache the cubin per GPU architecture on disk.

- **For it:** no CUDA binaries in galpy's wheels. Every GPU architecture is supported, including future ones.
- **Specialisation:** this is the most interesting point. The kernel can be specialised to the user's potential
  *at compile time*. A `#define` list of the potential types actually present shrinks the dispatch `switch` to
  those cases, and parameters can become compile-time constants. That directly attacks the instruction-cache
  problem the prototype hit, which a monolithic ~50-potential binary would make worse.
- **Costs:** a first-call compile (the prototype's nvcc builds took 40–120 s for two architectures with `ptxas
  -O3`; NVRTC on one architecture should be faster but will still be seconds to a minute per potential
  combination, then cached). It also needs a runtime dependency: `cuda-python` plus the `nvidia-cuda-nvrtc` wheel
  (~60 MB), or CuPy.
- **numba-cuda** is not a real option: it would mean re-implementing the potentials in Python, a fifth
  implementation.

**Preferred:** (a) + (c). Single-source C with `GALPY_HD`, compiled for the host by the existing `setup.py` and
for the device at runtime with NVRTC. Add optional prebuilt wheels (section 7) only if users ask.

### Python API and data flow

- **Opt-in through the method name:** `Orbit.integrate(ts, pot, method='dop853_cuda')`, and likewise
  `'rk4_cuda'`, `'leapfrog_cuda'`. This mirrors `_c`, and `Orbit.check_integrator` and
  `_check_method_c_compatible` already have the shape to validate and fall back. GPU code should never be
  auto-routed to. A launch lasts as long as its slowest orbit, ≈ 0.05–0.1 s on the H100 and 0.15–0.45 s on the
  TITAN even for N = 10³, so automatic routing would make every small integration slower. An optional `device=` keyword could select among GPUs.
- **Potential packing:** reuse `_parse_pot` unchanged. Its `(pot_type, pot_args, pot_tfuncs)` output already
  holds everything needed. A host-side serialiser flattens it into one device buffer: a type table, an offset
  table, a doubles table, and spline and wrapper sub-tables. The C-side `parse_leapFuncArgs_Full` (≈ 1 000 lines
  setting ~20 function pointers per potential) stays CPU-only.
- **Inputs:** numpy arrays are copied host → device. CuPy arrays, torch CUDA tensors and jax GPU arrays can be
  taken without copying through DLPack (`__dlpack__`) or `__cuda_array_interface__`.
- **Outputs:** by default, a pinned host buffer copied into the usual `Orbit` storage, so `getOrbit()` and the
  accessors work unchanged. Optionally, a device array of the caller's framework, so a GPU pipeline does not
  round-trip through the host. Output volume matters: 10⁷ orbits × 16 saves × 6 doubles is 7.7 GB, already a
  272 ms copy on an H100.
- **Coexistence with the jax/torch backends:** independent. The CUDA path is a non-differentiable fast forward
  integrator like `dop853_c`. It can be exposed *inside* jax/torch only as a custom op (section 8), as the C-STM
  bridge (`galpy/backend/_jax/orbit_stm.py`, `_torch/orbit_stm.py`) does for C today.

## 4. What in galpy's C is GPU-hostile

| feature (where) | problem on a GPU | what has to change | size |
|---|---|---|---|
| **Function-pointer dispatch** (`struct potentialArg`, ~20 pointers per potential set in `parse_leapFuncArgs_Full`; `calcRforce`… loop over them) | Device function pointers must be taken on the device, prevent inlining and constant propagation, and every call is indirect. | A type `switch` over a flat table on the device. A host-side serialiser replaces the C parse for the GPU. The CPU can keep its pointers. | medium; mechanical once designed |
| **`struct potentialArg` with heap pointers** (`args`, `gsl_spline**`, `interp_2d*`, `wrappedPotentialArg`, `tfuncs`, `pot_data`) | Host pointers are meaningless on the device. | Serialise into one contiguous device buffer with integer offsets. | medium |
| **Caching in `args`** (PowerSphericalwCutoff, SoftenedNeedleBar, Ellipsoidal, OblateStaeckel, RotateAndTilt, SCF, Multipole, dynamical friction, NonInertialFrame: `cach*` counts 3–84 per file) | `args` is shared by all threads, and OpenMP currently works around this by copying `potentialArg` per thread. | Per-thread cache state in registers or local memory, or recompute. Most caches only save a recomputation *within* one force evaluation (R vs z component) and become unnecessary once the R/z forces are fused. | medium |
| **GSL special functions** (`gsl_sf_gamma_inc*` in PSwC/Einasto; `gsl_sf_gamma` ×15, mostly constants; `gsl_sf_expint*` in ExpTruncNFW and FDM friction; Legendre and Gegenbauer arrays in SCF/Multipole; `hyp2f1` from xsf in TwoPower*) | No GSL on the device. GSL's algorithms are division- and branch-heavy. | Device ports, bit-identical on the host as the prototype's `gamma_inc_P` is, plus optional GPU-tuned variants. Constants such as Γ(a) are precomputed on the host. | ≈ 2–3 weeks for the set |
| **GSL splines** (`gsl_spline`/`gsl_interp_accel`: interpSpherical, Multipole, MovingObject, Chandrasekhar/FDM friction, NonInertialFrame, OblateStaeckel) and galpy's own `interp_2d` b-splines (interpRZ) | Accelerators are mutable per-call state, and the tables are large. | A small device cubic-spline evaluator over host-computed coefficients (binary search or uniform-grid index), with tables in global memory through the read-only cache. GSL's natural-spline coefficients can be reproduced on the host for bit-identity. | ≈ 1–2 weeks |
| **`malloc` per call or per orbit** (SCF: 6 sites, Multipole: 35, every integrator's work arrays) | Device `malloc` is slow and serialising. | Fixed-size per-thread arrays, or restructured coefficient contractions (SCF/Multipole sum over (n, l, m) on the fly). | part of those potentials' cost |
| **OpenMP orbit loop** (`#pragma omp parallel for` over orbits, per-thread `potentialArgs`) | n/a | Replaced by the CUDA grid. The refill loop handles the imbalance OpenMP's `dynamic` schedule handles today. | small |
| **Recursion through wrappers** (12 wrapper types call `calcRforce` on `wrappedPotentialArg`) | CUDA allows recursion but needs a stack and blocks inlining. Depth is data-dependent. | Flatten wrapped potentials into sub-ranges of the type table. Apply the wrapper's transformation (amplitude, rotation, tilt) around a non-recursive evaluation of the sub-range. One nesting level covers almost all use; deeper nesting needs an explicit small stack. | ≈ 1–2 weeks |
| **Time-dependent functions `tfuncs`** (TimeDependentAmplitudeWrapper, NonInertialFrameForce: Python callables turned into C via numba `cfunc` or ctypes in `_prep_tfuncs`) | A Python callable cannot run on the device. | Tabulate on the integration time grid plus a spline (an approximation, so document it), or compile simple expressions. A genuine design decision. | ≈ 1 week + design |
| **Interpolated / expansion tables** (interpRZ, interpSpherical, SCF `Acos/Asin`, DiskSCF, Multipole radial splines) | Too large for 64 kB of constant memory. Access is per-orbit, data-dependent and latency-bound. | Global memory and read-only cache. Expect these potentials to gain the least from a GPU; whether it pays is unmeasured. | per potential |
| **Dissipative forces** (Chandrasekhar, FDM dynamical friction: velocity-dependent, σ(r) splines, ln Λ) | Fine in principle (velocity is in the state). Needs splines and per-thread caches. | As above. | ≈ 1 week |
| **Symplectic integrators** (`leapfrog`, `symplec4`, `symplec6` in `bovy_symplecticode.c`: a per-orbit `*_estimate_step` picks a fixed dt, then substeps between outputs) | GPU-friendly: no accept/reject. The per-orbit dt makes substep counts differ, and refill handles that. | Port with `GALPY_HD`. Remove `malloc` and SIGINT handling. | ≈ 1 week for the family |
| **RK4/RK6, dopr54, dop853** (`bovy_rk.c`, `leung_dop853.c`) | Adaptive accept/reject creates warp divergence (refill fixes the imbalance part). Dense output: lazy evaluation saves little under SIMT, because a warp pays whenever any lane needs it. | Port. `dop853` is done. | ≈ 1–2 weeks for the rest |
| **IAS15** (`wez_ias15.c`, predictor-corrector with iteration) | Data-dependent iteration counts and larger state (7 force substeps × predictor arrays). | Port last, if at all. | ≈ 1–2 weeks |
| **Planar and 1D orbits** (`integratePlanarOrbit.c` 1 306 lines, `integrateLinearOrbit.c` 343, separate `planar*`/`linear*` potential functions) | Same issues, separate code paths. | Port in parallel with 3D, sharing the integrators via a `dim` template parameter. | ≈ 1–2 weeks |
| **Variational / C-STM equations** (`evalRectDeriv_dxdv`, `_stm`: Hessians R2deriv, z2deriv, Rzderiv, … per potential; state 12 or 42 doubles) | Every potential's Hessian must be on the device, roughly doubling per-potential work. A 42-double state through dop853's 12 stages does not fit in registers (504 doubles), so heavy spilling. | One thread per STM column (6 threads per orbit sharing the base orbit), or stage vectors in shared memory. Needs prototyping before committing. | ≈ 3–5 weeks + per-potential Hessians |
| **SIGINT handling, progress callbacks** (`handle_sigint`, `cb` in `integrateFullOrbit`) | Host-only. | Keep on the host around the launch. Long launches would need chunking to stay interruptible. | small |

## 5. Effort by potential and by integrator

The counts come from `galpy/potential/potential_c_ext/` (≈ 12.7 k lines in 59 files; 39 potential type codes plus 12
wrapper/force codes in `_parse_pot`). Mechanical means `GALPY_HD`, switch entry, serialiser entry and tests. Each
potential also needs its Hessian functions *if* the variational path is in scope.

| group | potentials | what is needed | estimate |
|---|---|---|---|
| **A. closed-form** | Burkert, Dehnen, DehnenCore, Hernquist, HomogeneousSphere, Isochrone, Jaffe, Kuzmin, KuzminKutuzovStaeckel, LogarithmicHalo, MiyamotoNagai, NFW, Plummer, PowerSpherical, PseudoIsothermal, FlattenedPower, DehnenBar, SoftenedNeedleBar, SteadyLogSpiral, TransientLogSpiral, CosmphiDisk, EllipticalDisk, LopsidedDisk, HenonHeiles, Zero/Null, KG/IsothermalDisk/vertical (1D) | ≥ 90% mechanical. Remove the `args` caches in SoftenedNeedleBar. Optional `pow` → `sqrt` rewrites (worth ~1.5× on the GPU). | ≈ 0.3–0.5 day each → **2–3 weeks** |
| **B. special functions** | PowerSphericalPotentialwCutoff (incomplete Γ), Einasto (incomplete Γ), ExpTruncNFW (`expint`), TwoPowerSpherical / TwoPowerTriaxial (`hyp2f1` via xsf) | Device incomplete Γ (done), `expint` and `hyp2f1` ports with host bit-identity tests, plus the GPU-tuned variants. `hyp2f1` is the risky one: xsf's implementation is large C++ and its edge cases matter (cf. the jax `hyp2f1` problems). | **2–4 weeks** |
| **C. quadrature-based** | EllipsoidalPotential family (TriaxialHernquist/NFW/Jaffe, PerfectEllipsoid, TriaxialGaussian, PowerTriaxial, TwoPowerTriaxial: Gauss–Legendre sums over tabulated nodes, with caching); DoubleExponentialDisk (sums over host-precomputed Bessel-zero nodes and weights); SpiralArms (940 lines of sums) | Mostly mechanical but long loops. Remove caching. The node tables go to constant or global memory. | **2–3 weeks** |
| **D. tables / expansions** | SCF (1 025 lines, Legendre/Gegenbauer, `malloc`), DiskSCF, MultipoleExpansion (1 506 lines, 35 `malloc`s, splines), interpRZ (2D b-splines), interpSpherical (GSL spline), OblateStaeckelWrapper (tabulated) | Restructuring: no `malloc`, coefficient loops rewritten for registers, device splines. Performance unknown (memory-latency-bound). | **4–6 weeks** |
| **E. wrappers** | DehnenSmooth, SolidBodyRotation, CorotatingRotation, GaussianAmplitude, RotateAndTilt, KuzminLike, CylindricallySeparable, TimeDependentAmplitude (`tfuncs`), MovingObject (orbit splines) | Flattening design + the `tfuncs` decision + the splines. | **2–3 weeks** |
| **F. non-conservative / non-inertial** | ChandrasekharDynamicalFriction, FDMDynamicalFriction, NonInertialFrameForce (`tfuncs`, splines) | Velocity-dependent dispatch, splines, per-thread caches. | **1.5–2 weeks** |

| integrator / component | estimate |
|---|---|
| `dop853` (productionise the prototype: init/step split, refill, dense output, error codes) | 1 week |
| `dopr54`, RK4, RK6 (`bovy_rk.c`) | 1–1.5 weeks |
| leapfrog, symplec4, symplec6 incl. `*_estimate_step` | 1 week |
| IAS15 | 1–2 weeks (optional) |
| planar + 1D variants of the above | 1–2 weeks |
| SOS integrators (`integrate*_sos`) | 1 week (low priority) |
| variational dxdv / STM kernels (excluding per-potential Hessians) | 3–5 weeks |
| infrastructure: serialiser from `_parse_pot`, NVRTC build and cache, Python API, DLPack I/O, pinned buffers | 2–3 weeks |
| test harness (host-build vs C, GPU vs C, CI jobs) | 1–2 weeks |
| optional binary wheels (section 7) | 2–3 weeks + recurring |

**Totals:** groups A–F ≈ 14–21 weeks of potential work; integrators and infrastructure ≈ 12–20 weeks. Full parity
is therefore ≈ 5–8 months. Group D and the variational kernels carry the largest uncertainty: they could each take
2× the estimate if register pressure forces a different kernel structure. The minimum viable version (group A + PSwC
+ `dop853`/RK4/leapfrog + infrastructure + tests, 3D only) is ≈ 6–8 weeks.

**Mechanical vs restructuring:** about 60% of the potential code is mechanical (groups A and C). The other 40%
needs restructuring (D, E, F, and B's special functions). The prototype's main lesson is that kernel *structure*
can't be skipped: the out-of-line force, refill loop and pinned I/O are designed once in the infrastructure
weeks, not per potential. A naive port that skips them leaves ~3× on the table. With ~50 potentials in one
`switch` the instruction-cache footprint grows again, which is another reason to specialise kernels at JIT time
(section 3c).

## 6. Testing and correctness

1. **Host build of the device code vs galpy C, bit-identical.** Compile the single-source files with gcc (or with
   `GALPY_HD` empty) and compare against `dop853_c` etc. on every potential in the existing C test matrix. The
   prototype shows this is achievable, including the GSL ports. It runs in ordinary GitHub CI (CPU only) and
   catches most logic errors: dispatch, packing, wrappers, splines.
2. **GPU vs C within truncation error.** Not bit-identical (libdevice ~1 ulp, FMA contraction), so compare against
   C at the same tolerance *and* against C at a much tighter tolerance. The right acceptance test is "the GPU error
   distribution matches C's": for MWPotential2014 the maximum relative position error vs the 1e-13 truth was
   1.49e-5 (GPU) vs 1.40e-5 (C). A `--fmad=false` build gets closer to C if a stricter test is wanted.
3. **Compile-only CUDA CI.** GitHub-hosted runners have no GPU by default (GPU "larger runners" exist on paid
   plans; I have not checked their cost or availability for this repository). The CUDA toolkit installs on
   `ubuntu-latest`, so an nvcc/NVRTC compile job can at least catch device-only build errors.
4. **Periodic real-GPU runs.** A self-hosted runner on the dev box (TITAN Xp; note section 7's CUDA 13 caveat)
   and/or a scheduled Trillium H100 job (as `run_h100*.sh` here), weekly and before releases. These run the GPU
   tests and a timing regression, since performance regressions are the likeliest silent failure.
5. **Invariants as cheap extra checks:** energy and Lz conservation for static axisymmetric potentials, and the
   Jacobi integral for rotating frames, on large random orbit sets, where C is too slow to compare directly.

## 7. Packaging and distribution

- **Runtime compilation (recommended first).** Ship sources and compile with NVRTC on first use. The GPU-capable
  extra (`pip install galpy[cuda]`) depends on `cuda-python` and `nvidia-cuda-nvrtc-cu12` (or CuPy). Wheels stay
  as they are; macOS and Windows users are unaffected. Costs: a one-time compile per GPU architecture and
  potential combination, cached in `~/.cache/galpy`, plus a dependency on the NVIDIA pip wheels' versioning.
- **Prebuilt binaries.** A fatbin for sm_70–sm_90 plus PTX for forward compatibility. The prototype's library is
  1.8–6 MB for 3 potentials × 2 architectures. A full build with ~50 potentials × ~6 architectures could be tens
  of MB; that is a guess and depends heavily on inlining choices. Binaries need a CUDA-major matrix (cu12, cu13),
  linux x86_64 (+ aarch64 for Grace-Hopper) only, built in a CUDA manylinux image. Practically that means a
  separate `galpy-cuda12`/`galpy-cuda13` package, as CuPy does, not galpy's main wheel. conda-forge supports
  CUDA builds through its `cuda-version` machinery, which is probably the easiest binary channel.
- **Toolkit and driver churn.** Per NVIDIA's CUDA 13.0 release notes, CUDA 13 removed support for the Maxwell,
  Pascal and Volta architectures, so the dev box's TITAN Xp (sm_61) can only be targeted with CUDA ≤ 12. Driver 580 supports up to CUDA 13.0:
  Trillium's `cuda/13.x` modules were too new for its driver, and `cuda/12.9` was used. Expect to re-test on each
  CUDA major release and to drop old architectures on NVIDIA's schedule.
- **Users without a GPU.** `method='*_cuda'` raises a clear error, or falls back to the matching `_c` method with
  a warning, as `_check_method_c_compatible` does today when the C extension is missing. Nothing else changes for
  them.

## 8. Differentiability

The kernels are not differentiable. Three levels are possible:

1. **Initial-condition gradients (state-transition matrix).** A device version of `integrateFullOrbit_stm` gives
   dx(t)/dx(0) for every orbit, exactly as the C-STM path does today. It plugs into jax as a `jax.custom_vjp`
   whose forward calls the kernel and whose backward applies Mᵀ (the existing `_jax/orbit_stm.py` pattern), and
   into torch as an `autograd.Function` (`_torch/orbit_stm.py`).

   For the data to stay on the GPU, the jax forward should be a `jax.ffi` custom call on jax's CUDA stream rather
   than `pure_callback`, which copies through the host. The torch forward should be a registered CUDA custom op.
   The cost per orbit is the variational system: 42 doubles of state and every potential's Hessian, which is
   several times the base integration on a GPU because of register pressure. It is first-order only, like C-STM
   (no JVP-of-VJP).
2. **Parameter gradients** (d orbit / d potential parameters) are *not* provided by the STM. They need either
   forward sensitivities (∂force/∂θ for every potential parameter, another per-potential derivative set) or a
   discrete adjoint of `dop853` with checkpointing. That is a research-sized project per integrator. For
   parameter gradients the diffrax/torch in-backend paths are the right tool.
3. **Nothing beyond that.** Higher-order derivatives, `vmap` over parameters and `jit` fusion with user code are
   all unavailable through a custom op.

So a GPU kernel would be the fast *forward* integrator, with optional IC Jacobians. It complements the
differentiable backends and does not replace them.

## 9. Recommendation and staged plan

| stage | scope | estimate | decision point |
|---|---|---|---|
| 0 (done) | prototype: MWPotential2014, `dop853`, TITAN + H100, correctness and diagnosis | – | – |
| 1 (minimum viable) | single-source `GALPY_HD` refactor of group A + PSwC; `dop853_cuda`, `rk4_cuda`, `leapfrog_cuda` (3D); serialiser from `_parse_pot`; NVRTC + cache; opt-in `method=`; numpy/DLPack I/O; host-build bit-identity tests in CI; periodic GPU runs | 6–8 weeks | Do people use it on A100/H100-class hardware for ≥ 10⁵-orbit workloads? |
| 2 | groups B + C, wrappers without `tfuncs`, planar/1D, remaining RK/symplectic; device STM + jax/torch custom op for IC gradients | 8–12 weeks | Is the STM fast enough to matter vs C-STM? |
| 3 | groups D, E (`tfuncs`), F; IAS15; binary packages if demand exists | 8–12 weeks | – |

**It is not worth doing if:**
- **workloads are small or consumer-GPU-bound.** Below ~10⁴ orbits the per-launch latency dominates. On FP64-poor
  GPUs (TITAN, GeForce, most laptops) the measured gain over a multi-core CPU node is only 1.6–5×, and plain
  OpenMP `dop853_c` on a workstation is already ~0.1 ms/orbit.
- **the main users need gradients.** Then improving the in-backend diffrax/torch path, for example per-orbit
  stepping with fused kernels, gives more.
- **target models are mostly table-based** (SCF, multipole, interpolated). The GPU advantage there is unmeasured
  and may be small.
- **no maintainer with CUDA performance experience is available.** Correctness can be kept by the CPU-proxy
  tests, but performance regressions and toolkit churn need someone who reads Nsight output.

**What makes it worth it:** large orbit ensembles (stellar streams, mock catalogues, orbit-based modelling of
10⁶–10⁷ stars) in analytic potentials on data-centre GPUs. There, a 2–3 µs/orbit forward integrator with galpy's
exact arithmetic is 30–40× a full CPU node and 7× the current jax path. A single-source design keeps the
maintenance cost to "write each potential once, with `GALPY_HD`", plus a CI matrix.

## 10. Addendum (v3): occupancy headroom and integrator choice

These are H100 results at N = 10⁶–10⁷ (jobs 1066194 and 1066672, exclusive GPU) and TITAN Xp results at N = 3×10⁴. TITAN Xp timings of the shared-memory dop853 kernels were **not run**: the first attempt failed on Pascal's 48 KB per-block shared-memory limit (fixed since), and afterwards the TITAN was shared with other users and then needed for the paper's timing campaign. The H100 v3/v3b runs cover these kernels. Details are
in the README section "v3".

**Headroom for dop853 is limited by state, not by code quality.** An adaptive dop853 orbit carries ≈ 880 B of live
state. The register file (256 KB per SM) therefore fits at most ~9 warps per SM; the refill kernel already runs 8.
Moving the stage vectors into shared memory adds at most another ~8 warps on the H100 (228 KB) and ~3 on the TITAN
(96 KB), for an occupancy ceiling of ≈ 25–30% on the H100. Two orbits per thread does not fit at all. Measured on the H100, the
best shared-memory split (10 stage slots, 168 registers, 19% occupancy) is 1.3× faster than the register kernel:
1.81 µs/orbit with galpy's arithmetic, 0.69 with fast special functions, at 10⁶. That is the headroom left for
dop853.

**Integrator choice matters as much as kernel engineering.** galpy's 6th-order symplectic integrator (`symplec6`)
with one common step for the whole batch gives:
- at matched accuracy (max position error 1.4e-5 after 2 Gyr, which is `dop853_c` at 1e-8), 15% *fewer* force
  evaluations than dop853 and a 2–3× smaller energy error;
- 15–30% fewer force evaluations at the looser accuracies 1e-4 and 1e-3;
- a tiny state (q, p), no accept/reject divergence, and lockstep warps.

On the H100 it runs at 0.25 µs/orbit with fast arithmetic and 0.78 with galpy's (10⁶; 0.23 / 0.75 at 10⁷), with
registers capped at 64 for 50% occupancy. That is 3.6× / 3.0× faster than the register dop853, about 2.8× / 2.3×
faster than the best shared-memory dop853, and ~40% of the FP64 issue rate. On the TITAN it is 1.7–1.9× faster, at
~85% of that GPU's FP64 rate. symplec4 is competitive only at low accuracy (1e-3), and leapfrog needs 3–13× more work.

Two caveats:
- The symplectic speed needs a *common* step. galpy's per-orbit `estimate_step` picks power-of-two steps that
  overshoot the accuracy.
- A symplectic integrator's common step must suit the most demanding orbit in the batch. The test is a smooth,
  cheap potential (MWPotential2014) with near-solar disc orbits of similar dynamical time. A batch mixing very
  different timescales (cusps, halo orbits), with close encounters, or in rapidly time-varying potentials would
  have to step at its fastest orbit's rate, and adaptive dop853 would win there.

**For a GPU implementation in galpy** (section 9), this argues for starting with the symplectic integrators. They
are smaller to port (≈ 290 lines with step estimation, vs ≈ 340 + 160 for dop853), are bit-identical to the C on the
host, and are the faster kernels. dop853 would follow for batches that need per-orbit adaptivity.

**What limits the best kernels now (H100 v3b, Nsight Compute).**
- *Achieved occupancy and stalls.* The shared-memory dop853 achieves 18% occupancy and symplec6 at 64 registers 39%.
  Both are limited by fixed-latency dependency stalls (~45–51% of stall cycles): too few independent instructions
  per warp to hide the 4–8-cycle FP64 latency. They reach 10% and 19% of the H100's FP64 peak.
- *Measured times.* At 10⁷ the kernels cost 1.74 / 0.66 µs/orbit (dop853, galpy / fast arithmetic) and 0.75 / 0.23
  µs/orbit (symplec6).
- *FMA contraction.* ncu counts 12.4×10⁹ unfused FP64 instructions against 6.4×10⁹ FMAs for dop853 (8.3 vs 3.1×10⁹
  for symplec6), and estimates up to 33–36% more FP64 throughput if add/mul pairs were fused. nvcc's FMA contraction
  (`--fmad=true`) is **already on** in these builds; only the `_nofma` variant turns it off. The pairs it leaves
  unfused are mostly not contractible without changing the rounding:
  - products that feed several consumers or carry no following add: `term *= x*inv[n]`, `(c*dt)` before a drift,
    `x*x`;
  - adds whose operands are not products: `sum += term`, error-norm accumulations;
  - expressions where fusing would require reassociation, which IEEE semantics forbid without fast-math.
- *Next step.* A cheap, untested next step is to rewrite the hot loops (the incomplete-gamma series, the drift/kick
  updates, the dop853 stage combinations) in explicitly fused forms (`fma(a,b,c)`, Horner-style recurrences). This
  changes the rounding at the ulp level, so it belongs in the "fast arithmetic" path. Under the
  matched-truncation-error criterion it is acceptable for both paths, but the galpy-arithmetic path would then no
  longer be a like-for-like port of the C. The ncu estimate is an upper bound, not a forecast.
