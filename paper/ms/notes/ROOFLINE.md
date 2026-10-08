# Roofline analysis of the paper's CPU-vs-GPU speed comparisons

How close each measured series is to the hardware limits (FP64 throughput, memory
bandwidth), what bounds it, and how a full CPU node would compare. Wall times are the
measured ones in the figure JSONs; no benchmark was re-timed. The one exception is a short
single-thread check of the galpy C integrator, see §0. Every number marked **(est.)**
is an estimate, and the assumptions behind each are listed in §7.

Scripts and their raw outputs are in `paper/ms/notes/roofline/`. §6 lists how to run them.

## 0. Read this first: the "single-thread C" orbit series is not single-threaded

`galpy_c_measure.py` labels its JSONs "dop853_c (compiled, single-thread, sequential)" and
passes `numcores=1`. But galpy's C orbit loop is an OpenMP `parallel for` over
`omp_get_max_threads()` threads (`integrateFullOrbit.c` l.1108–1184), and `numcores` never
reaches it. The script's own docstring says so. `tools/rerun_figures_gpu.sh` does not set
`OMP_NUM_THREADS`, so the dev-box C series ran on **all 72 hardware threads**. Two checks:

* `c_dop853.py` times `dop853_c` with `OMP_NUM_THREADS=1` on one pinned, idle core, using the
  same protocol (MWPotential2014, 2 Gyr, rtol=atol=1e-8, nt=16, best of 3). The result is
  **2.33 ms/orbit (N=300) and 2.38 ms/orbit (N=3000)**, against the JSON's 0.08–0.11 ms/orbit.
  With OpenMP unrestricted on 12 cores, I get 0.27 ms/orbit wall time and 2.7 ms/orbit of CPU time.
* The JSON's own low-N points show the same thing. N=1 costs 2.9 ms, which is the cost of one
  orbit, not "per-call setup". N=10 costs 5.0 ms in total. Per orbit, the cost then falls 30× by
  N=1e4, which a sequential loop cannot do.

So **the measured "galpy C" line in g8/g12 is the full dual-socket dev box** (36 cores, 72
threads; 29.7× the single-core speed, 82% of linear scaling on 36 cores). The single-core
cost is ~2.35 ms/orbit. The June value of ~2.1 ms/orbit that the g12 caption dismisses as
a "slow build" was most likely the true single-thread number. The g12 caption claims
"Single-thread C is the per-core baseline... galpy's C path also runs orbits in parallel
(`numcores`), which divides its line by the number of cores". That claim needs fixing, and
so does every GPU-vs-"single-thread C" ratio in the text.

## 1. Hardware peaks used

| device | FP64 peak | memory BW | source / derivation |
|---|---|---|---|
| Intel Xeon E5-2697 v4 (dev box), 1 core | 36.8 GFLOP/s at 2.3 GHz base (57.6 at 3.6 GHz turbo) | ~10–12 GB/s single-core stream (est.) | 18 C/36 T per socket, 2.3/3.6 GHz, AVX2+FMA (2× 256-bit FMA → 16 FP64 flop/cycle), 4× DDR4-2400 → 76.8 GB/s per socket ([Intel ARK](https://www.intel.com/content/www/us/en/products/sku/91755/intel-xeon-processor-e52697-v4-45m-cache-2-30-ghz/specifications.html)) |
| dev box, full node (2 sockets, 36 cores) | 1.32 TFLOP/s at base (lower at the AVX all-core clock) | 153.6 GB/s | 2 × the above |
| NVIDIA TITAN Xp (sm_61) | 0.380 TFLOP/s (= 12.15 TFLOP/s FP32 / 32) | 547.7 GB/s | 3840 cores × 2 × 1582 MHz boost = 12.15 TF FP32; GP102 FP64 rate 1/32 ([videocardz](https://videocardz.com/68039/nvidia-launches-titan-xp-with-3840-cuda-cores), [NVIDIA](https://www.nvidia.com/en-us/titan/titan-xp/)). GPU Boost can clock above 1582 MHz, so the %-of-peak values may be ~10% high |
| NVIDIA H100 80GB HBM3 **SXM5** | 34 TFLOP/s FP64 (67 with FP64 tensor cores, which these kernels do not use) | 3.35 TB/s | [NVIDIA H100](https://www.nvidia.com/en-us/data-center/h100/). It is SXM, not PCIe: `logs/h100/report.md` lists "H100 80GB HBM3 (sm_90), 1 of 4 on the node", and the PCIe 80 GB part uses HBM2e. The Trillium GPU nodes are listed as "AMD EPYC 9654 96C, NVIDIA H100 SXM5 80GB" ([TOP500](https://www.top500.org/system/180415/)) with 4 H100s, 96 cores and 768 GB per node ([SciNet](https://scinethpc.ca/trillium/)) |
| AMD EPYC 9654 (Trillium GPU node CPU), 1 core | 38.4 GFLOP/s at 2.4 GHz base (Zen 4: 2× 256-bit FMA → 16 FP64 flop/cycle) | 460.8 GB/s per socket (12× DDR5-4800) | 96 C, 2.4/3.7 GHz, 12 ch DDR5-4800 ([Wikipedia: Epyc](https://en.wikipedia.org/wiki/Epyc)). A 1-GPU job got 24 of the 96 cores |

Ridge points (FP64 flop per byte): TITAN 0.69; H100 10.1; one Xeon socket 8.6.

## 2. Method

* **Work per unit** comes from XLA's own accounting of the exact jitted function each make.py
  times, compiled on the **CPU backend** (all runs CPU-only, `taskset -c 24-35 nice -n 19`,
  `CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu`, under a 16 GB `MemoryMax` scope).
  `compiled.cost_analysis()` **counts every while-loop body once.** I checked this: a
  `fori_loop` of 1, 10 or 100 iterations reports the same flops. Most of the work here sits
  inside loops (jax's native `igamma`, the diffrax step/stage loops, the samplers' bisections,
  gV2's Newton iteration), so `hlo_cost.py` re-derives the census from the optimized HLO text
  with a per-loop trip count. It uses XLA's `known_trip_count` where a loop has one, and
  otherwise the trip count derived below. With every trip count set to 1, the census reproduces
  `cost_analysis()`: 1450 vs 1472 flops and 43 vs 43 transcendentals for the full Dopri8 solve;
  279 vs 311 flops and 7 vs 7 for the RHS; exact for g2.
* **FLOPs.** "XLA flops" follows XLA's convention: every elementwise op counts 1, including
  compare, select and integer ops. Transcendentals are counted separately and converted at
  **W = 20 flops each** (exp, log, pow, sqrt, sin, cos, atan2). This is a stated convention,
  not a measured cost: CUDA FP64 `pow` and `sin`/`cos` are likely 30–100 FP64 operations, and
  XLA-CPU's vectorized sin/cos are ~10 (§4 gV2). "FP64 flops" = add/mul/div + W × transcendentals,
  with compares, selects and integer ops excluded. That is the number compared with the FP64 peak.
* **Bytes** are reported as a bracket, because XLA-GPU fuses differently (more aggressively)
  than XLA-CPU:
  * *census*: the operand and result bytes of every top-level CPU-HLO kernel (XLA's "bytes
    accessed" model; in-place dynamic-update-slices count only their update). This is an
    upper estimate for the GPU.
  * *ideal*: perfect fusion. Each loop body is one kernel that reads and writes its carry once
    per iteration, and the entry computation moves only its inputs and outputs. This is a
    lower bound for the GPU.
* **Eager torch** is censused per aten op (`TorchDispatchMode`). One op is one kernel that
  streams its operands and result, so the eager traffic is exact up to caching.
* **Orbit trip counts** (`orbit_stats.py`): I integrated, on the CPU with galpy's diffrax
  configuration, 4096 random ICs plus the 1024 lowest-R and 1024 highest-R ICs out of the full
  N=3e5 and N=3e6 IC sets the measure scripts draw (`PRNGKey(N)`). Because vmap-of-solve runs
  in **lockstep**, every lane executes every iteration of the batched while-loop, and finished
  lanes are masked. Performed work is therefore set by the batch maximum (S_max steps; the
  `igamma` continued fraction runs to the slowest lane, i.e. the smallest r), while useful work
  is set by the mean.
* **Achieved rates** = work per unit × N / wall time at the largest N measured. The table also
  gives the *marginal* rate, the slope between the two largest N, which removes fixed per-call
  costs.

## 3. Summary table (largest N; † = estimate-dependent, see notes)

FP64 % = FP64 flop/s ÷ FP64 peak. BW = achieved bytes/s under the ideal–census bracket
(ideal % of peak first). "Node" = the full dev box (2× E5-2697 v4, 36 cores).

| benchmark (unit) | work / unit | series (N) | time / unit | FP64 GF/s (% peak) | BW (% peak) | bound | full CPU node vs GPU |
|---|---|---|---|---|---|---|---|
| **g2** Plummer F_R (point) | 7 flops + 1 `pow` (27 flop-eq); 24 B fused / 144 B eager (8 kernels) | numpy, Xeon 1 core (1e7) | 55 ns | 0.49 (1.3%) | 3.3 GB/s | scalar libm `pow` (0.29 s of 0.45 s) | numpy on 36 cores: ~0.65 Gpt/s† |
| | | numpy, EPYC 1 core (1e7) | 11 ns | 2.4 (6%) | 16 GB/s | `pow` / allocation | |
| | | jax jit TITAN (1e7) | 0.43 ns | 62 (16%; 40–65% if CUDA `pow` ≈ 60–100 ops†) | 55 GB/s (10%) | FP64 (`pow`) | TITAN ≈ 3.5× node numpy |
| | | jax jit H100 (1e7) | 0.015 ns | 1850 (5%) | 1.65 TB/s (49%); **marginal 2.97 TB/s (89%)** | **HBM** | H100 ≈ 105× node numpy; ≥11× a perfect 24 B/pt CPU kernel (6.4 Gpt/s cap)† |
| | | torch eager TITAN (1e7) | 0.68 ns | 40 (10%) | 210 GB/s (38%) | `pow` kernel + 7 streaming kernels | |
| | | torch eager H100 (1e7) | 0.059 ns | 460 (1.4%) | 2.45 TB/s (73%; marginal 80%) | HBM, but 6× the necessary traffic | |
| **orbit** g8/g12 (orbit, 2 Gyr, 1e-8) | useful: Dopri8 4.8 Mflop-eq, 254 steps (197 accepted); Dopri5 5.1 Mflop-eq, 593 steps; C ~4.9 Mflop-eq† | C `dop853_c`, 1 thread (this note, 3e3) | 2.35 ms | 2.1 (5.6%)† | – | latency-bound scalar code (GSL Γ_inc, divisions) | |
| | | C as measured = **72 OpenMP threads** (1e6) | 0.079 ms | 62 (4.7% of node)† | – | same, 29.7× parallel | **this is the node** |
| | performed (lockstep): Dopri8 12.9 Mflop-eq, 33–91 MB | jax Dopri8 TITAN (3e5) | 0.215 ms | 37 (10%) | 136–365 GB/s (25–67%) | mixed: FP64-heavy RHS + bandwidth-bound loop kernels | node C is 2.7× faster than the TITAN |
| | | jax Dopri8 H100 (3e6) | 0.0155 ms | 585 (1.7%); useful 0.9% | **2.14 TB/s (64%) → 5.9 (>peak)**; marginal ideal 66% | **HBM traffic of loop-carried state** | H100 5.1× node |
| | performed: Dopri5 10.2 Mflop-eq, 25–75 MB | jax Dopri5 TITAN (3e5) | 0.129 ms | 52 (14%) | 183–552 GB/s (33–100%) | bandwidth | node 1.6× faster |
| | | jax Dopri5 H100 (3e6) | 0.0125 ms | 565 (1.7%) | 1.99 TB/s (59%) → >peak | HBM | H100 6.4× node |
| | torch: 3416 shared RHS calls, 143 kernels each, 15.2 MB, ~4.1 Mflop-eq† | torch dopri5 TITAN (1e6) | 0.0756 ms | 54 (14%) | 201 GB/s (37%) | bandwidth + FP64 `pow`/`igamma`; launch floor | ≈ node (1.05×) |
| | | torch dopri5 H100 (3e6) | 0.0084 ms | 490 (1.4%) | 1.80 TB/s (54%; marginal 56%) | HBM (unfused eager) | H100 9.4× node |
| **gD3** spherical-DF sample (particle; eddingtondf NFW shown, the other 3 DFs in [ ]) | 2315 XLA ops [1873–2533]: FP64 arith 117 [107–144] + 22 transc; integer (threefry RNG) 1108; cmp/select 1090; 1.6 kB ideal [1.1–1.7] / 4.7 kB census; output is 48 B | jax jit TITAN (1e7) | 6.3 ns [3.9–6.8] | 88 (23%) [21–35%] | 248 GB/s (45%) [45–50%] → >peak | bandwidth | node jax-CPU ≥11× slower† |
| | | jax jit H100 (1e8) | 0.62 ns [0.38–0.72] | 890 (2.6%) | **2.52 TB/s (75%)** [72–84%] → >peak | **HBM** | node jax-CPU ~115× slower† |
| | | jax jit, Xeon 4 cores (1e7) | 0.65 µs | 0.86 (2.3% of 1 core) | 2.4–7.3 GB/s | neither: small-loop / dispatch overhead† | 36 cores: 0.72 s per 1e7† |
| | | numpy, Xeon 1 core (1e7) | 5.4 µs [1.8–10.8] | – (different algorithm, scipy spline) | – | – | 36 cores: 1.5 s per 1e7† |
| **gV2** VerticalInverse `xvFreqs`, one torus (angle) | 13.2k flops + 3229 sin/cos (8 Newton iterations × ~400) = 77.8k flop-eq (45.5k at W=10); 0.53 kB ideal / 5.2 kB census, i.e. ≥15 flop/B | jax jit TITAN (1e6) | 308 ns | 253 (**67%**; 39% at W=10) | 1.7–17 GB/s (≤3%) | **FP64** | node jax-CPU (55 ms/1e6†) is 5.6× faster than the TITAN |
| | | jax jit H100 (1e7) | 10.2 ns | 7600 (22%; 13% at W=10; ~35–45% if GPU sin/cos ≈ 35–40 ops†) | ≤0.5 TB/s (≤15%) | **FP64 transcendentals** | H100 5× node (≤26× = FP64 peak ratio) |
| | | torch eager TITAN / H100 (1e6 / 1e7) | 937 / 66 ns | 83 (22%) / 1180 (3.5%) | small | dispatch + unfused | |
| | | jax jit, Xeon 1 thread (1e6) | 1.99 µs | 39 (106% at W=20 → W=20 is too high on CPU; **62% at W=10**) | – | **FP64 compute (efficient)** | |
| | | numpy, Xeon 1 thread (1e6) | 9.6 µs | 8.1 (22%) | – | compute (unfused) | |
| **g17** Staeckel AA | – | skipped: `g17_staeckel_actionangle_speed/data/g17_staeckel_actionangle_speed.json` is absent (only `.bak` and `h100/`); it is being re-measured | | | | | |

## 4. Notes per benchmark

**g2.** The fused jax kernel is the textbook case: 24 B and 27 flop-eq per point, 1.1 flop/B.
On the H100 it is HBM-bound. At N=1e7 the 0.146 ms still includes the ~0.05 ms launch floor
visible at N ≤ 1e5; the slope from 1e6 to 1e7 gives 2.97 TB/s, **89% of peak**. On the TITAN
the same kernel moves only 10% of peak bandwidth: it is FP64-bound by the single `pow` at the
1/32 rate. At W=20 that is 16% of peak, and 40–65% if CUDA's FP64 `pow` costs 60–100
operations (est.). Eager torch launches 8 kernels and moves 144 B/pt, 6× the fused traffic.
On the H100 it streams them at 73–80% of peak, i.e. it is efficient per byte but moves 6× the
bytes, and so is 4× slower than jax. The numpy timing includes building the input grid
(~0.10 s of 0.553 s). F_R itself is 0.45 s, of which libm `pow` takes 0.29 s (micro-check on
one Xeon core). Single-core numpy is therefore `pow`-throughput-bound (29 ns/element), not
bandwidth-bound.

**Orbits.** The work decomposition per RHS evaluation (per orbit, Dopri8, `orbit_costs.json`)
is 192 loop-free flops + 7 transcendentals, plus **87 flops and 0.2–0.7 kB per iteration of
jax's native `igamma` continued fraction**. It takes 10 iterations at r=8 kpc and 14 at the
smallest r of a 3e6 batch. So the incomplete gamma of `PowerSphericalPotentialwCutoff` is
**~75% of the arithmetic and ~55% of the ideal traffic** of an RHS. Per step, Dopri8 is 13 RHS
+ ~3.6k flops of RK combinations and controller, and 51–63 kB ideal (130–175 kB census) per lane.
The jax line runs native `igamma`, two convergence while-loops, and not galpy's loop-free
series/CF fallback. Inside vmap-of-solve the router's `x.size < _JAX_MIN_SIZE` test
(`special/_fallback/gammainc.py`) sees the per-lane size 1, not the batch.

Lockstep costs **2.3–2.7×**: performed per-orbit work over useful work, from
S_max/S_mean = 545/254 for Dopri8 at 3e6 and 957/593 for Dopri5, times the CF count of the
slowest lane. S_max is a lower bound, since the sample covers only the R tails of the 3e6 ICs.

Torch dopri5 carries one (N, 6) state through a shared controller. Its error norm is an RMS
over all orbits, so nfev ≈ a typical orbit's: 3416 at N=1024, assumed the same at 3e6 (est.).
It runs 143 kernels and moves 4.45 kB per orbit per RHS call. Each RHS evaluation also
forces **4 host syncs**: `branch_where` in `galpy/backend/_coerce.py` reads `bool(nc)` and
`int(nc)` twice per call.

C has no counter. Its RHS count, 3344 mean over 24 orbits (range 2517–4650), comes from galpy's
line-for-line Python port of `leung_dop853`. Its arithmetic is *assumed* equal to the jax RHS
plus the Dopri8 stage overhead, ~1470 flop-eq per RHS (est.). The real C RHS differs: it uses
GSL `gamma_inc_P` and `gsl_sf_gamma`, `atan2`, and `sin`/`cos`. At 0.70 µs per RHS on one core,
C is latency-bound scalar code at ~5% of a core's SIMD peak.

**gD3.** Every loop has a known trip count: 5-step table bisections and 7/10/15-iteration
fixed loops. Real FP64 arithmetic is only ~5% of the ops. The rest is threefry integer RNG
(~1100) and compares/selects (~1100), which do not run on the FP64 pipe, so FP64 % is not the
relevant roof. Bandwidth is: the ideal model puts the H100 at 72–84% of HBM peak, and the census
exceeds peak, so the truth lies in between. That makes the jitted sampler **HBM-bound by its
loop carries**. It moves 1.1–1.7 kB per particle against 48 B of output, so a fused
one-thread-per-particle kernel has up to ~30× of headroom before the integer/select throughput
bounds it (est.).

**gV2.** At ≥15 flop/B this is compute-bound everywhere. The TITAN runs at 67% of its FP64
peak at W=20 (39% at W=10). The H100 runs at 13–22% at W=10–20 of the nominal 34 TF. Its
transcendentals probably cost more than 20 operations in CUDA FP64, which would put it at
~35–45% (est.). The single-thread XLA-CPU number is 106% of peak at W=20, which shows that
XLA-CPU's vectorized sin/cos cost about 10 flops; at W=10 it runs at 62% of the base-clock
peak. Because the kernel is FP64-bound, the GPU advantage is capped by the FP64 peak ratio:
H100 vs the 36-core node is ≤26×, the measured ratio is 5×, and the TITAN loses to the node.
Generating sin(nθ) and cos(nθ) for the harmonics by angle-addition recurrence instead of ~400
library calls per Newton step would cut the work ~5–10× (est.).

## 5. What limits the GPU orbit integration, and what a better integrator could reach

**Regime 1, small N (the ~1.5 s floor).** The H100 Dopri8 line is flat at 1.4–1.65 s up to
N≈1e4, and the TITAN line at 2.0–2.2 s. Every batched while-loop iteration on XLA:GPU launches
its body kernels and copies the predicate to the host before it can continue. Per step that is
1 step-loop + 13 stage-loop + 13 × (n_cf + 1) CF-condition + 13 series-condition + ~2 save-loop
iterations: ~200 for Dopri8 (n_cf ≈ 12) and ~95 for Dopri5. Over ~440 / ~800 steps that is
8.7e4 / 7.4e4 host-synchronised iterations, i.e. **~19 µs per iteration on both solvers**
(est.). The floor is therefore launch and synchronisation latency, not arithmetic. Torch's floor
works the same way per kernel: 4.06 s / 489k kernels = 8.3 µs/kernel on the H100, and
22 µs/kernel on the TITAN host.

**Regime 2, large N (1e6–3e6).** Per orbit-step, the H100 needs 28 ns for Dopri8 and 13 ns for
Dopri5. At FP64 that is only **1.7%** of peak (0.9% counting only useful work). Memory traffic
is **≥64–66%** of HBM peak under perfect fusion, and the CPU-fusion census exceeds peak. The
large-N GPU integration is therefore **memory-bound on the loop-carried state**, not on FP64
throughput. Three things feed it:

* the `igamma` CF carry: ~200 B per lane per iteration × ~14 iterations × 13 stages per step,
  ~55% of the traffic;
* the 14×6 RK stage buffer, re-streamed at every stage;
* the lockstep execution, which makes all lanes run S_max steps and n_cf(r_min) iterations:
  2.3–2.7× wasted work.

Torch is in the same regime (54–56% of HBM peak, unfused eager). Its traffic per RHS call
is similar to jax's (4.45 vs ~4.7 kB per orbit). It is still the fastest H100 line because the
shared controller performs ~3400 RHS evaluations per orbit, against jax-Dopri5's ~5700 lockstep
evaluations (957 × 6), and because torch's native `igamma` is a single per-element kernel, not
a batched while-loop. Per orbit that is 15.2 MB against ≥24.8 MB, a ratio of 1.6, and the
measured time ratio is 1.5.

**What a better-structured GPU integrator could reach (all est.):**

1. *Loop-free RHS* (route the fixed-length `igamma` fallback under vmap, or tabulate the enclosed
   mass M(<r), since α is fixed). This removes ~200 of the ~225 host-synchronised loop iterations
   per step: the floor drops from ~1.5 s to ~0.1–0.2 s. The ideal traffic drops from ~61 kB to
   ~28 kB per lane-step, so large-N runs ~2× faster (~7 µs/orbit).
2. *Fused per-orbit kernel*: one thread per orbit, state and stages in registers, the whole
   adaptive loop inside one kernel (Pallas/Triton/CUDA; a fixed-step `scan` alone does not get
   there under vmap). The only traffic left is the ICs and the 16 saves per orbit, so the kernel
   becomes FP64-bound. With per-thread adaptivity there is no batch-lockstep waste, only warp
   divergence. The useful 4.8 Mflop-eq per orbit at 25–50% of 34 TF gives **0.3–0.6 µs/orbit,
   25–50× the measured 15.5 µs**. That is ~130–260× the full 36-core dev box running C, and 3e6
   orbits in ~1–2 s.
3. *Cheaper force* (a spline or Chebyshev M(r) instead of Γ_inc, ~200 instead of ~1200
   flop-eq per RHS) would be worth another ~4× on top of (2), for ~0.1 µs/orbit.
4. *Fixed-step* stepping on its own gains little in the vmapped design. It removes the rejected
   steps (22% of Dopri8's) and the step-loop syncs, but still pays the inner loops and the
   stage-buffer traffic. Its main value is enabling (2) without per-thread control flow.

## 6. Reproducing (CPU only, ~10 min total)

```
cd paper/ms/notes/roofline
E="taskset -c 24-35 nice -n 19 env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONPATH=/var/tmp/fbp:/home/bovy/Repos/galpy-jax systemd-run --user --scope -q -p MemoryMax=16G"
$E python orbit_stats.py   # -> orbit_stats.json  (lockstep S_max/S_mean, r_min)
$E python orbit_costs.py   # -> orbit_costs.json  (RHS / per-step census, igamma CF counts)
$E python misc_costs.py    # -> misc_costs.json   (g2, gD3, gV2 census; g2 torch ops)
$E python torch_costs.py   # -> torch_costs.json  (eager torch kernels/bytes/RHS count)
taskset -c 35 env OMP_NUM_THREADS=1 PYTHONPATH=/var/tmp/fbp python c_dop853.py   # nfev + 1-thread check
python roofline_tables.py  # -> roofline_tables.json / .txt (all rates in §3)
```

`hlo_cost.py` is the loop-aware HLO census used by the drivers. galpy was
`/var/tmp/fbp` @ 9a877712 (feat/backends), with jax 0.10.1 / diffrax 0.7.2 / torch 2.12.0 on
the dev box. The H100 runs used jax 0.11.2 and torch 2.14.1. Op counts can differ slightly
across versions (est.).

## 7. Caveats and assumptions (every estimate)

* **W = 20 flops per transcendental** is a convention. The plausible range is ~10 (XLA-CPU
  SIMD) to 60–100 (CUDA FP64 `pow`), and gV2 and g2-TITAN percentages scale with it.
* **Bytes are a bracket.** The GPU kernels were not compiled; the CPU HLO stands in for them.
  Where the census exceeds peak bandwidth, the GPU evidently fuses better, and the ideal lower
  bound is the defensible number.
* Compares, selects and integer ops count in "XLA flops" but not in "FP64 flops". They are
  37% of the orbit ops and ~95% of the sampler ops.
* Orbit trip counts: S_max is the maximum over a 6144-orbit sample including both R tails of
  the actual N-orbit IC set, so it is a lower bound for the full batch. n_cf is taken at the
  sample's r_min. The C arithmetic per RHS is assumed equal to the jax RHS. C's nfev comes from
  the Python port, run on 24 orbits. Torch's nfev at 3e6 is assumed equal to its N=1024 count.
  The torch `igamma` is taken at ~200 flop-eq per call. The floor model's ~440/800 steps at
  N=1e3 are interpolated.
* The **1-thread C timing** (2.35 ms/orbit) is the only new timing in this note, a sanity
  check on one idle core of the shared box. So is the numpy `pow` micro-check.
* Full-node CPU figures assume ideal linear scaling for the embarrassingly parallel paths,
  capped by node memory bandwidth where relevant (g2 numpy). The orbit node figure is
  *measured*, since the C series already used all threads. No EPYC-node C orbit timing exists.
  If a Zen 4 core runs this scalar code 1.5–2× faster than a Broadwell core (not measured), a
  96-core EPYC 9654 would reach ~0.012–0.016 ms/orbit, i.e. parity with the H100 lines.
* TITAN %-of-peak uses the 1582 MHz rated boost; the actual clock may be higher.
* gD3 and gV2 censuses are at N=1e5 and scaled linearly. Per-point costs are N-independent
  except for constants such as the baked-in tables, which are negligible.
