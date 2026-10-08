**galpy's own in-backend differentiable orbit integrator: GPU vs CPU vs compiled
C.** The analogue of G8 (which compared galpy to galax), but for galpy's #906
in-backend integrator (`galpy.backend._reference.integrate_orbit`): the same
MWPotential2014, Sun-like orbit integrated 2 Gyr over a batch of N initial
conditions, wall time for the whole batch vs N. All lines use the *same*
backend-agnostic galpy forces, `rtol=atol=1e-8`, and per-orbit adaptive stepping.
The galpy-C and galpy-jax (CPU/GPU) series are loaded from the shared
`paper/figures/_shared/orbit_speed/data` and are **byte-for-byte identical to G8's**
(same orbit, tolerance, and nt convention — lowN saves 1000 dense steps, high-N GPU
saves 16), so the two figures can be read together.

Colours follow the paper-wide palette (framework × device: numpy black, galpy C
grey, jax blues and torch oranges, light → dark for CPU → TITAN Xp → H100); jit is
solid, eager dashed; lines are labelled directly. Every GPU line was measured on two
cards: the dev box's **NVIDIA TITAN Xp** (consumer, float64 at 1/32 of float32) and
an **NVIDIA H100 80GB** (Trillium; jax 0.11.2, torch 2.14.1+cu130). The small-N end
is launch/compile-bound and barely moves (~1–4 s per call on both). At large N the
H100 is 9–12× faster per orbit than the TITAN: jax Dopri8 0.025 ms/orbit at
N = 3×10⁵ (TITAN 0.21), the same-work jax Dopri5 0.018, and eager torch dopri5 0.009
ms/orbit at N = 10⁶ (TITAN 0.076). galpy's C orbit loop is OpenMP-parallel over
the orbits (`omp_get_max_threads()`; the `numcores` keyword does not reach it), so
C is drawn on **one core** (`OMP_NUM_THREADS=1`, pinned with `taskset`; 2.2–2.4
ms/orbit) and on the **whole node** (unpinned: all 72 hardware threads of the dual
18-core Xeon E5-2697 v4; ~0.08–0.10 ms/orbit), plus on 16 pinned threads (0.17–0.2
ms/orbit, 178 s at N = 10⁶; 13× one core, 2.2× slower than the node). Every GPU line overtakes one-core C at N ≈ 10³ (eager torch,
with its flat 4–11 s floor, at N ≈ 2–5×10³). On the H100 the GPU lines also overtake
the 72-thread node at N ≈ 3×10⁴ and at the largest N are 5–9× cheaper per orbit
than it (0.079 ms/orbit at N = 10⁶) and 150–280× cheaper than one-core C.

* **galpy (C, CPU)** — the compiled `dop853_c` integrator, on one core (grey,
  solid), on all 72 hardware threads (light grey, dash-dot) and on 16 pinned
  threads (mid grey, dash-dot-dot). One core: ~3 ms per-call setup, then 2.2–2.4
  ms/orbit at 1e-8 at every N ≥ 100 (716 s at N = 3×10⁵). 72 threads: ~0.1
  ms/orbit (0.079 at N = 10⁶): on the float64-limited TITAN Xp the node is the
  fastest line at every measured N (the in-backend GPU lines amortise their ~1.5 s
  compile/launch floor only down to ~0.13–0.21 ms/orbit). The June version of this
  figure had ~2.1 ms/orbit for C: that is the one-core cost (it matches the
  one-core line point by point), not a slow build.
* **galpy in-backend (JAX, GPU; Dopri8, shipped)** — diffrax `Dopri8` (8th order) of
  the backend-agnostic forces, `jax.vmap`-ed over the batch and jit-fused, on the
  NVIDIA TITAN Xp (sm_61). This is **what galpy ships on jax**. The vectorised solve
  amortises across the batch down to ~0.25 ms/orbit at N=1e5 and ~0.21 at 3e5
  (~11× faster per orbit than one-core C, ~2.5× slower than the 72-thread node);
  crosses one-core C at N≈10³.
* **galpy in-backend (JAX, GPU; Dopri5, same-work)** — diffrax `Dopri5` (5th order)
  through the **same** backend forces, jit+vmap, GPU. galpy ships Dopri8 on jax but
  `dopri5` on torch, so the shipped jax-vs-torch crossover is **not same-work**; this
  line **is** the apples-to-apples match (same 5th-order method, matching nfev/steps —
  see work counts below). Cheaper-per-step Dopri5 reaches ~0.13 ms/orbit at N=3e5,
  *beating* the shipped Dopri8 line on the GPU despite taking ~3× more steps.
* **galpy in-backend (JAX, CPU; Dopri8)** — the shipped diffrax `Dopri8` path forced
  onto the CPU, isolating the device contribution from the framework/vectorisation
  cost, on ONE core (pinned with `taskset`, the only way to keep XLA:CPU's thread
  pool to one core; 2026-10-08): 2.0 ms/orbit at N = 10³ (one-core C: 2.3), 2.6 at
  10⁴, 3.6 at 10⁵ and 4.6 ms/orbit at N = 3×10⁵ (1390 s; one-core C 716 s). The
  earlier unpinned run (up to 72 threads) was 5.5 ms/orbit at 10³ and 2.8 at 3×10⁵.
* **galpy in-backend (torch, GPU; dopri5, eager)** — the torchdiffeq `dopri5` path
  (5th order — torchdiffeq's `dopri8` *backward* pass is buggy, so galpy ships `dopri5`
  on torch; state carried as (6, N) so one `odeint` vectorises the batch). torchdiffeq
  runs **eager** — there is no jit-fuse of the adaptive-step loop. At small N the batch
  is *overhead-bound*: the whole batch takes a roughly *constant* ~12 s set by the
  thousands of Python-driven steps, the GPU is starved, and ms/orbit falls as ~1/N
  (here torch is tens of × slower per orbit than the jit-fused jax lines — exactly why
  G8 omits a torch line). Pushed to large N the line crosses into the *compute-bound*
  regime — the per-step GPU kernels finally dominate the Python overhead, wall time
  rises ~linearly, and ms/orbit flattens to ~0.069 (correct and fully differentiable;
  shown out to N=1e6). That this beats jax-Dopri5's ~0.13 is **not** the method (same
  work) but torchdiffeq's single *shared* (6,N) controller (one dt sequence + one fused
  per-step kernel for the whole batch) vs jax's per-orbit vmap-of-solve — the same
  shared-vs-per-orbit trade-off as G8's galax lines.

* **CUDA kernel** (reddish purple: TITAN Xp circles, H100 dark-purple diamonds). This
  is not a galpy backend. It is an experimental CUDA port of galpy's C forces and
  `dop853`, one thread per orbit (`experiments/cuda_orbits`), shown for scale: what
  GPU-native kernels would buy. **Solid**: galpy's own arithmetic (the GSL incomplete
  gamma and libm `pow`), with one out-of-line force function and a persistent-thread
  kernel that hands a finished lane the next orbit. **Thin, faint long-dashed (unlabelled)**: the same with
  division- and `pow`-free special functions and a cost-sorted schedule. Both use the
  same orbit, tolerance (1e-8) and 16 saves as the hiN points. Wall time includes the
  host↔device copies (pinned, reused buffer) and, for the fast line, the host argsort;
  one-time nvcc compilation is excluded. Both agree with `dop853_c` to within its own
  1e-8 truncation error: over 300 orbits, the maximum relative position difference
  after 2 Gyr is 1.6e-6 (galpy arithmetic) and 5.3e-6 (fast). galpy at 1e-8 differs
  from galpy at 1e-13 by up to 1.4e-5.
  On the H100 at N=10⁶ the kernel costs **2.36 µs/orbit** (galpy arithmetic) and
  **1.04 µs/orbit** (fast). That is 7× / 16× less than galpy's jax Dopri8 on the same
  GPU, 4× / 9× less than torch dopri5, 34× / 76× less than the 72-thread C node, and
  ~1 000× / ~2 250× less than one core. On the TITAN Xp it costs 49 / 15.6 µs/orbit.
  Measured down to N = 1: one orbit takes 45 ms on the H100 and 0.18 s on the TITAN with
  galpy arithmetic (fast: 20 and 65 ms). The wall time stays near 0.1 s (H100) or 0.4 s
  (TITAN) up to N ≈ 10⁴, because a launch lasts as long as its slowest orbit. The H100
  kernel therefore overtakes one-core C at N ≈ 50 and matches the 72-thread node at
  N ≈ 10³ (0.11 vs 0.12 s). The N < 10³ points come from separate low-N scans (H100: N = 1–3×10⁵; TITAN:
  N = 1–3×10⁴), run without the cost-sorted schedule. Where N
  overlaps, the high-N scans are plotted. For the fast line the two schedules differ by up
  to ~20% in the overlap: the sort costs a host argsort at small N and pays off at large N.
  These kernels are not differentiable.

**Same-work / work counts** (`measure/work_counts.py` → `data/work_counts.json`), for
one Sun-like orbit at `rtol=atol=1e-8` (accepted steps / RHS evals = nfev):
jax `Dopri8` **189 / 3056** · jax `Dopri5` **545 / 3547** ≈ torch `dopri5`
**~578 / 3470**. So jax-Dopri5 and torch-dopri5 are **genuinely same-work** (matching
step + nfev counts) — any wall-time gap between *those two lines* is pure backend
kernel/dispatch + controller architecture, not the method; while Dopri8 takes ~3×
fewer (bigger) steps with ~13 vs ~6 evals/step, netting a slightly *lower* nfev.

**Compile amortization** (the in-backend GPU lines pay a one-time XLA compile): the
first batched call costs ~5 s at small N growing to ~25 s at N=1e5 (the compile is
excluded from the plotted steady-state wall time and recorded as `compile_run_s` in
the JSON), then every later call runs the cached executable at the steady-state cost
— so the GPU only pays off when the same jitted solve is reused across many batches.

**Potential support:** as of the backends worktree (#949, 36c92e8f),
MWPotential2014 (PowerSphericalCutoff bulge + MiyamotoNagai disk + NFW halo) is
fully in-backend-integrable on both jax and torch — the earlier "Plummer/Isochrone
only" status no longer holds, so this figure uses the same MW model as G8.

All GPU/CPU measurements on the NVIDIA TITAN Xp (sm_61), jax 0.10.1 + diffrax,
torch 2.12.0+cu126 + torchdiffeq, against the backend-migrated galpy worktree
(`/var/tmp/backends-merge`, feat/backends @36c92e8f), which now ships `libgalpy` so
the C baseline and the in-backend jax/torch lines all run from the **same galpy**.
Reproducibility: the shared galpy series (C, JAX-CPU-Dopri8, JAX-GPU-Dopri8) come
from the `paper/figures/_shared/orbit_speed/` scripts (`galpy_c_measure.py`,
`galpy_jax_measure.py` for {CPU,GPU}×{lowN dense, hiN}) and are shared with G8. The
G12-local series are `measure/jax_dopri5_measure.py` (the same-work JAX-Dopri5 GPU
line, {lowN,hiN}), `measure/torch_measure.py` (torch `dopri5`, out to N=1e6 — run with
no jax import + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` so the full 1e6
batch fits in 12 GiB, no OOM), and `measure/work_counts.py` (the step/nfev counts in
`data/work_counts.json`). GPU jobs are run **strictly serially** (one device-timed
process at a time, GPU verified idle via `nvidia-smi` between each). Each writes a
machine- and version-stamped `data/*.json`; the `make.py` plot is byte-stable from
those JSONs and `verify` re-renders only the plot.
