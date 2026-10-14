**Orbit-integration throughput: galpy (C / JAX CPU & GPU) vs galax (JAX, CPU & GPU),
at matched solver settings.**
MWPotential2014 / BovyMWPotential2014, a Sun-like orbit integrated for 2 Gyr over
a batch of N initial conditions; wall time for the whole batch vs N. All lines
use `rtol=atol=1e-8` (the galax `OrbitSolver` default; galpy's C and in-backend
integrators are called with the same tolerances instead of their tighter 1e-10
default) and diffrax `Dopri8` / `dop853_c`, so solver class, tolerance, orbit, and
potential are matched. The galpy series (C, JAX-CPU, JAX-GPU) are loaded from the
shared `paper/figures/_shared/orbit_speed/data` and are **byte-for-byte identical
to G12's**, so the two figures can be read on a common reference. nt convention
(shared): lowN points save 1000 dense steps (small-N crossover); high-N GPU points
save 16 (memory) — integration is adaptive-step-bound not save-bound, so per-orbit
cost matches and the overlap points agree.

galpy's compiled `dop853_c` is drawn twice, because its C orbit loop is
parallelised with OpenMP over the orbits (`omp_get_max_threads()`; the `numcores`
keyword does not reach it): **on one core** (`OMP_NUM_THREADS=1`, pinned with
`taskset`; grey, solid) it costs 2.2–2.4 ms/orbit at every N ≥ 100 (2.3 s at
N=10³, 22 s at 10⁴, 716 s at 3×10⁵), and **on the whole node** (unpinned: all 72
hardware threads of the dual 18-core Xeon E5-2697 v4; light grey, dash-dot)
~0.1 ms/orbit once past its ~3 ms per-call setup, falling to 0.084 ms/orbit at
N=3×10⁵ (25 s) and 0.079 ms/orbit at N=10⁶ (79 s) — a ~28× parallel speed-up.
The GPU backends start from ~1–2 s of compile/kernel-launch overhead and
amortise it with N; all GPU lines end at N=3×10⁵. On the **TITAN Xp** (a consumer
GPU that runs float64 at 1/32 of its float32 rate) they reach 0.215 ms/orbit
(galpy in-backend, per-orbit steps; 64 s), 0.209 ms/orbit (galax, per-orbit
steps; 63 s) and **0.068 ms/orbit (galax, batched steps; 20 s)**: all three
overtake one-core C between N≈500 and 10³ and are 11–35× faster per orbit than
it at N=3×10⁵, whereas none of them beats the 72-thread node (re-measured
2026-10-13): galax's batched line comes closest (2.8 vs 1.9 s at N=3×10⁴, 8.0 vs
6.4 s at 10⁵, 20 vs 19 s at 3×10⁵) and the two per-orbit TITAN lines stay ~3.3×
slower than the node. galax per-orbit / galpy (re-measured back to back on the
TITAN 2026-10-14): 0.96, 0.95, 0.72, 0.70, 0.75 at N=1–10⁴ (low-N scan) and 0.63,
0.77, 0.93, 0.97 at N=10³–3×10⁵ (high-N scan); on the H100 1.09, 1.37, 1.12, 1.14,
1.08 and 1.15, 1.10, 0.87, 1.01. So the two agree to ≲10% only at N≥10⁵, where the
integration dominates. On an
**NVIDIA H100** (Trillium; galpy and galax both measured there; galpy on jax
0.11.2) every GPU line crosses below one-core C just under N=10³ and below the
72-thread node at N≈1–3×10⁴, and at N=3×10⁵ reaches **0.0247 ms/orbit for
galpy's per-orbit integrator (7.4 s)** — ~97× one-core C, ~2.6× the 72-thread
node and ~9× the TITAN Xp — 0.0246 ms/orbit for galax per-orbit (7.4 s, equal
to galpy) and 0.0130 ms/orbit for galax batched (3.9 s, ~5× the node).

**CUDA kernel** (reddish purple: TITAN Xp circles, H100 dark-purple diamonds). This
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
The lines show the register kernel (measured from N = 1 to 10⁷). Keeping ten of dop853's stage vectors in shared memory (168 registers, 19% occupancy) is 1.3× faster on the H100 at N = 10⁶ (1.81 / 0.69 µs/orbit), measured at 10⁵–10⁶ only. galpy's sixth-order symplectic integrator with one common step for the whole batch reaches the same worst-case accuracy over the 300 reference orbits with 15% fewer force evaluations and runs at 0.78 / 0.25 µs/orbit (galpy / fast arithmetic, H100, N = 10⁶): 3.0–3.6× the dop853 kernel. That holds because these near-solar disc orbits in a smooth potential share a dynamical timescale; a batch mixing very different timescales would have to step at its fastest orbit's rate.
These kernels are not differentiable.

Colours follow the paper-wide convention (galpy C grey — one core dark grey,
the 72-thread node light grey dash-dot — jax blue — sky blue on the CPU, darker on
the GPUs, navy for the H100 — galax green); all jax/galax lines are jit-compiled
(solid), galax's per-orbit control is dash-dotted. The individual lines:

* **galpy (JAX, CPU; per-orbit steps)** — galpy's in-backend diffrax integrator on
  the CPU: the framework/vectorisation cost without the device, the common galpy
  reference shared with G12, measured on ONE core (pinned with `taskset`, the only
  way to keep XLA:CPU's thread pool to one core; 2026-10-08). 2.0 ms/orbit at
  N=10³ — slightly *cheaper* than one-core C (2.3) — rising to 2.6 at 10⁴, 3.6 at
  10⁵ and 4.6 ms/orbit at N=3×10⁵ (1390 s, 1.9× one-core C). **galax (JAX, CPU;
  batched steps)**, also on one core: 1.4 ms/orbit at 10³ (cheaper than one-core
  C), 1.7 at 10⁴, 3.0 at 10⁵ (297 s, 1.2× one-core C; re-measured 2026-10-14). (Unpinned, with up to 72
  threads, both were 1.5–4× *slower* than on one core up to N ≈ 10⁴ — XLA's
  thread pool kept only ~1.1–1.3 cores busy — and faster, by 1.4–1.7×, only at
  their largest N: galpy at 3×10⁵, galax at 10⁵.)
* **galpy (JAX, GPU; per-orbit steps)** — the same in-backend diffrax integrator of
  the backend-agnostic forces (`jax.vmap` of a per-orbit solve, jit-fused) on the
  GPU; every orbit carries its own adaptive `PIDController`. ~0.25 ms/orbit at N=1e5,
  ~0.21 at 3e5.
* **galax (JAX, GPU; per-orbit steps)** — the *apples-to-apples* control: galax's
  `OrbitSolver` vmapped over single-orbit solves, i.e. the same stepping
  semantics as galpy (not how galax is normally used). 0.230 ms/orbit at N=1e5,
  0.209 at 3e5 on the TITAN Xp — **equal to galpy to within ~3–7%** (and to
  <1% on the H100, 0.0246 vs 0.0247 ms/orbit at 3e5): at matched semantics there
  is no framework gap between the two codes.
* **galax (JAX, GPU; batched steps)** — galax's native mode: the whole batch is
  one ODE system with a *single shared* `PIDController` (one global RMS error
  norm → one dt sequence for all orbits). ~0.07–0.08 ms/orbit at N≥1e5, i.e.
  ~3× faster than either per-orbit line. The speed comes from the shared/diluted
  error control: per-orbit error is only guaranteed in the batch-RMS sense, and
  an orbit's trajectory depends (at the tolerance level, ~1e-6) on what else is
  in the batch. galpy deliberately keeps per-orbit stepping for its in-backend
  integrator (per-orbit error guarantees, batch-independent results).

Both GPU per-orbit lines and galpy's in-backend orbits match the C integrator to
~1e-9 over 2 Gyr. NOTE on torch: batched `torchdiffeq` runs *eager* (no jit-fuse,
unlike jax/diffrax), so it is Python-overhead-bound and GPU-starved — impractical
for large batches, hence not shown here (G12 shows it explicitly, out to the
eager overhead→compute transition).

All measurements on the NVIDIA TITAN Xp (sm_61), except the H100 lines (galpy: `_shared/orbit_speed/data/h100/`; galax: `g8_galax_orbit_speed/data/h100/`). Reproducibility: the galpy series
(C, JAX-CPU, JAX-GPU) come from the shared measure scripts under
`paper/figures/_shared/orbit_speed/` (`galpy_c_measure.py`, `galpy_jax_measure.py`),
run against the backend-migrated galpy worktree (`/var/tmp/backends-merge`,
feat/backends @36c92e8f), which now ships `libgalpy` so the C baseline and the
in-backend jax lines run from the **same galpy**; galax is measured in its own
uv-locked venv with a matched CUDA-12 plugin (`galax_perorbit_measure.py`,
`galax_extended.py` — see ../_setup/galax_gpu_setup.md). Each writes a per-backend,
machine-stamped JSON; the plot is byte-stable from those. The full gap decomposition
(tolerance defaults ×1.5, batched stepping ×2.9, everything else ≈×1.0 — incl. proof
that XLA CSE dedupes galpy's duplicate `gammainc` and that backend dispatch is jitted
away) is in logs/gpu_speed_galpy_vs_galax/log.md.
