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

galpy's compiled `dop853_c` (single-thread C, `numcores=1`) costs ~0.1 ms/orbit
once past its ~3 ms per-call setup, falling to **0.084 ms/orbit at N=3e5 (25 s)
and 0.079 ms/orbit at N=10⁶ (79 s)**, its largest N. It is the fastest line up to
N≈10⁴. The GPU backends start from ~1–2 s of compile/kernel-launch overhead and
amortise it with N; all GPU lines end at N=3×10⁵. On the **TITAN Xp** (a consumer
GPU that runs float64 at 1/32 of its float32 rate) they reach 0.215 ms/orbit
(galpy in-backend, per-orbit steps; 64 s), 0.209 ms/orbit (galax, per-orbit
steps; 63 s) and **0.068 ms/orbit (galax, batched steps; 20 s)**: galax's batched
line is level with single-thread C from N=3×10⁴ to 10⁵ (2.8 vs 3.1 s, 8.0 vs 8.1 s)
and faster at 3×10⁵ (20 vs 25 s), while the two per-orbit lines stay ~2.5× slower
than C. On an **NVIDIA H100** (Trillium; galpy and galax both measured there;
galpy on jax 0.11.2) every GPU line crosses below C at N≈1–3×10⁴ and at N=3×10⁵ reaches **0.0247 ms/orbit for galpy's
per-orbit integrator (7.4 s)** — ~9× the TITAN Xp and ~3.4× faster than
single-thread C — 0.0246 ms/orbit for galax per-orbit (7.4 s, equal to galpy) and
0.0130 ms/orbit for galax batched (3.9 s, ~6× faster than C).

Colours follow the paper-wide convention (galpy C grey, jax blue — sky blue on the
CPU, darker on the GPUs, navy for the H100 — galax green); all jax/galax lines are
jit-compiled (solid), galax's per-orbit control is dash-dotted. The individual
lines:

* **galpy (JAX, CPU; per-orbit steps)** — galpy's in-backend diffrax integrator on
  the CPU: the framework/vectorisation cost without the device, the common galpy
  reference shared with G12 (~3–10 ms/orbit, 2.8 at N=3e5; isolates the GPU
  contribution below).
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
