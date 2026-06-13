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

galpy's compiled `dop853_c` (single-thread C loop) has ~zero startup and wins
decisively at small N — a single orbit takes ~3–10 ms vs ~1.6–2 s of
compile/kernel-launch overhead for the GPU backends — but its per-orbit cost is
constant (~2.1 ms/orbit). The JAX backends amortise the vectorised solve across the
batch and cross galpy-C around N≈100–300:

* **galpy (JAX, CPU; per-orbit steps)** — galpy's in-backend diffrax integrator on
  the CPU: the framework/vectorisation cost without the device, the common galpy
  reference shared with G12 (~5–8 ms/orbit; isolates the GPU contribution below).
* **galpy (JAX, GPU; per-orbit steps)** — the same in-backend diffrax integrator of
  the backend-agnostic forces (`jax.vmap` of a per-orbit solve, jit-fused) on the
  GPU; every orbit carries its own adaptive `PIDController`. ~0.25 ms/orbit at N=1e5,
  ~0.21 at 3e5.
* **galax (JAX, GPU; per-orbit steps)** — the *apples-to-apples* control: galax's
  `OrbitSolver` vmapped over single-orbit solves, i.e. the same stepping
  semantics as galpy (not how galax is normally used). 0.232 ms/orbit at N=1e5,
  0.209 at 3e5 — **equal to galpy to within ~1–5%**: at matched semantics there
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

All measurements on the NVIDIA TITAN Xp (sm_61). Reproducibility: the galpy series
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
