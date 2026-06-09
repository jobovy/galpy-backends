**Orbit-integration throughput: galpy (C / JAX-GPU) vs galax (JAX, CPU & GPU),
at matched solver settings.**
MWPotential2014 / BovyMWPotential2014, a Sun-like orbit integrated for 2 Gyr over
a batch of N initial conditions; wall time for the whole batch vs N. All GPU lines
use diffrax `Dopri8` with `rtol=atol=1e-8` (the galax `OrbitSolver` default;
galpy's in-backend integrator is called with the same tolerances instead of its
tighter 1e-10 default), so solver, tolerance, orbit, and potential are matched.

galpy's compiled `dop853_c` (single-thread C loop) has ~zero startup and wins
decisively at small N — a single orbit takes ~10 ms vs ~1.6–2 s of
compile/kernel-launch overhead for the GPU backends — but its per-orbit cost is
constant. The GPU backends amortise the vectorised solve across the batch and
cross galpy-C around N≈200–300:

* **galpy (JAX, GPU; per-orbit steps)** — galpy's in-backend diffrax integrator
  of the backend-agnostic forces (`jax.vmap` of a per-orbit solve, jit-fused);
  every orbit carries its own adaptive `PIDController`. 0.244 ms/orbit at N=1e5,
  0.211 at 3e5.
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
for large batches, hence not shown. The small-N points (galpy-C, galax-CPU, and
N≤1e4 on the GPU lines) save 1000 dense points; the high-N GPU points save 16
(memory) — orbit integration is adaptive-step-bound, not save-bound, and the
overlap points agree.

All measurements on the NVIDIA TITAN Xp (sm_61). Reproducibility: the backends
are measured in separate environments (galpy in the main py313 env; galax in its
own uv-locked venv with a matched CUDA-12 plugin — see ../_setup/galax_gpu_setup.md),
so the measure step is the standalone scripts under measure/ (`*_tol8.py`,
`galax_perorbit_measure.py` for the matched-tolerance/per-orbit lines), which
write the per-backend, machine-stamped data/*.json; the plot is byte-stable from
those. The full gap decomposition (tolerance defaults ×1.5, batched stepping
×2.9, everything else ≈×1.0 — incl. proof that XLA CSE dedupes galpy's duplicate
`gammainc` and that backend dispatch is jitted away) is in
logs/gpu_speed_galpy_vs_galax/log.md.
