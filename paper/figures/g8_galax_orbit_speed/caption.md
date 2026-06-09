**Orbit-integration throughput: galpy (C / JAX-GPU) vs galax (JAX, CPU & GPU).**
MWPotential2014 / BovyMWPotential2014, a Sun-like orbit integrated for 2 Gyr over
a batch of N initial conditions; wall time for the whole batch vs N.

galpy's compiled `dop853_c` (single-thread C loop) has ~zero startup and wins
decisively at small N — a single orbit takes ~10 ms vs ~1–2.6 s of
compile/kernel-launch overhead for the GPU backends — but its per-orbit cost is
constant. The GPU backends amortise the vectorised solve across the batch and
cross galpy-C around N≈300:

* **galpy (JAX, GPU)** — galpy's own in-backend diffrax integrator of the
  backend-agnostic forces (`jax.vmap`, jit-fused into one XLA graph). Down to
  ~0.31 ms/orbit at N=3×10⁵ — ~15–20× faster than galpy-C. This is now possible
  for the *full* MWPotential2014 because its bulge
  (`PowerSphericalPotentialwCutoff`) was migrated to be jit-clean.
* **galax (JAX, GPU)** — the modern `OrbitSolver` (Dopri8); down to ~0.07
  ms/orbit at N=3×10⁵, ~5× faster than galpy-jax-GPU (a more specialised field).

Both GPU backends, and galpy's in-backend orbits, match the C integrator to
~1e-9 over 2 Gyr. NOTE on torch: single orbits run on the GPU, but batched
`torchdiffeq` runs *eager* (no jit-fuse, unlike jax/diffrax), so it is
Python-overhead-bound and GPU-starved — impractical for large batches, hence not
shown. The small-N points (galpy-C, galax-CPU, and N≤10³ on the GPU lines) save
1000 dense points; the high-N GPU points save 16 (memory) — orbit integration is
adaptive-step-bound, not save-bound, so the per-orbit cost is the same and the
overlap points agree.

All measurements on the NVIDIA TITAN Xp (sm_61). Reproducibility: the backends
are measured in separate environments (galpy in the main py313 env; galax in its
own uv-locked venv with a matched CUDA-12 plugin — see ../_setup/galax_gpu_setup.md),
so the measure step is the standalone scripts under measure/, which write the
per-backend, machine-stamped data/*.json; the plot is byte-stable from those.
