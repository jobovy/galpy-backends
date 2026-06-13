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

* **galpy (C, CPU; sequential)** — the compiled `dop853_c` integrator, one orbit at
  a time in a single-threaded C loop. ~zero startup, constant per-orbit cost
  (~2.1 ms/orbit at 1e-8), so it wins decisively at small N — a single orbit is
  ~3 ms vs the GPU lines' ~1.5 s compile/launch-bound floor — but never amortises.
* **galpy in-backend (JAX, GPU)** — diffrax `Dopri8` of the backend-agnostic forces,
  `jax.vmap`-ed over the batch and jit-fused, on the NVIDIA TITAN Xp (sm_61). The
  vectorised solve amortises across the batch down to ~0.25 ms/orbit at N=1e5 and
  ~0.21 at 3e5 (~10× faster per orbit than galpy-C); crosses galpy-C around
  N≈few-hundred.
* **galpy in-backend (JAX, CPU)** — the identical diffrax path forced onto the CPU,
  isolating the device contribution from the framework/vectorisation cost
  (~5–8 ms/orbit).
* **galpy in-backend (torch, GPU; eager)** — the torchdiffeq `Dopri5` path (state
  carried as (6, N) so one `odeint` vectorises the batch). torchdiffeq runs **eager**
  — there is no jit-fuse of the adaptive-step loop. At small N the batch is
  *overhead-bound*: the whole batch takes a roughly *constant* ~12 s set by the
  thousands of Python-driven steps, the GPU is starved, and ms/orbit falls as ~1/N.
  Pushed to large N the line crosses into the *compute-bound* regime — the per-step
  GPU kernels finally dominate the Python overhead, wall time starts rising
  ~linearly, and ms/orbit flattens to torch's true per-orbit asymptote. It is correct
  and fully differentiable (the in-backend tests pass against the C integrator and
  finite-difference gradients) but ~40× slower per orbit than the jit-fused JAX path
  — exactly why G8 omits a torch line; here it is shown out to the OOM/time cap to
  make the eager overhead→compute transition visible.

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
Reproducibility: the shared galpy series (C, JAX-CPU, JAX-GPU) come from the
`paper/figures/_shared/orbit_speed/` scripts (`galpy_c_measure.py`,
`galpy_jax_measure.py` for {CPU,GPU}×{lowN dense, hiN}) and are shared with G8;
torch is measured by `measure/torch_measure.py`. GPU jobs are run **strictly
serially** (one device-timed process at a time). Each writes a machine- and
version-stamped `data/*.json`; the `make.py` plot is byte-stable from those JSONs and
`verify` re-renders only the plot.
