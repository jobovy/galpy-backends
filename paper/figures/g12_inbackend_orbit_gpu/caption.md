**galpy's own in-backend differentiable orbit integrator: GPU vs CPU vs compiled
C.** The analogue of G8 (which compared galpy to galax), but for galpy's #906
in-backend integrator (`galpy.backend._reference.integrate_orbit`): the same
MWPotential2014, Sun-like orbit integrated 2 Gyr over a batch of N initial
conditions, wall time for the whole batch vs N. All lines use the *same*
backend-agnostic galpy forces, `rtol=atol=1e-8`, and per-orbit adaptive stepping.

* **galpy (C, CPU; sequential)** — the compiled `dop853_c` integrator, one orbit at
  a time in a single-threaded C loop. ~zero startup, constant per-orbit cost
  (~7 ms/orbit), so it wins decisively at small N — a single orbit is ~10 ms vs the
  GPU lines' ~1.5 s compile/launch-bound floor — but never amortises.
* **galpy in-backend (JAX, GPU)** — diffrax `Dopri8` of the backend-agnostic forces,
  `jax.vmap`-ed over the batch and jit-fused, on the NVIDIA TITAN Xp (sm_61). The
  vectorised solve amortises across the batch down to ~0.24 ms/orbit at N=1e5
  (~30× faster per orbit than galpy-C); crosses galpy-C around N≈few-hundred.
* **galpy in-backend (JAX, CPU)** — the identical diffrax path forced onto the CPU,
  isolating the device contribution from the framework/vectorisation cost.
* **galpy in-backend (torch, GPU; eager)** — the torchdiffeq `Dopri5` path (state
  carried as (6, N) so one `odeint` vectorises the batch). Shown only for a few
  small N: torchdiffeq runs **eager** — there is no jit-fuse of the adaptive-step
  loop, so the whole batch takes a roughly *constant* ~12 s (the cost of the
  thousands of Python-driven steps), GPU-starved. Per-orbit cost therefore only
  reaches ~1.2 ms at N=1e4, still ~4× slower than the jit-fused JAX-GPU line and
  ~40× more wall time. This is correct and fully differentiable (the in-backend
  tests pass against the C integrator and finite-difference gradients), but it is
  not the batched-throughput path — exactly why G8 omits a torch line.

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
(`/var/tmp/backends-merge`, feat/backends @36c92e8f). Reproducibility: like G8, the
backends are measured by the standalone scripts under `measure/` (`jax_measure.py`
for {CPU,GPU}×{lowN dense, hiN}, `torch_measure.py`, and `galpy_c_measure.py` in the
C-built main env), each writing a machine- and version-stamped `data/*.json`; the
`make.py` plot is byte-stable from those JSONs and `verify` re-renders only the plot.
