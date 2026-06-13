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
* **galpy in-backend (JAX, GPU; Dopri8, shipped)** — diffrax `Dopri8` (8th order) of
  the backend-agnostic forces, `jax.vmap`-ed over the batch and jit-fused, on the
  NVIDIA TITAN Xp (sm_61). This is **what galpy ships on jax**. The vectorised solve
  amortises across the batch down to ~0.25 ms/orbit at N=1e5 and ~0.21 at 3e5
  (~10× faster per orbit than galpy-C); crosses galpy-C around N≈few-hundred.
* **galpy in-backend (JAX, GPU; Dopri5, same-work)** — diffrax `Dopri5` (5th order)
  through the **same** backend forces, jit+vmap, GPU. galpy ships Dopri8 on jax but
  `dopri5` on torch, so the shipped jax-vs-torch crossover is **not same-work**; this
  line **is** the apples-to-apples match (same 5th-order method, matching nfev/steps —
  see work counts below). Cheaper-per-step Dopri5 reaches ~0.13 ms/orbit at N=3e5,
  *beating* the shipped Dopri8 line on the GPU despite taking ~3× more steps.
* **galpy in-backend (JAX, CPU; Dopri8)** — the shipped diffrax `Dopri8` path forced
  onto the CPU, isolating the device contribution from the framework/vectorisation
  cost (~5–8 ms/orbit).
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
