# M3: make galpy `streamspraydf` fully jittable (jax) + GPU — implementation plan

Goal: `jax.jit(lambda theta: build_spdf(theta).sample(n))` and `.streamTrack()` run
under `jax.jit` on CPU+GPU, differentiable w.r.t. potential params AND progenitor ICs,
with `xv(theta)` flowing (generative). Keep the numpy path BYTE-IDENTICAL and torch working.

Prototype that proves the target end-to-end: `m2_generative_prototype.py` (+ multiparam,
fit). The jax-native recipe below is lifted from it.

## Prereqs (done)
- `o.x(t)`/`R(t)`/... jittable — committed **f7ac8b5d**. Read `o.orbit` directly when
  integrating on the grid you need; `o.x(t)` now also works under jit (off- and on-grid).
- Detection helper `_backend_sampling()` (forced-numpy force probe to find a GENUINE
  backend param, not a forced context) — committed WIP **38546eff** (eager routing is
  superseded; reuse the detection).

## The three barriers + fixes

### B1. Construction re-integrates the progenitor with the C integrator (streamspraydf.py:143)
`self._progenitor.integrate(self._progenitor_times, self._pot)` uses the default
(C) integrator → breaks on a traced amp (can't put a tracer through ctypes).
FIX: when `_pot` has a genuine backend param OR the progenitor IC is a backend array,
build the progenitor via the in-backend ODE: construct `Orbit(<backend IC>)` and
`integrate(..., method="diffrax"/"torchdiffeq")`. IC must be a backend array for diffrax.
Gate on detection so numpy construction is untouched.

### B2. Sampling numpy islands (`_sample_tail`, `_setup_rot`, `_calc_rtide`/`_calc_vc`, `spray_df`)
- `as_numpy(dt)` + querying the progenitor at numpy times → replace with backend `dt`
  and query the backend progenitor via `o.x(t)` (now jittable) or read `o.orbit` on the
  stripping-time grid.
- `_calc_rtide`/`_calc_vc` try/except numpy-island fallback → under jit the backend path
  (rtide/evaluateRforces on backend inputs) succeeds; ensure the except path is not
  traced (it can't be — it's a fallback for unmigrated potentials; jit requires a
  migrated potential).
- Reparameterized noise: `numpy.random.*` → `jax.random` with a key (already seamed via
  `key=` in `sample()`), so draws are theta-independent constants under jit.
- Sample-orbit integration: route to `method="diffrax"` (per-particle 2-point grid
  `ts=(N,2)` works and is jittable — validated).

### B3. Reconstruction (`streamTrack.py::_fit_track_from_particles`) — cKDTree + make_smoothing_spline
Not jittable (scipy on concrete values). Add a jax-native path (dispatch when the
particles/curve are backend arrays under jit):
- closest-point: brute-force `argmin` of `||xv - curve||^2` over the curve grid with the
  arm mask (`d2 = where(t<=0, d2, inf)`), `stop_gradient` the index.
- binning/smoothing: **raw-data P-spline** `s = (B'B + λ D'D)^-1 B'off`, B = linear-interp
  design (particle tp → nodes), λ via GCV grid-search on the raw residual (N >> NNODE so
  GCV finds a real λ; galpy's make_smoothing_spline is not jittable and its GCV-on-binned-
  means degenerates — DON'T bin-then-GCV). `stop_gradient` B and λ; offsets flow.
- covariance/track-width (optional, second pass): jnp.linalg.eigh + clip for the nearest-
  PSD projection (frozen eigenvectors); mirrors the existing frozen-operator design.

## Bound the extent (important)
Full tdisrupt (~20 orbital periods) → chaotically sensitive → rugged fit landscape.
Bound the reconstructed track to the coherent recent extent (galpy already estimates
`track_time_range`); the fit demo only converges on a younger/bounded stream.

## Validation (per [[stringent-gradient-tests]] + [[stream-track-two-gradients]])
- grad-vs-FD must **h-converge** (streams are sensitive; single-h FD is nonlinear garbage
  at h=1e-4, linear ~1e-5). Test a spread of h and take the min REL.
- numpy byte-identity for every numpy-path change (gate all jit paths on backend detection).
- jax==torch value/grad parity; GPU run (device-agnostic).
- per-module `tests/test_backend_streamspraydf.py` (physical generative grad-vs-FD, jit, GPU).

## Sequencing
1. B1 (construction) — smallest, unblocks everything; test `jax.jit(sample)` compiles.
2. B2 (sampling) — get `sample()` fully jittable + multiparam grad-vs-FD (h-converged).
3. B3 (reconstruction) — jax-native path; `streamTrack()` jittable end-to-end.
4. Extent bound + fit test; tests; ledger.

Risk: touches `__init__` + hot sampling paths + the 2500-line streamTrack.py. Do
incrementally, commit per barrier, verify numpy byte-identity + a torch smoke each step.
