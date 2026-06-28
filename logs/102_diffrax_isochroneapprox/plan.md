# #102 — IsochroneApprox via diffrax (GPU + higher-order / twice-differentiable AA)

**Base:** branch off `feat/backends` AFTER #1027 (feat/inbackend-batch) merges (uses the batched
in-backend diffrax/torchdiffeq path). Memory: `diffrax-higher-order-aa-blockers` — C-STM is
first-order-only (pure_callback has no JVP); diffrax for AA needs adjoint + bounded max_steps +
a twice-differentiable pipeline, not a config flip.

## Why C-STM can't do this
actionAngleIsochroneApprox default `_integrate_method="dopr54_c"`
(actionAngleIsochroneApprox.py:108). All orbit integrations (lines 781, 875, and the #1025 batched
path) go through `o.integrate(self._tsJ, pot=self._pot, method=self._integrate_method, dt=...)`.
`dopr54_c` → Orbit C-STM route → `jax.pure_callback`/`torch.autograd.Function` → FIRST-ORDER only +
CPU-only (C). So Hessian-of-actions and GPU are both impossible on the default path.

## In-backend path is twice-diff + GPU, but under-exposed
- `method="diffrax"` (jax) / `"torchdiffeq"` (torch) → `_integrate_inbackend` → `integrate_orbit`
  (inbackend_ode.py:132 `*, rtol=1e-12, atol=1e-12, max_steps=100000`) → `_jax/orbit_ode.py:34`
  `diffrax.diffeqsolve(term, Dopri8(), dt0=None, saveat, PIDController(rtol,atol), max_steps=...)`.
- `adjoint=` NOT set → diffrax default `RecursiveCheckpointAdjoint` (reverse-mode + higher-order
  safe; needs bounded max_steps — already bounded). Good default.
- GAP: `max_steps`, `solver`, `adjoint`, `dt0` are hardcoded / not threaded from the public
  `Orbit.integrate(...)` (Orbits.py `_integrate_impl`→`_integrate_inbackend(t,pot,method,rtol,atol)`).
  torch path (`_torch/orbit_ode.py` odeint dopri5) has NO max_steps.

## Twice-differentiability audit (explicit b)
integrate (diffrax) ✓ → getOrbit()/accessors ✓ (in-backend cubic Spline1D) → isochrone AA
(closed-form) ✓ → angle-average (algebraic) ✓ → angle-fit (xp.linalg.inv + sums) ✓. ONLY blocker:
`estimateBIsochrone` auto-b uses `_backend_brentq` with a `stop_gradient`'d bisection root + 1 Newton
step (backend/optimize.py, _jax/optimize.py) → kills 2nd derivs. With explicit `b=` (the test-suite
norm) the whole pipeline is twice-diff.

## Plan
1. **Thread the knobs.** Add `max_steps=`, `solver=` (string→diffrax/torchdiffeq solver map),
   `adjoint=` (jax), optional `dt0=` to `Orbit.integrate` → `_integrate_impl` → `_integrate_inbackend`
   → `integrate_orbit` → `diffeqsolve`. Keep numpy/C path byte-identical (new kwargs ignored unless
   in-backend method). Sensible defaults = today's (Dopri8/RecursiveCheckpointAdjoint/1e5).
2. **IsochroneApprox passthrough.** Accept `integrate_method="diffrax"/"torchdiffeq"` cleanly:
   handle `dt=self._integrate_dt` (adaptive → ignore/None for in-backend, don't error), and make the
   #1025 batched path route to in-backend (not force C-STM) when method is in-backend. Expose
   max_steps/solver/adjoint via constructor kwargs (store like `_integrate_dt`).
3. **estimateBIsochrone twice-diff.** Either (a) document/guard: auto-b is first-order only (warn),
   recommend explicit b for Hessians; or (b) make `_backend_brentq` twice-diff (drop stop_gradient via
   a proper implicit-function 2nd-order rule). Start with (a) (cheap, honest), file (b) as follow-up.
4. **Tests** (test_backend_actionAngle.py): IsochroneApprox + `integrate_method="diffrax"`/torchdiffeq
   — (i) value parity vs C-STM path + numpy; (ii) grad-vs-FD (1st order); (iii) **Hessian of actions
   wrt ICs/params** (jax.hessian / torch double-grad) finite & matches FD-of-grad (the new capability);
   (iv) max_steps too-low raises/handled; (v) GPU = Track-G follow-up (TITAN Xp sm_61 diffrax
   impractical; CPU twice-diff is the CI-checkable part).

## Risks
- diffrax IsochroneApprox is SLOW eager on CPU (10×Tdyn integration) → keep the test grid tiny; the
  win is GPU + Hessian, not CPU speed. Possibly slow_skip a heavy case under jax CI.
- Default max_steps=1e5 may be too low for 10×Tdyn at 1e-12 tol → may need a higher default for the
  AA path specifically (or raise tol). Measure before fixing.
