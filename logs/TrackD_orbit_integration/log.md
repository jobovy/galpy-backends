# Track D — in-backend differentiable orbit integration (diffrax / torchdiffeq)

## Approach
Two independent routes to differentiable orbit integration (see `plan/trackD_autodiff.md`):
- **Path A — in-backend ODE integrators.** Integrate galpy's equations of motion *inside* the backend (diffrax for jax, torchdiffeq for torch) using the **backend-agnostic force evaluators** (`evaluateRforces`/`evaluatephitorques`/`evaluatezforces`), so autodiff falls out of the solver. Needs only migrated forces — **no C variational work** — and gives higher-order autodiff + parameter gradients for free. Also the independent correctness *reference* for path B.
- **Path B — C-STM forward sensitivity.** Wrap galpy's fast C integrator with `jax.custom_vjp` / `torch.autograd.Function`, using the variational state-transition matrix `M = ∂x₁/∂x₀`. Needs the **full variational PR** (Pvar-rhs + per-potential Hessians, Track B) → which is exactly why the Pvar-pot fan-out is being pushed.

This log covers the **Path A pilot** (2026-06-05). Reproducible code: `orbit_inbackend_pilot.py` (this dir).

The RHS is galpy's cylindrical `_EOM`, state `y = [R, vR, phi, Omega(=dphi/dt), z, vz]`:
```
dR=vR;  dvR = Lz^2/R^3 + F_R;  dphi=Omega;  dOmega=(phitorque - 2 R vR Omega)/R^2;  dz=vz;  dvz=F_z
```
reimplemented backend-agnostically (the force evaluators dispatch on the array type of R,z,phi). diffrax: `Dopri8` + `PIDController(rtol=atol=1e-11)`. torchdiffeq: `odeint(method="dopri8", rtol=atol=1e-11)`.

## Results (Plummer, T=8, dop853_c reference)
- **Trajectory parity vs galpy C integrator:** jax/diffrax and torch/torchdiffeq both match the C `dop853_c` final state to **~1e-8** in (R,vR,Omega,z,vz). The only apparent discrepancy is in `phi`: exactly **2π** — galpy wraps φ to [−π,π], the solvers leave it unwrapped (−1.0816 vs −1.0816+2π=5.2015). Same orbit; a wrapping convention, not an error.
- **Differentiability (jax.grad through the ODE solve vs central finite-difference):**
  - IC sensitivity `d(final R)/d(vR0)`: autodiff **−0.32048196** vs FD **−0.32048195** ✓
  - **Parameter gradient** `d(final R)/d(Plummer b)`: autodiff **−1.01760338** vs FD **−1.01760339** ✓

The parameter gradient is the headline: a potential-parameter gradient **backpropagated through a full adaptive ODE solve**. This is the capability path B will reproduce via the C STM once the variational track lands; here it already works through path A, and depends on the parameter-differentiability enabler (Track C / galpy PR #903 — backend arrays passed through the unit parser).

## Dead-ends / gotchas
- `phi` wrapping (2π) initially looked like a failure in the max-abs-diff; it's galpy's `o.phi()` wrap. Compare wrapped, or ignore φ mod 2π.
- diffrax/torchdiffeq are not default galpy deps — installed into the GPU env (diffrax 0.7.2, torchdiffeq 0.2.5, optax 0.2.8). Runtime extras, to be gated behind `galpy[diffrax]`/`galpy[torchdiffeq]`.
- Parameter gradients require the coordinates to also be backend arrays (so the namespace resolver follows the data) — same usage contract as the Track C param-grad figures (G3/G5).

## Formalized — galpy PR #906 (`galpy.backend._reference.integrate_orbit`)
Module `galpy/backend/_reference/inbackend_ode.py`: `integrate_orbit(pot, vxvv, ts)`, input/output in
the `Orbit` convention `[R,vR,vT,z,vz,phi]`, dispatching on the input namespace (jax→diffrax Dopri8,
torch→torchdiffeq). Tests (`tests/test_backend_inbackend_ode.py`, 8 pass under jax+torch, self-skip
without the ODE extras): C-vs-in-backend match (Plummer, Isochrone), grad-vs-FD (IC + parameter), 6×6
STM via jacrev vs FD-of-flow, torch-grad-vs-jax-grad. Stacked on #903 (parameter-grad enabler).

### Two backend ODE-autodiff findings (paper-worthy gotchas)
1. **diffrax `diffeqsolve` is a `custom_vjp`** — reverse-mode only. `jax.jacfwd` raises
   "can't apply forward-mode autodiff (jvp) to a custom_vjp function". So the planned `jacrev==jacfwd`
   STM cross-check is impossible as written; the STM is instead validated by jacrev vs
   finite-difference-of-the-flow (column by column). (A forward-mode STM would need
   `adjoint=diffrax.ForwardMode()`.)
2. **torchdiffeq's `dopri8` *backward* is inaccurate** (~1e-5 relative gradient error), while
   `dopri5`/`rk4` give ~1e-8. Measured for `d(final R)/d(vR0)`: torch-dopri8 ad=0.07185247 vs the
   true 0.07185299 (dopri5/rk4/FD and **jax/diffrax Dopri8 all agree** at 0.07185299). Cause: the
   adaptive step controller is detached on the backward pass, so the high-order dopri8 backward
   accumulates error the lower-order methods don't. **Fix: the torch path uses `dopri5`** → accurate
   gradients matching jax. diffrax's Dopri8 backward is fine — this is torchdiffeq-specific.
   - Corollary: a finite-difference reference computed *with* adaptive dopri8 jitters by ~1e-3 (the
     +/-eps solves pick different step sequences) even when the autodiff gradient is correct — so the
     torch test cross-checks against jax.grad, not against a torch-dopri8 FD.

### Figure G7 (`paper/figures/g7_orbit_sensitivity/`)
Orbit sensitivity (STM elements ∂x(t)/∂x₀ vs t, autodiff vs FD) + a gradient-descent orbit fit
recovering the launch velocity (vR0,vz0) through the solver. Dead-end: the GD must be bounded —
an unconstrained step can push the launch velocity into a near-radial plunge whose adaptive solve
exceeds diffrax's `max_steps` (EquinoxRuntimeError); fixed by clipping (vR0,vz0) to a sane box +
solver headroom. This is the Track G "G3 showpiece" (orbit fit) that was blocked on the integrator.

## Landed — Orbit.integrate wiring (PR-A #973) + accessors (PR-B #974) (2026-06-18)

### PR-A — `Orbit.integrate` routes to the in-backend ODE path (#973, merged 2026-06-18 → feat/backends)
`Orbit.integrate(method=...)` now selects the in-backend diffrax/torchdiffeq integrator when the
method is one of the in-backend names **and** the IC is a backend array (numpy ICs keep the exact C/
Python paths — public API and numpy results byte-identical). Generalizations beyond the #906 pilot:
- **All phase-space dimensions**, not just 6: the RHS is dimension-aware over `phasedim` 2/3/4/5/6
  (1D linear, 2D planar-axi, 3D, 4D planar-non-axi, 5D, 6D full), reusing `_EOM`/`_EOM_2d`-equivalent
  cylindrical RHS per dim. Parity ~1e-9 vs the matching C integrator across dims.
- **Concrete (non-forced) backend ICs run the numpy/C path** rather than raising — a jax/torch IC
  under no forced context still integrates; only the in-backend *method* names trigger the ODE route.
- **rtol/atol match the C defaults** (`1e-12`, the C `_parse_tol` default — not the pilot's 1e-11) — answers the review question: the
  in-backend tolerances now equal galpy's C-integration defaults, and the loose pilot test tolerances
  were tightened to match (except one Quantity test where `_TS*u.Gyr` parses to ~28× longer
  integration, kept at 1e-6/1e-7 with a logged justification).
- **Coverage:** closed the one missing line (inbackend_ode.py:84) flagged in review.
- Extras `galpy[diffrax]`/`galpy[torchdiffeq]` + a CI job that installs them so the in-backend code is
  actually exercised (no coverage hiding).
- **Follow-up ledgered:** ICs must be in *internal units* for in-backend tracing — needs clear docs
  (deferred, recorded for a docs PR).

### PR-B — backend-aware time-evaluation accessors + off-grid interpolation (#974, open → feat/backends)
After integrating in-backend, the derived `Orbit` accessors (`x/y/z/r/R/vR/vT/vx/vy/...`) must read
back the stored backend trajectory without detaching it. The accessors now dispatch through `xp`:
- `_resolve_accessor_namespace(thiso)` resolves the namespace **and** coerces `thiso` onto it
  (`xp.asarray`) — fixing a forced-suite crash where `xp.cos(thiso)` with a numpy `thiso` and a forced
  torch `xp` made torch reject the numpy input (jax silently coerces; torch raises). This was the one
  **real bug** the triage caught (I had initially mislabeled it a ledger flip).
- **Off-grid interpolation**: evaluating an accessor at times *between* the integration grid points now
  uses `galpy.backend.interpolate.Spline1D` (the same differentiable cubic spline the potentials use),
  so `o.R(t)` is differentiable in `t` and the ICs even off the stored grid — parity with the C/numpy
  interpolation at ~1e-9.
- **Aliasing fix** (user flagged "earlier instances in 892"): `getOrbit` and both `_call_internal`
  backend branches return a `_backend_safe_copy` — torch `.clone()` (graph-preserving, prevents a user
  mutating the stored orbit in place), jax left as-is (immutable). The numpy path keeps its existing
  `.copy()` semantics, so users still cannot inadvertently mutate an orbit's internal array.

### Potential backend-completeness (#975, open → feat/backends) — "are we done with potential?"
Audit answer: the compute core (forces/2nd-derivs/density) was already done; what remained were the
**convenience / functional-interface** methods that still used bare numpy + the scipy-backed
root-finds/quadratures. #975 closes them:
- Module-level `vesc/rtide/ttensor/tdyn` + `Potential` convenience methods → `xp`.
- `rl/rE/LcE/zvc/zvc_range` energy/circular-radius root-finds → new `galpy.backend.optimize.brentq`
  (numpy→`scipy.brentq`; jax/torch→bisection + one Newton reparameterization step for first-order
  gradients).
- `mass()` Gauss-theorem quadrature → `galpy.backend.quadrature` GL quad, with a **node-by-node mode**
  for scalar-only forces (e.g. `DoubleExponentialDiskPotential`) so `mass()` never feeds an array to a
  scalar-only `_Rforce` (the one real bug found in #975's first cut).
- `Force.rforce` base-class plumbing + `planarPotential` convenience → `xp`; dissipative-force backend
  cases deferred (scipy.special.erf on a backend array coerces to numpy + emits the numpy-2.0
  `__array_wrap__` DeprecationWarning the focused CI promotes to error — a separate dissipative
  migration concern, gated off here).
- numpy path byte-identical throughout; new backend tests for optimize/quadrature/convenience/
  planar/linear/force.

### Triage of the all-backend shard reds (both PRs)
The all-backend CI re-runs the full suite under a *forced* jax/torch backend with a checked-in xfail
ledger; `feat/backends` itself is green, so every new red is introduced by the PR. A migration that
swaps `numpy.sqrt(x)→xp.sqrt(x)` legitimately makes a method return a backend array under the forced
suite (the **intended** progress), which flips downstream numpy-only tests — those are **ledger flips**,
not regressions, and the fix is a ledger add, not code. The triage workflow classified every new red:
- **#974:** 1 real bug (the accessor crash above, fixed) + 16 ledger-flip adds (1808→1824).
- **#975:** ~6 real bugs (mass/scalar-only quad, radicand coercion, etc., fixed) + 50 ledger adds
  (1764→1815 [this PR's branch baseline]).
- Local representative shards re-run green under `-n auto` (72-core box): #974 `test_orbits`/
  `test_actionAngle` torch+jax 0-failed; #975 `test_potential` torch (80 pass) + jax (191 pass) 0-failed.
  jax shows extra XPASSes (jax is far more migrated than torch) — left under `strict=False`, to be
  pruned at the next milestone, not per-PR.

## Next
- Cross-validate the in-backend path against **path B (C-STM)** once #972's wrappers route into
  `Orbit.integrate` (PR-C, task #57): same `∂x(t)/∂x₀` from two independent mechanisms (g16 figure).
- Optimize `c_stm_forward` to a single batched C `integrate_dxdv` instead of 6 callback solves (#59);
  forward-mode C-STM via `custom_jvp` for single-direction IC sensitivities (#60).
- Docs: ICs-in-internal-units note for in-backend tracing (ledgered).
