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

### Opt PR: in-backend ODE native batch (#99) → #1027 (2026-06-28)
The diffrax(jax)/torchdiffeq(torch) in-backend ODE path was single-orbit-only (multi-orbit Orbit
raised). Made it a NATIVE batch: thread a leading N axis through the rectangular EOM on the
TRAILING component axis so N orbits integrate in ONE adaptive solve. The diffrax/torchdiffeq
analogue of #1025; unlocks IsochroneApprox-on-GPU + higher-order AA (#102).
- **Design+critique workflow (wf_3a45aefe) chose Seam A (native batch) over outer vmap**, empirically
  validated on the real Plummer force layer (batched==per-orbit 1.6e-10). Outer vmap rejected:
  torchdiffeq has NO native vmap (forces a Python loop), jax.vmap(diffeqsolve) re-traces.
- **4 single-orbit assumptions fixed** (all "first axis == the 6 components"): _eom_rhs positional
  unpack y[0..5]→y[...,k] + explicit `dim` arg (len(y) is the batch size once batched, not phasedim);
  _to_eom pd=len(vxvv)→vxvv.shape[-1], vxvv[...,i], stack axis=-1; field stacks axis=-1; _integrate_inbackend
  drops the guard + flattens raw _ic_backend (2,3,6)→(size,6) [critic caught: _ic_backend is stored RAW,
  not pre-flattened] + moveaxis (nt,size,pd)→(size,nt,pd). _from_eom was already batch-ready (ys[...,i]).
- **gotcha:** pd==5 branch `(genexpr)+(zero,)` → TypeError (can't concat generator+tuple); split to
  `R..vz=(vxvv[...,i] for i in range(5)); phi=zero`. Caught by the pd5 phasedim test.
- **numpy byte-identical** (numpy in-backend branch still raises; only backend-array ICs reach the code).
- **Adversarial diff-review workflow (wf_1d9d50e5) → 2 confirmed, both folded in:** (MAJOR) multi-orbit
  off-grid accessors (o.R(t_offgrid)/o(t)) hit _call_internal_backend_interp's `orb=self.orbit[0]`
  (single-orbit) → opaque reshape crash; added `if self.shape!=(): raise NotImplementedError` — ALSO
  fixes the pre-existing #1024 C-STM multi-orbit path. (MINOR) batch shares one max_steps → stiff orbit
  can exhaust it; docstring warning. Both have tests.
- **shared adaptive controller is the only semantic cost** and it's sub-tolerance: error norm (rms) over
  the whole (N,dim) state, so a much-stiffer orbit shrinks dt for all; verified sub-1e-10 even on
  adversarial heterogeneous batches; per-orbit time grids unsupported (raise) since saveat is shared.
- **speed:** batched cost ~flat in N (one controller; force evals broadcast). N=8 jax CPU 1.67s vs loop
  13.9s = **8.36×**, N=1 = 1.02× (no win), growing with N. GPU large-N win (one kernel vs N launches) is
  the real target but the dev TITAN Xp (sm_61) + diffrax XLA-compile makes a clean timing impractical →
  Track-G follow-up on better HW. 119 (integrate+inbackend_ode) + 62 (accessors+new) backend tests green.

#### off-grid eval for multi-orbit batches (#1027 follow-up commit 196363eb, 2026-06-28)
User: "1027 looks good but we need to make off grid evaluation work." The first commit guarded
multi-orbit off-grid accessors with NotImplementedError; replaced with a real implementation.
- _call_internal_backend_interp was single-orbit (`orb=self.orbit[0]`). Key simplification: the
  in-backend BATCH is ONE solve, so all orbits share ONE integration grid → spline each orbit's
  coords on that shared grid (in-backend differentiable cubic Spline1D, mode 2) and stack on the
  trailing orbit axis → (phasedim, nt_query, size), the SAME layout _call_internal's on-grid batch
  branch returns (permute_dims of (size,nt,pd)). Refactored the per-orbit spline body into
  _interp_one(orb)->(pd,nt_query); size==1 reproduces the prior (pd,nt,1) BYTE-IDENTICALLY (single-orbit
  accessor tests unchanged). Backward-integration grid flip handled once (shared grid) + each orb.
- cubic_spline_coeffs is 1D-y only (not batched) → loop over orbits (size Spline1D per coord); fine for
  an eager accessor call, differentiable w.r.t. ICs/params/eval-times. Vectorized batched-coeffs = future.
- So o.R(t)/o.x(t)/o.vR(t)/o.z(t)/o(t)/o.E(t) at OFF-GRID times now work for a batch: (N,nt) for array t,
  (N,) scalar; o(t)→(N,)-shape Orbit. ALSO fixes the pre-existing #1024 C-STM multi-orbit off-grid path.
  Verified: 6 accessors match per-orbit single solve 1e-6; 67 backend tests green (single-orbit
  byte-identical + new multi-orbit off-grid param tests). _integrate_inbackend docstring corrected
  (accessors work on+off grid; only in-place mutators flip(inplace)/SOS remain a follow-up).

#### per-orbit time arrays (#1027 follow-up commit 82a00cdb, 2026-06-28)
User: "in C you can actually have different time arrays all of the same length. We should also
allow that for in backend integrators. Used in streamspraydf." My #99 raised NotImplementedError on
2-D ts — too restrictive. Verified C truly supports this: integrate*Orbit_c has indiv_t (per-orbit t,
same nt). Now both modes:
- ts (nt,): shared grid -> ONE native-batch solve (fast #99 path, unchanged).
- ts self.shape+(nt,): per-orbit grid -> each orbit its own times/span + independent controller.
  jax: jax.vmap(_solve, in_axes=(0,0), out_axes=1) -> (nt,N,dim) [out_axes=1 keeps the saveat-first
  layout so the downstream moveaxis is uniform with the shared path]. torch: per-orbit odeint loop
  stacked dim=1 (torchdiffeq has NO vmap). -> (nt,size,pd) -> moveaxis -> (size,nt,pd).
- _integrate_inbackend: drop the raise, validate ts.shape[:-1]==self.shape (clear ValueError), flatten
  ts to (size,nt) [row-order pairs orbit i with ts row i, same as the ic flatten]. self.t stored 2-D
  -> _call_internal auto-routes accessors to the existing _call_internal_indiv_t (works for CONCRETE
  backend orbits via numpy.asarray+scipy; non-differentiable, but getOrbit() stays backend/differentiable).
- jax.vmap is VECTORIZATION not jit -> composes with the user's outer jit, respects no-internal-jit.
  The shared native batch is kept (not replaced by vmap) precisely because vmap re-traces per call-site;
  vmap is used ONLY for the per-orbit case where a single shared saveat is impossible. numpy untouched.
- Verified jax+torch: per-orbit batch & (2,2)-grid getOrbit match per-orbit single solves ~3e-11
  (different t0 AND t1 per orbit); gradient off-diagonal EXACTLY 0.0 (vmap independence -> no
  shared-controller cross-coupling, unlike the shared native batch's ~1e-13); shape-mismatch raises;
  concrete o.R([t_i])/o.R(scalar) work; shared 1-D path unchanged. 127 backend tests green. Enables
  the future streamspraydf backend migration (Track F). Adversarial diff-review wf_4142bbd8 running.

#### Augmented 42-state C-STM integrator (#100, PR #1028, 2026-06-28)
User: "Let's do 100 and 102 first." #100 = the true 6× C-STM forward. `c_stm_forward` (the
host-side forward the jax/torch orbit_stm wrappers call) used SIX separate 12-state
`integrate_dxdv` solves per orbit — each re-integrates the base orbit and re-evaluates the force +
3D Hessian. Replaced with ONE augmented 42-state solve: base (6) + six 6-D deviation columns (36).
- **C** (`integrateFullOrbit.c`): new `evalRectDeriv_stm` — 42-state RHS, term-for-term identical to
  the validated `evalRectDeriv_dxdv` but the base accel `a[0..5]`, the full 3D-Hessian scalars
  (`R2deriv..zphideriv`→`dFxdx..dFzdz`, conservative-only `RforceK/phitorqueK`) and the dissipative
  rectangular `calcRectDissipativeForceJacobian(jac_x,jac_v)` are computed ONCE/step, then the 6×6
  Jacobian `J=[[0,I],[K+dF/dx, dF/dv]]` is applied to each column k at offset o=6+6k. New
  `integrateFullOrbit_stm` = `integrateFullOrbit_dxdv` copy with dim=42 in all 4 switch cases.
- **Python**: `integrateFullOrbit_stm_c` packs `yo`(6)+`dyo_block`(36)→42 IC, `result (nt,42)`,
  argtypes mirror the dxdv wrapper. `c_stm_forward` rewritten: cyl→rect base + `jac0.T.reshape(-1)`
  seed (the six cyl basis deviations folded to rect = columns of the cyl→rect Jacobian), ONE
  augmented solve, rect→cyl base (Z/vz copy-guard for views), fold each rect deviation column back to
  cyl via `M[t]=numpy.linalg.solve(jac_t, dev[t].T)`, `M[0]=eye6` exact. Old six-solve kept as
  `_c_stm_forward_loop` (the bit-identity reference).
- **Speedup ~5.8×** (MWPotential2014, nt=1000: 0.250s→0.043s; ~6 fewer force/Hessian evals/step).
- **Tolerance:** for the ADAPTIVE methods (dop853_c/dopr54_c) the joint 42-state error norm accepts a
  slightly different step sequence than six 12-state solves → M matches `_c_stm_forward_loop` ~1e-10
  (amplified to ~1e-8 in the worst STM entry; gradient unchanged, well inside all test tols). Fixed
  step (rk4_c/rk6_c) bit-identical. numpy `Orbit.integrate` byte-identical (only the STM path changes).
  User decision (AskUserQuestion): adaptive default, accept ~1e-10.
- **Verified:** augmented-vs-loop parity over generic / turning-point-edge (vR=0/vz=0/z=0) / all-nonzero
  / phi=0 ICs (M[0]=I); FD-of-flow column-by-column conservative 3.15e-9 + **dissipative 1.65e-9**
  (MWPotential2014+Chandrasekhar — the non-symmetric velocity-coupled J); batch shape (N,nt,6,6). 7
  new tests green under `-W error::DeprecationWarning` (dissipative list-of-pots filterwarnings).
  The C RHS is a verbatim per-column copy of the already-merged-and-validated dxdv RHS; FD-of-flow is
  the gold-standard independent end-to-end check of the cyl↔rect folding. Task #100.
