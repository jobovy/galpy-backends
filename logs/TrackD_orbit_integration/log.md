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

## Next
- Route `Orbit.integrate` by IC namespace (numpy-identical public API); `galpy[diffrax]`/`galpy[torchdiffeq]` extras + a CI job that exercises them.
- Cross-validate against path B (C-STM) once Pvar-rhs is in main.
