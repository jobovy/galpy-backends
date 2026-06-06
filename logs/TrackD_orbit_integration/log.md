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

## Next
- Formalize into `galpy/backend/_reference/inbackend_ode.py` + route `Orbit.integrate` by IC namespace; add tests (grad-vs-FD, jacrev==jacfwd, torch `gradcheck`, C-vs-in-backend agreement).
- Paper figure: orbit-sensitivity (STM columns) + a gradient-descent orbit fit (Track G G3 showpiece, currently blocked — this unblocks it).
- Cross-validate against path B (C-STM) once Pvar-rhs is in main.
