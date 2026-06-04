# Overview & motivation

galpy is numpy/scipy + C. This project makes **all of galpy** differentiable and
backend-agnostic (numpy, JAX, PyTorch, array-API) so galpy can sit inside
autodiff/ML models of galactic dynamics, while keeping the fast C paths; and it
adds the **3D variational equations** (state-transition matrix = full 3D
potential Hessian) and **Lyapunov exponents** in C as standalone features.

## Settled design decisions (see `galpy_backend_autodiff_discussion.md`)
1. **Follow the data.** Backend resolution order: explicit `xp=` → namespace of
   the array arguments → context-manager/global (`contextvars`) → numpy. Engine:
   `array-api-compat`'s `array_namespace`, wrapped behind `galpy.backend.get_namespace`
   with a vendored fallback. numpy/python inputs resolve to *plain numpy* so the
   numpy path is byte-identical.
2. **Dispatch at the private compute layer** (`_evaluate/_Rforce/...`); unit
   decorators, `_amp`, units stay backend-agnostic.
3. **Orbit autodiff = forward-sensitivity STM** (n=6): backward = `Mᵀ·λ`, no
   backward C call. Higher-order via first-class in-backend diffrax/torchdiffeq.
4. **Prefer native backend functions; isolate galpy fallbacks** (special functions).
5. **Tolerances:** numpy byte-identical; jax/torch pass the *same* tests at the
   *same* tolerances (jax x64 on), loosened only if absolutely necessary.

## Scope
All potentials · all orbit integration · all action-angle (incl. C-backed
Staeckel/Adiabatic via the Leibniz derivative integrands already in C) · the
entire `df` module (incl. streams/stream tracks) · the `util` compute modules
(`coords`, `conversion`, `special`). Pure-backend special functions
(assoc_legendre/gammaincinv/hyp2f1...). Full parameter gradients for all
potentials. **Out of scope:** `actionAngleTorus` (external C++ library).

## Tracks
- **0** infra (galpy.backend, special router, coords/conversion, CI, parametrization)
- **A** potentials namespace-swap
- **B** C 3D variational equations (→ main) + **Lyapunov** (→ main)
- **C** C parameter sensitivity / ∂f/∂θ (→ main)
- **D** autodiff wrappers (C-STM custom_vjp/autograd.Function) + in-backend integrators
- **E** action-angle (all methods)
- **F** df module (all DFs + streams)
