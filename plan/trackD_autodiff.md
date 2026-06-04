# Track D — Autodiff wrappers + in-backend integrators

Branch: `feat/backends`. Detail in master plan §"Track D". Depends on Track B's
STM (Pvar-rhs, on main) and Track 0 (P1, P0-util).

- **Pjax-orbit / Ptorch-orbit (C-STM):** forward = C orbit (jax via
  `jax.pure_callback`, returning STM residual M; `vmap_method="expand_dims"` reusing
  the multi-orbit C loop); backward = `Mᵀ·λ̄` (host-side, no backward C call).
  `jax.custom_vjp` / `torch.autograd.Function` (gradcheck mandatory). In
  `galpy/backend/_jax|_torch/orbit_stm.py`; `Orbit.integrate` routes by IC namespace;
  API numpy-identical. jax FFI deferred (`jax.ffi` is the current API in 0.9.2;
  pure_callback is enough for CPU-bound serial ODE).
- **Pinbackend (first-class diffrax/torchdiffeq):** in-backend ODE integration of the
  backend-agnostic forces → higher-order autodiff (Hessian-of-orbit) + parameter
  gradients for all potentials via AD; runtime extras; resolver dispatch; cross-validate
  vs C-STM + Liouville.
