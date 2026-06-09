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

## Performance note — resolve the namespace once per call (deferred opt)

`get_namespace(*arrays)` costs ~3 us/call and adds ~16% to a *scalar* pure-Python
force eval (measured on MWPotential2014 `_evaluateRforces`: ~40 us/call, of which
~6 us is two `get_namespace`/`namespace_from_arrays` resolutions). This does NOT
affect orbit integration via the C integrators (they call the C force functions
directly, never the namespaced Python `_evaluate*` layer), so it is not on the
hot path for `Orbit.integrate`. Where it bites is pure-Python evaluation loops
(plotting, Python-only methods, the in-backend diffrax/torchdiffeq RHS, df
quadratures).

Optimization (do during the orbit/df backend migration, not urgently): resolve
`xp` ONCE at the public-method / RHS-entry boundary and thread it down through the
private compute layer (mirrors how the Python integrators use the underscored,
decorator-free `_evaluateRforces`/`_evaluatephitorques`/`_evaluatezforces` to skip
the unit decorators). Expected ~15% on those Python paths; zero risk to the C
paths. Discovered while profiling the run-notebooks slowdown (which turned out to
be an unrelated slow live Gaia archive query in staeckel.ipynb, NOT the
namespace-swap).
