# galpy-jax — multi-backend (JAX/PyTorch) + autodiff + 3D variational equations

Working log + paper material for making **all of galpy** differentiable and
backend-agnostic (numpy/scipy, JAX, PyTorch, array-API), and implementing the
**3D variational equations** (state-transition matrix / full 3D potential
Hessian) + **Lyapunov exponents** in C.

- Master plan (index): `~/.claude/plans/i-want-to-start-snug-rocket.md`
- Design discussion (settled decisions): `galpy_backend_autodiff_discussion.md`
- Per-track detailed specs: `plan/`
- Per-PR logs (Approach / Dead-ends / Results / Plots): `logs/`
- Paper draft: `paper/`

## Branches
- `feat/variational3d`, `feat/lyapunov`, `feat/param-sensitivity` → merge to **main** (standalone C features the backends consume but don't depend on).
- `feat/backends` (long-lived) → all jax/torch/df/autodiff work; merges `main` in; final merge to main.

## Status

| PR | Track | Branch | State | Tests | Notes |
|----|-------|--------|-------|-------|-------|
| PR0 pilot (resolver + Plummer/Isochrone) | 0 | feat/backends | ✅ done | +70 (numpy/jax/torch green) | validates the whole approach |
| Pvar-struct (C 3D Hessian ABI) | B | feat/variational3d | ✅ done | 22 pass (inert) | serialization point for the C fan-out |
| P1 backend package (config, special router) | 0 | feat/backends | ⏳ next (pending review) | — | — |
| Pvar-pot.* / Pvar-rhs | B | feat/variational3d | ⏳ next (pending review) | — | — |

(Update this table as PRs land.)

## Backend support snapshot (this env)
- jax 0.9.2 ✅ · torch 2.12 (CPU) ✅ · array-api-compat 1.14 ✅
- `array_api_compat.array_namespace`: numpy → (normalized to plain numpy), jax → `jax.numpy`, torch → `array_api_compat.torch`.
