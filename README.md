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
| PR0+P1 backend foundation | 0 | feat/backends → main | **#892** (CI running) | 76 backend tests; galpy.backend 100% cov | integration branch |
| Pvar-struct (C 3D Hessian ABI) | B | feat/variational3d → main | **#891** (CI running) | 22 pass (inert) | variational feature branch (merges to main first) |
| P2.1 spherical | A | feat/P2.1-spherical → #892 | **#893** (draft) | 95 pass/15 skip | |
| P2.2 disk | A | feat/P2.2-disk → #892 | **#894** (draft) | 138 pass | |
| P2.3 ellipsoidal | A | feat/P2.3-ellipsoidal → #892 | **#895** (draft) | 258 pass | cache gated numpy-only |
| P2.4 halobar | A | feat/P2.4-halobar → #892 | **#896** (draft) | 60 pass | all 10; flagged pre-existing CosmphiDisk bug |
| P2.7 dissipative | A | feat/P2.7-dissipative → #892 | **#897** (draft) | 6 pass/2 xfail | deferred to Pspecial |
| Pvar-rhs | B | feat/pvar-rhs → #891 | **#898** (draft) | det M=1 + flow-dir/FD/2D-bridge K-validators all tight (MN+Plummer) | the variational keystone — VALIDATED |
| Pvar-pot.* (≈48 potentials' C Hessians) | B | → #891 | ⏳ next fan-out | — | RHS/ABI/plumbing now general; add derivs→wire→test |
| Pspecial, P2.5, P2.6, Pvar-lyapunov, Track C/D/E/F | — | — | ⏳ next | — | — |

(Update this table as PRs land.)

## Backend support snapshot (this env)
- jax 0.9.2 ✅ · torch 2.12 (CPU) ✅ · array-api-compat 1.14 ✅
- `array_api_compat.array_namespace`: numpy → (normalized to plain numpy), jax → `jax.numpy`, torch → `array_api_compat.torch`.
