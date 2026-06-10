# galpy-jax — multi-backend (JAX/PyTorch) + autodiff + 3D variational equations

Working log + paper material for making **all of galpy** differentiable and
backend-agnostic (numpy/scipy, JAX, PyTorch, array-API), and implementing the
**3D variational equations** (state-transition matrix / full 3D potential
Hessian) + **Lyapunov exponents** in C.

- Master plan (index): `master_plan.md` (committed snapshot; kept in sync with the working copy at `~/.claude/plans/i-want-to-start-snug-rocket.md`)
- Design discussion (settled decisions): `galpy_backend_autodiff_discussion.md`
- Per-track detailed specs: `plan/` (incl. `plan/trackG_figures.md` — reproducible paper figures)
- Per-PR logs (Approach / Dead-ends / Results / Plots): `logs/`
- Paper draft + figures: `paper/`, with `paper/figures/` (every figure ships `make.py` + a reproducibility hash; see Track G)

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
| Pvar-rhs | B | feat/pvar-rhs → #891 | **#898** (draft) | det M=1 + flow-dir/FD/2D-bridge K-validators all tight (MN+Plummer) | the variational keystone — VALIDATED; coverage fix: dxdv dispatcher now mirrors planar (non-symplectic cases only) so misses return to 27 |
| Pvar-pot.* (per-potential C Hessians) | B | → feat/variational3d | ✅ merged (#899–#920 family PRs) | det/symp + C-vs-Py + FD-of-flow | 38 potentials with full 3D C Hessian |
| SCF planar sign fix + FD-of-flow doctrine | B | → main | ✅ **#923** merged | FD-of-flow pins absolute values | det(M)=1 cannot catch sign errors — standing rule |
| SCF analytic 2nd derivs / DiskSCF planar | B | → main | ✅ **#921** merged / **#925** open | FD <1e-6 / <1e-5 | should-have-existed → main |
| Pvar-lyapunov (Orbit.lyapunov) | B | → feat/variational3d | **#926** open (CI green) | HH chaotic λ=0.044–0.051 vs lit 0.04–0.08; battery 121/121 | + doc notebook + CompositePotential hasC_dxdv3d fix |
| Pvar-dissip-struct (velocity-dependent J) | B | → feat/variational3d | **#933** open | Jac-vs-FD 2.1e-8; FD-of-flow 2.8e-5; det M=exp(∫tr∂F/∂v) to 3.1e-6 | Chandrasekhar wired; conservative path bit-identical |
| Variational tail (FDM, NonInertial, 3 wrappers, MovingObject, interpRZ, NeedleBar) | B | → varia3d/#933 | ⏳ 8-agent fan-out running | FD-of-flow doctrine + adversarial verify | planar dissipative: out of scope (no such force) |
| Pspecial (special-function router) | 0 | → feat/backends | ✅ **#916** merged | native-vs-fallback + grad-vs-FD | jax hyp2f1/hyp1f1 forced to fallbacks |
| P2.5 special-backed (Einasto…DiskSCF) | A | → feat/backends | **#927–#932** open (CI green) | numpy byte-parity vs pristine + AD | TD Multipole queued (P2.5b) |
| In-backend integrator (diffrax/torchdiffeq) | D | → feat/backends | ✅ merged (#906) | C-parity ~1e-9; param-grads via AD | per-orbit stepping kept (vs galax batched: 2.9x, by design) |
| Track G figures/diagnostics | G | (cross-cutting) | ⏳ continuous | g1–g8 in manifest | g8 matched-tolerance + galax per-orbit apples-to-apples |
| P2.6 wrappers, Track C/E/F | — | — | ⏳ next | — | — |

(Update this table as PRs land.)

## Backend support snapshot (this env)
- jax 0.9.2 ✅ · torch 2.12 (CPU) ✅ · array-api-compat 1.14 ✅
- `array_api_compat.array_namespace`: numpy → (normalized to plain numpy), jax → `jax.numpy`, torch → `array_api_compat.torch`.
