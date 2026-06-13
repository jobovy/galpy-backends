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
- `feat/variational3d` → ✅ **merged to main** (#891, 2026-06-13); branch can be deleted. Lyapunov landed via this branch (#926/#945) — no separate `feat/lyapunov` was needed.
- `feat/backends` (long-lived; **the only live branch**) → all jax/torch/df/autodiff work; rebase-only syncs with main. Pending: rebase onto post-#891 main, #949 (f32 exit-cast + device-anchoring), then final review of **#892** → main.
- `feat/param-sensitivity` → Track C, not started.

## Status

**Status 2026-06-13:** Track B (3D variational equations + Lyapunov) is
**complete and on main** — #891 (feat/variational3d → main) merged 2026-06-13,
carrying the whole family (#898–#920, #926, #933–#941, #944/#945, #950–#952)
on top of the main-side fixes/closeouts (#921/#923/#925, #943/#947/#948,
#953/#954). Track A potentials are **complete on feat/backends**
(P2.1–P2.4, P2.5 + P2.5b, P2.6 wrappers, P2.7 dissipative — every galpy
potential incl. all wrappers is backend-agnostic), with the f32 exit-cast /
device policy in flight as #949 (device-anchoring commit being added). Next:
rebase feat/backends onto post-#891 main, final review of #892, then Tracks
C/D/E/F.

| PR | Track | Branch | State | Tests | Notes |
|----|-------|--------|-------|-------|-------|
| PR0+P1 backend foundation | 0 | feat/backends → main | **#892** OPEN (conflicting — pending rebase onto post-#891 main) | 76 backend tests; galpy.backend 100% cov | integration branch; final review after rebase |
| 3D variational + Lyapunov (feature branch) | B | feat/variational3d → main | ✅ **#891 MERGED 2026-06-13** | full 8-family battery green | **Track B complete — entirely on main** |
| P2.1 spherical | A | → feat/backends | ✅ **#893** merged 2026-06-06 | 95 pass/15 skip | |
| P2.2 disk | A | → feat/backends | ✅ **#894** merged 2026-06-06 | 138 pass | |
| P2.3 ellipsoidal | A | → feat/backends | ✅ **#895** merged 2026-06-07 | 258 pass | cache gated numpy-only |
| P2.4 halobar | A | → feat/backends | ✅ **#896** merged 2026-06-08 | 60 pass | all 10; flagged pre-existing CosmphiDisk bug |
| P2.7 dissipative | A | → feat/backends | ✅ done (scaffold **#897** closed) | dissipative AD via in-backend integrator | migration delivered via the Pspecial-era work |
| Pvar-rhs | B | → feat/variational3d | ✅ **#898** merged 2026-06-05 | det M=1 + flow-dir/FD/2D-bridge K-validators all tight (MN+Plummer) | the variational keystone |
| Pvar-pot.* (per-potential C Hessians) | B | → feat/variational3d | ✅ merged (#899–#920, #936–#941, #950, #952) | det/symp + C-vs-Py + FD-of-flow | EVERY potential + wrapper has full C variational support (planar + 3D) |
| SCF planar sign fix + FD-of-flow doctrine | B | → main | ✅ **#923** merged 2026-06-09 | FD-of-flow pins absolute values | det(M)=1 cannot catch sign errors — standing rule |
| SCF analytic 2nd derivs / DiskSCF planar | B | → main | ✅ **#921** merged 2026-06-09 / **#925** merged 2026-06-10 | FD <1e-6 / <1e-5 | should-have-existed → main |
| Pvar-lyapunov (Orbit.lyapunov) | B | → feat/variational3d | ✅ **#926** merged 2026-06-11 (+ spectrum **#945** 2026-06-12) | HH chaotic λ=0.044–0.051 vs lit 0.04–0.08; battery 121/121 | + doc notebook + CompositePotential hasC_dxdv3d fix |
| Pvar-dissip-struct (velocity-dependent J) | B | → feat/variational3d | ✅ **#933** merged 2026-06-11 | Jac-vs-FD 2.1e-8; FD-of-flow 2.8e-5; det M=exp(∫tr∂F/∂v) to 3.1e-6 | Chandrasekhar wired; conservative path bit-identical |
| Variational tail (FDM, NonInertial, wrappers, MovingObject, interpRZ, NeedleBar) | B | → feat/variational3d | ✅ **#934–#941** merged 2026-06-11 | FD-of-flow doctrine + adversarial verify | planar dissipative: out of scope (no such force) |
| Variational closeout + audit fixes | B | → variational3d / main | ✅ **#950–#954** merged 2026-06-12 (+ z-aliasing **#944**) | closed-form harmonic+Kepler STM, E/Lz left-relations | log-spiral planar Hessians; parser silent-skip segfault (#953); RotateAndTilt cache-t (#954) |
| Main-side wrapper fixes | B | → main | ✅ **#943/#947/#948** merged 2026-06-11/12 | regression batteries | KD t-forwarding; wrapper list input; wrapper repr |
| Pspecial (special-function router) | 0 | → feat/backends | ✅ **#916/#917** merged 2026-06-08/09 | native-vs-fallback + grad-vs-FD | jax hyp2f1/hyp1f1 forced to fallbacks |
| P2.5 special-backed (Einasto…DiskSCF) | A | → feat/backends | ✅ **#927–#932** merged 2026-06-09/10 + P2.5b TD Multipole **#942** 2026-06-12 | numpy byte-parity vs pristine + AD | |
| P2.6 wrappers | A | → feat/backends | ✅ **#946** merged 2026-06-13 | 474 new tests; backend suite 2069; 18,822-value byte-parity | **Track A potentials complete** — every wrapper backend-agnostic |
| f32 exit-cast / device policy | A | → feat/backends | **#949** open | 46-entry dtype gate (zero skips); 92+324-record byte proofs | f32 in → f32 out for table-backed pots; device-anchoring commit being added |
| In-backend integrator (diffrax/torchdiffeq) | D | → feat/backends | ✅ **#906** merged 2026-06-08 | C-parity ~1e-9; param-grads via AD | per-orbit stepping kept (vs galax batched: 2.9x, by design) |
| Track G figures/diagnostics | G | (cross-cutting) | ⏳ continuous | g0–g11 in manifest | g9–g11 variational/Lyapunov/dissipative figures; found+fixed lyapunov 3D restart bug (z←vT aliasing, #944) |
| Tracks C/D/E/F | — | — | ⏳ next (after backends rebase + #892) | — | — |

(Update this table as PRs land.)

## Backend support snapshot (this env)
- jax 0.9.2 ✅ · torch 2.12 (CPU) ✅ · array-api-compat 1.14 ✅
- `array_api_compat.array_namespace`: numpy → (normalized to plain numpy), jax → `jax.numpy`, torch → `array_api_compat.torch`.
