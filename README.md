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
- `feat/backends` (long-lived; **the only live branch**) → all jax/torch/df/autodiff work; rebase-only syncs with main. #949 (f32 exit-cast + device policy) ✅ **merged 2026-06-13**. Rebased onto post-#891 main (task #37). **#892** remains the open integration branch → main; final review after the Track D wave settles.
- `feat/param-sensitivity` → Track C, not started.

## Status

**Status 2026-06-18:** Track B (3D variational equations + Lyapunov) is
**complete and on main** — #891 (feat/variational3d → main) merged 2026-06-13,
carrying the whole family (#898–#920, #926, #933–#941, #944/#945, #950–#952)
on top of the main-side fixes/closeouts (#921/#923/#925, #943/#947/#948,
#953/#954). Track A potentials are **complete on feat/backends** — every
galpy potential incl. all wrappers is backend-agnostic (P2.1–P2.4, P2.5 +
P2.5b, P2.6 wrappers, P2.7 dissipative), with the f32 exit-cast / device
policy landed (#949, 2026-06-13). The migration tail then cleared the
remaining systemic gaps: central torch scalar-input coercion (#960), the
SCF/Multipole vectorization that killed the per-point `ndindex` hang (#959),
the straightforward potential tail + td-amplitude wrappers (#962/#963),
backend interpolation/quadrature + `special` hardening (#964/#965/#966), and
CUDA device-mismatch / coerced-scalar fixes (#967/#968). The convenience
methods + functional interface (vesc/rtide/ttensor/tdyn, rl/rE/zvc via backend
root-find, mass() via backend GL quadrature) are now in flight as **#975**,
which completes the user-facing Track A surface. Test-infra: the **all-backend
existing-suite CI** is live (#956 — xfail-ledger, sharded, on
feat/backends/nightly/dispatch + per-PR backend smoke), and the **milestone-1
ledger prune** (#969) regenerated the ledger (torch 1684→1485, jax 335→292).
**Track D** (differentiable orbit integration) is underway: the fast C-STM
functional interface (#972) and in-backend `Orbit.integrate` diffrax/torchdiffeq
methods (#973) are merged; backend-aware accessors (#974) are open. Next:
land #974/#975, then final review of #892 → main, then Tracks C/E/F.

| PR | Track | Branch | State | Tests | Notes |
|----|-------|--------|-------|-------|-------|
| PR0+P1 backend foundation | 0 | feat/backends → main | **#892** OPEN (integration branch; rebased onto post-#891 main per task #37) | 76 backend tests; galpy.backend 100% cov | final review after the Track D wave settles |
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
| f32 exit-cast / device policy | A | → feat/backends | ✅ **#949** merged 2026-06-13 | 46-entry dtype gate (zero skips); 92+324-record byte proofs | f32 in → f32 out for table-backed pots; f64 interior preserved |
| all-backend existing-suite CI | 0 | → feat/backends | ✅ **#956** merged 2026-06-15 | xfail-ledger + sharded suite (feat/backends + nightly + dispatch); per-PR backend smoke | makes the migration tail's regressions visible — no hiding uncovered code |
| #39 SCF/Multipole jax/torch vectorize | A | → feat/backends | ✅ **#959** merged 2026-06-15 | killed per-point ndindex hang; batched (N,L,M) | numpy byte-identical; un-deferred 31 jax SCF tests |
| torch scalar-input coerce (central) | A | → feat/backends | ✅ **#960** merged 2026-06-16 | coerce at potential_physical_input for migrated pots only; jax estimates+torch jeans green, numpy byte-identical | `_backend_compatible` flag (hasC convention) + `_check_backend_compatible` gate; fixes systemic torch scalar rejection |
| Tail (a) PseudoIso/KuzminKutuzov/Ferrers/RazorThin | A | → feat/backends | ✅ **#962** merged 2026-06-17 | numpy byte-identical; jax+torch parity ~1e-15 | straightforward namespace-swaps |
| Tail (c) DehnenSmooth/GaussianAmplitude td-wrappers | A | → feat/backends | ✅ **#963** merged 2026-06-16 | numpy byte-identical; parity ≤1e-15 | _smooth backend + flag True (WrapperPotential bypasses wrapped decorator) |
| Tail infra-1: backend interpolation + quadrature | A/0 | → feat/backends | ✅ **#965** merged 2026-06-16 | interpolate.py (1D frozen + diff-in-y cubic + 2D) + quadrature.py; numpy byte-identical, AD-in-y 1.5e-10 | unblocks friction sigma_r, group-d, df/Track-F |
| Tail infra-2: special-fn audit (iv + sici) | 0 | → feat/backends | ✅ **#964** merged 2026-06-16 | iv (recurrence) + sici (jax native/torch fallback); vs scipy 1e-13..1e-15 + gradcheck | sici for FDM; iv for RazorThin Hessian |
| backend.special fallback edge hardening | 0 | → feat/backends | ✅ **#966** merged 2026-06-16 | poles, domain endpoints, AD, numpy params | hardens the special-fn router at edges (per special-fn-edge-testing doctrine) |
| CUDA device-mismatch on mixed scalar/array | 0 | → feat/backends | ✅ **#967** merged 2026-06-16 | mixed scalar/array coordinate cases green | SCF, coords, DoubleExp |
| DoubleExp 0-d-array R as scalar | A | → feat/backends | ✅ **#968** merged 2026-06-16 | fixes coerced-scalar reshape error | follow-on to the #960 central coercion |
| all-backend ledger: milestone-1 prune | 0 | → feat/backends | ✅ **#969** merged 2026-06-17 | torch 1684→1485, jax 335→292 (commit msg) | regen prune at migration milestone (ledger size = progress metric) |
| Tail (b) friction NonInertial/Chandrasekhar/FDM | A | → feat/backends | ⏳ after infra | full differentiability (user: full win) | uses backend interp + sici |
| Tail (d) AnySpherical/AnyAxiRazorThin/interpRZ | A | → feat/backends | ⏳ after infra | backend quadrature + 2D interp | the hard quad/interp consumers |
| In-backend integrator (diffrax/torchdiffeq) | D | → feat/backends | ✅ **#906** merged 2026-06-08 | C-parity ~1e-9; param-grads via AD | per-orbit stepping kept (vs galax batched: 2.9x, by design) |
| Track D fast C-STM orbit autodiff — PR-B (functional) | D | → feat/backends | ✅ **#972** merged 2026-06-17 | functional interface; param-grads via the C state-transition matrix | fast differentiable integration reusing the Track B C-STM |
| Track D in-backend `Orbit.integrate` — PR-A | D | → feat/backends | ✅ **#973** merged 2026-06-18 | diffrax / torchdiffeq integration methods on `Orbit.integrate` | in-backend differentiable ODE solve |
| Track D backend-aware Orbit accessors — PR-B | D | → feat/backends | **#974** open | accessor namespace-coerce (forced-torch rejected numpy thiso) + torch-aliasing clone; off-grid interp via backend Spline1D | real-bug fix; ledger +16 |
| Track A convenience methods + functional interface | A | → feat/backends | **#975** open | module-level vesc/rtide/ttensor/tdyn + convenience → xp; rl/rE/LcE/zvc via backend brentq; mass() via backend GL; numpy byte-identical | completes user-facing Track A; dissipative backend cases deferred; ledger +50 |
| jax orbit un-defer | D | → feat/backends | ✅ **#998** merged 2026-06-23 | 0→606 jax orbit tests run; `_backend_integrators` C-only helper; duration-balanced 5-way shard; +24-pot in-backend `_POTS` | Python ODE integrators stay slow-skipped (eager per-force stepping); see logs/orbit_backend_burndown |
| Orbit read-path reductions (e/rap/rperi/zmax + normalize + interp-warn) | A/D | → feat/backends | ✅ **#1001** merged 2026-06-23 | numpy.amax/amin/any/fabs → get_namespace/xp + builtin abs(); 12 forced-torch flips, 0 regressions, byte-identical numpy | code-only (no ledger); +backend accessor/normalize tests; segfault-noise caught by serial re-verify |
| Orbit accessor-reduction tests backend-agnostic | D | → feat/backends | ✅ **#1002** merged 2026-06-23 | 34 forced-torch comparison/energy/scalar tests wrap accessor outputs via shared `_to_numpy` (conftest); byte-identical numpy | accessors CORRECTLY return torch (load-bearing #974 coercion); the tests adapt, not the orbit code |
| C-STM routing into Orbit.integrate (PR-C) | D | → feat/backends | ✅ **#1003** merged 2026-06-24 | Orbit(jax/torch IC).integrate(dop853_c) → differentiable backend array via the dxdv STM; in-backend fallback (non-6D / no C-Hessian); numpy byte-identical | `d o.e()/dR0` over a C integration (jax+torch == FD); 28 routing tests + full 6×6 STM jacobian; params still in-backend |
| Orbit SOS / dissipative-minr / multi-orbit interp-warn reductions | A/D | → feat/backends | ✅ **#1004** merged 2026-06-24 | SOS init_psis via xp.atan2 + reductions; dissipative minr; multi-orbit interp-range warning → get_namespace/xp; byte-identical numpy | code-only (no ledger); completes the orbit read-path burndown |
| **Track E** closed-form action-angle (Harmonic + Isochrone fwd + HarmonicInverse) | E | → feat/backends | **#1005** open | forward Harmonic+Isochrone + HarmonicInverse namespace-swap; every masked write → xp.where/xp.clip; byte-identical numpy | +tests/test_backend_actionAngle.py (45): parity + grad-vs-FD + **dθ/dt=Ω** consistency (∇θ·ẋ==Ω to 1e-15, fwd Isochrone+Harmonic, pro+retrograde) + inverse round-trip. IsochroneInverse deferred (scipy Newton → root-find increment). streamdf-jax slow_skip (this PR's swap makes its track assembly real-eager-jax) |
| Orbit-milestone xfail-ledger prune | 0 | → feat/backends | ✅ **#1006** merged 2026-06-24 | full-matrix regen 28083178345 (all 45 shards complete); jax 289→287, torch 1029→961; 70 drops, 0 adds (test_orbit 31t+2j, test_orbits 31t, +6 incidental) | surgical prune off the 2026-06-22 milestone-2 base; caught stale-main-repo-ledger trap → diff vs origin/feat/backends |
| Track G figures/diagnostics | G | (cross-cutting) | ⏳ continuous | g0–g12 in manifest | g12 galpy in-backend orbit GPU throughput (vs C/jax-CPU/torch); g2/g6 re-measured on TITAN Xp; g9–g11 variational/Lyapunov/dissipative; found+fixed lyapunov 3D restart bug (z←vT aliasing, #944) |
| Tracks C/E/F | — | — | ⏳ next (after #974/#975 + #892) | — | — |

(Update this table as PRs land.)

**Ledger baseline (verified 2026-06-18, `origin/feat/backends:tests/backend_xfail.txt`):**
torch **1494** xfail / jax **254** xfail. The #969 milestone-1 prune regenerated
the ledger (commit message: torch 1684→1485, jax 335→292); subsequent tail merges
and the central-coercion fixes brought it to the current baseline. Open-PR triage
adds are branch-local until merge (#974 ≈ +16, #975 ≈ +50).

## Backend support snapshot (this env)
- jax 0.9.2 ✅ · torch 2.12 (CPU) ✅ · array-api-compat 1.14 ✅
- `array_api_compat.array_namespace`: numpy → (normalized to plain numpy), jax → `jax.numpy`, torch → `array_api_compat.torch`.
