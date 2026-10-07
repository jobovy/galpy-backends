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

**Status 2026-10-07.** feat/backends is rebased onto main (7c675eb63, applied
2026-10-04) and green. Ledgers: **3** xfail entries (all `--jit` pickling of unit
wrappers), **3** slow-skips (FDM `central_limit` under torch/torch-jit: eager
Python-integrator dispatch; ExpTruncNFW `analytic_ecc_rperi_rap` under jax-jit,
260 s after #1625), **16** permanent skips (actionAngleTorus via streamdf useTM;
four eager-jax per-op floors, measured not compile-bound), across 88 backend
test modules.

Open on 2026-10-07: **#1641** (VerticalInverse Newton polish: `guess=` for the
backend brentq's safeguarded Newton; angle inversion ~2x faster under jit) is
green; **#1642** (Staeckel `c=False`: chi-anomaly mesh matching C, 3.8x faster;
fixes the ~9x regression of Sept. 2) is open, CI running. The paper draft is at
`paper/ms`.

Since 2026-10-01:

- **actionAngleVerticalInverse is on the backends** (#1627/#1628/#1630/#1631;
  out of scope until it stabilized): the momentum-matched map is built and
  evaluated natively, differentiable in the action, the angles and the
  potential's parameters, and jit-able end to end; the legacy modes evaluate
  natively on a numpy construction. All 49 tests run under jax/torch (+jit) and
  on CUDA. Open: a backend-native (differentiable) legacy construction.
- **float64 by default** (#1613): galpy's internals run in float64 at torch's
  float32 default; a user's float32 tensor stays float32 in and out.
- **jit-mode correctness** (#1625): static objects keyed by type + attributes;
  an identity key had served stale traces (2x wrong forces) after a wrapped
  potential's `normalize()`.
- **eager-jax recompilation** (#1621): varying array lengths padded to
  power-of-4 buckets; C bridges skip pure_callback for concrete inputs
  (qdf sampleV_interpolate 2837 s -> 118 s under jax).
- differentiable `from_density` for SCF (#1612) and Multipole (#1618);
  TwoPower via the incomplete beta (#1597); jax gammainc ~100x faster (#1615).
- CI: CPU/AVX-512 runner reported in the pytest header (#1620); the #1614
  inductor SIMD cap confirmed necessary on an avx512f runner.

<details>
<summary>Previous status (2026-09-16)</summary>

**Status 2026-09-16.** The migration is essentially complete on the "does it
run" axis: the backend xfail ledger is down to **3** entries and the slow-skip
ledger to **5**, across 82 backend test modules. What remains is a
*differentiability-correctness* tail, audited on 2026-09-16 and written up in
[`plan/differentiability_audit_2026-09-16.md`](plan/differentiability_audit_2026-09-16.md).

Measured differentiable (AD vs a finite difference that REBUILDS from the
parameter, not an in-object perturbation):

| area | verdict |
|------|---------|
| potentials | **49 of 50** instantiable potentials differentiable w.r.t. `amp` under both `grad` and `jit`; only `DiskSCFPotential` fails under `jit` |
| orbits | `integrate` + accessors w.r.t. potential parameters — 1.3e-06 (Hernquist), 8.9e-06 (LogHalo q), 2.2e-08 (Plummer) |
| action-angle | Staeckel(c=False) 1.7e-10, Adiabatic 1.9e-10, Spherical 2.7e-11 |
| streamdf | `__init__` w.r.t. a potential parameter, after the `real_eig` unfreeze |
| streamspraydf | construction + sampling 1.2e-08 |
| diskdf | `dehnendf.surfacemass` d/d(hr) 6.7e-10 |

Known gaps (all in `plan/differentiability_audit_2026-09-16.md`): DF
**constructors** other than streamdf's are not traceable (streamgapdf, the
spherical-DF family, qdf, kingdf); `actionAngleStaeckel(c=True)` differentiates
w.r.t. initial conditions only, by design; a handful of potentials branch in
Python on a stored parameter, which is fine under `grad` but breaks under `jit`;
and two deliberate gradient freezes (`psd_project` eigenvectors, Staeckel `u0`)
are still unmeasured.

**Record-keeping note:** this README's per-PR table went un-updated between
2026-08-24 and 2026-09-16 while 89 PRs merged; the rows below were rebuilt from
the GitHub API rather than from notes, and the Status section above replaces one
that had been stale since 2026-06-18.

</details>

<details>
<summary>Previous status (2026-06-18)</summary>

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

</details>

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
| **Track E** closed-form action-angle (Harmonic + Isochrone fwd + HarmonicInverse) | E | → feat/backends | ✅ **#1005** merged 2026-06-24 | forward Harmonic+Isochrone + HarmonicInverse namespace-swap; every masked write → xp.where/xp.clip; byte-identical numpy | +tests/test_backend_actionAngle.py (45): parity + grad-vs-FD + **dθ/dt=Ω** consistency (∇θ·ẋ==Ω to 1e-15, fwd Isochrone+Harmonic, pro+retrograde) + inverse round-trip. streamdf-jax slow_skip + 2 VerticalInverse-torch ledger (HarmonicInverse→real-torch collateral) |
| **Track E** IsochroneInverse backend (Kepler root-find) | E | → feat/backends | ✅ **#1007** merged 2026-06-24 | (J,angle)→(x,v); Kepler η−(a·e/ab)sinη=angler; numpy keeps scipy.newton (byte-identical), jax/torch use the SHARED `backend.optimize.brentq` (bracketed bisection + 1-Newton-step implicit-diff) on [0,2π] | first AA consumer of the shared root-finder; parity ~3e-16, round-trip ~1e-16, d(ΣR)/djr AD==FD ~5e-10. Reused for Vertical/Spherical/Staeckel next |
| **Track E** Spherical backend PR-1 (actions + EccZmaxRperiRap) | E | → feat/backends | ✅ **#1008** merged 2026-06-24 | most complex AA method; additive `elif is_backend_array` branch (numpy byte-identical, 0 deletions); vectorised rperi/rap via shared backend.optimize.brentq + Jr via sin²-substitution fixed_quad | design fan-out + implement→adversarial-verify workflow; parity ~3.8e-12, grad-vs-FD ~6e-8; 60+20+18 tests green. PR-2 = freqs+angles |
| **Track E** Spherical backend PR-2 (frequencies + angles) | E | → feat/backends | ✅ **#1009** merged 2026-06-24 | additive `elif`; t²-two-panel Tr/I integrals via per-object-limit fixed_quad + circular epifreq/omegac branch + xp.where angle assembly; numpy byte-identical. **Edge-case hardening:** non-inclined `psi=phi`/`Omega=0` matched EXACTLY (finiteness mask from raw `z/r/sin(i)`, no NaN grad in `xp.where`); near-radial GL order 25→50 | **actionAngleSpherical fully migrated** (actions+ecc+freqs+angles); 8 edge cases × 4 pots × pro/retro match numpy (circular **0.0** exact, planar 1.5e-9; only pathological L/Lcirc<1e-4 hits the irreducible adaptive-quad floor ~1e-6). parity Or~2.8e-9/angles~1e-9, grad~1e-7; **200 backend**+20 numpy green |
| **Track E** Staeckel pure-Python freqs+angles | E | → main | **#1010** open 2026-06-24 | actionsFreqs/Angles were C-only; pure numpy/scipy port (6 derivative integrands, 2×2 Jacobian→Ω, verbatim angle quadrant tree, calcu0). c=False==C ~1e-9 freqs/~1e-7 angles. **Review fixes:** removed now-dead `_setup_actions` Orbit fallback; made actions C-consistent (v0=π/2, was v0=vx, off C ~6e-3) | impl→adversarial-verify workflow + my circular ZeroDivisionError fix; 100% diff coverage (test + pragmas); 200+ tests green. Reference for the vectorized rewrite (drop Single) next |
| **Track A** verticalPotential backend (actionAngleVertical prereq) | A | → feat/backends | ✅ **#1011** merged 2026-06-25 | namespace-swap the 3D→1D wrapper (`numpy.ones_like(z)`→backend, follow z's actual type), `_backend_compatible` from parent | numpy byte-identical 0.0, parity 4e-16, grad 2e-10; fixed 2 CI reds (list-deprecation, NullPotential numpy-under-forced-context). Unblocks Vertical AA backend |
| **Track E** Staeckel vectorized actions (PR-1 of 2) | E | → feat/backends | ✅ **#1013** merged 2026-06-26 | one unified vectorized numpy+backend path replacing the per-object Single scipy loop in _evaluate; turning points via shared bisect_root (fixed-iter expanding bracket + xp.where at-turn/circular), J_R/J_z via fixed_quad. C-consistency = plain GL order-10 (NO subst; integrand vanishes at turning pts) | c=False actions ≡ c=True **4e-16**; jax/torch parity 1e-17; +UnboundError vectorized + delta-namespace coercion; driven through 4 CI rounds (qdf regression, #1010-freqs/angles ledger, isort, codecov-cover, backend_managed). **gradients deferred to PR-2** (dJ integrands). Freqs core validated 1e-10, integrating now |
| **Track E** Staeckel jit-compat + GPU device-anchor | E | → feat/backends | ⏳ **#1016** open 2026-06-26 | jit-compat: the eager `if bool(xp.any(unbound))` raise in `_staeckel_prep` raised TracerBoolConversionError under jax.jit (couldn't jit the AA map); guarded behind `not is_backend_array(R)` (numpy byte-identical raise, backend falls through to NaN/garbage → traceable). +10 GPU device-anchor sites (fixed_quad GL nodes/delta/phi on input device) across Staeckel/Vertical/Spherical | `jit(actionsFreqsAngles)` traces + matches eager **9e-10**; GPU-validated 17/17 torch + 6/6 jax. **Compile is the catch:** ~8.6 min/shape (47,528-eqn unrolled jaxpr); rolled-`lax.fori_loop` bisection prototyped = bit-identical, **12× compile** at 1.7× warm cost → future jax kernel. See `logs/trackE_staeckel_jit/` |
| **Track E** Staeckel vectorized freqs+angles (PR-2) | E | → feat/backends | ✅ **#1014/#1015** merged 2026-06-26 | vectorized C-consistent frequencies (#1014: 6 Leibniz dJ integrands → 2×2 Jacobian → Ω, +3 bracketing-collapse fixes) and angles (#1015: 12-leaf quadrant tree; +restore actionAngleStaeckelSingle degenerate-branch coverage, codecov/project gated all-clear) | c=False ≡ c=True ~1e-9 freqs / 6.75e-10 angles across 648 orbits; backend parity ~1e-11. Drops the per-object Single freq/angle methods |
| **Track E** Vertical backend (J/Ω/angle) | E | → feat/backends | ✅ **#1012** merged 2026-06-25 | 1D analog of Spherical's radial Jr/Or/ar: additive `elif is_backend_array` (numpy byte-identical, **0 deletions**); xmax via shared backend.optimize.brentq + expanding bracket; J/Ω/angle via fixed_quad with x=xmax−t² turning-point substitution; (x,vx)-sign quadrant fix | parity ~1e-10 (IsoDisk/KG/vertMN × jax+torch), all edges match numpy (vx=0 turn, x=0, large-ampl, 4 quadrants), grad-vs-FD ~1e-9; 12 backend + 32 numpy green |
| **Track E** Adiabatic backend | E | → feat/backends | ⏳ **#1018** open 2026-06-27 | **NOT free** (empirically: numpy path recurses to `aAV(z[0],vz[0])` 0-d scalar → backends reject). Vectorised path: `Jr,Lz,Ω,ar,aphi` delegate to Spherical backend (`_Jz=` array + γ·Jz handled inside); `Jz,Ωz,az` reuse Vertical's GL machinery with per-R `Φ(R_i,z)−Φ(R_i,0)` via `_BatchedVerticalPotential` (folded into potential/verticalPotential.py) | threads per-R Φ(z) through Vertical's backend helpers. Task #89 |
| **Track E** VerticalInverse/Torus not-backend guards | E | → feat/backends | ⏳ **#1017** open 2026-06-27 | explicit `NotImplementedError` under forced/active backend or backend-array input: VerticalInverse "not yet migrated (under development)"; Torus "not backend compatible (and will not be)" — external C++ wrapper. Guards `__init__` (`backend()!="numpy"`) + eval methods | leave VerticalInverse alone (user); Torus permanent out-of-scope; 2 tests. Tasks #94/#43 |
| **Track E** estimateDeltaStaeckel + estimateBIsochrone | E | → feat/backends | ⏳ **#1019** open 2026-06-27 | estimator helpers namespace-swapped: in-place z→`xp.where`, `numpy.median`→nanmedian (torch quantile(.,0.5)); numpy byte-identical (return unchanged) | Task #98 |
| **Track E** IsochroneApprox backend (orbit-STM path) | E | → feat/backends | ⏳ **#1020** open 2026-06-27 | orbit-path method (no interp): stacked backend IC vector → in-backend integrator → isochrone AA → angle-average; `_backend_roll/_dangle/_dePeriod/_median` helpers. **+non-axi** (Lz via danglephi-weighted avg, angle-fit design matrix w/ gphi·anglephiT). triaxial-LogHalo parity ~1.7e-12 | Task #96 + user non-axi ask |
| **Track E/0** backend.interpolate map_coordinates + spline deriv/point-eval | 0/E | → feat/backends | ⏳ **#1021** open 2026-06-27 | extends galpy.backend.interpolate with the 3 AA-grid primitives: 1D `nu=` deriv, 2D `grid=` point-eval (Spline2D == RectBivariate.ev), ndimage cubic map_coordinates (spline_filter passthrough + order=3 + MapCoordinates). scipy-fit→backend-eval, no FITPACK reimpl | numpy byte-identical; jax/torch ~1e-16 vs scipy; grad-vs-FD; 96 tests, 100% cov. Unblocks the grid family. Task #97 |
| **Track E** Staeckel/Adiabatic Grid backend | E | → feat/backends | ⏳ **#1022** open 2026-06-27 (stacked on #1021) | additive jax/torch eval for StaeckelGrid (`_evaluate` + `_EccZmaxRperiRap`) + AdiabaticGrid (`_evaluate`); backend interp wrappers built from the SAME fitted scipy objects; masked writes→xp.where/clip. **On-grid only** (no per-point off-grid fallback under jit; documented in docstrings). actionAngleStaeckelSingle kept (still used) | numpy byte-identical (assert_array_equal, 40 ICs, 8 outputs); parity 1e-10; grad-vs-FD ~1e-10; 10 new + 16 grid tests green. **Last Track E breadth item.** Task #95 |
| Orbit-milestone xfail-ledger prune | 0 | → feat/backends | ✅ **#1006** merged 2026-06-24 | full-matrix regen 28083178345 (all 45 shards complete); jax 289→287, torch 1029→961; 70 drops, 0 adds (test_orbit 31t+2j, test_orbits 31t, +6 incidental) | surgical prune off the 2026-06-22 milestone-2 base; caught stale-main-repo-ledger trap → diff vs origin/feat/backends |
| Track G figures/diagnostics | G | (cross-cutting) | ⏳ continuous | g0–g12 in manifest | g12 galpy in-backend orbit GPU throughput (vs C/jax-CPU/torch); g2/g6 re-measured on TITAN Xp; g9–g11 variational/Lyapunov/dissipative; found+fixed lyapunov 3D restart bug (z←vT aliasing, #944) |
| **Track E breadth COMPLETE** (milestone) | E | → feat/backends | ✅ **2026-06-28** — #1016–#1022 all merged | every action-angle method migrated or explicitly guarded: Isochrone/Harmonic/Vertical/Spherical/Staeckel(+vectorized,jit)/Adiabatic/IsochroneApprox(+nonaxi)/Staeckel·Adiabatic·Grid migrated; VerticalInverse guarded (under-development); Torus guarded (permanent out-of-scope, external C++). interp helper (map_coordinates) + estimateDelta/BIsochrone migrated | next: milestone ledger prune (#85) + optimization phase |
| **Opt** Orbit C-STM multi-orbit (one stacked solve) | D/opt | → feat/backends | ✅ **#1024** merged 2026-06-28 | lift the single-orbit guard in Orbits._integrate_cstm: Orbit(N backend ICs).integrate(dxdv-C-method) → one stacked differentiable C-STM (size,nt,6); single-orbit byte-identical; machinery (wrapper+c_stm_forward) already batched | forward bit-identical to loop (Δ=0); gradient-exact (jacrev diag==single, off-diag==0); ~1.65× N=8 CPU. First optimization-phase PR. Enables IsochroneApprox batch (#1025) |
| **Opt** IsochroneApprox one batched Orbit solve | E/opt | → feat/backends | ✅ **#1025** merged 2026-06-28 | _parse_args_backend forward+backward built one Orbit PER ic and stacked getOrbit()/accessor outputs → now one Orbit(no,6) + getOrbit() columns (one stacked C-STM solve for fwd & time-reversed leg). backward reads getOrbit()[:,1:,col] (per-orbit accessors are single-orbit-only on multi-orbit Orbit; == accessor values at query times==tsJ) | numpy byte-identical (_parse_args_backend gated on is_backend_array; numpy stays on per-orbit loop); existing isochroneapprox_* battery covers it (3-orbit grid parity + grad-vs-FD); **~6.1× N=8 jax actionsFreqsAngles (47.4s→7.5s median CPU)**. Task #104 (consumes #1024) |
| **Opt** P0-1 roll bisect_root → lax.fori_loop (jax-traced) | infra/opt | → feat/backends | ⏳ **#1026** open 2026-06-28 | shared bisect_root halving loop rolled into lax.fori_loop ONLY under a trace (isinstance jax.core.Tracer); eager(CI)+torch+numpy keep the Python loop. Every backend turning-point solve flows through it (Staeckel×2/Spherical/Vertical/Potential.rl/IsochroneInverse) | **eager/traced branch MANDATORY**: naive roll regresses eager ~9× (137 vs 16ms); measured eager unchanged, **jit compile 592→152ms (~4×), jaxpr 717→41 (~17×)**, traced bit-identical to unrolled at equal n (Δ=0). numpy untouched (scipy). Roadmap P0-1. Defer static-step-count-thru-callers half. Task #105 |
| **Opt** in-backend ODE native batch (#99) | D/opt | → feat/backends | ✅ **#1027** merged 2026-06-28 | diffrax/torchdiffeq path was single-orbit-only → NATIVE batch axis on the TRAILING component axis (y[...,k], stack axis=-1, explicit dim) so N orbits integrate in ONE adaptive solve (shared saveat+controller); _integrate_inbackend flattens raw _ic_backend to (size,phasedim), moveaxis (nt,size,pd)→(size,nt,pd). NOT outer vmap (torchdiffeq has none; jax.vmap re-traces) | numpy untouched (backend-array ICs only); batch & (2,3)-grid match per-orbit 1.2e-10, grad off-diag exactly 0.0; **~8.4× N=8 jax CPU (1.67s vs 13.9s loop), ~flat in N**; GPU big win = Track-G follow-up (TITAN Xp sm_61 impractical for diffrax timing). Review fixes: per-orbit-t guard; max_steps-shared docstring. **+off-grid eval for batches** (user ask): _call_internal_backend_interp splines each orbit on the SHARED grid → o.R(t)/o.x(t)/o(t) off-grid work (N,nt)/(N,) differentiable; fixes #1024 C-STM path too; single-orbit byte-identical. 119+67 tests green. Unlocks #102. Task #99 |
| **Opt** Augmented 42-state C-STM integrator (#100) | D/opt | → feat/backends | ✅ **#1028** merged 2026-06-29 | `c_stm_forward` six 12-state `integrate_dxdv` solves (re-integrating the base per column) → ONE augmented 42-state C solve: base (6) + six deviation columns (36). New C `evalRectDeriv_stm` (42-state RHS, term-for-term identical to `evalRectDeriv_dxdv` but base accel + 3D-Hessian + dissipative `jac_x/jac_v` computed ONCE/step, applied to all 6 columns via `J=[[0,I],[K+dF/dx,dF/dv]]`) + `integrateFullOrbit_stm` (dim=42) + Python `integrateFullOrbit_stm_c`. cyl→rect `jac0.T` seed, fold `M[t]=jac_t⁻¹·dev[t]ᵀ`, `M[0]=I`. Old six-solve kept as `_c_stm_forward_loop` (bit-identity ref) | **~5.8× faster** (MWPotential2014 nt=1000: 0.250→0.043s). Adaptive joint-42-state error norm → M matches per-column ref ~1e-10 (gradient unchanged); fixed-step bit-identical; numpy `Orbit.integrate` untouched. FD-of-flow conservative 3.15e-9 + **dissipative 1.65e-9** (non-symmetric velocity-coupled J); parity over edge/turning-pt/phi=0 ICs; 7 new tests green under `-W error::Deprecation`. Task #100 |
| **Opt** IsochroneApprox via diffrax (#102) | E/D opt | → feat/backends | ✅ **#1029** merged 2026-06-29 | `integrate_method='diffrax'/'torchdiffeq'` already routed IsochroneApprox through the in-backend ODE solver and matched the C-STM path on actions+**first** derivs; the blocker for **second** derivs was the diffrax adjoint, not a flag. C-STM is pure_callback/autograd.Function (1st order); diffrax default `RecursiveCheckpointAdjoint` is reverse-only (`jax.hessian`=jacfwd∘jacrev hits missing fwd-mode; jacrev∘jacrev hits backward while_loop). Fix = thread `adjoint`/`solver`/`max_steps`: `Orbit.integrate(inbackend_kwargs=…)` → `_integrate_inbackend` → `integrate_orbit` → diffeqsolve/odeint; `actionAngleIsochroneApprox(integrate_kwargs=…)`. `adjoint='direct'` (`DirectAdjoint`, scans max_steps) unlocks `jax.hessian`; torchdiffeq plain odeint double-backprops already | **jax.hessian d²jr/dvR² = 0.7534993413 vs FD-of-grad 0.7534993487** (MWP2014); torch double-grad agrees; value+grad parity with default adjoint EXACT. adjoint passed only when set → default diffeqsolve byte-for-byte unchanged; numpy `Orbit.integrate` byte-identical (69 backend tests unchanged). estimateBIsochrone auto-b is stop_gradient'd → 1st-order only (explicit b for Hessians; follow-up to lift). +8 tests (AA parity/hessian jax+torch; orbit solver/adjoint/max_steps plumbing + per-orbit-t + error paths). Task #102 |
| **AA backend-native redesign** (drop numpy_island) | E | → feat/backends | ✅ **#1038** merged 2026-06-30 | replace `@numpy_island` with resolved-namespace dispatch so `test_actionAngle.py` runs the REAL jax/torch path under a forced backend (was "hollow green"). `is_backend_array`/`get_namespace` at the private layer | closes the #1033/#1034 anti-pattern; AA backend tests genuinely exercise the backend. Task #113 |
| Harmonic/Isochrone Inverse test reductions | E | → feat/backends | ✅ **#1039** merged 2026-06-30 | backend-agnostic `_to_numpy` reductions un-ledger 5 torch Inverse tests | Harmonic+Isochrone Inverse work under all backends (parity+grad). |
| **Staeckel jit-fast** (bisection→lax.fori_loop) | E/opt | → feat/backends | ✅ **#1040** merged 2026-07-01 | `bisect_root` dispatches to a rolled `lax.fori_loop` under a jax trace (direct Staeckel turning-point callers) + `iterate_bracket` rolls the 6 bracket loops. eager/torch/numpy keep the Python loop (byte-identical) | **jaxpr 47,556→2,283 (~21×), jit compile 134s→2.6s (~52×)**, jit-vs-eager 1.6e-16. `under_jax_trace`→`_namespaces`. Tasks #93/#106 |
| **AA-perf** (jacobian hoist + recompute-elim) | E/opt | → feat/backends | ✅ **#1043** merged 2026-07-01 | Staeckel `_staeckel_jacobian` once (was 2× in actionsFreqsAngles) + Isochrone sqrt hoists + StaeckelGrid potentialStaeckel(u0,π/2) dedup + Spherical Or/Op & angler/anglez radicand-share. numpy byte-identical | audit (workflow): only A1 (jacobian hoist) is a real cross-backend win; rest byte-identical hygiene/eager-only. M6/M7 dropped (numpy.sum reorder broke byte-id). Task #114 |
| **AA-wave ledger prune** | 0 | → feat/backends | ✅ **#1045** merged 2026-07-01 | full-matrix REGEN 28516574227 (all 45 shards complete); **jax 273→233, torch 916→886**; 70 drops (df tests that compute actions now run real jax/torch aA), 0 adds | reviewed: 70 deterministic drops; 1 unseeded-RNG (test_impulse_deltav_general_curved) RETAINED xfail. Task #115 |
| **Track F Pdf.1** df foundation | F | → feat/backends | ✅ **#1046** merged 2026-07-01 | backend-migrate `surfaceSigmaProfile` + `jeans` (sigmar/sigmalos); establishes the df dispatch + backend-quadrature pattern. numpy byte-identical; jax/torch parity ≤2e-9; sigmalos `x=√(R²+s²)` substitution cancels the 1/√(x²−R²) endpoint singularity | **leaf-vs-consumer hazard surfaced**: leaves consumed by still-numpy diskdf/sphericaldf use the `is_backend_array else numpy` data-guard (TRANSITIONAL — remove once Pdf.2/3 migrate consumers). Tasks #116/#117 |
| **Track F Pdf.2** spherical DFs | F | → feat/backends | ⏳ **split by family** 2026-07-02: base **#1052** + **#1053** isotropic + **#1054** constantbeta + **#1055** osipkovmerritt + **#1056** eddington+king | Review-driven restructure (helpers relocated to `galpy.backend`: as_numpy/exit_cast/resolve_namespace; astropy-8 exit_cast fix). #1052=base (sphericaldf+isotropicHernquist pilot); 4 family PRs off it, each df files + `test_backend_<family>.py`, all numpy byte-identical + jax/torch backend tests green. Family PRs target the base for clean per-family diffs → retarget+rebase onto feat/backends after #1052 merges (backend CI runs then) + one migration-flip ledger-add commit (10 torch/3 jax; see logs/pdf2_family_split/) | numpy-only remnants confirmed: general Eddington fE, general-β>0.5 Hernquist hyp2f1 (liftable via hyp2f1 Pfaff/Euler ext PR); osipkovmerritt-powerlaw direct-int torch floor ~7.8e-3 (tighten-vs-ledger, task #125) |
| **Staeckel action gradients** (Track E deriv fix, PR A) | E | → feat/backends | ✅ **#1047** merged 2026-07-01 | plain-GL VALUE kept (numpy tobytes-identical, C parity) + gradient grafted from t²-substituted donor (`_staeckel_t2_action` + `graft_gradient`, gated under_jax_trace/under_torch_grad). d(jr,jz)/d(x,v) 1.4e-4–4.7e-4 of FD gold (was 1–6% naive; jac-chain graft ~6× wrong on ∂Jr/∂R — u0=ux(R,z) geometry missed) | diagnosis workflow wf_c80e7340-0dd named the missing term to 4e-12; first-order only (freq/angle grads + Hessians = Phase 2). Follow-ups: PR B StaeckelGrid, PR C c=True, d/dθ test. Task #119 |
| **AA grad probes + Adiabatic jax d/dR fix** (PR D) | E | → feat/backends | ⏳ **#1048** open 2026-07-01 | probes CONFIRM Spherical/Vertical/Adiabatic action grads (4e-10–2e-8 vs FD); xfail(strict=False) Phase-2 flags with baselines: Staeckel dΩr/dR off ×(−54), grafted-action Hessian off ≤500×; FIX: _BatchedVerticalPotential float(R) concretization (jax d/dR errored) | Phase 2 = task #120 |
| **StaeckelGrid true-action gradients** (PR B) | E | → feat/backends | ❌ **#1049** CLOSED unmerged 2026-07-01 (review) | spline-AD grid grads were 12%–290× off FD-of-direct (bug-level); donor graft at query points → ≤5.2e-4; value stays interpolated; numpy tobytes-identical | stale test dropped its staeckel case (asserted AD==FD-of-interpolant) |
| **c=True differentiable Staeckel actions** (PR C) | E | → feat/backends | ❌ **#1050** CLOSED unmerged 2026-07-01 (review) | C forward value (pure_callback under jax trace, stop_gradient into the callback) + donor gradient; c=True backend forward == numpy c=True exactly; grads 4.6e-4; numpy 72/72 tobytes-identical | freqs/angles values pass through ungrafted (Phase 2) |
| **c=True C-native Staeckel gradients** (PR C-v2) | E | → feat/backends | ✅ **#1051** merged 2026-07-02 | C entry actionAngleStaeckel_actionsJac assembles full d(jr,jz)/d(x,v) (Ĩ3U + new dJz/du0 integrand) + jax.pure_callback custom_vjp / torch autograd (C-STM pattern; NO python-donor fallback). C-Jacobian vs donor 1.2e-12; numpy c=True tobytes-identical (200 orbits); 50+72 tests | replaces closed #1050 per user directive; first-order only (freqs/angles=Phase 2 #120) |
| **c=True C-native Staeckel FREQUENCY gradients** (#131 PR-A) | E | → feat/backends | ✅ **#1060** merged 2026-07-04 | fused C (5,5) Jacobian d(jr,jz,Ω)/d(x,v) via the 6 Leibniz dJ integrands' derivatives → 2×2 freq-Jacobian chain; actionsfreqs_with_jac custom_vjp/autograd; first-order only. numpy byte-identical | extends #1051 actions→freqs; angles = PR-B |
| **c=True C-native Staeckel ANGLE gradients** (#131 PR-B) | E | → feat/backends | ⏳ **#1063** open 2026-07-04 | stacked C (8,5) Jacobian: angle rows compose PR-A's action Hessians (dΩ/dcoord, d(dI3dJ)/dcoord) + angle-specific partial-integral boundary term [f(ux)/√S(ux)]·dux/dcoord; φ analytic (d(angle_φ)/dφ=1 via remainder). numpy byte-identical, first-order | **codecov CLOSED**: deleted 3 dead fns (c_value×2 + _staeckel_c_forward_values, orphaned by rewiring) + high_v_panel/degenerate C-line tests. #131 action-angle deriv chain COMPLETE |
| **Track F Pdf.4 qdf PR-4a** (backend __call__ + _rg) | F | → feat/backends | ⏳ **#1064** open 2026-07-04 | quasiisothermaldf `__call__` (log/non-log, lz<0/NaN sentinels) + `_rg` resolved-namespace dispatch (Spline1D eval; leaf data-guard keeps numpy callers numpy); flows #131 Staeckel c=True action grads through the DF core. numpy byte-identical | **codecov CLOSED**: test_rg_no_precompute covers the _precomputerg=False branch (line 3076 → potential.rl). grad-vs-FD rtol 8e-3 (1st-order Staeckel floor). Next: PR-4b GL-moment tiling |
| **Track F Pdf.4 qdf** moment engine + Adiabatic C-native | F | → feat/backends | ✅ **#1070/#1071** merged 2026-07-05 | Adiabatic C-native actions PR-2a (#1070) + qdf backend moment engine + pv-projections (#1071, builds on #1064 eval+_rg). differentiable disk-DF moments. numpy byte-identical | Pdf.4 qdf breadth done |
| **Track F Pdf.3 diskdf/evolveddiskdf** | F | → feat/backends | ✅ **#1073** (diskdf eval+moment) **/ #1076** (evolveddiskdf differentiable-through-evolution) merged 2026-07-06 | #1073 diskdf eval cores + moment quadrature; #1076 `_buildvgrid_backend` = ONE vectorized multi-orbit integrate so moments differentiate ∂/∂R THROUGH the orbit evolution. numpy byte-identical; grad-vs-FD through evolution | +#1074 Staeckel forced-backend→C-native routing. diskdf-family inline `is_backend_array` data-guard (transitional, task #143) |
| **in-backend ODE: planar/composite/1D** | D | → feat/backends | ✅ **#1075** merged 2026-07-06 (+**#1079** 1D test) | `_eom_rhs` single-object dispatch: native-planar (planarForce) + composite + 1D linear (`dim==2`, `_evaluatelinearForces`) all route through diffrax/torchdiffeq. #1079 adds 1D linear-potential test coverage (native-linear/toVertical/composite, jax+torch parity + grad-vs-FD) | composites-not-lists (user); evolveddiskdf evolution rides this |
| **DF+AA milestone ledger prune** | 0 | → feat/backends | ✅ **#1077** merged 2026-07-07 | full-matrix regen; **jax 255→109 (−146), torch 896→738 (−158)**; DF-family + AA tests that now run real jax/torch actions/moments drop out; retained flaky RNG xfail (test_impulse_deltav_general_curved #66) | **baseline 2026-06-16 jax 324/torch 1670 → now jax 109/torch 738** (jax −2/3, torch −56%). plaw-OM forced-sample intermittent shard flake tracked (#126, benign near-knot backend-grid ULP flip) |
| **Pdf.5 streams** keystone wave | F | → feat/backends | ✅ **#1083**/#1084/#1088 merged 2026-07-08; **#1085** rebased+open; **#1090** streamgapdf open | Streams resumed. #1083 `coords` transforms backend-agnostic (byte-identical, 18 arrays exactly equal) + the torch-streamdf-shard hang fix. **Hang root cause (Fable-confirmed):** torch+`os.fork` deadlock — `util.multi.parallel_map` forks while torch's OpenMP threadpool is live → deadlocked children → `multiprocessing` atexit join hangs pytest post-summary; NEW to #1083 (forced-torch coords tensorize streamdf hot loop → torch pool active at fork). **Fix = conftest force-exit mitigation** (daemon backstop timer armed before sessionfinish yield + SIGKILL leftover children before os._exit, gated jax/torch); real cure = migrate streamdf. #1085 einsum stacks via `xp.stack([...],axis=0).T` (F-contiguous byte-identical). #1088 de-flakes plaw forced-sample (near-knot ULP flip) | streamgapdf (#1090) in flight; streamspraydf integrate=True (#152) needs symplectic C-STM (#151) |
| **constantbetadf: eliminate `_rforce_jax`** (unify DF autodiff) | F/A | → feat/backends | ✅ **#1089** merged 2026-07-08 (maintainer-requested) | fE force divisor → backend-agnostic `evaluateRforces` (not per-potential `_rforce_jax`); `autodiff_ops(xp)` = single (grad,vmap) dispatch (jax OR torch). Deletes `_rforce_jax` from 8 pots; constantbetadf now works for ANY spherical potential (Jaffe test); un-ledgers 5 torch. **numpy fE NOT byte-identical** (flagged, user-approved): float-op reassociation at the rounding floor — Plummer exact, Hernquist 6e-16, NFW 5e-14 (x64). interpSpherical ~3% tail bug-fix (linear+clamp→cubic+Kepler). `_namespaces` +AttributeError (jax vmap tracer SingleDeviceSharding). +notebook-CI array-api-compat install + install-docs + torch-only-fallback test (pragmas removed) | follows #1087; leaf density derivs data-first |
| **streamgapdf** analytic impulse kernels | F | → feat/backends | ⏳ **#1090** open 2026-07-08 | plummer/hernquist ±curved + HernquistX + `_rotation_vy` backend-agnostic + differentiable (Streams PR-4, island-free). numpy tobytes-identical (10/10 vs baseline); jax/torch parity 40 each + grad-vs-FD. **data-first dispatch** (`namespace_from_arrays or numpy`) — a get_namespace-under-forced-torch version crashed `torch.stack([numpy.float64])` in the numpy streamgapdf setup (test_hernquist); fixed. Made ~13 torch streamgapdf tests newly pass. PR-2 (scipy-quad general* via backend.quadrature) + PR-3 (Orbit-integration variants) = follow-ups | leaf-migration-data-guard; carries pre-#1088 OM flake until re-run/rebase |
| **differentiable stream track** (streamTrack) | F | → feat/backends | ✅ **#1098**/**#1099**/**#1100** merged 2026-07-09 | Public differentiable stream track from a backend particle spray: `spdf.streamTrack(particles=<backend xv>)` → backend track + covariance, torch end-to-end. **#1098** `galpy.backend.interpolate` smoother (`make_smoothing_spline` GCV + `UnivariateSpline` FITPACK reuse, frozen linear operator `y→fit`, λ/knots/p stop-gradient hyperparam). **#1099** backend track assembly (`_bin_by_tp`/`_smooth_series`/`_fit_one_pass`) + `psd_project` (frozen-eigenvector nearest-PSD, avoids singular eigh JVP). **#1100** backend-aware `StreamTrack` class (`cubic_spline_coeffs` not-a-knot == IUS). numpy byte-identical throughout | frozen-operator differentiable reconstruction; jax.grad blocked by cKDTree structure → torch eager end-to-end + frozen-unit jax grad |
| **∂track/∂progenitor** (streamTrack) | F | → feat/backends | ✅ **#1101** merged 2026-07-11 | Backend progenitor curve threads through `_fit_track_from_particles` → `d(track)/d(progenitor)` (C-STM-achievable: dense differentiable prog curve; prog enters as offsets = particles−prog_at(ta) AND track = prog_grid+smooth, both via in-backend cubic spline). numpy byte-identical (max\|Δ\|=0 vs base, 6 configs). Frozen-GCV FD-vs-AD 1.4e-8, jax==torch 9e-16; torch end-to-end grad. Adversarially reviewed (byte-identity + grad + edge cases, all clean) | part 1: the fit-level threading |
| **∂track/∂progenitor part 2** (streamspraydf) | F | → feat/backends | ✅ **#1102** merged 2026-07-11 | Public `spdf.streamTrack()` builds the backend `track_prog_cart` when the progenitor Orbit has a backend IC: integrate fwd+back via `Orbit.integrate(dop853_c)` → **C-STM** (the tie-in already exists for RK dxdv methods; only symplectic default excluded, #1094 guard) → xp.flip/concat stitch. Adversarial review: numpy byte-identical (43043-value array_equal, max\|Δ\|=0), stitch=1e-12, track=3e-9 at common tp, torch e2e grad + curve FD 1e-10 | discovered the C-STM tie-in is DONE (backend orbit + dop853_c → `_integrate_cstm`); **next = symplectic-default C-STM routing fix (grad-tracking guard, #1094-safe)**, then in-backend ODE for ∂track/∂θ, then Track G |
| **symplectic C-STM routing** (Orbit.integrate) | D | → feat/backends | ✅ **#1103** merged 2026-07-11 | grad-tracking backend IC + symplectic dxdv C method (symplec4_c/leapfrog_c/symplec6_c) now auto-routes to the C-STM (was ValueError). #1094-safe via `not _ic_backend_concrete` gate: CONCRETE backend ICs (forced-backend suite) keep numpy/C, only GRAD-TRACKING route. Adversarially reviewed (#1094 NOT reintroduced: forced-backend spray routes only dop853_c, identical to base; grad-tracking symplec4_c matches numpy 0.0 + FD 1e-9; 282 backend + 112 numpy orbit tests pass). Pre-existing GPU-only concrete-CUDA edge noted | user flagged the exclusion; refines [[default-method-routing-reroutes-internals]] (you CAN distinguish via _ic_backend_concrete) |
| **stringent stream-track grad-vs-FD** | F | → feat/backends | ✅ **#1104** merged 2026-07-11 | Test-only: upgrade the four ∂(stream track) grad tests that asserted finite-and-nonzero (merged #1100/#1101). Frozen-unit tests → quadratic loss + random-directional grad-vs-FD (rtol 1e-5); end-to-end tests → production grad == FD-checked frozen-structure grad (particles reconstruct GCV-frozen cyl→cart→interior→mean+cov→psd_project closure). **User feedback: never finite-and-nonzero, always grad-vs-FD** ([[stringent-gradient-tests]]) | also strengthened #1102 (frozen-composition d(track)/d(prog IC), jax 5e-8/torch 7e-8) + #1103 (symplectic C-STM directional FD) in-place |
| **constantbetadf: cover L679** | F | → feat/backends | ⏳ **#1105** open 2026-07-11 | PR#892 codecov flagged 1 uncovered line: `constantbetadf.py:679`, the forced-backend branch of `_evalpot_asnumpy`. **Rebase artifact**: the Emin/potInf calibration was left on an inline copy of that coercion, so the helper's forced branch was only reachable via `_fEintegrand_raw` (numpy-only quadrature; forced backend uses the GL path) → unreachable. Fix: route calibration through `_evalpot_asnumpy` (dedup −15/+6, byte-identical numpy: `_potInf`/`_Emin` == original `_evaluatePotentials`); L679 now covered by existing `test_general_constantbetadf_fE_backend` (MISSING→COVERED) | user flagged; root-cause fix not band-aid test |
| **streamspray: fully jittable + differentiable (generative ∂track/∂θ)** | F/D | → feat/backends | ⏳ **#1106** open 2026-07-11, **expanded 2026-07-12** | **Option A: jit the real stateful class.** `fardal15spraydf(...).sample()` AND `.streamTrack()` now run under `jax.jit` (CPU+GPU) and give the **physical generative** `d(track)/dθ` — all N sample orbits + progenitor integrated on the backend, `xv=xv(θ)` FLOWS (spray particles are a theory intermediate, NOT frozen data; the earlier `particles=` frozen-xv gradient is non-physical for d(stream)/dθ — user correction). Construction routes progenitor to in-backend ODE (`_integrate_progenitor`, jit-safe detection, numpy-time grid → `self.t` concrete); k-vector noise via jax RNG key (reparam); `_fit_track_backend_jit` = jax-native reconstruction (static argmin + hat-basis P-spline + GCV λ + PSD-by-construction cov, structure `stop_gradient`'d). **Exact-match**: auto `track_time_range` (true traced particle-extent via normalized-axis + `tp_scale`), covariance, GCV λ. **C-STM preserved** (#1102: dop853_c for concrete backend IC). `_finalize_pot_args` byte-identical C-path coercion retained. `o.x(t)` jit fix. All grad-vs-FD **h-converged**: sample 2e-6, streamTrack 3.3e-7 (CPU)/1e-7 (GPU), extent 3.3e-7, cov 3.95e-6, GCV 4.9e-7, tp_scale 2e-11; numpy **byte-identical** (maxdiff 0); 35 backend + 38 numpy tests pass | **streamspray FULLY jittable+GPU+differentiable, no compromises** ([[stream-track-two-gradients]]); Track G artifact regen with the correct generative gradient unblocked ([[trackg-showpiece-artifact]]) |
| **@backend_input boundary decorator (explicit coords)** | 0/A | → feat/backends | ✅ **#1196** merged 2026-07-25 | 154/154 CI green; test_potential 1320 numpy/jax/torch **0 fail**; unforced test_backend shard 6251 **0 fail**; byte-identity hash == baseline; **no ledger change** | Extracts coercion OUT of `potential_physical_input` (now units-only, matches main) into `galpy/backend/_input.py`; **73 potential-side sites each DECLARE their coords** (`@backend_input("R","z","phi","t")`), validated at DECORATION time. Old block blanket-coerced every positional arg → `dR`/`dphi` derivative ORDERS, `forceint`, `M`, `nsigma`, `integrate_method` (a string), grid objects. **2 real bugs found+fixed**: (a) sequence coord `v=[vR,vT,vz]` was stacked by one `asarray` → **detached grad-carrying elements from the autograd graph** (broke 6 NonInertialFrameForce grad-vs-FD); now coerced ELEMENT-WISE keeping the container; (b) mixed backend+numpy coords (`Orbits.E` passes numpy `t=` with torch `R`/`z`) raised `Multiple namespaces` **on the numpy path** → namespace now follows the backend coords. Also: `**kwargs`-only coords declarable (`Force.rforce` phi/t regression). **Gate removal TRIED + REVERTED** — see notes | `_check_backend_compatible` kept as **TEMPORARY `_backend_ready`**: removing it broke 4 things — df (`evolveddiskdf` grid `deriv=`, `qdf.sample()`) turned WORKING calls into raising ones; `interpRZPotential` grid-edge divergence (force 3.3%, 2nd deriv 32%, float64); `MovingObjectPotential` under jit (`_orb.R(t)` numpy lookup gets a `DynamicJaxprTracer`; baseline gets a plain `float`). Guard also made **every df decoration a dead no-op** (returns False for any non-Force) → df left undecorated. Footprint is auditable: `_UNDECORATED_MODULES`/`_UNDECORATED_BY_DESIGN` in `tests/test_backend_input.py`, test-enforced = the burndown list. Audit is blind to VARARGS entry points (actionAngle `actionsFreqs(*args)`) — AA uses 80 `get_namespace` + 38 `promote_scalars`, **0 `coerce_coords`** |
| **jit trace mode** (`--jit`) | 0 | → feat/backends-develop | ✅ **#1200/#1201/#1202/#1205** merged 2026-07-26/27 | Runs the ENTIRE suite under `jax.jit`/`torch.compile` at the `@backend_input` seam. `under_trace` + `untraceable_setup` for the scipy-setup/quad dispatches; RazorThin + TwoPowerTriaxial made traceable; inductor fused-kernel `expm1` cancellation guarded (`_NOT_TRACEABLE` now EMPTY) | this is the harness every row below depends on |
| **jax-jit ledger burndown** | A | → feat/backends-develop | ✅ **#1206/#1208/#1209/#1210/#1211/#1212/#1213/#1214/#1215/#1216/#1217** merged 2026-07-27→29 | 2nd-deriv migrations (log spirals, RazorThin), noGL ellipsoidals via backend quadrature, `dePeriod` de-duplication, one f(E) spline builder on sphericaldf, constant-beta fE limits on-backend, whole-line surfdens + RazorThin degenerate radii. **jax-jit ledger 100 → 49 → 28 → 15 → 5**; `<backend>-jit` now INHERITS `<backend>` | traced `test_sphericaldf` 223 → **0** failures |
| **torch.compile silently-wrong surfdens** | A | → feat/backends-develop | ✅ **#1218** merged 2026-07-29 | `_quad_needs_backend` used a `float()` probe; dynamo makes that symbolic, so torch ran the **scipy branch inside the compiled region** and `AnyAxisymmetricRazorThinDisk.surfdens` returned **2.235e-15 instead of 0.0662, order-dependently**. Gate now asks `under_trace` (i.e. `torch.compiler.is_compiling()`), which dynamo cannot fool. Second site (`RazorThinExponentialDisk._R2deriv`) now returns NaN off-plane on BOTH tracers | found by running `--backend torch --jit` for the first time — the ledger had 51 jax / 5 jax-jit / 99 torch / **0 torch-jit** entries |
| **DoubleExponentialDisk array-safe** | A | → feat/backends-develop | ✅ **#1219** merged 2026-07-29 | `node_axis` on the closed-over R/z + `axis=-1` on the Ogata reduction; both no-ops for scalars. `check_potential_inputs_not_arrays` grows an opt-in (`_backend_accepts_arrays`) gated on every array arg being a BACKEND array, so the numpy scalars-only contract is unchanged. **AnyAxisymmetricRazorThinDisk deliberately does NOT opt in**: its traced GL degrades 1.6e-12 (\|z\|=1e-2) → 2e4x (1e-5) and the Poisson rule clusters nodes at z=0, so opening it would turn a loud TypeError into a silently wrong ~0 | numpy `test_potential` 1326/0; jax --jit 1326/0; torch --jit 8/0 |
| **traced-surface measurement** | 0 | (measurement, no code) | ✅ 2026-07-28/29 | torch `--jit` `test_orbit` **581 tests / 0 fail** (last unmeasured big surface). **jax `--jit` `test_orbit` measured 2026-07-31: 581 / 3** — 2 are 1800s *timeouts* (`test_integrate_indiv_t_1D/2D`, performance not correctness) and 1 (`sigint_fulldxdv`) is a LOCAL signal-delivery artifact, confirmed PASS in CI artifacts. Sweep of 13 never-traced suites: **309 tests / 2 fail**, both a characterised interpRZPotential over-tolerance (root cause: **XLA compiles NFW's `log1p(r/a)/(r/a)` differently from eager** — jax eager is bit-identical to numpy, only the jitted value moves) | remaining blind spots NAMED: `test_streamTrack` hidden by a FILE-level slow-skip (measured separately 40/45 pass); `test_noninertial` does not finish in 45 min traced |
| **eager-ledger burndown wave** | A | → feat/backends-develop | ✅ **#1220/#1221/#1222/#1223** merged 2026-07-29→30 | `test_ExpDisk_special` off-plane NaN contract; **1D linearPotential coerces coords under a forced backend** (7 torch tests, pre-existing); ledger reading-note; `test_quantity` expresses `turn_physical_off`'s contract backend-agnostically (`*_turnphysicaloff` ×6) | ledger 156 → 150 |
| **stale-ledger sweep from CI artifacts** | 0 | → feat/backends-develop | ✅ **#1224** merged 2026-07-30 | xfail is non-strict, so a FIXED entry reports XPASS, the shard still exits 0, and it keeps inflating the count — nothing goes red. Measured `(ledgered ∧ passed)` over a completed run's 44 `backend-junit-*` artifacts (no regen dispatch needed): **144 observed, 124 genuinely failing, 20 stale**. A local py3.13 run over-claimed by one entry (matrix is py3.14 **and** shards `test_orbit` `--splits 5`) | **ledger 150 → 130** |
| **1D actionAngle physical conversion** | A | → feat/backends-develop | ✅ **#1225** merged 2026-07-30 | `physical_conversion_actionAngle` used `len(out)` to tell "sequence of quantities" from "one quantity" — but a length-N array has a `len()` too, so a 1D `__call__` was scaled against a 3-long factor list: tuple-of-scalars below n=4 and **`IndexError` at or above it, on plain numpy**. `numpy` broadcasting `tuple - ndarray` hid it; jax/torch raise on `tuple - Tensor`. Fixed by dispatching on type | 16 float-byte probes identical; ledger 130 → **128** |
| **hyp2f1 domain: constant-beta DFs unlocked** | A | → feat/backends-develop | ✅ **#1226** merged 2026-07-30 | The Euler-integral fallback needs some `P ∈ {a,b}` with `P>0` and `c−P≥1`; galpy's own anisotropic DFs request three triples it refused. Adds **Euler's transformation** (keeps `z`, so the `z≤0` quadrature applies verbatim — 7e-15) and, where *both* parameters are non-positive and **no** transformation lands one in `(0,c)`, a **Pfaff-variable Gauss series** with `C−A−B=\|a−b\|>0`. In-regime params keep today's route and today's **bits** | **β ≥ 0.5 Hernquist family was numpy-only, now 1.55e-15 vs scipy**; `test_sphericaldf --jit` **223/0 on both backends**; ledger 128 → 122 |
| **Staeckel unbound-sentinel wrap** | E | → feat/backends-develop | ✅ **#1227** merged 2026-07-30 | The backend angle path applied `remainder(θ+φ, 2π)` to the 9999.99 *unbound sentinel*, turning a recognisable flag into an ordinary-looking angle. Guarded with `where(raw == 9999.99, raw, wrap(...))` in both `_torch/staeckel_c.py` and `_jax/staeckel_c.py` | ledger 122 → 120 |
| **assertion-site landing** | 0 | → feat/backends-develop | ✅ **#1228** merged 2026-07-30 | 11 torch entries were tests doing `ndarray OP Tensor` **at the assertion**, not source defects — the exception text is identical either way, so only the failing *frame* separates them. Landed each backend value where it enters the numpy comparison. Caught a jax-only regression mid-flight: `as_numpy(jax)` is **read-only**, so the in-place `+=` in `test_noninertial` raised — torch's cast is writable and hid it | ledger 120 → **109** |
| **`obs=` accepts backend arrays** | D | → feat/backends-develop | ✅ **#1229** merged 2026-07-31 | Under a forced backend a user builds `obs=[o.x(t), o.y(t), 0.]` the natural way and gets backend arrays, while the accessors consuming them are numpy by construction. One `_obs_asnumpy` helper at the four `isinstance(obs, (ndarray, list))` sites. `as_numpy` is scalar/array-only so it could not absorb this | ledger 109 → **107** |
| **potential picklability** | A | → feat/backends-develop | ✅ **#1230** merged 2026-07-31 | Three potentials could not be sent to a worker process: `dens=` defaults were lambdas (in three copies) and the dict-spec Sigma/hz profiles, the axisymmetric `dens` wrapper and the phiME density are closures. Fixed with galpy's own idiom (`ChandrasekharDynamicalFrictionForce`): named module-level defaults + `__getstate__`/`__setstate__` drop-and-rebuild. **A user lambda still raises** — asserted, since a `__getstate__` that silently dropped the density would "fix" the test by discarding the physics | **64/0 picklable** (was 61/3) on scipy 1.17; 858 method values bit-identical; 5 new tests fail on the parent. Surfaced an **upstream scipy 1.18** break: `_xp` (a module) on `PPoly`/`BPoly` makes *any* spline-holding object unpicklable |
| **qdf sampler numpy boundary** | F | → feat/backends-develop | ✅ **#1231** merged 2026-07-31 | The rejection samplers are numpy end to end, but under a forced backend `self(...)` returns a backend array for array coords and the accept step raises `Tensor > ndarray`. Lands the DF value **and** `fmin_powell`'s optimum where they ENTER the numpy arithmetic (4 sites), not at the comparison that trips. `as_numpy` on numpy returns THE SAME OBJECT, so the numpy path is structurally unchanged and the RNG draw order is untouched. A boundary cast, **not** a numpy island — the DF still evaluates on the backend | ledger **107 → 101**; 6 torch entries each verified to fail without it. Samples deliberately NOT asserted bit-identical: `fmin_powell` takes 46 vs 37 evaluations on a backend, shifting `maxVT`'s last bits and hence the proposals; measured 4.5e-10, asserted 1e-8 |
| **scipy 1.18 spline pickling** | A | → feat/backends-develop | 🔄 **#1232** open 2026-07-31 | scipy 1.18 caches the array namespace — a **module** — on `PPoly`/`BPoly` as `_xp`, so any holder raises `cannot pickle 'module' object`. Surfaced by #1230's new round-trip on the py3.14 runner (scipy 1.18; dev env is 1.17). ONE `__getstate__`/`__setstate__` on `MultipoleExpansionPotential` covers both affected classes (DiskMultipole holds them via `_me`); rebuild through the public constructor — stripping `_xp` does NOT work, scipy never recreates it | Radius **measured, not grepped**: 62 potentials pickle, **2** do not; interpRZ/interpSpherical/AA grids/qdf/diskdf/sphericaldf/integrated-Orbit all fine, because only `_interpolate.py`'s polynomial classes gained `_xp`. Bit-identical on scipy **1.17 AND 1.18**, incl. a NON-axisymmetric case — `_I_*_sin` exists only there, so an axi-only check would have shipped silently-wrong non-axi restores |
| **tolerance Group A: shared relative helper** | — | → feat/backends-develop | ✅ **#1250** merged 2026-08-01 | Group A of the approved tolerance batch: one shared relative-comparison helper replacing hand-rolled absolute bounds in test_quantity. Groups A/B approved by Jo for `test_quantity` only; `actionAngleVerticalInverse` and `actionAngleTorus` remain permanently out of scope | ledger **42 → 34** |
| **one shared `_BACKEND_GL_ORDER`** | E | → feat/backends-develop | ✅ **#1251** merged 2026-08-01 (= `c34abfa3`) | Hoisted the backend GL order out of two action-angle modules into one place, and **corrected its documented floor**: for ANGLES the deviation does not converge (3.3e-6 @50, 1.8e-6 @200, 3.5e-6 @400) — raising the constant is not a way to close a backend-vs-numpy angle gap | 152 green after an evicted-runner re-run |
| **AnyAxisym small-\|z\| band** | A | → **main** | ✅ **#1245** merged 2026-08-05 | Two defects. The `m`-clamp did not cure the `1-m` cancellation, it *created* infinities (`ellipk(1.0) = inf`, `R2deriv`/`z2deriv` `±inf` at isolated z). `1-m = ((a-R)²+z²)/((a+R)²+z²)` is exact, so `ellipkm1` takes it directly and the clamp is **deleted**. `zforce`/`Rzderiv` additionally resolve the width-\|z\| peak at `a=R` via `a = R + \|z\|t`. Title/body rewritten 2026-08-04 — the original described the disproved clamp diagnosis | `zforce → -2πΣ`, `Rzderiv → -2πΣ'` verified at two radii; documented `~eps/\|z\|` floor on Rzderiv. numpy values change deliberately. Merged as `cd81fbed` |
| **un-defer 27 jax spherical-DF slow-skips** | F | → feat/backends-develop | ✅ **#1252** merged 2026-08-05 | Track F's spherical-DF migration made these fast, as the list's own comment predicted. Measured, not assumed: 27 pass inside the 300 s cap (slowest 177.6 s). One kept — `test_king_sigmar` at **1028.6 s**, needing KingPotential vectorization rather than the sampling port | slow-skip **230 → 203**; merged as `7b6f5ae4`; shard 588 s green (was 1450 s skipped / 3111 s unskipped) |
| **backend `ellipkm1` + delete the AnyAxisym m-clamp** | A | → feat/backends-develop | ✅ **#1254** merged 2026-08-05 | The AGM fallback formed `b0 = sqrt(1-m)` by subtraction, so `K` was unreachable near `m=1`. New `ellipkm1` takes the complement directly; `1-m = ((a-R)²+z²)/((a+R)²+z²)` is exact and lies in [0,1], so `_ELLIP_M_CAP` is **deleted**, not relocated. Backend-only — distinct from #1245, which deliberately changes numpy values | numpy path byte-identical **measured** (7 methods, 336 values, 294 distinct, sha-equal); forced-backend gate 35/35 jax+torch; A/B shows no jit ledger delta; merged as `7696c88e` |
| **torch brentq Newton-polish** | E | → feat/backends-develop | ✅ **#1255** merged 2026-08-05 (`18ddfd9b`) | The no-grad early return handed back the raw bisection root, so the forward VALUE stopped at the final bracket half-width (~1e-14). Callers forming `sqrt(f(x0))` amplify that: actionAngleVertical's Ω divides by `sqrt(2(E-Φ))` and turned a 4e-14 residual into a 6.4e-10 frequency error, while jax (which always polishes) stayed at 1e-13 | xmax residual −3.87e-14 → +1.39e-17; Ω error 6.364e-10 → 1.279e-13 (= jax); no measurable cost (20.0 vs 20.3 ms/2000 orbits); A/B: 2 torch xfail→pass CAUSED by it, 2 remaining failures pre-existing. **Round 2 `2705e9cd`:** CI caught 3 real regressions (dehnendf nan mean sampled R) — `f(x0)` itself non-finite, so the polish subtracted a nan from a good bracket root; `newton_polish` now steps only where finite. test_diskdf torch 3 FAILED → 129/0. First diagnosis (nan `df/dx`) was WRONG and measured so |
| **forked-child coverage leak** | — | → feat/backends-develop | ✅ **#1256** merged 2026-08-05 (`5e52497a`) | `.coveragerc` sets `concurrency = multiprocessing`, so a process forked by `parallel_map` starts its OWN coverage — and `--cov galpy` is a **pytest-cov CLI flag** the child never inherits, so it measured the TEST module its target lives in. 26 "missing" lines, all in `tests/`, zero in library code. | Fixed in the coverage DATA (`omit = tests/*`), not in `codecov.yml`'s `ignore`, so the data never contains what the run does not intend to measure. A/B on the identical command: 195 files → 194, `tests/` gone, no galpy file lost. codecov patch 80.00% → **100.00%**, project 86.10% → **99.95%**. My first two diagnoses were WRONG — both fitted to a percentage computed on a PARTIAL upload set; the right move is to ask codecov WHICH LINES. |
| **ledger accuracy: 2 source fixes + 3 re-homes + readable logs** | — | → feat/backends-develop | ✅ **#1257** merged 2026-08-05 (`fa80b2f0`) | Ledger 28 → 20. `interpSphericalPotential` froze `_Phi0`/`_Phimax` as backend arrays while `_total_mass`/`_rmax` stayed numpy, so `_revaluate`'s numpy branch mixed the two and raised. SpiralArms `test_Rzderiv` `dx` 1e-8 → 1e-6 **strengthens** the test: the central difference amplified a 1.9e-13 float64 reassociation into 2.5e-6 IN THE REFERENCE. | Five timeouts re-homed xfail → slow-skip (a timeout OVERRIDES xfail and records FAILED, so they were never numerical gaps) — bookkeeping, NOT burndown, and called out as such. #1255 credited with 6 torch actionAngle entries, verified by A/B on a pre-#1255 tree + 4 CI artifact runs. **conftest force-exited BEFORE TerminalReporter**, so every forced-backend log ended in dots with no "N passed" line; deferred to a `trylast pytest_unconfigure`, verified with an atexit probe since `os._exit` skips atexit. |
| **map_coordinates boundary mode** | — | → feat/backends-develop | ✅ **#1258** merged 2026-08-07 (`4727b8c1`) | `test_backend_interpolate.py` 161 passed (numpy+jax+torch); burns the jax StaeckelGrid ledger entry | `map_coordinates` defaulted to `mode='nearest'` while its grids are prefiltered with `mode='mirror'`; a cubic spline filter and its evaluator MUST share a boundary rule. Interior queries never reach an outside tap, so every mode agreed there and no test could see it. At boundary nodes: **6.775 in a log-valued actionAngle grid ≈ 875× in the value**. Surfaced via a circular orbit needing `|JR| < 1e-16`: numpy gives exactly 0.0, backends gave 4.836e-14 — the query lands on the last node in TWO axes. With the fix `JR = 6.04e-24`. | Why it survived: the existing references PINNED scipy to `mode='nearest'`, so both sides shared the wrong rule and agreed by construction. New test asserts filter+evaluate reproduces the input at EVERY node, reporting interior and boundary SEPARATELY so a boundary-only regression cannot hide behind a healthy interior. A/B: fails on all three backends without the source fix. Also removed a `# pragma: no cover` on the mode guard and covered it for real. |
| **jeans semi-infinite quadrature scale** | — | → feat/backends-develop | ✅ **#1260** merged 2026-08-07 (`d8960062`) | `test_backend_jeans.py` 90 passed + `test_backend_quadrature.py`; burns 5 forced-backend jeans failures | The `sigmar`/`sigmalos` data-guard was a numpy island **suppressing 5 real backend defects** — fixing the dispatch alone exposed them, so it could not land alone. Root cause: `fixed_quad_semiinfinite`'s map `s = a - 1 + 1/u^2` **hardcoded its transition scale to 1**, so node spacing near the lower limit is O(1/n²) *absolute*; structure on scale ε≪1 is unresolvable at ANY n and accuracy **collapses as r→0 instead of converging**. Two more: `intFactor` integrated `beta(y)/y` over decades (fixed by the `y = exp(t)` substitution), and the dispatch discarded the forced namespace for plain-float input. | numpy path **bit-identical**: `scale=None` → `L=1.0`, exact in IEEE754, verified `.view(int64)` over 18 cases (2 kinds × 3 integrands × 3 node counts). A 1-ULP "regression" I first saw was **my probe's** associativity (`(w*f)*jac` vs `w*(f*jac)`), not the library's. `scale=R` for surfdens was tried and **REFUTED** — great on Hernquist, 3700× worse on Plummer at R=1e-3 — and reverted along with the dead `scale=` I'd added to `symmetric_quad`. |
| **AnyAxisym graded dyadic quadrature** | — | → feat/backends-develop | ✅ **#1259** merged 2026-08-07 | `test_backend_anyaxisymdisk.py`; burns 4 AnyAxisym jit ledger entries | Public `potential`/`Rforce`/`zforce` hit `if xp.abs(z) < 1e-6` (and `if R < 10.`) on TRACED values → `TracerBoolConversionError`. Replaced with `xp.where`, and the `[0,R]+[R,2R]+[2R,inf)` split is now graded with **dyadic panels** toward the `a=R` (m→1) singularity, issued as ONE batched `fixed_quad` — as separate calls the jit graph is ~16× more expensive to compile. Both degenerate radii guarded so traced matches concrete scipy: R=0 (first two panels zero-width, integrand 0/0 at a=0, tail alone is the whole integral) and R=∞ (all quantities vanish). | Each dead `xp.where` branch is evaluated at a **harmless R** so a NaN cannot leak back through a gradient. Audit found this is **not** a duplicate of `transformed_quad(interior_point=)` — that splits into 2 panels and clusters nodes *within* each via the `X=xi**k` remap; this builds 2K+2 dyadic panels halving toward `a=R` clamped at `dmin~4|z|`, which the within-panel clustering could not resolve. Follow-up: **hoist** (not copy) the dyadic panel builder into `galpy.backend.quadrature` when #202 becomes its second consumer. Also deleted the unreachable `z is None` branch rather than pragma-hiding it. |
| **diskdf/surfaceSigmaProfile numpy island** | — | → feat/backends-develop | ✅ **#1261** merged 2026-08-07 | forced jax+torch: surfacesigma 12/12, evolveddiskdf 35/35, diskdf jax 124 passed·torch 129 passed, **0 failures** | The four R-evaluators carried `get_namespace(R) if is_backend_array(R) else numpy`, which discards the forced namespace exactly when `get_namespace` would have honoured it — a plain float under a forced backend took the numpy branch, so the profile was a **numpy island**. Fixing the leaf alone (`@backend_input("R")` + `_backend_compatible`) **REGRESSES 47 of 129** forced-torch tests: `diskdf` is unmigrated and hands the returned Tensors to `scipy.integrate.quad` / `numpy.fabs` / `numpy.lib.scimath.log` (whose complex result promotes into a complex Tensor `_real` cannot unwrap). | Consumer guards its own boundary: **all 30** numeric call sites route through one `_ssp` accessor. Measured arc **47 → 11 → 0** (leaf-only → 2 sites → all 30), counts from `--junitxml`. numpy **bit-identical AND type-identical** over 16 cases (`.view(int64)`); scalar-in still returns `float64`. `diskdf.py` is the only consumer in the tree, so the boundary cannot be bypassed later. Cost stated not hidden: forced-torch shard 502s → 1384s (2.8×), which is per-call coercion in the velocity-grid loops — the leaf-only arm was already 1203s. |
| **jit-vs-eager helper adoption** | — | → feat/backends-develop | ✅ **#1262** merged 2026-08-07 | `test_backend_disk.py` 240 passed, 4 skipped | Test-only. `test_razorthin_public_methods_jit_traceable` hand-rolled the jit-vs-eager comparison `tests/backend_jit_helpers.py` exists to own. The helper adds the **jaxpr-dependence check** the hand-rolled version could not see: a trace that folds its arguments away returns a constant and compares equal, so a bare value check passes **vacuously** — the same blind spot a numpy fallback has under a forced backend. | Passes `ref=float(call(R0,z0))` to keep the **plain-float** eager reference against traced-array input; letting the helper compute its own reference would have quietly narrowed the test to array-vs-array while looking like a pure refactor. Measured before adopting: **0 of 24** (3 methods × 8 points) trip the dependence assertion — `zforce` is exactly 0.0 at every z=0 point yet still `depends=True`, because the check is *structural* (does the jaxpr consume its invars), not numerical. Verified the assertion **can** fail: a deliberately constant lambda is rejected. |
| **torch-jit streamdf: xfail → slow-skip** | — | → feat/backends-develop | ✅ **#1263** merged 2026-08-07 | measured with the ledger DISABLED (`GALPY_BACKEND_XFAIL_REGEN=1`): 80 tests, 73 passed, 1 skipped | An xfail-ledger entry still **RUNS** the test (`xfail(strict=False)`, not a skip), so a test that **HANGS** cannot usefully be xfailed — it burns the shard's whole budget and lands as a timeout instead of the expected failure. All three torch-jit `test_streamdf` entries hang: `test_streamTrack_multi_parallel` (**deadlock, observed directly**: 12 forked `parallel_map` children all parked in `futex_wait_queue`, parent in `do_wait`, 64 min, no CPU anywhere in the tree), `test_plotting` and `test_fardalwmwpot_trackaa` (Timeout >900s). | Does **not** claim a shared cause — `multi_parallel` is a fork deadlock; the other two block in `waiter.acquire`, possibly a different lock. They share only the property the ledger cares about. First torch-jit entries in a slow-skip file that had 207 entries and none for that backend. Follow-up: the eager-torch `parallel_map` fork fix does **not** cover the `--jit` path. |
| **`parallel_map`: do not fork while traced** | — | → feat/backends-develop | ✅ **#1266** merged 2026-08-07 (`b6bf4b54`) | A forked child inherits the parent's **compiled** kernels and the thread pool they run on, and deadlocks re-entering it — children self-reported via `faulthandler`, every one parked inside the generated inductor `call`. Neither knob helps, both checked: `restrict_to_single_thread()` cures the EAGER case but cannot touch parallelism baked in at compile time; `TORCHINDUCTOR_COMPILE_THREADS=1` governs *compile*, not runtime (A/B: identical timeout, same stack). So while traced, don't fork — mirroring the Windows escape three lines above. | A/B on the 3 slow-skipped streamdf tests: 3× RC=1 timeout → **27.7 s / 14.0 s / 45.4 s all PASS**. numpy byte-identical by construction (`jit_mode()` is "off" outside `galpy.backend.jit`). Explicitly NOT claimed: interpRZ's 2 torch-jit timeouts are `torch.compile` cost, not this deadlock (1530.20 s vs 1548.70 s, ~1% apart) |
| **slow-skip: 100 stale jax orbit entries** | — | → feat/backends-develop | ✅ **#1267** merged 2026-08-07 (`4f8ce41a`) | Lifted all 127 and re-ran **in the CI shard context** (same command `backend-tests.yml` uses, all five pytest-split groups, xfail ledger left active). 114 covered → **100 un-deferred**, 88% of what was measured. Three rules the numbers forced: only <240 s (80% of cap) un-defers, since a test that tips over records FAILED not skipped; **`passed` does not imply "within the cap"** (`--timeout-method=signal` cannot interrupt a long C call — one entry passed at **620 s**); entries no shard covered stay. | slow-skip **210 → 109** across #1266+#1267. Every retained entry now carries its measured seconds. Rewritten file parsed back with the real loader (`_load_backend_nodeids`), not eyeballed. +1 measured `jax-jit actionAngleStaeckelGrid` ledger entry: a coverage asymmetry (ledgered for eager torch, not jax; `-jit` inherits only the same backend), a tolerance case at 9.363e-12 vs an absolute 1e-16 |
| **diskdf: keep the scipy-quad scalar path off the backend** | F | → feat/backends-develop | ✅ **#1268** merged 2026-08-08 (`d627badc`) | Fixes a forced-backend regression **I introduced in #1261**, found by the traced sweep — `backend-tests` has no `--jit` dimension and eager passed, so CI could not have seen it. #1261 decorated `expSurfaceSigmaProfile`'s R-evaluators with `@backend_input` (correct for dispatch); `diskdf`'s scipy quadrature calls them with **plain scalars 133,332× per `oortA()`**, and under a forced backend a plain float resolves to that backend — so each was a scalar coerced in and converted back out. **Not jit-only**: one crossing is 178.2 µs eager vs 151.1 µs traced, so the routine eager shard was ~2.8× slow while staying green. | `oortA` numpy 2.82 s (pre-#1261: 2.72) · traced jax 25.26→**4.40 s** · traced torch 42.79→**4.09 s**. Whole file: jax-jit 124/5/0 (was 2 FAILED at 300 s), torch-jit 129/0/0, numpy 129/0. **numpy bit-identical**: 120 values hash identically across with-fix / without / pre-#1261, three worktrees with provenance + a distinguishing grep each. Guard watches `coerce_coords` not the return value — a value-based guard caught 4 (jax only), the path-based one catches 8, and **torch was the worse case**. Residual 0.29 s of 1.5 s is the numpy pin: measured, not chased. |
| **drop expired numpy tolerance branch** | — | → **main** | ✅ **#1253** merged 2026-08-05 | The SpiralArms FD tolerance was gated on `1.22 < numpy < 1.27`; the upper bound silently expired, so the *tight* branch applied again with ~2× margin against an amplification-bound reference. Per Jo, **deleted** the version machinery rather than widening it: `python_requires >= 3.10` implies numpy ≥ ~1.21, so the tight branch was only reachable on numpy ≤ 1.22 — dead code | whole file 19 passed; merged as `80a10d66` |
| **actionAngle: stop masking an inner AttributeError** | — | → **main** | ✅ **#1355** merged 2026-08-20 (`45e2dcde`) | The four public entry points (`__call__`, `actionsFreqs`, `actionsFreqsAngles`, `EccZmaxRperiRap`) wrapped the **call** to their implementation in `except AttributeError: raise NotImplementedError`, so ANY `AttributeError` raised *inside* a method that exists was relabelled "method not implemented" and its traceback lost. Not hypothetical: it is what an `actionAngleAdiabaticGrid` off-grid fallback handing backend arrays to the C implementation reports, which is what made that blocker hard to see. Narrowed the `try` to the attribute lookup. | The four handlers were `# pragma: no cover`; they are now **covered** instead, by tests asserting both halves of the contract for all four entry points. **A/B-checked**: without the source change the four "not masked" tests fail and the four "missing method" tests still pass, so the tests measure exactly this fix. Full `test_actionAngle.py` green (223 passed) incl. `test_nullpotential_error`, whose `NotImplementedError` is raised deliberately elsewhere. Coverage went **up** (39377 → 39385 hits at 99.93%) |
| Track F + optimization | — | — | ⏳ next | — | **DF breadth COMPLETE** (Pdf.1–4 all merged); **streams RESUMED** (keystone wave above). AA-perf + AA-milestone COMPLETE. **NEXT: rebase feat/backends→main** (consume ExpTruncNFW #976 + SCF time-dep #1058 + migrate + ledger), then the deferred speedup roadmap (`logs/backend_speedup_audit/`; anchor = combined force/Hessian evaluators) |  

(Update this table as PRs land.)

**Ledger baseline (verified 2026-06-18, `origin/feat/backends:tests/backend_xfail.txt`):**
torch **1494** xfail / jax **254** xfail. The #969 milestone-1 prune regenerated
the ledger (commit message: torch 1684→1485, jax 335→292); subsequent tail merges
and the central-coercion fixes brought it to the current baseline. Open-PR triage
adds are branch-local until merge (#974 ≈ +16, #975 ≈ +50).

## Traced (`--jit`) state, measured 2026-08-07

The `--jit` dimension is **not in `backend-tests`' normal matrix** — it only runs
via `workflow_dispatch(jit=true)` — so until this sweep, no module had ever been
run traced outside ad-hoc probes. Whole-file runs, regen mode (so ledgered
nodeids execute), `--timeout=300`:

| file | backend | passed | failed | un-ledgered |
|---|---|---|---|---|
| `test_potential.py` | jax-jit | 1262 | 5 | **1** — #1259 traced regression |
| `test_potential.py` | torch-jit | 1282 | 1 | **1** — same regression |
| `test_actionAngle.py` | jax-jit | 179 | 3 | **1** — ledger asymmetry (fixed) |
| `test_actionAngle.py` | torch-jit | 182 | 3 | 0 |
| `test_qdf` + `test_pv2qdf` | jax-jit | 58 | 1 | 0 |
| `test_sphericaldf.py` | jax-jit | **222** | **0** | 0 |
| `test_evolveddiskdf.py` | jax-jit | **35** | **0** | 0 |
| `test_diskdf.py` | jax-jit | 122 | 2 | 2 — Oort-A **timeouts**, not wrong values |
| `test_interp_potential.py` | torch-jit | 38 | 2 | 2 — `torch.compile` cost, now deferred |

**The spherical-DF and evolveddiskdf migrations are jit-complete.** The
`test_potential` entry is a genuine regression introduced by a PR that merged
green (see `logs/vacation_2026-07/log.md`, and the task ledger's #207): a
z-discontinuity in `AnyAxisymmetricRazorThinDiskPotential`'s traced quadrature,
7.3e-5 between z=0 and z=1e-8 where eager gives 4e-12.

Five distinct problems today were visible only under `--jit`. That is the case
for making `workflow_dispatch(jit=true)` a standing pre-merge run rather than an
occasional one.

### Where traced-green actually stands

The `diskdf` Oort-A entry above is **fixed** — it was a regression I introduced
in #1261, found by this sweep within twelve hours of merging it, and traced
`oortA` now runs 4.40 s on jax and 4.09 s on torch against 25.26 s / 42.79 s
before. Whole file: 124/5/0 on jax-jit, 129/0/0 on torch-jit.

That leaves **7 entries in the whole traced ledger**, and re-reading them
(2026-08-07) gives a cleaner headline than the burndown language implies:

| what | n | why it is there |
|---|---|---|
| `AnyAxisymmetricRazorThinDisk` | 4 | **by design** — the backend path uses GL so gradients exist; GL cannot resolve the `a=R` principal value or form the `z=0` Hadamard 2nd derivative |
| units-based density | 2 | **not traceable by construction** — astropy `Quantity` arithmetic strips a tracer to numpy |
| `actionAngleStaeckelGrid` | 1 | tolerance — a *computed* circular-orbit eccentricity asked to vanish to `1e-16` (measured 9.4e-12 / 1.5e-10) |

**None is an unmigrated potential.** Traced-green is achieved as far as code
changes can take it; what remains is maintainer decisions about tolerances and
two acknowledged design limits, not migration work.

## Backend support snapshot (this env)
- jax 0.9.2 ✅ · torch 2.12 (CPU) ✅ · array-api-compat 1.14 ✅
- `array_api_compat.array_namespace`: numpy → (normalized to plain numpy), jax → `jax.numpy`, torch → `array_api_compat.torch`.

<!-- 2026-07-07: feat/backends REBASED onto main (a0fe9e12, linear, no merge commit).
Consumed ExpTruncNFW #976, time-dep SCF #1058, from_nbody #1061, SCF<->Multipole
translate #1062, sphericaldf negative-df #1032. Only SCFPotential.py needed manual
merge (main refactor+time-dep vs backend migration; numpy byte-identical, backend
preserved). New numpy-only features ledgered (jax 121, torch 802) pending Phase-2
backend migration (ExpTruncNFW + SCF-time-dep). Reviewed via PR #1080 (closed). -->
| #1269 | slow-skip: un-defer 5 measured-stale jax diskdf sampling entries (109 -> 104) | MERGED `ffde6f86` |
| #1270 | slow-skip: un-defer 6 measured-stale jax entries (104 -> 98) | MERGED `1726834e` |
| #1271 | planar adapters: forward to the undecorated evaluators (989 -> 5 crossings) | MERGED `f1bb9f83` |
| #1272 | vertical adapters: forward to the undecorated evaluators (1708 -> 244 crossings) | MERGED `7418aed5` |
| #1273 | actionAngleVertical backend path -> undecorated linear evaluators (244 -> 0); slow-skip 98 -> 92 | MERGED `9f3c3166` |
| #1274 | interpRZPotential: keep grids + off-grid fallbacks in internal units (pre-existing numpy bug) | GREEN (151/0) — targets `main`, Jo's to merge |
| #1275 | SpiralArms tolerance cleanup + interpSpherical range check independent of FDM + NFW alpha/beta on conc/rmax paths | MERGED `44b06e31` |
| #1278 | AnyAxisym: geometric panel grading so the a=R peak resolves at any \|z\| (K=24 was 95% low at 1e-12) | MERGED `f963eb42` |
| #1279 | `--jit` gets its own slow-skip list, populated by measurement; backend-aware actionAngle bars (#171) | MERGED `d2ad93c9` |
| #1363 | ledger: split the streamTrack whole-file defer — 43 tests back into jax | MERGED `19f31a54` |
| #1364 | backend: `as_backend_constant` must not name the default device (1.5x eager win on wrapper forces) | MERGED `a4193e84` |
| #1367 | `ExpTruncNFW.from_nfw`: differentiable mass → rc solve (implicit function theorem; 8e-12 vs FD) | MERGED `08db5d90` |
| #1368 | `EinastoPotential`: differentiable d_n solve on jax/torch (regularized residual, same root) | MERGED `45221587` |
| #1369 | potential: build on the forced backend, not scipy, at construction (Einasto + ExpTruncNFW) | MERGED `77a74fdb` |
| #1370 | Ferrers/TwoPowerTriaxial: gamma normalizations on the backend (coerce at point of use) | MERGED `542eae05` |
| #1371 | power-law spherical DFs: fE normalizations on the backend; `from scipy import special` gone | MERGED `856ceccc` |
| #1372 | `parallel_map`: run serially under jax instead of forking into a deadlock; un-defers `test_streamdf.py` (~76 tests) | MERGED `a063e674` |
| #1374 | special: backend-op `gammainc`/`gammaincc` differentiable in the ORDER (native forward + custom backward; 485x → 2.08x) | MERGED `068de984` |
| #1380 | ledger: un-defer 3 measured-stale jax entries; measure the rest of Group 1 (unmeasured 10 → 0) | MERGED `eabc3a5a` |
| #1381 | special: guard the x=0 endpoint of d/dx gammainc on torch (#1204 review, 1/5) | MERGED `206ccd78` |
| #1382 | backend: finite_part_quad batching + symmetric_quad eager-torch limit + AnyAxisym torch construction (#1204 review, 2/5) | MERGED `06267d1d` |
| #1383 | coords: finish the inverse galcen transforms + fix the f64-constant/f32-data promotion (#1204 review, 3/4/6) | MERGED `4e2cc322` |
| #1384 | actionAngleVerticalInverse: canonical evaluation mode, on by default (main) | CLOSED (unmerged) |
| #1385 | actionAngleVerticalInverse: canonical evaluation for all point-transformation modes; old interpolated path removed | CLOSED (unmerged) |
| #1387 | actionAngleSphericalCanonical: the spherical inverse, built the Stäckel-canonical way (S1) (main) | CLOSED (unmerged) |
| #1388 | backend: map_coordinates must dispatch on the GRID too (#1204 review, batch 2 item 2) | MERGED `107a8060` |
| #1389 | actionAngleSphericalInverse, the canonical construction: the momentum-matched alignment PT (S2) | CLOSED (unmerged) |
| #1391 | backend: traceable constructors + differentiable hyp2f1/hyp1f1 parameters (#1204 review, batch 2 items 3+4) | MERGED `cfe23abe` |
| #1392 | constantbetaHernquistdf: route the gammas so d f_E/d beta works | MERGED `8980402a` |
| #1393 | tests: un-gate the Einasto d_n gradient to torch + pin float64 in test_backend_spherical | MERGED `366ffc04` |
| #1394 | surfdens: keep both routes on the backend under eager torch autograd | MERGED `bc69771e` |
| #1395 | actionAngleStaeckelInverse: the canonical momentum-matched construction (T1) (stacked on `aainv-staeckel-canonical-dev`) | OPEN — green (107/107) |
| #1396 | actionAngleStaeckelInverse: the canonical family (T2) (stacked on `aainv-staeckel-canonical-core`) | OPEN — green (107/107) |
| #1398 | actionAngleAdiabatic: fix multi-point _justjr raising TypeError (main) | MERGED `3df11b96` |
| #1399 | actionAngleAdiabaticGrid: exact off-grid actions on jax/torch | MERGED `5c3c8671` |
| #1400 | integrate_dxdv: transform the base state in C, restoring bit-identity with plain integrate (main) | MERGED `87547bf0` |
| #1401 | Review round 3: traced beta under jit, mixed quad limits, and the small-B hyp2f1 floor | MERGED `00a7153c` |
| #1402 | hyp2f1: one tanh-sinh rule for every B, and drop the c-B >= 1 constraint | MERGED `8c320b39` |
| #1403 | hyp2f1: the traced Euler-label selector must use the same rule as the concrete one | MERGED `26388056` |
| #1404 | actionAngleStaeckel: chi-anomaly quadrature at the turning points in C (main) | MERGED `f49fa016` |
| #1405 | actionAngleIsochroneInverse: solve Kepler's equation for all angles at once (main) | MERGED `72f96783` |
| #1406 | hyp2f1: bring the comments back in line with the code | MERGED `b1a8b03f` |
| #1407 | surfdens: eager backends take the batched vertical quadrature (ledger -6) | MERGED `852d69d6` |
| #1408 | mass(): ask the backend's array-safety question; re-ledger a wrong de-ledgering | MERGED `b331faa0` |
| #1409 | CI: fetch GSL from the nearest GNU mirror rather than ftp.gnu.org (main) | MERGED `96f40406` |
| #1410 | test_orbit: two tests bypassed _backend_integrators (ledger -4) | MERGED `ef360db0` |
| #1411 | SCF coefficients: lift the numpy pin so the expansion is differentiable w.r.t. the density | MERGED `e9afe588` |
| #1412 | actionAngleVerticalInverse: the momentum-matched anomaly map (construction) | CLOSED (unmerged) |
| #1413 | actionAngleStaeckelInverse: speed — vectorize the canonical family evaluation (stacked on `aainv-staeckel-canonical-family`) | OPEN — green (107/107) |
| #1414 | actionAngleVerticalInverse: the momentum-matched family | CLOSED (unmerged) |
| #1415 | actionAngleVerticalInverse: the momentum-matched canonical map (main) | MERGED `70032808` |
| #1416 | SCF axi coefficients: batched quadrature on backends (bit-exact, ~1.9x) | MERGED `d8098c7e` |
| #1417 | tests: roundoff tolerance for batched-vs-sequential mass quadrature | MERGED `5410ac6e` |
| #1418 | tests: cut the time-dependent SCF coefficient tests from ~76s to ~11s (main) | MERGED `a41eb8bd` |
| #1419 | actionAngleStaeckelInverse: canonicity for a varying focal length (stacked on `aainv-staeckel-vectorize`) | OPEN — green (107/107) |
| #1420 | actionAngleStaeckelInverse: build the family adaptively, delta(E, L_z) (stacked on `aainv-staeckel-adaptive`) | OPEN — green (107/107) |
| #1421 | actionAngleStaeckelInverse: delta='fit', and integrals in the local chart (stacked on `aainv-staeckel-adaptive-build`) | OPEN — green (107/107) |
| #1422 | Speed up ChandrasekharDynamicalFrictionForce construction ~100x for array-capable potentials (main) | MERGED `02d589bf` |
| #1423 | actionAngleStaeckelInverse: target= — localize the grid on an orbit or stream (stacked on `aainv-staeckel-adaptive-fit`) | OPEN — green (107/107) |
| #1424 | actionAngleStaeckel C: frequencies and angles near turning points (main) | MERGED `5925f0b1` |
| #1425 | Cherry-pick: ChandrasekharDynamicalFrictionForce ctor speedup (drop in the next rebase) | CLOSED (unmerged) |
| #1426 | actionAngleStaeckelInverse: u0-only adaptation — fixed delta, u0(E, L_z) (stacked on `aainv-staeckel-targetbox`) | OPEN — green (107/107) |
| #1427 | actionAngleStaeckelInverse: a single focal length, like the forward code (stacked on `aainv-staeckel-u0only`) | OPEN — green (107/107) |
| #1428 | backend: keep the jax isinstance out of dynamo's way (unblocks #1204 on torch 2.14) | MERGED `b0698589` |
| #1429 | actionAngleStaeckelInverse: array-J canonical evaluation (stacked on `aainv-staeckel-singledelta`) | OPEN — green (107/107) |
| #1430 | actionAngleTorusStaeckel: torus mapping on the canonical Staeckel-inverse auxiliary (stacked on `aainv-staeckel-arrayJ`) | OPEN — green (107/107) |
| #1431 | actionAngleStaeckel C: NaN and near-axis accuracy in the chi quadratures (main) | MERGED `c8b9e982` |
| #1432 | Backend shard split (1/n): trace the torch-compile sweep under eager | MERGED `3aaf0e5c` |
| #1433 | Backend shard split (2/n): duration-balanced pytest-split | MERGED `d35ed301` |
| #1434 | actionAngleStaeckelInverse: circular-edge anchor + edge-conscious frequency chains (stacked on `aainv-torusmapper`) | OPEN — green (107/107) |
| #1436 | Backend shard: eager torch.compile for jit tests (real ~10x on the heaviest) | MERGED `1bbddb38` |
| #1437 | qdf: float the precomputed rg grid bound so forced-backend _rg survives | MERGED `f053bee5` |
| #1438 | backend ledger: xfail the forced jax/torch reds the main rebase introduced | MERGED `c11eeba5` |
| #1439 | jeans: don't let scipy.quad's default epsabs truncate sigma_r's small outer tail (main) | MERGED `04f30362` |
| #1442 | backend burndown 1: numpy sigmar byte-identity + drop 4 stale xfail entries | MERGED `a637696a` |
| #1443 | qdf sampleV: backend-native inverse-CDF sampler (drop the numpy-forced rejection) | MERGED `7752807b` |
| #1444 | qdf sampleV_interpolate: inverse-CDF on the backends, rejection on numpy | MERGED `c2611a49` |
| #1446 | backend: vectorise the Staeckel c-vs-Python freqs/angles parity grid (slow-skip -1) | MERGED `629479d1` |
| #1447 | test_actionAngle: platform-proof the Staeckel chi-quadrature bars (3e-11 -> 1e-10) | MERGED `10b7ec96` |
| #1448 | sphericaldf: build the p(v\|r) inverse-CDF table on the backend (d(sample_v)/d(DF params)) | MERGED `831dd644` |
| #1449 | sphericaldf: cover the p(v\|r) backend-build branches #1448 left untested | MERGED `4f54b885` |
| #1450 | backend: stop re-doing two pieces of eager work on every force evaluation | MERGED `35a536c4` |
| #1451 | slow-skip: drop the traced Staeckel parity grid (ledger -1), re-measure three stale jit notes | MERGED `d1d59adf` |
| #1452 | Make the traced suite green: fix the vcirc probe, record the four traced timeouts | MERGED `a7fdc4fc` |
| #1453 | SCF: batch the general (non-axi) coefficient quadrature (traced test_density1_Order 300s+ -> 13s, ledger -1) | MERGED `396e13c2` |
| #1454 | SCF: contract the time-dependent coefficient quadrature over its nodes (jax 63s -> 4.3s, ledger -3) | MERGED `33f1f05b` |
| #1455 | Route the orbit integrator loops through _backend_integrators (SOS_3D 573s -> 12s, ledger -3) | MERGED `87362405` |
| #1458 | Do not fork a worker per task when numcores=1 (Orbit.lyapunov 94x faster) (main) | MERGED `a95dc88a` |
| #1459 | slow-skip: un-ledger three stale jax entries (ledger -3) | MERGED `be64f868` |
| #1460 | test_backend_dynamfric: validate the in-backend integrators against analytic friction | MERGED `e575bfcf` |
| #1461 | test_dynamfric_c: in-backend second arm under a backend, and chunk the loop | MERGED `f059c098` |
| #1463 | dynamfric + FDM: in-backend non-C arm, and un-ledger what it earns (ledger -9) | MERGED `cd7fec21` |
| #1464 | bessel_k: evaluate the ascending series in one pass (torch 2.3x on Bessel-K potentials) | MERGED `ca522e63` |
| #1465 | noninertial + Chandrasekhar orbits: in-backend non-C arm, and un-ledger what it earns (ledger -6) | MERGED `15a5f7d4` |
| #1467 | test_orbit: one test case per potential for the four all-potentials walks (main) | MERGED `2296c50b` |
| #1468 | OblateStaeckelWrapperPotential: optional tabulated C evaluation (ntab=), ~10-16x faster C integration (main) | MERGED `1bc09cac` |
| #1469 | dynamfric: one test case per potential, replacing the index chunks | MERGED `2b40ae8a` |
| #1470 | Avoid duplicate builds/tests on PRs (main) | MERGED `3e744b61` |
| #1471 | CI: run the all-backend suite once per commit, not twice | MERGED `e2ce72e5` |
| #1472 | Delete copilot setup (main) | MERGED `9b9bcd96` |
| #1473 | C bridge: keep a potential's tables whole in pot_args (streamTrack 7x under jax) | MERGED `b94cd0a8` |
| #1474 | test_orbit: per-potential walks on feat/backends, and ledger the potentials (not the walks) | CLOSED (unmerged) |
| #1475 | actionAngle_c: parse one potentialArg copy per OpenMP thread (main) | MERGED `76ce1807` |
| #1476 | streamTrack sky Jacobians batched + exp1 vectorized (ledger 18 -> 14) | MERGED `a381fcf1` |
| #1477 | exp1: one quadrature instead of an 80-step recurrence (torch 7.3x on ExpTruncNFW, and 8x more accurate) | MERGED `f9745db3` |
| #1478 | torch-jit walks: retire both whole-walk rows (ledger 11 -> 9) | MERGED `3bb4e40e` |
| #1480 | traced rows: re-measure the three that never were, retire all three (ledger 9 -> 6) | MERGED `2515e281` |
| #1481 | xfail ledger: the eager rows are gone (31 -> 6, and every survivor is jit-only) | MERGED `cbebad69` |
| #1482 | the numpy/backend boundary in orbit integration: warn on the demotion, and stop the friction force paying for it | MERGED `c5ce9565` |
| #1484 | ChandrasekharDynamicalFrictionForce: the force cache has never hit (2.1x on the friction path) (main) | MERGED `d243f989` |
| #1485 | OblateStaeckelWrapperPotential: ntab= works on the backend, differentiably | MERGED `43f10799` |
| #1486 | Orbit: a backend IC gets a backend trajectory back, whatever the integrator | MERGED `c81c1992` |
| #1489 | Orbit: the backend integration routes continue an integration instead of replacing it | MERGED `5aaf31ef` |
| #1490 | test_eccentricity: a traced-only bar for the one potential that integrates on GL | MERGED `ecce6294` |
| #1491 | eval_rect_ppoly: one gather for the cell, not sixteen | MERGED `7fc8e031` |
| #1492 | slow-skip: retire one FDM row, and make three stale notes true | MERGED `e2857725` |
| #1493 | actionAngleStaeckel: parallelise actionsFreqsAnglesJac's Jacobian loop | MERGED `bf84fca0` |
| #1494 | sici: two cumulative products instead of two 40-term Python loops | MERGED `84d9b1b6` |
| #1495 | map_coordinates: one gather for the whole cubic tap block, not 4**D | MERGED `51b65fa4` |
| #1496 | AnyAxisymmetricRazorThinDisk: accept backend array input; xfail 5 -> 3 | MERGED `25087f0f` |
| #1497 | constantbetadf: make eta sampling differentiable in beta | MERGED `a40b911c` |
| #1498 | orbit: keep a backend Orbit on its backend through the no-time accessors and o() | MERGED `0d070d2a` |
| #1499 | streamdf: run the progenitor/offset setup on the backend (Phase C.3) | MERGED `8c99e203` |
| #1500 | streamTrack: keep the accessor query axis on the backend so it can be differentiated | MERGED `12f37425` |
| #1501 | EinastoPotential: use regularized P so n=6 force stays finite (main) | MERGED `7a67da19` |
| #1503 | orbit: a backend potential parameter must not take the C-STM | MERGED `0c05087f` |
| #1504 | backend: integrate, spline and read an orbit back on a TRACED time grid | MERGED `ead2ccea` |
| #1505 | streamdf: determine the track on a theta-dependent (traced) time grid | MERGED `62c53ee8` |
| #1506 | tests: units-consistent bar for test_streamgapdf_sample's sky-frame comparison | MERGED `0d1c247a` |
| #1507 | streamgapdf: build the impact coordinate transform on the backend | MERGED `c548f0ad` |
| #1508 | potentials: branch on a stored parameter only when it is concrete | MERGED `3351b666` |
| #1509 | linalg: make psd_project's gradient exact where nothing is clipped | MERGED `a432d6e4` |
| #1510 | actionAngleVerticalInverse: select tori by their actual action with a polynomial point transformation (main) | MERGED `e0fb094e` |
| #1511 | sphericaldf: invert r(Phi) by root-find when the potential is traced | MERGED `54aa7e60` |
| #1512 | streamgapdf: put the kick interpolation on the backend, completing d/d(potential) | MERGED `0e78aa15` |
| #1514 | DFs: make the quasiisothermal and King constructors differentiable | MERGED `4d608f43` |
| #1515 | actionAngleVerticalInverse: speed up the momentum-matched evaluation (Newton on the closed-form derivative) and construction | MERGED `058e3d2a` |
| #1516 | actionAngleVerticalInverse: compute the polynomial point transformation's action offset in closed form and apply it (main) | MERGED `eb5aa8c7` |
| #1517 | actionAngleVerticalInverse: the map's action derivatives on each torus, stored as Hermite constraints of the family | MERGED `2db0c6f7` |
| #1519 | linearPotential: x2deriv and evaluatelinearx2derivs, the second derivative of 1D potentials (main) | MERGED `b19f8743` |
| #1521 | streamdf/streamgapdf: fix the nan d/d(potential) through the kick, and test the perturber chain end to end | MERGED `211dbe83` |
| #1522 | actionAngleSphericalInverse: the momentum-matched map alone, as in the paper and the 1D code | CLOSED (unmerged) |
| #1523 | actionAngleSphericalInverse: the family as Hermite interpolants with exact first partials at every node | CLOSED (unmerged) |
| #1524 | actionAngleSphericalInverse: the evaluation as one pass in the anomaly | CLOSED (unmerged) |
| #1525 | Orbit: let the analytic ecc/zmax/rperi/rap differentiate w.r.t. potential parameters | MERGED `85d6c4e2` |
| #1526 | actionAngleSphericalInverse: the circular orbits, as the edge of the family and as tori | CLOSED (unmerged) |
| #1527 | actionAngleSpherical: circular and near-circular orbits (main) | MERGED `5282778c` |
| #1528 | spherical DFs: differentiate w.r.t. potential parameters (construction + velocity moments) | MERGED `562f9fda` |
| #1529 | DFs: differentiate w.r.t. their own parameters (diskdf profile, qdf hr) | MERGED `4910fb6b` |
| #1530 | dynamical friction: coerce the caller-supplied velocity triple onto the backend | MERGED `15156dd5` |
| #1531 | Osipkov-Merritt dM/dE: integrate fQ's support and split at the ceiling corners | MERGED `f2844428` |
| #1532 | Orbit accessors: follow the gradient w.r.t. potential parameters | MERGED `8f74d0d2` |
| #1533 | Spline1D: vector-valued y, and default to the boundary condition numpy uses | MERGED `24bb56af` |
| #1534 | Spline1D consumers: streamTrack's dual representation and SCF's eval_ppoly clone | MERGED `1a1e2211` |
| #1535 | interpRZPotential and Orbits: one spline object per grid, not a spline plus a derived cache | MERGED `d3a92e44` |
| #1536 | CI: run the backend suite TRACED on every PR, as a matrix axis | MERGED `af709169` |
| #1537 | qdf: differentiate the moments w.r.t. potential parameters, and fix _rg's no-table array branch | MERGED `3b0b3ce9` |
| #1538 | actionAngle grids: delegate to the exact method when the potential carries a gradient | MERGED `bba922cb` |
| #1539 | evolveddiskdf: fix the vanishing-normalisation guard on torch (odd moments) | MERGED `1ea00491` |
| #1540 | evolveddiskdf: honour a gradient carried by the POTENTIAL, not just by R | MERGED `0172ce5c` |
| #1541 | sphericaldf: make sample() work when the potential carries a gradient | MERGED `64f3bb7e` |
| #1542 | evolveddiskdf: migrate the hierarchical velocity grid to the backend | MERGED `62ccd636` |
| #1543 | constantbetadf: differentiate fE w.r.t. the potential parameter | MERGED `1db57056` |
| #1544 | backend: namespace_of() — resolve a value's own namespace, not the forced one | MERGED `18cd0ace` |
| #1545 | actionAngleSphericalInverse: the inverse action-angle transformation for spherical potentials through a momentum-matched canonical map (main) | OPEN — green (105 pass, 3 skipped) |
| #1546 | constantbetadf: differentiable w.r.t. the potential under jax too (eager grad) | MERGED `87f4ec20` |
| #1547 | sphericaldf: differentiable backend-key sampling w.r.t. the potential | MERGED `c68bcbeb` |
| #1549 | dynamical friction: differentiable w.r.t. the host potential | MERGED `e24a1fc5` |
| #1550 | psd_project: exact gradient on clipped slices; diffrax accepts a potential list | MERGED `dce7195d` |
| #1551 | kingdf: differentiable w.r.t. W0 (forward sensitivities of the King ODE) | MERGED `7caff969` |
| #1552 | kingdf: solve the King ODE at rtol=1e-10 (scipy's default left ~1e-4) | MERGED `836e3f6b` |
| #1553 | actionAngleSpherical: decide boundedness at infinity, not at a fixed radius (main) | MERGED `a0d7dff6` |
| #1554 | potentials: finite parameter gradients at r = 0 and r = inf | MERGED `cfc9a77d` |
| #1555 | special: accurate backend hyp2f1 at large negative z | MERGED `516ac9fb` |
| #1556 | osipkovmerrittdf: differentiable velocity sampling w.r.t. the potential | MERGED `45dcc2a3` |
| #1557 | constantbetadf: ~4x faster construction; spline solves in O(n log n) | MERGED `e817b56d` |
| #1558 | constantbetadf: construct, evaluate and sample under jax.jit | MERGED `10afd54c` |
| #1559 | eddingtondf, kingdf (M, rt), osipkovmerrittdf: under jax.jit | MERGED `482ef217` |
| #1560 | actionAngleStaeckel: the focal length from a potential that supplies one (main) | MERGED `1966a1c0` |
| #1561 | PerfectEllipsoidPotential: the focal length of an axisymmetric oblate perfect ellipsoid | MERGED `7cfaa5c7` |
| #1562 | actionAngleStaeckelInverse: the momentum-matched canonical map for Staeckel potentials, explicit tori (stacked on `aainv-spherical-inverse`) | OPEN — green (107/107) |
| #1563 | actionAngleStaeckelInverse: the interpolated family (stacked on `aainv-staeckel-inverse`) | OPEN — green (107/107) |
| #1566 | kingdf: d/dW0 under jax.jit (diffrax) and torch.compile | MERGED `7236d252` |
| #1567 | orbit: C integrators under torch.compile (and no silent detach) | MERGED `07d68ed7` |
| #1568 | Fix spherical DF sampling velocity bias (main) | MERGED `fade6dae` |
| #1569 | orbit: method='torchode', a torch.compile-able in-backend integrator | MERGED `e5e73318` |
| #1571 | backend: linalg solve/inv/cholesky/eigvals that torch 2.14 can compile | CLOSED (unmerged) |
| #1572 | kingdf: correct second (and higher) W0-derivatives on jax and torch | MERGED `9606ee4a` |
| #1573 | tests: whole workflows under jax.jit / torch.compile, value and gradient | MERGED `ee3eff6c` |
| #1574 | qdf under torch.compile; Orbit.E at a traced time under jax.jit | MERGED `7b408d35` |
| #1575 | streamTrack: spray -> track under torch.compile | MERGED `a71a2be5` |
| #1578 | potentials, eddingtondf: backend side of #1585 and #1586 (small-r accuracy, edge limits, f(E) near Emin) | MERGED `3d8d14a5` |
| #1579 | eddingtondf: backend side of #1586 (f(E) accurate near Emin) | MERGED `c603f266` |
| #1582 | Add Python 3.15 support (main) | OPEN — green (113 pass, 3 skipped) |
| #1583 | Add Python 3.15 wheels (main) | MERGED `649b7506` |
| #1584 | backend: torchode default; stream workflows fully compile under inductor | MERGED `60310a17` |
| #1585 | potentials: machine-precision small r (NFW, Burkert, TwoPower), finite edge limits, Staeckel midplane vmin (main) | MERGED `a6f41359` |
| #1586 | eddingtondf: f(E) accurate near Emin (NFW was NaN/negative there) | MERGED `d0a8e24f` |
| #1588 | Programatically define galpy animation <style> to avoid duplication and HTML error (main) | MERGED `85788ef7` |
| #1589 | all-clear: ignore superseded check attempts (main) | MERGED `84166989` |
| #1590 | Build Pyodide wheels for Python 3.14 (main) | MERGED `ba066013` |
| #1591 | DO NOT MERGE: CI for feat/backends rebased onto main (+ rebuilt #1578/#1579) (main) | CLOSED (unmerged) |
| #1592 | DO NOT MERGE: CI for feat/backends rebased onto main (+ rebuilt #1578/#1579) | CLOSED (unmerged) |
| #1593 | torch.compile: the streamTrack fit and the qdf constructor compile (replace #1574/#1575 opt-outs) | MERGED `3432e0d5` |
| #1594 | actionAngleSpherical: the backend solves main's relative radial problem (retire the rebase's 8 xfails) | MERGED `03d417e9` |
| #1596 | TwoPowerSphericalPotential: forces, second derivatives and mass from the incomplete beta (accurate near beta = 3, finite at large r) (main) | MERGED `7c675eb6` |
| #1597 | TwoPowerSphericalPotential: forces, second derivatives and mass from the incomplete beta (numpy, C, backends) | MERGED `28d05c02` |
| #1598 | spherical DFs: osipkovmerrittdf.sample under jax.jit; exact traced r(Phi) near Emin; constantbetadf backend fE without the calibrated limit | MERGED `4599bdcb` |
| #1599 | Orbit.rperi/rap/zmax/e(analytic=True): backend-native and differentiable w.r.t. the phase-space point | MERGED `74cd6698` |
| #1600 | kingdf: W0 compiles under torch.compile (no opaque scipy solve) | MERGED `e20282d7` |
| #1602 | streamdf: vmap the differentiable track's chunk loop (shared constant steps keep d/dparam exact) | MERGED `8d5769a4` |
| #1603 | GPU (CUDA) validation: --device cuda test option and CUDA fixes | MERGED `756fe414` |
| #1604 | GPU numpy islands (1/5): coords + conversion; to_host for sequences, on_host | MERGED `74d69e87` |
| #1605 | GPU numpy islands (2/5): potential | MERGED `65f63abd` |
| #1606 | GPU numpy islands (3/5): actionAngle | MERGED `a316de7e` |
| #1607 | GPU numpy islands (4/5): Orbit | MERGED `164d4e51` |
| #1608 | GPU numpy islands (5/5): df, ars, scipy spline | MERGED `1a4c720f` |
| #1610 | Orbit: backend ICs in sky coordinates convert differentiably (radec/lb) | MERGED `0724b62b` |
| #1611 | Orbit: setting up from sky coordinates with uvw=True no longer modifies the input (main) | MERGED `cddcc8ff` |
| #1612 | SCFPotential.from_density: differentiable w.r.t. the density's parameters (static and time-dependent) | MERGED `986ade27` |
| #1613 | float64 by default: torch's float32 default no longer crashes galpy or degrades its internals | MERGED `2abae485` |
| #1614 | test_backend_compile_workflows: cap inductor SIMD at 256 bits for kingdf_W0 (GCC ICE on AVX-512 runners) | MERGED `99f52149` |
| #1615 | gammainc/gammaincc on jax: vectorized series/CF (~100x faster on CPU), scipy-accurate everywhere | MERGED `b88c1fe5` |
| #1616 | Run the Orbit.from_name (SIMBAD) tests under forced jax/torch | MERGED `abc8f48c` |
| #1617 | quasiisothermaldf: constructor compiles under torch.compile without an opt-out | MERGED `cee82aae` |
| #1618 | MultipoleExpansionPotential.from_density: differentiable w.r.t. the density's parameters (static and time-dependent) | MERGED `c278dbf6` |
| #1619 | Dynamical friction (backend, eager): one force factor per EOM step; FDM computes only its regime | MERGED `e8114d53` |
| #1620 | conftest: report the runner's CPU (model, AVX2/AVX-512/AMX, inductor vec ISA) in the pytest header | MERGED `1cffb1ca` |
| #1621 | qdf sampler: bucket varying lengths + skip pure_callback for concrete inputs; un-skip test_sampleV_interpolate under jax (2837 s -> 118 s) | MERGED `21965057` |
| #1623 | test_orbit_interface_staeckel_defaultdelta: compare the interfaces at the Orbit's own delta (1-ulp jit/eager delta difference) | MERGED `72029ea2` |
| #1624 | Remove two numpy islands (streamgapdf orbit-integration kick, constantbetadf torch sampling pin); classify the rest | MERGED `2eea28de` |
| #1625 | jit mode: key static objects by type + attributes (stale traces after a wrapped potential changes; per-call retraces) | MERGED `27e3c649` |
| #1626 | DO NOT MERGE: CI for feat/backends rebased onto main (2026-10-04) | CLOSED (unmerged) |
| #1627 | actionAngleVerticalInverse (1/3): the momentum-matched map evaluates natively under jax/torch | MERGED `7c06aead` |
| #1628 | actionAngleVerticalInverse (2/3): build the momentum-matched family natively, differentiable in the potential's parameters | MERGED `884defc3` |
| #1630 | actionAngleVerticalInverse (3/3): the legacy modes evaluate natively under jax/torch | MERGED `f65c4ae2` |
| #1631 | actionAngleVerticalInverse: runs on CUDA (energy grid to host; tests CUDA-clean) | MERGED `9f6c9d6f` |
| #1633 | Dynamical friction: sigma_r on a log-r grid (was 20-60% off at small r); Python evaluates C's natural spline (main) | MERGED `3ae536e1` |
| #1634 | Dynamical friction: sigma_r on a log-r grid; every path evaluates C's natural spline (backend side of #1633) | MERGED `793ceef7` |
| #1635 | backend brentq: bracket_width= keeps the traced halving schedule as short as eager (VerticalInverse jit 2.5-3.7x faster) | MERGED `1c352a0e` |
| #1636 | actionAngleVerticalInverse: tables follow the potential's device (CUDA parameters with torch's default device on the CPU) | MERGED `3504efdf` |
| #1637 | streamdf: a torch progenitor constructs (eager torch grad tensors stay on torch; a silently dropped gradient fixed) | MERGED `70165d10` |
| #1638 | Pyodide build: fetch GSL from independent mirrors, SHA-256 verified (GNU hosts down) (main) | MERGED `97096de7` |
| #1639 | gammainc (torch): a float order follows the argument's device (MWPotential2014 on CUDA failed) | MERGED `1bfce100` |
| #1640 | streamdf: a jitted construction matches eager (track solves on DirectAdjoint); jax.grad of the constructor runs | MERGED `253ad012` |
| #1641 | backend brentq: guess= for safeguarded Newton; VerticalInverse angle inversion ~2x faster under jit (4.8x incl. #1635) | OPEN — green (150/150) |
| #1642 | actionAngleStaeckel (c=False): chi-anomaly mesh matches C (3.8x faster; fixes the Sept. 2 regression) | OPEN — CI running (138 pending; codecov/project+patch red on partial uploads) |

## Upstream issues filed from this work (numpy-path defects, not backend)

Found while validating backend ports against independent references; both
reproduce identically on `main`, so they are pre-existing galpy bugs rather than
migration damage. Filed rather than fixed, per Jo (2026-08-08: "doesn't sound
urgent, maybe open as an issue").

| issue | defect | worst measured |
|---|---|---|
| [#1276](https://github.com/jobovy/galpy/issues/1276) | `RazorThinExponentialDiskPotential` forces lose ~3 digits in a **band** `1e-6 <~ \|z\| <~ 1e-2`. The `\|z\|<1e-6` closed form is accurate; the quadrature branch above it degrades as z->0, so the switch hands over in the wrong direction. | 5.2e-03 rel at R=0.5, z=1e-6 |
| [#1277](https://github.com/jobovy/galpy/issues/1277) | `AnyAxisymmetricRazorThinDiskPotential.R2deriv` at z=0 carries ~1e-5 relative error at small R, while reaching ~1e-9 at large R. | 9.9e-06 rel at R=0.2 |

Both were checked against an mpmath Hankel reference at 30 digits, and the
reference itself validated per quantity before use (against galpy's own closed
form at z=0 for `Rforce`, 3.5e-15; against FD-of-Hankel-`Rforce` for `R2deriv`,
11 digits). An unvalidated arbiter would have been the third plausible-but-wrong
number in the pair.

## Merged PRs #1381-#1573 (rebuilt from the GitHub API, 2026-09-25)

| PR | Base | Merged | Title |
|----|------|--------|-------|
| [#1381](https://github.com/jobovy/galpy/pull/1381) | feat/backends-develop | 2026-08-25 | special: guard the x=0 endpoint of d/dx gammainc on torch (#1204 review, 1/5) |
| [#1382](https://github.com/jobovy/galpy/pull/1382) | feat/backends-develop | 2026-08-25 | backend: finite_part_quad batching + symmetric_quad eager-torch limit + AnyAxisym torch construction (#1204 review, 2/5) |
| [#1383](https://github.com/jobovy/galpy/pull/1383) | feat/backends-develop | 2026-08-25 | coords: finish the inverse galcen transforms + fix the f64-constant/f32-data promotion (#1204 review, 3/4/6) |
| [#1388](https://github.com/jobovy/galpy/pull/1388) | feat/backends-develop | 2026-08-26 | backend: map_coordinates must dispatch on the GRID too (#1204 review, batch 2 item 2) |
| [#1391](https://github.com/jobovy/galpy/pull/1391) | feat/backends-develop | 2026-08-26 | backend: traceable constructors + differentiable hyp2f1/hyp1f1 parameters (#1204 review, batch 2 items 3+4) |
| [#1392](https://github.com/jobovy/galpy/pull/1392) | feat/backends-develop | 2026-08-26 | constantbetaHernquistdf: route the gammas so d f_E/d beta works |
| [#1393](https://github.com/jobovy/galpy/pull/1393) | feat/backends-develop | 2026-08-26 | tests: un-gate the Einasto d_n gradient to torch + pin float64 in test_backend_spherical |
| [#1394](https://github.com/jobovy/galpy/pull/1394) | feat/backends-develop | 2026-08-26 | surfdens: keep both routes on the backend under eager torch autograd |
| [#1398](https://github.com/jobovy/galpy/pull/1398) | main | 2026-08-27 | actionAngleAdiabatic: fix multi-point _justjr raising TypeError |
| [#1399](https://github.com/jobovy/galpy/pull/1399) | feat/backends-develop | 2026-08-27 | actionAngleAdiabaticGrid: exact off-grid actions on jax/torch |
| [#1400](https://github.com/jobovy/galpy/pull/1400) | main | 2026-08-27 | integrate_dxdv: transform the base state in C, restoring bit-identity with plain integrate |
| [#1401](https://github.com/jobovy/galpy/pull/1401) | feat/backends-develop | 2026-08-27 | Review round 3: traced beta under jit, mixed quad limits, and the small-B hyp2f1 floor |
| [#1402](https://github.com/jobovy/galpy/pull/1402) | feat/backends-develop | 2026-08-27 | hyp2f1: one tanh-sinh rule for every B, and drop the c-B >= 1 constraint |
| [#1403](https://github.com/jobovy/galpy/pull/1403) | feat/backends-develop | 2026-08-27 | hyp2f1: the traced Euler-label selector must use the same rule as the concrete one |
| [#1404](https://github.com/jobovy/galpy/pull/1404) | main | 2026-08-29 | actionAngleStaeckel: chi-anomaly quadrature at the turning points in C |
| [#1405](https://github.com/jobovy/galpy/pull/1405) | main | 2026-08-29 | actionAngleIsochroneInverse: solve Kepler's equation for all angles at once |
| [#1406](https://github.com/jobovy/galpy/pull/1406) | feat/backends-develop | 2026-08-28 | hyp2f1: bring the comments back in line with the code |
| [#1407](https://github.com/jobovy/galpy/pull/1407) | feat/backends-develop | 2026-08-28 | surfdens: eager backends take the batched vertical quadrature (ledger -6) |
| [#1408](https://github.com/jobovy/galpy/pull/1408) | feat/backends-develop | 2026-08-28 | mass(): ask the backend's array-safety question; re-ledger a wrong de-ledgering |
| [#1409](https://github.com/jobovy/galpy/pull/1409) | main | 2026-08-28 | CI: fetch GSL from the nearest GNU mirror rather than ftp.gnu.org |
| [#1410](https://github.com/jobovy/galpy/pull/1410) | feat/backends-develop | 2026-08-29 | test_orbit: two tests bypassed _backend_integrators (ledger -4) |
| [#1411](https://github.com/jobovy/galpy/pull/1411) | feat/backends-develop | 2026-08-30 | SCF coefficients: lift the numpy pin so the expansion is differentiable w.r.t. the density |
| [#1416](https://github.com/jobovy/galpy/pull/1416) | feat/backends-develop | 2026-08-31 | SCF axi coefficients: batched quadrature on backends (bit-exact, ~1.9x) |
| [#1417](https://github.com/jobovy/galpy/pull/1417) | feat/backends-develop | 2026-08-31 | tests: roundoff tolerance for batched-vs-sequential mass quadrature |
| [#1418](https://github.com/jobovy/galpy/pull/1418) | main | 2026-09-01 | tests: cut the time-dependent SCF coefficient tests from ~76s to ~11s |
| [#1422](https://github.com/jobovy/galpy/pull/1422) | main | 2026-09-01 | Speed up ChandrasekharDynamicalFrictionForce construction ~100x for array-capable potentials |
| [#1424](https://github.com/jobovy/galpy/pull/1424) | main | 2026-09-01 | actionAngleStaeckel C: frequencies and angles near turning points |
| [#1428](https://github.com/jobovy/galpy/pull/1428) | feat/backends-develop | 2026-09-02 | backend: keep the jax isinstance out of dynamo's way (unblocks #1204 on torch 2.14) |
| [#1431](https://github.com/jobovy/galpy/pull/1431) | main | 2026-09-03 | actionAngleStaeckel C: NaN and near-axis accuracy in the chi quadratures |
| [#1432](https://github.com/jobovy/galpy/pull/1432) | feat/backends | 2026-09-04 | Backend shard split (1/n): trace the torch-compile sweep under eager |
| [#1433](https://github.com/jobovy/galpy/pull/1433) | perf/backend-jit-eager | 2026-09-04 | Backend shard split (2/n): duration-balanced pytest-split |
| [#1436](https://github.com/jobovy/galpy/pull/1436) | refactor/backend-tests-subdir | 2026-09-04 | Backend shard: eager torch.compile for jit tests (real ~10x on the heaviest) |
| [#1437](https://github.com/jobovy/galpy/pull/1437) | feat/backends | 2026-09-04 | qdf: float the precomputed rg grid bound so forced-backend _rg survives |
| [#1438](https://github.com/jobovy/galpy/pull/1438) | backend/qdf-rg-forced-coercion | 2026-09-04 | backend ledger: xfail the forced jax/torch reds the main rebase introduced |
| [#1439](https://github.com/jobovy/galpy/pull/1439) | main | 2026-09-04 | jeans: don't let scipy.quad's default epsabs truncate sigma_r's small outer tail |
| [#1442](https://github.com/jobovy/galpy/pull/1442) | feat/backends | 2026-09-04 | backend burndown 1: numpy sigmar byte-identity + drop 4 stale xfail entries |
| [#1443](https://github.com/jobovy/galpy/pull/1443) | feat/backends | 2026-09-07 | qdf sampleV: backend-native inverse-CDF sampler (drop the numpy-forced rejection) |
| [#1444](https://github.com/jobovy/galpy/pull/1444) | backend/qdf-inverse-cdf-sampler | 2026-09-07 | qdf sampleV_interpolate: inverse-CDF on the backends, rejection on numpy |
| [#1446](https://github.com/jobovy/galpy/pull/1446) | feat/backends | 2026-09-06 | backend: vectorise the Staeckel c-vs-Python freqs/angles parity grid (slow-skip -1) |
| [#1447](https://github.com/jobovy/galpy/pull/1447) | feat/backends | 2026-09-06 | test_actionAngle: platform-proof the Staeckel chi-quadrature bars (3e-11 -> 1e-10) |
| [#1448](https://github.com/jobovy/galpy/pull/1448) | feat/backends | 2026-09-06 | sphericaldf: build the p(v\|r) inverse-CDF table on the backend (d(sample_v)/d(DF params)) |
| [#1449](https://github.com/jobovy/galpy/pull/1449) | feat/backends | 2026-09-07 | sphericaldf: cover the p(v\|r) backend-build branches #1448 left untested |
| [#1450](https://github.com/jobovy/galpy/pull/1450) | feat/backends | 2026-09-07 | backend: stop re-doing two pieces of eager work on every force evaluation |
| [#1451](https://github.com/jobovy/galpy/pull/1451) | feat/backends | 2026-09-07 | slow-skip: drop the traced Staeckel parity grid (ledger -1), re-measure three stale jit notes |
| [#1452](https://github.com/jobovy/galpy/pull/1452) | feat/backends | 2026-09-07 | Make the traced suite green: fix the vcirc probe, record the four traced timeouts |
| [#1453](https://github.com/jobovy/galpy/pull/1453) | feat/backends | 2026-09-08 | SCF: batch the general (non-axi) coefficient quadrature (traced test_density1_Order 300s+ -> 13s, ledger -1) |
| [#1454](https://github.com/jobovy/galpy/pull/1454) | backend/scf-batched-nonaxi | 2026-09-08 | SCF: contract the time-dependent coefficient quadrature over its nodes (jax 63s -> 4.3s, ledger -3) |
| [#1455](https://github.com/jobovy/galpy/pull/1455) | backend/scf-tdep-contract | 2026-09-08 | Route the orbit integrator loops through _backend_integrators (SOS_3D 573s -> 12s, ledger -3) |
| [#1456](https://github.com/jobovy/galpy/pull/1456) | main | 2026-09-08 | [pre-commit.ci] pre-commit autoupdate |
| [#1458](https://github.com/jobovy/galpy/pull/1458) | main | 2026-09-08 | Do not fork a worker per task when numcores=1 (Orbit.lyapunov 94x faster) |
| [#1459](https://github.com/jobovy/galpy/pull/1459) | feat/backends | 2026-09-08 | slow-skip: un-ledger three stale jax entries (ledger -3) |
| [#1460](https://github.com/jobovy/galpy/pull/1460) | feat/backends | 2026-09-09 | test_backend_dynamfric: validate the in-backend integrators against analytic friction |
| [#1461](https://github.com/jobovy/galpy/pull/1461) | backend/inbackend-friction-validator | 2026-09-09 | test_dynamfric_c: in-backend second arm under a backend, and chunk the loop |
| [#1463](https://github.com/jobovy/galpy/pull/1463) | feat/backends | 2026-09-10 | dynamfric + FDM: in-backend non-C arm, and un-ledger what it earns (ledger -9) |
| [#1464](https://github.com/jobovy/galpy/pull/1464) | feat/backends | 2026-09-09 | bessel_k: evaluate the ascending series in one pass (torch 2.3x on Bessel-K potentials) |
| [#1465](https://github.com/jobovy/galpy/pull/1465) | backend/unledger-dynamfric | 2026-09-10 | noninertial + Chandrasekhar orbits: in-backend non-C arm, and un-ledger what it earns (ledger -6) |
| [#1467](https://github.com/jobovy/galpy/pull/1467) | main | 2026-09-09 | test_orbit: one test case per potential for the four all-potentials walks |
| [#1468](https://github.com/jobovy/galpy/pull/1468) | main | 2026-09-10 | OblateStaeckelWrapperPotential: optional tabulated C evaluation (ntab=), ~10-16x faster C integration |
| [#1469](https://github.com/jobovy/galpy/pull/1469) | backend/unledger-noninertial | 2026-09-10 | dynamfric: one test case per potential, replacing the index chunks |
| [#1470](https://github.com/jobovy/galpy/pull/1470) | main | 2026-09-10 | Avoid duplicate builds/tests on PRs |
| [#1471](https://github.com/jobovy/galpy/pull/1471) | feat/backends | 2026-09-10 | CI: run the all-backend suite once per commit, not twice |
| [#1472](https://github.com/jobovy/galpy/pull/1472) | main | 2026-09-10 | Delete copilot setup |
| [#1473](https://github.com/jobovy/galpy/pull/1473) | backend/dynamfric-per-potential | 2026-09-10 | C bridge: keep a potential's tables whole in pot_args (streamTrack 7x under jax) |
| [#1475](https://github.com/jobovy/galpy/pull/1475) | main | 2026-09-10 | actionAngle_c: parse one potentialArg copy per OpenMP thread |
| [#1476](https://github.com/jobovy/galpy/pull/1476) | feat/backends | 2026-09-11 | streamTrack sky Jacobians batched + exp1 vectorized (ledger 18 -> 14) |
| [#1477](https://github.com/jobovy/galpy/pull/1477) | backend/sky-jac-batched | 2026-09-11 | exp1: one quadrature instead of an 80-step recurrence (torch 7.3x on ExpTruncNFW, and 8x more accurate) |
| [#1478](https://github.com/jobovy/galpy/pull/1478) | backend/sky-jac-batched | 2026-09-11 | torch-jit walks: retire both whole-walk rows (ledger 11 -> 9) |
| [#1480](https://github.com/jobovy/galpy/pull/1480) | backend/jit-rows-torch | 2026-09-11 | traced rows: re-measure the three that never were, retire all three (ledger 9 -> 6) |
| [#1481](https://github.com/jobovy/galpy/pull/1481) | feat/backends | 2026-09-11 | xfail ledger: the eager rows are gone (31 -> 6, and every survivor is jit-only) |
| [#1482](https://github.com/jobovy/galpy/pull/1482) | backend/xfail-rehome | 2026-09-11 | the numpy/backend boundary in orbit integration: warn on the demotion, and stop the friction force paying for it |
| [#1484](https://github.com/jobovy/galpy/pull/1484) | main | 2026-09-11 | ChandrasekharDynamicalFrictionForce: the force cache has never hit (2.1x on the friction path) |
| [#1485](https://github.com/jobovy/galpy/pull/1485) | feat/backends | 2026-09-13 | OblateStaeckelWrapperPotential: ntab= works on the backend, differentiably |
| [#1486](https://github.com/jobovy/galpy/pull/1486) | backend/oblatestaeckel-ntab-backend | 2026-09-13 | Orbit: a backend IC gets a backend trajectory back, whatever the integrator |
| [#1489](https://github.com/jobovy/galpy/pull/1489) | backend/orbit-returns-backend-array | 2026-09-13 | Orbit: the backend integration routes continue an integration instead of replacing it |
| [#1490](https://github.com/jobovy/galpy/pull/1490) | backend/cstm-continuation | 2026-09-13 | test_eccentricity: a traced-only bar for the one potential that integrates on GL |
| [#1491](https://github.com/jobovy/galpy/pull/1491) | backend/traced-ecc-tolerance | 2026-09-13 | eval_rect_ppoly: one gather for the cell, not sixteen |
| [#1492](https://github.com/jobovy/galpy/pull/1492) | backend/interp2d-dispatch | 2026-09-13 | slow-skip: retire one FDM row, and make three stale notes true |
| [#1493](https://github.com/jobovy/galpy/pull/1493) | backend/fdm-ledger-burn | 2026-09-13 | actionAngleStaeckel: parallelise actionsFreqsAnglesJac's Jacobian loop |
| [#1494](https://github.com/jobovy/galpy/pull/1494) | backend/jac-openmp | 2026-09-13 | sici: two cumulative products instead of two 40-term Python loops |
| [#1495](https://github.com/jobovy/galpy/pull/1495) | backend/sici-cumprod | 2026-09-13 | map_coordinates: one gather for the whole cubic tap block, not 4**D |
| [#1496](https://github.com/jobovy/galpy/pull/1496) | feat/backends | 2026-09-14 | AnyAxisymmetricRazorThinDisk: accept backend array input; xfail 5 -> 3 |
| [#1497](https://github.com/jobovy/galpy/pull/1497) | feat/backends | 2026-09-14 | constantbetadf: make eta sampling differentiable in beta |
| [#1498](https://github.com/jobovy/galpy/pull/1498) | feat/backends | 2026-09-14 | orbit: keep a backend Orbit on its backend through the no-time accessors and o() |
| [#1499](https://github.com/jobovy/galpy/pull/1499) | feat/backends | 2026-09-15 | streamdf: run the progenitor/offset setup on the backend (Phase C.3) |
| [#1500](https://github.com/jobovy/galpy/pull/1500) | feat/backends | 2026-09-14 | streamTrack: keep the accessor query axis on the backend so it can be differentiated |
| [#1501](https://github.com/jobovy/galpy/pull/1501) | main | 2026-09-16 | EinastoPotential: use regularized P so n=6 force stays finite |
| [#1502](https://github.com/jobovy/galpy/pull/1502) | main | 2026-09-15 | [auto] Update documentation notebook outputs (2026-09-15) |
| [#1503](https://github.com/jobovy/galpy/pull/1503) | feat/backends | 2026-09-16 | orbit: a backend potential parameter must not take the C-STM |
| [#1504](https://github.com/jobovy/galpy/pull/1504) | feat/backends | 2026-09-16 | backend: integrate, spline and read an orbit back on a TRACED time grid |
| [#1505](https://github.com/jobovy/galpy/pull/1505) | feat/backends | 2026-09-16 | streamdf: determine the track on a theta-dependent (traced) time grid |
| [#1506](https://github.com/jobovy/galpy/pull/1506) | feat/backends | 2026-09-16 | tests: units-consistent bar for test_streamgapdf_sample's sky-frame comparison |
| [#1507](https://github.com/jobovy/galpy/pull/1507) | feat/backends | 2026-09-19 | streamgapdf: build the impact coordinate transform on the backend |
| [#1508](https://github.com/jobovy/galpy/pull/1508) | feat/backends | 2026-09-17 | potentials: branch on a stored parameter only when it is concrete |
| [#1509](https://github.com/jobovy/galpy/pull/1509) | feat/backends | 2026-09-17 | linalg: make psd_project's gradient exact where nothing is clipped |
| [#1510](https://github.com/jobovy/galpy/pull/1510) | main | 2026-09-17 | actionAngleVerticalInverse: select tori by their actual action with a polynomial point transformation |
| [#1511](https://github.com/jobovy/galpy/pull/1511) | feat/backends | 2026-09-17 | sphericaldf: invert r(Phi) by root-find when the potential is traced |
| [#1512](https://github.com/jobovy/galpy/pull/1512) | backend/streamgapdf-traceable-init | 2026-09-19 | streamgapdf: put the kick interpolation on the backend, completing d/d(potential) |
| [#1514](https://github.com/jobovy/galpy/pull/1514) | feat/backends | 2026-09-18 | DFs: make the quasiisothermal and King constructors differentiable |
| [#1515](https://github.com/jobovy/galpy/pull/1515) | aa-vertical-canonical-mm | 2026-09-18 | actionAngleVerticalInverse: speed up the momentum-matched evaluation (Newton on the closed-form derivative) and construction |
| [#1516](https://github.com/jobovy/galpy/pull/1516) | main | 2026-09-18 | actionAngleVerticalInverse: compute the polynomial point transformation's action offset in closed form and apply it |
| [#1517](https://github.com/jobovy/galpy/pull/1517) | aa-vertical-mm-speed | 2026-09-18 | actionAngleVerticalInverse: the map's action derivatives on each torus, stored as Hermite constraints of the family |
| [#1519](https://github.com/jobovy/galpy/pull/1519) | main | 2026-09-18 | linearPotential: x2deriv and evaluatelinearx2derivs, the second derivative of 1D potentials |
| [#1521](https://github.com/jobovy/galpy/pull/1521) | feat/backends | 2026-09-19 | streamdf/streamgapdf: fix the nan d/d(potential) through the kick, and test the perturber chain end to end |
| [#1525](https://github.com/jobovy/galpy/pull/1525) | feat/backends | 2026-09-19 | Orbit: let the analytic ecc/zmax/rperi/rap differentiate w.r.t. potential parameters |
| [#1527](https://github.com/jobovy/galpy/pull/1527) | main | 2026-09-21 | actionAngleSpherical: circular and near-circular orbits |
| [#1528](https://github.com/jobovy/galpy/pull/1528) | feat/backends | 2026-09-20 | spherical DFs: differentiate w.r.t. potential parameters (construction + velocity moments) |
| [#1529](https://github.com/jobovy/galpy/pull/1529) | feat/backends | 2026-09-20 | DFs: differentiate w.r.t. their own parameters (diskdf profile, qdf hr) |
| [#1530](https://github.com/jobovy/galpy/pull/1530) | feat/backends | 2026-09-20 | dynamical friction: coerce the caller-supplied velocity triple onto the backend |
| [#1531](https://github.com/jobovy/galpy/pull/1531) | feat/backends | 2026-09-20 | Osipkov-Merritt dM/dE: integrate fQ's support and split at the ceiling corners |
| [#1532](https://github.com/jobovy/galpy/pull/1532) | feat/backends | 2026-09-21 | Orbit accessors: follow the gradient w.r.t. potential parameters |
| [#1533](https://github.com/jobovy/galpy/pull/1533) | feat/backends | 2026-09-21 | Spline1D: vector-valued y, and default to the boundary condition numpy uses |
| [#1534](https://github.com/jobovy/galpy/pull/1534) | feat/backends | 2026-09-21 | Spline1D consumers: streamTrack's dual representation and SCF's eval_ppoly clone |
| [#1535](https://github.com/jobovy/galpy/pull/1535) | feat/backends | 2026-09-21 | interpRZPotential and Orbits: one spline object per grid, not a spline plus a derived cache |
| [#1536](https://github.com/jobovy/galpy/pull/1536) | feat/backends | 2026-09-21 | CI: run the backend suite TRACED on every PR, as a matrix axis |
| [#1537](https://github.com/jobovy/galpy/pull/1537) | feat/backends | 2026-09-21 | qdf: differentiate the moments w.r.t. potential parameters, and fix _rg's no-table array branch |
| [#1538](https://github.com/jobovy/galpy/pull/1538) | feat/backends | 2026-09-21 | actionAngle grids: delegate to the exact method when the potential carries a gradient |
| [#1539](https://github.com/jobovy/galpy/pull/1539) | feat/backends | 2026-09-21 | evolveddiskdf: fix the vanishing-normalisation guard on torch (odd moments) |
| [#1540](https://github.com/jobovy/galpy/pull/1540) | feat/backends | 2026-09-22 | evolveddiskdf: honour a gradient carried by the POTENTIAL, not just by R |
| [#1541](https://github.com/jobovy/galpy/pull/1541) | feat/backends | 2026-09-22 | sphericaldf: make sample() work when the potential carries a gradient |
| [#1542](https://github.com/jobovy/galpy/pull/1542) | feat/backends | 2026-09-22 | evolveddiskdf: migrate the hierarchical velocity grid to the backend |
| [#1543](https://github.com/jobovy/galpy/pull/1543) | feat/backends | 2026-09-22 | constantbetadf: differentiate fE w.r.t. the potential parameter |
| [#1544](https://github.com/jobovy/galpy/pull/1544) | feat/backends | 2026-09-23 | backend: namespace_of() — resolve a value's own namespace, not the forced one |
| [#1546](https://github.com/jobovy/galpy/pull/1546) | feat/backends | 2026-09-23 | constantbetadf: differentiable w.r.t. the potential under jax too (eager grad) |
| [#1547](https://github.com/jobovy/galpy/pull/1547) | feat/backends | 2026-09-24 | sphericaldf: differentiable backend-key sampling w.r.t. the potential |
| [#1549](https://github.com/jobovy/galpy/pull/1549) | feat/backends | 2026-09-24 | dynamical friction: differentiable w.r.t. the host potential |
| [#1550](https://github.com/jobovy/galpy/pull/1550) | feat/backends | 2026-09-24 | psd_project: exact gradient on clipped slices; diffrax accepts a potential list |
| [#1551](https://github.com/jobovy/galpy/pull/1551) | feat/backends | 2026-09-24 | kingdf: differentiable w.r.t. W0 (forward sensitivities of the King ODE) |
| [#1552](https://github.com/jobovy/galpy/pull/1552) | feat/backends | 2026-09-24 | kingdf: solve the King ODE at rtol=1e-10 (scipy's default left ~1e-4) |
| [#1553](https://github.com/jobovy/galpy/pull/1553) | main | 2026-09-24 | actionAngleSpherical: decide boundedness at infinity, not at a fixed radius |
| [#1554](https://github.com/jobovy/galpy/pull/1554) | feat/backends | 2026-09-24 | potentials: finite parameter gradients at r = 0 and r = inf |
| [#1555](https://github.com/jobovy/galpy/pull/1555) | feat/backends | 2026-09-24 | special: accurate backend hyp2f1 at large negative z |
| [#1556](https://github.com/jobovy/galpy/pull/1556) | feat/backends | 2026-09-25 | osipkovmerrittdf: differentiable velocity sampling w.r.t. the potential |
| [#1557](https://github.com/jobovy/galpy/pull/1557) | feat/backends | 2026-09-25 | constantbetadf: ~4x faster construction; spline solves in O(n log n) |
| [#1558](https://github.com/jobovy/galpy/pull/1558) | feat/backends | 2026-09-25 | constantbetadf: construct, evaluate and sample under jax.jit |
| [#1559](https://github.com/jobovy/galpy/pull/1559) | feat/backends | 2026-09-25 | eddingtondf, kingdf (M, rt), osipkovmerrittdf: under jax.jit |
| [#1560](https://github.com/jobovy/galpy/pull/1560) | main | 2026-09-25 | actionAngleStaeckel: the focal length from a potential that supplies one |
| [#1561](https://github.com/jobovy/galpy/pull/1561) | staeckel-focal-length | 2026-09-25 | PerfectEllipsoidPotential: the focal length of an axisymmetric oblate perfect ellipsoid |
| [#1566](https://github.com/jobovy/galpy/pull/1566) | feat/backends | 2026-09-25 | kingdf: d/dW0 under jax.jit (diffrax) and torch.compile |
| [#1567](https://github.com/jobovy/galpy/pull/1567) | feat/backends | 2026-09-26 | orbit: C integrators under torch.compile (and no silent detach) |
| [#1569](https://github.com/jobovy/galpy/pull/1569) | feat/backends | 2026-09-26 | orbit: method='torchode', a torch.compile-able in-backend integrator |
| [#1572](https://github.com/jobovy/galpy/pull/1572) | feat/backends | 2026-09-26 | kingdf: correct second (and higher) W0-derivatives on jax and torch |
| [#1573](https://github.com/jobovy/galpy/pull/1573) | feat/backends | 2026-09-26 | tests: whole workflows under jax.jit / torch.compile, value and gradient |
| [#1574](https://github.com/jobovy/galpy/pull/1574) | feat/backends | 2026-09-29 | qdf under torch.compile; Orbit.E at a traced time under jax.jit |
| [#1575](https://github.com/jobovy/galpy/pull/1575) | feat/backends | 2026-09-29 | streamTrack: spray -> track under torch.compile |
| [#1584](https://github.com/jobovy/galpy/pull/1584) | feat/backends | 2026-09-29 | backend: torchode default; stream workflows fully compile under inductor |
| [#1585](https://github.com/jobovy/galpy/pull/1585) | main | 2026-09-28 | potentials: machine-precision small r (NFW, Burkert, TwoPower), finite edge limits, Staeckel midplane vmin |
| [#1586](https://github.com/jobovy/galpy/pull/1586) | main | 2026-09-28 | eddingtondf: f(E) accurate near Emin (NFW was NaN/negative there) |
| [#1578](https://github.com/jobovy/galpy/pull/1578) | feat/backends | 2026-09-29 | potentials, eddingtondf: backend side of #1585 and #1586 (with #1579 folded in) |
| [#1594](https://github.com/jobovy/galpy/pull/1594) | feat/backends | 2026-09-29 | actionAngleSpherical: the backend solves main's relative radial problem (retires the rebase's 8 xfails) |
| [#1593](https://github.com/jobovy/galpy/pull/1593) | feat/backends | 2026-09-30 | torch.compile: the streamTrack fit and the qdf constructor compile (replace the #1574/#1575 opt-outs) |
| [#1600](https://github.com/jobovy/galpy/pull/1600) | feat/backends | 2026-09-30 | kingdf: W0 compiles under torch.compile (no opaque scipy solve) |
| [#1598](https://github.com/jobovy/galpy/pull/1598) | feat/backends | 2026-09-30 | spherical DFs: osipkovmerrittdf.sample under jax.jit; exact traced r(Phi) near Emin; constantbetadf backend fE without the calibrated limit |
| [#1599](https://github.com/jobovy/galpy/pull/1599) | feat/backends | 2026-09-30 | Orbit.rperi/rap/zmax/e(analytic=True): backend-native and differentiable w.r.t. the phase-space point |
| [#1602](https://github.com/jobovy/galpy/pull/1602) | feat/backends | 2026-09-30 | streamdf: vmap the differentiable track's chunk loop (shared constant steps keep d/dparam exact) |
| [#1603](https://github.com/jobovy/galpy/pull/1603) | feat/backends | 2026-10-01 | GPU (CUDA) validation: --device cuda test option and CUDA fixes |
| [#1604](https://github.com/jobovy/galpy/pull/1604) | feat/backends | 2026-10-01 | GPU numpy islands (1/5): coords + conversion; to_host for sequences, on_host |
| [#1605](https://github.com/jobovy/galpy/pull/1605) | feat/backends | 2026-10-01 | GPU numpy islands (2/5): potential |
| [#1606](https://github.com/jobovy/galpy/pull/1606) | feat/backends | 2026-10-01 | GPU numpy islands (3/5): actionAngle |
| [#1607](https://github.com/jobovy/galpy/pull/1607) | feat/backends | 2026-10-01 | GPU numpy islands (4/5): Orbit |
| [#1608](https://github.com/jobovy/galpy/pull/1608) | feat/backends | 2026-10-01 | GPU numpy islands (5/5): df, ars, scipy spline |
| [#1610](https://github.com/jobovy/galpy/pull/1610) | feat/backends | 2026-10-01 | Orbit: backend ICs in sky coordinates convert differentiably (radec/lb) |
| [#1611](https://github.com/jobovy/galpy/pull/1611) | main | 2026-10-01 | Orbit: setting up from sky coordinates with uvw=True no longer modifies the input |
| [#1596](https://github.com/jobovy/galpy/pull/1596) | main | 2026-10-01 | TwoPowerSphericalPotential forces via the incomplete beta function (accurate for all alpha, beta) |
| [#1597](https://github.com/jobovy/galpy/pull/1597) | feat/backends | 2026-10-01 | TwoPowerSphericalPotential forces via the incomplete beta function, backend-native and differentiable |
| [#1612](https://github.com/jobovy/galpy/pull/1612) | feat/backends | 2026-10-01 | SCFPotential.from_density: differentiable w.r.t. the density's parameters (static and time-dependent) |
| [#1614](https://github.com/jobovy/galpy/pull/1614) | feat/backends | 2026-10-02 | test_backend_compile_workflows: cap inductor SIMD at 256 bits for kingdf_W0 (GCC ICE on AVX-512 runners; confirmed on an avx512f runner 2026-10-03) |
| [#1615](https://github.com/jobovy/galpy/pull/1615) | feat/backends | 2026-10-02 | gammainc/gammaincc on jax: vectorized series/CF (~100x faster on CPU), scipy-accurate everywhere |
| [#1616](https://github.com/jobovy/galpy/pull/1616) | feat/backends | 2026-10-02 | Orbit.from_name (SIMBAD) tests run under forced jax/torch |
| [#1617](https://github.com/jobovy/galpy/pull/1617) | feat/backends | 2026-10-03 | quasiisothermaldf: constructor compiles under torch.compile without an opt-out (last #1574 stopgap) |
| [#1618](https://github.com/jobovy/galpy/pull/1618) | feat/backends | 2026-10-03 | MultipoleExpansionPotential.from_density: differentiable w.r.t. the density's parameters (static and time-dependent) |
| [#1619](https://github.com/jobovy/galpy/pull/1619) | feat/backends | 2026-10-03 | Dynamical friction (backend, eager): one force factor per EOM step; FDM computes only its regime |
| [#1620](https://github.com/jobovy/galpy/pull/1620) | feat/backends | 2026-10-03 | conftest: the runner's CPU (model, AVX2/AVX-512/AMX, inductor vec ISA) in the pytest header |
| [#1613](https://github.com/jobovy/galpy/pull/1613) | feat/backends | 2026-10-04 | float64 by default: torch's float32 default no longer crashes galpy or degrades its internals (user float32 tensors stay float32) |
| [#1621](https://github.com/jobovy/galpy/pull/1621) | feat/backends | 2026-10-04 | qdf sampler: power-of-4 buckets + no pure_callback for concrete inputs; test_sampleV_interpolate un-skipped under jax (2837 s -> 118 s) |
| [#1623](https://github.com/jobovy/galpy/pull/1623) | feat/backends | 2026-10-04 | test_orbit_interface_staeckel_defaultdelta: compare at the Orbit's own delta (1-ulp jit/eager difference) |
| [#1624](https://github.com/jobovy/galpy/pull/1624) | feat/backends | 2026-10-04 | numpy islands: streamgapdf orbit-integration kick + constantbetadf torch pin removed; rest classified |
| [#1625](https://github.com/jobovy/galpy/pull/1625) | feat/backends | 2026-10-04 | jit mode: static objects keyed by type + attributes (stale traces after a wrapped potential changed; per-call retraces) |
| [#1626](https://github.com/jobovy/galpy/pull/1626) | (draft) | 2026-10-04 | CI for feat/backends rebased onto main (applied by Jo; closed) |
| [#1627](https://github.com/jobovy/galpy/pull/1627) | feat/backends | 2026-10-05 | actionAngleVerticalInverse (1/3): the momentum-matched map evaluates natively under jax/torch |
| [#1628](https://github.com/jobovy/galpy/pull/1628) | feat/backends | 2026-10-05 | actionAngleVerticalInverse (2/3): momentum-matched family built natively, differentiable in the potential's parameters, jit-able end to end |
| [#1630](https://github.com/jobovy/galpy/pull/1630) | feat/backends | 2026-10-05 | actionAngleVerticalInverse (3/3): the legacy modes evaluate natively (numpy construction); map_coordinates order 5 + multi-reflection mirror fix |
| [#1631](https://github.com/jobovy/galpy/pull/1631) | feat/backends | 2026-10-06 | actionAngleVerticalInverse runs on CUDA (energy grid to host; tests CUDA-clean) |
