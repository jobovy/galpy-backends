# galpy multi-backend (JAX / PyTorch / array-API) + autodiff + 3D variational equations

> Master plan / index. Per-track detailed specs live as separate files in `~/Repos/galpy-jax/` (created at execution start — see "Log & per-track plan files"). This is a large, multi-month, multi-PR effort.

## Context — why this is being done

galpy today is numpy/scipy + C. The goal is to make **all of galpy** differentiable and backend-agnostic — JAX, PyTorch, and (via the array-API) generically more — so galpy calculations can sit inside autodiff/ML models of galactic dynamics, while keeping the fast C paths. Defining success: **the entire galpy test suite passes under all backends (numpy/scipy, jax, pytorch), at the same tolerances.**

A second, independently-valuable pillar: implement the **3D variational equations** (state-transition matrix `M = ∂x₁/∂x₀`, i.e. the full 3D potential Hessian) in C for all potentials. This gives galpy 3D tangent dynamics / Lyapunov exponents in their own right, **and** makes fast C orbit integration differentiable (forward-sensitivity STM → first derivatives via `M^T·λ`, no backward C call).

Design rationale and settled decisions: `~/Repos/galpy-jax/galpy_backend_autodiff_discussion.md`. This plan grounds them in galpy's actual code (verified by exploration).

### Scope = ALL of galpy, maximal on every axis (user-confirmed)
- **Everything runs and differentiates under every backend**: all **potentials**, all **orbit** integration, all **action-angle** methods (**including the C-backed ones** — Staeckel/Adiabatic via analytic custom rules using the derivative integrands already in C), and the **entire `df` module** (disk DFs, spherical DFs, quasi-isothermal, and **streams + stream tracks**). Plus the `util` compute modules (`coords`, `conversion`, `special`) that these sit on.
- **Pure-backend special functions, native-preferring + cleanly swappable**: reimplement `assoc_legendre` (Bonnet recursion), `gamma/gammainc/gammaincinv`, `hyp2f1`, `erf`, etc. AD-friendly, so even SCF/Multipole/Einasto/spherical-DFs work on jax/torch — but route to the **native** backend implementation when present and fall back to galpy's isolated module otherwise, so native versions are adopted automatically and the fallback is trivially deletable.
- **Full parameter gradients, all potentials** (`df/dθ`), not only initial conditions.
- **First-class diffrax/torchdiffeq runtime integrators** (unlocks higher-order orbit autodiff the first-order C-STM cannot give).
- **Lyapunov exponents** from the variational equations — a separate PR.
- **Two branch streams (the C variational work is decoupled from the backends):**
  - The **3D variational equations + Lyapunov** (Track B) and the **C parameter-sensitivity P-block** (Track C C-side) are standalone galpy features that the backends *need* but that do *not* need the backends → they merge to **`main` directly** (each via its own short-lived feature branch, e.g. `feat/variational3d`, `feat/lyapunov`, `feat/param-sensitivity`).
  - All **jax/torch/array-API backend work** (Track 0 infra, Track A potentials, Track D autodiff wrappers + in-backend integrators, Track E action-angle backends, Track F df) lives on the **long-lived `feat/backends`** branch, which periodically merges `main` in to consume the variational/param C work; final review merges `feat/backends → main`.

---

## Goal & success criteria
1. `pytest` green under numpy, jax, **and** torch for the whole suite — **same tests, same tolerances** — **plus** substantial new backend-specific derivative tests (new CI jobs, not the main matrix). See **Testing infrastructure**.
2. Every potential, orbit method, action-angle method, and DF (incl. streams) evaluates and **differentiates** under all backends.
3. 3D variational equations in C for all potentials: `integrate_dxdv` works for `phasedim==6`; 3D Liouville + symplecticity tests pass; Lyapunov exponents available.
4. Differentiable C orbit integration (IC **and** parameter gradients) via STM custom rules + first-class in-backend diffrax/torchdiffeq integrators (higher-order autodiff).
5. Coverage held at **all-but-27-lines**; running log in `~/Repos/galpy-jax` → paper.

### Tolerance policy (hard requirement)
- **numpy path is byte-identical** — guaranteed, because numpy is the default and `get_namespace` is a pure pass-through for numpy/Python inputs; no numpy-path test or tolerance changes.
- **jax/torch must pass the *same* tests at the *same* tolerances.** Enable jax float64 (`jax_enable_x64`); torch defaults to float64. A tolerance may be loosened **only if absolutely necessary and then only slightly**, per-test, with a one-line justification logged in the coverage/tolerance ledger. No blanket per-backend tolerance relaxation.

---

## Settled architecture (do not re-litigate)
- **Resolve the backend by following the data.** One resolver, precedence: explicit `xp=` → `array_namespace(*inputs)` → `contextvars` context/global → numpy default. Engine: `array-api-compat`'s `array_namespace`, wrapped behind `galpy.backend.get_namespace` with a vendored fallback.
- **Inject dispatch at the PRIVATE compute layer** (`_evaluate/_Rforce/.../_phizderiv`, spherical `_revaluate/_rforce/_r2deriv`, and the DF/AA private methods). Public methods, the unit decorators (`@potential_physical_input`, `@physical_conversion`), and `_amp` stay backend-agnostic — verified: the input decorator passes jax/torch arrays through; the output decorator multiplies by a Python-float `fac` (one rare numpy-array branch needs a guard).
- **Backend-specific reimplementations live in clearly-named `galpy.backend` modules dispatched by the resolver** — never `if backend==…` in physics code.
- **Prefer native backend functions; keep galpy fallbacks cleanly separable** (special-functions router; auto-adopt native as it ships; fallback trivially deletable).
- **Orbit autodiff = forward-sensitivity STM, not adjoint** (n=6; backward = single `M^T·λ` matvec, no backward C call).
- **Preserve device & dtype** by anchoring constructed arrays on inputs (`xp.zeros_like`, `xp.where`, `*_like`); never bare `xp.zeros(...)` in a method that received arrays.

---

## Grounded facts that shape the work

**Backend surface.** Uniform `import numpy`. Hazards during namespace-swap: in-place mutation `out[idx]=val` (potentials: 11 files, DehnenBar 30+; `coords.py` clip/wrap at 249-250/253/330/610/708/780; AA `Omegaphi[indx]*=-1`; streamdf 20+); `isinstance(R,float)`/value branching (8 potential files; coords scalar/shape checks); `numpy.any/all` control flow; mutable caches (EllipsoidalPotential hash; quasiisothermaldf `_rgInterp`; diskdf `_surfaceInterpolate`). jax 0.9.2 installed; torch/array-api-compat/diffrax/torchdiffeq **not**. `SphericalPotential.py` is the single 3D→1D consolidation point.

**util.** `coords.py` (transforms used everywhere — backend-clean it, fix the clip/wrap mutations via `xp.where`/`clip`, replace `isinstance` shape checks with `getattr(x,'ndim',0)`) and `conversion.py` (clean scalar math — only the rare `physical_conversion` numpy-array-`fac` branch needs a guard) are the critical compute-path util modules. `util/special.py:compute_legendre` (scipy `assoc_legendre_p_all`/`lpmn`) is on the MultipoleExpansion compute path → folded into the pure-backend special-functions work. `quadpack/plot/multi/ars/leung_dop853.py/symplecticode.py/config/_optional_deps` stay numpy-only (sampling/tooling/Python-integrators not on the differentiable path — C is used there).

**3D variational (C).** `struct potentialArg` (galpy_potentials.h) has full-3D `R2deriv/phi2deriv/Rphideriv` pointers + planar variants, but **no `z2deriv/Rzderiv/zphideriv`** — must add. 2D RHS `evalPlanarRectDeriv_dxdv` (integratePlanarOrbit.c:1050-1101) is the correct template; 3D `evalRectDeriv_dxdv` (integrateFullOrbit.c:1195-1252) is a **broken stub** (`*a=0 //BOVY: PUT IN Z2DERIVS`, wrong indices, `LCOV_EXCL`). **No potential has the z-second-derivatives in C** except possibly **SpiralArmsPotential.c** (the only complete 3D Hessian — wire first). In Python, *many* potentials have analytic `_z2deriv/_Rzderiv` and some don't (they fall back to `NumericalPotentialDerivativesMixin`). So the per-potential C work is **implement** the 3D Hessian: port analytic Python where it exists; **derive** analytic formulas where it doesn't (or add a C numerical-derivative fallback). The forward orbit RHS uses no second derivatives, so the whole 3D dxdv path is dead today (consistent with `integrate_dxdv` restricted to `phasedim==4`).

**The 3D Jacobian math (verified).** Cartesian accel `ax=cosφ·FR − sinφ/R·Fφ`, `ay=sinφ·FR + cosφ/R·Fφ`, `az=Fz`. The accel Jacobian `K = ∂(ax,ay,az)/∂(x,y,z)` is the **symmetric** Cartesian Hessian `−∇∇Φ`. Its **(x,y) block is byte-for-byte the proven 2D code**; the **new z-coupling** is `dFxdz=−cosφ·Rzderiv+sinφ/R·zphideriv (=dFzdx)`, `dFydz=−sinφ·Rzderiv−cosφ/R·zphideriv (=dFzdy)`, `dFzdz=−z2deriv`. Symmetry of K is a built-in unit test. (Python `phizderiv` = C `zphideriv` = ∂²Φ/∂z∂φ.)

**Action-angle differentiation.** **Staeckel C already computes all the Leibniz derivative integrands** (dJR/dE, dJR/dLz, dJR/dI3, dJz/…, actionAngleStaeckel.c:1883-1976) and frequencies from them → a custom_vjp/autograd.Function can wrap the C forward and use these for the backward w.r.t. (E,Lz,I3); add ∂(E,Lz,I3)/∂(x,v) and ∂/∂θ for full chain. **Adiabatic C** exposes only actions (frequencies via Python Spherical+Vertical) — add derivative integrands or use the Python path. Pure-Python **Isochrone/Harmonic/Vertical** are closed-form (namespace-swap + fix masked writes); **Spherical** needs backend-agnostic root-find (implicit-diff) + fixed-order GL quadrature. **Grid** classes need backend-agnostic interpolation; **IsochroneApprox** differentiates via the orbit STM path. **`actionAngleTorus` is OUT OF SCOPE** — it wraps an external C++ Torus library (`actionAngleTorus_c_ext/torus/`, McMillan/Binney/Dehnen) we will not modify; it stays numpy-only and is not made backend-aware.

**Tests/CI.** `test_liouville_planar` (test_orbit.py:705) integrates 4 basis deviation vectors → `det==1`; generalize to 6/6×6. Most tests loop manually; `conftest.pytest_generate_tests` parametrizes 2. `build.yml` ≈20 `include` jobs (`TEST_FILES`+`REQUIRES_*`). `setup.py extras_require` = {test, docs}. Coverage only on py3.14+ubuntu (lcov for C). `.coveragerc` omits snapshot/plot/etc; codecov ignores torus+xsf.

---

## Track structure & PR roadmap

Notation `[blocked-by: …]`, `∥` parallel-safe. Big fan-outs run as orchestrated **workflows** (one agent per potential/DF family in its own worktree). Serialization points (must merge before their fan-out branches): **P1**, **Pvar-struct**, **Pparam-design**.

### Track 0 — Backend infrastructure
- **PR0 (pilot, pre-PR).** `galpy/backend/` resolver skeleton + migrate **Plummer + Isochrone** + `tests/test_backend_pilot.py` (parity numpy/jax/torch + grad-vs-FD + identity `d(_evaluate)/dR == −_Rforce`). Add `_TORCH_LOADED`/`_ARRAY_API_COMPAT_LOADED`. Validates the whole approach. `[blocks: all]`
- **P1 — `galpy.backend` package, numpy no-op.** Resolver + `set_backend()`/`backend()` context manager + config `[backend]` + `special.py` **native-preferring router** + `linalg.py`; AST convention guard (opt-in allowlist). numpy pass-through → suite unchanged. `[blocked-by: PR0]` `[blocks: everything]`
- **P0-util.** Backend-clean `coords.py` (transforms; fix clip/wrap mutations + shape branching) and the `conversion.py` `fac` branch. `[blocked-by: P1]` `[blocks: orbit/df backends]`
- **Pspecial — pure-backend special functions (native-preferring, isolated fallbacks).** AD-friendly `assoc_legendre` (Bonnet recursion, replaces `util/special.compute_legendre` on the MultipoleExpansion path), `gamma/gammainc/gammaincinv`, `hyp2f1`, `erf`. Router uses native (`jax.scipy.special`/`torch.special`/`scipy.special`) when present, else `galpy/backend/special/_fallback/<fn>.py`; a capability test pins the shrinking fallback list and asserts native-vs-fallback agreement. Validated ~1e-12 vs scipy + grad-vs-FD. `[blocked-by: P1]` `[blocks: SCF/Multipole + spherical-DF backends]`
- **Pci / parametrization (seeded here, finalized last).** `pytest_addoption(--backend)` + session autouse fixture wrapping tests in `with galpy.backend.use(<backend>)` so the **existing suite re-runs verbatim** per backend (jax x64 in fixture). New `build.yml` jax/torch jobs + full-suite re-run; `REQUIRES_TORCH` install; extras `jax/torch/backend/diffrax/torchdiffeq`. Coverage ledger to all-but-27. `[seeded by P1]`

### Track A — Potentials namespace-swap (fan-out, worktrees off P1)
~6 family PRs, each fixing hazards (`out[idx]=val`→`xp.where` w/ safe denominators; `isinstance`→shape-based; cache→functional), keeping numpy bit-identical, adding family parity + grad-vs-FD tests:
- **P2.1 Spherical family** (migrate `SphericalPotential.py` → covers 6+) — first. `[blocked-by: P1]`
- **P2.2 Disk/bar analytic** ∥ **P2.3 Ellipsoidal** ∥ **P2.4 Bars/heavy-mutation**
- **P2.5 special-backed** (Einasto, PowerSphericalwCutoff, DoubleExp, SCF/Multipole/DiskSCF) `[blocked-by: P1, Pspecial]`
- **P2.6 Wrappers** (compose children) — last `[blocked-by: P2.1–P2.5]`
- **P2.7 Dissipative / velocity-dependent forces** (`DissipativeForce`/`planarDissipativeForce`, `ChandrasekharDynamicalFrictionForce`, `FDMDynamicalFrictionForce`) — `f=f(x,v)`; thread velocity through the namespace-swap. Enables dissipative-orbit autodiff via the Track D in-backend integrator (see Track B "Open consideration"). `[blocked-by: P1]`

### Track B — C 3D variational equations (IC gradients)
- **Pvar-struct (serialization point).** Add `z2deriv/Rzderiv/zphideriv` pointers; add `calc*` aggregators (skip NULL safely); NULL-init at every `init_potentialArgs`/`parse_leapFuncArgs_Full` callsite; wire SpiralArms. RHS unchanged → suite unaffected. `[blocks: Pvar-pot.*, Pvar-rhs]`
- **Pvar-pot.x (fan-out, ~7 PRs, many parallel agents).** **Implement** the per-potential 3D Hessian in C (port analytic Python where present; derive where absent). Families: disk/bar analytic, spherical (via r2deriv+transform), ellipsoidal (psi/mdens+transform), SCF/Multipole/DiskSCF (harmonics), bars/spirals (non-axi, zphideriv), wrappers (compose) `[blocked-by: 1–5]`, interpRZ (spline). Each adds C-Hessian-vs-Python checks. `[blocked-by: Pvar-struct]`
- **Pvar-rhs (keystone).** Replace the broken stub with the correct 6×6-Jacobian RHS (reuse proven 2D (x,y) block + add z-coupling; fix indices; remove `LCOV_EXCL`). Lift `integrate_dxdv` to `phasedim==6` (6D cyl↔rect deviation transform; revive `integrateFullOrbit_dxdv_c`); batched C convenience returning `(nt,6,6) M`. Tests: **3D Liouville `det M=1`**, **symplecticity `MᵀΩM=Ω` (Cartesian, which IS canonical)**, **flow-direction `M·f(x₀)=f(x(t))`**, **closed-form Kepler/3D-harmonic STM**, **Lz conserved-integral**, **2D-reduction bridge** (planar IC matches trusted planar STM to 1e-10). `[blocked-by: Pvar-struct + enough Pvar-pot; blocks: Track D]`
- **Pvar-lyapunov (separate PR).** Lyapunov exponents from the variational equations (largest-eigenvalue growth of the STM / Benettin renormalization), exposed on `Orbit`. Validation per design doc §4: integrable→0 signature (isochrone/harmonic/spherical), Hénon–Heiles / flattened-log regression vs literature values. `[blocked-by: Pvar-rhs]`

#### Dissipative / velocity-dependent forces (in scope — full C-STM, "route 2")
Dissipative forces (`ChandrasekharDynamicalFrictionForce`, `FDMDynamicalFrictionForce`; base `DissipativeForce`/`planarDissipativeForce`; in C: `requiresVelocity` + `RforceVelocity/zforceVelocity/phitorqueVelocity`) are **not** derived from a potential and **depend on velocity**: `f = f(x, v)`. The maximal C-STM implementation is **in scope** (user-confirmed):
- The variational Jacobian is the general `J = [[0, I],[∂F/∂x, ∂F/∂v]]` — the lower-left block is the force's *position* Jacobian (not −∇∇Φ, generally **non-symmetric**) and there is a nonzero *velocity* block `∂F/∂v` (the friction/damping term). `Ṁ = J M` still holds; for conservative forces `∂F/∂v=0`, `∂F/∂x=K`, recovering the symplectic case. **The C variational RHS (Pvar-rhs) is written general enough from the start to carry this velocity-coupled, non-symmetric `J`** (the conservative case is the special case `∂F/∂v=0`).
- The flow is **non-conservative**: `det M = exp(∫ tr(∂F/∂v) dt) ≠ 1` and `MᵀΩM=Ω` does **not** hold — so dissipative potentials are **excluded from the Liouville/symplecticity tests**; validate via finite-difference-of-the-flow and known damping behavior. The STM `M = ∂x₁/∂x₀` still gives correct IC gradients.

PRs (Track B sub-track, → `main`):
- **Pvar-dissip-struct.** Add velocity-dependent Jacobian pointers to `struct potentialArg` for dissipative forces: `∂F/∂(R,z,φ)` and `∂F/∂(vR,vT,vz)` (the position- and velocity-Jacobian of each force component), with NULL-safe aggregators, mirroring Pvar-struct. The non-symmetric/velocity-coupled `J` assembly path in the RHS.
- **Pvar-dissip-force.x.** Implement `∂F/∂(x,v)` in C for each dissipative force (`ChandrasekharDynamicalFrictionForce`, `FDMDynamicalFrictionForce`), validated vs finite-difference of the force and vs the in-backend AD reference. `[blocked-by: Pvar-dissip-struct, Pvar-rhs]`

The **in-backend integrator (Track D)** additionally provides dissipative-orbit derivatives "for free" via AD of the namespace-swapped `f(x,v)` (Track A P2.7) — kept as the independent correctness reference for the C-STM dissipative Jacobians.

### Track C — C parameter sensitivity (parameter gradients, all potentials)
- **Pparam-design + struct.** Augment with `P=∂x/∂θ`, `Ṗ=J·P+∂f/∂θ`, `P(0)=0`; backward `θ̄=Pᵀλ`. C interface for `∂(force)/∂(param)` per potential. The **in-backend integrators (Track D) already give param-grads for all potentials via AD** of the namespace-swapped forces — immediate full coverage + correctness reference; this C track is the **fast** path rolled out family-by-family. `[blocked-by: Pvar-struct, P2.*]`
- **Pparam-pot.x (fan-out, worktrees).** `∂f/∂θ` in C per family + augmented integrator; validate vs in-backend AD + FD. `[blocked-by: Pparam-design]`

### Track D — Autodiff wrappers + in-backend integrators
- **Pjax-orbit ∥ Ptorch-orbit (C-STM).** `jax.custom_vjp` (forward = C orbit via `jax.pure_callback` returning STM residual; backward `M^T·λ̄`; `vmap_method="expand_dims"` reusing the multi-orbit C loop) and `torch.autograd.Function` (save M; backward `Mᵀ·grad`; `gradcheck` mandatory). In `galpy/backend/_jax|_torch/orbit_stm.py`; `Orbit.integrate` routes by IC namespace, API numpy-identical. `[blocked-by: P1, P0-util, Pvar-rhs]`
- **Pinbackend — first-class diffrax/torchdiffeq.** In-backend ODE integration of backend-agnostic forces (diffrax/torchdiffeq) as selectable runtime backends → **higher-order autodiff** + **parameter gradients for all potentials** via AD. Also the natural path for **dissipative / velocity-dependent forces** (`f=f(x,v)`): AD through the namespace-swapped force gives their orbit derivatives with no hand-coded Jacobian (see Track B "Open consideration"). Runtime extras; resolver dispatch; cross-validate vs C-STM + Liouville. `[blocked-by: Track A forces (incl. P2.7), P0-util]`

### Track E — Action-angle (ALL methods, incl. C-backed)
- **Paa-python.** Namespace-swap exact pure-Python methods: Isochrone, Harmonic, Vertical (closed-form + fix masked writes), then Spherical (backend-agnostic fixed-iteration root-find via implicit diff + fixed-order Gauss-Legendre quadrature, vectorized). Correctness reference + all-orders fallback. `[blocked-by: P1, P2.1]`
- **Paa-staeckel-c.** Wrap the C Staeckel (and Adiabatic) with `jax.custom_vjp`/`torch.autograd.Function`: forward = C `actionsFreqs(Angles)`; backward uses the **existing C Leibniz derivative integrands** (dJ/dE, dJ/dLz, dJ/dI3) composed with ∂(E,Lz,I3)/∂(x,v) and (Track C) ∂/∂θ. Add the missing derivative integrands to Adiabatic C (or route Adiabatic through the Python Spherical+Vertical path). `[blocked-by: P1, P0-util; Paa-analytic for the math]`
- **Paa-analytic.** Analytic IFT (turning points, with √-branch-point regularization via the existing `r=r_peri+t²` substitution) + Leibniz action-derivative rules for Spherical/Isochrone; mirror the C Staeckel structure. `[blocked-by: Paa-python]`
- **Paa-grid-approx.** Backend-agnostic interpolation for StaeckelGrid/AdiabaticGrid; IsochroneApprox differentiability via the orbit STM path. **`actionAngleTorus` is explicitly excluded** (external library — left numpy-only). `[blocked-by: Paa-staeckel-c, Track D]`

### Track F — df module (ALL DFs + streams) — the largest, mostly downstream
Depends on potential + orbit + action-angle backends. Batched by difficulty:
- **Pdf.1 easy** (df base, surfaceSigmaProfile, jeans) — establishes the DF backend-swap + backend-agnostic quadrature pattern. `[blocked-by: P1]`
- **Pdf.2 spherical DFs** (sphericaldf + isotropic/constantbeta/osipkovmerritt/eddington/king variants) — nested `quad`→backend GL quadrature, `gamma/hyp2f1`→Pspecial, CDF-inversion sampling→`jax.random`/implicit-diff. `[blocked-by: Pspecial, Pdf.1]`
- **Pdf.3 disk DFs** (diskdf, evolveddiskdf) — actionAngleAdiabatic backend, `integrate`/`integrate_dxdv` backend, scipy.optimize→backend root-find, spline caches→functional. `[blocked-by: Track E (Adiabatic), Track D, Pdf.1]`
- **Pdf.4 quasiisothermaldf** — actionAngleIsochrone/Staeckel backend, tplquad→backend, RectBivariateSpline→backend interp, `_rgInterp` cache→functional. `[blocked-by: Track E, Pdf.1]`
- **Pdf.5 streams** (streamspraydf, streamgapdf, streamTrack, then streamdf) — the ultra-hard set: sampling/RNG, orbit integration, action-angle (via the in-scope methods — IsochroneApprox/Staeckel/Spherical, **not** Torus), `make_smoothing_spline`/`cKDTree`, and streamdf's iterative track assembly (refactor ~60 loops + in-place mutations + eigen-slerp to functional/vectorized form). streamdf last; likely several sub-PRs. `[blocked-by: Pdf.3/Pdf.4, Track D, Track E]`

### Track G — Figures & diagnostics for the paper(s) (cross-cutting, continuous)
Summary/diagnostic plots that become paper figures, produced **as each track lands** (not batched at the end), doubling as living regression evidence for the PRs. **Hard requirement (user): every figure ships version-controlled code that re-runs deterministically and reproduces the *identical* plot** — so against the *final* galpy code we can re-render and diff every paper figure (a `verify-all` reproducibility gate). Detailed spec/log: `~/Repos/galpy-jax/plan/trackG_figures.md`.
- **G0 — figure harness (first).** `paper/figures/_harness/`: a `@figure(name, seed)` decorator (seeds numpy/jax/torch, applies shared style, writes a byte-stable PDF, records `sha256` + git sha + env into `manifest.json`) + `run`/`verify`/`list` CLI. Benchmarks split **measure** (machine-stamped `data/*.json`) from **plot** (byte-stable); `verify` regenerates and asserts the hash. `[blocks: G1–G4]`
- **G1 — backend comparison** (numpy/jax/torch/array-API): value-parity grids over potentials × methods × the 6 second-derivatives; AD-vs-analytic identity residuals; device/dtype preservation. *CPU, available now.* `[uses: Track 0/A]`
- **G2 — speed/scaling**: throughput vs N, orbit-integration speed (C-STM vs in-backend diffrax/torchdiffeq, single vs batched), autodiff overhead, jit/compile amortization — **CPU now; GPU figures after #891/#898 land (user moves me to a GPU box); identical plotting code, re-measure on GPU.** `[uses: Track 0/A/D]`
- **G3 — autodiff worked examples**: potential-parameter gradients; orbit sensitivity to ICs (STM columns) and to parameters (vs FD); a gradient-descent orbit-fit demo; action-angle gradients; **stream-track derivatives ∂(track)/∂(params)** — the showpiece. `[uses: Track A/D/E/F]`
- **G4 — variational/dynamics diagnostics**: Liouville/symplecticity drift vs t; Lyapunov integrable→0 vs chaotic (Hénon–Heiles / flattened-log); 2D-reduction-bridge residual. *Can start now from the #898 K-validator data.* `[uses: Track B]`
- **Sequencing.** G0+G1+G4 now (CPU); G2-GPU + G3 after #891/#898 land and the GPU machine is available. Each implementation PR with a paper-worthy result drops its figure under `paper/figures/<name>/` (`make.py` + README) and registers it, so the figure set grows with the code.

### Dependency graph (summary)
```
PR0 ─▶ P1 ─┬─▶ P0-util ─┬─▶ Pjax/Ptorch-orbit, Pinbackend, Track F
           ├─▶ Pspecial ─▶ P2.5, Pdf.2
           ├─▶ P2.1 ─▶ {P2.2 ∥ P2.3 ∥ P2.4} ─▶ P2.6
           ├─▶ Paa-python ─▶ Paa-analytic ─▶ Paa-staeckel-c ─▶ Paa-grid-approx
           └─▶ Pci(seed) ─▶ Pdf.1
Pvar-struct ─▶ Pvar-pot.{1..5,7 ∥} ─▶ Pvar-pot.6 ;  +above ─▶ Pvar-rhs ─▶ Pvar-lyapunov
Pvar-struct + P2.* ─▶ Pparam-design ─▶ Pparam-pot.*
Pvar-rhs ─▶ Pjax/Ptorch-orbit ─▶ Track F(disk/stream) ─▶ Pci(final)
```
Pvar-struct has **no Track-0/A dependency** — start it day one alongside PR0.

---

## Parallelization (workflows + git worktrees)
- **Branches.** Track B (variational) + Lyapunov + Track C C-side → short-lived feature branches off `main`, merged to `main`. Track 0/A/D/E/F → long-lived `feat/backends` (off `main`, merges `main` in to consume the C work; final merge to `main`). Track B touches C files / `parse_leapFuncArgs_Full`; Track A touches Python compute methods — mostly disjoint, so the two streams rarely conflict.
- **Fan-outs run as orchestrated workflows**, one agent per family in its own worktree off the merged base: Track A `P2.*` (off P1 on `feat/backends`), Track B `Pvar-pot.*` (off **merged Pvar-struct** on its `main`-bound branch), Track C `Pparam-pot.*` (off merged Pparam-design), Track F DF families. Each edits its own file + its own contiguous `parse_leapFuncArgs_Full` `case` → adjacent but non-overlapping; merge sequentially per family; each PR appends its **own** test module.
- Worktree build: `GALPY_COMPILE_SINGLE_EXT=1 GALPY_COMPILE_NO_OPENMP=1 python setup.py build_ext --inplace`.

---

## Log & per-track plan files (`~/Repos/galpy-jax/`)
At execution start, create the index + a detailed spec/log file **per track** (the original ask to break the plan into multiple files):
```
galpy-jax/
  README.md                 # index + status table (PR → state → coverage delta)
  00_overview.md            # motivation + settled decisions
  plan/                     # detailed per-track specs
    track0_infra.md  trackA_potentials.md  trackB_variational.md
    trackC_params.md trackD_autodiff.md   trackE_actionangle.md  trackF_df.md
    trackG_figures.md       # reproducible paper figures/diagnostics (cross-cutting)
  paper/
    paper.md  references.bib
    figures/
      _harness/{figure.py, style.py, backends.py, bench.py, manifest.json}
      <figname>/{make.py, <figname>.pdf, data/, README.md}   # every figure = code + artifact + repro hash
  logs/<PR-or-topic>/{log.md (Approach/Dead-ends/Results/Plots), plots/, results.md}
  appendices/{B_variational_equations.md, D_special_functions.md, E_coverage_tolerance_ledger.md,
              F_adiabatic_action_quadrature.md}
```
Paper sections map 1:1 to tracks. Every `log.md` opens with Approach / Dead-ends / Results / Plots so dead-ends are preserved. **Every paper figure lives under `paper/figures/<name>/` with reproducible `make.py` (Track G); `verify-all` re-renders all figures against the final code and fails on mismatch.**

---

## Testing infrastructure (the tests are the spec)
Two pillars, **both required**:

### Pillar 1 — the *entire existing suite* passes under every backend, at identical tolerances
- `pytest --backend={numpy,jax,torch}` re-runs the **unchanged** existing tests (autouse fixture sets the default backend; jax x64 on). numpy run is byte-identical. jax/torch must pass the *same* tests at the *same* tolerances; loosen only if absolutely necessary, slightly, with a logged justification (tolerance ledger).
- New CI jobs run the full suite under jax and under torch (Track 0 / Pci).

### Pillar 2 — substantial NEW backend-specific tests (derivatives of many functions)
Not exhaustive, but significant coverage of jax/torch-specific behavior that the existing suite never exercises:
- **Potentials:** parametrized grad-vs-finite-difference over many potentials × methods for forces and all six second derivatives; the consistency identities `d(_evaluate)/dR == −_Rforce`, `d(−_Rforce)/dR == _R2deriv`, etc.; `jit`/`vmap` survival; device/dtype preservation; numpy↔jax↔torch value parity. torch path adds `torch.autograd.gradcheck`.
- **Orbit autodiff:** grad-vs-FD of final phase-space coords w.r.t. ICs *and* potential parameters; jax `jacrev == M == jacfwd`; torch `gradcheck`; **C-STM vs in-backend diffrax/torchdiffeq agreement** (independent mechanisms); higher-order (Hessian-of-orbit via the in-backend integrator); param-grad C vs in-backend AD.
- **Action-angle:** analytic-rule vs pure-Python-AD vs FD for Jr / frequencies / angles; **Staeckel custom_vjp vs FD/gradcheck**; backend value parity.
- **df / streams:** parity + grad-vs-FD on DF evaluations, moments, and stream-track outputs (w.r.t. inputs and potential parameters).
- **Special functions:** native-vs-fallback agreement + grad-vs-FD (Track 0).

### The 3D variational-equations test battery (Pvar-rhs / Pvar-lyapunov) — all of the design doc's tests, not just Liouville
1. **Liouville:** `det M(t) = 1` (6×6, integrate 6 canonical basis deviation vectors), the 3D analogue of `test_liouville_planar`.
2. **Symplecticity:** `Mᵀ Ω M = Ω` in **canonical Cartesian** variables (Cartesian is canonical; cylindrical is not) — strictly stronger than Liouville; most sensitive to the new z-coupling/cross terms.
3. **Closed-form STM (tightest ground truth):** 3D harmonic oscillator (analytic sines/cosines) and **Kepler** (Lagrange f, g coefficients) — catches sign errors Liouville can miss.
4. **Flow-direction freebie:** `M(t)·f(x₀) = f(x(t))` exactly (no reference solution needed).
5. **Conserved-integral relations:** axisymmetric `Lz` and energy give exact left-relations `gᵀ(x(t)) M(t) = gᵀ(x₀)` — localize errors to specific K rows/cols.
6. **Finite-difference of the flow, column-by-column:** `Mᵢ ≈ (x(t; x₀+εeᵢ) − x(t; x₀))/ε` — direct per-column STM check.
7. **2D-reduction bridge:** planar IC with `dz=dvz=0` reproduces the trusted planar `integrate_dxdv` STM to 1e-10.
8. **Lyapunov:** integrable→0 signature (power-law not exponential deviation growth: isochrone/harmonic/spherical); Hénon–Heiles and flattened-logarithmic regression vs literature values.

These run across the integrators that support dxdv (rk4_c, rk6_c, dopr54_c, dop853_c, dop853, odeint) and across axisymmetric (MiyamotoNagai, MWPotential2014), fully non-axisymmetric (triaxial Logarithmic, SpiralArms, an Ellipsoidal — to exercise `Rphideriv`/`zphideriv`), and closed-form (Kepler, harmonic) potentials.

**Dissipative / velocity-dependent forces are excluded from tests 1–2** (Liouville `det M=1` and symplecticity fail by construction for non-conservative `f(x,v)` — see Track B "Open consideration"); validate those via test 6 (finite-difference of the flow) and known damping behavior instead.

---

## Review checkpoints (you review every PR; natural pauses)
1. After **PR0** — resolver API + "dispatch in private layer".
2. After **P1 + Pspecial** — freeze `galpy.backend` API, config, special-router contract + special-fn accuracy.
3. After **P2.1 + P0-util** — namespace-swap hazard pattern + coords/conversion before the P2 fan-out.
4. After **Pvar-struct** — C ABI change (serialization point) before the big C fan-out.
5. After **Pvar-rhs (+ Pvar-lyapunov)** — 3D STM correctness before any autodiff trusts M.
6. After **Pjax/Ptorch-orbit + Pinbackend + Pparam** — full differentiable orbit stack.
7. After **Track E** — action-angle differentiability (incl. C-backed Staeckel).
8. Per **Track F** batch (df) — esp. before/after the streamdf rewrite.
9. Final: whole suite green under all backends at identical tolerances, coverage all-but-27, `feat/backends → main`.

---

## Key files
- New: `galpy/backend/{__init__,_resolver,_namespaces,linalg}.py`, `special/{__init__,_router}.py` + `special/_fallback/<fn>.py`, `_jax/orbit_stm.py`, `_torch/orbit_stm.py`, `_reference/inbackend_ode.py`.
- `galpy/util/{_optional_deps.py, config.py ([backend]), conversion.py (fac branch), coords.py (transforms), special.py (→ Pspecial)}`.
- Potentials: `SphericalPotential.py` (family consolidation), Plummer/Isochrone, EllipsoidalPotential (cache), … + the 33 C potential files + `potential_c_ext/{galpy_potentials.h,galpy_potentials.c}`.
- Orbit: `orbit_c_ext/integrateFullOrbit.c` (evalRectDeriv_dxdv, parse_leapFuncArgs_Full, init_potentialArgs), `integrateFullOrbit.py`, `Orbits.py` (integrate, integrate_dxdv, getOrbit_dxdv, Lyapunov).
- Action-angle: `actionAngleIsochrone/Harmonic/Vertical/Spherical.py`, `actionAngleStaeckel_c.py` + `actionAngle_c_ext/actionAngleStaeckel.c` (derivative integrands 1883-1976), `actionAngleAdiabatic*`, `*Grid`, `IsochroneApprox`. (`actionAngleTorus*` excluded — external library, untouched.)
- df: `df/{df,surfaceSigmaProfile,jeans,sphericaldf+variants,diskdf,evolveddiskdf,quasiisothermaldf,streamspraydf,streamgapdf,streamTrack,streamdf}.py`.
- Tests/CI: `tests/{conftest.py (--backend), test_backend_*.py, test_orbit.py (3D Liouville)}`, `.github/workflows/build.yml`, `setup.py (extras)`, `.coveragerc`.

---

## Immediate first step
Two independent starts:
- On **`feat/backends`** (off `main`): build **PR0** — resolver skeleton + Plummer/Isochrone migration + `tests/test_backend_pilot.py` (numpy/jax/torch parity at existing tolerances + grad-vs-FD + the `d(_evaluate)/dR == −_Rforce` identity).
- On **`feat/variational3d`** (off `main`, → `main`): start **Pvar-struct** (add the C Hessian pointers/aggregators, wire SpiralArms) — fully decoupled from the backends.

Initialize `~/Repos/galpy-jax/` with `00_overview.md`, the `plan/` per-track specs, and `logs/PR0_pilot/`. Both PR0 and Pvar-struct are early review checkpoints.
