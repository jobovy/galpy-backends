# Track B — C 3D variational equations (→ main) + Lyapunov (→ main)

Standalone galpy features (the backends consume them but don't depend on them).
Branches off `main`: `feat/variational3d`, then `feat/lyapunov`.

## The math (verified)
Cartesian accel: `ax=cosφ·FR − sinφ/R·Fφ`, `ay=sinφ·FR + cosφ/R·Fφ`, `az=Fz`
(FR=Rforce, Fφ=phitorque, Fz=zforce). The acceleration Jacobian
`K = ∂(ax,ay,az)/∂(x,y,z)` is the **symmetric Cartesian Hessian −∇∇Φ**.
- (x,y) 2×2 block = **byte-for-byte the proven 2D `evalPlanarRectDeriv_dxdv`**
  (integratePlanarOrbit.c:1079-1098), with the second derivs now evaluated by the
  3D aggregators `calcR2deriv/phi2deriv/Rphideriv(R,z,φ,t)`.
- New z-coupling:
  - `dFxdz = −cosφ·Rzderiv + sinφ/R·zphideriv`  (= `dFzdx`)
  - `dFydz = −sinφ·Rzderiv − cosφ/R·zphideriv`  (= `dFzdy`)
  - `dFzdz = −z2deriv`
  Symmetry of K (`dFxdy==dFydx`, `dFxdz==dFzdx`, `dFydz==dFzdy`) is a built-in test.
Sign conventions: `Rforce=−∂Φ/∂R`; the `*2deriv`/`*deriv` methods return derivs of Φ.
Python `phizderiv` = C `zphideriv` = ∂²Φ/∂z∂φ.

12D state `(x,y,z,vx,vy,vz | dx,dy,dz,dvx,dvy,dvz)`: `d(pos)=vel`, `d(vel)=accel`,
`d(dpos)=dvel`, `d(dvel)=K·(dx,dy,dz)`. (The current stub gets the indices wrong.)

## PRs
- **Pvar-struct** (serialization point): add `z2deriv/Rzderiv/zphideriv` pointers to
  `struct potentialArg` (galpy_potentials.h ~29-40); add `calcz2deriv/calcRzderiv/
  calczphideriv` aggregators (galpy_potentials.c, model on `calcR2deriv`, skip NULL);
  NULL-init the new pointers in `init_potentialArgs` + every `parse_leapFuncArgs_Full`
  callsite; wire SpiralArms (the only complete 3D-Hessian C potential). RHS unchanged
  → suite unaffected.
- **Pvar-pot.x** (fan-out, worktrees off merged Pvar-struct): implement per-potential
  3D Hessian in C — port analytic Python where it exists (MiyamotoNagai 108-140 etc.),
  derive where it doesn't. Families: disk/bar analytic, spherical (r2deriv+transform),
  ellipsoidal (psi/mdens+transform), SCF/Multipole (harmonics), bars/spirals (zphideriv),
  wrappers (compose children, last), interpRZ (spline). Each: C-Hessian-vs-Python check.
- **Pvar-rhs** (keystone): fix `evalRectDeriv_dxdv` (integrateFullOrbit.c:1195) per the
  math above; lift `integrate_dxdv` to `phasedim==6` (6D cyl↔rect deviation transform;
  revive `integrateFullOrbit_dxdv_c`); batched C convenience returning `(nt,6,6) M`.
- **Pvar-lyapunov** (separate, `feat/lyapunov`): Lyapunov exponents (largest-eigenvalue
  STM growth / Benettin renormalization) on `Orbit`.

## Open consideration — dissipative / velocity-dependent forces (dynamical friction)
`ChandrasekharDynamicalFrictionForce`, `FDMDynamicalFrictionForce` (base
`DissipativeForce`; C `requiresVelocity` + `*Velocity` force fns) are NOT from a
potential and depend on velocity: `f=f(x,v)`. So the variational Jacobian is the
general `J=[[0,I],[∂F/∂x, ∂F/∂v]]` (lower-left non-symmetric, plus a nonzero
velocity block). `Ṁ=JM` still holds (conservative case = `∂F/∂v=0`, `∂F/∂x=K`).
Non-conservative ⇒ `det M=exp(∫tr ∂F/∂v dt)≠1` and `MᵀΩM≠Ω`, so dissipative
potentials are **excluded from the Liouville/symplecticity tests** (use FD-of-flow).
The STM still gives correct IC gradients. **IN SCOPE — route 2 (user-confirmed):**
the C-STM RHS (Pvar-rhs) is written general from the start to carry the
velocity-coupled, non-symmetric `J` (conservative = special case `∂F/∂v=0`), and a
sub-track implements `∂F/∂(x,v)` in C per dissipative force:
- **Pvar-dissip-struct:** velocity-dependent Jacobian pointers (`∂F/∂(R,z,φ)`,
  `∂F/∂(vR,vT,vz)`) + NULL-safe aggregators, mirroring Pvar-struct.
- **Pvar-dissip-force.x:** `∂F/∂(x,v)` for Chandrasekhar/FDM, validated vs FD-of-force
  and the Track D in-backend AD reference.
Track D in-backend (diffrax/torchdiffeq) gives the same derivatives via AD of the
namespace-swapped `f(x,v)` (Track A P2.7) — kept as the independent oracle.

## Test battery (Pvar-rhs / Pvar-lyapunov) — all of design-doc §4
1. Liouville `det M=1` (6×6) — 3D analogue of `test_liouville_planar`.
2. Symplecticity `MᵀΩM=Ω` in **canonical Cartesian** (Cartesian is canonical).
3. Closed-form STM: 3D harmonic + Kepler (Lagrange f,g).
4. Flow-direction `M·f(x₀)=f(x(t))`.
5. Conserved-integral: Lz / energy left-relations `gᵀ(x(t))M=gᵀ(x₀)`.
6. Column-wise FD of the flow.
7. 2D-reduction bridge (planar IC matches trusted planar STM to 1e-10).
8. Lyapunov: integrable→0 signature; Hénon–Heiles / flattened-log regression vs literature.
Run across rk4_c/rk6_c/dopr54_c/dop853_c/dop853/odeint and axisymmetric +
non-axisymmetric (triaxial Log, SpiralArms, Ellipsoidal) + closed-form potentials.

## Pvar-pot.6 wrappers (PR #919, → feat/variational3d)
3D C Hessian for the four "clean" wrappers whose Hessian = modulation ×
calc<deriv>(...wrapped...): DehnenSmooth, GaussianAmplitude, TimeDependentAmplitude,
SolidBodyRotation (SolidBody adds an additive φ-shift, invisible to its R/z
2nd-derivs). hasC_dxdv3d = _check_c(self._pot, dxdv3d=True). Also: WrapperPotential
now delegates `_phizderiv` to the wrapped potential (was missing) so the pure-Python
reference works for non-axi children. C-vs-Python dxdv ≈1e-17. Deferred hard wrappers:
Corotating (R-dep φ-shift→chain rule), RotateAndTilt (3×3 Hessian rotation),
coord-transform wrappers (KuzminLike/OblateStaeckel/CylindricallySeparable).

## Pvar-pot SCF/Multipole harmonic Hessian (feat/pvar-pot-scf)
SCF (and Multipole) store Φ = Σ A_nlm phiTilde_nl(r) P_l^m(cosθ) [cos/sin](mφ); the
Python side uses NumericalPotentialDerivativesMixin (no analytic 3D Hessian to port),
so the C 3D Hessian is DERIVED:
- Sum the full set of spherical-coordinate potential derivatives Φ_r, Φ_θ, Φ_rr,
  Φ_θθ, Φ_rθ, Φ_φφ, Φ_rφ, Φ_θφ (φ-derivs from m·[cos/sin], θ-derivs from the
  Legendre functions).
- Angular θ-derivatives from the existing x=cosθ arrays via exact identities
    dP/dθ   = -sinθ · dP/dx
    d²P/dθ² = cosθ · dP/dx − l(l+1)P + m²/sin²θ · P   (from the assoc.-Legendre ODE)
  The m²/sin²θ term is a genuine z-axis (R=0) coordinate singularity for m>0 (forces
  stay regular there); orbits never hit the axis exactly. m=0 (incl. axisymmetric)
  is regular everywhere.
- Spherical→cylindrical chain rule (R=r sinθ, z=r cosθ): R2/z2/Rz from Φ_rr,Φ_rθ,
  Φ_θθ + gradient×(second partials of r,θ); Rphi/zphi from Φ_rφ,Φ_θφ; phi2=Φ_φφ.
  Verified: planar (z=0) reduction recovers the existing PlanarR2deriv=Φ_rr etc.
- Cache widened to 6 value slots (one compute serves all 6 cyl components); nargs
  +7→+10 in both orbit parsers.

### Disk composites (DiskSCF/DiskMultipole/KuijkenDubinski) — same PR #920
Analytic [Sigma_i(r), Hz_i(z)] disk-approximation pairs (C type 26) get their 3D
Hessian: for Phi=Sigma(r)Hz(Z), R2/z2/Rz from Sigma',Sigma'',Hz,Hz' plus the
identity Hz''(z)=hz(z) (1D Poisson; reuse the existing vertical-density function).
phi-derivs identically zero (NULL, skipped by aggregators). hasC_dxdv3d set in
KuijkenDubinski._finish_init = (disk pairs C-capable) AND _me (SCF/Multipole) has
its 3D Hessian. Validated: DiskSCF det/symp ~1e-11/1e-10 FDflow ~1e-5; DiskMultipole
det/symp ~1e-8 FDflow ~6e-6 (smooth sech2 vertical profile; exp |z| is C^0 at z=0
like KuzminDisk -> FDflow at the kink, det/symp still pass).

### Harmonic+disk family status: SCF, MultipoleExpansion, DiskSCF, DiskMultipole,
### KuijkenDubinski all have full 3D C Hessian (PR #920 -> feat/variational3d).

## 2026-06-09 — SCF planar sign bug, FD-of-flow doctrine, DiskSCF planar (#923, #925)
- Adding a C-vs-analytic-Python consistency test to #921 exposed that
  SCFPotential.c's sum_spher_2nd_derivs returned the PLANAR 2nd derivs
  sign-flipped (force `-=` convention copied; R2deriv must be +d2Phi). The C
  planar variational STM was wrong for SCF since the March Multipole-C commit;
  test_liouville_planar passed throughout because a sign-flipped-but-symmetric
  Hessian is the Hessian of -Phi (still Hamiltonian, det M=1 holds).
  **Doctrine (now a standing project rule + tests): det(M)=1/symplecticity are
  structural, not absolute — every variational Hessian must be pinned by
  finite-difference-of-the-flow.** Fix + test_harmonic_planar_stm_vs_fd_of_flow
  = PR #923 (merged to main). Post-fix: C STM matches FD-of-flow to ~1e-6.
- #920's 3D path (sum_spher_full) was independently verified correct (FD-of-flow
  ~7.5e-7 axi+non-axi); planar/3D now consistent at z=0 (diff exactly 0).
- DiskSCF-family planar 2nd derivs in C (#925 → main): H(0)=0 identically for
  both vertical profiles so the correction's planar Hessian is exactly zero;
  wired defensively + hasC_dxdv propagation; FD-of-flow <1e-5; DiskSCF +
  DiskMultipole removed from the liouville exclusion list.

## 2026-06-09 — Pvar-lyapunov (#926 → feat/variational3d)
Orbit.lyapunov: Benettin renormalization on integrate_dxdv (planar+3D, C, multi-
orbit, frequency units). Validation: Hénon–Heiles chaotic orbit F (Skokos+02)
λ→0.044–0.051 vs literature 0.04–0.08 (Benettin+76); regular orbit λ≈ln(t)/t,
chaotic/regular ratio ~300; renormalized-vs-single-shot 2e-7; C-vs-Python <2%;
121-test dxdv battery green. + doc notebook tutorials/orbits/
phase_space_volumes_and_chaos.ipynb (2D/3D dxdv + Liouville + Lyapunov).
**Bug found by the notebook: CompositePotential/baseCompositePotential did not
aggregate hasC_dxdv3d → MWPotential2014 (a CompositePotential) silently fell
back to Python odeint for ALL 3D variational work (21s + NaNs vs 1s C).** Fixed
+ regression test in #926 (duplicated independently in #933; dedupe at rebase).

## 2026-06-09 — Pvar-dissip-struct (#933 → feat/variational3d)
Velocity-dependent variational equations in C: RectForceJacobian(t,q,jac_x,jac_v)
pointer in potentialArg (rectangular 3x3 dF/dx, dF/dv; NULL-safe everywhere);
evalRectDeriv_dxdv generalized to d(δv)/dt=(K+Σ∂F/∂x)δx+(Σ∂F/∂v)δv with
calcRforce/phitorqueConservative keeping friction out of the K-assembly;
conservative path verified BIT-IDENTICAL (pre/post builds, 3 pots × 3 integrators).
Chandrasekhar Jacobians fully analytic (F=A(x,|v|)v ⇒ closed forms incl.
dlnΛ/dv and clamped-spline σ'); single documented FD term: ∂ρ/∂x via central FD
of calcDensity (no density-gradient C interface). FDMDynamicalFriction explicitly
gated hasC_dxdv3d=False (would inherit Chandrasekhar's True with wrong Jacobians).
Python dxdv methods raise NotImplementedError for dissipative (Python EOM_dxdv is
conservative-only). Validation: Jacobian-vs-FD ≤2.1e-8; FD-of-flow ≤2.8e-5;
**phase-volume law det M(t)=exp(∫tr(∂F/∂v)dt') to 3.1e-6 rel, det M=0.518<1**
(the non-conservative replacement for Liouville); conservative battery 121/121.

## 2026-06-09 — remaining-tail fan-out COMPLETE (8 items → PRs #934–#941)
Audit: 38 potentials have the full 3D C Hessian. Remaining, launched as a
workflow fan-out (each with FD-of-flow doctrine + adversarial verifier):
fdm-jacobians + noninertial-frame (off #933; NonInertial physics test:
tr(∂F/∂v)=0 ⇒ det M=1 despite velocity dependence), rotateandtilt-conjugation,
corotating R-dependent-pattern-speed chain rule, kuzminlike z-substitution,
movingobject (planar+3D), interpRZ spline 2nd derivs, softenedneedlebar
(no 2nd derivs at all today; sympy-assisted derivation). Planar dissipative dxdv: confirmed OUT OF SCOPE by the
maintainer (2026-06-09) — no concrete planar dissipative force exists in galpy
to exercise it.

Tail fan-out results (16 agents: 8 impl + 8 adversarial verify; all complete,
all verified):
- **#934 FDM Jacobians**: fully analytic across ALL FDM regimes (zero-velocity
  Cin series, dispersion, mu-interpolated, classical cutoff×3 lnΛ, const);
  Jac-vs-FD 1.55e-8 (regime-asserted tests); FD-of-flow 2.45e-5 (suppression
  asserted active); phase-volume law 1.59e-6, det M=0.677.
- **#935 NonInertialFrameForce**: exact closed forms (−2[Ω]ₓ etc., all Omega
  configs); KEY: tr(∂F/∂v)=0 ⇒ det M(t)=1 despite velocity dependence — the
  volume-preserving velocity-block test friction can't provide.
- **#936 RotateAndTilt**: Hessian conjugation RᵀHR + offset; numpy byte-identical;
  2D-bridge tests skip by design (no invariant z=0 plane). Pre-existing
  force-cache-key-missing-t noted (separate issue).
- **#937 Corotating**: R-dependent pattern-speed chain rule. **Found+fixed a real
  Python bug: Rzderiv missed the −s_R·Φ_zφ′ cross term (13–99% wrong, β≠1,
  t≠to; nothing consumed it)**. C-vs-Python 5.1e-12 post-fix.
- **#938 KuzminLike**: z-substitution chain rule; b=0 kink documented
  (pre-existing); mutation-tested (wrong Hessian fails at O(1)).
- **#939 MovingObject**: planar+3D (was: ERROR on any dxdv attempt); kernel
  Hessian at shifted point; C-vs-Python track interp 6e-11.
- **#940 interpRZ**: spline 2nd derivs, interp-limited ~2-3e-5; R2deriv/z2deriv
  error→value behavior change flagged.
- **#941 SoftenedNeedleBar**: Hessian derived from scratch (sympy-verified,
  hand-written C), planar+3D+Python analytic; bare default stays excluded from
  liouville_planar — its fixed IC is strongly chaotic (‖STM‖~6e4, det error
  saturates ~0.1 from cancellation); covered via a non-chaotic config instead.
With these, the variational track is feature-complete: every C-capable
potential/force has dxdv support except by-design exclusions (planar
dissipative: no such force exists — maintainer-confirmed out of scope).

## 2026-06-10 — z-aliasing fix cascade (#944) + Lyapunov spectrum (#945)
- **#944 (merged)**: integrateFullOrbit_dxdv returned the base orbit with z
  overwritten by vT (rect_to_cyl returns Z by reference; in-place column
  assignment clobbered it). Found by the Track G figure measurements; invisible
  to ALL deviation-based validation (rectOut columns correct) and even to the
  lyapunov integrable test (a corrupted restart is just a different regular
  orbit). Mutation-verified regression test added.
- Cascade: #926/#933/#934–#941 rebased onto the fix; doc-notebook + #926
  examples refreshed — MW 3D λ: corrupted 1.18→1.06 /Gyr became correct
  0.38→0.21 /Gyr (HH planar values unchanged, planar path never had the bug).
- **#945**: Orbit.lyapunov(spectrum=True) — full Benettin/QR spectrum.
  HH chaotic orbit F: [0.0519, 0.0004, −0.0002, −0.0505] (λ1 in lit band,
  flow-direction pair ≈0, pairing 2.7%, sum rule 3.1%); Isochrone 3D: all six
  within ±ln(t)/t. 11/11 lyapunov tests on the fixed base.

## 2026-06-10 — #933 review rework (+ #934/#935 cascade)
Per maintainer review: RectForceJacobian → **RectDissipativeForceJacobian**
(galpy DissipativeForce nomenclature; NULL for conservative); the exported
ctypes evaluators REMOVED (galpy convention: C tested only through regular
orbit use) — per-entry Jacobian-vs-FD unit tests replaced by orbit-level
validation, with the phase-volume trace now from **central FD of the Python
forces** (independent cross-validation of the C STM; law still ≤1.6e-6);
evalRectDeriv_dxdv simplified to the NULL-safe-aggregator straight-line
pattern (conservative path re-verified **18/18 byte-identical** vs base);
stale force enumerations stripped from warnings. Batteries: 126/128/130.
g11 measure script updated to the Python-FD trace (committed data unchanged).

## 2026-06-11 — #933 round-2 review + coverage closure
- calcRforce/calczforce/calcphitorque generalized with a final
  include_dissipative int (macros supply 1 → all existing call sites
  unchanged; the variational RHS calls with 0); the Conservative duplicate
  aggregators deleted. Conservative path hash-verified bit-identical vs base.
- The 11 uncovered Chandrasekhar-Jacobian C lines closed via ORBIT-LEVEL
  tests only (maintainer convention; the ctypes evaluators are gone):
  parametrized FD-of-flow over const-lnΛ / rhm-Coulomb-log / minr-gate /
  σ-clamp configs, each with a non-vacuity branch guard, mutation-validated
  1:1. Physics note: the minr-gate test keeps the orbit entirely inside minr —
  a crossing orbit's flow has a saltation term at the force discontinuity that
  the zero Jacobian correctly omits (FD-of-flow ill-posed across the
  discontinuity; measured 0.68 mismatch for a crossing orbit).
- #945 spectrum consistency test made platform-robust: tight check now seeds
  the spectrum with the SAME dxdv0 as the single-vector run (|R_11|
  accumulation ≡ single-vector Benettin; agreement 1.4e-8) instead of
  comparing decorrelated chaotic finite-time estimates at 5%.

## 2026-06-11 — #934 (FDM) + #935 (NonInertial) retarget + full C-branch coverage
- Post-#933-squash retarget: both branches rebased
  `--onto origin/feat/variational3d 2e234049` (squash tree-identical to the
  old tip, so the replays were clean); phantom CONFLICTING → MERGEABLE.
- #934: all 39 uncovered FDM-Jacobian C lines closed via 7 orbit-level
  FD-of-flow configs (template = the Chandrasekhar branch battery):
  minr-gate / σ-spline-clamp / rhm-lnΛ+cdm-cutoff / const-lnΛ+cdm-cutoff /
  zero-velocity-Cin / μ-intermediate / const-FDMfactor. Non-vacuity via the
  `_fdm_c_regime` Python replication of the C regime logic (asserted regime
  set per orbit). gcov (CI-exact -O0 --coverage) 39/39 hit; mutations 8/8
  detected. Design note: the lnΛ-branch configs pair with the classical
  cdm-cutoff regime so the lnΛ *derivatives* are consumed (mutation-
  detectable), not dead.
- #935: the 10 uncovered NonInertial-Jacobian lines = 3 branches —
  translation-only early exit (J≡0), scalar Ω_z(t)-as-function
  (tdep/tdep_deriv spline path), constant-vector Ω+Ω̇ componentwise.
  3 new configs, gcov 10/10, mutations 9/9 (incl. a garbage-entry injection
  before the early return). Quirk: vector Omega must be a numpy array
  (parse_frequency rejects lists) — existing convention, not a bug.

## 2026-06-11 (later) — full tail retarget after #934 merge
- #935 re-conflicted post-#934 (both PRs append batteries at the same
  test_orbit.py anchor); resolved by reconstructing base + both blocks
  (noninertial battery placed after the FDM one), coverage commit replayed
  clean, diff purely additive, 21/21 dissipative-dxdv tests green → pushed
  4d699d3e, MERGEABLE.
- The six Pvar-pot tail PRs #936–#941 predated #926/#933/#934 entirely
  (merge-base 8f50486f) → 6-agent workflow fan-out, one per worktree, rebased
  all onto 198f7c07. 3 clean, 3 with conflicts (all purely-additive
  test_orbit.py interleaves at shared anchors + one duplicate
  CompositePotential hasC_dxdv3d hunk in movingobj dropped as already-merged).
  Verified per branch: base's rewritten evalRectDeriv_dxdv untouched (0
  touching lines in every diff; only per-potential wiring hunks), build OK,
  own battery + dissipative regression battery (14 tests) green under the
  worktree-import guard. All six pushed; notebooks cancelled.
- Tips: 936=0ae41d2c 937=d0c2caeb 938=86c5b059 939=4febaf48 940=b75d8354
  941=3c69d0db.

## 2026-06-11 (later still) — tail landings + final retarget pass
- Merged today in order: #934 (FDM), #935 (NonInertial), #936 (RotateAndTilt),
  #940 (interpRZ), #941 (SoftenedNeedleBar). #940/#941 needed no re-rebase
  (merge-tree-simulated clean in sequence); #941's windows streamgapdf job was
  a runner-setup infra flake (194-line log), rerun → green.
- User found a committed conflict marker in HISTORY.txt — full-branch sweep
  showed exactly ONE stray line (line 17, ">>>>>>> 0e732bc4"), introduced by
  the FDM stacking rebase and squashed into feat/variational3d via #934;
  AUTHORS.txt "=======" hits are RST underlines. Root cause: repo lacks the
  check-merge-conflict pre-commit hook AND diff-vs-base verification can't see
  inherited markers. Fix rides on #937 (marker deleted + hook added,
  9ab4f0b2). Lesson memorized: full-tree marker grep before every rebase push
  (now mandatory in fan-out agent prompts).
- Final 3-agent pass: #937/#938/#939 rebased onto dc291335 (conftest registry
  + test-anchor unions; movingobj keeps _planar_invariant adjacent to its
  consumer). Each: build OK, own battery + 21-test dissipative regression
  green, marker sweep clean (sole hit = the base's line that #937 removes).
  merge-tree chain sim: 937 clean; 938-after-937 and 939-after-938 WILL
  re-conflict (shared anchors) → one mechanical re-rebase per landing.
- Tips: 937=9ab4f0b2 938=d94a2289 939=034ef885.

## 2026-06-11 (evening) — Pvar-pot wrapper family COMPLETE: the two deferred wrappers
- User opted to finish OblateStaeckelWrapper + CylindricallySeparableWrapper
  ("otherwise we never will") — the last two #919 deferrals; KuzminLike (#938)
  was the template. Branch feat/pvar-tail-staeckelwrappers (df4ba616, worktree
  /var/tmp/pvt-staeckelwrap, off dc291335). Both also gain their first PLANAR
  hasC_dxdv (they had hasC_dxdv=False even pre-variational-track).
- CylSep trivial by separability: Phi_RR=Phi_w,RR(R,0), Phi_zz=Phi_w,zz(Rp,z),
  Phi_Rz≡0 (NULL in the Full parser; ZeroPlanarForce in the planar parser
  since calcPlanar* aggregators are not NULL-safe). OblateStaeckel: full
  (u,v) chain rule, C transcribed 1:1 from the Python ground truth (same
  intermediates: umvfac, d2Udu2/d2Vdv2, prefactor-derivative helpers); the
  planar functions use the exact v=pi/2 simplification (V(pi/2)=0,
  dVdv(pi/2)=0) since the planar parser only wires wrapped planar functions.
  amp subtlety: trailing 2*force/prefac^2 terms reuse the wrapper's own C
  forces (amp included) so only the leading bracket multiplies amp.
- Note: both wrappers drop t by design (Python _R2deriv calls
  evaluateR2derivs(...,t=0); existing C forces likewise) — checked against
  the KD-t-forwarding lesson, NOT a bug here (static-by-construction).
- Validation: parity planar+3D ~3e-11; physics-law cross-check — wrapper on
  exact-Staeckel KuzminKutuzov (delta matched) reconstructs KK's independent
  C Hessian to 2e-11 for any u0; FD-of-flow/det(M)/symplecticity/bridge via
  registry; gcov 100% new lines (CI-exact build); 13 mutations killed (11
  impl + 2 fresh from the coverage verifier). One term unfalsifiable by
  construction: d2Udu2's z0-suppressed Rzderiv term (z0~1e-17 at v0=pi/2) —
  kept for arithmetic fidelity to Python.
- 3 adversarial verifiers (independent-derivation math refuter, coverage,
  regression incl. actionAngle consumers): 0 blockers; minors = hardening
  suggestions (parity tests could assert absence of the fallback warning)
  + scratch debris (removed).
- Holding push until #937/#938/#939 land (HISTORY-marker deletion dedupes
  with #937's at rebase; conftest/test anchors will conflict mechanically).
