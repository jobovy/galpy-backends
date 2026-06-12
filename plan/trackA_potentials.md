# Track A — Potentials namespace-swap

Branch: `feat/backends` (fan-out worktrees off P1). Detail in master plan §"Track A".

Per-family PRs migrate the private `_evaluate/_Rforce/.../_phizderiv` (+ spherical
`_revaluate/_rforce/_r2deriv`) to `xp = get_namespace(R,z,phi,t)` + `numpy.`→`xp.`,
`numpy.pi`→`math.pi`, fixing hazards and keeping numpy bit-identical:
- in-place `out[idx]=val` → `xp.where(mask, special, expr)` with **safe denominators**
  (guard `0/0` so NaN from the dead branch doesn't poison reverse-mode gradients).
- `isinstance(R,float)`/value branching → shape-based (`getattr(R,'ndim',0)`).
- mutable caches (EllipsoidalPotential hash) → functional `_compute_forces`.

Batches: **P2.1 spherical** (migrate `SphericalPotential.py` first — covers 6+),
P2.2 disk/bar analytic ∥ P2.3 ellipsoidal ∥ P2.4 bars/heavy-mutation,
P2.5 special-backed (needs Pspecial), P2.6 wrappers (last, compose children).
Each PR adds family parity + grad-vs-FD tests (its own test module).

Pilot already migrated Plummer + Isochrone (PR0) — they fold into P2.1's family.

**P2.7 Dissipative / velocity-dependent forces** (DissipativeForce/planarDissipativeForce,
ChandrasekharDynamicalFrictionForce, FDMDynamicalFrictionForce): `f=f(x,v)` — thread
velocity through the namespace-swap. Enables dissipative-orbit autodiff via the Track D
in-backend integrator (see trackB_variational.md "Open consideration").

## 2026-06-09 — P2.5 special-function-backed fan-out complete (#927–#932)
Six-group workflow fan-out (one agent per group + independent adversarial
verifier doing numpy byte-parity vs a pristine checkout + test re-runs). All
migrated, all verified numpy-byte-identical:
- #927 Einasto: gamma/gammaincc via the router (native everywhere); r=0 core
  branch xp.where-guarded. Flagged deltas (values identical): scalar _revaluate
  returns 0-d ndarray (#922 pattern); old r=0 RuntimeWarning no longer emitted.
- #928 DoubleExpDisk: **no Bessel router needed** — j0/j1 live in the init-time
  Ogata weights (numpy, byte-identical); per-call path is elementary ops.
  _de_quadsum replicates numpy.nansum bitwise while zeroing NaN weights in the
  dead branch (AD survives). Flagged: R==0,z≠0 NaN bit pattern 0xfff8→0x7ff8.
  **Param-AD verified exact: quadrature nodes/weights carry zero parameter
  dependence; jax.grad vs FD-of-fully-reconstructed-potential 4e-11 (hr), 1e-12
  (hz).**
- #929 King: real work was interpSphericalPotential's spline evaluation →
  backend-evaluable power-basis PPoly (searchsorted+Horner); King init stays
  numpy by design.
- #930 SCF: full namespace-swap incl. analytic 2nd derivs; needed a #921
  cherry-pick (dropped after the main→feat/backends sync); trace-hostile
  per-point caches bypassed for non-numpy.
- #931 Multipole: static radial splines → PPoly coefficient stacks evaluated
  under xp; time-dependent (tgrid) explicitly numpy-only (NotImplementedError
  seam). Follow-up queued: TD support = stack a time axis + reuse #929's
  machinery for the CubicSpline-in-t.
- #932 DiskSCF/KuijkenDubinski/DiskMultipole correction layer (+logsumexp
  router); merge after #930/#931 for end-to-end backend eval.
Coverage: #931 had 4 uncovered axisymmetric phi=None guards → covered by test
(not excluded). Branch hygiene: main→feat/backends synced (temporarily a merge
commit; **maintainer rule: rebase-only on long-lived branches** — executed 2026-06-10
after the P2.5 merges: 18 commits replayed cleanly onto main (incl. #925),
merge commit dropped, history linear at d3cccc03; 185 backend tests pass).

## 2026-06-09 — also landed/learned
- galpy-vs-galax GPU: gap fully decomposed (tolerance 1.5x + batched-stepping
  2.9x; framework 1.0x) — see logs/gpu_speed_galpy_vs_galax/. galpy keeps
  per-orbit stepping (per-orbit error guarantees, batch-independent results).
- Backend dispatch (get_namespace) is jitted away (trace-time, zero runtime);
  XLA HloCSE merges duplicate identical igamma expansions (verified micro +
  real path + jax/XLA source) — no special-function call-site optimization
  needed in the spherical force layer.

## 2026-06-10 — P2.5b: time-dependent Multipole backend (#942)
TD (tgrid=) MultipoleExpansion now evaluates under jax/torch: the CubicSpline-
in-t over power-basis radial PPoly stacks re-packed into stacked tables, double
clamped-searchsorted gather + dt-Horner in numpy's exact op order → unchanged
static radial Horner; jit-traceable. Parity 2e-16–8e-16 (11 methods × on/
between/outside-tgrid × axi/non-axi × jax+torch); numpy byte-identical (2235-
line grid vs pristine, sha256-equal); 100% patch coverage; adversarial verifier
confirmed with its own independent TD potential (1.7e-15) incl. a d/dt gradient
check. **Surfaced pre-existing issue (decision needed): the KuijkenDubinski
layer does not forward t to the inner _me calls — TD DiskMultipole evaluates
the embedded expansion at t=0; honoring t is a numpy-path value change.**

## 2026-06-11 — P2.6 wrappers COMPLETE (#946): Track A potentials done
- All wrappers backend-agnostic: DehnenSmooth (dual-path t: numpy-scalar keeps
  original branching, traced t branch-free xp.where — DehnenBar precedent),
  GaussianAmplitude, RotateAndTilt (anchored rot/inv_rot/offset constants,
  input-guarded R==inf), KuzminLike, OblateStaeckel (U/V machinery), CylSep,
  TimeDependentAmplitude; SolidBody/Corotating already clean (tested only);
  AdiabaticContraction = interpSphericalPotential at runtime (inherits #929);
  WrapperPotential delegation audited clean.
- 5 coords transforms P0-util missed migrated (cyl_to_rect, rect_to_cyl,
  Rz_to_coshucosv, Rz_to_uv, uv_to_Rz) + _promote_scalars_for (torch scalar
  promotion anchored on array dtype/device).
- Byte-parity: 18,822 values bitwise vs pristine — ZERO deltas. Behavioral:
  coords list-input TypeError (P0 precedent), 0-d intermediates (outputs
  bit-equal), no new inf RuntimeWarnings. 474 new tests; backend suite 2069.
- Adversarial verify (independent byte-parity, AD stress + where-audit,
  conventions): 0 blockers. Pre-existing surfaced: WrapperPotential repr
  drops leading char; RotateAndTilt(raw list) fails; SpiralArms f32→f64.
- With #946, EVERY galpy potential (incl. all wrappers) is backend-agnostic.

## 2026-06-11 (late) — wrapper-oddities sweep: 2 main PRs + SpiralArms dtype fix
- #947 (main): list-of-Potential input crashed ALL 9 wrappers (factory ones at
  _dim in __new__, direct subclasses at first raw-_evaluatePotentials call);
  fixed via the canonical _check_potential_list_and_deprecate at the shared
  altitude (one __new__ + two __init__ sites), one DeprecationWarning, tests
  for all 9 + planar, fail-on-unpatched proven.
- #948 (main): wrapper repr ate the first character for the DIRECT 3D-only
  subclasses (RotateAndTilt v1.7.0+, KuzminLike v1.10.0+) — the factory
  underscore mechanism is INTACT (both dims); regression from PR #804's
  unconditional __name__[1:] (shipped v1.11.2); _repr_utils in the same PR was
  already conditional. Fix: removeprefix("_") both sites + factory×dim/direct/
  wrapper-of-wrapper/pickle test battery.
- feat/backends 362d9237: SpiralArms torch promotion root-caused — NOT numpy
  scalars (weak on torch 2.12); stored float64 numpy ARRAYS via bare
  xp.asarray (strong tensors). Same line fixes the 4e-8 float64 _dens bug:
  int64 Cs=[1] tensor × python-float rho0 computed in torch DEFAULT dtype
  (f32 round-trip of 1/4pi). dtype=getattr(R,'dtype',None) anchor; numpy
  44-entry byte-parity proof; _dens parity 4e-8 → 2e-16; f32 preserved.
- Systemic census (diagnosed, NOT yet fixed; user decision pending): same
  unanchored-asarray f32→f64 in SCF, DoubleExpDisk (Ogata), interpSpherical/
  King (spline), MultipoleExpansion (deliberately f64 per code comment),
  DiskSCF (via _me). SEPARATE likely-real bugs: TwoPowerSpherical generic
  alpha/beta calls scipy.special directly (bypasses router → torch autograd
  DETACHED); PerfectEllipsoid/TriaxialNFW all-tensor TypeError (unverified).

## 2026-06-12 — chase (b): both confirmed real, fixed on feat/backends
- b15fb8c3 TwoPowerSpherical generic (alpha,beta): direct scipy.special calls
  bypassed the router — torch no-grad SILENTLY degrades (numpy round-trip, no
  graph; loud error only with requires_grad), jax silently returns ndarray
  eager + TracerArrayConversionError under jit/grad. Test gap: CASES entry was
  ad_ok=False "scipy-deferred". Fixed via the Einasto router convention
  (fallback hyp2f1 <1e-12 vs scipy on the galpy regime); CASES upgraded to the
  full matrix; byte-parity pinned; grads match FD ~1e-10. Same idiom reported
  (not yet fixed): PowerSpherical._surfdens, TwoPowerTriaxial._psi.
- 7933ef06 EllipsoidalPotential: claim verified, mechanism CORRECTED — the
  default Perfect/TriaxialNFW are AXISYMMETRIC (b=c=1) and every method does
  `if not isNonAxi: phi = 0.0` (python float) before xp.cos(phi) → torch
  TypeError even all-tensor; ALSO mixed tensor-R,z + float-phi fails for ALL
  ellipsoidals. P2.3 missed both (only b=0.9,c=0.7 instances, only all-array
  inputs). Fix: _anchor_phi at 11 sites (numpy passthrough; 528-record
  byte-parity grid identical). 105/105 reproduction matrix post-fix.
- SYSTEMIC follow-up surfaced: mixed tensor-coords + python-float phi/t
  TypeErrors are PROJECT-WIDE under torch (SoftenedNeedleBar, LogHalo fail
  identically) — the backend suites always pass all-arrays. Needs a shared
  mixed-input gate + sweep (user decision pending).

## 2026-06-12 — f32 exit-cast policy landed as #949; #946 CI fixed
- #949 (→ feat/backends, c94d93f2): match_input_dtype exit cast for the
  table-backed five (SCF/mixin, DoubleExp, interpSpherical→King,
  Multipole→DiskSCF/DiskMultipole) — f64 interiors kept, f32 in → f32 out.
  The new 46-entry dtype gate (zero skips) caught a FIFTH instance while
  being built: the special-router fallbacks' f64 GL/trapezoid node tables
  (promoted f32 through the freshly-routed TwoPower) — same cast in
  _dispatch. Byte proofs: 92-record impl + 324-record independent verifier
  (incl. Quantity paths). Approved delta: numpy all-f32 inputs now return
  f32. Residual: ≤ tens of f32-ulp input-stage rounding in chain-rule
  prefactors (documented; pins at rtol=2e-6).
- #946 CI failure root-caused: actionAngleStaeckel public indivdelta API
  feeds plain LISTS to the migrated uv_to_Rz — the "galpy never passes lists
  internally" audit claim was wrong for user-supplied array-likes. Fixed at
  the AA boundary (_coerce_delta_arraylike at 3 delta entry points, 79fe4299),
  coords list policy intact.
- NEW KNOWN GAP (queued): CUDA device preservation broken for table-backed
  pots + SpiralArms (tables materialize on CPU); device-anchoring pass after
  #949, validate on the TITAN Xp box. Also pending user decision: project-wide
  mixed-input (tensor coords + python-float phi/t) gate + sweep.
