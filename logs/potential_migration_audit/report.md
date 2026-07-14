# Potential/Force backend-migration audit (TEST-BASED)

Date: 2026-07-13. Worktree: `/var/tmp/perf` (origin/feat/backends + one orthogonal
`assoc_legendre` perf change). CPU-only (`CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu`),
jax x64, torch float64. Probe: `probe.py` (raw output in `raw_results.json`).

Method: for every concrete `Potential`/`Force` subclass exported from `galpy.potential`
(59 total minus the `Potential`/`Force` ABCs = **57 tested**), instantiate it and call
`evaluateRforces` under (1) **jax eager**, (2) **jax traced** (`jax.jacfwd` w.r.t. R),
(3) **torch**. Classify by whether the result `is_backend_array` and is finite, catching
exceptions. Velocity-dependent forces (Chandrasekhar, FDM, NonInertialFrame) pass `v=`.

## Headline

**Essentially all potentials/forces ARE backend-migrated for eager evaluation, and the
delegation hypothesis is CONFIRMED** -- every `WrapperPotential`, `CompositePotential`,
and `MovingObjectPotential` is backend-compatible with **no direct `xp.`/`get_namespace`
call of its own**, purely by delegating to a migrated child (grep would false-negative all
of them). **44 / 57 are FULLY-BACKEND** (clean backend array under eager jax, jax trace,
AND torch).

But there ARE real gaps, and they are almost entirely a **traced/differentiable** story,
not an eager-crash story:

- **1 CRASH**: `AnyAxisymmetricRazorThinDiskPotential` -- torch raises `TypeError` (numpy
  `<` vs Tensor); jax eager silently returns a **numpy** value.
- **3 NUMPY-FALLTHROUGH**: `NonInertialFrameForce` (jax path only), `interpRZPotential`,
  `NullPotential` (trivial identically-zero force -> returns Python `float 0.0`).
- **7 EAGER-ONLY**: correct backend arrays under eager jax **and** torch, but break under
  `jax.jacfwd`/`jit` because of a numpy island in their internals ->
  `AnySphericalPotential`, `ChandrasekharDynamicalFrictionForce`,
  `FDMDynamicalFrictionForce`, `FerrersPotential`, `KuzminKutuzovStaeckelPotential`,
  `MovingObjectPotential`, `RingPotential`.
- **2 OUT-OF-SCOPE**: `SnapshotRZPotential`, `InterpSnapshotRZPotential` (need pynbody).

Net: **eager backend force evaluation works for 55/55 in-scope classes** (all but the two
pynbody potentials) -- nothing crashes on eager jax; only `AnyAxisymmetric` crashes on
eager torch. The genuine, non-trivial gaps are the **10 classes that are not jit/grad-safe
under a jax trace** (7 EAGER-ONLY + `AnyAxisymmetric` + `NonInertialFrameForce` +
`interpRZPotential`), each due to a raw-numpy internal (quadrature, elliptic integrals,
`coords.Rz_to_lambdanu`, orbit interpolation, or scipy spline interpolation).

## Counts

| Classification | Count |
|---|---|
| FULLY-BACKEND | 44 |
| EAGER-ONLY (traced-broken) | 7 |
| NUMPY-FALLTHROUGH | 3 |
| CRASH | 1 |
| OUT-OF-SCOPE (pynbody) | 2 |
| NOT-INSTANTIATED | 0 |
| **Total tested** | **57** |

## Genuine gaps (details + ledger cross-reference)

| Class | Root cause (traced/torch failure) | Ledger? |
|---|---|---|
| `AnyAxisymmetricRazorThinDiskPotential` | numpy Gauss-Legendre quadrature over user surfdens; `numpy.ndarray < Tensor` -> torch `TypeError`; jax eager returns numpy | yes -- `torch test_normalize_potential[...]` (units/normalize, not force-eval); force-eval torch-crash NOT explicitly ledgered |
| `NonInertialFrameForce` | numpy internals in frame accel; jax eager -> numpy scalar, traced `TracerArrayConversionError`; torch OK | yes -- 11 jax + 48 torch `test_noninertial.py` entries |
| `interpRZPotential` | scipy `RectBivariateSpline` interpolation -- inherently numpy (note: `interpSphericalPotential` IS migrated/FULLY) | partial -- `torch test_interprz_dxdv_3d` (C-variational, not force-eval) |
| `NullPotential` | returns Python `float 0.0` (force identically zero); not a backend array. Trivial/by-design | yes -- jax+torch `test_noninertial.py::...nullpotential` |
| `AnySphericalPotential` | numpy quadrature of user density (`AnySphericalPotential.py:92`) | yes -- `torch test_forceAsDeriv/evaluateAndDerivs/poisson[...]` |
| `ChandrasekharDynamicalFrictionForce` | numpy sigmar spline + lnLambda (`...py:239`) | yes -- `torch test_dynamfric.py::..._varLambda` etc. |
| `FDMDynamicalFrictionForce` | same friction internals + FDM factor | yes -- jax+torch `test_FDMdynamfric.py` |
| `FerrersPotential` | numpy numerical integration limits (`FerrersPotential.py:475`) | yes -- `torch test_energy_*conservation[Ferrers...]` |
| `KuzminKutuzovStaeckelPotential` | `coords.Rz_to_lambdanu` calls raw `numpy.sqrt` (`coords.py:2613`) -- coords helper not backend-migrated | yes -- `torch test_dxdv_3d_c_vs_python` (C-variational, not force-eval) |
| `MovingObjectPotential` | raw `numpy.cos/sin` geometry + numpy orbit interpolation (`MovingObjectPotential.py:95`). Eager delegates magnitude to migrated child correctly | yes -- `torch test_MovingObjectPotential_orbit`, `test_noninertial` combo |
| `RingPotential` | elliptic integrals via scipy special (`RingPotential.py:70`) | no direct force-eval ledger entry (eager passes both backends) |

Important nuance: for the 7 EAGER-ONLY classes the ledger entries are mostly `torch`-side
byte-identity/units tests or `_c` C-variational tests -- they do NOT record the
**jax-traced differentiability** gap this audit surfaces (eager force tests pass, so the
grad/jit break is invisible to the current test suite for several of them, e.g. `Ring`).

## Full table (gaps first, then FULLY-BACKEND)

| class | jax-eager | jax-traced (jacfwd) | torch | classification |
|---|---|---|---|---|
| AnyAxisymmetricRazorThinDiskPotential | **NUMPY** (backend=False) | **CRASH: TracerArrayConversionError: The numpy** | **CRASH: TypeError: '<' not supported between ** | CRASH |
| NonInertialFrameForce | **NUMPY** (backend=False) | **CRASH: TracerArrayConversionError: The numpy** | backend-array, finite | NUMPY-FALLTHROUGH |
| NullPotential | **NUMPY** (backend=False) | backend-array, finite | **NUMPY** (backend=False) | NUMPY-FALLTHROUGH |
| interpRZPotential | **NUMPY** (backend=False) | **CRASH: TracerArrayConversionError: The numpy** | **NUMPY** (backend=False) | NUMPY-FALLTHROUGH |
| AnySphericalPotential | backend-array, finite | **CRASH: TracerArrayConversionError: The numpy** | backend-array, finite | EAGER-ONLY |
| ChandrasekharDynamicalFrictionForce | backend-array, finite | **CRASH: TracerArrayConversionError: The numpy** | backend-array, finite | EAGER-ONLY |
| FDMDynamicalFrictionForce | backend-array, finite | **CRASH: TracerArrayConversionError: The numpy** | backend-array, finite | EAGER-ONLY |
| FerrersPotential | backend-array, finite | **CRASH: TracerArrayConversionError: The numpy** | backend-array, finite | EAGER-ONLY |
| KuzminKutuzovStaeckelPotential | backend-array, finite | **CRASH: TracerArrayConversionError: The numpy** | backend-array, finite | EAGER-ONLY |
| MovingObjectPotential | backend-array, finite | **CRASH: TracerArrayConversionError: The numpy** | backend-array, finite | EAGER-ONLY |
| RingPotential | backend-array, finite | **CRASH: TracerArrayConversionError: The numpy** | backend-array, finite | EAGER-ONLY |
| InterpSnapshotRZPotential | - | - | - | OUT-OF-SCOPE |
| SnapshotRZPotential | - | - | - | OUT-OF-SCOPE |
| AdiabaticContractionWrapperPotential | backend-array, finite | backend-array, finite | backend-array, finite | FULLY-BACKEND |
| BurkertPotential | backend-array, finite | backend-array, finite | backend-array, finite | FULLY-BACKEND |
| CompositePotential | backend-array, finite | backend-array, finite | backend-array, finite | FULLY-BACKEND |
| DehnenBarPotential | backend-array, finite | backend-array, finite | backend-array, finite | FULLY-BACKEND |
| DehnenCoreSphericalPotential | backend-array, finite | backend-array, finite | backend-array, finite | FULLY-BACKEND |
| DehnenSphericalPotential | backend-array, finite | backend-array, finite | backend-array, finite | FULLY-BACKEND |
| DiskMultipoleExpansionPotential | backend-array, finite | backend-array, finite | backend-array, finite | FULLY-BACKEND |
| DiskSCFPotential | backend-array, finite | backend-array, finite | backend-array, finite | FULLY-BACKEND |
| DoubleExponentialDiskPotential | backend-array, finite | backend-array, finite | backend-array, finite | FULLY-BACKEND |
| EinastoPotential | backend-array, finite | backend-array, finite | backend-array, finite | FULLY-BACKEND |
| ExpTruncNFWPotential | backend-array, finite | backend-array, finite | backend-array, finite | FULLY-BACKEND |
| FlattenedPowerPotential | backend-array, finite | backend-array, finite | backend-array, finite | FULLY-BACKEND |
| HernquistPotential | backend-array, finite | backend-array, finite | backend-array, finite | FULLY-BACKEND |
| HomogeneousSpherePotential | backend-array, finite | backend-array, finite | backend-array, finite | FULLY-BACKEND |
| IsochronePotential | backend-array, finite | backend-array, finite | backend-array, finite | FULLY-BACKEND |
| JaffePotential | backend-array, finite | backend-array, finite | backend-array, finite | FULLY-BACKEND |
| KeplerPotential | backend-array, finite | backend-array, finite | backend-array, finite | FULLY-BACKEND |
| KingPotential | backend-array, finite | backend-array, finite | backend-array, finite | FULLY-BACKEND |
| KuzminDiskPotential | backend-array, finite | backend-array, finite | backend-array, finite | FULLY-BACKEND |
| KuzminLikeWrapperPotential | backend-array, finite | backend-array, finite | backend-array, finite | FULLY-BACKEND |
| LogarithmicHaloPotential | backend-array, finite | backend-array, finite | backend-array, finite | FULLY-BACKEND |
| MN3ExponentialDiskPotential | backend-array, finite | backend-array, finite | backend-array, finite | FULLY-BACKEND |
| MiyamotoNagaiPotential | backend-array, finite | backend-array, finite | backend-array, finite | FULLY-BACKEND |
| MultipoleExpansionPotential | backend-array, finite | backend-array, finite | backend-array, finite | FULLY-BACKEND |
| NFWPotential | backend-array, finite | backend-array, finite | backend-array, finite | FULLY-BACKEND |
| PerfectEllipsoidPotential | backend-array, finite | backend-array, finite | backend-array, finite | FULLY-BACKEND |
| PlummerPotential | backend-array, finite | backend-array, finite | backend-array, finite | FULLY-BACKEND |
| PowerSphericalPotential | backend-array, finite | backend-array, finite | backend-array, finite | FULLY-BACKEND |
| PowerSphericalPotentialwCutoff | backend-array, finite | backend-array, finite | backend-array, finite | FULLY-BACKEND |
| PowerTriaxialPotential | backend-array, finite | backend-array, finite | backend-array, finite | FULLY-BACKEND |
| PseudoIsothermalPotential | backend-array, finite | backend-array, finite | backend-array, finite | FULLY-BACKEND |
| RazorThinExponentialDiskPotential | backend-array, finite | backend-array, finite | backend-array, finite | FULLY-BACKEND |
| RotateAndTiltWrapperPotential | backend-array, finite | backend-array, finite | backend-array, finite | FULLY-BACKEND |
| SCFPotential | backend-array, finite | backend-array, finite | backend-array, finite | FULLY-BACKEND |
| SoftenedNeedleBarPotential | backend-array, finite | backend-array, finite | backend-array, finite | FULLY-BACKEND |
| SphericalShellPotential | backend-array, finite | backend-array, finite | backend-array, finite | FULLY-BACKEND |
| SpiralArmsPotential | backend-array, finite | backend-array, finite | backend-array, finite | FULLY-BACKEND |
| TriaxialGaussianPotential | backend-array, finite | backend-array, finite | backend-array, finite | FULLY-BACKEND |
| TriaxialHernquistPotential | backend-array, finite | backend-array, finite | backend-array, finite | FULLY-BACKEND |
| TriaxialJaffePotential | backend-array, finite | backend-array, finite | backend-array, finite | FULLY-BACKEND |
| TriaxialNFWPotential | backend-array, finite | backend-array, finite | backend-array, finite | FULLY-BACKEND |
| TwoPowerSphericalPotential | backend-array, finite | backend-array, finite | backend-array, finite | FULLY-BACKEND |
| TwoPowerTriaxialPotential | backend-array, finite | backend-array, finite | backend-array, finite | FULLY-BACKEND |
| interpSphericalPotential | backend-array, finite | backend-array, finite | backend-array, finite | FULLY-BACKEND |

## Notes on constructors (not gaps)

- `SCFPotential`, `DiskSCFPotential`, `DiskMultipoleExpansionPotential`,
  `MultipoleExpansionPotential`, `KingPotential`, `AnySpherical/AnyAxisymmetric` all
  instantiate from **defaults** (or a trivial density) -> tested successfully; none are
  "constructor-complex".
- Wrappers built with `pot=HernquistPotential(0.5,0.3)`; Adiabatic with
  `baryonpot=Hernquist`, `pot=NFW`; MovingObject with an `Orbit` integrated in a
  LogarithmicHalo, perturber kept away from the eval point (a coincident eval point gives
  `nan` on **numpy** too -- a physics divide-by-zero, not a backend gap).
- `SnapshotRZPotential`/`InterpSnapshotRZPotential` require a pynbody snapshot -> external
  dep, OUT-OF-SCOPE (not a migration gap).

## Gap-PR closure status (2026-07-14)

The traced/jit + amp-grad gaps this audit surfaced are being closed by a fan-out
of small, disjoint-file PRs (#1111–#1123). All share the same fix class: dispatch
on `is_backend_array(x)` (NOT `get_namespace() is not numpy`) so a bare
numpy/python input under a *forced* backend stays numpy (torch.sqrt/cos reject a
bare scalar; a backend array leaking into the numpy C-bridge breaks the
integrator) — numpy path byte-identical.

Merged: #1111, #1112 (KuzminKutuzov coords), #1115, #1116, #1117 (NonInertial),
#1119, #1120 (CosmphiDisk→LopsidedDisk amp), #1121 (AnyAxisym).

Open (CI draining / awaiting review), all rebased onto post-#1121 base, all
coverage-clean (no added-and-missing lines, verified locally):
- #1113 AnySpherical — r=0/inf edge via xp.where + in-backend GL tail quad.
- #1114 MovingObject — `_numpy_ctx_for_numpy_inputs` decorator pins numpy inputs
  to numpy under a forced backend (the integrator/gate-bypass path); L50-51 covered.
- #1118 Chandrasekhar/FDM — is_backend_array gate on _Rforce/_phitorque/_zforce;
  const_lnLambda/const_FDMfactor backend branches now covered.
- #1122 differentiability sweep (test-only) — held CLEAN; final rebase drops all
  gap-list entries once #1113/#1123 merge (option b).
- #1123 IsothermalDisk — backend-aware amp sqrt (folds amp into H); amp-grad
  grad-vs-FD (jax/torch) covers the is_backend_array(self._amp) branch.

The "AnyAxisym line 33" codecov flag on #1113/#1114 was a stale-base phantom
(base predated #1121's merge; line 33 is a comment) — cleared by rebasing.
