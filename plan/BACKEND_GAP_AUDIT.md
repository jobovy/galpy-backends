# Backend gap audit: feat/backends @ 5702435d0 (2026-10-10)

**Question:** "Are there any other backend gaps besides the AdiabaticContraction wrapper and the torus code?"

**Answer: yes, many.** About 60 (class × parameter) combinations raise, or only work eagerly. More importantly, **9 gap families return finite gradients that are silently wrong, zero or disconnected.** Every one of them passes the value-only forced-backend suite. Most are constructor-time gaps: the class says `is_backend_compatible=True` and evaluates fine on backend coordinates, but it is not differentiable (or not jit-able) with respect to its own constructor parameters.

Each family below is grouped into one PR-sized unit for the stacked fix plan (section 4).

## 0. Method

- **Harness:** `/var/tmp/gapaudit/audit/{harness,driver}.py`, plus `spec_pot.py`, `spec_df.py` and `spec_aa.py`, giving 88 + 45 + 72 specs. Each spec builds the object **inside** the differentiated function.
- **What is compared:** for every numeric constructor parameter, plus one coordinate, the audit compares:
  - `jax.jit(jax.jacrev)`, falling back to eager to separate NOJIT from RAISE;
  - torch `autograd.grad`;
  - a Richardson central finite difference (FD) on numpy, using h = 2e-3·s and 1e-3·s, with convergence estimated from the difference of the two. Sampler and DF outputs use a same-backend FD instead.
- **Classes:** ok (rel < 1e-6, or within the FD error), WRONG, ZERO, DISC (no graph), NONF (NaN), RAISE, NOJIT (works eagerly only), ISLAND (a numpy value comes back), TIMEOUT (jit > 300 s).
- **Three input modes:**
  1. **data**: coordinates are backend arrays; only the differentiated parameter is a backend array.
  2. **FORCED**: Python-float coordinates under `use(bk, force=True)`.
  3. **float-coords**: Python-float coordinates with a backend parameter and no force (torch only; `floatcoords.py`).
- **Scope:** all public `galpy.potential` classes and wrappers, `galpy.df` including jeans and the impulse kicks, `galpy.actionAngle` (Torus excluded), `Orbit.integrate` per method family, `Orbit.analytic`, 22 `coords` transforms and 13 `conversion` helpers.
- **Not covered:** the plot helpers; Snapshot*/InterpSnapshotRZ (pynbody is not installed); TimeDependentAmplitude with a backend `A(t)`.

## 1. Ranked gaps

### 1a. Silently wrong, zero or disconnected gradients (most dangerous)

| # | Where | Symptom | Root cause (one line) | Size |
|---|---|---|---|---|
| W1 | torch `Orbit.integrate` with any C method (dop853_c, rk4_c, rk6_c, dopr54_c, symplec4_c, symplec6_c, leapfrog_c); also IsochroneApprox, StaeckelGrid, AdiabaticGrid and **streamdf/streamgapdf** built on them | d/d(potential parameter) is **DISC**. Through the IsochroneApprox default `dopr54_c`: streamdf d/dpR **45× wrong**, d/dq **exactly 0**; streamgapdf d/dtimpact **2× wrong** | `Orbits.py:364-395` `_pot_has_traced_param` uses `under_trace`, which is False for an eager torch tensor with requires_grad. The C-STM path is taken; it is first-order and IC-only, so it drops parameter, time-grid and second-order sensitivities without warning | M |
| W2 | `actionAngleStaeckel(c=True)`, `actionAngleStaeckelGrid`, `actionAngleAdiabaticGrid`; through them `quasiisothermaldf` | jax d/ddelta = **0.0**; jax AdiabaticGrid d/da, d/db = **0.0** under jit; torch a, b, delta **DISC**. qdf with Staeckel c=True under force: torch d/da **0.63–5× wrong**, d/ddelta DISC/0 | The C action Jacobians (`actionAngleStaeckel.py:829-862, 975-1016`, `stop_gradient(delta)` at :844/:994; `actionAngleAdiabatic.py:33-61`) close over the potential. Their `custom_vjp` has no cotangent for potential or delta. Documented only as "delta is a fixed reference" in a docstring | M |
| W3 | `actionAngleStaeckel(c=False)` frequencies and angles | d/ddelta **WRONG 2.6e-3 to 1.3e-2** (jax and torch agree with each other); angles d/dR **2.6e-5**. Actions are correct | The graph misses a delta dependency in the turning-point refinement (`_refine_tp` :161, FD Newton slope with h=1e-6) and/or the chi-mesh profiles (`_staeckel_chi_profiles`, :573-584). Values match numpy to 1e-13 | M (investigate first) |
| W4 | `quasiisothermaldf` with an unnormalized potential (Isochrone amp=1, b=0.6) | jax: **NaN gradient for every parameter** (f too); torch sigmaR2 and meanvT NaN | `quasiisothermaldf.py:392` computes `kappa*jr*exp(...)` before masking jr = inf/NaN at unbound GL nodes (dead-branch NaN). The tests only use normalize=1 | S |
| W5 | `streamdf` jax d/dq | ~3% off (AD -9.57e-6 vs FD -9.3e-6) | Frozen frequency-covariance eigenvectors (`real_eig` stop_gradient; already flagged in a test comment) | M |
| W6 | `interpRZPotential` under a forced backend | Returns **numpy float64** with a torch parameter, so the gradient is silently dropped (jax raises ValueError) | `_backend_compatible=False` (deliberate: the grid edge differs 3.3%, see `_input.py:_backend_ready`), so it is a numpy island under force; grid construction is numpy | M-L |
| W7 | `coords.radec_to_lb(epoch=1950)` | value and gradient differ from numpy by ~1e-6 | numpy+astropy uses FK4 with E-terms; the backend uses a plain rotation (`coords.py:388`, `get_epoch_angles:3616`) | S |
| W8 | `KingPotential` / `kingdf` d/dW0 | 1.1e-6 rel on Phi (2e-7 on forces), jax and torch agree | AD through the King ODE vs numpy solver tolerance. Probably acceptable; verify against a tighter-tolerance FD | S |
| W9 | `streamspraydf.sample(tail='both')` (coordinator report) | Fails under jit. Eagerly on jax the stripping-time output `dt` is silently converted to numpy (gradient dropped) | `streamspraydf.py:240` `dt = numpy.concatenate([dt_l, dt_t])` (the `out` arm right above it is already xp) | S |

Not a backend gap: AdiabaticContraction d/damp = 0 is correct, because **numpy ignores `amp` too**. See section 6.

### 1b. Raises (loud) and NOJIT

Ranked by user impact.

1. **R-NS, the namespace follows only the coordinates (cross-cutting, the biggest single gap).** `_evaluate`/`_Rforce` resolve `xp = get_namespace(R, z)` and never look at the parameters. Any Python-float coordinate combined with a backend parameter therefore reaches `numpy.sqrt(tensor)`.
   - **User-facing:** `jax.grad(lambda a: MiyamotoNagaiPotential(a=a)(1.0, 0.1))` raises. Same for Plummer, NFW, Isochrone, LogHalo and others: 60 of 88 specs fail in float-coords mode. Rforce often works by accident (no xp call on parameter terms), Phi does not. A few return numpy silently: Multipole a/q, Chandrasekhar rhm, GaussianAmplitude to/sigma, CylSep Rp.
   - **Internal call sites with numpy coordinates** (these fail even when the user passes backend coordinates):
     - `verticalPotential.py:59` `self._Pot(R0, 0.0)`: toVerticalPotential, RZToverticalPotential, actionAngleVertical, actionAngleAdiabatic d/da, d/db.
     - `Orbits.py:4268` `evaluatePotentials(pot, _INF, 0.0)`: Orbit.analytic staeckel/adiabatic.
     - `integrateFullOrbit.py:1291`: python integrators.
     - OblateStaeckel tabulation, CylindricallySeparable `Rp`, interpRZ grid.
     - `isotropicNFWdf.py:59`, `isotropicPowerLawdf.py:101` `pot(rmax, 0)`: NFW/PowerLaw sphericaldfs.
     - Kuijken–Dubinski `ts/tds` closures (`KuijkenDubinskiDiskExpansionPotential.py:206`).
   - **Under FORCED all of these work**, which is why the forced suite never saw them.
2. **R-ROT, rotation and axis-ratio setup in numpy.**
   - Affected:
     - EllipsoidalPotential b, c, pa, zvec: all 7 triaxial classes (TriaxialNFW/Hernquist/Jaffe/Gaussian, PerfectEllipsoid, PowerTriaxial, TwoPowerTriaxial);
     - Ferrers b;
     - RotateAndTilt galaxy_pa, inclination, sky_pa, offset.
   - Sites:
     - `EllipsoidalPotential.py:130` `numpy.fabs(self._b-1)` and :142-160 `numpy.array/cos/sin/dot`;
     - `FerrersPotential.py:134`;
     - `RotateAndTiltWrapperPotential.py:102-160`;
     - `util/__init__.py:169 _rotate_to_arbitrary_vector` (also `util/__init__.py:209` `to_host`, which breaks jit of the impulse kicks).
3. **R-SHAPE, scalar-parameter branching and numpy math in constructors.**
   - `TwoPowerSphericalPotential.py:297-299, 541-556`: `round(alpha)` and `round(beta)` special-casing (TwoPower, DehnenSpherical alpha/beta; jax unhashable or TAC, torch `__round__`).
   - `PowerSphericalPotential.py:56` and `PowerTriaxialPotential.py:87`: `if alpha != 3`.
   - `FlattenedPowerPotential.py:87`.
   - `ExpTruncNFWPotential.py:119` (mass); `KingPotential.py:49` (M).
   - NFW `conc`: `TwoPowerSphericalPotential.py:1180 numpy.log`.
   - `DehnenBarPotential.py:134` (beta, numpy.sqrt) and `:143 if omegab != 0`.
   - `SteadyLogSpiralPotential.py:84`.
   - `SpiralArmsPotential.py:109-119` (alpha, H).
   - `normalize=` under jit: `if normalize or (...)` in every constructor (Plummer :50, DoubleExponentialDisk :166, ...).
4. **R-DISS, dissipative forces.**
   - Never set `_backend_compatible`, so the forced boundary does not coerce. Chandrasekhar and FDM then take the numpy `lnLambda` path (`ChandrasekharDynamicalFrictionForce.py:274`, `FDMDynamicalFrictionForce.py:195`). Under FORCED, amp, GMs, gamma and a all RAISE.
   - `sigmar=` callable: `to_host` at :171.
   - NonInertialFrame Omega: `NonInertialFrameForce.py:147 numpy.atleast_1d`.
5. **R-AC, AdiabaticContraction (known).** nfwa, ha and f_bar RAISE. scipy `fixed_point` + `interp1d` + `quad` construction, `_backend_compatible=False`.
6. **R-MO, MovingObjectPotential.**
   - ISLAND under force (flag False).
   - NOJIT: `Orbits.py:7447` `numpy.atleast_1d(t)` on the trajectory lookup.
   - Orbit IC not differentiable: `Orbits.py:625` `numpy.array(to_host(vxvv))`.
   - pb (kernel parameter) RAISE under force.
7. **R-DISK, disk expansions.**
   - DiskSCF and DiskMultipole `Sigma={'h':…}` and `hz={'h':…}` RAISE. `KuijkenDubinskiDiskExpansionPotential._set_dens_funcs` gives 2-arg vs 3-arg lambda arity mismatches on the backend path, plus the R-NS closures.
   - `scf_compute_coeffs_axi` torch d/da: `SCFPotential.py:2180` `like(_CC, ...)` detaches. Jax d/da is NOJIT under force.
8. **R-AA, actionAngle (from the aA fork).**
   - IsochroneInverse b: `actionAngleIsochroneInverse.py:63 numpy.sqrt`.
   - StaeckelGrid delta: `actionAngleStaeckel_c.py:641 numpy.atleast_1d`.
   - Orbit.analytic delta: `Orbits.py:4287` indexing a 0-d delta.
   - AdiabaticGrid gamma: ConcretizationTypeError. Adiabatic gamma is NOJIT.
   - **0-d backend coordinates crash** Spherical, Adiabatic and Staeckel c=True (`actionAngleSpherical.py:938 rperi[:,None]`, `actionAngleStaeckel_c.py:551 len(R)`, and :205). This also hits qdf.
   - Python integrators (odeint, dop853, leapfrog) and ias15_c with a traced *parameter* give a cryptic TAC or numpy() error, while a traced IC gets the documented refusal.
9. **R-DF, distribution functions (from the df fork).**
   - diskdf beta: `diskdf.py:151 numpy.sqrt(2/(1+beta))`, :2089 `if beta == 0`.
   - dehnendf `correct=True`: `_createSavefilename` hash (`diskdf.py:3131`) + iterative numpy DFcorrection. It raises even for d/dR. L.
   - constantbeta family:
     - Hernquist beta: `constantbetadf.py:381 special.gamma`;
     - PowerLaw amp/alpha: `as_numpy` at :366;
     - general beta: `int(numpy.floor(1.5-beta))` at :500;
     - torch amp: scipy brentq via `sphericaldf.py:386` (float bracket).
   - streamdf:
     - `streamdf.py:596 numpy.log(sigangle)` (d/dsigv);
     - `sigangledAngle` `numpy.amax/sqrt` (:3800);
     - `approx=False` uses scipy quad (`streamgapdf.py:349`).
   - **streamgapdf `higherorderTrack=True`** (coordinator report, confirmed): `NotImplementedError` on jax/torch at `streamgapdf.py:465-469` and `:772-776`. Both branches are hidden with `# pragma: no cover`, a violation of the no-hiding rule. The backend only has the piecewise kick approximation.
   - Unmigrated impulse kicks: `impulse_deltav_general_curvedstream` (quad, :2278), `_plummerstream` (:2742), `_plummerstream_curvedstream` (:2819), `_general_fullplummerintegration` (Orbit + as_numpy). The plummer/hernquist/general kicks are NOJIT (`util/__init__.py:209 to_host`).
   - NOJIT:
     - eddingtondf denspot: `eddingtondf.py:127` `bool()` in `_finite`;
     - `jeans.sigmalos`: `jeans.py:282`;
     - kingdf M (see R-SHAPE).
10. **R-COORDS.**
    - `Rz_to_lambdanu` Delta: `coords.py:3096`.
    - `vxvyvz_to_galcencyl` Zsun: `coords.py:1963`, a traced Zsun mixed with a numpy `vsun` list.

### 1c. Numpy islands under force with float inputs

The values are right but the gradient dies, so these are invisible to the value suite.

- **sphericaldf `__call__(R,vR,vT,z,vz)`**: only the radius entry points coerce. This affects every spherical DF and kingdf.
- **diskdf `__call__` and moments**: by design (`test_ssp_plain_scalar_stays_on_numpy`); also evolveddiskdf.
- **streamdf/streamgapdf `density_par`/`meanOmega` and the impulse kicks**: no `@backend_input`.
- **Orbit.analytic jr, jz, Or, wr** return `numpy.float64`, while rperi, rap, zmax and e are backend arrays.
- **Smaller ones**: HarmonicInverse O, estimateBIsochrone, Rz_to_lambdanu, conversion helpers (benign).
- **AnyAxisymmetricRazorThinDisk under forced torch**: galpy calls the user's `surfdens(1)` with a Python int (`AnyAxisymmetricRazorThinDiskPotential.py:221`), so an xp-agnostic user callable raises.

### 1d. Flag (A-column) inconsistencies

These change whether the forced boundary coerces at all.

- **Report False** but evaluate on the backend: ChandrasekharDF, FDMDF, NonInertialFrameForce (flag unset); SolidBodyRotationWrapper and TimeDependentAmplitudeWrapper (False); every actionAngle object (never set).
- **Report True** but have parameter gaps: every R-ROT, R-SHAPE and R-DISK class.

The flag is **not** a differentiability signal. AdiabaticContraction (False) and interpRZ/MovingObject (False) are the only real "islands by flag".
## 2. Matrices

### 2a. galpy.potential (88 specs)

- **Columns:**
  - A = `is_backend_compatible`.
  - C/D data = jax / torch with backend coordinates.
  - C/D FORCED = jax / torch with Python-float coordinates under `use(bk, force=True)`.
  - FC = torch, float coordinates with a backend parameter and no force.
- **Cells:** a cell lists only the failing keys. For B (backend output under force), every row returned backend arrays unless marked ISLAND, except MovingObject and interpRZ, which are ISLANDs under force.
- **All-ok rows** (all 5 columns ok): DehnenCoreSpherical, DoubleExponentialDisk, Hernquist, KuzminDisk, HenonHeiles, AnySpherical, interpSpherical (both forms).
- **Ok in all 4 data/forced columns, failing only in FC** (= the R-NS gap): Burkert, Einasto (+rs), ExpTruncNFW, HomogeneousSphere, Isochrone, Jaffe, Kepler, KuzminKutuzovStaeckel, LogarithmicHalo, MN3ExponentialDisk, MiyamotoNagai, NFW (+rmax/vmax), Plummer, PowerSphericalwCutoff, PseudoIsothermal, RazorThinExponentialDisk, Ring, SphericalShell, SoftenedNeedleBar, SCF, the DehnenSmooth/GaussianAmplitude/SolidBodyRotation/CorotatingRotation/TimeDependentAmplitude/KuzminLike wrappers, CosmphiDisk (+rb), EllipticalDisk, LopsidedDisk, TransientLogSpiral, RZToplanar, planar SolidBodyRotation, KG, IsothermalDisk.

| spec | A | C data jax | D data torch | C FORCED jax | D FORCED torch | FC torch |
|---|---|---|---|---|---|---|
| DehnenSphericalPotential | T | alpha:NOJIT | alpha:RAISE | alpha:NOJIT | alpha:RAISE | alpha:RAISE |
| FlattenedPowerPotential | T | alpha:NOJIT | ok | alpha:NOJIT | ok | ok |
| PerfectEllipsoidPotential | T | b:RAISE, c:RAISE | b:RAISE, c:RAISE | b:RAISE, c:RAISE | b:RAISE, c:RAISE | b:RAISE, c:RAISE |
| PowerSphericalPotential | T | alpha:NOJIT | ok | alpha:NOJIT | ok | ok |
| PowerTriaxialPotential | T | alpha:NOJIT, b:RAISE, c:RAISE | b:RAISE, c:RAISE | alpha:NOJIT, b:RAISE, c:RAISE | b:RAISE, c:RAISE | b:RAISE, c:RAISE |
| TriaxialGaussianPotential | T | b:RAISE, c:RAISE | b:RAISE, c:RAISE | b:RAISE, c:RAISE | b:RAISE, c:RAISE | sigma:RAISE, b:RAISE, c:RAISE |
| TriaxialHernquistPotential | T | b:RAISE, c:RAISE | b:RAISE, c:RAISE | b:RAISE, c:RAISE | b:RAISE, c:RAISE | b:RAISE, c:RAISE |
| TriaxialJaffePotential | T | b:RAISE, c:RAISE | b:RAISE, c:RAISE | b:RAISE, c:RAISE | b:RAISE, c:RAISE | a:RAISE, b:RAISE, c:RAISE |
| TriaxialNFWPotential | T | b:RAISE, c:RAISE | b:RAISE, c:RAISE | b:RAISE, c:RAISE | b:RAISE, c:RAISE | b:RAISE, c:RAISE |
| TwoPowerSphericalPotential | T | alpha:RAISE, beta:RAISE | alpha:RAISE, beta:RAISE | alpha:RAISE, beta:RAISE | alpha:RAISE, beta:RAISE | a:RAISE, alpha:RAISE, beta:RAISE |
| TwoPowerTriaxialPotential | T | TIMEOUT(jit>300s); eager b,c RAISE | b:RAISE, c:RAISE | TIMEOUT(jit>300s) | b:RAISE, c:RAISE | alpha:RAISE, beta:RAISE, b:RAISE, c:RAISE |
| FerrersPotential | T | b:RAISE | b:RAISE | b:RAISE | b:RAISE | a:RAISE, b:RAISE, c:RAISE, omegab:RAISE, pa:RAISE |
| DehnenBarPotential | T | rolr:NOJIT, beta:RAISE | beta:RAISE | rolr:NOJIT, beta:RAISE | beta:RAISE | chi:RAISE, rolr:RAISE, barphi:RAISE, tform:RAISE, tsteady:RAISE, beta:RAISE, alpha:RAISE |
| SpiralArmsPotential | T | alpha:RAISE | alpha:RAISE, H:RAISE | alpha:RAISE | alpha:RAISE, H:RAISE | alpha:RAISE, r_ref:RAISE, phi_ref:RAISE, Rs:RAISE, H:RAISE, omega:RAISE |
| DehnenBarPotential[omegab,rb,Af] | T | omegab:NOJIT | ok | omegab:NOJIT | ok | omegab:RAISE, rb:RAISE, Af:RAISE |
| NFWPotential[conc,mvir] | T | conc:RAISE | conc:RAISE | conc:RAISE | conc:RAISE | conc:RAISE, mvir:RAISE, H:RAISE, Om:RAISE |
| ExpTruncNFWPotential[mass] | T | mass:NOJIT | ok | mass:NOJIT | ok | a:RAISE, rc:RAISE |
| PlummerPotential[normalize] | T | normalize:NOJIT | ok | normalize:NOJIT | ok | b:RAISE |
| MiyamotoNagaiPotential[normalize] | T | normalize:NOJIT | ok | normalize:NOJIT | ok | a:RAISE, b:RAISE |
| DoubleExponentialDiskPotential[normalize] | T | normalize:NOJIT | ok | normalize:NOJIT | ok | ok |
| TriaxialNFWPotential[zvec,pa] | T | pa:RAISE | pa:RAISE | pa:RAISE | pa:RAISE | pa:RAISE |
| AnyAxisymmetricRazorThinDiskPotential | T | ok (with an xp-agnostic surfdens) | ok | amp,R:NOJIT | amp,R:RAISE (galpy calls `surfdens(1)` with a python int, l.221) | ok |
| scf_compute_coeffs_axi->SCFPotential | T | TIMEOUT(jit>300s; eager ok) | a:RAISE, q:RAISE | a:NOJIT, q:NOJIT, R:NOJIT | a:RAISE | a:RAISE, q:RAISE |
| DiskSCFPotential | T | h:RAISE, hz:RAISE | h:RAISE, hz:RAISE | amp:NOJIT, h:RAISE, hz:RAISE, R:NOJIT | h:RAISE, hz:RAISE | h:RAISE, hz:RAISE |
| MultipoleExpansionPotential | T | TIMEOUT(jit>300s; eager ok) | ok | TIMEOUT(jit>300s; eager ok) | ok | a:ISLAND, q:ISLAND |
| ChandrasekharDynamicalFrictionForce | F | ok | ok | amp:RAISE, GMs:RAISE, gamma:RAISE, rhm:NOJIT, a:RAISE, R:NOJIT | amp:RAISE, GMs:RAISE, gamma:RAISE, a:RAISE | amp:RAISE, GMs:RAISE, gamma:RAISE, rhm:ISLAND, a:RAISE |
| DiskMultipoleExpansionPotential | T | h:RAISE, hz:RAISE | h:RAISE, hz:RAISE | h:RAISE, hz:RAISE | h:RAISE, hz:RAISE | h:RAISE, hz:RAISE |
| ChandrasekharDynamicalFrictionForce[sigmar] | F | s0:RAISE | s0:RAISE | GMs:RAISE, s0:RAISE | GMs:RAISE, s0:RAISE | GMs:RAISE, s0:RAISE |
| FDMDynamicalFrictionForce | F | ok | ok | amp:RAISE, GMs:RAISE, gamma:RAISE, rhm:NOJIT, m:RAISE, a:RAISE, R:NOJIT | amp:RAISE, GMs:RAISE, gamma:RAISE, m:RAISE, a:RAISE | amp:RAISE, GMs:RAISE, gamma:RAISE, rhm:ISLAND, m:RAISE, a:RAISE |
| NonInertialFrameForce | F | Omega:RAISE | Omega:RAISE | Omega:RAISE | Omega:RAISE | Omega:RAISE, Omegadot:RAISE |
| RotateAndTiltWrapperPotential[galaxy_pa] | T | galaxy_pa:RAISE | galaxy_pa:RAISE | galaxy_pa:RAISE | galaxy_pa:RAISE | galaxy_pa:RAISE, mna:RAISE, mnb:RAISE |
| RotateAndTiltWrapperPotential[incl] | T | inclination:RAISE, sky_pa:RAISE | inclination:RAISE, sky_pa:RAISE | inclination:RAISE, sky_pa:RAISE | inclination:RAISE, sky_pa:RAISE | inclination:RAISE, sky_pa:RAISE, mna:RAISE, mnb:RAISE |
| RotateAndTiltWrapperPotential[offset] | T | ox:RAISE | ox:RAISE | ox:RAISE | ox:RAISE | ox:RAISE, mna:RAISE, mnb:RAISE |
| OblateStaeckelWrapperPotential | T | mna:RAISE, mnb:RAISE | mna:RAISE, mnb:RAISE | ok | ok | delta:RAISE, mna:RAISE, mnb:RAISE |
| CylindricallySeparablePotentialWrapper | T | mna:RAISE, mnb:RAISE | mna:RAISE, mnb:RAISE | ok | ok | Rp:ISLAND, mna:RAISE, mnb:RAISE |
| AdiabaticContractionWrapperPotential | F | nfwa:RAISE, ha:RAISE, f_bar:RAISE | nfwa:RAISE, ha:RAISE, f_bar:RAISE | amp:NOJIT, nfwa:RAISE, ha:RAISE, f_bar:RAISE, R:NOJIT | nfwa:RAISE, ha:RAISE, f_bar:RAISE | amp:ISLAND, nfwa:RAISE, ha:RAISE, f_bar:RAISE |
| MovingObjectPotential | F | amp:NOJIT, pb:NOJIT, oR:RAISE, R:NOJIT | oR:RAISE | pb:RAISE, oR:RAISE | pb:RAISE, oR:RAISE, R:RAISE | pb:RAISE, oR:RAISE |
| interpRZPotential | F | mna:RAISE, mnb:RAISE | mna:RAISE, mnb:RAISE | mna:RAISE, mnb:RAISE, R:NOJIT | mna:ISLAND, mnb:ISLAND, R:RAISE | mna:RAISE, mnb:RAISE |
| KingPotential | T | TIMEOUT(jit>300s); eager W0 1.1e-6 | W0: 1.1e-6 (marginal) | TIMEOUT(jit>300s) | W0: 1.1e-6 (marginal) | ok |
| SteadyLogSpiralPotential | T | omegas:NOJIT | ok | omegas:NOJIT | ok | omegas:RAISE, A:RAISE, alpha:RAISE, gamma:RAISE, tform:RAISE, tsteady:RAISE |
| toPlanarPotential(DehnenBar) | T | omegab:NOJIT | ok | omegab:NOJIT | ok | omegab:RAISE, rb:RAISE |
| RZToverticalPotential(MN) | T | mna:RAISE, mnb:RAISE | mna:RAISE, mnb:RAISE | ok | ok | mna:RAISE, mnb:RAISE |
| toVerticalPotential(MN) | T | mna:RAISE, mnb:RAISE | mna:RAISE, mnb:RAISE | ok | ok | mna:RAISE, mnb:RAISE |

### 2b. actionAngle, Orbit and util (72 specs)

- **Inputs:** 1-d coordinates. 0-d crashes are listed separately under R-AA.
- **FORCED:** identical to data mode except for one change. StaeckelGrid and AdiabaticGrid torch a,b go from DISC to RAISE (`len` of an unsized object).
- **A column:** every actionAngle object reports A=F (the flag is never set).

| spec | B forced | C jax | D torch |
|---|---|---|---|
| Staeckel c=False | bk | TIMEOUT jit (57 s eager); eager delta **WRONG 2.6e-3** | a,b ok; **delta WRONG 2.6e-3 to 1.3e-2** (Or, angles); **R WRONG 2.6e-5** (angles); vz ok |
| Staeckel c=True | bk | a,b RAISE(TAC); **delta ZERO**; R,vz ok | **a,b,delta DISC**; R,vz ok |
| Adiabatic c=False/True | bk | a,b RAISE (R-NS); gamma ok (c=F) / NOJIT; R,vz ok | a,b RAISE; gamma,R,vz ok |
| Isochrone (+ip), Harmonic, Spherical (Hernquist, LogHalo) | bk | ok | ok |
| Vertical(toVertical(MN)) | bk | a,b RAISE (R-NS) | a,b RAISE |
| Vertical(IsothermalDisk), VerticalInverse (out of scope) | bk | ok | ok |
| IsochroneApprox | bk | ok (jit) | **a,b DISC** (W1); bI,R ok |
| StaeckelGrid | bk | a,b,delta RAISE; R,vz ok | **a,b DISC**; delta RAISE |
| AdiabaticGrid | bk | **a,b ZERO (jit)**, RAISE eager; gamma RAISE | **a,b DISC**; gamma ok |
| HarmonicInverse | ISLAND (O float) | ok | O not a tensor |
| IsochroneInverse | bk | b RAISE | b RAISE |
| estimateDeltaStaeckel | bk | ok | ok |
| estimateBIsochrone | ISLAND | ok | ok |
| Orbit.integrate diffrax / torchdiffeq | bk | a,b,R ok | a,b,R ok |
| Orbit.integrate dop853_c, rk4_c, rk6_c, dopr54_c, symplec4_c, symplec6_c, leapfrog_c | bk | a,b,R ok | **a,b DISC** (W1); R ok |
| Orbit.integrate ias15_c | bk | a,b RAISE; R RAISE (documented) | a,b DISC (warns); R RAISE (documented) |
| Orbit.integrate odeint, dop853, leapfrog (python) | bk | a,b RAISE (cryptic); R RAISE (documented) | same |
| Orbit.analytic staeckel / adiabatic | ISLAND (jr,jz,Or,wr) | a,b RAISE (R-NS); delta RAISE (0-d index) | same |
| Orbit.analytic spherical; Orbit.E/Jacobi/Lz | ISLAND / bk | ok | ok |
| 20 coords transforms | bk | ok | ok |
| coords.radec_to_lb epoch=1950 | bk | ra **WRONG 1.5e-6** | same |
| coords.vxvyvz_to_galcencyl | bk | Zsun RAISE (mixed namespaces) | same |
| coords.Rz_to_lambdanu | ISLAND | Delta RAISE | Delta RAISE |
| 13 conversion helpers | ISLAND (benign scalars) | ok | ok |

### 2c. galpy.df (45 specs)

FD: numpy Richardson, or a same-backend self-FD for samplers.

| spec | A | B forced | C jax (data / FORCED) | D torch (data / FORCED) |
|---|---|---|---|---|
| isotropicHernquist, isotropicPlummer, eddingtondf (+Hernquist) | T | `__call__` ISLAND | ok / ok | ok / f NOT_T |
| isotropicNFW, osipkovmerrittNFW (a, amp) | T | ISLAND | a RAISE (R-NS) / ok | amp,a RAISE / ok |
| isotropicPowerLaw, osipkovmerrittPowerLaw (alpha, amp) | T | ISLAND | alpha RAISE / ok | amp,alpha RAISE / ok |
| osipkovmerrittHernquist, eddingtondf[denspot], kingdf | T | ISLAND | ok; denspot and king M NOJIT | ok |
| constantbetaHernquistdf beta | T | ISLAND | RAISE / RAISE | RAISE / RAISE |
| constantbetaPowerLawdf amp, alpha, beta | T | ISLAND | RAISE / RAISE | RAISE / RAISE |
| constantbetadf (general) | T | ISLAND | beta RAISE; R NOJIT | amp,beta RAISE |
| osipkovmerrittdf (general) | T | ISLAND | ok | ok |
| jeans.sigmar, sigmalos | – | ok | ok; sigmalos NOJIT | ok |
| dehnendf, shudf, schwarzschilddf | F | ISLAND (all outputs) | beta RAISE, rest ok / everything RAISE | same |
| dehnendf correct=True | F | ISLAND | all RAISE (even d/dR) | all RAISE |
| evolveddiskdf | F | ISLAND | ok, omegas NOJIT / hR RAISE | ok / TIMEOUT |
| qdf [Isochrone, unnormalized] | F | bk | **NONF for every parameter** | **sigmaR2, meanvT NONF** |
| qdf [MN + Adiabatic c=False] | F | bk | a RAISE (R-NS), R RAISE (0-d) / TIMEOUT | a,R RAISE / a ok, R RAISE |
| qdf [Staeckel c=True] | F | bk | a,delta,R RAISE / **delta ZERO**, a RAISE | all RAISE / **a WRONG 0.63–5×, delta DISC** |
| impulse_deltav_plummer, _hernquist, _general | F | ISLAND | ok, NOJIT; general forced b,amp,pb RAISE | ok; general forced NOT_T |
| impulse_deltav_*_curvedstream (plummer, hernquist), _general_orbitintegration | F | bk | ok | ok |
| impulse_deltav_general_curvedstream, _fullplummerintegration, _plummerstream, _plummerstream_curvedstream | F | ISLAND | RAISE | RAISE |
| fardal15spraydf, chen24spraydf (streamspraydf alias) | F | bk | ok (self-FD); **sample(tail='both') NOJIT + dt detached** (W9) | ok |
| streamdf (diffrax aA on jax; default C on torch) | F | ISLAND | sigv RAISE; sigangledAngle RAISE; **q ~3% off** (W5); pR, lamp ok (eager; jit TIMEOUT) | **pR WRONG 45×, q ZERO** (W1) |
| streamgapdf | F | ISLAND | GM, timpact ok (diffrax, eager; jit TIMEOUT); **higherorderTrack=True NotImplementedError** | GM, rs, impactb ok; **timpact WRONG 2×** (W1) |

## 3. Why the test suite didn't catch these

1. **Forced-backend classic tests compare values.**
   - A numpy island (AdiabaticContraction, interpRZ, MovingObject, the sphericaldf `__call__`, the diskdf moments, Orbit.analytic actions) and a gradient-dropping path (W1, W2) produce the *right numbers*.
   - The `numpy_island`/`_backend_ready` routing is designed to keep these green. That is "hollow green".
2. **Forced mode hides the data-mode constructor failures, and the reverse.**
   - Under force, Python-float coordinates are coerced, so R-NS never shows (toVertical, OblateStaeckel and CylSep pass FORCED and fail in data mode).
   - The `test_backend_*` gradient tests pass backend coordinates, which hides the forced-only failures: R-DISS lnLambda, R-DISK forced NOJIT, interpRZ silent detach.
   - No test runs the same (class, parameter) in both modes.
3. **Parameter-gradient coverage is hand-picked.**
   - What exists:
     - `test_backend_paramgrad.py`: Plummer and Isochrone {amp, b}, MN a, MN3 hz, Chandrasekhar rhm;
     - `test_backend_amp_grad.py`: **amp only**, 13 classes;
     - `test_backend_constructor_jit.py`: ExpTruncNFW rc, TwoPowerTriaxial alpha, constantbetaPowerLaw beta.
   - **No test differentiates** the axis ratios b and c, pa/zvec, the TwoPower alpha/beta, NFW conc, DehnenBar beta, SpiralArms alpha/H, the disk-expansion Sigma params, dissipative-force params, a wrapped potential's params, diskdf beta, constantbeta moments, the actionAngle delta/gamma, or any potential parameter through a C integrator on torch.
   - The actionAngle gradient tests differentiate w.r.t. phase-space coordinates only.
4. **Integrator and fixture choices sidestep the bugs.**
   - The stream tests always use `integrate_method="diffrax"` (jax), so the torch default C path (W1) is never differentiated w.r.t. parameters.
   - `test_traced_potential_parameter_routes_to_inbackend` is jax-only, and jax tracers *are* detected.
   - qdf tests use normalize=1 potentials, avoiding W4.
   - Tests feed 1-d arrays, so the 0-d crashes never appear.
5. **Assertions of the wrong kind.**
   - Value parity cannot see a wrong gradient (W3: values match to 1e-13).
   - Where gradients are tested, a missing graph only shows up as an exception if the test calls `.backward()` on the exact output. ZERO and DISC need an explicit grad-vs-FD comparison with `grad_fn is None` / `allow_unused` detection.
6. **The flag is not a differentiability signal.** `is_backend_compatible` is True for most R-ROT, R-SHAPE and R-DISK classes. A flag sweep (like the quick 45/46 scan) therefore reports green for exactly the classes that fail here.

## 4. Stacked-PR plan

Every PR below targets develop and lands as one GitHub stack. Files are disjoint between PRs unless a PR is marked "stacked on X (same file)".

- **Sizes:** S ≤ 1 day, M = 1–3 days, L > 3 days.
- **Testing:** every PR adds entries to, or deletes known-failure entries from, the sweep test (PR 0), plus a focused grad-vs-FD test in the class's `test_backend_<module>.py` at tight tolerance, in both jax and torch, in data and FORCED modes, and with jit for jax.

### Bottom of the stack: the sweep test and shared prerequisites

| PR | Scope (files) | Root cause | Fix approach (reuse) | Test | Size |
|---|---|---|---|---|---|
| **0. Parameter-gradient sweep** | `tests/test_backend_param_sweep.py` + `tests/backend_param_specs.py` (new) | n/a | Port `/var/tmp/gapaudit/audit/{harness,spec_*}.py`. Parametrize over (spec, param or one coordinate, backend ∈ {jax, torch}, mode ∈ {data, forced, floatcoords}) plus a jit axis for jax. Numpy Richardson FD is cached per spec; samplers use a same-backend self-FD. The assertion is grad-vs-FD at rtol 1e-6 with explicit ZERO, DISC (`grad_fn is None`), NOT_TENSOR and NaN detection. Includes a 0-d coordinate case. **Known failures** go in `KNOWN_GAPS = {(spec, param, backend, mode): "PR id: reason"}` as `xfail(strict=True)` | itself | M |
| **1. Parameter-aware namespace** | `backend/_input.py`, `backend/_namespaces.py`; hoist `_pot_grad_namespace` from `potential/Potential.py:3686` into `galpy.backend` (generic-helper rule) | R-NS: the namespace is resolved from the coordinates only | In `backend_input`, when the target holds a backend-array parameter (`_pot_grad_namespace(target, any_backend=True)`, cached per instance), lift numpy/Python coordinates into that namespace via `namespace_from_arrays` + `coerce_coords`, anchored with `device_of(param)`. The numpy path stays byte-identical (no backend parameter means no lift). This fixes every public-call internal site at once: `verticalPotential.py:59`, `Orbits.py:4268`, `isotropicNFWdf.py:59`, `isotropicPowerLawdf.py:101`, the CylSep and OblateStaeckel evaluations, and the whole FC column | the FC sweep column; toVertical, actionAngleVertical, Adiabatic a/b, Orbit.analytic a/b, NFW/PowerLaw sphericaldf (data mode) | M |
| **2. Backend rotation helpers** | `util/__init__.py` (`_rotate_to_arbitrary_vector` :169, `to_host` :209) | numpy rotation matrices from parameters | An xp-native rotation built with `get_namespace` / `as_backend_constant`; the numpy path stays byte-identical | unit grad-vs-FD of the rotation w.r.t. angle and zvec | S |
| **3. Backend `fixed_point`** | `backend/optimize.py` (+ `_jax`/`_torch` files per the separate-files rule) | scipy `fixed_point` in AdiabaticContraction | Concrete forward iteration plus an implicit-function-theorem VJP: a jax `custom_vjp` and a torch `autograd.Function`, mirroring how `brentq` is done there. Reuse `interp_linear` from `backend/interpolate.py` instead of `interp1d` | grad of a known fixed point vs analytic | M |

### Wrong-gradient PRs (first after the prerequisites)

| PR | Scope | Root cause | Fix | Test | Size |
|---|---|---|---|---|---|
| **4. W1: C-integrator parameter gradients on torch** | `orbit/Orbits.py` (`_pot_has_traced_param` :364-395), `backend/_torch/orbit_stm*` | `under_trace` misses an eager torch requires_grad tensor, so the C-STM drops parameter, time-grid and second-order sensitivities | Use `requires_backend_grad`/`_pot_grad_namespace` (eager-torch aware). Auto-route to the in-backend integrator (torchdiffeq/torchode), as jax already does to diffrax. **Raise** whenever a second-order or parameter gradient would go through the C-STM. Also give the Python integrators and ias15_c the same documented refusal for a traced *parameter* (today it is a cryptic TAC) | torch d/da through each C method; torch streamdf d/dpR, d/dq and streamgapdf d/dtimpact vs self-FD; IsochroneApprox a,b | M |
| **5. W2: actionAngle C-Jacobian parameter gradients** | `actionAngle/actionAngleStaeckel.py`, `actionAngleAdiabatic.py`, `actionAngleStaeckelGrid.py`, `actionAngleAdiabaticGrid.py` | The C `custom_vjp` closes over the potential and `stop_gradient(delta)` | If `_pot_grad_namespace(pot)` or delta/gamma is differentiating, route to the c=False backend path (the ecczmax `delta_grad` branch pattern); otherwise keep C. Never return 0 or DISC. Depends on PR 1 (Adiabatic a,b) | Staeckel c=True and Grid delta,a,b; AdiabaticGrid a,b,gamma (jit and eager); qdf+Staeckel(c=True) d/da | M |
| **6. W3: Staeckel c=False frequency and angle gradients** | `actionAngle/actionAngleStaeckel.py` (stacked on 5, same file) | missing delta dependency in `_refine_tp` and/or `_staeckel_chi_profiles` | Bisect with the sweep: differentiate the turning points, then the chi profiles, w.r.t. delta separately. Likely an FD-Newton slope that should be analytic, or a delta-dependent mesh frozen as a constant | Or, Oz and the angles vs FD at rtol 1e-8 w.r.t. delta and R | M |
| **7. W4: qdf NaN gradients** | `df/quasiisothermaldf.py` | `kappa*jr*exp` evaluated on unbound nodes before the mask | `xp.where` guard on jr (dead-branch pattern) before the product | qdf with unnormalized Isochrone, all parameters | S |
| **8. W5 + streamdf raises** | `df/streamdf.py` | frozen `real_eig` eigenvectors; `numpy.log(sigangle)` :596; `sigangledAngle` `numpy.amax/sqrt` :3800 | Differentiable eigenvectors (unfreeze, or degenerate-safe custom eig JVP); xp log/amax | d/dq rtol 1e-6, d/dsigv, sigangledAngle | M |
| **9. W6: interpRZPotential backend-native** | `potential/interpRZPotential.py` | island by flag; numpy grid build | Build the grids by evaluating the (now parameter-aware, PR 1) potential on backend nodes, fit with `native_rect_cubic_coeffs`/`eval_rect_ppoly`, fix the 3.3% grid-edge divergence (prefilter/eval mode pairing, see memory), and flip the flag. Then drop interpRZ from `_backend_ready`'s docstring | value parity at the grid edge (R,z)=(0.5,0); mna/mnb grads in all modes | M-L |
| **10. W7 + R-COORDS** | `util/coords.py` | B1950 without E-terms; `Rz_to_lambdanu` numpy; `vxvyvz_to_galcencyl` mixed namespaces | Add the FK4 E-terms on the backend path (or document and match astropy); xp in `Rz_to_lambdanu`; lift `vsun` into Zsun's namespace | radec_to_lb 1950 parity vs astropy; Delta, Zsun grads | S |
| **11. W9: streamspraydf both tails** | `df/streamspraydf.py:240` | `numpy.concatenate` on backend `dt` | `get_namespace(dt_l).concatenate` (same as the `out` arm) | `sample(tail='both')` under jit; d(dt)/dθ | S |
| **12. W8 + kingdf NOJIT** | `potential/KingPotential.py`, `df/kingdf.py` | ODE-solve tolerance (1.1e-6) and `if newM != M` (:49) | Tighten the backend ODE tolerance or confirm against a high-precision FD; use a concrete-safe unit check | W0 grad at rtol 1e-8 vs mpmath/tight FD; M under jit | S |

### Raises, islands and NOJIT

| PR | Scope | Root cause | Fix | Size |
|---|---|---|---|---|
| **13. R-ROT** (on 2) | `EllipsoidalPotential.py`, `FerrersPotential.py`, `RotateAndTiltWrapperPotential.py` | numpy rotation and `numpy.fabs(b-1)` axisymmetry checks | Rotation via PR 2. `isNonAxi` via `concretely_true` (with a static attribute fallback under trace). Under trace, keep the general triaxial path (never the aligned shortcut) | M |
| **14. R-SHAPE spherical** | `TwoPowerSphericalPotential.py`, `PowerSphericalPotential.py`, `PowerTriaxialPotential.py`, `FlattenedPowerPotential.py`, `ExpTruncNFWPotential.py` | `round(alpha/beta)` special-case dispatch; `if alpha != 3`; `numpy.log(conc)`; the `newmass != mass` bool | Dispatch only when `has_concrete_truth_value` (else the general closed form, as TwoPowerTriaxial alpha already does in `test_backend_constructor_jit`); xp log | M |
| **15. R-SHAPE non-axisymmetric** | `DehnenBarPotential.py`, `SteadyLogSpiralPotential.py`, `SpiralArmsPotential.py` | `numpy.sqrt(beta)`; `if omega != 0`; `numpy.sin/tan(alpha)`; numpy `_ns0*H` | xp math; `xp.where` for the omega=0 period | S-M |
| **16. normalize= under jit** | `Potential.py` (`normalize`) + the `if normalize or (...)` line in each constructor (stacked after 14/15 where files overlap) | Python truthiness of a traced normalize | `normalize is not False and normalize is not None` (structural, not value) | S |
| **17. R-DISS** | `DissipativeForce.py`, `ChandrasekharDynamicalFrictionForce.py`, `FDMDynamicalFrictionForce.py`, `NonInertialFrameForce.py` | flag unset, so no forced coercion and the numpy lnLambda path is taken; `to_host(sigmar)`; `numpy.atleast_1d(Omega)` | Set `_backend_compatible`; dispatch lnLambda on the resolved namespace (the coordinates *and* the parameters); backend sigmar grid; xp atleast_1d | M |
| **18. R-DISK** | `KuijkenDubinskiDiskExpansionPotential.py`, `SCFPotential.py` (scf_compute_coeffs) | 2- vs 3-arg lambda arity on the backend path; closures take xp from R only; `like(_CC, …)` detaches a | Unify the closure arity; resolve xp from (R, h); keep a on the graph | M |
| **19. R-AC AdiabaticContraction** (on 1, 3) | `AdiabaticContractionWrapperPotential.py` | scipy fixed_point / interp1d / quad; numpy mass grid | Backend `fixed_point` (PR 3), `interp_linear`, `fixed_quad` for Phi0, mass grid on the backend; flip the flag. (Fix the ignored-amp numpy bug on main first, see section 6) | M-L |
| **20. R-MO MovingObjectPotential** | `MovingObjectPotential.py`, `Orbits.py` (:625 IC, :7447 `atleast_1d(t)`; stacked on 4 and 23) | numpy trajectory lookup; IC `to_host` | Backend dense-output spline of the in-backend-integrated orbit; keep the IC backend array (`_ic_backend`); flip the flag | L |
| **21. R-AA misc** | `actionAngleIsochroneInverse.py`, `actionAngleStaeckel_c.py`, `actionAngleSpherical.py`, `actionAngleAdiabaticGrid.py` (gamma), `actionAngle.py` (flag) | numpy sqrt/atleast_1d; 0-d coordinates (`rperi[:,None]`, `len(R)`); gamma bool | xp math; `atleast_1d` at entry plus a reshape on exit; set `_backend_compatible` on aA objects | M |
| **22. Orbit.analytic** | `Orbits.py` (stacked on 4) | actions exit as numpy.float64; 0-d `_delta[indx]` | `exit_cast` to the forced namespace; `atleast_1d(delta)` | S |
| **23. diskdf** | `diskdf.py`, `evolveddiskdf.py` | `numpy.sqrt(2/(1+beta))`, `if beta==0`; moments return numpy by design | xp + `branch_where`; replace `test_ssp_plain_scalar_stays_on_numpy` with forced coercion (a contract change, flag it) | M |
| **24. dehnendf correct=True** | `diskdf.py` (stacked on 23), `DFcorrection` | parameter-hash savefile + iterative numpy correction | Skip the cache under trace; backend iteration (fixed count) | L |
| **25. sphericaldf + constantbeta** | `sphericaldf.py`, `constantbetadf.py`, `constantbetaHernquistdf.py`, `constantbetaPowerLawdf` | `__call__` island (only the radius entry points coerce); scipy brentq float bracket (:386); `special.gamma(beta)` (:381); `as_numpy` (:366); `int(floor(1.5-beta))` (:500) | `@backend_input` on `__call__`; anchor the bracket on the potential namespace (`galpy.backend.optimize.brentq`); `galpy.backend.special.gamma`; `as_numpy_constant` for the integer index | M |
| **26. streamgapdf higherorderTrack + islands** | `df/streamgapdf.py` | `NotImplementedError` (hidden by `pragma: no cover`); density_par/meanOmega lack `@backend_input`; `approx=False` uses scipy quad | Port the higher-order moment recursion to `_density_par_approx_backend`/`_meanOmega_num_approx_backend`; remove the pragmas; `@backend_input`; GL quadrature for approx=False | M-L |
| **27. Impulse kicks** | `df/streamgapdf.py` impulse functions (stacked on 26) | scipy quad; in-place writes; Orbit + `as_numpy` | `fixed_quad`/`transformed_quad`; functional updates; in-backend orbit | L (splittable in 2) |
| **28. Small NOJIT and island leftovers** | `eddingtondf.py:127`, `jeans.py:282`, `AnyAxisymmetricRazorThinDiskPotential.py:221`, `actionAngleHarmonicInverse.py`, `actionAngleIsochroneApprox` (estimateBIsochrone), SolidBodyRotation/TimeDependentAmplitude wrapper flags | `bool()` on traced values; a Python int passed to a user callable; scalar exits; flags False without a reason (verify first) | `concretely_true`/`xp.where`; `scalar_like` before calling `surfdens`; `exit_cast` | S |

**Suggested stack order (bottom to top):** 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 20, 23, 24, 25, 26, 27, 28.

- PRs 4–12 (wrong gradients) sit directly on the prerequisites.
- Same-file pairs are adjacent: 5/6, 4/22/20, 23/24, 26/27.
- **Total:** about 7 S + 15 M + 5 L, roughly 6–8 weeks of focused work.
- **Natural cut:** 0–12, which removes every silent-wrong gradient, is about 2.5 weeks.

### Sweep test: land first with a known-failures list (recommended)

Land PR 0 at the bottom, with `KNOWN_GAPS` as strict xfails, rather than last:

- **Regression protection now.** Roughly two thirds of (class, parameter, mode) combinations already pass. Landing first protects them immediately, whereas landing last leaves 6–8 weeks of unguarded churn.
- **Fixes become measurable.** With `strict=True`, a fixed gap XPASSes and fails the run until its entry is deleted, so each PR proves its fix by shrinking the list. The remaining count is always checkable, the same reason "removing an exception hides the count".
- **New gaps cannot hide.** A new class or parameter that is not in the list must pass.

Practicalities:

- Numpy FDs are cached per spec.
- jit only runs on the jax rows.
- Heavy specs go into `backend_slow_skip.txt` with timings: King W0 (eager 270 s), Multipole jit, TwoPowerTriaxial jit, Staeckel c=False jit, streamdf/streamgapdf construction.
- Cost: the subprocess harness took ~90 CPU-minutes for potentials (2 backends × 3 modes), dominated by per-process imports and numpy FDs; in-process with cached FDs expect ~20–40 CPU-minutes per group. Measure in PR 0 and slow-skip outliers. Split across the existing backend shards with the per-module test-file convention: `test_backend_param_sweep_{potential,df,aa}.py`.
- Add a test that **every** public class in galpy.potential, galpy.df and galpy.actionAngle has a spec, so coverage can't silently erode.

## 5. Direct answer for Jo

- **Besides AdiabaticContraction and Torus, the gaps that return wrong numbers without complaint are:**
  - torch parameter gradients through every C integrator, which also makes torch streamdf and streamgapdf gradients wrong (45×, 0, 2×);
  - actionAngle C-Jacobian paths that zero or drop delta and potential-parameter gradients (Staeckel c=True, StaeckelGrid, AdiabaticGrid, and qdf on top);
  - Staeckel c=False d/ddelta of frequencies and angles, wrong at the 1e-3–1e-2 level;
  - qdf NaN gradients for unnormalized potentials;
  - streamdf d/dq about 3% off;
  - interpRZ silently detaching under a forced backend.
- **The broad "raises" class:** most non-amp shape parameters of the triaxial, TwoPower and bar/spiral families; NFW conc; dissipative forces under force; disk expansions; MovingObject; diskdf beta; constantbeta; the impulse kicks; streamgapdf higherorderTrack.
- **One cross-cutting namespace bug:** a backend parameter evaluated at float coordinates. `jax.grad(lambda a: MN(a=a)(1.,.1))` raises.

## 6. Pre-existing numpy bugs (fix on main, per the numpy-fixes-go-to-main rule)

- **`AdiabaticContractionWrapperPotential` silently ignores `amp`.** `Force.__init__(self, ro=ro, vo=vo)` at :82 and `interpSphericalPotential.__init__` are called without amp, so the numpy FD w.r.t. amp is exactly 0.
- **qdf + `actionAngleAdiabatic(c=False)` + `_precomputerg=False`** with an unnormalized MN potential raises "cannot unpack non-iterable" in `_vmomentdensity` (`quasiisothermaldf.py:878`).

Artifacts (scratch, not committed): `/var/tmp/gapaudit/audit/` contains `res_pot*.json`, `res_df*.json`, `res_aa*.json`, `fc_pot.txt`, the spec files and the analysis scripts.
