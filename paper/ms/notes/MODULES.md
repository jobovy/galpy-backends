# galpy jax/PyTorch backends: per-module notes for the methods section

Sources as in ARCHITECTURE.md (galpy `feat/backends` 9f6c9d6f7 at `/var/tmp/fbp`;
design notes in `/home/bovy/Repos/galpy-jax`). Examples are condensed from the cited
tests; imports omitted. **[unconfirmed]** marks what I could not verify from source.
PR numbers come from `git log` / README rows; the big landing PRs per area are listed
at the end of each section.

Legend for "gradients": IC = orbit initial conditions; theta = potential parameters;
DF = distribution-function parameters; t = time.

## 1. Potentials

**What works.** Every potential, force and wrapper in `galpy.potential` has a
namespace-swapped private compute layer (`_evaluate/_Rforce/_zforce/_phitorque/
_dens/_surfdens/_R2deriv/...`; spherical family via `SphericalPotential`), Track A
complete 2026-06-11 (README Status 2026-06-18; P2.1-P2.7, #893-#946, #959-#963).
Public entry points carry `@backend_input("R","z","phi","t")` (73 sites, #1196).
Special-function-backed potentials (SCF, Multipole, DiskSCF/DiskMultipole, Einasto,
PowerSphericalwCutoff, DoubleExponentialDisk, King via `interpSphericalPotential`)
go through `galpy.backend.special` and the PPoly spline evaluators (#927-#932, #942).
Convenience functions (`vcirc`, `epifreq`, `rl`, `rE`, `LcE`, `zvc`, `mass`, `vesc`,
`rtide`, `ttensor`, `tdyn`) are backend-aware, root-finds via `galpy.backend.optimize`
and `mass()` via GL quadrature (#975). `from_density` for SCF (#1612) and Multipole
(#1618) is differentiable. Measured (`plan/differentiability_audit_2026-09-16.md`):
"49 of 50 instantiable potentials differentiable w.r.t. `amp` under both `grad` and
`jit`; only `DiskSCFPotential` fails under `jit`".

**Gradients.** d/d(coordinates) and d/d(theta) for every migrated potential, by AD
through the namespace-swapped code; `conversion.parse_*` passes a traced parameter
through unscaled (`galpy/util/conversion.py:542-553`). Checked grad-vs-FD with
Richardson-extrapolated references at rtol 1e-9 (`tests/test_backend_paramgrad.py`),
and the identities dPhi/dR == -Rforce, d(-Rforce)/dR == R2deriv under AD
(`tests/test_backend.py:9-16`).

**Not supported / caveats.**
- `SnapshotRZPotential`/`InterpSnapshotRZPotential` (pynbody) have no backend code
  (grep: no `get_namespace`); `amuse.py` likewise. **[unconfirmed]** whether they are
  documented as out of scope anywhere.
- `interpRZPotential` and `MovingObjectPotential` are the two names behind the
  temporary `_backend_ready` gate (`galpy/backend/_input.py:59-82`), but both now have
  backend test modules (`tests/test_backend_interprz.py`: same tensor-product PPoly
  evaluated natively, ~1 ulp from scipy; `tests/test_backend_movingobject.py`: object
  orbit stays numpy-integrated, field coordinates may be backend). Status of the gate
  for them: **[unconfirmed]**.
- `AdiabaticContractionWrapperPotential` sets `_backend_compatible = False`
  (`galpy/potential/AdiabaticContractionWrapperPotential.py:133`).
- Dissipative forces (`ChandrasekharDynamicalFrictionForce`, `FDMDynamicalFrictionForce`,
  `NonInertialFrameForce`) use `get_namespace`/`coerce_coords` internally but do not
  set `_backend_compatible` (grep), so they coerce themselves rather than via the
  boundary. **[unconfirmed]** by execution.
- `AnyAxisymmetricRazorThinDiskPotential` and `DoubleExponentialDiskPotential` opt in
  to node-array inputs with `_backend_accepts_arrays` (for vectorised traced
  quadratures; `galpy/potential/Potential.py:83-97`); the numpy scalars-only contract
  is unchanged. `AnySphericalPotential`'s array opt-out is deliberate.
- Python branches on a stored parameter (`MiyamotoNagai` `if self._a == 0.0`,
  Chandrasekhar `rhm == 0`, `MN3ExponentialDisk` `brd < 0`) work under `grad` but
  break inside a jitted ODE solve (audit S D). **[unconfirmed]** whether fixed since.
- Fixed-order GL replaces adaptive `quad` for mass/surface density on the backends:
  "differentiable APPROXIMATION, not adaptive" (`galpy/backend/quadrature.py:36-38`).

**Examples** (from `tests/test_backend_paramgrad.py:177-205, 234-251`;
`tests/test_backend_jit_mode.py:69-82`; `tests/test_backend_device.py:139-160`):

```python
# jax.grad w.r.t. a potential parameter, under jit and vmap over a batch of b values
R, z = jnp.asarray(1.2), jnp.asarray(0.3)
def phi_of_b(b):
    return PlummerPotential(amp=1.0, b=b)._evaluate(R, z)
grads = jax.vmap(jax.jit(jax.grad(phi_of_b)))(jnp.asarray([0.5, 0.7, 1.0]))
```

```python
# the opt-in global trace mode (the test suite's --jit); user code normally jits itself
pot = MiyamotoNagaiPotential(normalize=1.0)
with galpy.backend.jit("jax"):
    fr = pot.Rforce(jnp.asarray(1.1), jnp.asarray(0.2))   # jax.jit at the boundary
```

```python
# GPU placement follows the data: scalar R, z plus a CUDA phi return on CUDA
phi = torch.tensor([0.1, 0.2, 0.3], device="cuda")
out = scf(1.5, 0.3, phi=phi)                       # out.device.type == "cuda"
fr = evaluateRforces(scf, 1.5, 0.3, phi=phi)
```

## 2. Orbits

### Integration paths

`Orbit.integrate` (`galpy/orbit/Orbits.py:2604`, default `method="symplec4_c"`
unchanged) chooses between three mechanisms when the Orbit was built from a jax/torch
initial condition (`self._ic_backend`, :512-543) or the potential carries a backend
parameter:

1. **In-backend ODE** (`method='diffrax'` for jax; `'torchdiffeq'` or `'torchode'` for
   torch; Orbits.py:1894, 2340-2500; `galpy/backend/_reference/inbackend_ode.py`,
   `_jax/orbit_ode.py`, `_torch/orbit_ode.py`). The equations of motion are evaluated
   through the decorator-free force layer in rectangular variables (matching
   `integrateFullOrbit.c`) and integrated by the backend's own solver (diffrax Dopri8,
   torchdiffeq dopri5, torchode dopri5/tsit5), default rtol=atol=1e-12
   (`inbackend_ode.py:150-275`). Gradients w.r.t. **IC and theta** fall out of AD; jax
   second derivatives need `inbackend_kwargs={'adjoint': 'direct', 'max_steps': 4096}`
   (Orbits.py:2647; `_jax/orbit_ode.py:26-46`). N orbits integrate in ONE solve on a
   native batch axis (#1027); per-orbit time grids are supported (`ts` shape `(N, nt)`).
   torchode is the `torch.compile`-able engine (#1569); torchdiffeq is not.
2. **C-STM bridge** (`galpy/backend/_reference/inbackend_stm.py`, `_jax/orbit_stm.py`,
   `_torch/orbit_stm.py`; #972/#973/#1003/#1028/#1103). The forward is galpy's compiled
   C variational integrator, which returns the trajectory and the state-transition
   matrix M(t)=dx(t)/dx(0); the custom VJP applies M^T to the cotangent ("a single
   matrix-transpose-vector product, no backward C call", `inbackend_stm.py:1-22`). For
   6D orbits with an RK method the augmented 42-state solve computes all six columns
   at once (`_c_stm_forward_augmented`, :90; #1028); symplectic and lower-dimensional
   cases propagate the six basis deviation vectors per column (:157). **IC gradients
   only**: "potential-parameter gradients use the in-backend ODE path (the C integrator
   carries no d x / d theta sensitivity)" (:20-22). First order only (the VJP is opaque
   to a second differentiation; `galpy_backend_autodiff_discussion.md` S5).

```python
# galpy/backend/_jax/orbit_stm.py:48-62 (abridged)
@jax.custom_vjp
def _stm(v):
    xt, _ = _call(v)            # jax.pure_callback into the C variational integrator
    return xt
def _fwd(v):
    xt, M = _call(v)
    return xt, M                # residual = M(t)
def _bwd(M, ct):                # IC cotangent = sum_t M(t)^T ct[t]
    ...
```

3. **numpy/C with a warning**: a *concrete* backend IC and a method that is not
   routed integrates in C and returns a backend array that is "not differentiable"
   (Orbits.py:1987-2001).

Routing rules (Orbits.py:1911-1986; `inbackend_stm.py:32-34`):
`_C_RK_METHODS = ("rk4_c","rk6_c","dopr54_c","dop853_c")`, `_C_SYMPLEC_METHODS =
("leapfrog_c","symplec4_c","symplec6_c")`. A backend IC with an RK C method routes to
the C-STM when `_check_c(pot)` holds, `t` is not traced, no potential parameter is
traced (#1503) and the potential has the (3D) C Hessian (`hasC_dxdv3d`/`hasC_dxdv`);
symplectic methods route only for a grad-tracking IC (#1103); otherwise the call falls
back to `inbackend_ode_method(xp)` (`'diffrax'`; torch: `'torchode'` if installed else
`'torchdiffeq'`, `galpy/backend/_namespaces.py:823-831`). The public `integrate`
docstring lists the new methods (Orbits.py:2669) but not the automatic C-STM routing
(only the private `_integrate_cstm` docstring does, and its method list is stale:
it still says 6D/RK-only).

Accessors (`o.R(t)`, `o.x(t)`, `o.E()`, `o.e()`, `rperi/rap/zmax`) are backend-aware;
off-grid times interpolate with `galpy.backend.interpolate.Spline1D` (#974, #1001,
#1027). `Orbit.rperi/rap/zmax(analytic=True)` through the backend Spherical/Staeckel
code (#1594 + "Next" in `plan/trackE_actionangle.md`) **[unconfirmed as landed]**.

### Variational equations (dxdv) and Lyapunov exponents (numpy/C, on main)

The 3D variational equations are Track B, merged to main as #891 (2026-06-13;
HISTORY.txt v1.12.0 :247-259). Where the math lives:
- Python RHS: `_EOM_dxdv` (`galpy/orbit/integrateFullOrbit.py:1758`, 12D rectangular
  state), `_planarEOM_dxdv` (`integratePlanarOrbit.py:1979`), `_linearEOM_dxdv`
  (`integrateLinearOrbit.py:597`).
- C RHS: `evalRectDeriv_dxdv` (`galpy/orbit/orbit_c_ext/integrateFullOrbit.c:1607`),
  the augmented `evalRectDeriv_stm`/`integrateFullOrbit_stm` (:1794/:1360), planar
  (`integratePlanarOrbit.c:970`) and 1D (`integrateLinearOrbit.c:250`); symplectic
  tangent maps `leapfrog_dxdv/symplec4_dxdv/symplec6_dxdv`
  (`galpy/util/bovy_symplecticode.c:1053/1144/1269`, #1086). The Jacobian K is the
  Cartesian Hessian -grad grad Phi built from the new C `z2deriv/Rzderiv/zphideriv`
  pointers in `struct potentialArg` (`plan/trackB_variational.md` "The math"), extended
  to the velocity-coupled non-symmetric J of dissipative forces (#933; HISTORY :261).
  Derivation for the symplectic case: `plan/symplectic_variational_derivation.md`.
- Supported integrators for `integrate_dxdv` (Orbits.py:2927, docstring :2981-2995):
  odeint, rk4_c, rk6_c, dopr54_c, dop853, dop853_c, leapfrog_c, symplec4_c, symplec6_c;
  pure-Python leapfrog and ias15_c rejected.
- Checks (the "test battery", `plan/trackB_variational.md`): Liouville det M = 1 and
  symplecticity M^T Omega M = Omega in canonical Cartesian coordinates
  (`tests/test_orbit.py:863` `test_liouville_3d`; `tests/test_symplectic_dxdv.py:239`),
  closed-form STMs (3D harmonic, Kepler), flow-direction M f(x0) = f(x(t)),
  conserved-integral relations, column-wise finite differences of the flow (the only
  valid check for dissipative forces and for the cylindrical-frame backend STM,
  `tests/test_backend_orbit_stm.py:572, 645`).
- `Orbit.lyapunov(ts, pot, method="dop853_c", renorm_every=10, spectrum=False, ...)`
  (Orbits.py:3151; #926, spectrum #945, #1458): Benettin renormalisation on the C
  variational integrators, 4D and 6D orbits; Henon-Heiles chaotic lambda=0.044-0.051
  vs literature 0.04-0.08 (README row Pvar-lyapunov).

### What is NOT supported

- theta-gradients through the C integrators (by construction; use the in-backend ODE).
- Second derivatives through the C-STM; through diffrax only with `adjoint='direct'`.
- Differentiating a diffrax solve under `jax.vmap` with adaptive steps gives a ~7% wrong
  gradient (DirectAdjoint/adaptive interaction); constant steps (`nsteps=`,
  `_shared_step_aA`) fix it (#1602; `inbackend_ode.py:199-202`).
- torchdiffeq has no vmap rule; torch batches by a Python loop.
- Pure-Python integrators (`odeint`, `leapfrog`, `dop853`) run eagerly on the backend
  and are slow (the remaining FDM `central_limit` slow-skip, `tests/backend_slow_skip.txt`).

### Examples (`tests/test_backend_orbit_integrate.py:200, 235, 763`;
`tests/test_backend_paramgrad.py:234-251`)

```python
# jax: d(final R)/d(Plummer b) through a differentiable orbit integration
def final_R(b):
    o = Orbit(jnp.asarray(ic))
    o.integrate(jnp.asarray(ts), PlummerPotential(amp=1.0, b=b), method="diffrax")
    return o.getOrbit()[-1, 0]
g = jax.grad(final_R)(0.6)
```

```python
# jax.vmap over initial conditions (each IC its own Orbit)
batched = jax.vmap(final_R_of_ic)(ics)          # ics.shape == (N, 6)
```

```python
# torch autograd through torchdiffeq w.r.t. a parameter
b = torch.tensor(0.6, requires_grad=True)
o = Orbit(torch.as_tensor(ic))
o.integrate(torch.as_tensor(ts), PlummerPotential(amp=1.0, b=b), method="torchdiffeq")
o.getOrbit()[-1, 0].backward();  b.grad
```

```python
# fast C integration with IC gradients: a grad-tracking jax IC and a C method
J = jax.jacrev(lambda v: _integrate_c("jax", pot, v, ts, "dop853_c"))(jnp.asarray(ic))
# equals the C state-transition matrix M(t_end) (test_backend_orbit_stm.py:125)
```

PRs: in-backend ODE #906, wiring #973, batch #1027, planar/1D #1075/#1079, C-STM
#972/#1003/#1024/#1028/#1086/#1091/#1094/#1103/#1503, torchode #1569, traced-parameter
routing #1503, vmap/constant steps #1602; variational3d #891 (+#898-#952), Lyapunov
#926/#945.

## 3. Action-angle

Everything except Torus runs on the backends; the public varargs API dispatches with
`get_namespace`/`promote_scalars` inside (no `@backend_input`, see ARCHITECTURE S3).
No actionAngle class sets `_backend_compatible` (agent grep).

**actionAngleStaeckel.** Two paths:
- `c=False`: vectorised, identical code for numpy/jax/torch. Turning points by
  `bisect_root` (xtol 1e-13, maxiter 200) plus one Newton refinement for the gradient
  (`galpy/actionAngle/actionAngleStaeckel.py:162, 195-266`); actions by the composite
  chi-anomaly rule (`leggauss(10)` panels, nchi = max(2 x order, 20), :1994, :335) that
  "main's pure-Python path uses (gh#1357) and the one C is regularized to, so c=True
  and c=False agree to machine precision" (:306-333). Gradients w.r.t. (x,v) and theta
  by AD; the action gradient is grafted from a t^2-substituted donor quadrature
  ("Value unchanged (C parity), first-order only", :472-495, #1047). Two stale comments
  still describe plain GL matching gsl_glfixed (:110-118, :1196-1200).
- `c=True` on a backend: the C code computes the actions AND the Jacobian
  d(jr,jz)/d(R,vR,vT,z,vz) natively (2x5; 5x5 with frequencies; angles too; 4x5 for
  EccZmaxRperiRap), wrapped as `jax.custom_vjp`/`torch.autograd.Function`
  (`galpy/backend/_jax/staeckel_c.py:1-10`, `_torch/staeckel_c.py`; #1051, #1060,
  #1063, #1068). **IC gradients only, first order; no theta gradients** (audit S C:
  "differentiates w.r.t. the INITIAL CONDITIONS ... not w.r.t. potential parameters").
  `delta` is a fixed reference without gradient (:769-778); d/d(delta) by central
  difference in EccZmax (:825-828).
- `estimateDeltaStaeckel` runs on jax/torch (:1999-2045, #1067); jit-traceable (#1016;
  the eager `if xp.any(unbound)` raise became traceable; GPU anchors +10 sites).
- `actionAngleStaeckelGrid`/`AdiabaticGrid`: under a forced backend the grid is built
  natively (GPU-resident tables, differentiable through to the query); a potential that
  carries a gradient delegates to the exact class with a warning (#1538, #1022/#1182/
  #1187/#1399).

**actionAngleAdiabatic.** `c=True` on a backend uses the fused C 2x5 Jacobian
(`_jax/adiabatic_c.py`, "order=20 matches the numpy c=True path, whose C hardcodes
20"); `c=False` delegates the radial part to the migrated `actionAngleSpherical` and the
vertical part to `actionAngleVertical`'s GL/root-find machinery (:593-601; #1018,
#1070, #1072). Frequencies/angles always take the Python path.

**actionAngleSpherical / Isochrone / Harmonic / Vertical / *Inverse.** Namespace-swapped
closed forms (Isochrone, Harmonic, #1005/#1007); Spherical with backend root-finds and
`_BACKEND_GL_ORDER = 50` quadrature (#1008/#1009; relative radial problem, circular and
near-circular orbits #1527/#1553/#1594); Vertical with backend `brentq` for xmax
(#1012/#1209/#1273). theta-gradients: Staeckel(c=False) 1.7e-10, Adiabatic 1.9e-10,
Spherical 2.7e-11 relative vs rebuild-FD (audit S A).

**actionAngleIsochroneApprox.** Orbits integrated by the in-backend path
(`integrate_method='diffrax'/'torchdiffeq'`), GPU and twice differentiable with
`integrate_kwargs={'adjoint': 'direct', ...}` (:64-77, :1022-1029; #1020/#1025/#1029).

**actionAngleVerticalInverse** (on the backends since 2026-10-04; #1627-#1631).
The default momentum-matched map is built and evaluated natively: x_max by backend
`brentq`; spectral sweep (FFT antiderivative); anomaly map by batched Gauss-Newton
without gradient plus ONE differentiable IFT step; Hermite-spline tables evaluated with
`eval_ppoly`; angle -> anomaly by backend `brentq` on the monotone angle relation. A
catastrophic cancellation in dv/dtheta near tau = pi was removed by a cos-series
rewrite (`plan/trackE_actionangle.md` 2026-10-06). Differentiable in J, theta and the
potential's parameters; the whole pipeline traces under `jax.jit` and `torch.compile`;
runs on CUDA (#1631). Legacy modes (`momentum_matched=False`, point transforms)
evaluate natively but are constructed on numpy, so they cannot take a potential with
backend parameters (`actionAngleVerticalInverse.py:213-220`, `_LEGACY_BACKEND_MSG`).

**actionAngleTorus: out of scope, permanently.** Wraps the external C++ Torus library;
`_reject_backend` (`galpy/actionAngle/actionAngleTorus.py:29-38`); `test_actionAngleTorus.py`
excluded from the all-backend CI and the four streamdf `useTM` tests are permanent skips
(`tests/backend_skip.txt`).

**Examples** (`tests/test_backend_staeckel_grad.py:202-240`;
`tests/test_backend_actionAngleVerticalInverse.py:224, 341`):

```python
# jax.jit + jax.grad of Staeckel actions (c=True: C-native Jacobian in the VJP)
aAS = actionAngleStaeckel(pot=MWPotential2014, delta=0.45, c=True)
def djr_dR(R):
    return jnp.sum(aAS(R, vR, vT, z, vz)[0])
g = jax.jit(jax.grad(djr_dR))(jnp.asarray([1.1]))
```

```python
# torch autograd through the same public API
args = [torch.tensor([x], requires_grad=True) for x in (R, vR, vT, z, vz)]
aAS(*args)[0].sum().backward()
```

```python
# theta-gradient needs the Python path (c=False)
def jr_of_a(a):
    aA = actionAngleStaeckel(pot=MiyamotoNagaiPotential(a=a, b=0.3), delta=0.4, c=False)
    return aA(R, vR, vT, z, vz)[0].sum()
jax.grad(jr_of_a)(jnp.asarray(0.5))
```

PRs: Staeckel #1013-#1016, #1038, #1040, #1043, #1047, #1051, #1059-#1068, #1074,
#1493; Adiabatic #1018, #1070, #1072; Spherical #1008/#1009/#1527/#1594; Vertical
#1012/#1209/#1273; IsochroneApprox #1020/#1025/#1029; grids #1022/#1182/#1187/#1399/
#1538; VerticalInverse #1627/#1628/#1630/#1631; shared GL order #1251.

## 4. Distribution functions

The consistent pattern (audit S B): DF *evaluation* and *sampling* are differentiable
nearly everywhere; DF *construction* is differentiable for streamdf, streamspraydf,
constantbetadf, eddingtondf, kingdf (W0, M, rt) and the VerticalInverse tables, and
still numpy for quasiisothermaldf, streamgapdf and the closed-form spherical DFs'
`dMdE`. Sampling dispatches on an explicit `key`: `key=None` keeps the historical
numpy output (and global numpy RNG); a backend key returns backend arrays that are
differentiable in the DF and potential parameters (reparameterised inverse-CDF).

### Spherical DFs (`sphericaldf.py` + Eddington/isotropic/King/constant-beta/Osipkov-Merritt)

- `sphericaldf._backend_compatible = True` (:392). Radius sampling by `interp_linear` on
  the cumulative-mass grid (differentiable in the CDF knots, :1023-1060) or a closed-form
  `_icmf`; velocity sampling by a dual-path p(v|r) inverse-CDF table (frozen scipy
  `RectBivariateSpline` for numpy, `interp_bilinear` natively, :214, :1514);
  r(Phi) by backend `brentq` with IFT gradients (`_RphiRootFind`, :320). The f(E) table
  (`_ensure_fE_interp`, :558) is kept on the backend when the potential is differentiated
  ("frozen knots miss their motion with the potential, ~7% off on p(v|r)").
- **eddingtondf**: backend f(E) by fixed GL (`_QUAD_N_FE = 100`) over three
  substitutions, rphi refined with 3 Newton steps (:258-330); exact near Emin (#1586
  main / #1578). **constantbetadf**: f(E) by fixed GL, differentiable through the
  integrand, no calibrated lower limit on the backend (#1598); eta sampling rebuilt with
  backend `hyp2f1`/`gamma` so d(eta)/d(beta) flows (:232); construction, evaluation and
  sampling under `jax.jit` (#1558; `rmin` must be explicit under jit); ~4x faster
  construction, O(n log n) spline solves (#1557). The numpy f(E) path is the one
  deliberate byte-identity exception (#1089). **osipkovmerrittdf**: f(Q) table on the
  backend when differentiated; `sample` under `jax.jit` (#1598; log-r bisection for
  r(Phi)); differentiable velocity sampling w.r.t. the potential (#1556).
  **kingdf**: the Poisson ODE is solved with scipy eagerly (`@untraceable_setup`), with
  d/dW0 grafted from forward sensitivities (`graft_derivative`, #1551/#1572); under
  `jax.jit`/`torch.compile` the same ODE is solved in-backend with diffrax
  (DirectAdjoint, for d2/dW0^2) / torchode (`_jax/king_ode.py`, `_torch/king_ode.py`;
  #1566). Gradients w.r.t. W0, M, rt.
- Gradients measured: sampled r and v vs FD to 4.3e-6 / 7.7e-5; constant-beta Hernquist
  d r_i/da = r_i/a to 1e-13 (#1547); King W0 tables < 1e-7 (#1551) (`STATUS.md`).
- Not differentiable: `dMdE` of the closed-form Hernquist/NFW DFs (TracerArrayConversion,
  audit S B); numpy key + grad-carrying potential raises `NotImplementedError` in
  constantbetadf (:387-404). Known sampler bias: the n_v_vesc=100 grid biases sigma_r
  ~0.7% low (gh#1343, pre-existing). jax eager f(E) for constantbetadf stays numpy by
  decision ("outside jit is faster", `STATUS.md`).

```python
# torch: gradient of sampled radii/speeds w.r.t. the Hernquist scale a
# (tests/test_backend_sphericaldf.py:1146-1200)
a = torch.tensor(1.0, requires_grad=True)
with galpy.backend.use("torch", force=True):
    d = constantbetadf(pot=HernquistPotential(amp=2.0, a=a), beta=-0.2)
    o = d.sample(n=4, rmin=0.1, key=grandom.key(7, backend="torch"))
torch.autograd.grad(o.r()[0], a)
```

```python
# jax: d/dW0 of King-model quantities, eager (graft) and under jit (diffrax)
# (tests/test_backend_kingdf.py:331-360, 398)
def sigr(W0):
    with galpy.backend.use("jax", force=True):
        return kingdf(W0=W0, M=2.3, rt=1.4).sigmar(jnp.asarray(0.5))
jax.grad(sigr)(3.0);   jax.jit(jax.grad(sigr))(3.0)
```

### quasiisothermaldf

Evaluation and moments are backend-native and differentiable w.r.t. potential
parameters (#1537) and the DF scale lengths (#1529); the constructor compiles under
`torch.compile` (#1617) but is not jax-traceable (audit S B: `TracerArrayConversionError`).
`sampleV` is an inverse-CDF sampler on one 3-D velocity mesh (vT -> vR|vT -> vz|vR,vT;
`quasiisothermaldf.py:2270-2383`, #1443/#1444), differentiable with a backend key
(`tests/test_backend_qdf.py:386`); `sampleV_interpolate` is still rejection sampling with
numpy proposals, vectorised and bucket-padded on the backend (#1621), not differentiable.
Still numpy: `_setup_rg_table` (`@untraceable_setup`), the guiding-radius `brentq` in the
constructor, the non-fixed-order surface-mass path. Plan on record
(`memory: qdf-inverse-cdf-sampler-plan`, `qdf-samplev-modefinding-boundary-peaks`):
inverse-CDF for a single (R,z), rejection for many; spurious vT-boundary peaks need a
local mode finder.

```python
# d(sampled vR^2)/d(hsr) with a jax key (tests/test_backend_qdf.py:386-420)
def total(hsr):
    qdf._hsr = hsr
    v = qdf.sampleV(0.9, 0.05, n=6, key=grandom.key(0), use_physical=False)
    return (v[:, 0] ** 2).sum()
with galpy.backend.use("jax", force=True):
    g = jax.grad(total)(jnp.asarray(1.0))
```

### diskdf / evolveddiskdf

dehnendf/shudf evaluation (`_eval_backend`, byte-matching the numpy scimath real part)
and moments by `nested_quad` (n=50) are backend-native (#1073); `surfacemass` d/d(hr)
6.7e-10 (audit). `evolveddiskdf` moments differentiate through the orbit evolution via
one vectorised multi-orbit in-backend integrate (`_buildvgrid_backend`, #1076). Not
migrated: `correct=True` (DFcorrection), the scipy `brentq`/`fmin_bfgs` sites, and the
rejection samplers (`numpy.random`). The surface-density profile boundary is guarded by a
single `_ssp` accessor after the island fix (#1261/#1268).

### streamdf (Track F "last hard df")

- Track computation on the backends: `_determine_stream_track_backend`
  (`galpy/df/streamdf.py:1684`) is "pure, `jax.vmap`-batched over the chunks ...
  differentiable end-to-end"; gradients flow to theta (through the diffrax/torchdiffeq
  auxiliary integration and the AD `calcaAJac` = `galpy.backend.jacobian`, replacing the
  numpy finite-difference Jacobian) and to the progenitor IC. The frequency-covariance
  eigendecomposition uses `real_eig(..., freeze_vectors=False)` after the unfreeze that
  fixed a 70% gradient error (#1584; `plan/differentiability_audit_2026-09-16.md`).
  `nTrackIterations` is a structural integer and raises under a trace (:1454-1466).
  torch batches the chunks with a Python loop (no vmap through the torchdiffeq custom
  Function); constant-step `_shared_step_aA` avoids the adaptive-step vmap defect (#1602).
- Construction: `__init__` differentiable w.r.t. a potential parameter (audit S A:
  d/dq relative ~1e-2 vs FD after the real_eig unfreeze); `jax.jit` of construction
  works (723 s, 1.1e-5 relative from eager) and torch progenitor construction, in the
  open PR #1637 (`STATUS.md`); `jax.grad` of the constructor "fails in diffrax adaptive
  backward (follow-up)".
- **Two kinds of gradient** (`memory: stream-track-two-gradients`; test headers):
  differentiating a track with the sampled phase-space points held fixed ("frozen-xv
  d/dtheta") is NON-PHYSICAL; the generative gradient lets the sample xv(theta) flow and
  is checked against h-converged finite differences (`tests/test_backend_streamdf.py`
  header; `tests/test_backend_streamspraydf.py:1220`). The grafted `_sample_aAt` is
  inverse-CDF, not ARS (:5080-5105).
- Misalignment: traceable in #1637 (open). `useTM=True` (Torus) is out of scope; the
  deprecated `calc_stream_lb` pipeline stays numpy (:1966).
- A ~10% gap to a full numpy-rebuild FD remains from the frozen frequency-covariance
  eigendecomposition (`tests/test_backend_streamdf.py:898` comment).

```python
# jax: d(track)/d(q) of a LogarithmicHalo stream (tests/test_backend_streamdf.py:874-898)
def loss(q):
    lp = LogarithmicHaloPotential(normalize=1.0, q=q)
    sdf._pot = lp; sdf._aA = actionAngleIsochroneApprox(pot=lp, b=0.8, integrate_method="diffrax")
    sdf._progenitor = Orbit(jnp.asarray(prog_vxvv))
    sdf._determine_stream_track_backend()
    return jnp.sum(W * sdf._ObsTrack)
jax.grad(loss)(jnp.asarray(0.9))
```

### streamspraydf

Fully jittable and differentiable (#1106): `fardal15spraydf(...).sample()` and
`.streamTrack()` run under `jax.jit` (CPU and GPU) and give the generative
d(track)/d(theta); the progenitor switches to the in-backend ODE when the IC, a
potential parameter, the mass M / M(t) or a backend `stripping_pdf` is a backend value
(`galpy/df/streamspraydf.py:760`); the k-vector noise is reparameterised through the key;
`_fit_track_backend_jit` reconstructs the track natively (P-spline + GCV lambda +
PSD-by-construction covariance). Measured grad-vs-FD, h-converged: sample 2e-6,
streamTrack 3.3e-7 (CPU) / 1e-7 (GPU); construction + sampling 1.2e-8 (audit). A backend
`stripping_pdf` gives a differentiable inverse-CDF grid (bracket index frozen, weights
differentiable). Roadmap (`memory: streamspray-differentiability-roadmap`): M(t),
`stripping_pdf`, `center=`, soft assignment. **[unconfirmed]** which of these beyond M(t)
and stripping_pdf have landed; `center`/`centerpot` have backend-trigger detection
(`_integrate_center`, :865).

```python
# jax.jit(sample()) differentiable in a potential parameter (test_backend_streamspraydf.py:558)
key = grandom.key(seed, backend="jax")
def loss(amp):
    spdf = fardal15spraydf(mass, progenitor=Orbit(ic),
                           pot=LogarithmicHaloPotential(amp=amp, q=0.9), tdisrupt=td)
    return spdf.sample(n, return_orbit=False, tail="leading", key=key)[0].sum()
g = jax.jit(jax.grad(loss))(jnp.asarray(1.1))
```

### streamgapdf

Analytic impulse kernels (Plummer/Hernquist, curved) backend-agnostic and
differentiable (#1090); kick interpolation and impact coordinate transform on the
backend (#1507/#1512); the orbit-integration numpy island removed (#1624; backend inputs
route to the batched in-backend ODE, otherwise the byte-identical per-orbit C path,
`streamgapdf.py:2544-2546`). Construction is NOT traceable (73 in-place assignments in
`_determine_impact_coordtransform` etc., audit S B). `test_streamgapdf_sample` is a
bit-identity test (1 ulp -> 2-5e-5), so it is never a precision fix target
(`memory: streamgapdf-sample-is-a-bitidentity-test`).

### jeans

`sigmar`/`sigmalos` dispatch to backend GL with a corrected semi-infinite transition
scale (#1260, which exposed 5 real defects the former island hid).

PRs (df): spherical #1052-#1056 (family split), #1181, #1392-#1401, #1448/#1449,
#1497, #1511, #1528-#1559, #1578, #1598; King #1514, #1551/#1552, #1566, #1572, #1600;
qdf #1064, #1071, #1231, #1437, #1443/#1444, #1537, #1574, #1593, #1617, #1621; diskdf
#1073/#1076, #1529, #1539-#1542; streamdf/streamTrack #1476-#1505, #1534, #1575, #1584,
#1602, (#1637 open); streamspraydf #1106; streamgapdf #1090, #1507, #1512, #1521, #1624;
cross-cutting #1089 (autodiff_ops), #1432, #1442, #1498, #1544, #1549, #1608, #1613.

## 5. Status table (2026-10-06)

| area | jax eager | jax jit | torch eager | torch compile | GPU | grads |
|---|---|---|---|---|---|---|
| potentials (all but Snapshot/amuse) | yes | yes (DiskSCF no) | yes | yes | yes | coords, theta |
| Orbit.integrate in-backend ODE | yes | yes | yes | torchode only | yes | IC, theta; 2nd order jax (direct adjoint) |
| Orbit.integrate C-STM | yes | pure_callback | yes | no (host call) | host | IC only, 1st order |
| integrate_dxdv / lyapunov | numpy/C | - | - | - | - | n/a |
| Staeckel c=False / Adiabatic / Spherical | yes | yes | yes | yes | yes | (x,v), theta |
| Staeckel/Adiabatic c=True | yes | yes | yes | yes | host | (x,v) only |
| StaeckelGrid/AdiabaticGrid | native build | yes | yes | yes | yes | delegates on theta-grad |
| actionAngleVerticalInverse (mm) | yes | yes | yes | yes | yes | J, theta, params |
| actionAngleTorus | no | no | no | no | no | out of scope |
| spherical DFs | yes | constantbeta/OM/King yes | yes | King yes | yes | theta, beta, W0/M/rt, samples |
| quasiisothermaldf | eval/moments/sampleV | ctor no | yes | ctor yes | - | theta, hr/hsr/hsz |
| diskdf/evolveddiskdf | eval/moments | - | yes | - | - | profile params, through evolution |
| streamdf | track | ctor (#1637 open) | track | - | - | theta, prog IC |
| streamspraydf | yes | yes | yes | - | yes | theta, M |
| streamgapdf | kernels/kicks | ctor no | yes | - | - | kick params |

Cells marked "-" were not measured in the sources I read.
