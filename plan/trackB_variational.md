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
