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
