# Pvar-struct — full-3D Hessian C ABI (serialization point)

Branch: `feat/variational3d` (off `main`, worktree `../galpy-variational3d`). The
enabling, behaviorally-inert change that every per-potential 3D-Hessian C PR
(Pvar-pot.*) branches off of.

## Approach
- `struct potentialArg` (galpy_potentials.h): added `z2deriv`, `Rzderiv`,
  `zphideriv` function pointers next to the existing full-3D `R2deriv/phi2deriv/
  Rphideriv`.
- `init_potentialArgs` (galpy_potentials.c): NULL-init the 3 new pointers.
- Aggregators `calcz2deriv/calcRzderiv/calczphideriv` (galpy_potentials.c, in the
  LCOV_EXCL block, model on `calcR2deriv`): **NULL-safe** — a potential that does
  not provide a term contributes 0. (Correctness is guaranteed by gating the 3D
  dxdv path on every component supporting the full Hessian; the NULL-skip is a
  safety belt against segfaults during the incremental Pvar-pot rollout.) Declared
  in the header.
- Wired SpiralArms' existing 3D second derivs (R2deriv, z2deriv, phi2deriv,
  Rzderiv, Rphideriv) in `parse_leapFuncArgs_Full` (case 27). It was the only C
  potential with any of these (previously commented out because the struct lacked
  the pointers).

## Dead-ends / notes
- **SpiralArms is NOT a complete 3D Hessian**: it has no C `zphideriv` (∂²Φ/∂z∂φ),
  which is generally nonzero for a 3D spiral. Left NULL here; must be added in
  Pvar-pot before SpiralArms can be used in the 3D STM validation. (The design-doc
  claim that SpiralArms is "the only complete 3D Hessian" is off by this one term.)
- Kept the change minimal/additive (no RHS change): `evalRectDeriv_dxdv` is still
  the broken stub, and the new aggregators are not called yet, so the suite is
  unaffected. Pvar-rhs will fix the RHS and call the aggregators.

## Results (worktree build, GALPY_COMPILE_NO_OPENMP=1)
- Compiles clean (libgalpy.so; torus skipped — external lib absent, expected).
- Behaviorally inert: **22 passed** — `test_liouville_planar`,
  `test_fixedstepsize_dxdv`, `test_integrate_dxdv_errors`, `test_SpiralArmsPotential`.
- SpiralArms 3D orbit integrates with relative energy drift ~5e-13.

## Next (Track B)
- Pvar-pot.*: implement per-potential 3D Hessian in C (incl. SpiralArms `zphideriv`),
  fan out by family in worktrees off merged Pvar-struct.
- Pvar-rhs: replace the stub `evalRectDeriv_dxdv` with the correct 6×6-Jacobian RHS
  (reuse the proven 2D (x,y) block + the new z-coupling), lift `integrate_dxdv` to
  `phasedim==6`, add the full test battery.
