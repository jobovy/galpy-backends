# Track B — Pvar-pot fan-out (per-potential 3D C Hessian)

Building on the variational framework (`feat/variational3d`): the 3D variational RHS +
`integrate_dxdv` phasedim==6, the `z2deriv`/`Rzderiv`/`zphideriv` struct pointers + NULL-safe
`calc*` aggregators, the `hasC_dxdv3d` capability gate, and a **parametrized test registry**
(`tests/conftest.py::liouville3d_registry`, categories spherical/axisymmetric/nonaxisymmetric)
driving `test_dxdv_3d_c_vs_python` (C-vs-Python finite-difference, **unit deviation**),
`test_liouville_3d` (det M=1 / symplecticity / flow), `test_liouville_3d_2d_bridge`.

## Approach
Per-potential **closed-form** Hessians (user decision: not a shared helper). For a spherical
Φ(r): R2deriv=Φ″R²/r²+Φ′z²/r³, z2deriv=Φ″z²/r²+Φ′R²/r³, Rzderiv=Rz(Φ″/r²−Φ′/r³), with
Φ″=PlanarR2deriv, Φ′=−PlanarRforce; phi2deriv/Rphideriv/zphideriv=0. Axisymmetric disks need
only R2/z2/Rz (the three φ-derivatives vanish). The C-vs-Python FD test is the independent
correctness gate; a wrong Hessian fails it (this caught the latent #899/#902 PlanarR2deriv bugs
earlier — invisible to det-M/symplecticity, which hold for *any* symmetric K).

Fan-out = one sub-agent per family, each in a worktree off `feat/variational3d`, → its own PR.

## Branch hygiene (done 2026-06-05)
- #902 (TwoPower PlanarR2deriv fix) merged to **main**; #901 (SpiralArms) squash-merged to
  `feat/variational3d`; then `feat/variational3d` **rebased onto main** (linear, no merge commit,
  per user preference) so it carries #902. 3 foundation commits replay cleanly over #902;
  31-test variational battery green post-rebase. Fan-out branches off this rebased base.

## Landed / in-flight PRs (into feat/variational3d)
- **#900** spherical Hernquist/NFW/Jaffe (in base).
- **#901** SpiralArms (first non-axisymmetric; zphideriv) (in base).
- **#904** spherical batch: PowerSpherical, PowerSphericalwCutoff, DehnenSpherical, DehnenCore,
  Burkert, **Isochrone**, HomogeneousSphere. 52 variational tests pass. NOTE: making Isochrone
  advertise `hasC_dxdv3d=True` required switching the gate-test's negative example
  (`test_integrate_dxdv_3d_c_requires_full_hessian`) from Isochrone → PseudoIsothermal.
- **#905** axisymmetric disks: KuzminKutuzovStaeckel (prolate-spheroidal chain-rule Hessian).
  - **KuzminDisk dead-end / judgment call:** Φ ~ −amp/√(R²+(a+|z|)²) is only **C⁰ at z=0** →
    z2deriv discontinuous at the disk plane. The registry IC crosses z=0 five times; the two
    adaptive integrators legitimately diverge there (C-vs-Python 4.1e-6 > 1e-6 gate). The C
    Hessian is *correct* (off-plane C-vs-Python 1.3e-11; analytic-vs-FD ~1e-10). Agent kept the
    Hessian + `hasC_dxdv3d=True` but **excluded KuzminDisk from the strict registry** with a
    comment. **Resolved in review (2026-06-05):** keep `hasC_dxdv3d=True` (the Python path has the
    same z=0 discontinuity, so C is no worse and faster), and add a standalone off-plane
    correctness test `test_kuzmindisk_dxdv_3d_c_vs_python_offplane` (orbit with large z0 moving
    away from the plane, verified min|z|>1 → smooth regime → C-vs-Python agree ~1e-11, asserted
    <1e-8). So KuzminDisk has real C-Hessian coverage without the bogus det-M-at-z=0 failure.

## Review of #904 / #905 (2026-06-05, independent re-verification)
- **#904** (spherical batch): transform verified correct (Isochrone template: R2deriv=Φ″R²/r²+Φ′z²/r³
  etc., amp via the planar functions); Isochrone→PseudoIsothermal gate-test swap valid; numpy
  byte-identical (additive C + flags only); **52 variational tests pass** on an independent
  rebuild. Recommend merge, no changes.
- **#905** (Kuzmin disks): KuzminKutuzov chain-rule Hessian verified (correct structure +
  dVdl sign); **34 tests pass** independently; added the off-plane KuzminDisk test (above) and
  referenced it from the registry-exclusion note (pushed 1e15e91b). Recommend merge.
  - MiyamotoNagai reported already done (axisymmetric template).

## numpy byte-identical
All Pvar-pot changes ADD C 2nd-derivative functions consumed only by the 3D-variational C path,
plus `hasC_dxdv3d=True` Python flags — the numpy/pure-Python numerical path is untouched. No flags.

## Remaining families
ellipsoidal/triaxial · SCF/Multipole (needs the **zphideriv z-asymmetry guard** — these can be
z-asymmetric, so zphideriv(z=0)≠0) · bars (DehnenBar etc.) · wrappers · interpRZ · special-fn
disks. Then Pvar-lyapunov, Pvar-dissip.
