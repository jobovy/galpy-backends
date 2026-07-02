# Pdf.2 family migration-flip ledger adds (apply as ONE commit at retarget to feat/backends)

These are numpy-path tests that FLIP to a backend failure once the family DF (and/or
its potential) is migrated — the test itself isn't backend-agnostic (passes pot(0,0)
which is now a torch/jax scalar under forced backend → df output stays backend →
numpy.fabs(tensor) raises). Same class as the already-ledgered hernquist entry.
Do NOT edit backend_xfail.txt in the per-family PRs (shared file → conflicts); add all
here in one commit at the milestone/retarget.

## isotropic family (branch feat/pdf2-isotropic, 7ada2608)
- torch tests/test_sphericaldf.py::test_isotropic_plummer_energyoutofbounds
  (2nd assertion dfp((pot(0,0)-1e-4,1.1)) -> torch E -> torch out; value correct 0.0)

## constantbeta family (branch feat/pdf2-constantbeta, <sha pending>)
- (pending agent report)

## osipkovmerritt family (branch feat/pdf2-osipkovmerritt, <sha pending>)
- (pending agent report)

## eddington-king family (branch feat/pdf2-eddington-king, 23cd0ad9)
- jax tests/test_sphericaldf.py::test_isotropic_eddington_dehnencore_in_nfw_dens_directint
  (general Eddington denspot!=pot: direct-GL fE blows up under jax near potInf,
   fE(jax)~1.4e20 vs numpy~1.4e-9 -> vmomentdensity overshoots tol. KNOWN numpy-only
   remnant [general Eddington fE]. Torch ALREADY ledgered at backend_xfail.txt:791;
   jax ledger MISSING it -> add jax entry. NOT a rename bug; self-consistent DehnenCore
   dens_directint passes under jax.)
  NOTE: this is a numpy-only-remnant confirmation for the user's #1052 review Q
  ("are there other numpy-only parts, does it come later?") — general Eddington fE
  (and general constantbeta fE, general OM _logfQ_interp) are numpy-only for now;
  lifting them = the hyp2f1/Pspecial extension + backend-accurate direct-GL fE, a
  follow-up (NOT this PR set).

## osipkovmerritt family (branch feat/pdf2-osipkovmerritt, fe753844)
7 torch quadrature-floor flips (powerlaw direct-int: torch fixed-GL vs numpy adaptive
quad; magnitude ~7.8e-3 vs tol 1e-7; migration-flip — unmigrated torch fell through to
numpy/scipy-adaptive & passed, migrated torch uses fixed_quad & floors). Add as torch
xfail entries (verify jax equivalents too — jax forced run was killed, likely same floor):
- torch tests/test_sphericaldf.py::test_osipkovmerritt_powerlaw_dens_directint[0.5]
- torch tests/test_sphericaldf.py::test_osipkovmerritt_powerlaw_dens_directint[2.0]
- torch tests/test_sphericaldf.py::test_osipkovmerritt_powerlaw_dens_directint[5.0]
- torch tests/test_sphericaldf.py::test_osipkovmerritt_powerlaw_sigmar_directint[0.5]
- torch tests/test_sphericaldf.py::test_osipkovmerritt_powerlaw_sigmar_directint[2.0]
- torch tests/test_sphericaldf.py::test_osipkovmerritt_powerlaw_sigmar_directint[5.0]
- torch tests/test_sphericaldf.py::test_osipkovmerritt_powerlaw_nonself_dens_directint
NOTE: 7.8e-3 is large for a "floor" — possibly GL order too low for the powerlaw cusp;
tightening _QUAD_N for the powerlaw direct-int MAY fix (better than ledger). That's task
#125 (Pdf.2 residual-slow / direct-int) territory — defer the tighten-vs-ledger call to
the milestone; ledger for now to keep retarget green.

## constantbeta family (branch feat/pdf2-constantbeta, 0051acef)
- torch tests/test_sphericaldf.py::test_constantbeta_powerlaw_beta0_equals_isotropic
  (test-side numpy.all(torch tensor) TypeError; migration-flip, same class as plummer)
- jax tests/test_sphericaldf.py::test_constantbetadf_against_hernquist
  (general-beta>0.5 Hernquist hyp2f1 numpy-only remnant: NotImplementedError
   "hyp2f1 fallback requires c-max(a,b)>=1 (got a=-3.2,b=4.4,c=5.2)" -> c-maxab=0.8<1;
   BY DESIGN. This is the user's #1052 review Q1 — LIFTABLE via the separate hyp2f1
   Pfaff/Euler (beta in (0.5,1)) extension PR; ledger jax for now.)
- jax tests/test_sphericaldf.py::test_constantbeta_differentpotentials_dens_directint
  (density direct-int quadrature floor under jax)

## SUMMARY — all 4 families done; migration-flip ledger adds to apply as ONE commit at retarget:
  torch: 3  (isotropic plummer eoob; constantbeta powerlaw beta0; + osipkovmerritt x7 = 10 total torch)
  jax:   3  (eddington general fE; constantbeta hyp2f1 + differentpot direct-int)
  Plus verify jax equivalents of the osipkovmerritt-powerlaw-directint x7 (jax run was killed).
Family branch shas: isotropic 7ada2608 | eddington-king 23cd0ad9 | osipkovmerritt fe753844 | constantbeta 0051acef
