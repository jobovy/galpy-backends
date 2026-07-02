# Pdf.2 by-family split (review-driven restructure of #1052)

## Approach
On #1052 review the user asked to (a) move generic backend helpers out of the df
file into `galpy.backend`, and (b) split the monolithic sphericaldf+11-variant PR
into per-family reviewable chunks (chosen: "By family, 5 PRs"). Restructured:

- **#1052 → base only** (branch `feat/pdf2-sphericaldf`, commit 26dce79f): sphericaldf
  base machinery + isotropicHernquist pilot; **relocated** `as_numpy` / `exit_cast` /
  `resolve_namespace` into `galpy.backend` (`_namespaces.py` + package `__all__`) — every
  family and the coming disk/qdf/stream/orbit work imports them from there. Plus the
  **astropy-8 exit_cast fix** (forced-backend scalar df call returned a backend-array
  Quantity; astropy 8 rejects it in a dimensionless-ratio subtract → exit_cast at the
  physical output boundary of `__call__`/`dMdE`/`vmomentdensity`/`sigmar`/`sigmat`/`beta`).
- **4 family PRs** off the base, each = its df files + one `tests/test_backend_<family>.py`,
  rewired to import helpers from `galpy.backend`. Targeted at the base branch for clean
  per-family diffs; **retarget + rebase onto feat/backends after #1052 merges** (that's
  when `backend-tests.yml` runs — it only fires for PRs to feat/backends).

Extracted each family's already-migrated files from the pre-split commit 4598635b and
did the mechanical helper-rewire (`_resolve_xp`→`resolve_namespace`, `_as_numpy`→`as_numpy`).
Ran each family in its own worktree (import-shadow workaround: cd into worktree so cwd=''
resolves galpy to it + sitecustomize drops the editable meta_path finder).

## Results
| family | PR | branch | commit | numpy (byte-ident) | new backend test |
|---|---|---|---|---|---|
| base+pilot | #1052 | feat/pdf2-sphericaldf | 26dce79f | 30 backend + pilot | 30 (jax+torch) |
| isotropic (NFW/Plummer/PowerLaw) | #1053 | feat/pdf2-isotropic | 7ada2608 | 32 | 24 jax + 24 torch |
| constantbeta (+Hernquist/PowerLaw) | #1054 | feat/pdf2-constantbeta | 0051acef | 44 | 11 jax + 11 torch |
| osipkovmerritt (+Hern/NFW/PowerLaw) | #1055 | feat/pdf2-osipkovmerritt | fe753844 | 56 | 18 jax + 18 torch |
| eddington + king | #1056 | feat/pdf2-eddington-king | 23cd0ad9 | 45 | 20 jax + 20 torch |

All numpy paths byte-identical. Migration-flip ledger adds (apply as ONE commit at
retarget — see migration_flips_ledger_adds.md): 10 torch (1 isotropic + 1 constantbeta +
7 osipkovmerritt-powerlaw-directint + 1 constantbeta), 3 jax (eddington general-fE,
constantbeta hyp2f1, constantbeta direct-int). Do NOT edit the shared ledger per family
(conflicts) — one commit at retarget.

## Numpy-only remnants confirmed (user's #1052 review Q2)
- **general Eddington fE** (denspot≠pot): direct-GL fE blows up under jax near potInf.
- **general-β>0.5 Hernquist** constantbeta fE: backend hyp2f1 domain limit c−max(a,b)<1.
- (general OM `_logfQ_interp`, `pvr` RectBivariateSpline — numpy-only, not exercised as fails here.)
Lifting = the separate hyp2f1 Pfaff/Euler β∈(0.5,1) extension PR + backend-accurate
direct-GL fE. NOT this PR set.

## Follow-ups
- Retarget: after #1052 merges → feat/backends, `gh pr edit <fam> --base feat/backends` +
  rebase each family branch onto feat/backends (drops the base commit → clean family diff)
  + one ledger-add commit (the flips above). Then backend CI runs on clean diffs.
- osipkovmerritt powerlaw direct-int torch error ~7.8e-3 is large for a floor — a `_QUAD_N`
  bump for the powerlaw cusp may FIX it (task #125 tighten-vs-ledger).
- hyp2f1 β∈(0.5,1) extension PR (separate) to lift general-β Hernquist numpy-only pin.
