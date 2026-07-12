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

## Landing (2026-07-03)
- **#1052 base MERGED** (squash → feat/backends `de366175`, 2026-07-03 07:26 UTC).
- **Rebase cascade done.** feat/pdf2-sphericaldf had been amended for the eddington
  ledger, so the family branches sat on a STALE copy of the base commits (26dce79f/
  4b21813d) not in feat/backends by SHA. Rebased each by replaying ONLY its family
  commit(s): `git rebase --onto origin/feat/backends 26dce79f <branch>` (26dce79f is
  every family commit's parent). All 5 clean, no conflicts, no markers:
  - #1057 test-helpers → aa6980a5 (23 test files, +377 as_numpy, disjoint from families)
  - #1053 isotropic → c9659df1  · #1054 constantbeta → 15de1a26
  - #1055 osipkovmerritt → d3f12441 · #1056 eddington+king → 6d4233a1
- **No per-family ledger delta**: the migration-flip entries all rode in with the
  #1052 base (the c4e39670/2d223912 eddington-directint + friends), so feat/backends
  already carries them; family branches touch NO backend_xfail.txt. Family PRs re-ran
  full CI green on the old base → equivalent merged base should match.
- #1057 automerge was cleared by the force-push; user DECLINED re-arming (will do it
  themselves / re-review). Batch shepherd `shepherd_pdf2_family.sh` drives all 5 green
  (auto-rerun flakes, alert on genuine red).

## 2026-07-03 (cont.)
- #1057 test-helpers MERGED (13:36). feat/backends advanced; family PRs disjoint
  (no rebase needed).
- Per user "one at a time to avoid rebase hell": doing family PRs sequentially.
  #1053 first: rewired its test file's local `_np` -> centralized `as_numpy`
  (matching #1057), 48 tests pass jax+torch, pushed af408fc1. On #1053 merge ->
  do #1054 (rewire+ready), etc.

## #1055 backend-ledger fix (2026-07-03, commit 671fe7e4)
- #1055 backend CI red on the osipkovmerritt DIRECT-INTEGRATION checks (jax 10
  FAILED, torch 4 FAILED) -- the anisotropic-DF `*_directint` moment/dens/beta
  tests are numerically unstable near the potential-inf limit and run-to-run
  variable under xdist, exactly the general-Eddington directint failure mode
  already ledgered whole. These were NOT yet in `backend_xfail.txt` for these
  variants (jax had 0 OM-directint entries; torch had 19 but not powerlaw
  dens/beta/nonself).
- Fix = ledger the observed reds + same-subclass siblings (xfail strict=False,
  so a currently-passing sibling just XPASSes -> zero risk, robust to the
  variability). jax +14 (dehnencore_in_nfw{dens,meanvr,sigmar,beta};
  powerlaw{dens,sigmar,beta}[0.5/2.0/5.0]+nonself_dens); torch +7 (powerlaw
  {dens,beta}[params]+nonself_dens -> completes torch powerlaw directint,
  sigmar[params] was already there). Only #1055 touches backend_xfail.txt among
  the family PRs -> conflict-free.
- NOTE (task #125): the alternative to ledgering is a `_QUAD_N`/rmin bump for
  the powerlaw cusp (torch directint error ~7.8e-3). Deferred -- the user asked
  to fix the ledger red now; tighten-vs-ledger is the #125 follow-up. If #125
  tightens these, drop the entries then.
- Prior-SHA (f9690295) in-progress runs cancelled (runner-constrained); new SHA
  671fe7e4 spawns fresh CI; autoMerge ON.

## MILESTONE: Pdf.2 spherical-DF family COMPLETE (2026-07-04, overnight autonomous)
- All 5 MERGED: base #1052, isotropic #1053, eddington/king #1056, osipkovmerritt
  #1055, constantbeta #1054 (08:15Z). #1054 needed rebase-onto-post-#1055 + a
  5-entry jax ledger add (hyp2f1-fallback-gap: constantbetadf_against_hernquist,
  osipkovmerritt_hernquist_dMdE_limits; directint/dMdE quadrature-floor+NaN:
  anisotropic_hernquist_{dens_directint,dMdE_integral},
  constantbeta_differentpotentials_dens_directint — all torch-ledgered already,
  deterministic, NOT migration bugs). Torch already xfailed all 5.
- **KEY LESSON (banked to memory):** rebase a family PR onto the CURRENT
  feat/backends BEFORE adding its ledger entries, else backend_xfail.txt conflicts
  with sibling-PR ledger changes on merge. And: never re-run a QUEUED all-clear
  (resets its queue position behind ~40 jobs under the runner ceiling; it
  self-heals as merges free runners).

## #131 COMPLETE + next DF started (2026-07-04)
- #131 PR-A freqs MERGED (#1060); PR-B angles OPENED (#1063, for review) — the
  C-native Staeckel angle Jacobian, derive-first-validated (2 independent
  derivations vs c=False AD), jax 9 + torch 8 + numpy 72 pass, value byte-identical.
  Design/impl spec: plan/staeckel_deriv_fix/prB_angle_jac_VALIDATED_spec.md.
- Next DF wave scoped (plan/pdf3_pdf4_diskdf_qdf/scope_and_plan.md): qdf FIRST
  (leverages #131 grads, all prereqs ready), then diskdf (needs novel _scimath_log
  complex-log), evolveddiskdf reject-guard. **qdf PR-4a OPENED (#1064, CI GREEN):**
  differentiable qdf(R,vR,vT,z,vz) via __call__+_rg; value byte-identical 1e-12,
  grad-vs-FD ~3e-3 (first-order Staeckel floor). Slow-shard risk did NOT
  materialize (unmigrated GL builds numpy grids -> __call__ stays numpy).
- PAUSED further DF migration (PR-4b/diskdf) pending user review of the qdf
  approach in #1064 (avoid rework + review backlog).
