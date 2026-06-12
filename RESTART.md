# Restart instructions — galpy backend project (paused 2026-06-08 for reboot)

> Read this first on resume. Nothing on disk is at risk — all worktrees are committed
> and pushed, no uncommitted tracked changes anywhere, no OS-level daemons running.
> The only thing the reboot kills are this session's **in-memory Monitor watchers**
> (CI/merge monitors). They do NOT auto-restart — re-arm the ones still needed (below).

## TL;DR — what to do on resume
1. `cd /var/tmp/pspecial` (the active worktree; branch `feat/pspecial`, at `be02d802`).
2. Re-check CI on the open PRs (`gh pr checks <n>`) and **re-arm monitors** as needed.
3. First task: **fix #916 codecov** (see below), then continue **Pspecial Tier 3** (K-Bessel).
4. After Pspecial: the **dissipative-forces** migration (user's stated next priority).

## Monitors that were running (all stopped by the reboot — re-arm if wanted)
These were harness `Monitor` background tasks watching CI / merges. None survive a reboot.
There were NO nohup/cron daemons for this project (the only crontab entry, `monitor_geir`,
is an unrelated project). Helper scripts live in `/var/tmp/*.sh`
(`cancel_superseded.sh <pr>` = cancel superseded CI runs after a push; plus various
`*_monitor.sh` poll loops). Re-arm a CI watcher with `Monitor` on the PR(s) you care about.

## Open PRs and their state (as of pause)
| PR | branch → base | state | CI |
|----|---------------|-------|-----|
| **#916** | feat/pspecial → feat/backends | **OPEN, active work** | 106 pass; **codecov/patch + codecov/project FAIL** (not tests) |
| #911 | feat/pvar-pot-disks2 → feat/variational3d | open | green |
| #910 | feat/pvar-pot-ellipsoidal → feat/variational3d | open | green |
| #909 | feat/pvar-pot-bars → feat/variational3d | open | green |
| #906 | feat/inbackend-ode → ? | DRAFT | — |
| #892 | feat/backends | DRAFT (integration branch) | — |
| #891 | feat/variational3d | DRAFT (feature branch) | — |

#909/#910/#911 were green at pause — if still green on resume, they may be ready for the
user to review/merge into `feat/variational3d`.

## #916 (Pspecial) — current status & next steps
**Shipped (committed + pushed on `feat/pspecial`):**
- Tier 1 (`424f1d7b`): router + gammaln/gamma/gammainc/gammaincc/erf/erfc/i0/i1/xlogy.
- Tier 2 (`be02d802`): hyp2f1, hyp1f1, ellipk, ellipe.
  - ellipk/ellipe via AGM (~1e-15). hyp2f1 via Euler integral + boundary-layer Gauss-Legendre
    (~1e-13 at r/a≲50, ~1e-6 at r/a~500). hyp1f1 b=a+1 exact via gammainc.
  - **Key finding:** jax native hyp2f1 AND hyp1f1 are catastrophically wrong for z<-1 →
    router forces fallback via `_NATIVE_UNRELIABLE` (vs `_NATIVE_MISSING`); tripwire tests
    auto-promote them if jax fixes them. (Saved to memory + appendices/D_special_functions.md.)
- 78 tests in `tests/test_backend_special.py`, all green locally.

**FIRST on resume — fix #916 codecov fail (the only red):**
- Tests pass (106) but codecov/patch + codecov/project fail → the new `galpy/backend/special/
  _fallback/*.py` lines are almost certainly NOT counted by the coverage job. Confirm whether
  the coverage-reporting job (py3.14 ubuntu) installs jax+torch and runs `test_backend_special.py`;
  if not, the fallback branches are exercised only on a non-coverage backend job. Fix by making
  the coverage job run the backend tests with jax+torch (per the "no hiding uncovered code" rule —
  do NOT pragma-hide). Check `.github/workflows/build.yml` REQUIRES_JAX/REQUIRES_TORCH +
  PYTHON_COVREPORTS_VERSION job. Then `cancel_superseded.sh 916` after any push.

**THEN — Pspecial Tier 3 (K-Bessel), on the same PR/branch:**
- Needed by `RazorThinExponentialDisk` forces: `k0, k1, kn` (modified Bessel 2nd kind, real x>0).
  (`DoubleExponentialDisk`'s j0/j1/jv/jn_zeros are construction-time constants — NO backend Bessel.)
- Native availability (probed): scipy all ✓. **jax has none** (k0/k1/kn fallback). **torch has
  `modified_bessel_k0`/`modified_bessel_k1`** (different names → add a per-backend name-alias map
  in `_router._backend_special`/`_dispatch`) but **no kn** → kn via upward recurrence
  `K_{n+1}=K_{n-1}+(2n/x)K_n` from k0,k1 (so kn(2,x)=k0+(2/x)k1).
- Fallback for k0,k1 (jax; also feeds torch's kn): series (small x, includes the log term via
  i0/i1 which are Tier-1 native) + asymptotic (large x). Validate to ~1e-12 vs scipy + grad-vs-FD,
  same pattern as Tier 2 (prototype in a scratch script first, then write module + tests).

**THEN — Tier 4:** assoc_legendre (Bonnet recursion, replaces util/special.compute_legendre on the
Multipole path) + Gegenbauer (3-term recurrence, SCF radial basis) + the fixed-order GL-quadrature
helper (already started in `_fallback/_quadrature.py`). Plus `sici` for dissipative friction.
**THEN —** wire the special-backed potentials (P2.5: Einasto, PowerSphericalwCutoff, TwoPowerSpherical
forces, disk potentials, SCF/Multipole/DiskSCF) — watch the hyp2f1 ~1e-6 ceiling at extreme r/a vs
the numpy/scipy path (bump GL nodes there if an existing test probes huge radii).

## After Pspecial — dissipative forces (user's explicit next priority)
P2.7 proper migration: `DissipativeForce`/`planarDissipativeForce`,
`ChandrasekharDynamicalFrictionForce`, `FDMDynamicalFrictionForce` — f=f(x,v), thread velocity
through the namespace-swap, with tests. (The #897 scaffold was closed; do the real migration.)
Note: `feat/P2.7-dissipative` worktree exists at `.claude/worktrees/wf_51aa2234-2b6-5` (2ca98330).

## Standing rules (don't forget)
- Every commit ends with `Co-authored-by: Claude Opus 4.8 <noreply@anthropic.com>`.
- pre-commit ruff-format ABORTS the commit on reformat → re-`git add` + re-commit (this bit us on
  the Tier-1 push; always verify `git ls-remote origin <branch>` after pushing a new branch).
- Worktree tests: `PYTHONPATH=<worktree>` + assert `galpy.__file__` (editable-install import shadow).
- Flag any non-byte-identical numpy-path change for the user.
- Cancel superseded CI after each push: `/var/tmp/cancel_superseded.sh <pr>`.

## Plan / logs
- Master plan: `~/.claude/plans/i-want-to-start-snug-rocket.md`.
- Pspecial spec + running log: `~/Repos/galpy-jax/appendices/D_special_functions.md`
  (Tier 1-2 implementation log appended at the end).
