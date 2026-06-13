# Full track re-assessment — 2026-06-13

After #891 (variational track) merged to main and feat/backends rebased onto it
(36c92e8f), a 6-agent re-assessment mapped done/not-done per track. Headline:
**backend project ~1/3 done; README overstated Track A.**

## Done (feat/backends)
- Track 0 infra COMPLETE; Track A 43 potential files migrated + f32/device policy
  (#949); SpiralArms jax f32→f64 fixed (8064e3f0); Track D in-backend integrator
  (#906); Track B variational COMPLETE on main.

## Track A gaps (NOT as complete as README)
- P2.7 dissipative forces NOT migrated (Chandrasekhar/FDM bare numpy + scipy.erf;
  zero backend tests) — corrects a 'done' claim.
- Conventions guard protects only 11/43 classes (P2.1-P2.5 unregistered).
- Log-spiral #952 2nd-derivs bare numpy (variational ingestion onto migrated files).
- RazorThinExpDisk only _surfdens migrated; linalg.py shim never built; helper
  centralization; SCF where-without-out warning.

## Unstarted
- Track D C-STM autodiff wrappers (orbit_stm.py) + Orbit backend-agnosticism/routing.
- Track E action-angle; Track F df+streams (largest); Track C param C-side.
- Track B residuals: batched (nt,6,6) M; dissipative-vs-AD cross-check.

## Pillar 1 — QUANTIFIED
- jax ~86% pytest / 96% per-potential; 4/5 fails = SCF _gaussianQuadrature jax-default
  leak; + df unmigrated; 1 tol; 1 flaky shard.
- torch lower; dominant = torch namespace rejects scalar inputs (xp.sqrt(float)).

## CI plan (agreed): all-backend suite per-push-to-feat/backends + nightly + dispatch,
SHARDED TEST_FILES×{jax,torch} (parallel ~1 cycle), CPU, green-via-xfail-ledger
(shrinking list = burndown), step-summary, tiny per-PR smoke. Not 3x/commit.

## Order: 1) Pillar-1 harness 2) Track A truly-complete wave 3) Track D C-STM→Orbit
4) E/F/C as deps allow. Track G figures flow as each lands (GPU now avail).
