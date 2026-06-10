# Track A — Potentials namespace-swap

Branch: `feat/backends` (fan-out worktrees off P1). Detail in master plan §"Track A".

Per-family PRs migrate the private `_evaluate/_Rforce/.../_phizderiv` (+ spherical
`_revaluate/_rforce/_r2deriv`) to `xp = get_namespace(R,z,phi,t)` + `numpy.`→`xp.`,
`numpy.pi`→`math.pi`, fixing hazards and keeping numpy bit-identical:
- in-place `out[idx]=val` → `xp.where(mask, special, expr)` with **safe denominators**
  (guard `0/0` so NaN from the dead branch doesn't poison reverse-mode gradients).
- `isinstance(R,float)`/value branching → shape-based (`getattr(R,'ndim',0)`).
- mutable caches (EllipsoidalPotential hash) → functional `_compute_forces`.

Batches: **P2.1 spherical** (migrate `SphericalPotential.py` first — covers 6+),
P2.2 disk/bar analytic ∥ P2.3 ellipsoidal ∥ P2.4 bars/heavy-mutation,
P2.5 special-backed (needs Pspecial), P2.6 wrappers (last, compose children).
Each PR adds family parity + grad-vs-FD tests (its own test module).

Pilot already migrated Plummer + Isochrone (PR0) — they fold into P2.1's family.

**P2.7 Dissipative / velocity-dependent forces** (DissipativeForce/planarDissipativeForce,
ChandrasekharDynamicalFrictionForce, FDMDynamicalFrictionForce): `f=f(x,v)` — thread
velocity through the namespace-swap. Enables dissipative-orbit autodiff via the Track D
in-backend integrator (see trackB_variational.md "Open consideration").

## 2026-06-09 — P2.5 special-function-backed fan-out complete (#927–#932)
Six-group workflow fan-out (one agent per group + independent adversarial
verifier doing numpy byte-parity vs a pristine checkout + test re-runs). All
migrated, all verified numpy-byte-identical:
- #927 Einasto: gamma/gammaincc via the router (native everywhere); r=0 core
  branch xp.where-guarded. Flagged deltas (values identical): scalar _revaluate
  returns 0-d ndarray (#922 pattern); old r=0 RuntimeWarning no longer emitted.
- #928 DoubleExpDisk: **no Bessel router needed** — j0/j1 live in the init-time
  Ogata weights (numpy, byte-identical); per-call path is elementary ops.
  _de_quadsum replicates numpy.nansum bitwise while zeroing NaN weights in the
  dead branch (AD survives). Flagged: R==0,z≠0 NaN bit pattern 0xfff8→0x7ff8.
  **Param-AD verified exact: quadrature nodes/weights carry zero parameter
  dependence; jax.grad vs FD-of-fully-reconstructed-potential 4e-11 (hr), 1e-12
  (hz).**
- #929 King: real work was interpSphericalPotential's spline evaluation →
  backend-evaluable power-basis PPoly (searchsorted+Horner); King init stays
  numpy by design.
- #930 SCF: full namespace-swap incl. analytic 2nd derivs; needed a #921
  cherry-pick (dropped after the main→feat/backends sync); trace-hostile
  per-point caches bypassed for non-numpy.
- #931 Multipole: static radial splines → PPoly coefficient stacks evaluated
  under xp; time-dependent (tgrid) explicitly numpy-only (NotImplementedError
  seam). Follow-up queued: TD support = stack a time axis + reuse #929's
  machinery for the CubicSpline-in-t.
- #932 DiskSCF/KuijkenDubinski/DiskMultipole correction layer (+logsumexp
  router); merge after #930/#931 for end-to-end backend eval.
Coverage: #931 had 4 uncovered axisymmetric phi=None guards → covered by test
(not excluded). Branch hygiene: main→feat/backends synced (temporarily a merge
commit; **maintainer rule: rebase-only on long-lived branches** — executed 2026-06-10
after the P2.5 merges: 18 commits replayed cleanly onto main (incl. #925),
merge commit dropped, history linear at d3cccc03; 185 backend tests pass).

## 2026-06-09 — also landed/learned
- galpy-vs-galax GPU: gap fully decomposed (tolerance 1.5x + batched-stepping
  2.9x; framework 1.0x) — see logs/gpu_speed_galpy_vs_galax/. galpy keeps
  per-orbit stepping (per-orbit error guarantees, batch-independent results).
- Backend dispatch (get_namespace) is jitted away (trace-time, zero runtime);
  XLA HloCSE merges duplicate identical igamma expansions (verified micro +
  real path + jax/XLA source) — no special-function call-site optimization
  needed in the spherical force layer.
