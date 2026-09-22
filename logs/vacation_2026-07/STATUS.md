# Vacation status snapshot — 2026-09-22

**This file is OVERWRITTEN, not appended.** "Where are things now";
`log.md` is the trace. Refresh on every merge / branch change.

---

## Now

    feat/backends   0172ce5c4   (= #1540)
    merged since the last snapshot:
      #1535 interpRZPotential -> Spline2D, Orbits one-spline-per-orbit
      #1536 jit as a CI matrix axis + traced-share reporting
      #1537 qdf moments w.r.t. potential params + _rg array branch
      #1538 actionAngle grids delegate under a potential gradient
      #1539 evolveddiskdf torch xp.where guard (odd moments)
      #1540 evolveddiskdf honours a gradient carried by the POTENTIAL

## Open PRs

| PR | what | state |
|----|------|-------|
| #1541 | `sphericaldf.sample()` under a gradient-carrying potential | **green 150/150** |
| #1542 | hierarchical velocity grid migrated to the backend | coverage fix pushed (nlevels=0 branch), rebuilding |
| #1543 | constantbetadf fE differentiable w.r.t. the potential (stacked on #1541) | red was the transient JUnit-upload step; re-running |

## The audit sweep is closed out

Every finding is landed or has an open PR. `streamspraydf` needed nothing.
The remaining known gap is jax differentiability for constantbetadf (below).

## What is actually differentiable now

* evolveddiskdf: direct, regular grid AND hierarchical grid all carry
  d/d(potential parameter) (#1540, #1542).
* actionAngle grid classes: delegate to the exact method (#1538).
* qdf moments and `_rg` (#1537).
* sphericaldf `sample()`: runs under a gradient-carrying potential for
  isotropicHernquist / osipkovmerrittHernquist / isotropicNFW, BIT-IDENTICAL to
  numpy (#1541).
* constantbetadf `fE`: d(fE)/d(a) = +22213.654, FD-verified to 2.7e-05 (#1543),
  **torch eager only**.

## The one structural gap left

**jax cannot construct constantbetadf under a trace**, and construction is where
the potential parameter is baked in, so jax has no d/d(parameter) for it. Two
blockers, both in the construction path:

* `_evalpot_asnumpy` calls `as_numpy` on a tracer (TracerArrayConversionError),
* the startt calibration is `while numpy.any(startval == 0.0)` -- data-dependent
  control flow.

Torch EAGER autograd hits neither (values stay concrete), which is why #1543 is
torch-only. Making jax work is a calibration redesign, not a lift.

NEW (2026-09-22, unfiled): `constantbetadf.sample()` still fails under a
gradient-carrying potential inside `_RToxi` via `_make_cmf_interpolator`, even
with #1541's operand lift. Characterised but not yet fixed.

## Recurring traps this round (all cost real time)

* **A fix that changes WHICH ALGORITHM runs breaks the naive A/B.** The numpy FD
  arm then measures a different function. Caught 4x: qdf table-vs-rootfind, both
  actionAngle grids, and constantbetadf fE (scipy-adaptive vs GL: 5.1e-04 of
  "error" that was pure quadrature difference).
* **`Tensor * ndarray` returns NotImplemented** so numpy's `__rmul__` calls
  `.numpy()`; jax accepts the same mix. Several sites were invisible on jax.
  Test mixed arithmetic on TORCH.
* **Read the error's TYPE, not its name.** `LinearizeTracer` looked like a
  torch.func limit; `type(r).__module__` said `jax._src.interpreters.ad`.
* **Coverage drops here are orphaned branches, not untested new code.** Three
  this round (#1537, #1540, #1542), each one line or two, each caused by a
  change making a previously-incidental path unreachable.

## Standing constraints

* Jo merges the main PRs. I can merge develop, not main.
* I **cannot** edit `.github/workflows/**` — provide patches; Jo applies.
* Never name a working branch `feat/...` — use `backend/...`.
* numpy must stay byte-identical; coverage never decreases, not one line.
* Rebase only, no merge commits. Cancel the PRIOR sha's CI on push.
