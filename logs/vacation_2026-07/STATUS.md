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

### constantbetadf.sample(): WIP branch, NOT a PR -- blocked on a NaN gradient

Branch `backend/constantbetadf-sample` (9ec9e779e, stacked on #1543) is PUSHED
WITHOUT a PR. Do not merge it.

**Works:** backend-key sampling runs under a gradient-carrying potential and
returns a backend Orbit, where it previously raised. Five fixes, each scoped to
ONE path so numpy and backend cannot bleed (details in the commit message). One
of them is a REAL correctness fix to #1541's own code: `get_namespace` returns
the FORCED namespace ahead of the data, so inside constantbetadf's
`use("numpy", force=True)` it handed back numpy and the next line was
`ndarray * Tensor`. `namespace_from_arrays` is the data-only resolver.

numpy-key sampling with such a potential now raises a clear NotImplementedError
naming the backend key; making it actually work needs a DETACHED view of the
potential, which galpy has no construct for.

**Blocked:** `d(sum r)/da` is **NaN** (FD = +57.417). Values are fine
(`forward_nan=False`) -- only the backward is poisoned. Not shipped precisely
because a NaN gradient is worse than the error it replaces: it propagates
silently into an optimiser, where the original at least failed loudly.

`torch.autograd.set_detect_anomaly` localises it to **DivBackward0** from
`grad(func)(r) / rforce(r)` (constantbetadf ~636), evaluated by
`_fEintegrand_smallr` at very small r during the construction-time startt
calibration, where `rforce -> 0`.

`stop_gradient` at that calibration boundary was tried -- legitimate, since the
loop only calibrates an integration LIMIT -- and did NOT clear the NaN, so
there is at least one more path into that division. Next step: anomaly
detection again, with the stop_gradient in place, to find the second path.

Note this NaN is LATENT on the fE path too, yet #1543's FD check passes at
2.7e-05: fE's value is integrated live by `_fE_backend`, so its backward does
not traverse the calibration. Sampling's does.

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
