# AA backend redesign — drop numpy_island, run existing tests under the backend for real

**2026-06-30 (overnight).** Major course-correction on the action-angle backend story,
triggered by the user spotting that `@numpy_island` makes the existing AA tests run
**numpy** under `--backend=jax/torch` — i.e. the un-ledgered AA tests were passing
*hollowly*, not exercising the backend. "That's not what we want — the whole point of
the project is that the existing tests run meaningfully under all backends."

## Approach

### The discovery
- The all-backend harness wraps every test in `use(backend, force=True)` (conftest.py:298).
- `@numpy_island` forces numpy whenever **no argument is a backend array**. Every existing
  test passes numpy inputs → an islanded method runs **numpy**, not jax/torch.
- So #1033's "AA backend completion" (numpy-island the query paths + un-ledger 181) made the
  forced-backend AA forward tests pass by *running numpy*. Hollow.
- Worse, on `feat/backends` the jax AA forward tests were ALSO hollow pre-#1033: jax arrays
  round-trip through `scipy.integrate.quad` as numpy (the eager-slow fall-through), so they
  "passed" without being jax-native. torch can't round-trip → ledgered.

### Salvage check on #1033 → empty
- `numpy_island` and `promote_scalars` and the `_backend_compatible` decorator-promotion are
  **#1033 inventions — not on `feat/backends`**.
- The apply_amp / scalar-coerce / interpRZ-ledger fixes I landed on #1033 during the day were
  patching problems #1033's OWN `promote_scalars` inflicted. Verified on `feat/backends`:
  `test_amp_mult_divide[AnyAxisymmetricRazorThinDiskPotential]` (torch) and
  `test_interpolation_potential_diffinputs_c` (jax) **pass** without any of them.
- → Nothing to cherry-pick. Closed #1033 and #1034 (same islanding).

### The fix: resolved-namespace dispatch
Gate the backend path on the **resolved namespace**, not `is_backend_array(input)`:
```python
xp = get_namespace(R, vR, vT, z, vz)      # honours use(backend, force=True)
if xp is not numpy:
    R, vR, ... = (xp.asarray(a) for a in (R, vR, ...))   # promote numpy inputs
    return self._<method>_backend(...)     # the vectorised path that already exists
else:
    ...                                    # numpy/scipy core — byte-identical
```
`get_namespace(numpy_array) is numpy` → True for default numpy, False under forced jax/torch
(verified). So numpy stays byte-identical and the existing tests run the backend for real.

### Shared torch root-find fix (`galpy/backend/_torch/optimize.py`)
`brentq_backend` unconditionally did `x0.clone().requires_grad_(True)`, so the forward ALWAYS
returned a grad-requiring tensor → broke plain `.numpy()` in the existing tests. Fix: skip the
implicit-function reparameterisation when `f(x0)` needs no grad (the plain-forward case) →
clean detached forward; the gradient tests still enter the reparam and pass. Shared by every
AA module that root-finds (Spherical, Adiabatic, Vertical, ...).

## Results (measured, raw, backend running for real)

**Principle proven:** under forced jax with numpy inputs, `actionAngleSpherical(...)` now
returns a genuine `jax ArrayImpl` (backend=True) matching numpy values. Not a fall-through.

**Spherical** (existing `test_actionAngle` Spherical suite, raw):
- jax: 16 passed + 3 xpass + 1 xfail (essentially done, jax-native).
- torch: after the root-find fix — 3 passed + 6 xpass + 10 xfail + 1 regression
  (down from 3-fail/16-xfail before the fix).
- numpy: 20 passed (byte-identical — the change is a no-op when `xp is numpy`).
- Pillar-2 backend-array tests still pass: jax 148 / torch 148.

**Applied the same dispatch to Spherical (4), Vertical (3), Adiabatic (4).** Committed as
`c09f160e` on `feat/aa-backend-native`.

## Key finding: torch is harder than jax (test-side, not backend)
Making outputs genuinely torch surfaces **test-side** incompatibilities: existing tests do
`numpy.mean(output)` / `numpy.fabs(output)`, which work on jax arrays (jax duck-types as numpy)
but raise on torch tensors (`numpy.mean(tensor)` → `tensor.mean(axis=,dtype=,out=)` signature
mismatch). So:
- **jax**: the redesign is a clean win — existing AA tests become genuinely jax-native + mostly
  pass. Un-ledger candidates.
- **torch**: the backend *code* is correct (Pillar-2 passes), but existing tests need
  backend-agnostic reductions (like the orbit tests got, task #81) OR honest ledgering for the
  `numpy.<reduction>(output)` cases.

## Dead-ends / corrections
- Initially (during the day) I "fixed" AnyAxisymmetric/interpRZ on #1033 and added ledger
  entries — all unnecessary once `promote_scalars` is dropped (self-inflicted). Reverted by
  closing #1033.
- `xp is numpy` gate required checking that `get_namespace` returns the bare `numpy` module
  (it does) — `array_api_compat.numpy` would have broken it.

## Remaining work (per the principle: Python forwards backend-native; C ledgered)
1. **Harmonic** (closed-form, no `_backend` method): numpy inputs under forced backend still
   give numpy output (no promotion). Needs an entry-level `xp.asarray` promote (watch scalar
   handling). Vertical/Adiabatic done.
2. **Staeckel / grids / IsochroneApprox**: apply dispatch + measure-then-triage.
3. **torch test-side**: backend-agnostic reductions for the `numpy.mean/fabs(output)` tests, OR
   honest ledger.
4. **C paths (`c=True`)**: stay ledgered until analytic derivatives (separate future track).
5. **Honest ledger regen**: un-ledger only tests that pass BY RUNNING the backend (verify the
   output is a backend array); ledger torch test-incompat regressions honestly.

PRs #1033 + #1034 closed. Branch `feat/aa-backend-native` off `feat/backends`.
