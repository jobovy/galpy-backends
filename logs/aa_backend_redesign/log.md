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

## Full-suite measurement (test_actionAngle, backend running for real)

| | passed | xpass (now backend-native for real) | xfail | failed (regressions) |
|---|---|---|---|---|
| jax  | 101 | **33** | 66  | 6 |
| torch| 33  | **16** | 150 | 8 |

**33 jax + 16 torch ledgered tests now pass genuinely** (the win). Regressions triaged:
1. **C-path routing bug (mine)** — Adiabatic backend gate was before the c=True check, so
   c=True under a forced backend took the backend path not C. FIXED (commit: gate on `not use_c`);
   4 jax + 2 torch resolved.
2. **Vertical backend-path gap** (`Vertical_unbound`, `physical_vertical`; both backends) — the
   numpy path returns the J=9999.99 unbound sentinel, but `_evaluate_backend` returns garbage
   (~2e30); the vectorised path doesn't replicate the unbound mask. Real backend-code fix.
3. **torch `numpy.mean(output)` test-incompat** (`*_conserved_EccZmaxRperiRap*`) — torch tensors
   don't duck-type through numpy reductions. Backend-agnostic test reductions (like the orbit
   tests, task #81) OR honest ledger.

After the c-routing fix, remaining regressions: ~2 jax + ~5 torch (buckets 2 & 3).

## Two decisions for the next session
- **torch test-side** (bucket 3): backend-agnostic reductions in the existing tests vs ledger.
- **Rollout order**: Vertical unbound-mask, then Harmonic (promote-at-entry), Staeckel/grids/
  IsochroneApprox (dispatch + triage), then the honest ledger regen (un-ledger only verified-
  backend-native passes). C `c=True` stays ledgered.

## Checkpoint (overnight, commits c09f160e..a9c40afd on feat/aa-backend-native)

Done & pushed:
- Resolved-namespace dispatch: **Spherical, Vertical, Adiabatic** (drop is_backend_array gate).
- Shared **torch root-find clean-forward** fix (_torch/optimize).
- **c=True routing** fix (Adiabatic backend gate now respects explicit C requests) — cleared 4 jax + 2 torch regressions.
- **Vertical J/Omega unbound mask** (9999.99 sentinel) in the 3 backend methods.
- numpy byte-identical; Pillar-2 148/148 each backend; 33 jax + 16 torch ledgered tests now genuinely backend-native.

Remaining (next push), per the agreed plan (Continue rollout + backend-agnostic test reductions):
1. **torch `numpy.mean/fabs(output)` tests** (`*_conserved_EccZmaxRperiRap*`): wrap outputs in the
   existing `_to_numpy()` helper in test_actionAngle.py (user-approved: backend-agnostic reductions,
   like the orbit tests). Focused test-file pass.
2. **Vertical edge cases**: `test_actionAngleVertical_unbound` (degenerate angle-sentinel formula)
   and `test_physical_vertical` (physical-units decorator returns a tuple under backend) — match or
   ledger honestly.
3. **Harmonic** (closed-form): promote-at-entry so numpy inputs run jax/torch-native (watch scalars).
4. **Staeckel / StaeckelGrid / AdiabaticGrid / IsochroneApprox**: apply dispatch + measure-then-triage.
5. **Honest ledger regen**: un-ledger the verified backend-native passes (33 jax + 16 torch already;
   more as modules land); ledger genuine backend gaps + C(`c=True`) paths.

Net so far: the principle is proven and productionised on 3 of ~8 AA modules, with a clean pattern
(dispatch + promote + the shared torch fix) the rest follow. No PR open; branch is review-ready.

## Rollout (continued) — 7 of 8 modules dispatched

Applied the resolved-namespace dispatch to all AA modules that have a vectorised
`_*_backend` path:
- **Spherical, Vertical, Adiabatic** (commit c09f160e) + Adiabatic c=True routing fix (ff403a5c).
- **Vertical** J/Omega unbound-sentinel mask (a9c40afd).
- **Harmonic** _evaluate promote-at-entry (matches _actionsFreqs).
- **AdiabaticGrid, StaeckelGrid, IsochroneApprox** (dd667f85) — grid interpolation + orbit-STM
  paths; torch: 12 more previously-ledgered tests xpass, 0 regressions.
- **Backend-agnostic test reductions**: the EccZmaxRperiRap conserved helper now `_to_numpy`s the
  AA outputs before numpy reductions (fixes the torch numpy.mean-on-tensor cases) — byte-identical.

Only **Staeckel** (the C-wrapped one: 13 is_backend_array gates, 0 `_backend` methods — its forward
is C, no jax-native vectorised path) is left undispatched; per the agreed principle its `c=True`
forward stays ledgered until analytic derivatives land. (A pure-Python Staeckel forward exists but
isn't vectorised into `_backend` methods; vectorising it is a follow-up.)

**Honest ledger regen** (GALPY_BACKEND_XFAIL_REGEN=1) running for test_actionAngle under jax+torch
to produce the true failing set → the new AA ledger un-ledgers the now-genuinely-backend-native
passes and keeps only real gaps (Staeckel C, Vertical unbound-angle/physical, etc.).

Known residual backend gaps to ledger honestly: Vertical `unbound` (angle sentinel) + `physical`
(units decorator tuple); Staeckel `c=True` forward. Everything else in the 7 modules now runs the
backend for real.

## Correction: Staeckel c=False IS backend-native (8/8 modules done)

The user caught my error: I'd written off Staeckel as "C-only" because `grep "def _.*_backend"`
returned 0. Wrong — Staeckel's c=False path is **fully vectorised + backend-agnostic** (the
module-level `_staeckel_actions` / `_staeckel_actions_freqs` / `_staeckel_prep` helpers from
**#1013 / #1014 / #1015**, with #1010's pure-Python freqs/angles). It just selected `xp` via
`get_namespace(R) if is_backend_array(R) else numpy` — the same hollow gate. The C path is only
the `c=True` branch.

Fixed all 4 input-dispatch sites (`_evaluate`, `_actionsFreqs`, `_actionsFreqsAngles`,
`_uminumaxvmin`) to resolve the active namespace + promote. Now Staeckel c=False runs genuinely
backend-native: **jax 17 pass + 2 xpass, torch 3 pass + 5 xpass, 0 regressions**; numpy
byte-identical (65 pass). Commit bbff62f8.

**So all 8 AA modules now run the existing tests on the backend for real**: Spherical, Vertical,
Adiabatic, Harmonic, AdiabaticGrid, StaeckelGrid, IsochroneApprox, Staeckel(c=False). Only the
Staeckel/Adiabatic **c=True C forward** stays ledgered (no jax-native C), per the agreed principle.

Re-running the honest ledger regen on the complete code now.

## Honest ledger applied + 1 fix (2026-06-30)

Regen ran the full test_actionAngle.py per backend in REGEN mode (xfail off):
**jax 69 fail / 137 pass / 1 skip, torch 121 fail / 86 pass, 207 collected, 0
collection errors** — clean and authoritative. Built the new AA ledger by
consolidating ALL scattered `tests/test_actionAngle.py` entries (main sorted
blocks + milestone-appended + the c=False FREQS/ANGLES comment block) into the
two sorted jax/torch blocks, replacing 99 jax + 166 torch with the regen sets.

**Verification (verify-burndown principle):**
- jax regen finished 10:29, the test_actionAngle.py `_to_numpy` reductions edit
  landed 10:33 → jax regen was STALE w.r.t. those edits (torch regen 10:36 was
  current). The `_to_numpy` edits can only make tests pass (never break), so
  jax_new is at worst a safe superset. Verified live: re-ran the 12 most
  boundary-relevant affected tests under jax against the current tree →
  **2 failed (both ledgered), 10 passed (all claimed-pass)** → ledger consistent.
- The 2 "newly-exposed" adds turned out to be 1 fix + 1 honest ledger:
  - **test_actionAngleVertical_unbound — FIXED.** `_actionsFreqsAngles_backend`
    returned the raw (garbage) angle for unbound orbits; numpy sets angle=9999.99
    BEFORE `angle*=Omega`(=9999.99) then `%2pi`, so the sentinel is
    `(9999.99**2)%(2pi)≈3.0047`, not 0. Added `xp.where(unbound, (9999.99*9999.99)
    %(2*pi), angle)`. Passes jax+torch; numpy untouched (backend-only branch).
  - **test_physical_vertical — ledgered (honest).** `aAV(x,vx)` with ro/vo returns
    a 1-element TUPLE on numpy too (physical decorator wraps a len-1 1D-action
    output in `(val,)`); the test passes on numpy ONLY because numpy broadcasts
    `tuple - ndarray`. jax/torch raise `TypeError` on `tuple - array`. Pre-existing
    decorator quirk, not introduced by the dispatch; fixing needs a shared-decorator
    change (numpy byte-identity risk) → out of scope, ledger honestly.

**Final ledger: jax 99→68, torch 166→120** (net −77 AA lines, 1286→1209). Only
AA lines + the orphaned c=False comment changed (non-AA lines byte-identical).
Remaining ledgered: Staeckel/Adiabatic c=True C forward, the Inverse modules (not
in the 8-module set), and ~1e-7 torch precision gaps the real backend path now
exposes (linear_angles / kepler / against_orbit). Follow-ups: redesign the Inverse
modules; chase the torch 1e-7 gaps; consider fixing the 1D physical-decorator tuple.

## Coercion unification folded into #1038 (2026-06-30)

User review caught that the AA dispatch hand-rolled `R,... = (xp.asarray(a) for a
in (...))` for input coercion, diverging from the house style. A cross-subsystem
survey (workflow wf_9e2bc78f-ae5) confirmed: `get_namespace` is the universal
resolver (honors `use(...,force=True)` via a contextvar -- there is NO
`force_backend` fn), but coercion is forked -- potentials/df use
`coerce_coords` at the boundary decorator; orbit + (most) AA hand-roll
`xp.asarray`; Harmonic/Isochrone already use `promote_scalars`. Inline asarray
skips device-anchoring (numpy scalar beside a GPU array -> default device) and
the float64-scalar dtype policy.

Unified all 7 split-structure AA modules (Spherical/Vertical/Staeckel/Adiabatic/
both Grids/IsochroneApprox) onto `promote_scalars(xp, ...)` -- matching
Harmonic/Isochrone and aligned with potentials' coerce_coords (same _coerce.py
home). Kept inside the `if xp is not numpy` gate (numpy never reaches it ->
byte-identical), kept the isinstance(float) SHAPE preamble (backend methods
index [:, None] -> need 1-D, which the dtype-only coercers don't give),
preserved Staeckel's device-aware phi/delta folds + IsochroneApprox internal
xp.asarray. Verified: test_backend_actionAngle 299 passed jax + 299 torch
(parity + gradcheck). Commit cc1d51e0.

## #1038 CI fallout — triage + fixes (2026-06-30, commits 7c244c2e..bd35905e)

The AA-native CI surfaced that the redesign's blast radius reaches AA CONSUMERS
(jax only — torch consumers were already ledgered). 5 distinct issues, all
caused/exposed by the PR:
1. brentq[torch] regression — my detach-forward guard ignored bracket-endpoint
   grad. FIX (7c244c2e): also reparam when a/b requires grad. test_backend_optimize
   24 passed.
2. test_actionAngleAdiabaticGrid_basicAndConserved_actions — CI 300s timeout that
   my fast-box regen (no --timeout) PASSED, so I wrongly dropped it. -> slow_skip.
3-5. diskdf-consumer exposures: diskdf.sample -> orbit(t) with a 0-d jax time hit
   Orbits._call_internal len(t) (TypeError: len() of unsized object). FIX (4112620b):
   treat 0-d backend time as scalar (numpy byte-identical). After the fix, of the
   10 diskdf.py sample tests: 5 PASS-but-slow (39-238s) -> slow_skip; 5 still FAIL
   at a deeper diskdf-jax gap -> xfail. test_galpypaper::test_diskdf pass-slow ->
   slow_skip; 2 test_quantity::test_diskdf_* fail -> xfail.

Ledger (bd35905e): slow_skip += 5 diskdf + galpypaper + basicAndConserved; xfail +=
5 diskdf + 2 quantity. No slow_skip/xfail overlap; all node ids collectible.
Key lesson reinforced: local fast-box regen CANNOT see CI's 300s per-test timeout
(numpy_island removal flips consumers eager-slow). Next: Inverse modules
(Harmonic/Isochrone/Vertical) after #1038 merges (user steer).

## #1038 MERGED + Inverse follow-up #1039 (2026-06-30)

#1038 (AA backends run for real) merged to feat/backends as squash b9454149 —
all-backend CI green. Then started the in-scope Inverse work (Harmonic/Isochrone
ONLY; VerticalInverse stays _reject_backend-guarded, user-confirmed out of scope;
Torus excluded).

#1039 (feat/aa-inverse-reductions): the 5 ledgered torch Inverse tests were purely
TEST-SIDE — numpy.amax/all(numpy.fabs(orb.X - inverseOut)) on torch tensors (torch
doesn't duck-type numpy reductions' axis=/out=; jax does, hence jax already passed).
The 2 _orbit tests also mixed numpy ts * torch xvFreqs and built Orbit from torch
ICs. Fix = test-only _to_numpy reductions + bring xvFreqs output to numpy before the
orbit plumbing (inverse __call__ still runs the backend → non-hollow). All 5 pass
numpy+jax+torch. Harmonic/Isochrone Inverse now FULLY un-ledgered (0 jax/torch);
VerticalInverse keeps its 48 entries. No library code change.
Lesson: do NOT use @numpy_island (the closed-#1033/#1034 anti-pattern); the Inverse
modules were already partially resolved-namespace-migrated, only the tests lagged.
