# Orbit-under-backends burndown (jax un-defer + torch read-path reductions)

Burndown of the `test_orbit.py`/`test_orbits.py` failures under forced jax/torch,
following the Track D in-backend integrator + accessor work. Two landed pieces:

## #998 (MERGED 2026-06-23) — jax orbit un-defer
Un-deferred the whole jax orbit suite (0 → 606 tests run). The Python ODE
integrators (odeint/dop853/leapfrog) step **per force call** under eager jax/torch
(~83k scalar force calls/test, ~1.1ms each — half irreducible jax dispatch, half
galpy per-call overhead), so they are unrunnable eager. Helper `_backend_integrators`
restricts the integrator lists to the `*_c` C integrators under a non-numpy backend
(numpy keeps the full list → byte-identical). Duration-balanced 5-way orbit shard
split (`tests/.test_orbit_durations` = max(jax,torch) per test) + slow-skip of the
66 heaviest jax tests keeps every group under the 75-min cap. Also broadened the
in-backend diffrax/torchdiffeq `_POTS` to 24 potentials.

## #1001 (open) — backend-agnostic orbit read-path reductions
**Approach.** Data-driven: ran `test_orbit.py --backend=torch` (regen, C integrators)
and clustered the 199→278 failures by one-line traceback. The dominant
`max()/min()/any()/abs() received an invalid combination of arguments` errors are
galpy calling `numpy.<reduction>(x)` on a **backend tensor**: under the forced-torch
fixture (`backend.use("torch", force=True)`) an integrated orbit's phase-space arrays
ARE torch tensors (`type(o.r(o.t)) == Tensor`), so:
- `Orbit.e/rap/rperi/zmax` → `numpy.amax/amin(rs)` crash (zmax's `numpy.fabs` instead
  silently coerces Tensor→numpy, dropping backend/grad);
- `Orbit._warn_radii_outside_interpolation_range` → `numpy.any(...)` crashes on
  interp-backed potentials (King/Multipole/interpRZ);
- `Potential.normalize` → `numpy.fabs(Rforce(1,0))` crashes when Rforce is a tensor.

Fix = resolve namespace from the data (`get_namespace`) and use `xp.max/min/any`;
`normalize` uses **builtin `abs()`** (`__abs__` → stays float/ndarray/Tensor/jax.Array,
byte-identical to `numpy.fabs` on the numpy scalar).

**Results.** 12 forced-torch `test_orbit.py` tests flip fail→pass (verified
**serially**, not via xdist): 8× `energy_jacobi_conservation[*Multipole*/King]` +
4× `orbinterp_reset_*` (all via the interp-range warning). 0 real regressions.
numpy byte-identical (56 numpy e/rap/rperi/zmax tests unchanged). +backend coverage
tests (diffrax/torchdiffeq accessors; normalize under forced jax/torch).

**Dead-ends / gotchas.**
- First `normalize` attempt used `xp.abs(rf)` with `xp=get_namespace(rf)`. Under
  `force=True`, `get_namespace` returns torch **even for a python-float `rf`** (RazorThin's
  scalar Rforce), so `torch.abs(float)` crashed — a *regression* the measurement caught.
  Builtin `abs()` is the right tool (dispatches on the value, not a forced namespace).
- The accessor fixes flip 0 forced-torch `test_orbit.py` tests directly (those tests
  use C integration → numpy internal state via the helper); they matter for the
  **diffrax/torchdiffeq** orbit path (backend internal state) — hence the backend tests.
- **xdist + segfault = false flips/regressions.** Two `*_dxdv_planar_c_vs_python` tests
  segfault under torch **on the unmodified base too**. Root cause: the dev box's **CUDA
  torch build** (`2.12.0+cu126`) segfaults these dxdv-planar-C tests even with
  `CUDA_VISIBLE_DEVICES=""`; **CI uses CPU-only torch** (`--index-url .../whl/cpu`) where
  they PASS (not ledgered, torch shard green). So this is *local* noise. A segfault kills
  the xdist worker and pytest mis-attributes the failure to whichever tests shared it →
  they appeared as "regressions" in the 2-pass xdist measurement. Always re-verify
  flips/regressions **serially** (the segfault only hits the dxdv tests); for CI-faithful
  local torch runs, install CPU torch in a venv. [follow-up:
  root-cause the torch dxdv-planar-C segfault]
- Worktree `libgalpy.so` copied from the main repo (different commit) → ABI mismatch
  segfaults the actionAngle Staeckel C path (analytic ecc/rap); rebuild the worktree's
  own `.so` before any analytic-AA validation.

**Ledger.** Code-only PR (no `backend_xfail.txt` change) → independent of #1000/#999.
The 12 flipped entries (+ the ~35 helper-flipped torch entries from #1000) drop at the
next **orbit-milestone** ledger regen (separate PR, after #1000 merges).

## Orbit close-out (post-#1001): triage → "3 then 2"

Clean CPU-torch re-characterization (CI-faithful; the earlier 270-fail set was
inflated by a stale-`.so` ABI artifact — 304 undefined-symbol AttributeErrors).
**186 real torch orbit fails.** A triage workflow (8 cluster agents + synthesis)
classified them: ~19 genuine orbit-infra reductions; ~119 DEFER (Python-eager 64,
Track-A non-migrated potentials 39, Track-E action-angle 15, dissipative 1, flaky
1); ~47 **test-side** (the orbit code is RIGHT — under forced torch the accessors
correctly return torch tensors and the *existing tests* call `numpy.amax/all/std`
on them).

Key architecture clarification (user Q "shouldn't outputs stay backend?"): yes,
for the **in-backend** path; the fast **C integrator is a numpy/ctypes boundary**,
re-lifted to a differentiable backend array by the C-STM wrapper. `force=True` is
a test stressor — the crashes are at the seams where galpy legitimately holds
numpy internally. `_resolve_accessor_namespace`'s coercion is **load-bearing**
(derived accessors' cos/sin need torch), so the accessor-policy "return numpy"
alternative is NOT viable → fix the tests, not the accessors.

User chose **"3 then 2"**:
- **#1002 (3):** 34 test-side comparison/energy/scalar tests made backend-agnostic
  via a shared `_to_numpy(x)` helper (conftest, imported by test_orbit/test_orbits;
  renamed from `_anp` per review). 34/34 torch-green + 34/34 numpy byte-identical;
  scope-verified. MERGED 2026-06-23.
- **#1003 (2):** route `Orbit.integrate(<dxdv C method>)` on a jax/torch IC →
  C-STM (`d o.e()/dR0` over a C solve == FD, jax+torch); 6D+C-3D-Hessian → C-STM,
  else in-backend diffrax/torchdiffeq fallback (pre-checked `_check_c(dxdv3d)`);
  numpy byte-identical. 28 routing tests incl. full 6×6 STM jacobian. Open.
  [Track-G speed check = backlog #58; C param-grads deferred.]

Remaining orbit reds = Track-A potentials + Track-E action-angle + Python-eager.
The ~19 orbit-infra SOS/bruteSOS/dissip-warning reductions + the orbit-milestone
ledger regen were NOT done (user deprioritized option 1).

## Orbit-milestone ledger prune — #1006 (open 2026-06-24)

After #1003 + #1004 merged, ran the **full-matrix backend regen**
(`workflow_dispatch regen=true`, run 28083178345) off the 2026-06-22 milestone-2
baseline (run 27949897402). **All 45 jax+torch shards ran to completion** (no
session-timeout truncation → the failing set is complete, not the
[triage-gather-misses-timed-out-shards] blind spot). Built the new failing set
from the 22+22 per-shard `backend_xfail_new_<be>.txt` fragments; diffed vs the
base ledger by mirroring the conftest `_matches` base/param/file logic.

**Delta: jax 289→287 (-2), torch 1029→961 (-68); 70 drops, 0 adds.**
- `test_orbit.py`: 31 torch + 2 jax · `test_orbits.py`: 31 torch
- incidental (same 6/22 → 6/24 window): Multipole 2, dynamfric 2, FDM 1,
  actionAngle 1 (all torch).
**0 adds** = the suite was genuinely all-green; nothing un-ledgered.

**Gotcha (caught):** the **main repo `/home/bovy/Repos/galpy` was STALE**
(c6014372 → old 2026-06-14 ledger, 296/1683). The real base is
`origin/feat/backends` (28c55f29 → the 2026-06-22 milestone-2 ledger, 289/1029).
Diffing against the stale main-repo ledger falsely showed 815+ drops / 98 adds
(incl. phantom noninertial/kuzminkutuzov "adds"). Always diff against
`git show origin/feat/backends:tests/backend_xfail.txt`, never the main worktree
([[worktree-stale-base-reverts-merged-prs]]).

**Method choice:** surgical prune (remove exactly the 70 now-passing entries,
keep the rest verbatim) rather than a full regen-snapshot replace — the diff IS
the burndown delta (-70 lines), reviewable, and can't un-mask/re-express kept
entries. Drops are CI-verified passing (regen ran them xfail-free); #1006's own
backend-tests run is the hard re-check.
