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
  segfault under torch **on the unmodified base too** (pre-existing C crash). A segfault
  kills the xdist worker and pytest mis-attributes the failure to whichever tests shared
  it → they appeared as "regressions" in the 2-pass xdist measurement. Always re-verify
  flips/regressions **serially** (the segfault only hits the dxdv tests). [follow-up:
  root-cause the torch dxdv-planar-C segfault]
- Worktree `libgalpy.so` copied from the main repo (different commit) → ABI mismatch
  segfaults the actionAngle Staeckel C path (analytic ecc/rap); rebuild the worktree's
  own `.so` before any analytic-AA validation.

**Ledger.** Code-only PR (no `backend_xfail.txt` change) → independent of #1000/#999.
The 12 flipped entries (+ the ~35 helper-flipped torch entries from #1000) drop at the
next **orbit-milestone** ledger regen (separate PR, after #1000 merges).
