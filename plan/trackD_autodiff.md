# Track D — Autodiff wrappers + in-backend integrators

Branch: `feat/backends`. Detail in master plan §"Track D". Depends on Track B's
STM (Pvar-rhs, on main) and Track 0 (P1, P0-util).

- **Pjax-orbit / Ptorch-orbit (C-STM):** forward = C orbit (jax via
  `jax.pure_callback`, returning STM residual M; `vmap_method="expand_dims"` reusing
  the multi-orbit C loop); backward = `Mᵀ·λ̄` (host-side, no backward C call).
  `jax.custom_vjp` / `torch.autograd.Function` (gradcheck mandatory). In
  `galpy/backend/_jax|_torch/orbit_stm.py`; `Orbit.integrate` routes by IC namespace;
  API numpy-identical. jax FFI deferred (`jax.ffi` is the current API in 0.9.2;
  pure_callback is enough for CPU-bound serial ODE).
- **Pinbackend (first-class diffrax/torchdiffeq):** in-backend ODE integration of the
  backend-agnostic forces → higher-order autodiff (Hessian-of-orbit) + parameter
  gradients for all potentials via AD; runtime extras; resolver dispatch; cross-validate
  vs C-STM + Liouville.

## Performance note — resolve the namespace once per call (deferred opt)

`get_namespace(*arrays)` costs ~3 us/call and adds ~16% to a *scalar* pure-Python
force eval (measured on MWPotential2014 `_evaluateRforces`: ~40 us/call, of which
~6 us is two `get_namespace`/`namespace_from_arrays` resolutions). This does NOT
affect orbit integration via the C integrators (they call the C force functions
directly, never the namespaced Python `_evaluate*` layer), so it is not on the
hot path for `Orbit.integrate`. Where it bites is pure-Python evaluation loops
(plotting, Python-only methods, the in-backend diffrax/torchdiffeq RHS, df
quadratures).

Optimization (do during the orbit/df backend migration, not urgently): resolve
`xp` ONCE at the public-method / RHS-entry boundary and thread it down through the
private compute layer (mirrors how the Python integrators use the underscored,
decorator-free `_evaluateRforces`/`_evaluatephitorques`/`_evaluatezforces` to skip
the unit decorators). Expected ~15% on those Python paths; zero risk to the C
paths. Discovered while profiling the run-notebooks slowdown (which turned out to
be an unrelated slow live Gaia archive query in staeckel.ipynb, NOT the
namespace-swap).

## ⛔ CI obligation — RE-ENABLE the deferred jax orbit shard when Track D lands

The all-backend CI harness (PR #956, `feat/pillar1-backend-ci` → `feat/backends`)
**defers the ENTIRE jax orbit test surface to Track D**. Until the in-backend
integrator + C-STM + jit routing make jax orbit integration fast, the shard is
unrunnable: `tests/test_orbit.py` + `tests/test_orbits.py` collect **579 jax
tests** but only ~120 run before the 4500 s session cap, so the shard
self-truncates every run (never completing, never yielding a stable ledger).

**Deferral mechanism:** two FILE-LEVEL lines in `tests/backend_xfail`-adjacent
`tests/backend_slow_skip.txt`:
```
jax tests/test_orbit.py
jax tests/test_orbits.py
```
`conftest._matches` treats a path with no `::` as "every test in this file";
matched tests are `pytest.mark.skip("backend-slow-skip: …")`, counted in the
status report's separate **`deferred`** burndown bucket (NOT xfail).

**To re-enable (DO THIS as the closing step of Track D):**
1. Delete the two lines above from `tests/backend_slow_skip.txt`.
2. Run the jax orbit shard(s) to completion (now fast) and harvest a real ledger
   for whatever still fails — `GALPY_BACKEND_XFAIL_REGEN=1` regen, then add the
   genuine failures back to `tests/backend_xfail.txt` (jax). The 25 jax
   `test_orbit`/`test_orbits` ledger entries removed in the deferral commit
   (c35ba05a) are the pre-Track-D failures — most should now PASS; re-harvest
   rather than blindly restoring them.
3. Confirm the jax orbit shard completes under the session cap and the run is
   green; the `deferred` count for orbit drops to 0.

Same obligation pattern (different track): the jax **spherical-DF** tests are
deferred to **Track F** (Pdf.2) via per-nodeid `jax tests/test_sphericaldf.py::…`
slow-skip lines — re-enable those when the spherical-DF backend migration lands
(see `trackF_df.md`).
