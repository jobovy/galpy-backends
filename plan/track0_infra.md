# Track 0 — Backend infrastructure

Branch: `feat/backends`. See master plan for the full dependency graph.

## PRs
- **PR0 pilot** ✅ — `galpy/backend/` resolver + Plummer/Isochrone + `test_backend_pilot.py`. (see `logs/PR0_pilot/`)
- **P1 — galpy.backend package, numpy no-op.** Productionize:
  - config `[backend]` section in `galpy/util/config.py` (+ `set_default_backend` writing the rc); seed contextvar default at import.
  - `galpy/backend/special/` **native-preferring router**: per `(backend, fn)` use native (`jax.scipy.special`/`torch.special`/`scipy.special`) if present, else `_fallback/<fn>.py`; capability test pins the shrinking fallback list + asserts native-vs-fallback agreement.
  - `galpy/backend/linalg.py` (det/solve/matmul shims for STM/Liouville).
  - AST convention guard `tests/test_backend_conventions.py` (opt-in allowlist; bans bare `numpy.` in compute methods of migrated potentials; flags `isinstance(arg, ndarray/float)` and `out[...]=` subscript-assign).
  - numpy pass-through → existing suite unchanged.
- **P0-util** — backend-clean `coords.py` (fix in-place clip/wrap at 249-250/253/330/610/708/780 via `xp.where`/`clip`; replace `isinstance`/shape checks with `getattr(x,'ndim',0)`; `scalarDecorator`/`degreeDecorator` backend-safe) + the rare `physical_conversion` numpy-array-`fac` branch (conversion.py ~1008-1013).
- **Pspecial** — pure-backend `assoc_legendre` (Bonnet recursion; replaces `util/special.compute_legendre` on the MultipoleExpansion path), `gamma/gammainc/gammaincinv`, `hyp2f1`, `erf`. ~1e-12 vs scipy + grad-vs-FD.
- **Pci** — `pytest_addoption(--backend)` + session autouse fixture (`with galpy.backend.use(<backend>)`, jax x64); new build.yml jax/torch jobs + full-suite re-run; `REQUIRES_TORCH` install; extras; coverage/tolerance ledger.

## Design notes (verified)
- Decorators pass jax/torch arrays through untouched; only the rare numpy-array `fac` branch in `physical_conversion` needs a guard.
- `SphericalPotential.py` is the single 3D→1D consolidation point — migrate it to cover the whole spherical family.
- Device/dtype preserved by anchoring on inputs (`*_like`, `xp.where`); never bare `xp.zeros(...)`.
