# PR0 — backend resolver pilot (Plummer + Isochrone)

Branch: `feat/backends` (off `main`). Goal: validate the whole multi-backend
approach end-to-end on the smallest meaningful slice before scaling to ~86
potentials. "Not even a first PR" — a spike that becomes PR0.

## Approach
- New `galpy/backend/` package:
  - `_namespaces.py`: `namespace_for_name` (str→module), `namespace_from_arrays`
    (data-first detection; normalizes numpy-compat wrapper → **plain numpy** for
    the byte-identical guarantee), vendored fallback when array-api-compat absent.
  - `_resolver.py`: `get_namespace(*arrays, xp=None)` with precedence
    `xp= → data → contextvar → numpy`; `use()` context manager; `backend()`;
    `set_default_backend()`.
  - `__init__.py`: re-exports `get_namespace, backend, use, set_default_backend`.
- `galpy/util/_optional_deps.py`: added `_TORCH_LOADED`, `_ARRAY_API_COMPAT_LOADED`.
- Migrated `PlummerPotential` and `IsochronePotential`: only methods that call a
  numpy *function* get `xp = get_namespace(R, z)` and `numpy.<fn>`→`xp.<fn>`
  (`sqrt`, `arctan`); `numpy.pi`→`math.pi` (a plain float, namespace-free). The
  force/Hessian methods (`_Rforce/_zforce/_R2deriv/_z2deriv/_Rzderiv`) are pure
  arithmetic and needed **no change** — confirming the "only swap what calls
  numpy" convention. Removed Plummer's unused `_rforce_jax` stub.
- `tests/test_backend_pilot.py` (70 tests): value parity numpy/jax/torch (rtol
  1e-12); grad-vs-finite-difference; the identities `d(_evaluate)/dR == −_Rforce`
  and `d(−_Rforce)/dR == _R2deriv` under AD; jax jit+vmap; torch dtype
  preservation; public-method (decorator pass-through) parity; resolver
  precedence/data-first/unknown-backend.

## Results
- `pytest tests/test_backend_pilot.py` → **70 passed** (numpy + jax 0.9.2 + torch 2.12 CPU).
- No regression: `test_energy_jacobi_conservation[Plummer/Isochrone]` pass;
  `test_actionAngle.py -k isochrone` (17) pass.
- Decorator pass-through confirmed: public `pot.Rforce(jax/torch array)` matches numpy.
- Data-first confirmed: a numpy array inside `with use("jax")` resolves to numpy.

## Dead-ends / notes
- Considered globally enabling `jax_enable_x64` at `galpy.backend` import — rejected
  (intrusive global side effect on the user's session); enabled in the test
  fixture instead. Revisit: the resolver could enable x64 lazily only when jax is
  first selected, but document it rather than mutate silently.
- `array_api_compat.array_namespace(numpy_array)` returns the compat wrapper, not
  plain numpy; we normalize it back to plain numpy so the numpy path is identical.
- torch float32 in → float32 out (dtype preserved); galpy's tolerances assume
  float64, so the `--backend torch` full-suite run will use float64 tensors.

## Next
- P1: productionize (config `[backend]` section, special-functions native-preferring
  router, AST convention guard, `--backend` pytest option). Then the P2.x potential
  fan-out starting with `SphericalPotential` (the family consolidation point).
