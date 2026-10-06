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

## Coercion boundary: `@backend_input` (#1196, merged 2026-07-25)

Backend coordinate coercion lived inside the legacy `potential_physical_input`
unit decorator (a `coerce_backend=True` parameter). Extracted into its own
decorator, `galpy/backend/_input.py`, stacked just inside a now units-only
`potential_physical_input`; the four `coerce_backend=False` opt-outs dropped
their flags. Supersedes the abandoned per-method `@backend_kernel` (#1194) and
the per-method jit seam (#1195), both closed.

**Explicit coords, not guessed.** `@backend_input("R","z","phi","t")` at 73
potential-side entry points. The old block coerced EVERY positional arg after
the first while its kwarg path was whitelisted — so the same parameter was
protected as a keyword and coerced as a positional: `dR`/`dphi` (derivative
ORDERS), `forceint`/`forcepoisson`, `M`, `eigenval`, `nsigma`/`romberg`, and in
evolveddiskdf `integrate_method` (a string) + `grid` objects. Declared names are
validated against the signature at DECORATION time (typo → ImportError).

**Two real bugs the switch exposed (both fixed):**
- A SEQUENCE coordinate (`v=[vR,vT,vz]`, the dissipative-force velocity) was
  stacked into one array by a single `asarray`, which **detaches grad-carrying
  elements from the autograd graph** — it broke 6 NonInertialFrameForce
  grad-vs-FD tests. Now coerced element-wise, container preserved.
- Mixed backend+numpy coordinates: `Orbits.E` passes a numpy `t=` alongside
  torch `R`/`z`; probing the mix raises `Multiple namespaces for array inputs`,
  which surfaced as a failure **on the numpy path**. The namespace now follows
  the backend coordinates and the numpy ones are coerced across.

**Dead-end: removing the migration gate.** Tried at the user's request, measured,
REVERTED; survives as `_backend_ready`, marked TEMPORARY. Four breakages:
1. df — `evolveddiskdf` grid `deriv=` raises `NotImplementedError` on jax/torch and
   `qdf.sample()` crashes: the guard had made **all 48 df decorations dead
   no-ops** (`_check_backend_compatible` returns False for any non-Force), so
   going live turned WORKING calls into raising ones. Not hollow-green cleanup.
2. `interpRZPotential` — scipy-interpolation island; out-of-grid fallback
   diverges at the grid edge: at (R,z)=(0.5,0.0) force off 3.3%, 2nd deriv 32%,
   both float64 (branch difference, not precision).
3. `MovingObjectPotential` under jit — `self._orb.R(t)` is a numpy trajectory
   lookup; a coerced `t` arrives as a `DynamicJaxprTracer` (probe: baseline gets
   a plain `float`). **Do NOT retry `as_numpy(t)` inside the class** — it raises
   on grad-requiring torch tensors and made things strictly worse (1 → 7 fails).
4. NonInertialFrameForce grads — the real bug above.

**Method note that generalizes:** forced-backend runs (`--backend jax/torch`)
short-circuit `get_namespace` (forced beats data), so they NEVER exercise the
mixed numpy/backend probe. Only the UNFORCED `tests/test_backend*.py` shard does
— it is what caught the `Multiple namespaces` bug and the 4 orbit-STM failures
that 1320-test forced runs passed clean. Always run that shard for boundary work.
Also: the xfail-ledger only applies when `--backend` is jax/torch, so failures in
that shard CANNOT be ledgered — they must be fixed.

**Scope is auditable, not historical.** `_UNDECORATED_MODULES` /
`_UNDECORATED_BY_DESIGN` in `tests/test_backend_input.py`: every public
coordinate-taking entry point must be decorated or listed WITH its blocker,
test-enforced. Those lists are the burndown list. Known limitation: the audit
reads NAMED parameters, so varargs entry points are invisible to it —
actionAngle's public API is `actionsFreqs(*args)`/`actionsFreqsAngles(*args)`/
`EccZmaxRperiRap(*args)`/`__call__(*args)`, and AA handles backends with 80
`get_namespace` + 38 `promote_scalars` and **zero** `coerce_coords`. Covering
varargs needs positional-slot declaration (out of scope).

**Numbers:** numpy byte-identical (hash over forces/potentials/densities/masses,
single + list potentials, incl. the `dR=1` control-kwarg path); numpy path
slightly FASTER than before (the old block ran two deferred imports on every
numpy call: `Rforce` ~9.4 → ~7.4 us, interleaved+pinned min-of-N — naive
back-to-back timing on the 72-core box is pure noise). test_potential 1320 tests
0 fail under numpy/jax/torch; unforced test_backend shard 6251 tests 0 fail;
tests/test_backend_input.py 24 tests; **154/154 CI checks green; no ledger change**.

**Follow-ups (ordered):** (1) make the df boundary live — blockers:
`evolveddiskdf` grid `deriv=`, `quasiisothermaldf` sampling + adiabatic
action-Jacobian C callback on 0-d arrays, `sphericaldf.sigmar`/`sigmat`
returning NaN for a coerced radius (breaks kingdf); (2) migrate
`interpRZPotential` + `MovingObjectPotential` → then DELETE `_backend_ready`;
(3) port the scipy root-finders (`rl`/`rE`/`LcE`/`lindbladR`/`vterm`/`rhalf`/
`mvir`/`zvc`) onto the backend root-find helpers, then decorate; (4) audit the
266 inline `coerce_coords` sites — almost all in `galpy/potential/*` leaf classes
inside `_evaluate`/`_Rforce`, reachable WITHOUT a decorated entry point (Python
integrator, wrappers, `_nodecorator`), so most are NOT redundant; (5) boundary-jit
test dimension (jit the whole public method; the decorators trace away).

## 2026-10-06 — dtype policy, jit-mode key, CI runner header

- **float64 by default (#1613).** Policy (Jo): Python floats, numpy inputs and
  everything galpy creates are float64; a USER float32 tensor stays float32 in and
  out (compute in f64 where needed: `at_least_float64` / `match_input_dtype`).
  Mechanism: scoped default dtype (`float64_default`, nesting-safe, skipped under
  torch.compile) entered by `use("torch")`, the `@backend_input` wrapper and
  `float64_default_if_torch_args` on df/actionAngle/Orbit entry points. Found via
  a TorchFunctionMode detector of dtype-less factory calls (local tool).
  CI-only failures fixed: a closure over `sys.modules` made dynamo trip typing's
  `_UnionGenericAlias` DeprecationWarning on py3.14; tensor-amp * ndarray via the
  deprecated `__array_wrap__` (numpy 2.5) in sphericaldf._vmax_at_r.
- **jit-mode static key (#1625).** `_object_key` was id(obj) + own scalars:
  (1) stale jax traces after a WRAPPED potential's normalize() (2x wrong
  forces); (2) a retrace per call from fresh toPlanar() wrappers
  (vcirc/epifreq: 293/466 compiles for ExpTruncNFW analytic_ecc). Now type +
  attributes: scalars by value, galpy sub-objects recursively, everything else
  by HELD identity (`_Ident`). **Dead end, caught by CI:** recursing into
  anything with `__dict__` recursed into jax arrays/torch tensors (data held
  elsewhere) -> every array keyed alike -> time-dep vs static SCF/Multipole
  shared traces (0.7 wrong). ExpTruncNFW jax-jit 501 s -> 260 s.
- **CI runner header (#1620).** pytest header prints CPU model, SIMD
  (avx2/avx512f/amx) and inductor vec ISA; ~40% of jobs land on AVX-512 runners;
  #1614's inductor simdlen=256 cap confirmed needed (kingdf_W0 passed on EPYC
  9V74 avx512f).
- **Rebase onto main (2026-10-04, #1626 draft for CI).** main 2 ahead (#1596,
  #1611), feat 505 ahead. Plain replay stops at 3/505; used `rebase -X theirs`
  + a hand-resolved reference merge tree; `-X theirs` silently dropped main's
  EOF-appended #1611 test at #1336 (329/505) — fixed at that commit. Tip ==
  old feat + main's HISTORY entry + #1611 test, 0 deletions.
