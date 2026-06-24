# Track E — action-angle backends (Paa-python)

Namespace-swap the action-angle methods so they evaluate AND differentiate under
jax/torch, numpy byte-identical. Started after the orbit milestone (#1003/#1004).

## Method taxonomy (drives the PR split)
Re-reading the sources corrected the plan's grouping:
- **Closed-form / algebraic** — `actionAngleHarmonic`, `actionAngleIsochrone`.
  Pure arithmetic + masked writes; no quadrature/root-find. → **PR #1005** (this).
- **1D numerical** — `actionAngleVertical` is **NOT** closed-form: it uses
  `scipy.integrate.quad` + `optimize.brentq` in a per-object Python loop. It is a
  1D sibling of Spherical → belongs with the **GL-quadrature + backend-root-find**
  PR, not the closed-form one. (The master plan mis-grouped it as "closed-form".)
- **3D→1D numerical** — `actionAngleSpherical` (quad + root-find + implicit diff).
- **C-backed** — Staeckel/Adiabatic via custom_vjp reusing the C Leibniz integrands.
- **Approx/grid** — `IsochroneApprox` (orbit-STM path), `*Grid` (backend interp).

## PR #1005 (open 2026-06-24) — closed-form Harmonic + Isochrone

**Approach.** `from ..backend import get_namespace, promote_scalars`; per method
`xp = get_namespace(...)`, keep the original `isinstance(R,float)→numpy.array([R])`
scalar-wrap (preserves the 1-element-array output shape on numpy, byte-identical),
then `promote_scalars(xp, …)` (no-op on numpy; coerces float/numpy-array inputs to
backend arrays on torch/jax — handles BOTH the forced-suite numpy-array input and
genuine backend input). `numpy.* → xp.*`; **every in-place masked write →
`xp.where`/`xp.clip`**:
- `Omegaphi[indx] *= -1` → `xp.where(indx, -Omegaz, Omegaz)`
- `coseta`/`sinpsi`/`sinu` `[-1,1]` clamps → `xp.clip` (and guarded `& xp.isfinite`)
- `eta`/`tan11`/`tan12`/`psi`/`u` branch updates → `xp.where`
- `Lz/L` safety clamp → `xp.where`, **moved below** so the returned `Jphi` still
  tracks the clamped `Lz` (the original aliased `Jphi = Lz` then clamped in place)
- `anglephi` `+=`/`-=` split → one `xp.where(indx, Omega-anglez, Omega+anglez)`
- `copy.copy(Omegaz)` dropped (no longer needed).
`actionAngleHarmonic._evaluate` is pure arithmetic — left unchanged.

**Results.**
- **numpy byte-identical** — direct base-vs-port probe (200-pt batches): actions,
  freqs, angles, `EccZmaxRperiRap`, both Lz signs, array + scalar inputs,
  `b ∈ {0, 0.8, 1.5}` → **all 25 batches `array_equal(equal_nan)`**, identical
  RuntimeWarnings (the arcsin/divide warnings at non-inclined orbits pre-exist).
- **jax/torch** — `tests/test_backend_actionAngle.py` (22 tests): parity 1e-12 +
  grad-vs-FD of `Jr`/`Ωr`/`θr` (Isochrone) and `J` (Harmonic) wrt an IC. All green.

**Ledger.** Code-only (no `backend_xfail.txt` change). The existing
`test_actionAngle.py` closed-form cases stay ledgered: their remaining torch/jax
failures are now **test-side** (the tests do `numpy.sum/mean/all` on backend-tensor
orbit accessors — the #1002 pattern) plus cross-checks against the **un-migrated
`IsochroneApprox`** (`astype` on a Tensor) and an `Orbits.py:4276 abs()` accessor
case. The migration itself is correct (the `aAI(...)` call returns fine); these
flip at the **action-angle milestone** regen once the test-side reductions +
IsochroneApprox land.

**CI consequence — streamdf jax timeout (caught + fixed in #1005).** The migration
is a *correctness* win that exposed an eager-jax perf cliff: streamdf's
IsochroneApprox track assembly calls `actionAngleIsochrone` per trackpoint. The
OLD code did `numpy.sqrt(jax_array)` → silently returned **numpy** (fast); the
migrated code returns **real jax**, so the track assembly now does genuine
per-trackpoint XLA dispatch. The `backend jax test_streamdf` shard went **~12 min →
>90 min** (job cap → cancelled → all-clear red); torch/regen/#1006 (without the
migration) all passed at ~12 min, isolating the cause. Fix: file-level
`jax tests/test_streamdf.py` in `backend_slow_skip.txt` (streamdf isn't vectorized
for jax — Track F; 80 jax tests skip in 8 s). The 17 now-redundant test_streamdf
jax xfail entries self-clean at the next regen. **Lesson: a numpy→backend migration
of a function on a hot eager path can flip a shard from accidentally-fast (numpy
fall-through) to genuinely-slow (eager XLA) — watch for shard-timeout, not just
correctness.** Mirrors the test-side entanglement: the migration is right, the
infra (shard balance / Track F vectorization) must catch up.

**Dead-ends / gotchas.**
- Can't replace the `isinstance(R,float)` wrap with `promote_scalars` alone:
  `promote_scalars` passes a numpy/python scalar THROUGH on numpy (stays a float),
  so the scalar-input outputs would become 0-d instead of the original 1-element
  arrays → shape change, not byte-identical. Keep the wrap; promote AFTER it.
- `numpy.pi` kept as a literal (scalar constant; `float - tensor` is fine) rather
  than `xp.pi`, to keep the byte-identity argument trivial.
- `xp.clip(x,-1,1)` verified on torch+jax (MultipoleExpansion already uses it).

## Next increment (Vertical + Spherical) — machinery already exists
`galpy.backend.quadrature` has `fixed_quad`, `gauss_legendre`, `transformed_quad`
(finite GL with a near-singular **interior-point split** — the `xmax-t²` /
`r=r_peri+t²` turning-point substitution), `fixed_quad_semiinfinite`, `nested_quad`;
`galpy.backend.optimize` has `brentq` + `bisect_root` (backend-agnostic). So the
1D-numerical `actionAngleVertical` (`quad` + `brentq` for `xmax`) and the 3D→1D
`actionAngleSpherical` (quad + root-find + implicit diff) map directly onto these
— next Track-E PR, after #1005 lands and the action-angle milestone ledger prune.
