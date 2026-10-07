# galpy jax/PyTorch backends: architecture notes for the methods section

Sources: galpy `feat/backends` at 9f6c9d6f7 (checkout `/var/tmp/fbp`; paths below
are relative to it), design notes in `/home/bovy/Repos/galpy-jax` (README.md,
master_plan.md, galpy_backend_autodiff_discussion.md, plan/track0_infra.md,
logs/vacation_2026-07/STATUS.md). Line numbers are from that checkout. Items I could
not confirm from source are marked **[unconfirmed]**.

## 0. The philosophy in one paragraph

`master_plan.md` ("Settled architecture (do not re-litigate)") fixes six rules:
(i) resolve the backend by *following the data* (explicit `xp=` > array arguments >
context/global default > numpy); (ii) inject dispatch at the *private* compute layer
(`_evaluate/_Rforce/...`) so the public API and unit decorators are untouched;
(iii) backend-specific reimplementations live in named `galpy.backend` modules, never
`if backend == ...` in physics code; (iv) prefer native backend functions, keep the
galpy fallbacks separable; (v) orbit autodiff through C = forward-sensitivity STM
(`M^T lambda`), not an adjoint; (vi) preserve device and dtype by anchoring new arrays
on inputs. The hard requirement ("Tolerance policy") is: **the numpy path is
byte-identical and jax/torch pass the same tests at the same tolerances.** The
success criterion is that the entire existing test suite passes under numpy, jax and
torch (master_plan.md "Goal & success criteria").

## 1. numpy byte-identity

**How it is achieved.** Every helper in the backend layer is a strict pass-through
when the namespace is numpy. The invariant is stated and enforced at the data-coercion
layer (`galpy/backend/_coerce.py:23-31`, "THE CORE INVARIANT ... returns its inputs
OBJECT-IDENTICALLY (no asarray, no copy, no dtype touch)"):

```python
# galpy/backend/_coerce.py:76-107
def coerce_coords(xp, *coords, device=None):
    ...
    if xp is numpy:
        return coords
```

The same `if xp is numpy` guard heads `promote_scalars` (:131), `as_backend_constant`
(:161) and `branch_where` (:267). The resolver returns the *plain* `numpy` module for
numpy data (`_namespaces.py:777-791`), so migrated code written as `xp.sqrt(...)` runs
the identical numpy ufunc. The scipy-like shims delegate to scipy on numpy data:
`galpy.backend.optimize.brentq` -> `scipy.optimize.brentq` (`optimize.py:58-114`),
`galpy.backend.quadrature.quad` -> `scipy.integrate.quad` (`quadrature.py:8-13`),
`galpy.backend.interpolate` keeps calling the fitted scipy spline (`interpolate.py:4-6`),
`galpy.backend.random` with `key=None` calls the global `numpy.random.*`
(`random.py:10-13`), and `galpy.backend.special` routes numpy to `scipy.special`
(`special/_router.py:65-83`). The boundary decorator is a no-op on numpy
(`_input.py:31-33`; the wrapper only coerces when `xp is not numpy`, `_input.py:178`).

**How it is enforced.** (a) `tests/test_backend_input.py:54`
`test_numpy_path_is_byte_identical_passthrough`; (b) per-PR byte-identity hashes
("hash over forces/potentials/densities/masses, single + list potentials, incl. the
`dR=1` control-kwarg path", `plan/track0_infra.md`, #1196) and `.view(int64)` /
`tobytes` comparisons against a pristine checkout (README rows #1090, #1260, #1261);
(c) `tests/test_backend_conventions.py`, an AST guard that "bans bare `numpy.` in
compute methods of migrated potentials" (`track0_infra.md` P1); (d) adversarial
verifier agents doing "numpy byte-parity vs a pristine checkout" per family
(`plan/trackA_potentials.md` 2026-06-09); (e) the standing rule "numpy must stay
byte-identical (flag any exception ...)" (`STATUS.md`, Standing constraints).

**The `_backend_ready` gate** (`galpy/backend/_input.py:59-82`). `@backend_input`
coerces only when `is_backend_compatible(target)`; the docstring calls this "a
TEMPORARY compatibility guard, not part of the decorator's design", kept because
`interpRZPotential` (scipy interpolation diverged 3.3%/32% at the grid edge) and
`MovingObjectPotential` (numpy trajectory lookup fails on a jit tracer) broke when
coerced. Removing the gate was tried and reverted (`track0_infra.md`, "Dead-end:
removing the migration gate": four breakages, including making 48 df decorations live
that then raised). Compatibility is a per-object flag `_backend_compatible` set in
`__init__` (default False, `galpy/potential/Potential.py:349-351`), read recursively
for lists/composites/wrappers by `galpy/backend/_compat.py:18-83`. **[unconfirmed]**
whether the gate still blocks `interpRZPotential`/`MovingObjectPotential` today:
`tests/test_backend_interprz.py` and `tests/test_backend_movingobject.py` now exercise
jax/torch paths for both, but neither file sets `_backend_compatible` (grep), so they
may rely on being handed backend arrays directly.

**Where the numpy result WAS allowed to change** (each flagged and approved):
- `constantbetadf` f(E): the force divisor moved from a per-potential `_rforce_jax` to
  the generic `evaluateRforces`; "numpy fE NOT byte-identical (flagged,
  user-approved): float-op reordering" (README row #1089).
- King model: "the King tolerance is one [exception], approved" (`STATUS.md`,
  Standing constraints). **[unconfirmed]** which test/tolerance.
- `streamdf._sample_aAt` (`galpy/df/streamdf.py:5080-5083`): "numpy is intentionally
  NOT byte-identical to the old ARS path (the draw is a different, exact sampler on a
  shifted RNG stream); the sampled DISTRIBUTION is preserved".
- `quasiisothermaldf.sampleV` now uses the inverse-CDF sampler on every path including
  numpy (`galpy/df/quasiisothermaldf.py:2254-2262`; `_sampleV_icdf` "replaces the old
  fmin_powell + numpy.random rejection sampler"), so numpy samples differ from v1.12.
- numpy *bug fixes* found by the migration and shipped (to main where possible): 1D
  `physical_conversion_actionAngle` IndexError (#1225); SCF planar second-derivative
  sign in C (#923); `TwoPowerSphericalPotential` via the incomplete beta (#1596);
  `eddingtondf` f(E) near Emin (#1586). Rule: numpy/C fixes go to main, backend-only
  deltas to `feat/backends`.

## 2. Interface preservation

Same classes, same methods, same `method=` strings; three new ingredients only:

```python
# galpy/backend/_resolver.py:64-90
@contextlib.contextmanager
def use(name, force=False):
    namespace_for_name(name)  # validate eagerly
    token = _BACKEND_CTX.set((name, force))
    try:
        if name == "torch":  # galpy's internals in float64 (see float64_default)
            with float64_default():
                yield namespace_for_name(name)
        else:
            yield namespace_for_name(name)
    finally:
        _BACKEND_CTX.reset(token)
```

- `galpy.backend.use(name, force=False)` / `set_default_backend` / config
  `[backend] default = numpy|jax|torch` (`doc/source/installation.rst:585`; seeded at
  import by `_resolver.py:102-115`). With `force=False` the default only applies when
  there is no array to dispatch on; **the primary mechanism is the data**: pass a jax
  array or torch tensor and galpy computes with that framework, with no context manager
  at all (`_resolver.py:1-15`).
- Arrays in -> same-framework arrays out. `exit_cast` (`_namespaces.py:380-388`)
  returns numpy for scalar/numpy inputs even under a forced backend ("astropy can't
  hold a backend-array Quantity, so consumption outputs stay numpy") and leaves backend
  arrays alone when any input is one. `match_input_dtype` (:434-489) gives
  float32-in/float32-out at float64 quality for the table-backed potentials (#949).
  dtype policy (#1613, `track0_infra.md` 2026-10-06): Python floats, numpy inputs and
  everything galpy creates are float64 (`float64_default`, :637-671, entered by
  `use("torch")`, by the `@backend_input` wrapper and by
  `float64_default_if_torch_args` on df/actionAngle/Orbit entry points); a user's
  float32 tensor stays float32 in and out.
- Orbits: an `Orbit` built from a jax/torch initial condition keeps it in
  `self._ic_backend` (`galpy/orbit/Orbits.py:512-543`); `integrate` then routes by
  namespace (section MODULES.md/Orbits).
- Units. Potential parameters given as backend arrays "are accepted and passed through
  unscaled -- [they carry] no units, so [they are] treated as already in galpy's
  internal units" (`galpy/util/conversion.py:542-553`), which is what makes d/dtheta
  possible. Quantity *inputs* are parsed to floats by the legacy unit decorator which
  sits *outside* the backend boundary (`_input.py:20-25`: `@potential_physical_input`
  outer, `@backend_input` inside, `@physical_conversion` innermost); a Quantity that
  reaches the boundary is passed through untouched (`_input.py:92-99`). Quantity
  *outputs* are built as `units.Quantity(to_host(out * fac), unit=u)`
  (`conversion.py:1038-1041`). Quantity is **not traceable**: the remaining xfail
  ledger consists solely of three `--jit` pickling-of-unit-wrapper tests
  (`tests/backend_xfail.txt:262-264`); the 104 jax-jit `test_quantity` failures seen
  earlier were a harness artifact (configure-time galpy import locking in
  `astropy-units=False`, `tests/conftest.py:440-450`). **[unconfirmed]** behaviour of a
  Quantity-returning call whose *input* is a backend array (whether `to_host` + astropy
  yields a Quantity wrapping a tensor or converts).

## 3. Backend-agnosticism: namespace dispatch, the coercion boundary, devices

**Resolver** (`galpy/backend/_resolver.py:28-55`):

```python
def get_namespace(*arrays, xp=None):
    if xp is not None:
        return namespace_for_name(xp)
    ctx = _BACKEND_CTX.get()
    if ctx is not None and ctx[1]:  # forced default beats the data
        return namespace_for_name(ctx[0])
    ns = namespace_from_arrays(arrays)
    if ns is not None:
        return ns
    if ctx is not None:
        return namespace_for_name(ctx[0])
    return numpy
```

The engine is `array_api_compat.array_namespace` (`_namespaces.py:834-866`): numpy ->
the plain `numpy` module; jax -> `jax.numpy`; torch -> `array_api_compat.torch` (a fast
path returns it directly because `array_namespace` itself is not `torch.compile`-able).
Every migrated compute method begins `xp = get_namespace(R, z, phi, t)` and uses
`xp.*`; `numpy.pi` -> `math.pi` (`plan/trackA_potentials.md`). Variants:
`resolve_namespace` (backend args else forced/context default, `_namespaces.py:390`) and
`prefer_backend_namespace` (backend args else probe all; for legitimately mixed
backend+numpy inputs, :399-416). Dispatch cost is ~3 us/call eager and is
constant-folded away under jit (`plan/trackD_autodiff.md`, "Performance note";
`logs/gpu_speed_galpy_vs_galax/log.md` S2).

**The coercion boundary `@backend_input(*coords)`** (`galpy/backend/_input.py:106-223`,
#1196, 73 potential-side entry points). Declared coordinate names are checked against
the signature at decoration time; only those arguments are coerced onto the active
namespace, "whether passed positionally or by keyword; every other parameter
(derivative orders, flags, masses, grids, option strings) is left untouched":

```python
# galpy/backend/_input.py:172-212 (abridged)
xp = prefer_backend_namespace(*probe)
if xp is not numpy and _backend_ready(args[0]):
    with (float64_default() if name_of_namespace(xp) == "torch" else _NULLCTX):
        dev = device_of(*probe)             # ONE device anchor for the whole call
        ...newargs[ii] = _coerce_one(xp, newargs[ii], dev)...
        out = traced_call(method, args, kwargs, slots, nargs, xp)
        if out is not NOT_TRACED:
            return out
        return method(*args, **kwargs)
return method(*args, **kwargs)
```

Sequence coordinates (`v=[vR,vT,vz]` of the dissipative forces) are coerced
element-wise to keep each element's autograd graph (`_input.py:84-103`). The same
declaration doubles as the traced/static split for jit mode (section 5). The
actionAngle public API is varargs (`actionsFreqs(*args)`), so it is *not* decorated; it
dispatches with `get_namespace` + `promote_scalars` inside (`track0_infra.md`, "Known
limitation").

**`@backendNative`** (`galpy/util/coords.py:248-262`) is a marker on coordinate
transforms that have been migrated: `scalarDecorator` promotes scalars through numpy by
default (which would strip the framework off a backend array) and only for marked
functions resolves `xp = get_namespace(*args)` and coerces with `coerce_coords`
(`coords.py:265-283`). "Migrate a function, then mark it -- no flag day."

**Forced vs data namespace.** `use(name, force=True)` makes the test harness pin a
whole run to one backend even for scalar/numpy inputs; production code never forces.
Design note that generalises (`track0_infra.md`): forced runs short-circuit the
resolver, so only the *unforced* `tests/test_backend*.py` shard exercises the mixed
numpy/backend probe.

**Devices.** `device_of(*coords)` returns the device of the first backend array
(`_namespaces.py:492-516`); stored numpy constants (rotation matrices, expansion
coefficients, quadrature nodes, spline coefficients) are brought in with
`as_backend_constant(xp, value, ref)` / `asarray_on_device` anchored on it
(`_coerce.py:156-171`, `_namespaces.py:741`). The boundary computes one anchor per call
so a numpy sibling of a CUDA coordinate cannot land on the CPU (`_input.py:183-190`).
Asking an explicit device is not free (397.6 vs 134.6 us/call on jax CPU), so
`default_device` lets callers omit it when it cannot change placement
(`_namespaces.py:518-536`, #1300). Host transfers at genuine numpy boundaries use
`to_host` (keeps autograd; `_namespaces.py:359-371`) or `as_numpy` (detaches, :350).

**Context variables under `torch.compile`.** The backend and jit-mode selections are
`TracedContextVar`s (`galpy/backend/_tracectx.py:27-78`): a `ContextVar` whose `get()`
reads a thread-local mirror while dynamo is tracing, because "`ContextVar.get()` is
opaque to dynamo: every read severs the graph".

```python
# galpy/backend/_tracectx.py:56-59
def get(self):
    if is_compiling():
        return getattr(self._mirror, "value", self._default)
    return self._var.get()
```

## 4. "Basic jax/torch only; own implementations" (special functions, root finding, quadrature, splines, sampling)

Rationale (`galpy_backend_autodiff_discussion.md` S1, "Tier 2 -- scipy"): scipy does
not dispatch, `jax.scipy`/`torch.special` have per-function holes, so galpy ships a
"scipy-like" layer in `galpy.backend` that is native-preferring with separable
fallbacks.

**Special functions** (`galpy/backend/special/_router.py`): dispatch by namespace to
`scipy.special` / `jax.scipy.special` / `torch.special` unless the function is in the
per-backend fallback set:

```python
# galpy/backend/special/_router.py:15-57 (abridged)
_NATIVE_MISSING = {
    "jax":   frozenset(("ellipk", "ellipkm1", "ellipe", "k0", "k1", "kn", "iv")),
    "torch": frozenset(("gamma", "ellipk", "ellipkm1", "ellipe", "hyp2f1", "hyp1f1",
                        "k0", "k1", "kn", "iv", "sici", "exp1")),
}
_NATIVE_UNRELIABLE = {
    "jax":   frozenset(("hyp2f1", "hyp1f1", "gammainc", "gammaincc")),
    "torch": frozenset(("gammainc", "gammaincc")),
}
```

Why: `jax.scipy.special.hyp2f1`/`hyp1f1` "are catastrophically wrong for z < -1
(return +-inf with NaN gradients), which is exactly galpy's regime (z = -r/a, ...)"
(`_router.py:40-44`; tripwire `tests/test_backend_special.py:519`
`test_jax_native_hyp2f1_hyp1f1_are_unreliable`); `jax` `gammainc` iterates per element
in a `while_loop` (~100x slower than the vectorised fallback, #1615); `torch`
`gammainc` has no derivative w.r.t. the order and an algorithm switch at a~20 costing
~6 digits (`_router.py:47-56`; `_fallback/gammainc.py:1-48`). A capability test asserts
the table matches the installed backends so entries are removed as natives improve.
Fallbacks (`special/_fallback/`): `hyp2f1` (Euler integral + tanh-sinh, ~1e-15 vs
mpmath to |z|=1e12; series route for the Euler transformation, #1226), `hyp1f1`
(closed form for galpy's only case b=a+1), `gammainc` (ascending series N=120 /
modified-Lentz N=50 with a Stirling-form prefix; worst 7.3e-15 vs mpmath, "at least as
accurate as the numpy path everywhere"), `elliptic` (AGM, `_AGM_NITER = 16`, :12),
`bessel_i/k`, `exp1`, `sici`, `gamma`, `xlogy`, `assoc_legendre` (Bonnet recursion),
`gegenbauer`. Fixed iteration counts everywhere (jit/vmap-compatible).

**Root finding** (`galpy/backend/optimize.py`): `brentq` is a drop-in for scipy's;
on backend data it runs a fixed schedule of sign-preserving bisection halvings
(`n_bisect_steps`, :116-143, `ceil(log2(width/xtol))+1`, full `maxiter` when the bracket
is traced) followed by one Newton step that both polishes the value and carries the
implicit-function-theorem gradient (first order only; header :1-55):

```python
# galpy/backend/optimize.py:145-156, 249-263
def bisect_step(lo, hi, slo, f, xp):
    mid = 0.5 * (lo + hi)
    same = xp.sign(f(mid)) == slo
    return xp.where(same, mid, lo), xp.where(same, hi, mid)

def newton_polish(x0, fx0, dfdx, xp):
    step = fx0 / nonzero_slope(dfdx, xp)
    return x0 - xp.where(xp.isfinite(step), step, xp.zeros_like(step))
```

Under a jax trace the loop is rolled into `lax.fori_loop` so the n copies of f do not
unroll into the jaxpr (:159-176; #1040). One shared helper is reused by every AA/DF
root-find (`plan/trackE_actionangle.md` task #86) instead of per-class solvers.

**Quadrature** (`galpy/backend/quadrature.py`): fixed-order Gauss-Legendre
(`_QUAD_N = 100`, :56; `fixed_quad` default n=50), with semi-infinite,
boundary-layer, symmetric, finite-part, nested and Simpson variants; float64 node tables
anchored on the input device; "a differentiable APPROXIMATION, not adaptive" (:36-38).
AA uses `_BACKEND_GL_ORDER = 50` (`galpy/actionAngle/actionAngle.py:27`); Staeckel uses
the composite chi-anomaly rule with `leggauss(10)` panels
(`actionAngleStaeckel.py:1994`, 335), matching C to machine precision (gh#1357).

**Interpolation** (`galpy/backend/interpolate.py`, #965): mode 1 converts a fitted
scipy spline to a power-basis PPoly table evaluated by namespace ops (differentiable in
the evaluation point; `interpRZPotential`, `interpSphericalPotential`); mode 2 builds
cubic coefficients in-backend by parallel cyclic reduction (differentiable in the
y-values; parameter-dependent tables such as dynamical-friction sigma_r). Also
`map_coordinates` with an in-backend B-spline prefilter (orders 3 and 5, mirror
boundary; #1630), `Spline1D/Spline2D`, smoothing splines with GCV.

**Sampling and RNG** (`galpy/backend/sampling.py`, `random.py`): piecewise-linear
inverse-CDF sampling as a reparameterised transform of uniforms (static shapes, no
rejection loop), with explicit keys (`jax.random` keys; a seed-based `_TorchKey`; a
`_NumpyKey` opt-in; `key=None` = historical global numpy RNG).

## 5. No internal compilation

`galpy/backend/_jit.py:1-22`: "galpy never jits itself (see the no-internal-jit rule):
the library is written to be jit-COMPATIBLE and the user decides when to trace."
Repeated in `interpolate.py:31-34`, `quadrature.py:34-36`, `jacobian.py:11-13`. Users
wrap their own code in `jax.jit`/`torch.compile`; the optional global trace mode
`galpy.backend.set_jit("jax"|"torch")` / `jit(mode)` (`_jit.py:43-90`, #1200-#1205)
exists so the *test suite* can run everything traced. It traces at the
`@backend_input` boundary: declared coordinates are the traced arguments, everything
else is one static bundle (`traced_call`, :254-297), only the outermost boundary traces
(`_TRACING`, :38), and a boundary on a different framework than the mode runs eager.

What had to change to be traceable (all measured, see `track0_infra.md`,
`plan/differentiability_audit_2026-09-16.md`):
- Fixed iteration counts instead of data-dependent loops (`n_bisect_steps`, AGM,
  series/CF lengths above; streamdf's ~60-iteration track refinement with a fixed
  count, `plan/trackF_df.md`).
- `xp.where` instead of Python branching, with *dead-branch guards*: both sides are
  evaluated eagerly, so the unused side is clamped to a safe argument so its NaN cannot
  poison reverse-mode AD (`_coerce.py:181-236` `radial_limits`, `:238-262`
  `branch_where`; e.g. `galpy/potential/DehnenBarPotential.py:171`). Validation
  branches use `concretely_true` / `has_concrete_truth_value`
  (`_namespaces.py:27-61`), which skip themselves under a trace.
- Data-dependent Python control flow on *parameters* (`if self._a == 0.0`) is fine under
  `grad` but raises inside a jitted ODE solve (audit S D).
- Static shapes: `set_at` is eager-only (:150-167); shrinking-array loops pad to
  power-of-4 buckets, because eager jax recompiles per shape
  (`bucket_size`, :170-178, #1621: qdf `sampleV_interpolate` 2837 s -> 118 s):

```python
# galpy/backend/_namespaces.py:170-178
def bucket_size(n, minimum=16):
    """Smallest ``4**k >= n``, at least ``minimum`` (a power of 4); 0 stays 0."""
    if n <= 0:
        return 0
    return max(minimum, 1 << 2 * ((n - 1).bit_length() + 1 >> 1))
```

- `under_trace` (:125-147) distinguishes "no concrete value" from eager, because dynamo
  turns `float(x)` into a symbolic scalar and a `float()` probe silently took the scipy
  branch inside a compiled region (#1218, wrong `surfdens`).
- `untraceable_setup` (:255-289) wraps lazy numpy/scipy table builders in
  `torch.compiler.disable` so a first call inside a compiled region does not trace scipy.
- Static-key structural hashing for jitted galpy objects (`_jit.py:124-175`, #1625):
  a potential passed as a static argument is keyed by type + attributes (scalars by
  value, galpy sub-objects recursively, everything else by held identity), so
  `pot.normalize(0.5)` retraces and fresh `toPlanar()` wrappers do *not*:

```python
# galpy/backend/_jit.py:124-154 (abridged)
def _object_key(obj, depth=0):
    for name, attr in vars(obj).items():
        as_float = _scalar_or_none(attr)
        if as_float is not None:               entries.append((name, as_float))
        elif isinstance(attr, (str, type(None))): entries.append((name, attr))
        elif depth < _KEY_DEPTH and type(attr).__module__.startswith("galpy.") ...:
            entries.append((name, _object_key(attr, depth + 1)))
        else:                                   entries.append((name, _Ident(attr)))
    return (type(obj), tuple(sorted(entries)))
```

- torch graph breaks from `ContextVar` reads: the `TracedContextVar` above; and
  `compilable_singledispatchmethod` (:213-253).
- The C bridges (`pure_callback`) skip the callback entirely for concrete inputs,
  because a fresh closure per call compiled anew (~25 ms each; `_jax/staeckel_c.py:16-25`).

## 6. The test suite in five modes

The *existing* suite is run verbatim under numpy, jax eager, jax `--jit`, torch eager
and torch `--jit`, plus `--device cuda` locally. Mechanism (`tests/conftest.py`):
options `--backend numpy|jax|torch`, `--device cpu|cuda`, `--jit` (:369-397); an
autouse fixture wraps every test:

```python
# tests/conftest.py:699-720 (abridged)
@pytest.fixture(autouse=True)
def _galpy_force_backend(request):
    backend_name = request.config.getoption("--backend")
    if backend_name == "numpy" or request.node.get_closest_marker("backend_managed"):
        yield; return
    if backend_name == "jax":   jax.config.update("jax_enable_x64", True)
    elif backend_name == "torch": torch.set_default_dtype(torch.float64)
    if request.config.getoption("--jit"):
        with backend.use(backend_name, force=True), backend.jit(backend_name): yield
    else:
        with backend.use(backend_name, force=True): yield
```

Backend-specific test modules (88 `tests/test_backend_*.py`) mark themselves
`backend_managed` and parametrise over installed backends. The **xfail ledger**
(`tests/backend_xfail.txt`, format `<backend> <nodeid>`, matched at full/param-stripped/
file granularity, `conftest.py:500-577`) marks known failures `xfail(strict=False)` so
the all-backend CI stays green while the port proceeds; a scheduled regen run
(`GALPY_BACKEND_XFAIL_REGEN=1`) rewrites it from real outcomes, and the diff is the
burndown (`tests/backend_status_report.py`). Two companion lists: `backend_slow_skip.txt`
(unrunnable-until-vectorised, a burndown) and `backend_skip.txt` (permanent: out-of-scope
`actionAngleTorus`/`useTM` and four eager-jax per-op floors). "-jit" entries inherit
eager xfails but not eager slow-skips (:151-192).

Burndown (README g13, `tests/backend_xfail.txt` history): jax 355 / torch 1670
(2026-06-14) -> jax 109 / torch 738 (#1077, 2026-07-07) -> 156 -> 42 -> 28 -> 20
(2026-08-05) -> **3 xfail, 3 slow-skip, 16 permanent skips** (2026-10-06, README
Status). The 3 xfails are all `--jit` pickling of unit wrappers.

**numpy islands.** A dual-path `get_namespace(R) if is_backend_array(R) else numpy`
guard runs a forced-backend test on numpy and produces a "hollow green". They were
eliminated systematically: AA redesign dropping `@numpy_island` (#1038); jeans (#1260,
which exposed 5 real backend defects the island had hidden); diskdf/surfaceSigma
(#1261); streamgapdf orbit integration (#1624). CUDA runs are the island detector (every
hidden numpy conversion fails loudly).

**The `--jit` boundary tracer** (`conftest.py:402-492`): a `--jit` run can pass while
every entry point falls back to eager, so `traced_call` is monkey-patched to count
traced vs `NOT_TRACED` returns; the share is printed (`JITCOV: boundary calls=...
traced=... NOT_TRACED=...`), written into the junit XML, and the shard fails below a
25% floor (measured 99.0% on `test_sphericaldf.py --backend jax --jit`). In CI the torch
trace mode runs dynamo with `GALPY_JIT_TORCH_BACKEND=eager` (tracing is galpy's
contract; inductor codegen is torch's, `conftest.py:26-35`; one dedicated test pins
inductor).

CI (`.github/workflows/backend-tests.yml`): matrix BACKEND {jax, torch} x MODE {eager,
jit} x 22 TEST_FILES shards (mirroring the numpy build matrix; `test_potential` and
`test_orbit(s)` split 5 ways), per-test `--timeout=300`, 180-min shard cap, status table
rendered per PR. Counts: `test_potential` 1320 tests x 3 backends 0 fail and the
unforced `test_backend*` shard 6251 tests (#1196, 2026-07-25); `test_orbit` 581 tests
under torch `--jit` 0 fail and jax `--jit` 581/3 (README "traced-surface measurement").
**[unconfirmed]** the total collected count of the full suite per mode; I did not run
collection.

## 7. GPU specifics

- `--device cuda` (`conftest.py:308-328`): torch -> `torch.set_default_device("cuda")`
  (every `asarray` without an explicit device lands on the GPU); jax -> require
  `jax.default_backend() == "gpu"`. Fails loudly without a GPU.
- Library side: device anchoring (section 3) everywhere a constant table meets
  coordinates (SCF, coords, DoubleExp #967/#968; +10 Staeckel/Vertical/Spherical sites
  #1016; #1603/#1606); quadrature takes a `device=` hint when limits are Python scalars
  but the integrand closes over CUDA tensors (`quadrature.py:20-27`); `to_host` at
  genuine numpy boundaries (VerticalInverse energy grid, #1631; "all 49 main-suite VI
  tests pass under torch and jax" on the TITAN Xp, `plan/trackE_actionangle.md`).
- Orbits: the in-backend ODE path runs on the GPU with a native batch axis (N orbits in
  one adaptive solve, #1027); streamspraydf `sample()`/`streamTrack()` run under
  `jax.jit` on CPU+GPU (#1106). The C-STM path is host-side by construction.
- Measured (`plan/trackG_figures.md` g8/g12, 2026-06-13, TITAN Xp sm_61): galpy-C
  ~2.1 ms/orbit; galpy-jax-GPU 1.65 s for one orbit (launch floor) -> 0.29 ms/orbit at
  N=1e4. The galax comparison (`logs/gpu_speed_galpy_vs_galax/log.md`): the ~4.5x gap
  decomposed into a tolerance mismatch (1e-10 vs 1e-8, ~1.6x), the `gammainc` bulge
  (37% of MWPotential2014 cost), and batched vs per-orbit stepping; backend dispatch is
  zero cost under jit.
- Environment used (memory notes, not in the repo): jax 0.10.1, torch 2.12.0+cu126.
  GPU-only tolerance loosening is acceptable; pure-Python integrators on CUDA are not
  supported, users are pointed to torchode/torchdiffeq **[unconfirmed in code]**.
