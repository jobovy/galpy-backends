# Differentiability audit — end-to-end follow-ups (2026-09-19/20)

Follow-on to `differentiability_audit_2026-09-16.md`, widened from "does a
gradient exist" to "is the gradient CORRECT end-to-end w.r.t. the inputs users
actually differentiate": orbit methods and action-angle w.r.t. potential
parameters, DFs w.r.t. both their own and potential parameters.

Scope note agreed with Jo: `c=True` cannot carry potential-parameter
derivatives (only the Python path can), so everything below is evaluated
within that constraint.

## Sweep result

* **Potentials** — 267 cells, 216 ok, **zero** real failures for `d/d(amp)`.
  Cleared as non-issues: every `beta` MISM (beta is constant by construction,
  so the FD is noise around a constant, ~8.5e-12) and `DehnenBarPotential.dens`
  (values ~1e-19, i.e. comparing noise to noise).
* **DFs** — 109 cells, only two genuine defects (T1, T2 below).

## The defects, and what made them hard to see

### T1 — `osipkovmerrittNFWdf.dMdE` (PR: om-dmde-panels)

`d/d(NFW a)` returned **+19.6** against a truth of **-0.721**: wrong sign, 27x
too large. The *value* was still right to four digits, because the quadrature
error was **oscillating with the parameter**. No value-only test can catch
this class, which is why it survived.

Two independent causes, both required:

1. With `Q = -E - L^2/(2 ra^2)`, fQ's support `Q > Qmin` caps `L` at a constant
   `Lsupp`. NFW's fQ is truncated at `rmax` and **jumps** to zero there
   (0 -> 5.19e-3) rather than tapering, so integrating out to the `Q=0`
   boundary put a jump INSIDE the range -> O(1/n). Hernquist/PowerLaw taper as
   `sqrt(Q)` with support `Q>0`, so their ceiling is unchanged.
2. The ceiling `min(Lmax(r), Lsupp)` corners the outer integrand at the two
   radii where `Lmax(r) = Lsupp`, which sat inside a single GL panel.

A third, free improvement: after splitting, ALL residual error sat in the
middle panel, because in `u = L^2` the inner integral is
`int_0^Lsupp^2 g(u)/(2 sqrt(Lmax(r)^2 - u)) du` — its endpoint singularity
collides with the support edge exactly AT the panel edges, so the outer
integrand has a sqrt branch point at both ends of every panel. A cosine map
clusters nodes there and buys ~8 digits at the same node count.

Result (amp=2, a=1.3, ra=1.4, E=-0.6): 4.166919464 (rel 3.9e-05, d/da +19.57)
-> 4.167082856 (rel 2.0e-08, d/da -0.721045). galpy's own numpy nested adaptive
quad gives 4.167082429, 1.0e-07 off — the backend is now the tighter of the two.

### T2 — dynamical friction velocity triple (PR #1530)

`_calc_force_backend` takes `v` as a caller-supplied TRIPLE, which
`@backend_input` does not coerce the way it does `R`/`phi`/`z`. A numpy `v`
reaching a torch namespace makes the first `xp.sqrt` raise, so `d/d(amp)`
through friction never starts. One `coerce_coords` per kernel.

### T4g — the grid actionAngle classes (branch `backend/aagrid-pot-grad`)

`actionAngleStaeckelGrid` / `actionAngleAdiabaticGrid` raised
`TracerArrayConversionError` in `__init__` for ANY potential carrying a
gradient: the node build materialises in numpy (`rl` root-finds for Staeckel,
`Rs`/`EzZmax` for adiabatic). So `d(action)/d(potential parameter)` did not
exist at all through either class.

Fix: detect with `_pot_grad_namespace(self._pot)`, warn (`galpyWarning`), skip
the build, and delegate to the exact `actionAngleStaeckel` /
`actionAngleAdiabatic` the grid interpolates. Not `stop_gradient` on the nodes:
that freezes the table in the differentiated parameter, so every action would
return a **silently zero** gradient — strictly worse than the error it replaces.

Verified AD-vs-FD against the exact class: Staeckel 1.22e-09, adiabatic
1.31e-09; `test_actionAngle.py` unchanged at 234 passed.

### T4i — evolveddiskdf moments: the gate asks about R, not about the gradient

`vmomentsurfacemass` picks its namespace with

    xp = get_namespace(R) if is_backend_array(R) else numpy
    _direct_backend = is_backend_array(R)

so `d(meanvR|meanvT|sigmaR2)/d(potential parameter)` with a plain-float `R`
computes entirely in numpy and comes back a bare `float64` — the gradient is
SILENTLY lost (probed on torch eager autograd; all three moments).

The intent behind the guard is sound and documented ("TRANSITIONAL ... a forced
backend does NOT flip a numpy R", task #117): it stops the fast numpy suite
being routed onto the unmigrated direct/hierarchical paths. But the predicate
conflates *merely forced* with *carries a real gradient* — the same conflation
`_pot_grad_namespace` was added for. Widening it is feasible, and now MEASURED (torch eager, d/d(potential a) with
R passed as a backend array so the gate opens):

| path | carries the gradient? |
|------|----------------------|
| direct (`grid=False`) | yes, +0.0712 |
| grid, non-hierarchical | yes, +0.1087 |
| grid, **hierarchical** | NO -- returns a bare `float64` |

So two of the three routes already work once taken; the hierarchical grid is the
genuinely unmigrated one the code comment warns about. A correct fix therefore
widens the gate for the direct and regular-grid routes and must NOT silently
widen it for the hierarchical one -- there it should keep numpy (or say so)
rather than return a detached float.

Note the direct and grid numbers differ (0.071 vs 0.109) at `gridpoints=8`:
that is the coarse grid's own quadrature error, the same different-algorithm
caveat as everywhere else in this document, not a second defect. Any test must
compare each route against ITSELF under FD.

Also required: `xp` must come from the potential when R is a plain float --
`get_namespace(R, sigmaR1, ...)` resolves numpy for an all-float argument list,
so widening `_direct_backend` alone would run the "backend" path on numpy. Same
two-part shape as the `rl`/`rE` fix in #1532.

Deliberately NOT folded into #1539.

### T4i-b — evolveddiskdf direct path: torch crashes on ZERO initdf moments

Separate from the gate, and NOT about gradients at all. `vmomentsurfacemass`
guards a vanishing normalisation with

    initvmoment = xp.where(initvmoment == 0.0, 1.0, initvmoment)   # line 620

`initvmoment` is a numpy scalar, so `initvmoment == 0.0` is a numpy bool and the
whole call is `torch.where(numpy.bool_, float, numpy.float64)` -> TypeError.
jax accepts Python/numpy scalars there, so it is TORCH-ONLY.

It fires only when the initial DF's moment is EXACTLY zero, i.e. the odd moments
(`meanvR`, `sigmaRT`, `vertexdev`) — by symmetry `<vR>_initdf == 0`. The
existing backend tests all call `vmomentsurfacemass(R, 0, 0, ...)`, whose
`initvmoment` is never zero, which is why the guard's own branch was never
executed on torch.

| backend | grid=False (direct) | grid=True |
|---------|---------------------|-----------|
| jax     | ok                  | ok        |
| torch   | TypeError line 620  | ok        |

Fix: build the guard from backend scalars (`xp.asarray(initvmoment)`) before the
comparison, and add an odd-moment case to the direct-path tests. NOTE the
comment on that line already says it is "kept OFF the numpy path deliberately"
to preserve the float return type — so the fix must not touch numpy.

### T4j — DF sample() crashes under a gradient-carrying potential

`isotropicHernquistdf.sample()` and `quasiisothermaldf.sampleV()` raise
`RuntimeError: Can't call numpy() on Tensor that requires grad` when the
potential carries a gradient. A crash, not a silent loss, so it cannot ship a
wrong number — but it does mean sampling is unreachable for a differentiable
potential even when the caller only wants the SAMPLES (no gradient through the
draws). Differentiable sampling itself is a roadmap item (reparameterised
inverse-CDF), so the near-term fix is the narrower one: detach at the sampler
boundary so sampling works, and let the gradient-through-samples question stay
with the roadmap.

## Method-error log (things that cost time, worth not repeating)

* **I almost ledgered 104 tests as a galpy limitation when the bug was in my own
  harness.** The jax-jit test_quantity shard went red; the shape matched a
  remembered "Quantity is not traceable" boundary, so my first move was to
  reach for the ledger. Two things stopped it. (1) My own note on that boundary
  says, in capitals, *do not write a ledger policy for these* -- it records that
  I had ALREADY once ledgered 33 of them on the strength of an error message and
  had to retract it. (2) Today's symptom did not actually match the remembered
  one: not TracerArrayConversionError inside the trace, but units silently NOT
  ATTACHED, with the correct physical VALUE. That mismatch is the whole signal.
  Root cause: this PR's own traced-share counter imports galpy inside
  `pytest_configure`, which runs BEFORE test modules are imported, locking in
  the default `astropy-units=False`; `conversion.py` binds `_APY_UNITS` as an
  import-time copy, so `test_quantity.py`'s module-level config set never
  reaches it. Proven by A/B on import order alone, no pytest involved. Fix: move
  the install to `pytest_collection_finish`. 104 failed -> 344 passed, and the
  counter still reports (92.7% traced), which is the check that matters -- a
  "fix" that silenced the measurement would have looked identical on the
  pass line.
  Three transferable rules: a remembered diagnosis is a hypothesis, not a
  verdict; "my new CI dimension went red" should suspect the NEW DIMENSION
  before the code it is measuring; and when you disable a measurement to fix
  its output, assert the measurement still fires.
* **Scattered vs cascading failures is a one-line discriminator.** 104 failures
  could have been one leaked global poisoning everything after it. Printing the
  pass/fail ribbon in file order settled it in seconds: 215 passes AFTER the
  first failure, so not a cascade, so not a leaked flag -- which is what sent me
  to the import-order question instead of hunting a missing try/finally.

* **Gating an entry point means gating EVERY entry point, not the ones that
  share a body.** Staeckel's `Jz`/`JR` are one-liners over `__call__`, so the
  `_evaluate` gate covered them — and I generalised that to adiabatic, where
  `Jz` has its own body reading `_jzInterp`/`_EzZmaxsInterp` directly. It
  needed its own gate. Enumerating which methods READ the attributes that stop
  existing is a 20-line script; assuming the shape from a sibling class is how
  it gets missed. Same shape as the `interpSnapshotRZPotential` writer miss.
* **A fallback changes WHICH ALGORITHM runs, so the naive A/B measures two
  different functions.** After the bypass the traced arm runs the exact method
  while a numpy FD arm still builds the grid; that reads as a 20% (Staeckel) /
  0.45% (adiabatic) "mismatch" which is pure interpolation error. Caught the
  same trap three times this week (qdf `_rg` table-vs-root-find, then both grid
  classes). Whenever a fix INTRODUCES a path switch, the reference arm must be
  forced onto the same path before any number it prints means anything.
* **`actionAngleAdiabatic` has no `Jz`.** Delegating `Jz` to `self._aA.Jz`
  would have been an `AttributeError` reachable only under a trace — i.e. only
  in the case the gate exists for. Check the delegate's API, don't mirror the
  caller's.

* **`import galpy.df.osipkovmerrittdf as om` binds the CLASS, not the module.**
  A GL-order sweep monkeypatching `om._QUAD_N_VMOM2D` therefore set a class
  attribute and never reached the quadrature — which produced a confident and
  WRONG "converged from n=60 to n=320". Use `importlib.import_module`.
* **The reference can be the looser path.** scipy's nested `integrate.quad` runs
  at DEFAULT tolerances on both levels here; its own error (~1e-07) exceeded the
  tolerance the backend was being held to. Build an independent arbiter before
  declaring which side is wrong.
* **Localise before tuning.** Raising `n` globally cost 2-4x for one digit;
  splitting the error per panel and per dimension showed the inner dimension was
  already exact and only one panel was bad, which pointed at the real cure
  (a change of variable) at zero cost.

## Not defects

* `RazorThinExponentialDiskPotential._R2deriv` — IS implemented in-plane;
  off-plane has no closed form upstream. A note, not a gap.

## Still open

* **T4** sweep COMPLETE. Open fixes it produced: T4i + T4i-b `evolveddiskdf`, T4j DF
  `sample()`. `streamspraydf` needs nothing — it already probes for a backend
  theta/mass/stripping_pdf and discriminates a merely-forced context. (Orbit
  accessors done #1532; grid variants done `backend/aagrid-pot-grad`; qdf
  `pvR`/`pvT`/`meanvR` + `_rg` done #1537.)
* **T6** `backend/spline1d-vector-valued` + five consumer migrations.
* **T5** add jit to CI (needs Jo — I cannot edit `.github/workflows/**`).
