# galpy: Multi-Backend / Autodiff Design Discussion

A working discussion on adding JAX/PyTorch backends to galpy (currently numpy/scipy + C),
covering the overall backend architecture, differentiable orbit integration via the C code,
variational equations, validation strategy, higher-order derivatives, and analytic
derivatives of the action-angle machinery.

---

## 1. High-level approach to multi-backend support

**Q:** I have galpy (Python + C for speed-critical paths), currently on numpy/scipy. I want to
support JAX and PyTorch backends for autodiff, deep-learning models of galactic dynamics, etc.
I want to reuse existing code as much as possible (numpy is imported as numpy and used as
numpy, so a JAX backend could often be `from jax import numpy`), stay backend-agnostic in most
places, keep backend-specific code in a few central locations, and support a global backend,
per-function overrides (necessary?), and a context manager. How should I do this at a high level?

**A:**

### The key reframe: dispatch on the data first, fall back to global/context second

Global backend + per-function override + context manager is the right set of *controls*, but the
*primary* resolution mechanism should be "follow the data," with global/context state as a
fallback. The reason is autodiff: the point of a JAX/PyTorch backend is differentiating *through*
a galpy calculation sitting inside a larger model. There, the arrays flowing in are tracers/tensors,
and the backend is an intrinsic property of those objects, not a global mood. If the global backend
is numpy but a JAX array is passed, you *must* use JAX ops or you break tracing. Data wins over the flag.

Resolver order:

1. If the caller passed an explicit `xp=`/`backend=` kwarg, use it (the per-function override —
   mainly an escape hatch).
2. Inspect array arguments; if any are JAX/torch arrays, use that namespace (handles autodiff for free).
3. Otherwise consult the context-manager/global state.
4. Otherwise default to numpy.

Steps 1–2 are what `array-api-compat`'s `array_namespace()` gives you — the direction scipy and
scikit-learn have taken. For step 3 use a `contextvars.ContextVar`, not a plain module global, so
the context manager is thread- and async-safe (matters inside training loops or Dask graphs).
All three controls fall out of one resolver:

```python
def get_namespace(*arrays, xp=None):
    if xp is not None:
        return xp
    if any(_is_tracked_array(a) for a in arrays):
        return array_namespace(*arrays)
    return _backend_ctx.get()  # ContextVar, default numpy
```

Every compute function starts with `xp = get_namespace(R, z, ...)` instead of `import numpy as np`.
That's your single central location.

### The namespace-swap instinct is right but only carries the first tier

Sort the codebase into tiers:

- **Tier 1 — pure elementwise/reduction math.** Most analytic potential/force/density evaluations.
  Namespace swap mostly works, but three idioms still bite (worth a lint rule each): in-place mutation
  (`a[i] = x` is illegal in JAX — needs `.at[i].set()`), data-dependent Python control flow
  (`if R > 0:` dies on a tracer — must become `xp.where`), and RNG/`astype`/some reductions that
  differ across backends. Even Tier 1 is "touch the un-functional bits," not zero-touch.

- **Tier 2 — scipy.** Bigger gap; its own sub-project. `scipy.special/.integrate/.interpolate/.optimize`
  don't dispatch. `jax.scipy` and torch cover some, but the special functions galactic dynamics needs
  (Bessel, elliptic integrals, gamma-family) will have per-backend holes. Build a parallel "scipy-like"
  abstraction that routes to backend implementations and raises clearly where one is missing. Keep it
  separate from the array namespace — different coverage and failure modes.

- **Tier 3 — the C code.** A hard wall, not a swap. C extensions are numpy-only and not differentiable.
  Three ways to handle backend calls that hit a C path:
  - *Refuse and redirect* — raise an informative error pointing to a pure-backend alternative. Fine for v1.
  - *Reimplement in-backend* — e.g. differentiable orbit integration via `diffrax` (JAX) or
    `torchdiffeq` (PyTorch). Where the actual ML value lives.
  - *Wrap the C integrator with custom differentiation rules* — `jax.custom_vjp` /
    `torch.autograd.Function`, supplying the VJP analytically via the variational/adjoint equations.
    Best long-term answer for orbit integration: gradients through the *fast* integrator without
    reimplementing it.

The differentiable-integrator decision is the most consequential one after the namespace plumbing,
and it's cleanly separable.

### Migration shape

Build the backend module + resolver first as a no-op for numpy (zero behavior change, shippable).
Then pick one self-contained, high-value, pure-math subsystem (a handful of analytic potentials) as
the pilot: numpy → JAX → torch. Establish conventions early (no bare `import numpy` in compute code,
no in-place mutation, no data-dependent branching in agnostic code) so they can be applied
mechanically. Make the test suite the spec: parametrize tests over backends; for JAX/torch add
gradient-vs-finite-difference checks — passing those proves a function is backend-clean far better
than "it ran."

Bake in from the start (painful to retrofit): preserve input **device and dtype** rather than
silently coercing (torch carries cpu/cuda + dtype as part of array identity); keep the resolver and
shared shims in one `galpy.backend` package, and put genuinely backend-specific reimplementations
(the diffrax integrator, the torch one) in clearly named modules the resolver dispatches to, rather
than scattering `if backend == ...` through the physics code.

Prior art worth skimming: `array-api-compat` and `array-api-extra`, scipy's and scikit-learn's
array-API support (the model to follow), and Keras 3's backend system as the *contrast* — global-first
because it's an application framework, which a library like galpy should not copy.

---

## 2. What `jax.custom_vjp` / `torch.autograd.Function` require

**Q:** If we go the C route, what's required for the `jax.custom_vjp` / `torch.autograd.Function` approach?

**A:** Both decorators do the same thing: tell the framework "don't differentiate the inside, here's
the VJP by hand." So it splits into the math (shared) and the per-backend plumbing (very different).

### The math you can't avoid

Wrapping the C integrator frees you from differentiating *the integrator*, but not from supplying
the Jacobian of *the vector field*. Orbit integration solves ẋ = f(x, θ, t) with x = (q, p).
Reverse mode turns a cotangent on the output, λ₁ = ∂L/∂x(t₁), into cotangents on x₀ and θ. Two routes:

- **Backward adjoint** (reverse-mode-native): integrate λ̇ = −Jᵀλ from t₁ back to t₀ with J = ∂f/∂x;
  then ∂L/∂x₀ = λ(t₀), and parameter gradient accumulates as −∫ λᵀ(∂f/∂θ) dt. The Neural-ODE /
  Pontryagin adjoint.
- **Forward variational** (sensitivity equations): propagate Ṁ = JM, M(0)=I alongside the orbit
  (plus ∂x/∂θ blocks), contract with λ₁ at the end. Natural for JVP; for VJP it computes more than
  needed, but it's forward-in-time.

Either way, **J = ∂f/∂x is the tidal tensor** (Hessian of the potential) with the symplectic block
structure, and ∂f/∂θ is the force derivative w.r.t. potential parameters. Good news: galpy already has
analytic second derivatives for many potentials, so the expensive ingredient mostly exists. Where only
forces are analytic, add the Hessian (or differentiate the force through the other backend's autodiff).

### The storage/chaos problem — the part that bites in dynamics

The backward adjoint needs the forward trajectory x(t) during the reverse solve, to evaluate J(t).
Either store it (dense output/checkpointing) or reconstruct it by integrating backward from x(t₁).
Reverse-reconstruction is the "free memory" trick — and a trap for galactic orbits: the adjoint's
growth rate *is* the linearized growth rate of the flow, i.e. the Lyapunov exponent. For chaotic
orbits (triaxial, near resonances, long integrations), the adjoint genuinely blows up (real
sensitivity — gradients through long chaotic integrations are physically ill-conditioned no matter
what) and reverse-reconstruction diverges catastrophically. **So for anything possibly chaotic, store/
checkpoint the forward trajectory, don't reverse-integrate it** — which is exactly what diffrax/
torchdiffeq already implement.

Quieter correctness point: the continuous adjoint gives the exact gradient of the *continuous*
solution, discretized possibly differently from the forward pass — so it's *not* exactly the gradient
of the discrete map C ran (optimize-then-discretize vs discretize-then-optimize). Usually fine for
smooth potentials; can hurt a sensitive optimizer. And the naive adjoint of a symplectic scheme isn't
itself symplectic, so you can lose the structure that justified leapfrog for long integrations
(there's a "symplectic adjoint" literature for this).

### Per-backend plumbing

- **PyTorch (easier):** subclass `torch.autograd.Function` with static `forward(ctx, x0, theta, ...)`
  and `backward(ctx, grad_out)`. In `forward`, detach to numpy, call C, stash via
  `ctx.save_for_backward(...)` (tensors) and plain `ctx` attributes (times, options). `backward` runs
  the adjoint/variational solve and returns one gradient per `forward` input — `None` for
  non-differentiated ones. Eager mode makes the boundary trivial (convert at the edge, mind the device).
  `torch.autograd.gradcheck` is effectively mandatory. `vmap`/`torch.compile` support needs extra
  `setup_context`/`vmap` staticmethods — out of scope initially.
- **JAX (same math, harder boundary):** `jax.custom_vjp` needs primal, `f_fwd` (output + residuals),
  `f_bwd` (residuals + cotangent → input cotangents), registered with `.defvjp()`. A black-box C call
  *cannot be traced into XLA*. Two options:
  - `jax.pure_callback` to escape XLA — lowest effort, but a host round-trip per call, needs a batching
    rule for `vmap`, and can't sit inside everything cleanly.
  - The FFI (currently `jax.ffi`, formerly under `jax.extend` — **verify the current name/signature
    against the installed version**, this API has moved) to register C as a proper custom XLA call.
    Faster, more work; the path if galpy orbits run inside a JIT-compiled training step.

  Clean division of labor: let C do the *forward* orbit with dense output; implement the *adjoint
  backward solve in JAX* (e.g. diffrax integrating λ̇ = −Jᵀλ reading the stored trajectory for the
  tidal tensor). Then `custom_vjp` just glues them, and only the forward C call crosses the callback.

### Price it against not doing it at all

Weigh wrapping C against just integrating the (backend-agnostic) forces with **diffrax** (JAX) or
**torchdiffeq** (PyTorch). They already implement the adjoint *and* the checkpointing/dense-output the
chaos problem forces on you — diffrax offers `RecursiveCheckpointAdjoint` (discretize-then-optimize,
consistent) vs `BacksolveAdjoint` (continuous, reverse-integrated) explicitly; torchdiffeq splits
`odeint` vs `odeint_adjoint`. Correct, maintained, JIT/vmap-composable. You give up galpy's C symplectic
integrator in the inner loop.

Custom-VJP-around-C is right specifically when (a) C is meaningfully faster than a diffrax/torchdiffeq
solve, (b) you need that speed inside the differentiated loop, and (c) you can live with the
continuous-adjoint inconsistency and the symplectic-structure question. Otherwise the maintainable
answer is the ODE-library route, reserving C-wrapping for the one or two hot paths. Prototype the
diffrax/torchdiffeq version first regardless — it gives a known-correct gradient to validate the custom
VJP against, beyond `gradcheck`/finite differences.

---

## 3. Implement the variational equations in C first?

**Q:** If we go the C route, would it make sense to first implement the variational equations in C?
Right now they're implemented in 2D (evolution of (dx, dv)), but not 3D.

**A:** Yes — and it's probably the right *first* concrete C task, and it partly updates the adjoint lean.

The variational equations propagate the state-transition matrix Ṁ = JM, M(0)=I, where J = ∂f/∂x is
exactly the tidal tensor with symplectic block structure (plus, in cylindrical coords, metric/Coriolis-type
cross terms). J is the most error-prone physics ingredient in *either* scheme — forward STM uses J,
adjoint uses Jᵀ. Building/validating it in the variational-equation context is the right place to get it
right, because there you have clean independent tests (finite-difference of the flow map; Lyapunov
exponents on known chaotic potentials) with nothing to do with autodiff. You de-risk the hard part where
correctness is checkable on its own terms.

The bigger point: the full 6×6 STM is cheap. For a 6-dim phase-space state, the entire Jacobian via
forward sensitivity is only 36 extra ODEs. Once you have M(t₁), the VJP w.r.t. initial conditions is
∂L/∂x₀ = Mᵀλ₁, a single transpose-matvec. So:

- No backward solve, no trajectory storage, **no reverse-reconstruction** — the chaos/storage problem
  evaporates for the IC-gradient case. The STM still grows at the Lyapunov rate (real ill-conditioning,
  unavoidable), but you store/reconstruct nothing.
- The backward pass crosses *no* boundary back into C. Forward calls C and returns (x₁, M); backward is
  pure host-side linear algebra in any backend. Big simplification for JAX — only the forward needs the
  callback; backward is a matmul.

This is the forward-vs-reverse tradeoff resolving in galpy's favor *because the state is tiny*: forward
sensitivity scales with #inputs differentiated, adjoint with #scalar-outputs. The earlier adjoint reflex
came from neural-ODE huge-state settings; for n=6 it inverts. For IC gradients (stream modeling,
sensitivity of final phase-space coords to launch conditions, finite-time Lyapunov work) forward STM is
the better method, and the 3D variational equations *are* that method.

Where it doesn't dominate: parameter gradients when n_θ is large. Extend with Ṗ = J P + ∂f/∂θ, P(0)=0,
giving ∂x₁/∂θ with backward = Pᵀλ (same free-backward structure), but forward cost scales as
6·(1+n_θ). For many-parameter fits against a scalar likelihood, that's where the adjoint (cost
independent of n_θ) eventually wins. Staging is clean: the adjoint reuses the *same* J transposed, and
you can validate the adjoint against the STM (related by transpose/inverse of the propagator) — the test
most likely to catch a sign error in the cylindrical cross terms.

Suggested sequence:

1. **3D variational equations in C as a standalone, independently useful feature** — not framed as
   autodiff infra. galpy lacks 3D Lyapunov / tangent dynamics / neighboring-trajectory propagation; this
   delivers them. Validate against finite-difference of the flow and against Lyapunov exponents.
2. **STM-based `custom_vjp` / `autograd.Function`** as a thin wrapper: forward returns (x₁, M), backward
   is Mᵀλ. Identical for JAX and torch, no backward C call. Covers IC gradients fully and few-parameter
   gradients via P.
3. **Adjoint, only if parameter-gradient cost demands it**, reusing J and validated against step 2.

Scope to size in step 1 (enumerable, not open-ended): every C potential needs its full 3D Hessian
exposed (R2deriv/z2deriv/Rzderiv and the φ second-derivatives, or Cartesian equivalents); some only carry
what the planar dxdv path needed. The 3D Jacobian in cylindrical coords isn't a mechanical "add z" — the
coordinate curvature contributes cross terms (angular-momentum and centrifugal pieces with R- and
z-dependence feed J), so the 2D code is the template for *structure* but each new block wants checking
against a finite-difference of the corresponding force component. Put real test coverage there before
trusting any gradient on top.

---

## 4. What Lyapunov-exponent checking actually means

**Q:** What kind of checking against Lyapunov exponents do you have in mind? I don't know whether we have
known Lyapunov exponents for any existing potentials (aside from zero for non-chaotic orbits).

**A:** Fair pushback — overstated. There's essentially no potential with an *analytically known* positive
Lyapunov exponent; the only closed form is λ = 0 for integrable systems. So "check against known Lyapunov
exponents" can't mean comparing to a ground-truth number, and the load-bearing tangent-space tests don't
need a known exponent at all.

### What the Lyapunov check really is (neither needs an analytic exponent)

- **Integrable → zero, with the right signature.** For known-regular cases (isochrone, Kepler, harmonic,
  any spherical, Stäckel/perfect ellipsoid), confirm the finite-time estimate decays the *right way*. For
  a regular (quasi-periodic) orbit, deviation grows linearly (torus shear), |δ(t)| ~ t, so
  FTLE λ(t) = t⁻¹ ln(|δ(t)|/|δ₀|) ~ ln(t)/t → 0. The diagnostic is the *power-law vs exponential*
  distinction: a sign error or missing cross term in J injects spurious instability → exponential
  deviation growth where there should be none. You're testing "no exponential growth where there should be
  none," which a wrong J fails loudly.
- **Literature regression on standard chaotic testbeds.** Well-established *numerical* (not analytic)
  values to reproduce. Hénon–Heiles (trivial to add; tabulated Lyapunov exponents/times at given energies
  everywhere in the literature); the flattened logarithmic potential ("Binney's") box/loop orbits (the
  galactic-dynamics-native version). Cross-code regression — catches gross errors, not a tight proof.

So the Lyapunov layer is real but the *weakest* validation — a zero-test plus a regression. Don't trust
the 3D J on it alone.

### The tests that actually pin down J — no known answer required

In decreasing strength of ground truth:

1. **Closed-form STM where it exists.** 3D harmonic oscillator has analytic M(t) (sines/cosines at
   eigenfrequencies); Kepler has the classical two-body STM (Lagrange f, g and derivatives). Genuine
   analytic ground truth for the *entire matrix*, far stronger than any Lyapunov statement. If the 3D
   variational code can't reproduce the Kepler STM, there's a bug, full stop.
2. **Symplecticity of M.** For a Hamiltonian flow, MᵀΩM = Ω at all times to tolerance, for *every*
   potential, chaotic or not. Most sensitive to the cylindrical curvature/Coriolis cross terms — a sign
   error there breaks symplecticity even when diagonal blocks look fine. det M = 1 (Liouville) is the
   weaker corollary. **Caveat:** must be evaluated in canonical variables. galpy's
   (R, vR, vT, z, vz, φ) aren't canonical (canonical momenta: pR = vR, p_φ = R·vT = Lz, pz = vz), so run
   the check on (R, φ, z, pR, p_φ, pz) or carry the variable-transform Jacobian into Ω.
3. **Flow-direction propagation — exact and free.** For an autonomous system the vector field itself
   solves the variational equations: M(t) f(x₀) = f(x(t)) exactly, at every step (since d/dt f(x(t)) = J f
   with IC f(x₀)). Check, with zero extra integration, that propagating the initial force/velocity vector
   through M reproduces the force/velocity at the current point. One column's worth of J validated
   exactly, continuously, no reference solution.
4. **Conserved-integral relations.** Axisymmetric → Lz conserved; spherical → full L. The gradient of each
   integral gives an exact tangent-space relation M must respect (∇I·δx conserved; ∇I propagated by the
   inverse-transpose). Extra exact constraints exercising the φ and angular-momentum blocks.
5. **Finite-difference / nearby-orbit cross-checks.** Column i of M is
   (x(t; x₀ + ε eᵢ) − x(t; x₀))/ε — the FD-of-the-flow-map check *is* a direct STM check, column by
   column. The "two nearby orbits, difference and divide" (shadow-orbit) version agrees with the STM for
   short times before the finite separation saturates. Neither needs a known answer; both probe J directly.

Honest hierarchy: Kepler/harmonic closed-form STM and symplecticity are the tight correctness tests;
flow-direction and conserved-integral relations are exact freebies that localize errors to specific
blocks; FD/nearby-orbit cross-checks confirm the whole matrix numerically; the Lyapunov layer reduces to
the integrable-zero signature plus Hénon–Heiles/log-potential regression. Lyapunov is worth having (the
only test exercising long-time accumulated behavior and the chaotic regime) but it's the confirmation,
not the proof. The proof is symplecticity plus the closed-form cases.

---

## 5. Second derivatives of a C orbit integration

**Q:** What if you want a second derivative of a C orbit integration function? Does that just work once
you have the variational equations?

**A:** No, not for free — and this is where the STM-VJP wrapper quietly stops composing.

The STM gives the *first* derivative as a value, not as a differentiable computation. Wrapping C with
`custom_vjp`/`autograd.Function` tells the framework "stop tracing here." The gradient Mᵀλ is a number out
of C; the framework can't differentiate through it because it never saw how M was produced. The autodiff
graph dead-ends at the boundary you drew.

Having the variational equations buys the *value* of the first-order sensitivity (M). Second derivatives
need the next thing in the hierarchy:

- **The trap:** a `custom_vjp` is not automatically twice-differentiable. JAX will try to differentiate
  the *backward* rule — but it calls back into C / treats M as opaque, so it errors or silently gives
  garbage. Torch similarly requires the backward written in differentiable ops (or `once_differentiable`,
  which refuses rather than lies). So it specifically does *not* "just work," and the JAX failure can be
  silent — the dangerous part.

- **Option A — extend the propagated system (second-order variational equations).** Just as Ṁ = JM, the
  second-order sensitivity (Hessian of the flow, ∂²x(t)/∂x₀²) satisfies its own linear ODE driven by M and
  by **K = ∂²f/∂x²** (derivative of the tidal tensor): schematically Ṅ = J N + K[M, M]. Same move one level
  up: add ODEs, integrate alongside orbit + M, read off at t₁. Exact, reuses all machinery. Costs: the new
  physics ingredient K = ∂³(potential)/∂x³ (the *third* derivative of the potential — adds force Hessians
  beyond the J Hessians), and size — N is rank-3, 6×6×6 = 216 components (126 with input-index symmetry).
  Trivial for n=6 computationally; the real cost is the third-derivative implementation per potential.

- **Option B — let the backend differentiate a backend-side first derivative.** To get a genuine second
  derivative this way, M itself must be a differentiable function of x₀ in the backend → integrate the
  variational equations *in the backend* (diffrax/torchdiffeq), not in C. At which point you're on the
  ODE-library route, where second (and higher) derivatives *do* just work because nothing is opaque —
  `jax.hessian`/double `grad` compose by construction. The strongest argument for the diffrax/torchdiffeq
  option: higher-order autodiff is exactly where wrapping opaque C hurts most.

Decision rule:

- **First derivatives only** (most fitting, IC sensitivity, FTLE) → STM-in-C wrapper, composes fine.
- **Second derivatives, keep the C integrator** → Option A: second-order variational equations in C,
  requiring third potential derivatives. Coherent and fast; real cost in per-potential derivatives.
- **Second/higher derivatives, willing to integrate in-backend** → diffrax/torchdiffeq, automatic, paying
  the inner-loop speed cost instead of the derivative-implementation cost.

Name what you need the second derivative *for*: a Hessian of a scalar loss w.r.t. ICs or a few parameters
(Laplace approximation, Newton steps, Fisher information for stream/likelihood work) needs the full
second-order object → Option A pays off. But Hessian-*vector* products (what most second-order optimizers
and much UQ actually consume) are cheaper: an HVP is a directional derivative of the gradient, so propagate
the second-order equations contracted along a *single* direction (~6 extra ODEs per probe, not 126), or do
forward-over-reverse in the backend without materializing the Hessian. Before committing to full ∂³V in C,
pin down whether the downstream task wants the Hessian or just HVPs — it's the difference between 126 extra
ODEs plus full ∂³V and a handful of extra ODEs per probe.

---

## 6. Derivatives of the action-angle (root-find + 1D quadrature) code

**Q:** Some action-angle stuff is also in C — generally root-finding for turning points then 1D numerical
integrals. Can I easily do derivatives of that analytically and implement them, or should I just use the
Python jax/torch implementations (currently incomplete)?

**A:** Clean theory answer, and it cuts *toward* analytic derivatives — root-finding and quadrature are
exactly where autodiff's "differentiate the algorithm" instinct is wrong and a hand-supplied rule is both
correct and cheap.

### Turning points: implicit function theorem, not differentiated bisection

Turning points are roots of an equation like p_r²(r; E, L) = 0, i.e. solve g(r; θ) = 0 for r. Naive AD
differentiates *through the solver iterations* — expensive (unrolls into the graph) and numerically poor
near convergence. The IFT gives the derivative of the *converged* root directly:

> dr*/dθ = −(∂g/∂θ) / (∂g/∂r), evaluated at r*

Ingredients are partials of the *defining equation* — energy/angular-momentum/force-type derivatives you
already have. No solver differentiation, no iteration storage. This is what `jax.lax.custom_root` /
implicit-diff machinery does under the hood.

**Subtlety:** ∂g/∂r → 0 at exactly the turning points for the radial action, because radial momentum has a
square-root branch point there (p_r ~ √(r − r_peri)). So the IFT denominator vanishes and dr*/dθ looks
singular — not a real divergence, the well-known coordinate singularity of the action integral at the
endpoints, but you can't naively apply IFT with g = p_r². Fix: define the root via a regularized quantity
(work with p_r² whose r-derivative is finite at the root, or change variables) — ties directly into the
integral handling below.

### Action integral: differentiate under the integral sign, watch the endpoints

J = (1/π) ∫ p_r(r; θ) dr between turning points. Want dJ/dθ. Use Leibniz's rule:

> dJ/dθ = (1/π) [ ∫ ∂p_r/∂θ dr + p_r·(dr/dθ)|_at the limits ]

**Payoff: the boundary terms vanish**, because p_r = 0 *at* the turning points by definition. So the
moving-endpoint contributions (usually the annoying part) are exactly zero, leaving:

> dJ/dθ = (1/π) ∫ (∂p_r/∂θ) dr

a *new integral of the same kind*, same limits, of an analytic integrand
(∂p_r/∂θ = (∂p_r/∂E)·(∂E/∂θ) + …, all algebraic in the potential and its derivatives). Compute it with the
*same C quadrature machinery*. You're not differentiating the integrator; you hand it a different integrand
and reuse it verbatim.

**Caveat connecting to the IFT singularity:** p_r = 0 at the limit kills the boundary term, but ∂p_r/∂θ has
an integrable singularity there (same √ branch point), so the derivative integral has the same
square-root-singular endpoint behavior as the original. Practically: reuse whatever endpoint handling the C
action integral already uses (sin/cos or u-substitution regularizing endpoints, e.g.
r = r_peri + (r_apo−r_peri)·sin²(½θ); Gauss–Chebyshev weighting) for the derivative integral — equally
well-behaved. If C integrates J cleanly, it integrates dJ/dθ cleanly. The IFT-singularity and
integral-singularity issues are one issue with one shared fix.

### So: analytic derivatives, genuinely easy here

Both pieces are "reuse existing machinery with a different integrand/expression," not "implement something
new and hard":

- **Turning points** → IFT, one algebraic expression in partials you have, wrapped as the root's custom
  derivative.
- **Actions** → Leibniz with vanishing boundary terms → one more quadrature of an analytic integrand, same
  limits, same substitution, same C integrator.
- **Frequencies and angles** are the same story one layer on — Ω = ∂H/∂J etc. are derivatives of the same
  objects, so once dJ/dθ and the action machinery are differentiated, the canonical structure hands you the
  rest. (The Hessian ∂²H/∂J² giving the frequencies is again a derivative-of-integral of the same form.)

Wrapping is parallel to the orbit case: `custom_vjp`/`autograd.Function` whose forward calls the C
root-find + quadrature and whose backward evaluates these closed-form sensitivities — and backward needn't
even call back into C *if* you precompute the few extra quadratures in the forward pass and stash them (or
make one more C quadrature call for the derivative integrand). Either way, no solver/integrator
differentiation crosses the boundary.

### When to reach for the Python jax/torch side instead

1. **Second derivatives of the actions.** You differentiate the dJ/dθ integral again — doable analytically
   (another Leibniz pass; boundary terms need rechecking because ∂p_r/∂θ is *not* zero at the endpoints, so
   the second derivative *does* pick up boundary contributions), but bookkeeping grows and in-backend traced
   quadrature composes more safely. Same higher-order story as the orbit case: opaque-C wrapping is fine to
   first order, fragile above it.
2. **Not a clean 1D-integral-of-known-integrand** — Stäckel-fudge approximations, or any scheme with
   auxiliary fits/interpolation/iteration that isn't a single well-defined quadrature. The "analytic
   derivative of a clean mathematical object" story breaks because the *object* isn't clean; differentiating
   the actual algorithm in-backend is more honest. Sort action methods by exact-quadrature (IFT+Leibniz
   applies beautifully) vs fudge/interpolated (lean Python/AD).
3. **Want it working now and speed isn't critical** — the in-backend implementation, once it exists, gives
   all-orders derivatives for free and is less code to maintain than per-method custom rules in C. Since the
   Python side is incomplete, there's a real argument to *finish the Python side first* as the correctness
   reference and all-orders fallback, then add C custom-derivative rules only for the exact-quadrature
   methods where the analytic derivative is this clean and speed matters.

**Recommendation:** exact-quadrature actions (spherical J_r, isochrone/epicycle analytics, anything that's
genuinely root-find-then-1D-integral) are worth the analytic C derivatives — easy for the reasons above and
the hot paths. The fudge/approximate methods are not; lean on backend AD there. Regardless, finishing the
Python jax/torch implementation buys the cross-check that the C analytic derivatives are right — same role
the diffrax solve plays for the orbit VJP.

---

## Quick reference: the recurring pattern

Across orbits and actions, the same shape recurs:

- **First-order derivatives through opaque C are cheap and clean** via a hand-supplied rule (STM/Mᵀλ for
  orbits; IFT + Leibniz for actions), because the small phase-space dimension makes forward sensitivity
  trivial and the math reuses ingredients galpy largely already has (Hessians/forces).
- **The hard part is always the physics ingredient**, not the autodiff plumbing: J = ∂f/∂x (and its
  transpose) for orbits; partials of the defining equation / integrand for actions.
- **Higher-order derivatives are where opaque-C wrapping stops composing.** Either extend the propagated
  system (more variational equations + one more potential derivative) or move the computation in-backend
  (diffrax/torchdiffeq / traced quadrature), where all orders are automatic.
- **Validation hierarchy matters more than "it ran":** closed-form cases + structural invariants
  (symplecticity, conserved-integral relations, flow-direction propagation) are the tight tests;
  finite-difference and a known-correct backend implementation are the cross-checks; Lyapunov regression is
  confirmation, not proof.
