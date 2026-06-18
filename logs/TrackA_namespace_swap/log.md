# Track A — potential namespace-swap fan-out (5 parallel agents, worktrees)

Orchestrated via the Workflow tool: 5 worktree-isolated agents, one per family, each
migrating its family to backend-agnostic compute, writing its own test module,
validating numpy byte-identity + jax/torch parity + grad-vs-FD, committing to its own
`feat/P2.x-*` branch. ~700k subagent tokens, ~34 min wall-clock. All coordinated
correctly (each rebased/reset onto feat/backends for get_namespace; none touched the
shared conventions allowlist).

## PRs (drafts into feat/backends / #892)
| PR | Family | Migrated | Tests | Notes |
|----|--------|----------|-------|-------|
| #893 | spherical | SphericalPotential base + 12 | 95 pass/15 skip | caught torch-`power`/`fabs` bugs |
| #894 | disk | MN, Kuzmin, FlattenedPower, IsothermalDisk, KG, RazorThin(surfdens) | 138 pass | MN isinstance fast-path gated on `xp is numpy` |
| #895 | ellipsoidal | EllipsoidalPotential base + GL quad + 7 triaxials | 258 pass | cache gated numpy-only; traced path pure |
| #896 | halobar | all 10 (DehnenBar 30+ mutations, SpiralArms, …) | 60 pass | nothing deferred; flagged pre-existing CosmphiDisk bug |
| #897 | dissipative | 0 (all need scipy.special) | 6 pass/2 xfail | scaffold + convention; deferred to Pspecial |

## Deferred to Pspecial (scipy.special/interpolate) — the P2.5 batch
TwoPower/PowerSpherical/NFW `_evaluate` (hyp2f1/gamma/xlogy), Einasto, interpSpherical,
Ring (ellipk/e), DoubleExp, RazorThin forces (Bessel), various `_surfdens` (complex-sqrt
branch), Chandrasekhar/FDM dyn-fric (erf/sici), KuzminKutuzov (coords transform).

## CI wiring
- feat/backends backend job TEST_FILES set to glob `tests/test_backend_*.py` (commit acfa0e64),
  so each merged family's test module auto-runs under jax+torch. Family PR CI uses the
  merge with feat/backends (glob), so their family tests run in their PR CI.

## Findings worth carrying to the paper
- torch's array-api-compat namespace lacks `power`/`fabs` → use `**`/`xp.abs` (portable).
- The safe-denominator `xp.where` idiom is essential: a dead `0**negative`/`0/0` branch
  poisons reverse-mode grads otherwise.
- Mutable per-instance caches (Ellipsoidal, SoftenedNeedleBar, SpiralArms self-state) must be
  gated on `xp is numpy`; traced path computes pure locals.
- Pre-existing bug surfaced: CosmphiDiskPotential._Rphideriv (R<rb branch) off by factor m.

## TODO (follow-ups)
- Extend tests/test_backend_conventions.py MIGRATED allowlist to the migrated families
  (after each merges into feat/backends — can't do before, the unmigrated source would fail the guard).
- P2.5 (scipy.special-backed) after Pspecial.
- P2.6 wrappers last.
- Separate fix PR for the CosmphiDisk _Rphideriv bug.

---

## Track A tail + completeness (2026-06-18)

The P2.x fan-out (above) + P2.5/Pspecial + P2.6 wrappers brought every potential's
*compute core* (value, all six 2nd derivatives, analytic density) onto the namespace.
After P2.6, a tail of correctness/coverage PRs closed the remaining systemic gaps. All
merged into feat/backends (#892); all numpy byte-identical unless noted. Dates/states
verified via `gh pr view`.

| PR | Merged | What it fixed |
|----|--------|---------------|
| #959 | 2026-06-15 | SCF/Multipole jax/torch vectorization — killed the O(P) per-point `numpy.ndindex` graph that effectively hung along orbits/at scale; `SphericalHarmonicPotentialMixin` vectorized too, so MultipoleExpansion got it for free; un-deferred the 31 now-passing jax SCF tests (test_scf.py 47/47 jax). |
| #960 | 2026-06-16 | Central coordinate coercion — fixed the *systemic* torch scalar-input rejection at one boundary (`potential_physical_input` in conversion.py, the outermost decorator on every coordinate-taking method/`evaluate*`); new `coerce_coords` in `_namespaces.py` (numpy strict pass-through → byte-identical; python scalars/int arrays lift to backend float64). Coercion gated to migrated potentials (the `_backend_compatible` opt-in flag), fixing ~39 at once instead of per-file whack-a-mole; torch test_potential.py 140 passing (was the dominant failure mode). |
| #962 | 2026-06-17 | Migrated the straightforward potential/ tail: PseudoIsothermal, KuzminKutuzov, Ferrers, RazorThinExponential. |
| #963 | 2026-06-16 | torch/jax time-dependent amplitude for the DehnenSmooth + GaussianAmplitude wrappers. |
| #964 | 2026-06-16 | backend.special: added `iv` (modified Bessel I_n) and `sici` (sine/cosine integral); vs scipy 1e-13..1e-15 + gradcheck. |
| #965 | 2026-06-16 | General backend interpolation + quadrature utilities (galpy.backend.interpolate / galpy.backend.quadrature). |
| #966 | 2026-06-16 | backend.special: hardened fallback edge cases (poles, domain endpoints, AD, numpy params). |
| #967 | 2026-06-16 | Fixed CUDA device-mismatch on mixed scalar/array coordinates (SCF, coords, DoubleExp). |
| #968 | 2026-06-16 | DoubleExp._evaluate: treat a 0-d-array R as scalar (fixes the coerced-scalar reshape error introduced by #960's coercion). |

Of these, #960 is the structurally important one: the per-family compute migration was
done family-by-family, but *input handling* was still per-file — torch rejected bare
python/numpy scalars all over the convenience surface. Pushing the backend coercion up to
the single `potential_physical_input` decorator turned ~39 per-file fixes into one, and
exposed the residual 0-d-vs-scalar edge (#968).

## Are we well and truly done with potential?

The **compute core was done** after the P2.x + P2.5 + P2.6 work. What remained was the
**convenience layer + functional interface**, which still carried bare `numpy.*` on the
compute path: silently fine under eager torch, but failing under `jax.grad`
(`TracerArrayConversionError`). That is the FINAL Track A potential gap, in flight now as
**#975** (feat/potential-backend-complete, OPEN, base feat/backends):

- module-level + class-method `vesc / rtide / ttensor / tdyn` (incl. `eigenval=True` via
  `xp.stack` + `xp.linalg.eigvals`), plus the rest of the `Potential.py` convenience
  methods (`r2deriv, vcirc, dvcircdR, omegac, epifreq, verticalfreq, flattening, vterm`),
  `planarPotential.py` (epifreq/vcirc/omegac/vesc), and `Force.rforce` plumbing — all
  `get_namespace`-dispatched (numpy `xp IS numpy` → byte-identical).
- the hard part: new **galpy.backend.optimize.brentq** (numpy → scipy.optimize.brentq
  byte-identical; jax/torch → branch-free bisection + one-Newton reparameterization
  giving the first-order implicit-function-theorem sensitivity `dx*/dθ = −(∂f/∂θ)/(∂f/∂x)`,
  jit/vmap-compatible, no internal jit) and **galpy.backend.quadrature** (numpy →
  scipy.integrate.quad; jax/torch → fixed-order Gauss–Legendre, differentiable w.r.t.
  params and limits). `rl, rE, LcE, zvc, zvc_range, mass` retrofit onto these for jax/torch.
- ledger +50 (branch-local until #975 merges).

So the answer to "are we done with potential?": the compute core was already done; **#975
closes the convenience/functional surface**. The single deferred Track A potential item is
**dissipative-force backend differentiability** (Chandrasekhar/FDM dyn-fric etc.) — scaffolded
back at P2.x (#897) and not yet on the namespace; per-potential follow-ups (#975's out-of-scope
list: Ferrers quadrature forces, MovingObject/NonInertialFrame, KuzminKutuzov coords) are
normal long-tail, not systemic gaps.
