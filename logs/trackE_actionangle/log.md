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

## #1005 round 2 (review feedback) — dθ/dt=Ω tests + HarmonicInverse

**dθ/dt = Ω autodiff consistency (the showcase).** Angle variables evolve
linearly in time, so dθ_i/dt = ∇_x θ_i · ẋ (Hamiltonian flow) MUST equal the
returned frequency Ω_i. New tests differentiate the forward angle (jax.grad /
torch.autograd) wrt the phase-space vector, dot with the EOM (galpy internal
coords), and assert == the SAME call's Ω (so the sign/wrap convention is
auto-referenced, no manual bookkeeping). **Holds to ~1e-15** for Harmonic
(dθ/dt=ω) and all three Isochrone angles, prograde AND retrograde (Lz<0).
- **Adversarial-verify catch:** the investigation workflow's test-design agent
  claimed dθ/dt≠Ω for the Isochrone (ratios ~0.11/0.37) and wanted a strict
  xfail — but it used the **private amp-less `_Rforce`**, giving a
  self-consistent-but-wrong dynamics where dE/dt=0 yet dθ/dt≠Ω. The
  dtheta-verify agent (public `Rforce`, amp included) got ~1e-15. I reproduced
  BOTH myself: public Rforce → 1e-15, private _Rforce → 0.11/0.37 (= amp factor
  4.5). The test uses **public Rforce** and is a PASSING test (not xfail). Two
  agents disagreed → resolved empirically, never trust an un-verified claim.

**actionAngleHarmonicInverse** — closed-form ((J,θ)→(x,v): amp=√(2J/ω),
x=amp·sinθ, vx=amp·ω·cosθ); trivial namespace-swap, byte-identical. Migrated +
round-trip/grad tests. No reason it was separate beyond subclassing
actionAngleInverse.

**actionAngleIsochroneInverse — DEFERRED to the root-find increment.** It is NOT
closed-form: it solves Kepler's equation eta - a·e/ab·sin(eta) = ar per-element
via `scipy.optimize.newton`+brentq in a Python loop (+ numpy.empty index-assign
loop, masked writes). Needs a backend Newton/fixed-point solver with
implicit-diff — same machinery as Vertical/Spherical. So the inverses SPLIT:
HarmonicInverse (closed-form, done in #1005); IsochroneInverse (root-find, joins
Vertical+Spherical). numpy round-trip tested now as the reference.

## IsochroneInverse backend (#1007, stacked on #1005) — DONE

The first root-find migration. Key win: **the shared backend root-finder already
exists** — `galpy.backend.optimize.brentq` (numpy→scipy byte-identical; jax/torch→
vectorised sign-preserving `bisect_root` + one-Newton-step implicit-diff, documented
in that module). So task #86's "shared helper" is satisfied by the infra; I just
*reuse* it. IsochroneInverse is its first AA consumer + reference.

- Kepler eq `f(η)=η−(a·e/ab)·sin η−angler`, `a·e/ab<1` ⇒ strictly monotone on
  `[0,2π]` (always a valid bracket; `f(0)≤0≤f(2π)`).
- **Branch on namespace**: numpy keeps the exact `scipy.optimize.newton`(+brentq
  fallback) loop → byte-identical; jax/torch call `backend.optimize.brentq` on
  `[0,2π]` vectorised over the angler array → differentiable wrt the actions.
- Masked writes → `xp.where`; double floor-mod `(angler%−2π)%2π` maps directly
  (verified torch/jax `%` are floor-mod, **not** fmod — the agent's fmod worry was
  unfounded).
- **Verified:** numpy byte-identical (b∈{0.3,0.8,1.5}, prograde/retrograde/jz=0);
  jax+torch parity ~3e-16; backend brentq vs scipy.newton ~9e-16 (near-radial
  c=0.99769 ~5e-13); forward∘inverse round-trip Jr/Jphi/Jz ~1e-16; d(ΣR)/djr AD==FD
  ~5e-10. 52 backend tests green.
- **Adversarial-verify catch (2nd this session):** the design workflow's
  byte-identity-policy + implicit-diff agents FAILED to return structured output;
  the kepler agent's byte-identity strategy *assumed* the branch-scipy-for-numpy
  policy — I confirmed it myself by reading `galpy.backend.optimize` (it IS the
  documented policy) rather than trusting the assumption.

PR opens after #1005 automerges (rebase the stacked branch onto feat/backends).

## actionAngleSpherical PR-1 (#1008, open) — actions + EccZmaxRperiRap

The hardest AA method. Design fan-out (3 agents) numerically verified every piece
vs scipy BEFORE implementing; an implement→adversarial-verify workflow then built
+ independently checked PR-1 (I re-ran byte-identity + tests myself).

- **Purely ADDITIVE** (`elif is_backend_array(R):` in `_evaluate` + `_EccZmaxRperiRap`;
  numpy scipy-loop unchanged → byte-identical, 0 deletions).
- **rperi/rap**: shared `backend.optimize.brentq` on `E−Φ(R)−L²/2R²`, vectorised, with
  a **fixed-schedule** expanding bracket search (vectorises `_rapRperiAxiFindStart`)
  + `xp.where` for circular/peri/apo/plunge. roots ~5e-15 vs scipy; implicit-diff
  `d(rperi,rap)/d(E,L)`==FD.
- **Jr**: `r=rperi+(rap−rperi)sin²θ` substitution (kills the √ at BOTH turning points)
  + `backend.quadrature.fixed_quad(0,π/2,n=25)`; clip radicand ≥0 (NaN-poison guard).
  matches scipy 3-4e-16.
- **Key design call (verified):** for Jr the **sin² single-panel** substitution beats
  `transformed_quad`'s interior-split/X³ boundary-layer (which is for endpoint
  *near*-singularities, not a real √ turning point — only reached 5e-8 vs sin²'s
  3e-16). `transformed_quad` / the t² two-panel split is the right tool for the
  FREQ/ANGLE integrals (PR-2), where galpy already uses `r=rperi+t²`/`r=rap−t²`.
- **Validation:** numpy byte-identical (Log/NFW/Hernquist, all branch cases + scalar +
  fixed_quad + gamma=1.0). jax+torch parity ~3.8e-12, grad-vs-FD ~6e-8. gamma≠0 backend
  → NotImplementedError. 60 backend + 20 numpy + 18 adiabatic green.

**PR-2 (#1009, open) — frequencies + angles → actionAngleSpherical FULLY migrated.**
Additive `elif` in `_actionsFreqs`/`_actionsFreqsAngles`; new `_panel_backend` (one
t²-substituted GL panel, per-object limit folded in via `t=lim·s` on a fixed `[0,1]`
panel so `fixed_quad`'s a,b stay scalar), `_calc_or_op_backend` (Tr/I two panels →
Or=2π/Tr, Op=I·Or/2π; circular `Jr<1e-9` → `xp.where(epifreq/omegac)`), and the
`_calc_angler`/`_anglez`/`_long_asc_backend` angle assembly (every if → `xp.where`:
r<Rmean panel, vr<0/vT<0 quadrant fixes, sinpsi inclination + non-finite→phi). numpy
byte-identical (base-vs-port array_equal, pro+retrograde). jax+torch parity Or~2.8e-9,
Op/angles~1e-9, grad~1e-7, circular branch exact. 68 backend + 20 numpy green. Built
design→implement→adversarial-verify (SHIP), re-checked by me. One documented edge: the
fully non-inclined z=0 IC (measure-zero non-smooth angle kink; numpy warns too).

## Vertical (still pending) — needs linearPotential/verticalPotential migrated first
`galpy.backend.quadrature` has `fixed_quad`, `gauss_legendre`, `transformed_quad`
(finite GL with a near-singular **interior-point split** — the `xmax-t²` /
`r=r_peri+t²` turning-point substitution), `fixed_quad_semiinfinite`, `nested_quad`;
`galpy.backend.optimize` has `brentq` + `bisect_root` (backend-agnostic). So the
1D-numerical `actionAngleVertical` (`quad` + `brentq` for `xmax`), the 3D→1D
`actionAngleSpherical` (quad + root-find + implicit diff), AND
`actionAngleIsochroneInverse` (Kepler-equation Newton root-find) all map onto a
shared backend root-find + implicit-diff pattern — the next Track-E PR(s), after
#1005 lands.
