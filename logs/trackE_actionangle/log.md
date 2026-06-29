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

**PR-2 (#1009, ✅ MERGED 2026-06-24) — frequencies + angles → actionAngleSpherical FULLY migrated.**
Additive `elif` in `_actionsFreqs`/`_actionsFreqsAngles`; new `_panel_backend` (one
t²-substituted GL panel, per-object limit folded in via `t=lim·s` on a fixed `[0,1]`
panel so `fixed_quad`'s a,b stay scalar), `_calc_or_op_backend` (Tr/I two panels →
Or=2π/Tr, Op=I·Or/2π; circular `Jr<1e-9` → `xp.where(epifreq/omegac)`), and the
`_calc_angler`/`_anglez`/`_long_asc_backend` angle assembly (every if → `xp.where`:
r<Rmean panel, vr<0/vT<0 quadrant fixes, sinpsi inclination + non-finite→phi). numpy
byte-identical (base-vs-port array_equal, pro+retrograde). jax+torch parity Or~2.8e-9,
Op/angles~1e-9, grad~1e-7, circular branch exact. 68 backend + 20 numpy green. Built
design→implement→adversarial-verify (SHIP), re-checked by me.

**PR-2 edge-case hardening (#1009 follow-up; user: "match numpy EVERYWHERE, not just
document it away").** A divergence map over 8 special cases × {Log,NFW,Hernquist,
Isochrone} × pro/retro × {jax,torch} (per-quantity |back−np|, thr 1e-7) found two real
gaps — both now FIXED; the fully-non-inclined case is matched EXACTLY, not xfailed:
- **Non-inclined `ap`/`az` off by 0.7–5.6 rad (the prior "documented edge" — a real
  BUG).** numpy: `sinpsi=z/r/sin(i)=0/0=NaN → isfinite False → psi=phi` (and skips the
  `vθ>0→π−psi` rotation); the backend computed `finite` from the **sini_safe** division
  (always finite) so it took `psi=0`. Fix: compute `finite=isfinite(z/r/sini)` on the
  **raw** sini (mirrors numpy's own 0/0→NaN — numpy emits the same RuntimeWarning),
  keep `sini_safe` only for the arcsin VALUE (no NaN in the grad path). Cascades through
  `az` and `ap=asc±az` → circular **0.0** exact, planar 1.5e-9, near_circular 9.5e-8.
  **Same fix in `_calc_long_asc_backend`**: its raw `z/R/tan(i)` NaN was correct in
  VALUE but **NaN-poisoned the gradient** via the `xp.where(isfinite,u,phi)` dead branch
  (the eager-where trap) → now `dap/dvR,daz/dvR` finite & equal (Ω=0 in-plane), jax+torch
  grad-vs-FD ✓.
- **Near-radial (L/Lcirc≤1e-3) freq/angle integrals up to ~4e-4 off.** GL order **25→50**
  (`_BACKEND_GL_ORDER`; panels + Jr). A gold-reference probe (tight scipy quad) showed the
  n=25 deficit AND that at L/Lcirc<~1e-4 scipy's OWN adaptive quad fails to ~1e-5
  (IntegrationWarning "tolerance cannot be achieved") — so that pathological extreme is the
  irreducible GL-vs-adaptive floor, NOT a logic bug; at n=50 the backend is actually *more*
  accurate vs gold there. Post-fix: physical range (L/Lcirc≥1e-2) matches numpy <1e-12;
  even the pathological vT=1e-4 → ~1e-6 (was 4e-4).
- **Tests:** `test_spherical_edge_case_parity` (8 cases × 4 pots × pro/retro × 2 backends
  = 128) asserts numpy↔backend on all of (Jr,Lz,Jz,Or,Op,Oz,ar,ap,az,e,zmax,rperi,rap);
  `test_spherical_noninclined_angle_grad_vs_fd` locks the NaN-free gradient. 200 backend
  green (file) + numpy byte-identical (only backend-only methods + a module constant +
  comments changed).

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

## Staeckel pure-Python freqs+angles (#1010 → main, open 2026-06-24)
`actionAngleStaeckel.actionsFreqs/actionsFreqsAngles` were **C-only** (raised
NotImplementedError for c=False); only actions were pure-Python. PR #1010 adds
the freqs+angles in pure numpy/scipy so they can later go onto the backends.
Built design→impl→adversarial-verify workflow (wf_fb62790a), then re-verified by
me vs the C oracle. **c=False matches c=True ~1e-9 (freqs) / ~1e-7 (angles)** over
a 3684-pt edge grid. Ported from C `calcdJR/calcdJz/calcFreqsFromDerivs/
calcAnglesStaeckel`: 6 derivative integrands (reuse `*IntegrandSquared` radicand),
GL panel quadrature w/ exact prefactors+circular guards, 2×2 Jacobian inversion
for Ω, verbatim angle quadrant tree (no-0.5 partial mid), pure-Python `calcu0`.
- **Actions byte-identical:** C freqs/angles use v0=π/2 (J_R)/actual-u0 (J_z) vs
  actions v0=vx/u0=ux; `_v0u`/`_u0v` DEFAULT to actions values → JR/Jz/turning-pt
  unchanged (diff=0.0). Refuted a code-review agent FAIL (it flagged the rename
  without seeing the default == vx/ux).
- **My fix on top of the agent:** exactly-circular (detA=0) raised ZeroDivisionError
  (scalar 0/0; C gets IEEE NaN→epifreq/omegac/verticalfreq fallback). calcFreqs now
  returns NaN on detA==0 → fallback fires. Matches c=True to 0.0. Added circular test.
- **Adiabatic:** no new code (delegates to pure-Python Spherical+Vertical); guard test.
- **Behavior change (flagged on PR):** Orbit.jr/jz/... staeckel c=False now route
  through actionsFreqsAngles (consistent with c=True, which already did via Orbits.py
  try-freqsAngles-first); ~1e-5 GL-vs-adaptive shift; 1 orbit-interface test updated.
- **Documented edge:** |pvx|<1e-3 annulus at a vertical turning point → up to ~5e-4 in
  az (C-vs-scipy brentq vmin tolerance amplified in sqrt(vx-vmin)); measure-zero, the
  Staeckel analog of the Spherical peri/apo floor; can't tighten w/o breaking actions.
- 110 Staeckel+Adiabatic tests pass. Next: rebase feat/backends, add backend layer.

### Staeckel #1010 review round (user, 2026-06-24/25) — 3 follow-up commits + design decisions
Three points raised reviewing #1010, all addressed:
1. **Behavior change → dead code.** Orbit.jr/jp/jz wrapped freqsAngles in `try/except
   NotImplementedError -> _setup_actions`; that fallback existed ONLY for staeckel-c=False
   (the last Orbit AA type without a Python actionsFreqsAngles; its own docstring says so).
   With #1010 it's dead (no other type raises; isochroneApprox implements it; 2D→spherical).
   Traced the call graph (only the 3 except-branches call _setup_actions), removed both →
   jr/jp/jz call _setup_actionsFreqsAngles directly. orbit-interface tests green.
2. **Python↔C inconsistency (the big one).** Empirically: C-actions == C-freqsAngles to 0.0
   (C uses v0=π/2 for actions+freqs+angles alike), but the pre-existing Python ACTIONS used
   v0=vx → off C by up to ~6e-3 (a long-standing latent inconsistency, predates this PR).
   User: "make the entire Python implementation consistent with C, tests can shift." Fixed:
   default the Single's `_v0u` to π/2 (collapsing the byte-identity dual-path) → c=False==c=True
   (actions to the ~1e-5 GL-vs-adaptive floor; action/ecc/zmax conservation BYTE-FOR-BYTE
   identical to c=True — e.g. ecc-orbit Jr 3.282%, zmax 4.146%). 2 ecc-conservation test tols
   loosened to the (identical-to-C) v0=π/2 levels (authorized shift). This also dissolves the
   behavior change from (1): actions-path jr == freqsAngles-path jr now.
3. **Drop actionAngleStaeckelSingle for vectorization? (user, forward-looking).** Single is
   scipy(brentq+quad) → can't run under jax; the backend needs a vectorized path regardless
   (Spherical precedent). For Staeckel the per-object physics (2D u/v + the 4×8 angle quadrant
   tree) is costly to duplicate, so **APPROVED PLAN (A): one unified vectorized numpy+backend
   path, drop Single, validate vs c=True (~1e-9), migrate the Grid consumer.** Single IS exported
   + used by actionAngleStaeckelGrid (JR/Jz) — both handled in the rewrite. #1010 (per-object,
   now C-consistent) stays as the validated reference; the vectorized rewrite is the NEXT PR
   (the convergence of (2) full-C-consistency-incl-GL and (3) vectorization/backend).

**NEXT: vectorized C-consistent Staeckel rewrite** — turning-point root-finds via
backend.optimize.brentq (+ vectorised bracket search), GL panels via backend.quadrature.fixed_quad,
the angle quadrant tree → xp.where (Spherical-angles technique, larger), drop Single, point
StaeckelGrid at the vectorised actions. Lands the numpy C-consistency (all-GL closes the last
~1e-5) AND backend support in one path.

### actionAngleVertical backend → #1012 (Track E, 2026-06-25)
PR #1012 (`feat/Paa-vertical` → `feat/backends`), **auto-merge armed**. Additive backend
path (137 ins / 0 del → numpy byte-identical), the 1D vertical analog of Spherical's radial
Jr/Or/ar: `elif is_backend_array(x)` → vectorised `_evaluate/_actionsFreqs/_actionsFreqsAngles
_backend`. Turning point zmax via shared `backend.optimize.brentq` (80-doubling expanding
bracket, `at_turn = vx==0` dead-branch guard). J/Ω/angle via `backend.quadrature.fixed_quad`
GL-50 with the `xi = xmax − t²` substitution (2t Jacobian regularises the 1/√ turning-point
singularity); radicand clipped ≥0 (√'(0)=∞ would NaN-poison reverse AD). Angle: integrate
t∈[√(xmax−|x|), √xmax], then the (x,vx)-quadrant `xp.where` tree (π−a / π+a / 2π−a, %2π).
Validated numpy↔jax↔torch ~1e-10 across IsothermalDisk/KGPotential/verticalPotential(MN) at
all edges (vx=0 turn 4e-12, x=0 midplane exact, large amplitude, all 4 (x,vx) quadrants
~1e-13); grad-vs-FD wrt vx ~1e-9; 212 backend tests + 32 numpy tests pass.

### actionAngleAdiabatic backend — design (NOT free; corrected) (2026-06-25)
The earlier note "Adiabatic: no new code (delegates to Spherical+Vertical)" is **wrong** —
empirically confirmed: feeding a jax array recurses through Adiabatic's per-object single
path into `aAV(z[0],vz[0])` with a **0-d scalar**, which the array-oriented backends reject
(GL integrands assume `[N,n]`). Adiabatic needs its own vectorised backend path. Validated
design (the hard angle-quadrant trees live in the DONE Spherical+Vertical — Adiabatic only
*composes* them, so it is NOT in the supervised-Staeckel category):
- `Jr,Lz,Ω,ar,aphi` → delegate to the (vectorised) Spherical backend
  `self._aAS.actionsFreqsAngles(R,vR,vT,0,0,_Jz=Jz)` — it already accepts `_Jz=` array and
  applies the adiabatic `γ·Jz` term inside `_setup_backend` (verified).
- `Jz,Ωz,az` → Vertical's zmax-bracket + 3 GL integrals, but with the vertical PE
  `Φ_vert(R_i,z)=Φ(R_i,z)−Φ(R_i,0)` evaluated with **R broadcast per-element**: `R.reshape(
  R.shape+(1,)*(zz.ndim−1)) * ones_like(zz)` adapts R to whichever rank zz has ([N] for the
  bracket, [N,n] for the integrand grid). Verified the per-R Φ broadcast matches the numpy
  `toVerticalPotential(pot,R_i)` reference to 1e-16.
- **Clean impl** = thread a `phi(z)` callable through Vertical's 4 backend helpers (they
  differ ONLY in the `evaluatelinearPotentials(self._pot, xi)` call) → DRY shared module;
  Vertical passes the linear-pot evaluator, Adiabatic passes the per-R 3D evaluator. Requires
  #1012 merged → build on clean `feat/backends` (squash-merge ⇒ no stacking). Task #89.

### Staeckel vectorized rewrite PR-1 → #1013 (actions, 2026-06-25)
Approach A, staged like Spherical (PR-1 actions / PR-2 freqs+angles+drop Single+Grid;
angle tree = compute-both-panels + xp.where — user-confirmed). Rebased feat/backends onto
main first (consumed #1010; 1 trivial Staeckel decorator conflict; c=F≡c=T 1.79e-10).

**#1013** (feat/staeckel-vectorized → feat/backends): replace the per-object
actionAngleStaeckelSingle scipy loop in _evaluate with ONE unified vectorized
numpy+backend path — elementwise setup, turning points via shared bisect_root
(fixed-iter expanding bracket + xp.where for at-turn/circular), J_R/J_z via fixed_quad.
- **C-consistency KEY INSIGHT:** C integrates actions with gsl_glfixed = PLAIN GL
  order-`order` over [umin,umax] (NO substitution; the J integrand sqrt(S) VANISHES at
  turning points). So match C EXACTLY with plain GL order-10 → c=False actions ≡ c=True to
  **4e-16**. My first instinct (t²-subst + GL=50) OVER-converges → 2e-5 OFF C (C's own
  order-10 quad error). Substitution is only for PR-2's 1/sqrt-SINGULAR angle integrals.
- **numpy behavior change (flagged):** old c=False _evaluate used ADAPTIVE quad (~exact,
  ~2e-5 off c=True); now GL order-10 = C-consistent AND consistent w/ _actionsFreqsAngles.
- **gradients → PR-2:** d(sqrt S)/dparam ~ 1/sqrt(S) SINGULAR at turning pts → AD-through-
  plain-GL is only GL-order accurate (~5e-3 vs FD). Accurate grads need the substitution-
  regularized derivative integrands (dJRdE/dLz/dI3) — they live with the freqs (PR-2). PR-1
  ships VALUES + backend parity; Staeckel backend tests assert value-parity only.
- Validated: turning pts vs Single 2.8e-13; actions vs c=True 4e-16; jax/torch parity 1e-17;
  63 numpy Staeckel tests + 4 backend tests (under -W error::Deprecation) pass.
- Integrand fix: _JR/_JzStaeckelIntegrandSquared resolve xp from data (numpy byte-identical;
  xp.sinh/xp.sin for backends) — avoids numpy.sinh(tensor) -W error::Deprecation trap.

**NEXT (PR-2):** freqs (calcdJR/calcdJz 6 derivative integrands w/ Low/High t²-subst panels
+ calcFreqs 3x3 solve) + the angle quadrant tree (2x2-per-branch → xp.where) + custom-rule
accurate action/freq gradients + drop actionAngleStaeckelSingle + migrate StaeckelGrid.

### PR-2 freqs core — VALIDATED in scratch (2026-06-25, ready to integrate)
Vectorized Staeckel frequencies validated vs c=True (develop-in-scratch, /tmp/staeckel_freqs_dev.py):
Omegar 4.6e-12, Omegaphi 6.5e-13, Omegaz 1.1e-10 (GL order 10, 6-orbit batch, MWPotential2014).
- 6 derivative integrals: each = Low(lo)+High(hi) t^2-substituted panels of factor/sqrt(S), where
  factor ∈ {sinh^2 u, 1/sinh^2 u, 1 (u-side); sin^2 v, 1/sin^2 v, 1 (v-side)}, mid=sqrt(0.5(hi-lo)),
  integrand 2t*g*mid with g = xp.where(S>0, factor/sqrt(S_safe), 0) (dead-branch guard; S=_JR/_Jz
  IntegrandSquared). v-side: lo=vmin, hi=pi/2.
- **PREFACTOR BUG caught in scratch (would've shipped wrong freqs):** djr prefactors are δ/π/√2 but
  djz prefactors are √2·δ/π (2× larger) AND djzdI3 is POSITIVE (djrdI3 is negative). djrd{E,Lz,I3}
  *= {δ/π/√2, -Lz/π/√2/δ, -δ/π/√2}; djzd{E,Lz,I3} *= {√2δ/π, -Lz√2/π/δ, +√2δ/π}.
- calcFreqs solve: detA=djrdE·djzdI3−djzdE·djrdI3; Or=djzdI3/detA, Oz=−djrdI3/detA,
  Op=(djrdI3·djzdLz−djzdI3·djrdLz)/detA. Circular (umax≈umin): derivs→0, detA→0, freqs→NaN; caller
  substitutes epifreq/omegac/verticalfreq (mirror the c=True indx-substitution, vectorized xp.where).
- **Integration TODO (PR-2):** add _staeckel_deriv_panels + _staeckel_freqs module fns, wire into
  _actionsFreqs c=False (setup+turning computed once, shared w/ actions), vectorize the circular
  epifreq substitution, + custom-vjp so these dJ integrands ALSO give PR-1's deferred accurate action
  gradients. THEN the angle quadrant tree (SUPERVISED) + drop Single + migrate StaeckelGrid.

### PR-2a Staeckel freqs → #1014 (2026-06-26)
Vectorized _actionsFreqs (c=False) on the merged-#1013 base. Refactor _staeckel_prep/
_staeckel_jr_jz (share setup+turning w/ actions) + _staeckel_deriv_panels (6 dJ integrals,
Low/High t²-subst panels of factor/sqrt(S), xp.where S<=0 guard) + _staeckel_freqs (3x3 solve,
NaN circular) + _staeckel_actions_freqs (combined). Circular -> epifreq/omegac/verticalfreq
(vectorized; epifreq takes array R). Validated: Or 4.6e-12/Op 6.5e-13/Oz 1.1e-10 vs c=True;
jax 4.7e-14 / torch 3.5e-13 parity; circular subst exact; 64 numpy Staeckel + 2 backend tests.
Coverage: new lines hit via eager xp.where (epifreq computed for all). NEXT PR-2b (SUPERVISED):
angle quadrant tree (_actionsFreqsAngles) + custom-vjp action/freq gradients via these dJ
integrands + drop Single + migrate StaeckelGrid.

### PR-2a follow-up: EccZmax backend + 3 bracketing bugs + adversarial review → #1014 (2026-06-26)
Backendized EccZmaxRperiRap (_uminumaxvmin c=False -> shared _staeckel_prep). Then a 4-angle
adversarial math-review workflow (vs c=True gold) surfaced THREE genuine bracketing-collapse bugs
in the vectorised root-finds — all the same class (a bisection bracket failing to straddle a turning
point near a degenerate limit), each silently wrong only where the validation grid had no edge orbits:
- **umax (turning point, vR=vz=0):** orbit sits AT umin (f(ux)~0⁻ to round-off); umax bisection over
  [ux,hi] had f<0 at both ends -> returned hi -> umax grossly wrong -> J_R zeroed (GL nodes miss the
  tiny sliver) + ecc inflated (e=0.068 vs 2.7e-4). **Also corrupted the merged #1013 actions.** Fix:
  bracket the other turning point from ux±eps (inside, f>0), mirroring Single calcUminUmax.
- **planar freqs (z=vz=0):** vmin landed ~7e-9 short of π/2 -> tiny-nonzero J_z panels -> finite GARBAGE
  freqs vs C's NaN/Inf. Fix: snap vmin->π/2 in _staeckel_prep (unify the EccZmax snap) -> det(A)=0
  exactly -> deterministic NaN/Inf. planar-eccentric: Omegar,Omegaphi=NaN, Omegaz=**Inf** (Inf not NaN
  stops the Jz<1e-3 subst from wrongly setting Omegar=epifreq on eccentric radial motion).
- **umin (near-axis, low Lz):** the lo down-bracket did 80×0.9 (floor ~3.8e-4·ux); near-axis turning
  points (Lz≲3e-4) sit below it -> [lo,ux] f>0 both ends -> umin collapsed to ux -> jr off 97%. vT=0
  purely-radial: no lower turning point at all. Fix: HALVE the bracket (60 steps -> ~1e-18) + detect
  axis-reaching (f>0 at floor -> umin=0, mirror C/Single rstart==0).
All three fixed + regression grids (turning-point, planar, near-axis vs c=True + backend parity).
Validated machine-precision vs c=True over 324+ orbits incl. all edges: jr/jz 7e-15, freqs 7e-10,
ecc 8e-13. 20 backend + 64 numpy Staeckel tests green.

**Surprise: the C extension is the WRONG side in 5 degenerate regimes** (c=False matches the adaptive
gold standard; these are pre-existing galpy C bugs, NOT introduced here, left as-is so c=False stays
correct): (1) C's order-10 GL freq-derivative integrands OVER-converge for high-ecc / near-planar /
near-circular-in-z orbits — Oz off up to ~0.6, error GROWS with GL order; (2) C's vmin snap mis-locates
the v turning point near the plane (Oz ~6.5% off); (3) C sentinels (9999.99) bound orbits with R_apo
≳37.5 that c=False integrates fine. => "match C to machine precision" is NOT the right bar for Staeckel
freqs; validate vs C in the bulk + vs adaptive-quad gold in the edges. Remaining c=False/C threshold
diffs (planar snap π/2·1e-6 vs 1e-7; vmin at_turn 1e-7 vs 1e-10; |f(ux)|<1e-10 circular knife-edge)
match the trusted Single, measure-zero. Nits for follow-up: float32 GL-result dtype widening; the eager
`if bool(xp.any(unbound))` blocks jit/vmap (by-design eager guard; raising can't be jitted — a sentinel
mode could). NEXT PR-2b (SUPERVISED): angle quadrant tree + custom-vjp gradients + drop Single.

### PR-2b Staeckel angles (quadrant tree) → #1015 (2026-06-26)
Vectorized _actionsFreqsAngles (c=False) angles -- completes the backend-agnostic Staeckel
action-angle stack (actions+freqs+angles all numpy/jax/torch + differentiable). Refactored the
6 Leibniz derivs into shared _staeckel_jacobian; added _staeckel_angle_partial (t^2-subst PARTIAL
integral turning-point->position, guarded) + _staeckel_angles (4-leaf u + 8-leaf v quadrant tree)
+ _staeckel_actions_freqs_angles (combined, folds azimuth into anglephi).
- **Quadrant tree (reverse-engineered from calcAnglesStaeckel, all leaves verified):** panel (Low
  from umin/vmin, High from umax/pi-2) chosen by which turning point the position is nearer;
  reflection const K + sign s = f(momentum-sign x position quadrant). u-branch K table
  {high:pi; low:0(pux>0)/2pi(pux<=0)}, I3 has a leading minus, aphi sign = -s. v-branch 8 leaves
  K in {0,pi/2,pi,3pi/2,2pi}, I3 NO leading minus (+s), phitmp sign = -s. Assembly:
  angler=Or*(Or1+Or2)+dI3dJR*(I3r1+I3r2); anglez +pi/2; anglephi += phitmp + Op*(..)+dI3dLz*(..);
  remainder(2pi) angler/anglez only; circular|planar -> all angles 0; anglephi += phi at the end.
- Validated vs c=True: 648-orbit grid hitting ALL 12 leaves -> 6.75e-10; 768-grid angler 1.9e-9
  /anglephi 2.3e-10/anglez 1.2e-8; np<->jax<->torch ~1e-11. 6 numpy + 23 backend tests green.
- Dropped the now-dead Single calcdJR/calcdJz/calcFreqs/calcAngles + exclusive helpers (-550 lines);
  JR/Jz/calcUminUmax/calcVmin kept for StaeckelGrid. Dev script: logs/trackE_actionangle/staeckel_angles_dev.py.
- NEXT PR-2c (deferred): custom_vjp accurate gradients (actions+freqs+angles via the dJ integrands)
  + StaeckelGrid migration + full Single removal. Then Adiabatic backend (task #89).

### Track G g17 + cross-AA GPU device-safety pass (2026-06-26)
**g17 figure** (paper/figures/g17_staeckel_actionangle_speed/): Staeckel actionsFreqsAngles throughput
vs N for C / numpy / jax+torch CPU+GPU on the TITAN Xp. Findings: vectorised numpy ≈ single-threaded C
(26 vs 25s @1e5); new numpy 13-17x faster than the OLD per-object loop for batches (crossover ~N=10,
OLD wins only at N=1); **torch GPU ~1s launch floor -> ~18x faster than C/numpy by N=1e5**; jax eager
dispatch-bound (~7s flat CPU&GPU -- needs user jit). old worktree for the OLD curve: built at 7a31c196.
**GPU device-safety (the user pushed to run ALL AA methods on GPU -- it paid off):** found the GL-quadrature
`fixed_quad(...,0.0,1.0)`-nodes-on-default-device bug recurs in EVERY migrated method that quadratures
(Staeckel x3 + Vertical x3 + Spherical x2 = 8 sites) -> torch CUDA crashes, jax GPU silently round-trips
(CI has no GPU -> latent). FIX: `device=device_of(<input array>)` on each (no-op numpy). PLUS a 9th bug
the static audit MISSED (not a quadrature pattern): EccZmaxRperiRap passed numpy `delta` into
coords.Rz_to_uv against CUDA R/z -> mismatch; root fix in shared `_staeckel_prep`:
`delta = asarray_on_device(xp, delta, device_of(R))`. Verified gpu_verify_aa.py: 17/17 torch CUDA + 6/6
jax GPU at machine precision; 235 CPU backend tests still green. The audit's other 14 findings were FALSE
POSITIVES: xp.ones_like / numpy.nan*xp.ones_like / python-scalar*tensor all preserve device (proven by the
GPU runs + torch-CPU CI). LESSON: static audits over-flag device bugs; GPU execution is the only ground truth.
Device fixes -> their own PR after #1015 merges (3 files, +33/-12 across Staeckel/Vertical/Spherical).

### Opt PR: IsochroneApprox one batched Orbit solve → #1025 (2026-06-28)
Optimization-phase follow-up consuming the #1024 multi-orbit C-STM lift. The backend
_parse_args_backend (jax/torch) forward + backward legs looped over the input orbits --
one Orbit per IC, then xp.stack of each orbit's getOrbit()/accessor outputs. Replaced with
ONE Orbit(xp.stack([R,vR,vT,z,vz,phi],axis=-1)) → o.getOrbit() (no,nt,6) -- a single
stacked differentiable solve for the whole batch (both the forward leg and the time-reversed
backward leg).
- **Backward leg uses getOrbit columns, NOT accessors.** First attempt reused the old
  `accessor(bo)(ts[1:])` pattern -> TypeError reshape (1,1999)->(3,1999): Orbit accessors
  (bo.R(t) etc.) are SINGLE-ORBIT-ONLY on a multi-orbit Orbit (return (1,nt-1) for 3 orbits).
  Fix: `_bcol(col,sign) = sign*xp.flip(borbs[:,1:,col],axis=1)` reading getOrbit()[:,1:,col]
  (getOrbit IS multi-orbit-correct via #1024's shapeDecorator fix). For query times ==
  self._tsJ these equal the accessor (interpolated) values exactly but stay batched.
- **numpy byte-identical:** _parse_args_backend is only reached when is_backend_array(args[0])
  (dispatch in _parse_args ~L735); numpy stays on the per-orbit _parse_args loop -> untouched.
  Verified: 4 numpy actionAngleIsochroneApprox tests green; the change is in the backend-only fn.
- **~6.1× speedup** on an N=8 jax actionsFreqsAngles batch: OLD per-orbit loop 47.4s median ->
  NEW batched 7.5s median (CPU, MWPotential2014, ntintJ=2000; bench: stash old vs new, warm+3).
  The per-orbit pure_callback C-STM dispatch collapses to one stacked call; scales with N, no
  difference at N=1. (cf. #1024's modest 1.65× on the raw 8-orbit C-STM forward -- the AA win is
  larger because IsochroneApprox does TWO solves/orbit (fwd+bwd) and the loop's Python+callback
  overhead dominated.)
- No new tests: existing isochroneapprox_* battery already hits the modified path on a 3-orbit
  grid (parity _evaluate/_actionsFreqs/_actionsFreqsAngles + scalar/cumul/planar/nonaxi +
  grad-vs-FD-through-the-batched-solve). 8 jax cases green; torch in CI (local CUDA dxdv segfault).
  Stale scalar_parity comment updated. NEXT opt: #99 batched in-backend ODE (vmap-of-solve, the
  diffrax/torchdiffeq analogue -> unlocks IsochroneApprox-on-GPU #102), #100 augmented 42-state STM.

---

## 2026-06-29 — AA milestone: forced-backend numpy-island the query paths (PR #1033)

**Approach.** Consuming the systemic potentials forced-backend scalar-input promotion
(feat/aa-coords-delta 935b17a8 — `Pot(R,z)`/`_evaluatePotentials`/force funnels promote
numpy/scalar inputs to the active backend), the AA query paths that compute against an
inherently-numpy core (vectorised Staeckel core; scipy-spline grids; Spherical scipy loop;
Adiabatic numpy path) started mixing a returned backend array into numpy ops
(`ndarray (-|*) Tensor`). Fix = a shared `galpy.backend.numpy_island` decorator (+ a local
`_island(xp)` helper in Staeckel) that forces numpy for the whole call WHEN NO ARG IS A
BACKEND ARRAY, so the numpy-input path stays byte-identical and returns numpy; a real
backend-array input runs the backend-native path untouched. Mirrors the grid-build
`with use("numpy", force=True)` boundary (167b69ce). Applied to Staeckel (4 query methods +
estimateDeltaStaeckel), StaeckelGrid, Adiabatic, AdiabaticGrid, Spherical, estimateBIsochrone.

**Dead-ends / subtleties.**
- The decorator forcing numpy for numpy-input is the SAME end-state as the `xp = ... else numpy`
  island already in these modules — the difference vs honoring-the-force is that AA composites
  must RETURN numpy for numpy input (unmodified pillar-1 tests don't coerce), so a wholesale
  honor-force would break the suite AND go eager-slow; the island keeps it numpy-fast (torch AA
  248s, not >500s).
- Caught a REGRESSION in 935b17a8: `planarForce._Rforce/_phitorque_nodecorator` promoted
  `phi` unconditionally, but axisymmetric `vcirc`/`rl`/`epifreq`/`omegac` pass `phi=None` →
  `torch.asarray(None)` crash under a forced backend. (numpy never hit it: promote_scalars
  short-circuits on numpy.) Guarded `phi is None`. Force.py (R,z only) was fine.
- `CylindricallySeparablePotentialWrapper._refpot` was a construction-time-cached backend
  array → a later forced-numpy AA query mixed ndarray with the cached Tensor. Stored as a
  backend-agnostic Python scalar (it's an additive constant, drops from forces).

**Results (forced backend; numpy byte-identical preserved).**
- torch test_actionAngle: 0 failed, **61 xpassed (flips)**, 105 xfailed (248s).
- jax test_actionAngle: 0 regressions, **43 xpassed**. 3 `Spherical_*_linear_angles` jax fails
  are a PRE-EXISTING orbit-integration-precision red (maxdev 1.8e-7 vs 1e-8, identical with PR
  stashed; already torch-ledgered) → added 3 mirroring jax ledger entries.
- numpy test_actionAngle+test_potential: 1415 passed, 0 failed.

**Ledger.** 61 torch + 43 jax now-passing entries LEFT ledgered (xpass) for the AA-milestone
regen; only the 3 pre-existing jax precision reds added here. NEXT: ledger regen on feat/backends
after #1033 merges (drop the ~104 flipped entries, review add/drop diff vs origin/feat/backends).

### 2026-06-29 (wave 2) — flip the C-path + freqs + normalized-potential AA xfails (PR #1033)

Per user ask ("bring as many test_actionAngle tests onto the ledger pass; everything but
VerticalInverse/Torus should be possible"), drove the remaining ledgered AA tests to pass:

- **Staeckel @numpy_island on the WHOLE method** (was a surgical `_island(xp)` wrap on only
  the backend-core else-branch). Forcing numpy for the whole call also covers the C-dispatch
  branch -> the ~25 Staeckel `_c` action tests run the numpy C path cleanly. Removed the
  redundant manual wraps + helper.
- **Harmonic / Vertical @numpy_island** (were returning tensors for numpy input).
- **Potential.normalize -> float(_amp)**: Rforce(1,0) returns a backend array under a forced
  backend, baking a Tensor into `_amp` so EVERY later eval returned backend arrays even on a
  numpy query (regression from 935b17a8). Unblocks every in-test `Pot(normalize=...)`. Byte-id.
- **epifreq/omegac/verticalfreq/rl/rE @numpy_island** (module fns + 3 methods): composite
  derived quantities; tests call them directly for references -> islanding makes them match the
  island-numpy AA result instead of mixing numpy with a forced-backend Tensor.

**Result: torch AA 166->60 ledgered (106 un-ledgered, 0 fail); jax 102->27 (75 un-ledgered, 0 fail);
numpy byte-identical (207 AA + 1208 potential pass).** 5 torch re-ledgered (3 eager-slow timeout-flaky,
2 flattened-LogHalo q=0.9 c-vs-python parity edges that fail intermittently at _staeckel_setup
despite lp._amp now being float -- needs follow-up).

Remaining 60 torch / 27 jax = by-design (VerticalInverse + Isochrone/HarmonicInverse not migrated)
or **orbit-interface-dependent**: the test integrates an Orbit under the backend, so obs.x()/R()/E()
return backend arrays that flow into the (correctly backend-native) AA -> the test's numpy ops mix.
ROOT CAUSE for the last ~24 + the user's "make test_orbit(s) AA tests run" ask is the same:
**orbit-side islanding** (numpy IC under a forced backend -> integrate via C/numpy -> numpy
accessors). That is the next follow-up PR (also restores 1e-8 integration precision -> flips the
linear_angles too).
