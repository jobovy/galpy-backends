# Overnight deep-review report — open PRs (2026-06-05)

Seven adversarial deep reviews (one per open PR / group), each by an independent agent
instructed to verify against the real code and treat the **numpy byte-identical** rule
as cardinal. Several agents built both base and PR branches and compared raw IEEE-754
bytes. This found real cardinal-rule violations that green CI missed (CI parity tests
use `rtol=1e-12`, ~4 orders too loose to catch the ~1e-16 drifts below).

## Scorecard

| PR | family | verdict | numpy byte-identical |
|----|--------|---------|----------------------|
| #893 | P2.1 spherical | minor issues + 1 to flag | **violated** at one measure-zero point |
| #894 | P2.2 analytic disk | **clean (LGTM)** | verified bit-identical (40 combos) |
| #895 | P2.3 ellipsoidal | **request changes** | **violated** (2 root causes, ~all methods) |
| #896 | P2.4 halo/bar/non-axi | **request changes** | **violated** (DehnenBar, CosmphiDisk/Lopsided) |
| #897 | P2.7 dissipative | clean (test scaffold) | preserved (no source change) |
| #903 | param-grad enabler | **approve** | verified |
| #904 | spherical variational | **approve** | preserved (additive C) |
| #905 | Kuzmin variational | **approve** | preserved (additive C) |
| #906 | in-backend integrator | **approve** | n/a (additive) |

## The recurring root cause (and the model to follow)

The byte-identical violations are all in the **complex potentials**, and all from the
migration **reassociating or re-deriving arithmetic** while swapping `numpy.`→`xp.`:
- `a += b + c` → `a = a + b + c` (changes associativity → 1-2 ULP) — #895.
- `numpy.dot(M, v)` → explicit `M00*x+M01*y+M02*z` (BLAS/pairwise vs left-to-right) — #895.
- substituting `r2` where the original used `sqrt(r2)**2`, and `R**2/r2` for `1/(1+z**2/R**2)` — #896 DehnenBar.
- hoisting a `cos/sin` factor to a different position relative to the radial factor — #896 CosmphiDisk.
- a safe-`xp.where` changing a genuine singular value (`inf` → finite) — #893 `_surfdens`.

The **clean** migrations (#894 all; #896 **SpiralArms**, despite the heaviest refactor;
#903) introduced `xp.where` **only around the branch selector** and preserved the exact
algebra/ordering/associativity. **SpiralArms is the reference pattern.** The fix for the
others is mechanical: keep the original expression byte-for-byte, wrap only the branch in
`xp.where`, never re-derive an algebraically-equal form.

## Systemic test gap (applies to #893, #895, #896)

The per-family backend tests assert numpy↔jax/torch **parity** at `rtol=1e-12` and
grad-vs-FD at `rtol=1e-5` — both far looser than "byte-identical", so they pass despite
the ~1e-16 numpy-path drift. **Recommendation:** add a numpy-new-vs-frozen-baseline
`assert_array_equal` regression for these families (the existing `test_potential.py`
analytic/FD checks also won't catch ULP drift). This is the check that would have caught
all three findings.

---

## Per-PR findings

### #893 — P2.1 spherical  (minor issues; 1 to flag)
- **[major, flag] `SphericalShellPotential._surfdens` (~L88-100)** — numpy value change at the
  exact shell edge `R==a`: original returned `inf` (true surface-density singularity,
  `h=sqrt(a²−R²)=0`); the `safe_h=where(h==0,1,h)` rewrite returns a finite `1/(2π a)`.
  Measure-zero point, and `inf` is itself degenerate — but it IS a numpy output change.
  *Decision:* restore `inf` at `h==0` to be byte-identical, or consciously accept.
- **[minor] scalar `float`→0-d `ndarray` return** for the `xp.where`-rewritten methods
  (HomogeneousSphere all; SphericalShell all; PowerSpherical `_evaluate` for `alpha>2`).
  Value-identical; publicly collapses to `numpy.float64` via `*_amp` (still passes
  `isinstance(x,float)` since float64 subclasses float). Flagged per the cardinal rule.
- **[minor] test gap:** the safe-`where` guards are never tested at the singular points
  they protect (`r==0`, `r==a`); `_surfdens` not tested at all. Guards verified correct by
  hand today, but a regression reintroducing `1/r` in a dead branch would pass CI.
- Physics, isinstance/shape, device/dtype anchoring, leftover `numpy.`: all clean.

### #894 — P2.2 analytic disk  (CLEAN / LGTM)
- Empirically **bit-identical** across 40 method×potential combos (incl. scalar fast-paths
  and ±z); MN `isinstance(R,float)` fast-paths correctly gated on `xp is numpy`. No
  `xp.where` used (no nan-poison surface). `numpy.fabs`→`xp.abs` necessary (array-api-compat
  torch lacks `fabs`) and byte-identical for the float inputs involved.
- Awareness note (pre-existing, not this PR): KuzminDisk `_zforce` returns 0 at exactly z=0
  while autodiff gives the right-derivative — the inherent `|z|` kink; relevant to any future
  work differentiating Kuzmin through the disk plane.

### #895 — P2.3 ellipsoidal  (REQUEST CHANGES — 2 byte-identical violations, both proven)
- **[high] `EllipsoidalPotential._2ndDerivInt_all` (~L551-556)** — `xx += dd*xi + w*...` rewritten
  to `xx = xx + dd*xi + w*...` changes associativity (`a+(b+c)` vs `(a+b)+c`) → 1-2 ULP drift in
  the **diagonal** 2nd-deriv accumulators (`xx,yy,zz`) → corrupts `_R2deriv/_z2deriv/_phi2deriv/
  _Rphideriv` for all 7 ellipsoidal potentials (aligned). *Fix:* restore `+=` for those lines.
- **[high] `_rotate_to_aligned` / `_rotate_force_back` (~L137-156)** — `numpy.dot(rot, vec)` →
  explicit component arithmetic rounds differently → 1 ULP in every rotated (`zvec`/`pa`) case
  (`_evaluate/_Rforce/_zforce/_phitorque/_dens`). *Fix:* keep `numpy.dot` on the numpy branch
  (`if xp is numpy: numpy.dot(...) else: explicit`).
- **[medium] test gap:** `rtol=1e-12` too loose; **every test case is aligned** — the rotated
  path (and the `numpy.dot` regression) has zero coverage. Agent separately verified the rotated
  jax/torch path is numerically correct + differentiable, so it's a coverage gap, not a 2nd bug.
- Cache (`xp is numpy` gate), `xp.where` (only `isinf(R)`, safe), physics, leftover `numpy.`: clean.
- Both fixes were **empirically confirmed** to restore bit-identity over 11,133 points.

### #896 — P2.4 halo/bar/non-axi  (REQUEST CHANGES — DehnenBar + CosmphiDisk byte-identical drift)
- **[medium] `DehnenBarPotential` — every migrated method**, ~1e-16-1e-19 drift. Causes:
  (1) substituting precomputed `r2` where the original used `r**2` with `r=sqrt(r2)`
  (`sqrt(r2)**2 != r2` in the last ULP); (2) `_evaluate` outer branch rewritten
  `1/(1+z²/R²)`→`R²/r2safe`; (3) trig factor hoisted ahead of the radial factor.
  *Fix:* keep `r=sqrt(r2)` and `r**2` in the derivatives, keep the `1/(1+z²/R²)` form, preserve
  multiply order; introduce `xp.where` only at the branch selector.
- **[medium] `CosmphiDiskPotential` (+ `LopsidedDisk` subclass) — `_evaluate/_phitorque/_R2deriv/
  _phi2deriv`**, ~1e-17: `cos/sin` hoisted before the `where`, reordering the product vs the
  original. *Fix:* keep trig in the original slot relative to the radial factor.
- **[low] DehnenBar NaN reverse-mode gradient at the exact origin R=z=0** — numerator
  `(r/rb)**3` with `r=sqrt(r2)` gives `0/0`; write as `r2**1.5/rb**3` (verified grad→0).
  Origin is a hard singularity / never an integration node, but undercuts "safe under tracing".
- **[low] DehnenBar `_evaluate` array path at R=0,z≠0,r≤rb** changes from `((r/rb)³−2)` (old
  `numpy.divide where=R!=0` fill 1.0 — itself a latent bug, inconsistent with old scalar=0) to 0.
  Arguably a correctness *improvement*, but not byte-identical to the old array path.
- **[medium] test gap:** parity `rtol=1e-12` too loose; no numpy-old-vs-new regression; no
  gradient test at the singular origin (would have caught the NaN).
- **CLEAN** (byte-identical): EllipticalDisk, HenonHeiles, Steady/TransientLogSpiral,
  LogarithmicHalo, SoftenedNeedleBar, **SpiralArms (all 11 methods, the model to follow)**.

### #897 — P2.7 dissipative  (CLEAN — test scaffold only)
- Adds `tests/test_backend_dissipative.py` only; **zero source changes** → numpy byte-identical
  by construction. Pins reference values, the `v=[vR,vT,vz]` convention, the below-`minr` branch;
  `strict=True` xfail documents that `jax.grad` on the dissipative path currently raises (a
  forcing function for the future Pspecial migration). 6 passed, 2 xfailed.
- **[nit]** dead code (`AD_BACKENDS`/`_asarray`/`_tonumpy` unused; torch probed but unused).
- **[info, pre-existing]** `ChandrasekharDynamicalFrictionForce._calc_force` never assigns
  `self._force_hash = new_hash` → the force cache is a latent no-op. Predates this PR; worth a
  separate fix since the test narrative leans on the cache as the tracing hazard.

### #903 — param-grad enabler  (APPROVE / clean)
- numpy byte-identical **confirmed**: `is_backend_array` returns False for number/None/ndarray/
  generic/Quantity (early-returns before any jax/torch isinstance), so the new guard branch only
  fires for genuine backend arrays; all other inputs hit the identical pre-existing path. No
  import cycle; the lazy import is a cached dict lookup. Pass-through-unscaled semantics are
  consistent (non-Quantities were never ro/vo-scaled). Tightened tolerances justified + non-flaky.
- **[minor] coverage gap:** only Plummer/Isochrone, only amp + one scale length, scalar coords.

### #904 / #905 — variational (APPROVE / clean, independently verified)
- Built + ran both suites (52 pass #904, 37 pass #905). Each spherical potential's C transform
  uses its OWN planar functions (no cross-contamination); KKS chain-rule matches
  `coords.Rz_to_lambdanu_jac/hess` and the V(l,n) derivatives; KuzminDisk off-plane test genuinely
  exercises the Hessian (min|z|=2.0, z2/Rz substantially nonzero). numpy byte-identical (additive
  C consumed only by the variational RHS). Isochrone→PseudoIsothermal gate swap valid.
- **[nit] #905 `tests/conftest.py` ~L33-39:** the KuzminDisk explanatory comment sits above the
  *KuzminKutuzov* registry tuple, reading as if it annotates it. Move/reword.

### #906 — in-backend integrator (APPROVE / clean)
- EOM matches galpy `_EOM` line-by-line (diff 0.0); coordinate maps round-trip + differentiable;
  dispatch cycle-free; both documented findings (diffrax custom_vjp → no jacfwd; torchdiffeq
  dopri8 backward inaccurate → dopri5) real and correctly handled.
- **[minor → fixed]** `_eom_rhs` omitted `v=[vR,R·Ω,vz]` that `_EOM` passes (harmless for
  non-dissipative, fail-loud for dissipative). Added `v=` for exact parity + future dissipative.
- **[minor] gaps:** no torch-side STM/param-grad test (covered indirectly by torch-vs-jax check);
  STM only for Plummer.

---

## Prioritized action list (for the morning)

**Byte-identical fixes (cardinal rule) — mechanical, agents gave exact fixes:**
1. **#895** — restore `+=` in `_2ndDerivInt_all`; keep `numpy.dot` on the numpy branch. (2 spots, verified.)
2. **#896** — DehnenBar: keep `r=sqrt(r2)`/`r**2`, the `1/(1+z²/R²)` form, multiply order; CosmphiDisk:
   keep trig in original slot. (Mechanical; follow the SpiralArms pattern.)
3. **#893** — decide on `_surfdens` `inf` vs finite at `R==a` (restore `inf` for byte-identity, or accept).

**Robustness:**
4. Add a numpy-frozen-baseline `assert_array_equal` regression to the #893/#895/#896 backend tests
   (the gap that let all three through). Add a gradient-at-singularity (finite, not NaN) test.
5. **#896** DehnenBar origin NaN gradient → `r2**1.5/rb**3`.

**Nits / pre-existing (low priority):**
6. #905 misplaced conftest comment; #897 dead test code + (pre-existing) Chandrasekhar cache no-op.

**Ready to merge as-is:** #894, #903, #904, #905, #906 (and #893 once the `_surfdens` decision is made).
**Need a (small, mechanical) change first:** #895, #896.
