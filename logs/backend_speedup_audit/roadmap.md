# Backend speed-up roadmap — jit-fast / backend-specific (no internal jit)

Systematic 6-subsystem audit of `feat/backends` (2026-06-26, workflow `wf_d3e67870-c45`,
7 agents). Policy: **never `jax.jit` inside galpy** — make code jit-*compatible* and
jit-*fast* (rolled `lax` loops, `custom_vjp`, fused/vectorized ops); backend-specific
`_jax`/`_torch` kernels OK where they buy a significant speedup; **numpy byte-identical**.

Two dominant themes:
1. **Unrolled Python `for` loops** bake giant subgraphs into the user's jit/grad trace
   (Staeckel actions = 47,528-eqn jaxpr, ~8.6 min compile). Fix: rewrite control flow as
   `lax.while_loop`/`fori_loop`/`scan` (jax-specific; numpy/torch keep the eager loop).
2. **Redundant recompute** — the numpy md5/key caches are deliberately disabled for
   jax/torch, so backend paths recompute forces 3×, Hessians 6×, base orbits 6×.

---

## P0 — now (highest leverage)

1. **Roll the shared `bisect_root` for-loop → `lax.fori_loop` (jax)** + thread a static step count.
   `optimize.py:165` `for _ in range(n)` unrolls ~100 copies of the physics closure. It's the
   ONE root-finder behind **every** backend turning-point solve: `Potential.rl/rE/zmaxgrid/
   rliftedgrid`, Spherical `rperi/rap`, Vertical `xmax`, IsochroneInverse `eta`, ×2 per Staeckel.
   Bonus (verified): `n_bisect_steps` falls back to **100** iters under a tracer (vs ~46–51
   concrete; Staeckel passes maxiter=200) → threading a static `nsteps`/width-hint roughly
   **halves every backend bisection**. *backend-specific, effort M, compile+gradient.*
   **✅ #1026 LANDED:** rolled the loop into `lax.fori_loop` **only under a
   trace** (`isinstance jax.core.Tracer`); eager (CI) + torch + numpy keep the Python loop.
   **KEY FINDING:** a naive unconditional roll *regresses eager ~9×* (137ms vs 16ms; `fori_loop`
   compiles a loop primitive per call) — galpy's CI runs EAGER, so the eager/traced branch is
   mandatory, not optional. Measured (N=64): eager 16ms unchanged; **jit compile 592→152ms (~4×);
   jaxpr 717→41 eqns (~17×)**; traced result bit-identical to the unrolled loop at equal n
   (max-diff 0). DEFERRED follow-up: the static-step-count-through-callers half (use ~50 not 100
   halvings under jit) — composes with P0#2/#93.
2. **Roll the expanding-bracket loops** (Staeckel `60+80+80`, Spherical `2×80`, Vertical `80`)
   → `lax.while_loop`/`fori_loop`. Bodies are already single branch-free `xp.where` updates.
   The concrete 47k-jaxpr instances in the hottest subsystem; stacks with P0#1.
   `actionAngleStaeckel.py:127,133,164`; `Spherical.py:564,569`; `Vertical.py:336`. *jax, M.*
   **✅ LANDED:** all three (`actionAngleStaeckel`/`Spherical`/`Vertical`) now import & use the
   shared `optimize.iterate_bracket` helper (the eager/traced-gated roll from #1026).
3. **Collapse `c_stm_forward`'s 6-per-IC variational integrations → ONE stacked `integrate_dxdv`.**
   `inbackend_stm.py:59-83` re-integrates the identical base orbit 6× (once per basis column).
   The stacked single-call idiom is already proven in-repo (Lyapunov `Orbits.py:2893-2907`,
   `rectIn/rectOut=True`, `numcores`). ~6× forward-cost cut on the headline C-STM gradient path.
   *backend-agnostic, effort M, warm+gradient.*
   **CLOSED/HELD → PR #1023 closed 2026-06-27 (user decision).** Prototyped: stacked the 6N
   calls into ONE `integrate_dxdv` on 6N orbits; bit-identical M (24/24 exact), 117 STM tests
   green; N=20 measured 881ms → 750ms (nc=1) → 480ms (nc=8). **NOT 6×** — `integrate_dxdv` is
   base+1-deviation, so each of the 6 stacked copies *re-integrates the base*; the win is just
   call-overhead (~1.2×) + `numcores` parallelism (~Ncore×). Too marginal standalone given the
   runner-constrained CI → closed and **folding the batching into #100 (augmented 42-state
   integrator, true ~6×) + #101 (multi-orbit C-STM lift)** where it compounds. Verified work
   preserved on branch `feat/cstm-batch`.

## P1 — next

4. **Vectorize the EllipsoidalPotential GL node loop** (`_potInt/_forceInt_all/_2ndDerivInt_all`,
   `EllipsoidalPotential.py:469,510-522,575-598`): `for k in range(len(glx))` (K~20–50) →
   one broadcast `dens(m)`/`psi(m)` over a node axis + one `xp.sum`. Universal inner loop for
   every triaxial potential; collapses K dispatches → 1 and shrinks the graph K-fold. *agnostic, M.*
   **⏳ PR #1130 open 2026-07-16:** backend-only vectorize via a shared `_gl_node_factors` helper,
   gated on `is_backend_array(x)` (numpy keeps the verbatim loop = BYTE-IDENTICAL, 105 repr-exact).
   Measured glorder=20: evaluate jaxpr 294→30, R2deriv 1154→91 (~10–13×); jit compile ~2–3×; parity
   ~1e-14; grad jax==torch |AD-FD|~1e-11; existing test_backend_ellipsoidal (737) pass. NOTE: the
   `dens(m)`/`psi(m)`-broadcast assumption verified across all subclasses. Also confirmed: the
   `_forceInt_all`/`_2ndDerivInt_all` "combined single-pass" evaluators (part of P0#5) already landed.
5. **Combined force / Hessian evaluators** (stop the backend 3×/6× recompute): one
   `_all_cyl_forces` → `(FR,Fz,Fphitorque)` and one combined Hessian from the single 8-tuple.
   numpy shares via md5 caches that are **disabled for jax/torch** → SCF/Multipole/Ellipsoidal
   recompute the whole expansion per component. Amplified by `inbackend_ode._eom_rhs` calling all
   three force evaluators every ODE stage. *agnostic, M.*
6. **Gate the C-STM forward on requires-grad** — a no-gradient backend-IC `integrate` (routed
   through `_integrate_cstm`, `Orbits.py:1748`) silently pays the 6× STM solve. jax: primal calls
   base-orbit-only; torch: skip STM when `not requires_grad`. With P0#3 → forward = one base solve.
7. **DEFERRED (gated on Track F — streamdf not yet backend-migrated).** General principle worth
   keeping: now that #1016 made Staeckel `actionsFreqsAngles` AD-traceable, *any* consumer that
   needs `d(J,Ω,θ)/d(x,v)` should use `jax.jacfwd`/`torch.func.jacfwd` instead of finite
   differences — exact + cheaper. The concrete first application (streamdf `calcaAJac`'s 6-point
   FD Jacobian) waits until streamdf itself is on a backend; do NOT touch streamdf now.
8. **Hoist `_staeckel_jacobian`** — compute the 6 derivative panels once and share between
   `_staeckel_freqs` and `_staeckel_angles` (each recomputes them today). *agnostic, M.*
9. **Native batch axis through the in-backend diffrax/torchdiffeq RHS** (`_eom_rhs/_to_eom/
   _from_eom`) so N orbits integrate in one adaptive solve, avoiding an outer `vmap(diffeqsolve)`
   that re-traces. *agnostic, L, warm-gpu-largeN.*
10. **`assoc_legendre` Bonnet recurrence → `lax.scan` over l** (carry `P_{l-1},P_{l-2}` for all m);
    vectorize the derivative tables. SCF/Multipole hot path. *jax, L.*
11. **gegenbauer L-comprehension + N-recurrence → `lax.scan`**; share `C^a/C^{a+1}/C^{a+2}` per SCF
    point. *jax, M.*

## P2 — later (gradient + warm fusions)
- **`custom_jvp` on the Staeckel/Spherical/Vertical action integrals** using the existing analytical
  Leibniz dJ panels (forward = plain sqrt-GL, byte parity; backward skips AD-through-the-quadrature). *jax, L.*
- Cache GL node/weight tables per `(xp,n,device)` in `_gl01_on` (kills a per-call host→device transfer). *S.*
- Share per-node `S(u)/S(v)` across the 3 Staeckel angle/jacobian factor-integrands (3 node-grids → 1).
- Share node potential evals across the spherical Or/Op/angler/anglez panels.
- Fuse the per-RHS-step force dispatch into one pass over `_potlist`; hoist `isDissipative` out of the loop.
- Thread diffrax potential params as `diffeqsolve` args (not a closed-over object); shrink `max_steps`.
- Contract the STM VJP residual `(N,nt,6,6)` → `(N,6)` host-side (smaller save/transfer).
- Batch `constantbetadf.fE`/`sphericaldf._dMdE` per-energy quadratures into 2D GL tensors.
- Batch `streamgapdf` per-star impulse Orbit integration into one stacked Orbit.
- Batch the umin/umax bisections into one `[2,N]` solve reusing bracket f-values.
- Special-fn table caches + positive-only torch gamma fast path.

## Cross-cutting "shared wins" (one fix, many consumers)
- **Roll `bisect_root` once** → shrinks the graph for *every* backend root-find consumer.
- **Static bisection step count** → ~halves iterations for every backend bisection.
- **GL node/weight `_gl01_on` cache** → removes a per-call host→device node transfer everywhere.
- **Combined force/Hessian evaluators** → kills the 3×/6× recompute across SCF+Multipole+Ellipsoidal at once.
- **`lax.scan`-recurrence idiom** (assoc_legendre) → reuse for gegenbauer + any 3-term recurrence.
- **Stacked single-call `integrate_dxdv` idiom** → fixes both `c_stm_forward` and `streamgapdf` batching.

## Rejected (respect no-internal-jit / low ceiling) — confirms policy
- **No audit proposed an internal `jax.jit`/`torch.compile`** — all items are jit-compatible rewrites. ✓
- numpy-only `ndindex` expansion-assembly loops (vectorizing changes numpy reduction order → not byte-identical).
- `glorder=None` scipy.quad / Ferrers numpy.roots fallbacks (traceability coverage gap, separate concern).
- iv/kn integer-order recurrences (only consumer n=2 → unrolls to ~1 step, negligible).
- `pure_callback` STM host round-trip (structural to a C host call; only shape-bucketing helps).
- get_namespace memo, transformed_quad 2-panel fusion, streamdf eigen/slerp loops (low ceiling).

## Recommended first moves
1. Roll `bisect_root` → `lax.while_loop` + static step count (one change, every consumer benefits).
2. Collapse `c_stm_forward` → single stacked `integrate_dxdv` + gate STM on requires-grad.
3. Vectorize EllipsoidalPotential GL loop + the assoc_legendre/gegenbauer recurrences.

Raw per-subsystem findings: `raw_audit_result.json`.

---

# Addendum — deep potential + orbit audit (NEW levers, 2026-06-26, wf_afb96d85-01d)

**Dominant structural insight:** the force/Hessian layer wires every cylindrical component
(Rforce/zforce/phitorque) and every Hessian entry independently, and the numpy md5/key result
caches are *disabled* under jax/torch — so every shared subexpression (`sqrt(R²+z²)`, `log(1+r/a)`,
`hyp2f1`, `gamma·gammainc`, the Gegenbauer C-ladder, SCF tables, the cyl↔rect transform) is
recomputed **3× (force) → 6× (Hessian) per ODE step**, multiplied straight into the variational/
gradient hot path (`inbackend_ode._eom_rhs:50-63` calls all three force evaluators every step).

## Cross-cutting "shared wins" (one pattern, many consumers)
1. **Combined force + combined-6-Hessian per-point evaluators** — compute the shared spherical/
   radial pieces once, return all 3 forces / all 6 Hessian entries. Pays off for NFW
   (`TwoPowerSphericalPotential.py:967-1023`), TwoPowerSpherical (hyp2f1 6×→2×, :251-300),
   PowerSphericalwCutoff (gammainc 3×→1×), Plummer/MiyamotoNagai, all spherical leaves, SCF/Multipole
   (`SphericalHarmonicPotentialMixin.py:121-161`), RotateAndTilt. **The #1 lever** — generalizes the
   first audit's combined-evaluator item to the whole analytic-potential layer.
2. **Skip structurally-zero components per point** — axisymmetric `phitorque≡0`, planar `zforce≡0`;
   `phasedim`/symmetry is statically known at `integrate_orbit` time → drop 1 of 3 force dispatches
   per step on the common classes AND remove the dead reverse-mode sub-tape. (`inbackend_ode.py:58-62`,
   SCF `dPhi_dphi` azimuthal sum / `Asin` branch.)
3. **Cache per-`(backend,device,dtype)` constant tables** — SCF Acos/Asin, Multipole table dict,
   DoubleExpDisk Ogata nodes (re-materialized twice in `_R2deriv`), RotateAndTilt matrices: a per-call
   `asarray_on_device` host→device transfer with no cache. Memoize on `(id(xp),device,dtype)`; numpy
   (`dev=None`) stays byte-identical pass-through.
4. **Single `_call_internal` feeds all derived accessors** — `L()` calls it 6× via x/y/z/vx/vy/vz;
   SkyCoord 3–4×; off-grid this *also* multiplies the spline rebuild 6×. Compute `thiso` once.

## P0 (highest (impact×reach)/effort)
- Combined NFW/TwoPowerSpherical/PowerSphericalwCutoff evaluators (heavy special once).
- SCF/Multipole combined force (3×→1×) + combined-6 Hessian (6×→1×).
- **Off-grid backend interpolator: 4–6 (up to 36 via `L`) dense O(n³) tridiagonal solves → one
  multi-RHS solve** (`Orbits.py:7050-7060`).
- Gegenbauer C-ladder: build `_C` at N+2 once, derive C/dC/d2C by roll (3×/6× rebuild → 1).
- **3D dxdv rect→cyl: per-(orbit×timestep) `numpy.linalg.solve` → the closed-form analytic inverse
  the planar twin already uses** ⚠ byte-identity flag (below).
- Skip the structurally-zero z-force (planar) / phi-torque (axisymmetric) RHS dispatch.

## P1 / P2
- P1: WrapperPotential `__getattr__` per-call lambda synthesis + `_wrap_pot_func` 11-branch chain →
  module-level map; RotateAndTilt frame-transform recompute (3–9×→1); L()/SkyCoord single-call;
  device-table caching; diffrax `dt0`/`max_steps`/`jump_ts` hygiene (skip initial-step root-solve,
  declare the bar/spiral C¹ kink).
- P2: `potential_physical_input` re-parsing `inspect.signature`/`_check_backend_compatible` per call;
  `_C` carry-alpha-vector recurrence; skip `arctan2` for axisymmetric in the RHS; packed
  lower-triangular Legendre/SCF; jax SaveAt/pure_callback shape-bucketing; per-step `xp.stack`,
  `_parse_pot` memo, hoist `Orbit()` out of the 6-basis loop.

## ⚠ Byte-identity flags (standing rule)
1. **3D dxdv analytic inverse** (`integrateFullOrbit.py:1037-1042`) — mathematically exact but NOT
   bit-identical to `numpy.linalg.solve`; needs review before it touches the numpy path.
2. **Off-grid multi-RHS spline solve** — backend-only path (mode-2, `is_backend_array`), so numpy
   stays byte-identical, but diff the stacked-solve reassociation.

All respect no-internal-jit (combined evaluators, dead-branch elimination, multi-RHS batching,
device-table memoization, solver-arg hygiene — no jit/compile inside galpy).

---

# After the optimizations land: RE-MEASURE all speed figures
Every perf change shifts timings, so the Track G speed figures must be re-measured (machine-stamped
JSON re-generated, byte-stable plots re-rendered) once the optimizations land — **g17** (Staeckel AA
speed, jax+torch), **g2** (force throughput), **g8** (galpy-vs-galax orbit), **g12** (in-backend orbit
GPU), **g15** (IC-deriv speed). Re-run on the GPU box; the plotting code is byte-stable so only the
`data/*.json` change. (User flagged 2026-06-26.)

---
## UPDATE 2026-06-30: #93 + P0#2 DONE → PR #1040 (feat/staeckel-jit-fast)

Track E complete (#1038/#1039 merged). First perf PR knocks out the biggest item.
KEY CORRECTION to this roadmap: #1026 (P0#1, merged) only rolled the brentq-routed
bisections; **Staeckel's umin/umax/vmin call bisect_root DIRECTLY** (not via brentq),
so they were never rolled — that was the real 47k-jaxpr / 134s-compile blow-up.

PR #1040: (1) bisect_root now dispatches to the lax _bisect_root under a jax trace
(so direct callers benefit, zero call-site change); (2) new iterate_bracket(step,x0,n)
rolls the 6 expanding-bracket loops (Staeckel x3, Spherical x2, Vertical x1). Same
#1026 eager/traced split via _under_jax_trace (numpy/torch/jax-eager keep Python loop,
byte-identical). Measured (Staeckel actions, N=8): jaxpr 47,556→2,283 (~21x), jit
compile 134.0s→2.6s (~52x), jit-vs-eager bit-identical 1.6e-16, grad correct.
Regression: numpy 91 / torch 197 passed. → P0#2 and #93 both resolved by #1040.

Also on #1040 (2026-06-30): promoted `_under_jax_trace` → `galpy.backend._namespaces.under_jax_trace`
(shared eager-vs-`lax.fori_loop` predicate, next to `is_backend_array`; DRY'd the duplicate
inline tracer-check in `_jax/optimize._bisect_root`) + added `test_staeckel_jit_grad_rolls_direct_bisection`
(jit/grad THROUGH Staeckel — the only test that traces the DIRECT `bisect_root` caller;
closed the codecov/patch gap on optimize.py:219-222).

---
## UPDATE 2026-06-30: comprehensive AA-method perf audit (workflow wf_71e05065-2e9, 7 agents)

Full read-only audit of ALL action-angle methods (Staeckel, Spherical+Vertical, Adiabatic,
IsochroneApprox, Isochrone+Harmonic, StaeckelGrid+AdiabaticGrid) after #1040, scored by the
value lens (agnostic+byte-identical recompute-elim = HIGH; eager-only = LOW because jit CSE
dedups). **12 findings, ALL byte-identical-safe. Honest ceiling: exactly ONE cross-the-board
win; the rest is numpy/eager-only micro-hygiene.** Adiabatic (non-grid) audited clean.

**HIGH (the anchor):**
- **A1 — Staeckel `_staeckel_jacobian` hoist** (`actionAngleStaeckel.py:515-527`). The 6 Leibniz
  derivative panels (dJr/dJz d{E,Lz,I3}, each a t²-subst singular-integrand GL quadrature) are
  computed TWICE in `actionsFreqsAngles`: `_staeckel_freqs` (L341) + `_staeckel_angles` (L409).
  Hoist to compute once/orbit, thread into both. Docstring L301 already declares the intent.
  Benefits numpy (byte-identical) + eager + shrinks the jit trace graph. Effort S. **LAND FIRST.**

**MEDIUM/LOW (free but not fast — land for cleanliness):**
- A2/A3/M1/M2 Isochrone scalar-`sqrt` hoists (`actionAngleIsochrone.py`): sqrt(L²+4·amp·b),
  sqrt(R²+z²), sqrt(L²−Lz²) each 2×. Byte-identical but jit-noise (CSE folds them).
- A4/M3 grid numpy consolidations (StaeckelGrid `_grid_common_numpy`+ErEz-combo drops the dup
  `potentialStaeckel(u0,π/2)`; AdiabaticGrid shared Jz vertical-setup). numpy-only, zero jit impact
  (backend path already consolidated via `_grid_common_backend`).
- M4/M5/L1 Spherical radicand-share across Or/Op + angler/anglez panels (same GL nodes/E/L).
  EAGER-ONLY (CSE reclaims under jit). Fold L1 into M5.
- M6/M7 IsochroneApprox `tile`→broadcast (numpy path; backend already broadcasts). numpy temp churn.

**Recommended single AA-perf PR (after #1040 lands):** A1 first (verify byte-id vs c=True) →
Isochrone sqrt bundle → grid numpy consolidations → Spherical radicand-share (last) →
IsochroneApprox broadcast. Flag A4/M3 + M6/M7 for numpy byte-diff review. Keep #88 (Spherical
numpy-loop vectorization — RELAXES byte-identity) OUT on its own track.

---
## UPDATE 2026-07-01: AA-perf PR MERGED → #1043 (squash 2794c74a on feat/backends)

CI 154/154 green (codecov cleared after the jac-required patch fix + all sessions
uploaded; one VizieR notebook flake re-run to green). Regressions:
test_actionAngle.py (numpy, affected classes) 166 passed;
test_backend_actionAngle.py (full jax+torch) 300 passed.

**AA-perf roadmap now fully burned down.** Remaining AA item: only #88 (Spherical
numpy-loop vectorization), deliberately on its own track because it RELAXES numpy
byte-identity. Non-AA speedup-audit items (potentials/orbit/df) remain in the
sections above.


Built with a byte-identity gold-reference harness (`/tmp/aa_ref.py`: dumps numpy AA outputs
on a fixed grid; `numpy.array_equal` before/after each edit). 4 commits, all verified:

- **A1 — Staeckel jacobian hoist** (9544765c). `jac=` param on _staeckel_freqs/_staeckel_angles;
  computed once in _staeckel_actions_freqs_angles. numpy byte-identical (15 arrays) + backend
  parity/jit-grad (24 passed). THE real win (all backends, incl. smaller jit trace).
- **Isochrone sqrt hoists** (8dfc2e7c). sqrt(L2+4ab), sqrt(R²+z²), sqrt(L²-Lz²) DRY'd.
  **KEY:** the audit's suggested L306 reuse (1/sqrt(1+4ab/L2) = L/s) is a DIFFERENT float op →
  NOT byte-identical; left untouched. Byte-identical (15) + backend (64 passed).
- **StaeckelGrid ErEz dedup** (dc8ea6cc). potentialStaeckel(u0,π/2) once per orbit (was 2×).
  Byte-identical (10) + grid backend parity (18 passed).
- **Spherical M4/M5** (b19e97f0). `_panel_backend_both` shares the radicand across Or/Op;
  merged `_calc_angles_backend` shares it across angler/anglez. Bit-identical backend output
  (exact op order preserved) → parity+grad (148 passed). numpy untouched (scipy). EAGER-ONLY.

**DROPPED / SKIPPED (with cause — validates the honest ceiling):**
- **M6/M7** (IsochroneApprox tile→broadcast): audit claimed "same order → byte-identical" but
  numpy.sum's pairwise grouping depends on array CONTIGUITY; tile+transpose vs broadcast give
  different layouts → ~1e-13 diff, AMPLIFIED through inv(AᵀA) into the angles. Reverted. The
  numpy-only IsochroneApprox angle path can't take this rewrite byte-identically.
- **_grid_common_numpy extraction (A4 big part) + AdiabaticGrid-Jz (M3)**: pure cross-method
  DRY — dedups CODE not per-call COMPUTE (separate entry points each run their setup once) →
  no speedup for real byte-identity risk on branchy masked-assignment numpy. Skipped.

**Verdict:** the ceiling held EXACTLY as audited — A1 is the one cross-the-board win; Isochrone
+ StaeckelGrid are tiny byte-identical numpy savings; Spherical is eager-only. Everything
past A1 is marginal, and the one item that CHANGED an op (M6/M7) broke byte-identity.

---
## UPDATE 2026-07-16: night perf pass (4-agent re-audit) → PR #1130 + #1131

Roadmap had gone stale — re-verified against feat/backends. **Already landed since
2026-07-01:** #1026 (P0#1 bisect_root), #1040 (P0#2 bracket-loops), #1043 (AA jacobian
hoist A1), #1111 (P1#10 assoc_legendre lax.scan). Two NEW graph-shrink PRs this pass:

- **#1130 — EllipsoidalPotential GL node-loop vectorize** (P1#4). Backend-only vectorize
  of `_potInt`/`_forceInt_all`/`_2ndDerivInt_all` (all nglx nodes on a trailing axis +
  one xp.sum), `is_backend_array`-gated so numpy keeps the verbatim loop (BYTE-IDENTICAL,
  105 repr-exact). jaxpr: evaluate 294→30, R2deriv 1154→91 (~10-13×); jit compile ~2-3×.
- **#1131 — gegenbauer lax.scan under trace** (SCF radial basis; the pre-#1111 analog).
  jax-trace-only scan branch, numpy untouched by construction. SCF jaxpr now N-INDEPENDENT
  (Rforce 224, R2deriv 438) vs growing (N=24: 556/1112). numpy byte-identical (stash).

**Remaining PENDING (4-agent audit, all deferred as marginal/higher-surface for autonomous
work — documented for review):**
- **phitorque-skip for axisymmetric** (`_reference/inbackend_ode.py:_eom_rhs` 3D branch calls
  all 3 force evals every step; `_evaluatephitorques≡0` for axisym). Thread a static `axisym`
  bool from `integrate_orbit` through `_jax`/`_torch/orbit_ode`. Backend-only (numpy raises
  NotImplementedError). BUT jit DCEs the ×0 term → mostly an EAGER + trace-size win. S/M.
- **alpha-vectorize the SCF L-loop** (`SCFPotential._C` backend branch stacks
  `gegenbauer(N,alpha(ll),xi)` over L → still O(L) scan calls). Broaden the #1131 scan to a
  broadcastable alpha vector → one scan. Stacks on #1131 (same files) + touches SCF backend
  branch. The natural #1131 follow-up; higher graph-shrink but stacked-PR + SCF surface.
- **combined spherical force/Hessian evaluators** (`SphericalPotential.py` base funnels the
  whole family through `_rforce`/`_r2deriv`; add numpy-md5-cached `_compute_*` like
  EllipsoidalPotential). NUMPY byte-identical win (5×/3×→1× heavy special) — but the jax/torch
  3×/6× recompute needs a fused consumer / XLA-CSE already dedups under jit → mostly a NUMPY-path
  win, not the backend goal. S/M, low risk.
- **DoubleExpDisk Ogata device-table cache** (de_n=10000 tables re-transferred per call ×6
  methods + a duplicate `_de_j1_xs` in `_R2deriv:411-412`). Per-(id(xp),device) instance cache,
  numpy pass-through. EAGER-ONLY (jit bakes constants) → LOW value. Fully designed (worktree
  removed); the `_R2deriv` duplicate collapse is the one trivially-safe sub-fix if revisited.
- **multi-RHS spline batching** — ✅ **PR #1133 (2026-07-16)**. `cubic_spline_coeffs` accepts 2-D
  y (n,m) → one multi-RHS solve (coeffs 4,n-1,m); `eval_ppoly` gives dr a trailing axis for
  cb.ndim==3; `_backend_spline_orbit` batches all phase-space columns (k==3), reconstructs R,phi
  for pd 4/6. numpy single-spline BYTE-IDENTICAL (stash). multi==per-col ~1e-14; 82 orbit tests
  pass + new pd-5 off-grid test. Measured 2-orbit pd-6 off-grid jaxpr 1056→256 (~4×). NOTE: only the
  per-orbit column batching (pd→1); the cross-orbit shared-grid batching (36→1 via
  `_call_internal_backend_interp`) was left as a further follow-up.
- **cross-orbit spline batching — ✅ PR #1136 (2026-07-16)**, the #1133 follow-up. All orbits
  share ONE grid → all size*pd columns in a single multi-RHS solve (vs one per-orbit call looped).
  Backend-only (is_backend_array). 83 orbit tests pass + new multi-orbit pd-5. Measured 24-orbit
  off-grid o.R eager 396→45 ms (~8.7×, scales with N). NOTE: concrete-orbit value-eval jaxpr
  unchanged (solve baked at trace); win = eager multi-orbit + differentiating-interp-wrt-ICs.
