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

1. **Roll the shared `bisect_root` for-loop → `lax.while_loop` (jax)** + thread a static step count.
   `optimize.py:165` `for _ in range(n)` unrolls ~100 copies of the physics closure. It's the
   ONE root-finder behind **every** backend turning-point solve: `Potential.rl/rE/zmaxgrid/
   rliftedgrid`, Spherical `rperi/rap`, Vertical `xmax`, IsochroneInverse `eta`, ×2 per Staeckel.
   Bonus (verified): `n_bisect_steps` falls back to **100** iters under a tracer (vs ~46–51
   concrete; Staeckel passes maxiter=200) → threading a static `nsteps`/width-hint roughly
   **halves every backend bisection**. *backend-specific, effort M, compile+gradient.*
2. **Roll the expanding-bracket loops** (Staeckel `60+80+80`, Spherical `2×80`, Vertical `80`)
   → `lax.while_loop`/`fori_loop`. Bodies are already single branch-free `xp.where` updates.
   The concrete 47k-jaxpr instances in the hottest subsystem; stacks with P0#1.
   `actionAngleStaeckel.py:127,133,164`; `Spherical.py:564,569`; `Vertical.py:336`. *jax, M.*
3. **Collapse `c_stm_forward`'s 6-per-IC variational integrations → ONE stacked `integrate_dxdv`.**
   `inbackend_stm.py:59-83` re-integrates the identical base orbit 6× (once per basis column).
   The stacked single-call idiom is already proven in-repo (Lyapunov `Orbits.py:2893-2907`,
   `rectIn/rectOut=True`, `numcores`). ~6× forward-cost cut on the headline C-STM gradient path.
   *backend-agnostic, effort M, warm+gradient.*
   **DONE (partial) → PR #1023 (2026-06-27).** Stacked the 6N calls into ONE `integrate_dxdv`
   on 6N orbits; bit-identical M (24/24 exact), 117 STM tests green. Measured N=20: 881ms →
   750ms (nc=1) → 480ms (nc=8). **NOT 6×** — the premise was wrong: `integrate_dxdv` is
   base+1-deviation, so each of the 6 stacked copies *re-integrates the base*. The actual win
   is call-overhead (~1.2×) + `numcores` parallelism across the 6N orbits (~Ncore×). The true
   6× needs a base+6-deviation **augmented 42-state** variational C integrator (integrate the
   base ONCE alongside the 6×6 deviation block) — new follow-up (task #100).

## P1 — next

4. **Vectorize the EllipsoidalPotential GL node loop** (`_potInt/_forceInt_all/_2ndDerivInt_all`,
   `EllipsoidalPotential.py:469,510-522,575-598`): `for k in range(len(glx))` (K~20–50) →
   one broadcast `dens(m)`/`psi(m)` over a node axis + one `xp.sum`. Universal inner loop for
   every triaxial potential; collapses K dispatches → 1 and shrinks the graph K-fold. *agnostic, M.*
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
