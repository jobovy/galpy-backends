# Staeckel action gradients: t²-donor graft (Track E derivative fix, PR A)

**Status:** ⏳ **PR #1047** open 2026-07-01 (branch `feat/staeckel-actions-jvp`,
commit e839ca0f); all validation passed (numpy byte-identity tobytes-equal on
24 orbits × 4 quantities, numpy Staeckel suite 72/72, backend suites 978/0).
**Design docs:** `plan/staeckel_deriv_fix/design_scoping_wpo9oiys0.md` (scoping),
`plan/staeckel_deriv_fix/RESUME.md` (session handoff that carried the bug report).

## Approach

Problem: Staeckel actions use plain order-10 GL of √S between turning points
(C parity, byte-identical numpy). The VALUE is fine (√S vanishes at the turning
points) but backend AD through it computes ∂S/(2√S), **singular at the turning
points** → silently wrong `d(actions)/d(x,v)` (~1–6% at order 10, converging
only algebraically, p≈1).

First attempt (the scoping doc's design): graft the analytic chain
`dJ/d(E,Lz,I3) · d(E,Lz,I3)/d(x,v)` from the validated `_staeckel_jacobian`
onto the plain-GL value via a stop-gradient reparameterisation. **Velocity
components matched FD to 1e-9; position components were wildly wrong**
(`∂Jr/∂R` +0.86 excess, ~orbit-independent).

Root cause (4-probe diagnosis workflow, `wf_c80e7340-0dd`): the `(E,Lz,I3)`
boundary is **incomplete for position derivatives**. `_staeckel_setup` sets
`u0 = ux(R,z)` (comment: "u0 does not matter for a single action evaluation" —
true for the value, false for the gradient): the reference-geometry terms carry
direct R,z dependence. Probe A3 named the missing term analytically:
`−∂Jr/∂I3U · d[(sinh²u0+sin²v0u)·potu0v0]/d(R,z)` (per-entry attribution
reproduces the missing piece to 4e-12; for Jz, u0 additionally enters the
integrand directly via `potentialStaeckel(u0, v)`).

Final fix (simpler than the scoped design): **donor graft**. Keep the plain-GL
forward value; take the gradient from a t²-substituted VALUE quadrature
(`_staeckel_t2_action`: low panel `u = lo + t²`, high panel `u = hi − t²`;
`du = 2t dt` cancels the `1/√S ~ 1/t`, so AD through it is turning-point-regular)
via `graft_gradient(value, donor) = sg(value) + donor − sg(donor)` — the same
first-order stop-gradient construction as `bisect_root`'s Newton reparam. The
donor carries the FULL dependence (E, Lz, I3, u0/v0u geometry, and potential
parameters) automatically — no hand-derived integrands, nothing to go stale.
Turning-point limits are `stop_gradient`-ed (Leibniz boundary terms vanish
exactly, S=0 there). Gated on `is_backend_array(jr) and (under_jax_trace(jr) or
under_torch_grad(jr))` → zero cost on plain forwards and on the forced-backend
suite; numpy path structurally untouched.

## Dead-ends

1. **`dJ/d(E,Lz,I3)` jacobian graft** (`graft_jacobian`, removed): exact for
   velocities, misses the u0/v0u reference-geometry position dependence. Would
   need per-geometry-entry Leibniz integrands (incl. the direct u0-in-integrand
   path for Jz) — strictly more code than the donor for less coverage.
2. **Raising naive-AD GL order instead**: converges only algebraically (order
   300 → ~1e-4); donor at order 10 already beats it, at 30× less work.
3. **torch `gradcheck`**: inapplicable by design — the graft intentionally makes
   the analytic gradient (true action derivative) differ from FD-of-the-forward
   at the plain-GL-vs-substituted quadrature-offset level (~5e-4).

## Results (order 10, MiyamotoNagai a=0.5 b=0.0375, δ=0.45)

- FD gold (`wf_c80e7340-0dd` probe A1, Richardson, spread ≤1.1e-5):
  `∂Jr/∂R` = 0.13946886 (O1) / 0.08151310 (O2) / 0.00096265 (O3).
- Grafted (donor) vs FD: **all 10 components, both backends, 1.4e-4–4.7e-4
  relative** (the plain-GL-vs-donor value-quadrature offset — the donor is the
  gradient of the *true* action). Pre-fix: naive AD ~1–6%; jacobian graft
  ~620% on `∂Jr/∂R`.
- Forward value: numpy byte-identical (gate excludes numpy); backend forward
  0–1 ulp vs numpy; value under AD trace == no-grad forward (donor cancels).
- Conservation witness: FD/donor gradients give `dJ/dt = ∇J·flow` ≈ 2e-3
  (Staeckel-fudge drift); the jacobian-graft value violated it by ~120×.
- `tests/test_backend_staeckel_grad.py`: 18 passed, 67 s — 6 orbits (incl.
  vR=0 / vz=0 / z=0 turning-point edges) × jax+torch, jit survival, public-API,
  value-unchanged, numpy-identity of the helpers.
- First-order only (like C-STM/bisect_root): Hessians & freq/angle gradients
  remain Phase 2 (need second-derivative integrands; ∂²√S ~ 1/t² diverges even
  substituted).

## Files

`galpy/backend/_namespaces.py` (+`under_torch_grad`, `stop_gradient`,
`graft_gradient`), `galpy/actionAngle/actionAngleStaeckel.py`
(+`_staeckel_t2_action`, graft in `_staeckel_jr_jz` → also covers the
actionsFreqs/actionsFreqsAngles action outputs), `tests/test_backend_staeckel_grad.py`.

## Follow-ups

- PR B: StaeckelGrid gradients (value=spline / grad via direct prep+donor).
- PR C: the C `c=True` path (pure_callback custom_vjp; C derivative integrands
  or a host round-trip through the Python donor).
- Parameter-gradient test (d(actions)/dθ through the donor's potential evals —
  mechanism present, untested).
- Phase 2: second-derivative integrands (freq/angle gradients, Hessians).
- g17 gradient panel once this + C path land.
