# Symplectic-integrator variational (state-transition-matrix) scheme for galpy C

**Goal.** Make galpy's symplectic C orbit integrators (`leapfrog`, `symplec4`,
`symplec6` in `galpy/util/bovy_symplecticode.c`) differentiable by propagating a
phase-space deviation vector (or the full 6×6 state-transition matrix, STM)
alongside the base orbit, exactly as the RK integrators already do through
`evalRectDeriv_dxdv` (12-wide) and `evalRectDeriv_stm` (42-wide) in
`galpy/orbit/orbit_c_ext/integrateFullOrbit.c`.

This document is a math + implementation blueprint. **No new potential-side code
is required**: the Cartesian Hessian assembly (`K = −∇∇Φ`) already exists in
`evalRectDeriv_dxdv` from the Track B variational work and is reused verbatim.

All coefficient constants below were read directly from
`galpy/util/bovy_symplecticode.c` (not from memory) and the drift/kick structure
was traced from the actual integrator loops.

---

## 0. galpy conventions (read these first)

### 0.1 State layout and the separable Hamiltonian

galpy integrates unit-mass particles, so the "momentum" array `p` **is the
velocity** `v`. The Hamiltonian is separable:

    H(q,p) = ½ |p|²  +  Φ(q, t),        q, p ∈ ℝ³ (Cartesian x,y,z)

giving the equations of motion

    q̇ = p            (drift part, from ½|p|²)
    ṗ = a(q,t) = −∇Φ (kick part, from Φ)

The two elementary operators in `bovy_symplecticode.c` are:

- `leapfrog_leapq(dim, q, p, h, qn)`:  `qn = q + h·p`   — **DRIFT** (advance
  position by the velocity), velocity unchanged.
- `leapfrog_leapp(dim, p, h, a, pn)`:  `pn = p + h·a`   — **KICK** (advance
  velocity by the acceleration), position unchanged.

The Cartesian acceleration is assembled from the cylindrical forces exactly as
`evalRectForce` does (lines 1348–1366):

    R = √(x²+y²),  cosφ = x/R,  sinφ = y/R
    aₓ = cosφ·Rforce − (1/R)·sinφ·phitorque
    a_y = sinφ·Rforce + (1/R)·cosφ·phitorque
    a_z = zforce

### 0.2 The acceleration Jacobian K (the Cartesian Hessian)

The tangent (variational) map needs `K = ∂a/∂q = −∇∇Φ` (3×3, symmetric for a
conservative potential). galpy already builds it in `evalRectDeriv_dxdv`
(lines 1586–1625) from the six cylindrical second-derivative aggregators plus
the **conservative-only** force (`RforceK`, `phitorqueK`, obtained with the
`include_dissipative = 0` flag):

    R2deriv   = calcR2deriv(R,z,φ,t,…)
    phi2deriv = calcphi2deriv(R,z,φ,t,…)
    Rphideriv = calcRphideriv(R,z,φ,t,…)
    z2deriv   = calcz2deriv(R,z,φ,t,…)
    Rzderiv   = calcRzderiv(R,z,φ,t,…)
    zphideriv = calczphideriv(R,z,φ,t,…)
    RforceK   = calcRforce(R,z,φ,t,…,0,0,0,0)      // conservative only
    phitorqueK= calcphitorque(R,z,φ,t,…,0,0,0,0)   // conservative only

    Kxx = −cosφ²·R2deriv + 2·cosφ·sinφ/R²·phitorqueK + sinφ²/R·RforceK
          + 2·sinφ·cosφ/R·Rphideriv − sinφ²/R²·phi2deriv
    Kxy = −sinφ·cosφ·R2deriv + (sinφ²−cosφ²)/R²·phitorqueK − cosφ·sinφ/R·RforceK
          − (cosφ²−sinφ²)/R·Rphideriv + cosφ·sinφ/R²·phi2deriv
    Kyy = −sinφ²·R2deriv − 2·sinφ·cosφ/R²·phitorqueK − 2·sinφ·cosφ/R·Rphideriv
          + cosφ²/R·RforceK − cosφ²/R²·phi2deriv
    Kxz = −cosφ·Rzderiv + sinφ/R·zphideriv
    Kyz = −sinφ·Rzderiv − cosφ/R·zphideriv
    Kzz = −z2deriv

    K = [[Kxx, Kxy, Kxz],
         [Kxy, Kyy, Kyz],       (symmetric: Kyx=Kxy, Kzx=Kxz, Kzy=Kyz)
         [Kxz, Kyz, Kzz]]

These are the `dFxdx, dFxdy, dFydy, dFxdz, dFydz, dFzdz` variables in
`evalRectDeriv_dxdv`. **The symplectic-STM kick reuses this block unchanged.**

### 0.3 Deviation-state layout (matches the RK path)

- **Single deviation (12-wide, `_dxdv`):**
  `q = (x,y,z, vx,vy,vz | δx,δy,δz, δvx,δvy,δvz)`.
- **Full STM (42-wide, `_stm`):** 6 base + **six** 6-D deviation columns at
  offsets `6 + 6k`, `k = 0…5`; the k-th column is initialized to the k-th unit
  vector so the propagated block is the 6×6 STM `M`.

The symplectic step naturally uses **split position/velocity arrays**. For an
`n`-column augmented system (`n=1` for dxdv, `n=6` for stm) hold

    Q = [ q ; δq₁ ; … ; δqₙ ]   (3(n+1) doubles, block 0 = base position)
    V = [ v ; δv₁ ; … ; δvₙ ]   (3(n+1) doubles, block 0 = base velocity)

On output, repack into the 12-/42-wide interleaved `(pos₃, vel₃)` layout the
Python caller (`integrateFullOrbit_dxdv_c` / `integrateFullOrbit_stm_c`) expects.

---

## 1. The tangent (variational) map of a drift and a kick

Let a phase-space deviation of one column be `w = (δq, δv) ∈ ℝ⁶`, `δq,δv ∈ ℝ³`.
Ω denotes the 6×6 symplectic form `Ω = [[0, I₃],[−I₃, 0]]`.

### 1.1 Drift tangent map `M_D(h)`

Base drift: `q → q + h·v`, `v → v`. Differentiating w.r.t. the state:

    δq → δq + h·δv
    δv → δv

    M_D(h) = [[ I₃ , h·I₃ ],
              [ 0  ,  I₃  ]]                        (6×6, block form)

This is **exact and linear** (the drift is affine in the state, so its Jacobian
is state-independent). It is the exact time-`h` flow of the free-particle
Hamiltonian ½|p|², hence symplectic: `M_D(h)ᵀ Ω M_D(h) = Ω`, `det M_D = 1`.
Note the group property `M_D(a)·M_D(b) = M_D(a+b)` (used for the micro-step
drift merges below).

Implementation: **`leapfrog_leapq` applied blockwise to the whole `Q`/`V`
arrays** — the deviation drift `δqⱼ += h·δvⱼ` is structurally identical to the
base drift `q += h·v`, so the same loop covers all `3(n+1)` entries.

### 1.2 Kick tangent map `M_K(h)` (conservative)

Base kick: `v → v + h·a(q,t)`, `q → q`, with `a = −∇Φ`, `∂a/∂q = K` (§0.2):

    δq → δq
    δv → δv + h·K·δq

    M_K(h) = [[ I₃ ,  0  ],
              [ h·K,  I₃ ]]                          (6×6, block form)

`K` **must be evaluated at the same intermediate position the base kick uses**
(the position the base orbit holds at the *input* of that kick, i.e. after the
preceding drift) and, for a time-dependent potential, **at the same substep
time** the base kick uses. Because `K` is symmetric (conservative Hessian),
`M_K` is the exact time-`h` flow of the linearized potential and is symplectic:
`M_K(h)ᵀ Ω M_K(h) = Ω`, `det M_K = 1`.

Implementation of the deviation kick, per column j (K from base position):

    δvxⱼ += h·(Kxx·δxⱼ + Kxy·δyⱼ + Kxz·δzⱼ)
    δvyⱼ += h·(Kxy·δxⱼ + Kyy·δyⱼ + Kyz·δzⱼ)
    δvzⱼ += h·(Kxz·δxⱼ + Kyz·δyⱼ + Kzz·δzⱼ)

This is exactly the conservative part of `evalRectDeriv_dxdv` (lines 1635–1643)
with the dissipative blocks `jac_x = jac_v = 0` — which is guaranteed on the
symplectic path (see §4). `K` is assembled **once** per kick from the base
position and applied to all `n` columns (as `evalRectDeriv_stm` already does).

### 1.3 Why the composed per-step STM is EXACT and symplectic

A single symplectic step is an explicit composition of drift and kick maps
`Φ_step = S_m ∘ … ∘ S₂ ∘ S₁` (each `Sᵢ` a drift or a kick). By the chain rule
the Jacobian of the step is the ordered product of the substep Jacobians:

    M_step = J_{S_m} · … · J_{S₂} · J_{S₁}
           = (last substep's Jacobian on the LEFT, first on the RIGHT)

Each `J_{Sᵢ}` is one of the **closed-form** elementary matrices `M_D(cᵢh)` or
`M_K(dᵢh)` above. Therefore:

1. **Exactness.** `M_step` is the *exact* derivative of the discrete numerical
   map that advances the base orbit — computed in closed form from the same
   force + Hessian evaluations the base step already needs, with **no extra RK
   stages and no ODE re-integration**. The transported deviation is consistent
   with the base orbit to machine precision at every step, not merely to
   integrator order.

2. **Exact symplecticity / volume preservation.** Every `M_D` and `M_K` is
   symplectic (`det = 1`, `MᵀΩM = Ω`) for a conservative (symmetric-K) system,
   and a product of symplectic matrices is symplectic. Hence `M_step` satisfies
   `det(M_step) = 1` and `M_stepᵀ Ω M_step = Ω` **exactly per step** (machine
   precision). This is the sharp advantage over the RK `dxdv`/`stm` STM, which
   is only *approximately* symplectic (symplecticity error at the method order).

The RK variational RHS `w' = A·w` with `A = [[0,I],[K,0]]` (the
`evalRectDeriv_dxdv` RHS) is integrated by the same adaptive RK tableau and is
subject to its own truncation/step-control error; the symplectic-STM instead
carries the analytic Jacobian of the actual discrete map.

---

## 2. Per-integrator drift/kick decompositions and STMs

Notation: `D(α) ≡` drift of length `α·dt`; `K(β) ≡` kick of length `β·dt`
(with `K` evaluated at the current base position and the accumulated substep
time). Composition is written left-to-right in *execution order* (first
operation leftmost). The STM per step is the product of the corresponding
tangent maps with **execution order reversed** (first-applied on the right).

### 2.1 `leapfrog` — DRIFT-KICK-DRIFT (DKD), a.k.a. position Verlet

**Subtlety (confirmed from source):** galpy's leapfrog is **DKD**, not KDK. The
first operation of every step is `leapfrog_leapq` (a drift); the kick sits in
the middle and is evaluated at the **half-step position and the midpoint time**
`t₀ + dt/2` (`func(to+dt/2., q12, …)`, lines 158–178). The integrator loop does
one initial half-drift, then `(ndt−1)` interior `[K(1) D(1)]` micro-steps, then
a final `[K(1) D(½)]`; consecutive half-drifts merge because
`M_D(½)·M_D(½) = M_D(1)`. The fundamental step is therefore:

    Φ_step(dt) = D(½) · K(1) · D(1) · … · D(1) · K(1) · D(½)   (chained)
               ≡ [ D(½) K(1) D(½) ]  repeated ndt times

**One fundamental DKD step** (input `q₀,v₀`; deviation `δq₀,δv₀`), time `t₀`:

    1. D(dt/2):  q_h  = q₀ + (dt/2)·v₀ ;      δq_h = δq₀ + (dt/2)·δv₀ ;   (v,δv unchanged)
    2. K(dt):    a  = accel(q_h, t₀+dt/2) ;  K = Hess(q_h, t₀+dt/2)
                 v₁  = v₀ + dt·a ;            δv₁ = δv₀ + dt·(K·δq_h) ;   (q,δq unchanged)
    3. D(dt/2):  q₁  = q_h + (dt/2)·v₁ ;      δq₁ = δq_h + (dt/2)·δv₁ .

**Per-step STM:**

    M_step = M_D(dt/2) · M_K(dt) · M_D(dt/2),   K evaluated at (q_h, t₀+dt/2).

Order (McLachlan/Yoshida order 2): local error O(dt³), global O(dt²).

### 2.2 `symplec4` — 4th-order Forest–Ruth / Yoshida composition

Coefficients (verbatim, lines 230–236; `∑cᵢ = ∑dᵢ = 1` to 1e-16, and they equal
the closed forms `d₁ = 1/(2−2^{1/3})`, `c₁ = d₁/2`, `d₂ = −2^{1/3}/(2−2^{1/3})`):

    c1 = c4 =  0.6756035959798289
    c2 = c3 = -0.1756035959798288
    d1 = d3 =  1.3512071919596578
    d2      = -1.7024143839193153      (d4 = 0)

**One full step** (execution order, lines 313–332):

    D(c1) K(d1) D(c2) K(d2) D(c3) K(d3) D(c4)          (4 drifts, 3 kicks; d4=0)

Kick positions/times: `K(d1)` at `t₀+c1·dt`, `K(d2)` at `t₀+(c1+c2)·dt`,
`K(d3)` at `t₀+(c1+c2+c3)·dt` (each at the base position reached after the
preceding drift — mirror the `to += cᵢ*dt` accumulation in the source).

**Per-step STM** (reversed order):

    M_step = M_D(c4·dt) · M_K(d3·dt) · M_D(c3·dt) · M_K(d2·dt)
             · M_D(c2·dt) · M_K(d1·dt) · M_D(c1·dt)

Interior micro-step merge: the source fuses the trailing `c4` and next leading
`c1` into one `D((c4+c1)·dt)` (line 304); the STM merges identically via
`M_D(c4·dt)·M_D(c1·dt) = M_D((c4+c1)·dt)`.

### 2.3 `symplec6` — 6th-order Yoshida composition (solution A)

Coefficients (verbatim, lines 383–397; `∑cᵢ = ∑dᵢ = 1`):

    c1 = c8 =  0.392256805238780
    c2 = c7 =  0.510043411918458
    c3 = c6 = -0.471053385409758
    c4 = c5 =  0.0687531682525198
    d1 = d7 =  0.784513610477560
    d2 = d6 =  0.235573213359357
    d3 = d5 = -1.17767998417887
    d4      =  1.31518632068391        (d8 = 0)

**One full step** (execution order, lines 498–541):

    D(c1) K(d1) D(c2) K(d2) D(c3) K(d3) D(c4) K(d4)
          D(c5) K(d5) D(c6) K(d6) D(c7) K(d7) D(c8)     (8 drifts, 7 kicks; d8=0)

`K(dᵢ)` at accumulated time `t₀ + (c1+…+cᵢ)·dt` and the base position reached
after the preceding drift.

**Per-step STM** (reversed order):

    M_step = M_D(c8·dt)·M_K(d7·dt)·M_D(c7·dt)·M_K(d6·dt)·M_D(c6·dt)·M_K(d5·dt)
             ·M_D(c5·dt)·M_K(d4·dt)·M_D(c4·dt)·M_K(d3·dt)·M_D(c3·dt)·M_K(d2·dt)
             ·M_D(c2·dt)·M_K(d1·dt)·M_D(c1·dt)

Interior merge: `D((c8+c1)·dt)` (line 489); same `M_D` additivity applies.

---

## 3. Augmented-state update in galpy's Cartesian conventions (drop-in blueprint)

The implementer adds **two augmented operators** that mirror
`leapfrog_leapq`/`leapfrog_leapp` and reuse the *existing* K-assembly. The base
6-vector is stepped exactly as today; only the deviation columns are new.

### 3.1 Augmented drift `leapq_aug(h, Q, V)`

For every 3-block `b = 0…n` (block 0 = base, blocks 1…n = deviation columns):

    Q[3b : 3b+3] += h · V[3b : 3b+3]

i.e. one `leapfrog_leapq` over the full `3(n+1)`-long arrays. No forces, no K.

### 3.2 Augmented kick `leapp_aug(h, t, Q, V)`

1. Base acceleration from the base block `q = Q[0:3]` (identical to
   `evalRectForce`): `a = accel(q, t)`; then `V[0:3] += h·a`.
2. Assemble `K` **once** from the base block `q` at time `t` using the §0.2
   aggregators (the same calls as `evalRectDeriv_dxdv`).
3. For each deviation column `j = 1…n`, with `δq = Q[3j:3j+3]`:

       V[3j+0] += h·(Kxx·δq₀ + Kxy·δq₁ + Kxz·δq₂)
       V[3j+1] += h·(Kxy·δq₀ + Kyy·δq₁ + Kyz·δq₂)
       V[3j+2] += h·(Kxz·δq₀ + Kyz·δq₁ + Kzz·δq₂)

   **`δq` here is the pre-kick deviation position** (position is unchanged by a
   kick), exactly as `M_K` prescribes. Note K uses the base block only; the
   deviation blocks never feed back into K.

### 3.3 Step drivers

Replace, inside each integrator's step, every `leapfrog_leapq` with `leapq_aug`
and every `[func + leapfrog_leapp]` pair with `leapp_aug` (which folds the force
call and the base+deviation velocity update together, evaluating K at the same
`(q, t)` the base kick uses). The exact drift/kick sequence and coefficients are
unchanged from §2; the micro-step merges carry over verbatim. Concretely:

- **leapfrog:** `leapq_aug(dt/2)`; then `[ leapp_aug(dt) ; leapq_aug(dt) ]`
  repeated `(ndt−1)` times; then `leapp_aug(dt) ; leapq_aug(dt/2)`.
- **symplec4:** `leapq_aug(c1·dt)`; then per micro-step
  `leapp_aug(d1·dt) leapq_aug(c2·dt) leapp_aug(d2·dt) leapq_aug(c3·dt)
   leapp_aug(d3·dt) leapq_aug((c4+c1)·dt)` (final micro-step splits the last
  drift into `c4·dt` for output), matching the source's two branches.
- **symplec6:** analogous, with the 8-drift/7-kick sequence of §2.3.

### 3.4 Output repacking

After the last drift of an output step, write each column back into the
12-/42-wide interleaved layout: base `(x,y,z,vx,vy,vz)` then, per column,
`(δx,δy,δz,δvx,δvy,δvz)` — the byte layout `integrateFullOrbit_dxdv_c` /
`integrateFullOrbit_stm_c` already consume. For `_stm`, initialize the six
columns to the identity (column k = unit vector eₖ) so the output block is `M`.

### 3.5 Python gating to lift

Currently `Orbit.integrate_dxdv` calls `check_integrator(method,
no_symplec=True)` (`Orbits.py:2478`), and `integrateFullOrbit_dxdv` /
`_stm` only wire the four RK steppers (`case 1,2,5,6`; the symplectic
`odeint_type`s never reach the C switch). To enable the symplectic-STM:
(a) drop `leapfrog_c/symplec4_c/symplec6_c` from the `no_symplec` block **for
the dxdv/stm entry points**, and (b) add `case 3 (leapfrog_c) / 7 (symplec4_c)
/ 8 (symplec6_c)` branches in `integrateFullOrbit_dxdv` and `_stm` that call the
new augmented symplectic steppers with `dim = 12` / `dim = 42`. (Verify the
integer `odeint_type` codes via `_parse_integrator`.)

---

## 4. Velocity-dependent (dissipative) forces — the harder case

**Bottom line: galpy's symplectic path never sees a velocity-dependent force, so
the symplectic-STM kick is always the conservative, explicit, symplectic map of
§1.2 — no implicit solve is ever needed.**

### 4.1 galpy already excludes symplectic + dissipative

`Orbits._check_method_dissipative_compatible` (`Orbits.py:1464–1475`) reroutes
**any** dissipative potential away from symplectic integrators:

    if _isDissipative(pot) and ("leapfrog" in method or "symplec" in method):
        method = "dopr54_c"  (C)  or  "odeint"  (pure Python)   # + a warning

Consistently, the symplectic force evaluator `evalRectForce` (used by
`leapfrog`/`symplec4`/`symplec6`) does **not** pass velocity to the force
functions, and `evalRectDeriv_dxdv`'s `K` uses the conservative-only
`RforceK/phitorqueK`. So on the symplectic path `∂a/∂v = 0` and the position
Jacobian is the symmetric conservative `K` — the dissipative blocks
`jac_x, jac_v` in `evalRectDeriv_dxdv` are identically zero.

### 4.2 What the kick tangent map *would* be if `∂a/∂v ≠ 0`

For completeness (should the exclusion ever be relaxed): a velocity-dependent
acceleration `a(q,v,t)` has `K_x = ∂a/∂q` (no longer symmetric) and
`K_v = ∂a/∂v` (nonzero). galpy's `leapfrog_leapp` performs an **explicit** kick
`v_new = v + h·a(q, v, t)` (a evaluated at the *pre-kick* v). Its tangent map is
still explicit:

    δq → δq
    δv → δv + h·(K_x·δq + K_v·δv) = (I + h·K_v)·δv + h·K_x·δq

    M_K = [[ I₃   ,   0        ],
           [ h·K_x, I₃ + h·K_v ]]

- **No linear solve** is needed for this explicit kick — `δv_new` is written
  purely from old `(δq, δv)`. This matches how the RK path already handles
  dissipation (via `jac_x = K_x`, `jac_v = K_v` in `evalRectDeriv_dxdv`).
- **Symplecticity is lost by design:** `det(M_K) = det(I + h·K_v) ≠ 1`
  (dissipation contracts phase-space volume), and `M_K` is not symplectic
  (`K_x` non-symmetric). The Liouville/symplectic checks of §5(b) therefore do
  **not** apply here — correctly, because the underlying flow is not
  conservative. The explicit velocity-dependent kick is also only first-order
  accurate and breaks the time-reversal symmetry that makes the base method
  symplectic — which is exactly why galpy forbids it.
- **If a reversible (symplectic-like) kick were desired** (e.g. implicit-midpoint
  on the velocity sub-ODE `v̇ = a(q,v)` with q frozen), then `δv` appears on both
  sides and each kick needs a **3×3 linear solve per column**:
  `(I − ½h·K_v)·δv_new = (I + ½h·K_v)·δv + h·K_x·δq`. galpy does **not** do this;
  it is noted only to delimit the design space.

**Recommendation:** keep the symplectic-STM strictly conservative (mirror the
existing `no_symplec` / dissipative-reroute gates). Do not add a
velocity-dependent branch to the symplectic kick.

---

## 5. Validation battery (hand to the tester)

Let `M` be the propagated 6×6 STM (the 42-wide block), `Ω = [[0,I₃],[−I₃,0]]`.

**(a) Closed-form STM on separable/known systems.**

- *3D isotropic harmonic oscillator* `a = −ω²q` ⇒ `K = −ω²I₃` (constant).
  Because the system is linear, the numerical step map *is* its own Jacobian, so
  the propagated `M` must equal the base symplectic step map to machine
  precision, and both approximate the analytic flow `exp(t·A)`,
  `A = [[0,I],[−ω²I,0]]`, to the integrator order (2 / 4 / 6). Worked one-DOF
  DKD step (per axis), useful as a unit check:

      M_step = [[1, dt/2],[0,1]]·[[1,0],[−ω²dt,1]]·[[1,dt/2],[0,1]]
             = [[1 − ½ω²dt² ,  dt − ¼ω²dt³ ],
                [ −ω²dt      ,  1 − ½ω²dt²  ]],     det = 1 exactly.

- *Kepler* `a = −q/|q|³`: nonlinear; `M` must match the finite-difference-of-flow
  STM (test d) to ~1e-6–1e-8 and the analytic variational solution to integrator
  order; energy/`L` drift bounded (symplectic).

**(b) Exact per-step symplecticity / Liouville (conservative — strong check).**
For every step (and cumulatively):

    | det(M) − 1 | ≲ 1e-12          (few·ULP·steps)
    ‖ Mᵀ Ω M − Ω ‖_∞ ≲ 1e-12

These hold to machine precision **by construction** for the conservative
symplectic-STM (product of symplectic elementary matrices) — a much tighter
bound than the RK STM can meet, so it is the primary correctness gate. Assert it
after each output step, not only at the end.

**(c) Agreement with the RK dxdv/stm STM (`dop853_c`).**
On a shared orbit + IC, compare the symplectic `M` against the `dop853_c`
`evalRectDeriv_stm` `M`. Both converge to the same continuous STM as `dt → 0`;
at finite tolerance they agree to ~ the looser of the two integrators' accuracy.
Test at a few `dt` and confirm the difference shrinks at the expected order.

**(d) Finite-difference-of-the-flow, per column.**
For each basis direction `eᵢ` (i = 0…5), integrate the **base** orbit (no
variational state) from `y₀ ± ε·eᵢ` with the same symplectic method/`dt`, and
form the central difference:

    column_i(M) ≈ [ φ(y₀ + ε·eᵢ) − φ(y₀ − ε·eᵢ) ] / (2ε),   ε ≈ 1e-6

Each STM column must match to ~1e-6–1e-8 (central-difference floor in double).
This directly certifies that `M` is the Jacobian of the *actual discrete
symplectic map*, independent of the symplecticity argument.

**Also recommended:** initialize the six `_stm` columns to the identity and
confirm the base-orbit trajectory is **bit-identical** to the plain
`leapfrog/symplec4/symplec6` run (the deviation machinery must not perturb the
base integration).

---

## 6. Key subtleties found (summary)

1. **galpy's leapfrog is DKD (position Verlet), drift-first**, with the kick at
   the half-step position and **midpoint time** `t₀+dt/2`. Not KDK.
2. **symplec4 = Forest–Ruth 4th order** (`D K D K D K D`, 4 drifts/3 kicks,
   `d4=0`); **symplec6 = Yoshida-1990 6th order** (`D…D`, 8 drifts/7 kicks,
   `d8=0`). Exact coefficient constants in §2.2/§2.3 (read from source, sums
   verified `=1`).
3. **The per-step STM is an exact, closed-form, exactly-symplectic product of
   elementary drift/kick Jacobians** — no extra force evaluations beyond the base
   step, and `det=1`, `MᵀΩM=Ω` hold to machine precision per step (vs only
   approximately for the RK STM). This is the whole point.
4. **No new potential-side code:** the deviation kick reuses the exact
   `R2deriv/z2deriv/Rzderiv/phi2deriv/Rphideriv/zphideriv → Cartesian-K`
   assembly already in `evalRectDeriv_dxdv`, with conservative-only
   `RforceK/phitorqueK`.
5. **Dissipative forces need no implicit solve** in galpy: the symplectic path
   already excludes them (`_check_method_dissipative_compatible` reroutes to
   `dopr54_c`/`odeint`), so the symplectic kick stays conservative and explicit.
   The velocity-dependent generalization (`M_K = [[I,0],[hK_x, I+hK_v]]`) is
   given only for completeness and is *still explicit* (a solve would arise only
   for a reversible implicit kick galpy does not use).
