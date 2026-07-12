# #131 C-native Staeckel freq/angle Jacobians — implementation plan (scoped 2026-07-03)
# (produced by workflow wf_ec134124-225; full text below)

I have everything grounded. Here is the plan.

---

# Implementation Plan — galpy #131: C-native differentiable Staeckel FREQUENCY (and ANGLE) Jacobians (`c=True`)

Extends PR #1051 (c=True action Jacobian) and #1059 (backend-native freq gradients / differentiable turning points). Worktree `/var/tmp/staeckel-cnative`, branch `feat/backends`. **Supervised C — maintainer review before any code.** Reviewer directives honored: c=True grads must be C-native (no Python-donor fallback), first-order accepted (#1050); numpy path byte-identical; separate `_jax`/`_torch` files; no internal jit; concise comments.

---

## 1. GOAL + SCOPE

Make `Omega_{r,phi,z}` (and, in a follow-up, `angle_{r,phi,z}`) carry a correct **first-order** gradient w.r.t. `(R,vR,vT,z,vz)` when `actionsFreqs` / `actionsFreqsAngles` run with `c=True` on a jax/torch input array — today those outputs are `stop_gradient`'d/`detach`'d (`_staeckel_c_forward_values`, `actionAngleStaeckel.py:670`; jax `c_value:17`, torch `c_value:14`).

- **Deliverable:** a native-C `(N,3,5)` Jacobian `d(Omega_r,Omega_phi,Omega_z)/d(R,vR,vT,z,vz)` (and later `d(theta)/d(coords)`), exposed through the SAME `jax.custom_vjp` / `torch.autograd.Function` matvec pattern as #1051's action Jacobian (`backend/_jax/staeckel_c.py:41`, `_torch/staeckel_c.py:31`) and the C-STM (`orbit_stm`).
- **First-order only, explicitly.** The C emits the analytic first derivative of Omega (which is a *second* derivative of `J`) as a value; the backend tie exposes it as a Jacobian. `jax.pure_callback` has no JVP and `autograd.Function.backward` is not itself differentiable, so `jax.hessian`/double-backward on Omega is out of scope (same blocker as diffrax-higher-order-AA; second-order would need `S^{5/2}` integrands + a second turning-point cancellation).
- **numpy path untouched / byte-identical.** The new C rule engages only under `is_backend_array`. The frequency/action *values* returned by the new fused C entry must equal the existing `c=True` numpy values (`actionAngleStaeckel_actionsFreqs`) to rtol 1e-15 — enforced by test (§4).
- **Reframing (drives all the math):** Omega and theta *are* first derivatives of `J` (`Omega = calcFreqsFromDerivsStaeckel(dJ/d(E,Lz,I3))`, `actionAngleStaeckel.c:1225`), so their coordinate Jacobian is the **action Hessian** chained through the elementary `d(E,Lz,I3Ũ,I3V,u0)/d(coords)` block that `actionsJac` already assembles (`actionAngleStaeckel.c:680–767`). This is why #1051 shipped actions-only.

---

## 2. NEW C FUNCTIONS

Two routes exist for the one genuinely-missing piece — the action Hessian / regularized second-derivative integrals. **I recommend building Route B (finite-difference of the integrals of motion) first as the primary implementation, and keeping Route A (analytic regularized integrands) as a documented accuracy upgrade.** Rationale in §6; both are equally "C-native." Everything downstream (chain to coords, new EXPORT entry, ctypes, backend tie, tests) is identical for either route.

### Shared new EXPORT entry

`actionAngleStaeckel_actionsFreqsJac` — clone `actionAngleStaeckel_actionsFreqs` (proto `actionAngleStaeckel.c:131–134`) + splice the elementary-derivative chain out of `actionAngleStaeckel_actionsJac` (def `:573–777`, inner chain `:680–767`). Signature mirrors `actionsJac` (`:135–137`) with an extra output buffer:

```
EXPORT void actionAngleStaeckel_actionsFreqsJac(
    int ndata, double *R,*vR,*vT,*z,*vz,*u0,
    int npot, int *pot_type, double *pot_args, tfuncs_type_arr pot_tfuncs,
    int ndelta, double *delta, int order, int useu0,
    double *jr,*jz, double *Or,*Op,*Oz,      // values (byte-identical to actionsFreqs)
    double *ojac,                             // (N*15) = d(Or,Op,Oz)/d(R,vR,vT,z,vz)
    int *err);
```

Add its prototype next to `:135`. Body: run existing setup + `calcUminUmax`(`:167`)/`calcVmin`(`:170`) + `calcJRStaeckel`/`calcJzStaeckel` + `calcdJRStaeckel`(`:156`)/`calcdJzStaeckel`(`:160`)/`calcdJzdU0Staeckel`(`:164`) + `calcFreqsFromDerivsStaeckel`(`:1225`) exactly as today (this reproduces the values), then compute the new `dOmega/dP` block and chain.

### Route B (recommended primary) — FD of the integrals of motion

New helper `calcFreqDerivsFromIoMStaeckel` (or inline): for each inner variable `P ∈ {E, Lz, I3Ũ, I3V, u0}`, form central differences by re-evaluating the existing pipeline at the perturbed IoM and recomposing Omega:

- `dOmega/dE`, `dOmega/dLz`: perturb `E`/`Lz`, **recompute both turning points** (`calcUminUmax`/`calcVmin`) and all six `calcdJR/dJz` integrals, recompute `calcFreqsFromDerivsStaeckel`, difference. Recomputing the turning points at the perturbed IoM captures the turning-point-motion term automatically — no explicit `dumin/dumax/dvmin` sensitivity, no `S^{-3/2}` regularized integrand.
- `dOmega/dI3Ũ`: perturb `I3Ũ` in the `S_R` (u-integral) side only (`djrd*` + `umin/umax`).
- `dOmega/dI3V`: perturb `I3V` in the `S_z` (v-integral) side only (`djzd*` + `vmin`).
- `dOmega/du0`: perturb `u0` in the `S_z` side only. (Precedent for FD-in-this-C: the `useu0=2` branch already central-differences the stationarity residual — `fpp`, `actionAngleStaeckel.c:706–708`.)

The `I3Ũ/I3V` split is load-bearing: `S_R`'s `potu0v0` gauge cancels so `djr*` carry no `u0` term (do NOT add one), while `S_z` genuinely depends on `u0` — mirroring how `actionsJac` adds `djzdU0` for Jz but nothing for Jr (`:765`). Perturbing `I3Ũ` on the u-side and `I3V` on the v-side respects this.

Cost: ~5 extra turning-point solves + integral sets per orbit — cheap vs. the correctness risk it removes. Accuracy: ~1e-7 rel (central FD), which is the trade-off the maintainer must accept (§6, Q1).

### Route A (accuracy fallback, per the math spec) — analytic regularized second-derivative integrands

New helpers `calcd2JRStaeckel` / `calcd2JzStaeckel` mirroring `calcdJRStaeckel`(`:1276`)/`calcdJzStaeckel`, producing the symmetric **3×3** u-Hessian `d²jr/d(E,Lz,I3Ũ)²` and the **4-column** v-Hessian `d²jz/d(E,Lz,I3V,u0)²`. Each panel is a t²-substituted GL integral of the finite regularized integrand (math spec §2.2):

```
G_low(t;P) = 2t·[ f_u(u)·A_a/√S − ½·f(u)·(∂_P S(u) + S_u(u)·A_a)/S^{3/2} ],   u = a+t²
A_a = ∂a/∂P = −∂_P S(a)/S_u(a)   (implicit turning-point derivative, evaluated once at tp a)
```

finite at `t→0` because the bracket is `O(t²)` while `S^{3/2}=O(t³)`. Requires **new** `S^{-3/2}` integrand functions (none exist — all current integrands are first-order `factor/√S`, decls `:177–197`), the closed forms `S_u`, `f_u` (math spec §1), the only surviving second-partial `∂²S/∂Lz² = −1/(δ²·{sinh²u,sin²v})` (§2.3), and turning-point sensitivities `A_a,A_b` via implicit diff of `S(u_tp)=0` (absent from C today; the c=False analog is `_refine_tp`, `actionAngleStaeckel.py:138`). Add matching `Low`/`High` wrappers per the `dJRd*Low/High` pattern.

### Compose Omega Jacobian (both routes)

`calcFreqsFromDerivsStaeckel` (`:1248–1252`) is a **rational function** of the six `djXdY` (no linear solve). Quotient-rule partials `∂Omega/∂(djXdY)` are elementary closed forms (e.g. `∂Omega_r/∂djzdI3 = −djzdE·djrdI3/det²`). For Route B these are folded into the FD directly (FD of the whole composed Omega); for Route A, `∂Omega/∂P = Σ_k (∂Omega/∂djXdY_k)·(∂djXdY_k/∂P)` using the §Route-A Hessians. Block structure: `∂(djr*)/∂I3V = ∂(djr*)/∂u0 = 0`, `∂(djz*)/∂I3Ũ = 0`.

### Chain to coordinates (reuse verbatim, both routes)

Reuse the elementary `dE[5], dLz[5], dux/dvx, dpux/dpvx, dI3Ut, dI3V, du0[5]` block from `actionsJac` (`:680–767`):

```
ojac[+k] = Σ_P (dOmega/dP)·dP[k],   P ∈ {E, Lz, I3Ũ, I3V, u0},   dP ∈ {dE, dLz, dI3Ut, dI3V, du0}
```

with `du0[k]` selected by `useu0` mode exactly as `:701–716` (default mode 0: `du0=dux ≠ 0`, so the `dOmega/du0` term is required — its omission was the old ∂Jr/∂R-off-by-1000× class of bug, now resolved via the `I3Ũ` geometry terms `dI3Ut_dux`, `:736`).

### Sentinels / edge cases

Preserve `9999.99` / `-9999.99` and near-circular/planar sentinels (`(umax−umin)/umax<1e-6`, `(π/2−vmin)<1e-7`). Where `calcFreqsFromDerivsStaeckel` sets the `9999.99` freq sentinel (`:1242`), set the corresponding `ojac` rows to 0. The Python `_staeckel_c_freq_circ_fix` (`:712`) NaN→epifreq/omegac/verticalfreq value substitution stays in Python on the returned values; **zero the freq-Jacobian rows at substituted indices** (documented first-order limitation for jr<1e-3 / jz<1e-3 — those substituted values' true derivatives are out of scope).

### Angles (follow-up PR — see §5)

`actionAngleStaeckel_actionsFreqsAnglesJac`, cloning `actionAngleStaeckel_actionsFreqsAngles` (proto `:125–130`), adding a second `(N,3,5)` buffer `ajac = d(angle_r,angle_phi,angle_z)/d(R,vR,vT,z,vz)`. Angles (`calcAnglesStaeckel`, `:1541`) = `Omega·Or_sum + dI3dJ·I3_sum (+π/2, +phitmp)` with `dI3dJ = calcdI3dJFromDerivsStaeckel` (`:1256`). Two extra ingredient types beyond freqs:

- `∂(dI3dJ)/∂P` — same Hessian/FD machinery, quotient-rule on `:1271–1273`.
- **Current-position boundary term (the one structural addition for angles):** the partial integrals `Or_sum,I3_sum` (`_staeckel_angle_partial`, `actionAngleStaeckel.py:484`) run from a turning point to the *current* position `ux/vx`, where `S>0`, so `d(partial)/dcoord` gets a finite non-vanishing `[f(ux)/√S(ux)]·(dux/dcoord)` term (+ `vx`). **Route-B handles this for free:** additionally FD `calcAnglesStaeckel` w.r.t. its position inputs `ux,vx` at fixed IoM and chain through `dux[k]/dvx[k]` (already in the actionsJac block). Route A requires hand-coding the boundary evaluation. Freqs have no such term (both limits are turning points).
- **phi:** `angle_phi = (...) + phi`. Keep the C value byte-identical (phi included) and set the phi column of the angle residual analytically in the backend as `(0,1,0)` (i.e. `d angle_phi/dphi = 1`); the angle custom_vjp then takes 6 inputs `(R,vR,vT,z,vz,phi)`.

---

## 3. PYTHON WIRING CHANGES

Mechanical mirrors of the #1051 scaffolding; the only new logic is more output rows in the matvec.

**(a) ctypes wrapper** — new `actionAngleStaeckelFreq_actionsJac_c` in `actionAngleStaeckel_c.py`, cloning `actionAngleStaeckel_actionsJac_c` (`:149–287`). Allocate `ojac = numpy.empty(len(R)*15)`; add three value out-buffers `Or,Op,Oz` and one `ndpointer(float64)` for `ojac` to `argtypes` (like `jac` at `:222`); keep the `c_int(order)`/`c_int(refu0mode)` slots (`:218–219`); pass them in the call (`:248–268`); `return (jr,jz,Or,Op,Oz, ojac.reshape((len(R),3,5)), err.value)`. (Angles PR: a second wrapper with two out-Jac buffers `ojac,ajac`.)

**(b) jax backend** (`backend/_jax/staeckel_c.py`) — generalize `actions_with_jac` (`:41–90`) to an N-output custom_vjp `actionsfreqs_with_jac`:
- primal → `(jr,jz,Or,Op,Oz)`; `_call` output structs (`:65–69`) grow to 5 `ShapeDtypeStruct((N,))` + one `((N,3,5))` Jacobian residual (or stack action+freq Jac into `(N,5,5)` if fusing — see §5).
- `_bwd` (`:83–87`): replace the hardcoded 2 rows with a loop over the value outputs, `g = Σ_o ct_o[:,None]·jac[:,o,:]` → `(N,5)`, then `tuple(g[:,k] for k in range(5))`. Byte-identical matvec structure to `:86`.

**(c) torch backend** (`_torch/staeckel_c.py`) — generalize `_ActionsJacFunction` (`:31–50`): `forward` saves the stacked `(N,n_out,5)` Jac and returns all `n_out` values on the input device/dtype (`:37–42`); `backward` receives one cotangent per output, `g = Σ_o g_o[:,None]·jac[:,o,:]` (`:48` generalized), returns `(None,) + 5 coord grads`.

**(d) routing** (`actionAngleStaeckel.py`) — in the `actionsFreqs` backend block (`:973–1018`), **replace** the split of differentiable `_staeckel_c_grad_actions` (`:990`) + value-only `_staeckel_c_forward_values(_host_freqs,...)` (`:1015`) with a single new dispatcher `_staeckel_c_grad_actionsfreqs` (mirror `_staeckel_c_grad_actions:632–667`) that builds a `host_jac` closure around the point-(a) wrapper (applying `_staeckel_c_freq_circ_fix` to the returned Or,Op,Oz and zeroing the corresponding `ojac` rows) and routes into the point-(b)/(c) backend fn by namespace name (`"jax"`/`"torch"`). `Lz=R*vT` stays a differentiable backend multiply (`:978`). `_staeckel_c_backend_refu0`/`refu0mode` reused unchanged (`:683`, `:979`). Same for `actionsFreqsAngles` (`:1174–1222`), adding phi and the angle Jac. Keep `_staeckel_c_forward_values`/`c_value` in place for any output deliberately left value-only.

---

## 4. TEST PLAN

New `tests/test_backend_staeckel_freq_grad.py` (per-module file convention). Mirror the #1059 backend probes but for `c=True`, under **both jax and torch**. Reuse the MiyamotoNagai probe and FD helpers from `test_backend_staeckel_grad.py` (`test_staeckel_action_grad_vs_fd:123`, `_useu0_grad_vs_fd:329`, `_grad_public_api:225`); gradcheck pattern from `test_backend_orbit_stm.py:134`.

1. **grad-vs-FD of Omega_r w.r.t. R** at the MiyamotoNagai probe: assert `d(Omega_r)/dR ≈ −1.93` (given target) via `jax.grad` / torch autograd vs central-difference of the `c=True` `Omega_r`. Extend to the full `(3,5)` `d(Omega_{r,phi,z})/d(R,vR,vT,z,vz)` on a small grid.
2. **c=True vs c=False backend agreement:** the c=False AD path (`_staeckel_freqs:436` + `_refine_tp:138`) computes the same `dOmega/dcoord`; assert the C Jacobian agrees. **Tolerance is route-dependent** — GL-convergence (~1e-10) for Route A, ~1e-6 rel for Route B (FD floor). This test is the correctness witness (as the action Jac was validated to ~4e-16 vs freqs).
3. **Byte-identity of values:** extend `test_staeckel_actionsfreqs_c_true_backend` (`:244–287`) so the fused entry's `(jr,jz,Omega_r,Omega_phi,Omega_z)` equal the numpy `c=True` values to rtol 1e-15 (Hessian outputs are purely additive; must not perturb the value quadrature).
4. **Edge orbits (required):** include `vR=0`, `vz=0`, `z=0` turning-point-edge ICs (per the merged-#1013 umax-bracketing lesson) plus near-circular/planar degenerates — assert no NaN leaks into grads and the documented zeroed rows behave.
5. **useu0 modes 0/1/2** grad-vs-FD (mirror `_useu0_grad_vs_fd:329`) — verifies the `du0` term and the mode selection.
6. **torch.autograd.gradcheck** on the freq outputs (first-order FD-vs-AD; does not need second-order).

Angles PR adds: grad-vs-FD of `angle_{r,z}` w.r.t. coords; `d(angle_phi)/dphi == 1`; c=True-vs-c=False angle agreement; angle value byte-identity.

**CI/coverage:** add the jax+torch exercises so the new `if is_backend_array`/`if xp is not numpy` branches are covered (backend-tests.yml uploads no coverage — an uncovered backend line fails codecov/patch). Force CPU for validation (`CUDA_VISIBLE_DEVICES=""`, `JAX_PLATFORMS=cpu`); verify torch flips serially (local CUDA torch dxdv segfault noise). PYTHONPATH=worktree + assert `galpy.__file__` (editable-shadow).

---

## 5. PHASING (recommended PR breakdown)

Freqs and angles differ structurally (angles add the current-position boundary term + phi), so split:

- **PR-A — frequency Jacobian (the bulk).** New second-derivative machinery (Route B FD, or Route A integrands) + `actionAngleStaeckel_actionsFreqsJac` + ctypes wrapper + backend generalization + `actionsFreqs` rewiring + tests §4.1–6. This is where all the new C math lives; angles then reuse it.
- **PR-B — angle Jacobian (follow-up).** `actionAngleStaeckel_actionsFreqsAnglesJac` reusing PR-A's Hessian/FD for `∂Omega,∂dI3dJ`, adding the partial-integral position-boundary term + phi handling + `actionsFreqsAngles` rewiring + angle tests.

**Fuse vs. separate custom_vjp:** for PR-A, recommend a **single fused** `actionsFreqsJac` entry returning `(jr,jz,Or,Op,Oz)` + one stacked `(N,5,5)` residual (rows jr,jz,Omega_r,Omega_phi,Omega_z), replacing BOTH the action custom_vjp and the value-only freq passthrough in `actionsFreqs`. It matches the C flow (`actionsFreqs` already computes actions + action-derivatives + freqs in one pass; the Hessian terms are additive), avoids recomputing the six derivative integrals twice, and the backward is the same matvec with 5 rows. Keep #1051's `actions_with_jac` (2-row) untouched for `__call__`.

Optional internal milestone inside PR-A: land the second-derivative infrastructure (Route B FD helper or Route A `calcd2JR/d2Jz`) validated against the c=False `_staeckel_jacobian` Hessian before wiring Omega — but it is not independently user-visible, so a single PR-A is fine (PR-granularity: group related fixes).

---

## 6. RISKS / OPEN QUESTIONS FOR THE MAINTAINER

**Q1 (primary strategic fork — needs your call). Route B (FD-of-IoM) vs Route A (analytic regularized integrands).** This directly answers "is there a simpler implicit-function route?": **yes.** Because Omega is a rational function of six integrals each parameterized by `(E,Lz,I3,u0)`, and the turning points are recomputable at any perturbed IoM, central-differencing `calcFreqsFromDerivsStaeckel∘(calcUminUmax,calcVmin,calcdJR,calcdJz)` w.r.t. each IoM gives `dOmega/dP` with the turning-point-motion term captured automatically — **no `S^{-3/2}` integrands, no explicit `A_a=−∂_P S/S_u` cancellation, no new turning-point sensitivities.** It reuses 100% of existing C (precedent: the `useu0=2` FD `fpp` at `:706`). For angles it also captures the current-position boundary term (FD w.r.t. the `ux,vx` inputs). Cost: ~5× turning-point-solve+integral work per orbit. **Trade-off: accuracy.** FD floors the c=True-vs-c=False gradient agreement (§4.2) at ~1e-6 rel, vs ~1e-10 for the analytic route — but #1050 accepts first-order gradients, and C-STM is first-order too, so FD is defensible. **My recommendation:** implement Route B first (fastest to correct, lowest risk, maximal reuse); keep Route A as a drop-in accuracy upgrade if you want the tight tolerance. Do you accept an FD-accuracy gradient for c=True freqs, or require analytic tolerance?

**Q2. Are angles harder than freqs? Yes — one structural addition.** The angle partial integrals end at the *current* position (not a turning point), so `dtheta/dcoord` carries a finite non-vanishing boundary term `[f(ux)/√S(ux)]·dux/dcoord` absent from freqs (both limits are turning points). Route B absorbs it (FD w.r.t. `ux,vx`); Route A must hand-code it. Additional delicacy: when the current position sits *near* a turning point (`ux→umin`), `S(ux)→0` makes the boundary factor `f(ux)/√S(ux)` large — needs a guard/regime check. Hence the PR-A/PR-B split.

**Q3. Does the matrix inverse in `calcFreqsFromDerivsStaeckel` complicate the backward? No.** It is an explicit rational function (`:1248–1252`), not a linear solve — quotient-rule partials are closed-form and there is no implicit-function step in the composition. The only implicit-function piece is the turning-point sensitivity, which Route B sidesteps entirely and Route A localizes to `A_a`. The `custom_vjp`/`autograd.Function` backward is a plain matvec of the C-supplied Jacobian (as in #1051), unaffected by how Omega was differentiated inside C.

**Q4. Near-circular / planar first-order gap.** The Python `_staeckel_c_freq_circ_fix` (`:712`) substitutes epifreq/omegac/verticalfreq for NaN Staeckel freqs at `jr<1e-3`/`jz<1e-3`; those substituted values have true nonzero coordinate derivatives we are **zeroing**. Acceptable as a documented first-order edge limitation, or do you want the epifreq/omegac/verticalfreq derivatives grafted (they have their own C/backend gradient paths)?

**Q5. `useu0` FD-step and I3Ũ/I3V split.** Route B needs a chosen FD step per IoM (relative, `~1e-6·|P|` with a floor). Confirm the `dOmega/du0` FD should perturb `u0` on the `S_z` side only (S_R's `potu0v0` gauge cancels — no `djr*` u0 term), matching the actionsJac Jr/Jz asymmetry (`:765`). This asymmetry is the resolution of the old ∂Jr/∂R root-cause (I3Ũ geometry, not a u0 gauge term).

**Q6. Fused `(N,5,5)` value byte-identity.** The fused `actionsFreqsJac` must reproduce `actionAngleStaeckel_actionsFreqs`'s `jr,jz,Omega` bit-for-bit; confirm you want the Hessian/FD purely additive (extra output buffers, zero perturbation to the value quadrature order/nodes) so test §4.3 holds.

---

**Key files:** `/var/tmp/staeckel-cnative/galpy/actionAngle/actionAngle_c_ext/actionAngleStaeckel.c` (`actionsJac`:573 + chain 680–767, `calcFreqsFromDerivsStaeckel`:1225, `calcdI3dJFromDerivsStaeckel`:1256, `calcdJRStaeckel`:1276, `calcAnglesStaeckel`:1541, integrand decls 177–197, `calcUminUmax`:167/`calcVmin`:170); `/var/tmp/staeckel-cnative/galpy/actionAngle/actionAngleStaeckel_c.py` (`actionAngleStaeckel_actionsJac_c`:149–287); `/var/tmp/staeckel-cnative/galpy/actionAngle/actionAngleStaeckel.py` (routing 973–1018 / 1174–1222, `_staeckel_c_grad_actions`:632, `_staeckel_c_forward_values`:670, `_staeckel_c_freq_circ_fix`:712; c=False reference `_refine_tp`:138, `_staeckel_freqs`:436, `_staeckel_angles`:501); `/var/tmp/staeckel-cnative/galpy/backend/_jax/staeckel_c.py` (`actions_with_jac`:41), `/_torch/staeckel_c.py` (`_ActionsJacFunction`:31); tests `tests/test_backend_staeckel_grad.py` (:123/:225/:244/:329), `tests/test_backend_orbit_stm.py:134`.

---
## Route-A scheme REFINEMENT (validated 2026-07-03, Python prototype)

The scoping plan's G_low formula (t²-substituted, mid-split) is INCOMPLETE: the
Low/High split point `mid=√(0.5(umax−umin))` itself moves with the IoM P, so
differentiating the mid-split panels drops the measure/domain-motion terms (a naive
fixed-mid implementation was ~1000× wrong; verified). The bracket-cancellation is
fine (bracket = S_u·A_a+∂_P S = O(t²) at the turning point, confirmed numerically).

FIX — use a **fixed-domain θ-map** (no boundary terms, so d/dP is purely
under-integral): u = c − r·cosθ, θ∈[0,π], c=(umin+umax)/2, r=(umax−umin)/2.
  I_{P0} = ∫₀^π g_{P0}(u)/√S · r sinθ dθ  (= ∫ g/√S du, the existing first-deriv)
  d(djr_{P0})/dP_k = pref(P0)·∫₀^π [ r sinθ·(g_u·u_P/√S − ½ g(S_u·u_P + S_P)/S^{3/2})
                                     + r_P·g/√S·sinθ ] dθ  + dpref(P0)/dP_k · I_{P0}
  u_P = c_P − r_P·cosθ; c_P=(A_min+A_max)/2, r_P=(A_max−A_min)/2;
  A_min = −S_P(umin)/S_u(umin), A_max = −S_P(umax)/S_u(umax)  [turning-pt sensitivities]
Integrand finite at θ=0,π (bracket O(θ²) / S^{3/2}=O(θ³) × measure rθ). Geometric
g (sinh²u, 1/sinh²u, 1) → ∂_P g=0. Only djrdLz's prefactor depends on P (dpref term).
Validated vs jax.jacfwd of the differentiable djr(E,Lz,I3): max abs 2.6e-7 (order-limited).
Prototype: scratchpad/validate_jr_hessian_v2.py. Jz analogous (u0 dir non-geometric).
Needs closed-form S_u, S_P in C (prototype used jax.grad).

## PR-A (freqs) DONE — PR #1060 (2026-07-03)
C-native c=True Omega Jacobian implemented + validated + committed (0a5ef809),
PR #1060 -> feat/backends. calcd2JRStaeckel/calcd2JzStaeckel (theta-map action
Hessians) + fused actionsFreqsJac (5,5) + ctypes + fused jax/torch custom_vjp +
actionsFreqs routing. Validated vs c=False #1059 AD: values 2e-11, dOmega/dcoord
3e-7, end-to-end dOmega_r/dR grad-vs-FD 2.9e-10 (jax AND torch). 16 new tests +
30 numpy + 70 #1051 regression pass. numpy byte-identical. Angles = PR-B follow-up
(they add the current-position boundary term [f(ux)/sqrt(S(ux))]*dux/dcoord).
Worktree /var/tmp/staeckel-cnative, branch feat/staeckel-cnative-jac.
