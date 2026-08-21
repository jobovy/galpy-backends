# Handoff — Staeckel action quadrature carries a systematic ~1e-4 error

**From:** the adiabatic-quadrature work (gh#1354 issue / gh#1356 fix).
**To:** whoever is working on `actionAngleStaeckel`.
**Status:** measured and documented, **not actioned** — no Staeckel code touched, no PR opened.
**Date:** 2026-08-20.
**Background/derivation:** [`appendices/F_adiabatic_action_quadrature.md`](appendices/F_adiabatic_action_quadrature.md) (§10 is Staeckel-specific).

---

## Claim

galpy's Staeckel actions carry a **systematic ~4.6e-4 ($J_R$) / ~1.45e-4 ($J_z$)** relative
error at the default quadrature order, in **both** the C and the Python/backend paths. It is
the same defect just fixed for `actionAngleAdiabatic`.

## Root cause

The integration limits are *defined* by the integrand vanishing: $F(u_{\min})=F(u_{\max})=0$
and $F(v_{\min})=0$. At a generic turning point $F$ has a **simple zero**, so $\sqrt{F}$ has a
**square-root branch point exactly at the limit**.

Gauss–Legendre is geometrically convergent only for integrands analytic on the interval.
Against an algebraic endpoint $(1-x)^{\alpha}$ the classical result (Donaldson & Elliott 1972;
Trefethen, *SIAM Review* 2008) gives

$$\varepsilon_n = O\!\left(n^{-(2\alpha+2)}\right) \;\xrightarrow{\ \alpha=1/2\ }\; O(n^{-3}).$$

## Sites

| file:line | integral | endpoints |
|---|---|---|
| `actionAngle_c_ext/actionAngleStaeckel.c:590` | `gsl_integration_glfixed(JRInt, umin, umax, T)` | **both** are turning points → two-sided |
| `actionAngle_c_ext/actionAngleStaeckel.c:657` | `gsl_integration_glfixed(JzInt, vmin, M_PI/2, T)` | **only `vmin`**; $v=\pi/2$ is the equatorial plane → one-sided |
| `actionAngleStaeckel.py:234` `_staeckel_gl_action` | the backend mirror | same |

## The reasoning to overturn

Stated explicitly at `actionAngleStaeckel.py:92`:

> *"Matches the C gsl_glfixed exactly: plain GL of `order` points over `[umin,umax]`/`[vmin,pi/2]`
> (the J integrands **VANISH** at the turning points, so no t^2-substitution is needed ...)"*

Vanishing is not sufficient — it vanishes like a *square root*, which is a branch point, not an
analytic zero. The identical split exists in the adiabatic C: the **derivative** routines were
correctly substituted, the **action** routines were not, on exactly this reasoning.

## A trap worth knowing before you measure

Comparing `c=True` against `c=False` shows agreement to **1e-12** — and that proves **nothing**,
because the Python path *deliberately* reproduces C's rule. Two implementations of the same
quadrature agreeing is not evidence of accuracy. (This is the "the reference can be the broken
one" failure mode; it nearly fooled me.)

The discriminator is **convergence in the order**.

## Evidence

Convergence, MWPotential2014, $\delta=0.45$, one point:

| order | $\Delta J_R$ | $\Delta J_z$ | ratio |
|---:|---:|---:|---:|
| 10→20 | 3.97e-04 | 1.25e-04 | — |
| 20→40 | 5.32e-05 | 1.68e-05 | 7.46 |
| 40→80 | 6.90e-06 | 2.18e-06 | 7.71 |
| 80→160 | 8.80e-07 | 2.78e-07 | 7.84 |
| 160→320 | 1.11e-07 | 3.51e-08 | 7.93 |
| 320→640 | 1.40e-08 | 4.41e-09 | 7.93 |

Ratio $\to 8 = 2^3$, confirming $O(n^{-3})$.

Error of the **default order 10** against order 1280, four phase-space points:

| $(R,v_R,v_T,z,v_z)$ | $J_R$ err | $J_z$ err |
|---|---:|---:|
| (1.1, 0.05, 0.9, 0.15, 0.12) | 4.576e-04 | 1.443e-04 |
| (0.7, −0.15, 1.05, 0.05, −0.2) | 4.585e-04 | 1.451e-04 |
| (1.6, 0.2, 0.7, 0.3, 0.1) | 4.672e-04 | 1.463e-04 |
| (0.9, 0.0, 0.95, 0.25, 0.18) | 4.569e-04 | 1.454e-04 |

Tight spread → **systematic**, not point-specific. (For comparison, the adiabatic defect was
4.6e-4 / 1.4e-4 — the same rule on the same class of integrand gives the same relative error.)

**Reproduce:** `actionAngleStaeckel(pot=MWPotential2014, delta=0.45, c=False, order=N)` and
sweep `N`.

## The fix shape — the machinery already exists in your module

- **One-sided ($J_z$):** $v = v_{\min} + (\pi/2 - v_{\min})\,t^{2}$ makes the integrand
  $\propto t^{2}$, analytic. This is exactly what `_staeckel_t2_action`
  (`actionAngleStaeckel.py:253`) already does.
- **Two-sided ($J_R$):** either $u = c - \rho\cos\theta$ with
  $c=\tfrac{u_{\max}+u_{\min}}2,\ \rho=\tfrac{u_{\max}-u_{\min}}2$, or split at a midpoint into
  two one-sided $t^2$ panels — which is what the C **derivative** routines already do
  (`actionAngleStaeckel.c:1098-1112`, `glfixed(dJRInt, 0., mid, T)` twice).

So the derivative routines already implement, in both forms, precisely the regularization the
action routines need.

## Cost

The substitution is **free** — same node count, only the abscissae change.

If you then raise the order: in the adiabatic case that cost **+33%, not +100%**, because the
turning-point root-find is order-independent and dominates. Solving $F+Q=2.48,\ F+2Q=3.30$ gave
quadrature $\approx33\%$ of the call and root-finding $\approx67\%$. **Re-measure for Staeckel**
rather than assuming the same split.

## Coordination points

1. This **changes `c=True` output at ~1e-4** → not byte-identical. gh#1356's pattern: a
   `HISTORY.txt` entry, and two separate commits (substitution = free; order bump = +33%) so the
   cost decision is reviewable and revertable on its own.
2. The **backend/jax/torch path deliberately matches C bit-for-bit**, so C and backend must move
   in lockstep or the parity tests break.
3. It may interact with the Staeckel **gradient** work (gh#1047 / gh#1051, the t²-donor graft) —
   the action integrands are shared.

## What I did *not* verify

- I measured the **Python** path directly and inferred C via the documented parity plus the
  1e-12 agreement. I did not independently instrument the C.
- I did not check whether the `if (out <= 0.) return 0.` clamp inside
  `JRStaeckelIntegrand`/`JzStaeckelIntegrand` interacts with a substitution near the endpoint.
  Worth a look — with the endpoint regularized, the clamp may become unnecessary, or may mask a
  sign issue at the node closest to the turning point.

## Precedent to copy

gh#1354 (issue, with the same style of measurement) and gh#1356 (fix: substitution + order,
in two separate commits, with a regression test A/B-checked against an unmodified build).
