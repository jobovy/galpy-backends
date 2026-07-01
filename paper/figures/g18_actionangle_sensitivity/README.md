# G18 — Differentiable action-angle: sensitivity of the actions (CPU, float64)

The action-angle analogue of **G7** (the "derivatives show-off", but for the
action-angle backend instead of the orbit integrator). The action-angle map
`J(x) = (J_r, L_z, J_z)` is now backend-agnostic, so it runs — and, crucially,
*differentiates* — under jax/torch. This figure overlays the autodiff
sensitivities `dJ/d(·)` (`jax.grad` straight through the action map) on top of the
action curves and validates every one against a central finite difference. The
autodiff lines sit exactly on the finite-difference markers.

## Panels

**(a) Initial-condition sensitivity.** A single isochrone orbit (scaled so
`vc(1)=1`) is swept in its radial velocity `v_R` at fixed `R, v_T, z, v_z`. Left
axis: the radial and vertical actions `J_r(v_R)`, `J_z(v_R)`. Right axis: the
autodiff sensitivities `dJ_r/dv_R`, `dJ_z/dv_R` (dashed) with open-circle
finite-difference markers. `J_r` is a clean parabola in `v_R` (its slope crosses
zero at the vertex — a nearly-circular radial orbit); `J_z` responds through the
total angular momentum.

**(b) Potential-parameter sensitivity** — the showpiece for *fitting a potential
to action data*. One orbit is held fixed and the isochrone scale `b` is swept
(amplitude fixed). Left axis: `J_r(b)`. Right axis: `dJ_r/db` (dashed) with
finite-difference markers. This `dJ_r/db` is exactly the gradient an inference
loop backpropagates when it tunes the potential to reproduce measured actions.

## Autodiff vs finite-difference residuals

Across the full sweeps (worst case over both actions / all points):

| panel | quantity | max `|AD−FD|` | rel-to-scale | worst pointwise rel |
|-------|----------|--------------|--------------|---------------------|
| (a)   | `dJ_r/dv_R` | 4.2e-10 | 1.5e-09 | 1.5e-07 |
| (a)   | `dJ_z/dv_R` | 7.3e-11 | 2.5e-09 | 6.3e-09 |
| (b)   | `dJ_r/db`   | 4.9e-10 | 7.1e-10 | 4.4e-08 |

All far below the 1e-4 bar. The only place the *pointwise* relative residual rises
to ~1e-7 is the parabola vertex in (a), where `FD → 0` so the ratio is dominated
by the finite-difference truncation floor — the autodiff itself is exact to
round-off (it agrees with FD to ~1e-10 absolute everywhere).

## Method choice — why isochrone, not Staeckel

`actionAngleIsochrone` has **closed-form** actions (no quadrature), so `jax.grad`
through the action map is machine-exact and each panel is a clean `autodiff == FD`
statement that also renders byte-stably.

`actionAngleStaeckel` is the branchier flagship — its actions flow through
turning-point root-finds + Gauss-Legendre quadrature and *do* run under the
backends — but it is **not** used for the autodiff overlay here, on purpose. On
this build the standalone-actions `c=False` path uses a *plain* GL quadrature of
the `sqrt`-vanishing integrand, chosen to reproduce the C action bit-for-bit.
`jax.grad` through that plain-GL integral converges to `dJ/d(·)` only
*algebraically* (the differentiated integrand `~ (dS/dθ)/(2√S)` is singular at the
turning points where `S→0`): empirically ~5% off at the C-matching order 10, and
still ~0.2% off at order 300 — it cannot reach 1e-4 by raising the order. The
*accurate* derivative integrands (the `t²`-substituted `dJ/d(E,L_z,I_3)` panels)
live on the **frequency** path, and differentiating the frequencies is worse still
(a second-derivative singularity). A `<1e-4` `autodiff == FD` *Staeckel* statement
therefore waits on exposing those derivative integrands as the action gradient; it
is a known follow-up, not a bug in this figure. The Staeckel action *values*
themselves are exact (C-parity) and are showcased for throughput in **G17**.

## Reproduce

```
cd galpy-jax
PYTHONPATH=/var/tmp/pdf1 CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \
    python -m paper.figures._harness run    g18_actionangle_sensitivity
PYTHONPATH=/var/tmp/pdf1 CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \
    python -m paper.figures._harness verify g18_actionangle_sensitivity
```

Pure-CPU float64 and deterministic (jax pinned to the CPU device, `x64` enabled),
so the PDF is byte-stable. Needs jax.
