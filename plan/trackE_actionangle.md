# Track E — Action-angle (ALL methods, incl. C-backed)

Branch: `feat/backends`. Detail in master plan §"Track E".

## Strategic refinements (user, 2026-06-24)
- **Shared root-find helper (task #86).** MANY AA modules root-find (IsochroneInverse
  Kepler-eq, Vertical xmax, Spherical rperi/rap, Staeckel umin/umax/vmin, ...). Build
  ONE reusable backend Newton/bisection + implicit-diff helper in
  `galpy.backend.optimize` and reuse it everywhere — don't write per-class solvers.
  IsochroneInverse (the current closed-form-tail item) is the first consumer/reference.
- **C-only freqs/angles → pure Python on MAIN first (task #87).** `actionAngleStaeckel.
  _actionsFreqs`/`_actionsFreqsAngles` are **C-only** (raise NotImplementedError without
  C; the Leibniz integrands live only in `actionAngleStaeckel.c:1883-1976`). To put
  Staeckel on the backends we must FIRST implement these in pure Python. That work is
  **backend-independent** → it ships straight to **main** (own feature branch), benefiting
  numpy users; THEN `feat/backends` rebases and namespace-swaps it. This supersedes the
  "just wrap the C with custom_vjp" plan for Staeckel: prefer a true backend-native Python
  path (custom_vjp can still wrap the C as the fast path). Mirrors the variational3d→main
  →feat/backends pattern. (Check Adiabatic: its freqs route via Python Spherical+Vertical,
  so likely already Python — verify; do the same if C-only.)

- **Paa-python:** namespace-swap exact pure-Python methods — Isochrone, Harmonic,
  Vertical (closed-form + fix masked writes like `Omegaphi[indx]*=-1` → `xp.where`);
  Spherical (backend-agnostic fixed-iteration root-find via implicit diff + fixed-order
  Gauss-Legendre quadrature, vectorized). Correctness reference + all-orders fallback.
- **Paa-staeckel-c:** wrap C Staeckel (and Adiabatic) with custom_vjp/autograd.Function.
  **Staeckel C already has the Leibniz derivative integrands** (dJR/dE, dJR/dLz, dJR/dI3,
  dJz/…, actionAngleStaeckel.c:1883-1976) and frequencies from them → backward reuses
  them, composed with ∂(E,Lz,I3)/∂(x,v) and (Track C) ∂/∂θ. Adiabatic C lacks them
  (frequencies via Python Spherical+Vertical) → add integrands or route through Python.
- **Paa-analytic:** IFT turning-point rule `dr*/dθ=−(∂g/∂θ)/(∂g/∂r)` (√-branch-point
  regularized via the existing `r=r_peri+t²` substitution) + Leibniz action rule
  `dJ/dθ=(1/π)∫∂p_r/∂θ dr` (boundary terms vanish). Mirror the C Staeckel structure.
- **Paa-grid-approx:** backend-agnostic interpolation for StaeckelGrid/AdiabaticGrid;
  IsochroneApprox via the orbit STM path.

**OUT OF SCOPE:** `actionAngleTorus` (external C++ library `actionAngleTorus_c_ext/torus/`)
— left numpy-only; streams differentiate via IsochroneApprox/Staeckel/Spherical instead.

**2026-09-29 status.** actionAngleSpherical's backend path solves main's
relative radial problem (#1527/#1553 ported, #1594 merged): epicycle branch,
small-orbit and near-circular relative potentials, main's turning-point rules,
u = 1/r azimuthal integrals. Parameter gradients vs h-converged FD 5e-9..1e-6.
Next: Orbit.rperi/rap/zmax(analytic=True) through the backend C-STM.

**2026-10-06 status — actionAngleVerticalInverse is ON the backends.** It had been
out of scope (`_reject_backend`) while under development; Jo brought it in once
stable (2026-10-04). Stack #1629: #1627 → #1628 → #1630, plus #1631 (CUDA).

- **#1627 — momentum-matched evaluation.** Tables (Hermite splines in J: D_m(J),
  K(J), E(J)) are scipy PPolys evaluated with `eval_ppoly` (values + exact
  J-derivatives). Angle → anomaly by backend `brentq` on the monotone angle
  relation over [0, 2π] (Newton polish = implicit-function gradient). Shared
  kernel `_mm_kernel` (numpy byte-identical). **Dead end:** autodiff of
  p = 2J sin²η η′/(x_max sin τ) cancels catastrophically where sin τ ≈ 1e-16
  (τ = π): dv/dθ off by 8e-5 at θ = π/2. Fix: sin η/sin τ = cos δ + cos τ ·
  sinc δ · q(τ), with q = Σ D_m sin(mτ)/sin τ = cos-series via
  sin(mτ)/sin τ = 2 Σ_{odd l<m} cos(lτ) (tail-summed coefficients). Values vs
  numpy ≤ 7.9e-15; d(x,v)/dJ 1.3e-11, d(x,v)/dθ 1.7e-12 (FD floor), {x,v} = 1.
- **#1628 — native construction, differentiable in potential parameters.**
  x_max by backend brentq; spectral sweep (FFT antiderivative); anomaly map by
  batched Gauss–Newton from D=0 (resolved tori ≤ 4 iterations, under-resolved
  ~11; 14 fixed) without the gradient + ONE differentiable step (IFT); slopes by
  normal equations; `cubic_hermite_coeffs` (new, == scipy CubicHermiteSpline.c).
  Whole pipeline traces under jax.jit; jit(jacfwd) == eager to 1e-12.
  d(x,v,Ω)/d(param) vs FD of the backend construction ≤ 9e-11 (numpy's calcxmax
  roots, E−Φ(x_max) ~ 2e-14, are too noisy for a sharper reference than ~1e-8;
  backend-built vs numpy-built values 7e-12 for that reason — backend more
  accurate). **Found:** an unbound energy (E > Φ(∞)) made the backend bracket run
  off to x_max = 1e38 silently → residual check → NaN action → same RuntimeError
  as numpy. Bottom tori use the exact √Φ″(0) (numpy approximates by the torus
  at E+1e-5 when Φ(0) ≠ 0; galpy's own potentials all have Φ(0)=0).
- **#1630 — legacy modes evaluate natively** (momentum_matched=False, polynomial
  and exact point transforms, ± setup_interp); construction stays numpy (a legacy
  mode can't take a backend-parameter potential). `_SplineXP` wraps setup_interp's
  1D splines; nSn/dSndJ/pt_coeffs via backend map_coordinates (scipy 'constant'
  == 'mirror' for in-grid queries). map_coordinates: **order 5** (quintic
  B-spline, 6 taps) + **mirror fold generalized to period 2(n−1)** (a 2-column
  table reflected a tap twice → index out of range; affects every backend
  user). Legacy values vs numpy ≤ 4.8e-14; derivatives ≤ 5e-11.
  E=300 isothermal: numpy calcxmax gives up (RuntimeError), backend finds the
  turning point → "not converged" warning; test expects each.
- **#1631 — CUDA.** Library: the energy grid went to numpy as CUDA tensors →
  `to_host`. Tests: numpy-side arithmetic on outputs → `as_numpy`. On the TITAN
  Xp all 49 main-suite VI tests pass under torch and jax; dedicated file (52) too.

Open: backend-native (differentiable) legacy construction — deferred by Jo.
