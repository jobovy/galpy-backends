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
