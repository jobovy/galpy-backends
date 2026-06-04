# Track E — Action-angle (ALL methods, incl. C-backed)

Branch: `feat/backends`. Detail in master plan §"Track E".

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
