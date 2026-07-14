# G19 — Neural inverse of the Staeckel action-angle map (no analytic inverse)

**Paper point:** galpy's Staeckel action-angle machinery for a *realistic* Milky
Way potential (`MWPotential2014`) is differentiable and jittable, so a neural
network can learn its **inverse** — a map for which **no closed form exists** —
purely by backpropagating through galpy's forward action-angle computation. This
is the harder sibling of G18 (there the 1D vertical inverse could be checked
against `actionAngleVerticalInverse`; here there is *nothing* to check against,
so the inverse *is* the deliverable).

## Setup (meridional plane, fixed L_z)

- Potential: `MWPotential2014`; `delta` from `estimateDeltaStaeckel`.
- Forward `F: (R, vR, z, vz) -> (J_R, J_z, theta_R, theta_z)` at fixed
  `L_z` (`vT = L_z/R`, `phi = 0`) via `actionAngleStaeckel(..., c=False)` —
  the pure-Python, **jittable** path (chosen because jit of the Python-backend
  quadrature is faster for the training loop than the C-callback custom-VJP, and
  it supports higher-order AD). ~0.5 s / 512-orbit step on the GPU.
- Inverse `G_phi: (J_R, J_z, theta_R, theta_z) -> (R, vR, z, vz)`, a 4-hidden-
  layer MLP with `sqrt(J)*(cos,sin)` amplitude features and a bounded output box.

## Loss & stability

`L = <w * ||G - (R,vR,z,vz)||^2> + lambda(step) * <w * cycle>` with
`w = 1/(J_R+J_z+0.02)` (action-weighting so small near-circular / near-planar
orbits are fit as precisely as big ones). The cycle term
`(F_J(G)-J)^2 + (1-cos(F_theta(G)-theta))` **backpropagates through the Staeckel
quadratures** — the autodiff showcase.

Two things keep it stable (the physics term can otherwise propose a degenerate /
unbound orbit whose Staeckel gradient is NaN): a **supervised warmup**
(`lambda` ramps 0->1 over the first 400 steps) and an `optax.zero_nans()`
backstop plus a finite-mask on the per-sample cycle residual.

## Validation (there is no ground-truth inverse)

- **Orbit reconstruction** (panel a): integrate real orbits in `MWPotential2014`,
  read off their `(J_R, J_z)` and angle evolution `theta(t)`, and reconstruct the
  meridional `(R, z)` track with `G` — the neural inverse traces the integrated
  orbit.
- **Round-trip closure** (panel c): on in-distribution tori,
  `F_J(G(J,theta)) ≈ J` (the inverse composed with the forward is the identity).
- Panel (b): the training curve (data + physics-cycle), backprop through Staeckel.

## Reproduce

```bash
cd paper/figures/g19_neural_staeckel_inverse
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
  PYTHONPATH=<galpy>:. python make_staeckel_figure.py   # -> g20_neural_staeckel_inverse.png
```

## Notes

- Torch works too (same `actionAngleStaeckel` autograd path) but eager torch is
  much slower on this 2D-quadrature forward, so the headline run is JAX.
- The full 6D problem (free `L_z`, angle `theta_phi`) is a straightforward
  extension of this fixed-`L_z` meridional demo.
