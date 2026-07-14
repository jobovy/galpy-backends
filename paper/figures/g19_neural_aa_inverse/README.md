# G18 — Neural action-angle inverse by differentiating through galpy

**Paper point:** galpy's action-angle machinery is now *differentiable* and
*jittable* under JAX and PyTorch, so it can sit inside a machine-learning
training loop. Here a small neural network learns the **inverse** action-angle
map of a realistic anharmonic potential — a map that has *no closed form* — by
backpropagating through galpy's differentiable *forward* action-angle
computation. This single example exercises all three capabilities the backend
project adds: **autodiff**, **jit**, and **ML in both backends**.

## Setup

- Potential: the 1-D vertical potential of `MWPotential2014` at `R0` (via
  `toVerticalPotential`) — genuinely **anharmonic** (unlike a harmonic
  oscillator, its inverse is not analytic).
- Forward map `F: (z, vz) -> (J_z, theta_z)` = `actionAngleVertical`.
  Backend-migrated: vectorized, `jax.jit`-able, and differentiable (backend
  `brentq` for the turning point + fixed-order Gauss–Legendre with a
  `z = zmax - t^2` turning-point substitution so reverse-mode AD is finite).
- Inverse we learn: `G_phi: (J_z, theta_z) -> (z, vz)`, a 3→2 MLP
  (3 hidden tanh layers, width 128), with the output squashed to a bounded box
  so training samples stay on bound orbits.

## Loss (hybrid, data + physics-informed)

```
L = <w * || G(J,theta) - (z,vz) ||^2>                            # data term
  + lambda * [ <((F_J(G)-J)/J)^2> + <1 - cos(F_theta(G) - theta)> ]   # cycle
    with  w = 1/(J + 0.015)
```

The data pairs are generated *by the jittable forward itself* (sample `(z,vz)`,
push through `F`); the **cycle term backpropagates through galpy's forward AA**,
so the gradient literally flows through the Gauss–Legendre action/angle
quadrature. Pure physics-informed (cycle only) also trains but is less stable —
the net can wander into high-energy orbits — so the data term keeps it on the
bound tori.

Two details that matter for the **small-action** tori (tiny orbits near the
plane, where `z, vz -> 0`): (1) the loss is **action-weighted** `w ~ 1/J` and the
action residual is **relative**, otherwise an unweighted MSE is dominated by the
big orbits and the tiny tori are systematically off; (2) the input features
include the **harmonic amplitude coordinates** `sqrt(J)*(cos theta, sin theta)`
alongside the raw `(J, cos theta, sin theta)`. With these the small-`J` tori are
the *most* accurate (RMS ~5e-3), and the error grows smoothly toward the largest
orbits (~2e-2). (Naively baking `z,vz = sqrt(J)*NN` into the *output* instead
hurts — it fights the anharmonic torus shape.)

## Backends

- `nn_vertical_inverse_jax.py` — `flax.linen` MLP + `optax` Adam, `jax.jit`
  training step, runs on GPU.
- `nn_vertical_inverse_torch.py` — `torch.nn` + `torch.optim.Adam`, eager.
- Identical physics and loss; only the autodiff engine differs.

## Results (validated against numpy `actionAngleVerticalInverse`)

The learned inverse reproduces the exact (numpy) inverse to RMS phase-space
error ~5e-3 (small `J_z`) growing smoothly to ~2e-2 (largest orbits), with no
systematic bias. See `g19_neural_aa_inverse.png`:
(a) learned tori vs exact; (b) JAX vs PyTorch training curves; (c) accuracy vs
`J_z`. `actionAngleVerticalInverse` is used **only** as ground truth — never
inside the autodiff graph.

## Reproduce

```bash
cd paper/figures/g18_neural_aa_inverse
# train each backend in its own process (avoids jax+torch both grabbing the GPU)
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
  PYTHONPATH=<galpy>:. python train_and_dump.py jax
USE_GPU=1 CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
  PYTHONPATH=<galpy>:. python train_and_dump.py torch
python make_figure.py         # -> g19_neural_aa_inverse.png
```

## Notes / next

- A harder showcase (no analytic inverse *at all*): learn the **Staeckel**
  action-angle inverse for `MWPotential2014` in the meridional plane
  (`actionAngleStaeckel`, differentiable via the C-native Jacobian vjp /
  `autograd.Function`, or the pure-Python `c=False` jittable path). Left as a
  follow-on figure.
