# gA1 — Staeckel actions differentiated w.r.t. a potential parameter (CPU, float64)

`jax.grad` straight through galpy's pure-Python Staeckel action code
(`actionAngleStaeckel(..., c=False)`) with respect to the **halo flattening** `q`
of `LogarithmicHaloPotential(normalize=1., q=q)`, validated against central
finite differences. The potential *and* the `actionAngleStaeckel` instance
(`delta=0.45`, GL order 10) are constructed inside the differentiated function,
so `q` flows through the normalisation, the Staeckel setup `(E, L_z, I_3)`, the
turning-point root-finds and the action quadrature. The finite differences use
the numpy path and also rebuild everything at `q ± h` — nothing is frozen.

Orbit: `R=1, vR=0.1, vT=1.1, z=0.1, vz=0.05, phi=0` (natural units).

## Panels
- **(a)** `J_R(q)`, `J_z(q)` over `q ∈ [0.7, 1.1]` (30 points).
- **(b)** `∂J_R/∂q`, `∂J_z/∂q`: `jax.grad` (lines) vs numpy central FD at
  `h=1e-5` (circles, every 5th point).
- **(c)** `|AD−FD|/|FD|` at `q=0.9` vs the FD step `h = 1e-2 … 1e-7`: the
  `h²` truncation branch, the round-off branch, and AD at the floor of the V.

## Measured (data/gA1_staeckel_param_sensitivity.json)
| quantity | value |
|---|---|
| `∂J_R/∂q` at q=0.9 (AD) | −1.78155880479e-3 |
| `∂J_z/∂q` at q=0.9 (AD) | −5.02186420212e-3 |
| best `|AD−FD|/|FD|`, `J_R` | 1.1e-10 at h=1e-6 |
| best `|AD−FD|/|FD|`, `J_z` | 1.3e-10 at h=1e-5 |
| sweep markers (h=1e-5), worst | 2.3e-9 (`J_R`), 3.3e-10 (`J_z`) |
| jax-vs-numpy forward action parity along the sweep | 1.3e-15 |
| `jax.jit(value_and_grad)` vs eager | ≤ 9e-17 |

The gradient is accurate because the backend path grafts the gradient of a
`t²`-substituted, turning-point-regular donor quadrature onto the plain-GL
(C-parity) action value (`_staeckel_jr_jz` / `_staeckel_t2_action`); this is
what g18's README said was still missing for a <1e-4 Staeckel AD==FD statement.

Not plotted (in `checks`): the frequency gradients `∂Ω_R/∂q`, `∂Ω_z/∂q` agree with
FD only to ~5e-7–2e-6 (best h=1e-4), because the numpy frequency *values* carry
~1e-11 relative noise (jax vs numpy `Ω_R` differ by 1.4e-11) that FD amplifies.

## Reproduce
```
cd galpy-jax/paper/figures/gA1_staeckel_param_sensitivity/measure
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu PYTHONPATH=/var/tmp/fbp \
    systemd-run --user --scope -p MemoryMax=32G python measure.py   # ~4.5 min
cd galpy-jax
python -m paper.figures._harness run    gA1_staeckel_param_sensitivity
python -m paper.figures._harness verify gA1_staeckel_param_sensitivity
```
Pattern A: the measure step writes the JSON (galpy sha 9a877712e, jax 0.10.1);
the plot is byte-stable from it. The run must use `systemd-run MemoryMax`, not
`ulimit -v`: jax's XLA CPU compiler reserves a large virtual address space and
aborts with `std::bad_alloc` under `ulimit -v 32G` (resident use is small).
