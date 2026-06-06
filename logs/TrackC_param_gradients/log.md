# Track C (AD side) — differentiable potential parameters

## Approach
Parameter gradients (`dΦ/dθ`, forces and Hessians w.r.t. amplitude / scale lengths) require a
potential parameter supplied as a backend array (jax tracer / torch tensor) to **survive
construction**. Coordinate-gradient autodiff already worked (the `dΦ/dR == −F_R` identity); the
blocker for parameter gradients was the **constructor unit parser**.

## Result — galpy PR #903 (into feat/backends)
`conversion.parse_x_wrapper` (the shared guard behind `parse_length`/`parse_mass`/...) rejected
anything not a number / numpy array / astropy Quantity → a traced parameter raised
`RuntimeError: ... should either be a number or an astropy Quantity` at construction.

Fix: new `galpy.backend.is_backend_array(x)` (import-light, gated on the optional-dependency
flags; direct `isinstance` vs `jax.Array` / `torch.Tensor`) + accept such arrays in the guard,
passing them through **unscaled** (a backend array carries no units → treated as already in
internal units, like a plain float). **numpy / number / Quantity paths byte-identical** — the new
branch only triggers for genuine backend arrays. Tests: grad-vs-FD of Φ/F_R/F_z w.r.t. amp & scale
length (Plummer, Isochrone) under jax & torch; jit+vmap survival; numpy construction byte-identical.

Usage contract: pass coordinates as backend arrays too, so the namespace resolver follows the data.

## Dead-end worth noting
Initial param-grad tests differentiated the **private** `_evaluate`/`_Rforce` — these do NOT apply
`self._amp` (the public wrapper does), so amp-gradients were zero (jax) / a disconnected-graph
error (torch). Fix: parameter-gradient tests must use the **public** evaluators.

## Downstream
Unblocks figures **G3** (parameter-gradient correctness + gradient-descent fit) and **G5**
(autodiff overhead), and the parameter gradient through an ODE solve in **Track D** (orbit
integration pilot — see `logs/TrackD_orbit_integration/`). The fast C parameter-sensitivity path
(Track C C-side, `Pparam`) is separate and still future; this AD side is the immediate enabler and
the correctness reference for it.
