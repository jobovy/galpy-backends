# Track C — C parameter sensitivity (∂f/∂θ → parameter gradients), → main

Branch: `feat/param-sensitivity` (off main; standalone C, decoupled from backends).
Detail in master plan §"Track C".

Augment the variational system with `P=∂x/∂θ`, `Ṗ=J·P+∂f/∂θ`, `P(0)=0`; backward
`θ̄=Pᵀλ`. Add a C interface for `∂(force)/∂(param)` per potential.

Note: Track D's first-class in-backend integrators already give parameter gradients
for **all** potentials via autodiff of the namespace-swapped forces — that is the
immediate full-coverage path and the correctness reference. This C track is the
**fast** path, rolled out family-by-family (Pparam-pot.x, worktrees off Pparam-design),
each validated vs in-backend AD + finite differences.
