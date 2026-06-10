**Validity of galpy's variational equations: conservation laws, ground-truth
finite differences, and the 2D-reduction bridge.**
All panels use `MWPotential2014` and the C `dop853_c` integrator at
`rtol=atol=1e-12`; state-transition matrices (STMs) are built column-by-column
by propagating the canonical rectangular basis deviations with
`Orbit.integrate_dxdv(rectIn=rectOut=True)`.

**(a)** The Liouville invariant: $|\det M(t)-1|$ for the $6\times6$ STM of a
generic 3D orbit and the $4\times4$ STM of a planar orbit, over $t=0$–$100$
($\approx3.6\,$Gyr for $R_0=8\,$kpc, $v_0=220\,$km s$^{-1}$). The defect is
pure accumulated integration error: max $2.6\times10^{-8}$ (3D) and
$1.3\times10^{-6}$ (planar; its deviation is propagated through an extra
cylindrical–rectangular transform layer, hence the slightly larger floor).
**(b)** The symplecticity defect $\|M^{\mathsf T}\Omega M-\Omega\|_{\max}$ in
canonical rectangular coordinates, same orbits: max $1.0\times10^{-6}$ (3D)
and $1.3\times10^{-6}$ (planar). Symplecticity constrains all
$\binom{2n}{2}$ antisymmetric invariants rather than the single determinant,
but $\det M=1$ and $M^{\mathsf T}\Omega M=\Omega$ are still only *necessary*
conditions — they hold for the flow of *any* symmetric Hessian, right or
wrong. **(c)** is therefore the ground-truth test: at $t=10$, STM columns
$M(t)e_i$ are compared against finite differences of independently integrated
perturbed orbits — which exercise only the (separately validated) forces,
never the Hessian — as a function of the FD step $\varepsilon$. The residual
traces the classic V: the forward-difference truncation arm
$\propto\varepsilon$, the central-difference arm $\propto\varepsilon^2$, and
the roundoff arm $\propto\varepsilon^{-1}$ rising as $\varepsilon$ shrinks
below the orbit-integration accuracy. The STM agrees with the best central
difference at the V's floor, $1.3\times10^{-8}$ at $\varepsilon=10^{-6}$ —
i.e. the analytically propagated derivative is at least as accurate as the
best finite difference achievable at these tolerances, which is what makes
the variational route the preferred derivative. **(d)** The 2D-reduction
bridge: for a planar initial condition, the $(x,y,v_x,v_y)$ block of the 3D
STM is compared with the independently implemented planar STM at every output
time; max difference $1.0\times10^{-8}$ over $t=0$–$100$, with *exactly zero*
out-of-plane leakage ($\partial^2\Phi/\partial R\partial z =
\partial^2\Phi/\partial z\partial\phi = 0$ at $z=0$ for the $z$-symmetric
potential, so in-plane deviations stay in-plane to machine precision).

Measured against the `feat/pvar-lyapunov` galpy build (commit `7ca49a1b` plus
a local fix of the base-orbit $z$-column packing in
`integrateFullOrbit_dxdv`, which does not affect the deviation columns used
here); standalone measure script in `measure/`, machine- and
worktree-stamped data in `data/`, plot byte-stable from the JSON.
