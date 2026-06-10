**The dissipative phase-volume law: $\det M(t)=\exp\int_0^t
\mathrm{tr}\,(\partial F/\partial v)\,\mathrm dt'$ — the non-conservative
replacement for Liouville's theorem.**
For a velocity-dependent force the variational Jacobian is the general
$J=\begin{pmatrix}0&I\\K+\partial F/\partial x&\partial F/\partial
v\end{pmatrix}$: symplecticity and $\det M=1$ fail *by construction*, and the
Abel–Jacobi–Liouville identity instead fixes the volume contraction through
the trace of the velocity block. Both sides are computed independently: the
left side is the determinant of the $6\times6$ STM built from the six
canonical deviations via `Orbit.integrate_dxdv` (`dopr54_c`,
`rtol=atol=1e-12`); the right side evaluates $\mathrm{tr}\,(\partial
F/\partial v)$ along the orbit with the C-extension rectangular force
Jacobian (`evalRectForceJacobian` — the same analytic Jacobian the C
variational integrator consumes) and integrates it with the trapezoidal rule
on the output grid ($n_t=4001$ over $t=0$–$10$).

**(a)** A decaying orbit in `MWPotential2014` + dynamical friction
(`GMs=0.008`, `rhm=0`, identical for both forces). Chandrasekhar friction
sinks the orbit from $r\approx1.0$ to $r\approx0.05$ and contracts phase
volume by over five decades, $\det M(10)=6.1\times10^{-6}$; the dashed
trace-integral prediction is overplotted and indistinguishable. FDM friction
at the *same* `GMs` (with $m\hbar^{-1}=10$, verified to stay in the FDM
dispersion regime with quantum-pressure suppression
$C_{\rm FDM}/C_{\rm CDM}\in[0.48,0.59]$ along the whole orbit) contracts far
more weakly — $\det M(10)=0.52$, $r\to0.79$ — directly visualizing the
suppressed FDM drag at fixed satellite mass. **(b)** The relative difference
between the two sides: max $5.7\times10^{-6}$ (Chandrasekhar) and
$3.5\times10^{-7}$ (FDM). The Chandrasekhar residual is dominated by the
$O(h^2)$ trapezoidal quadrature of the rapidly growing trace during the final
plunge, not by the variational integration itself (at $n_t=2001$ it is
$2.3\times10^{-5}$, scaling as expected), so the $\sim10^{-5}$ agreement is a
quadrature floor, not an accuracy limit of the STM.

Measured against the `feat/pvar-dissip-struct` build (commit `43451e7a`,
Chandrasekhar) and the `feat/pvar-tail-fdm` build (commit `36615802`, FDM);
standalone measure script in `measure/` (run once per worktree), machine- and
worktree-stamped data in `data/`, plot byte-stable from the JSONs.
