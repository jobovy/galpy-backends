# gD4 — derivatives through spherical-DF construction and sampling

**Caption.** Autodiff through the *construction* and the *sampling* of galpy's
spherical distribution functions. Top: King models with $M=1$, $r_t=10$. (a)
Density profiles for $W_0=3,5,7,9$. (b) $\partial\ln\rho/\partial W_0$ at fixed
$(M,r_t)$, obtained by differentiating through the King ODE solve, eagerly (scipy
DOP853 + forward sensitivities; solid) and under `jax.jit`, where the ODE is solved
in-graph with diffrax (dashed), against a central finite difference that rebuilds
the DF on the plain numpy path (circles). (c) The AD–FD difference vs step $h$
reaches $5\times10^{-8}$–$10^{-7}$ (eager and jit alike). Bottom: an Osipkov–Merritt DF ($r_a=1.5$)
of a Hernquist sphere is constructed *and* sampled ($4\times10^6$ particles)
inside one jitted function of the scale radius $a$, with a fixed PRNG key so
that every particle is a deterministic function of $a$ (common random numbers).
(d) $\sigma_r$ in bins of enclosed mass, compared with the Jeans-equation
expectation for the drawn radii. (e) $d\sigma_r/da$ from `jax.jvp` through the
sampler (dots) agrees with a finite difference of the same sample (crosses; to
$6\times10^{-8}$) and, within Monte-Carlo noise ($|{\rm pull}|\le2.1$; inset),
with the Jeans expectation, so the pathwise derivative is an unbiased estimator.
(f) AD vs FD vs $h$: the per-particle median follows $h^2$ down to $5\times10^{-11}$;
a fraction $\propto h$ of particles sits near a kink of the tabulated inverse CDF,
which makes the binned FD converge linearly in $h$.

## Key numbers (CPU, float64)
| quantity | AD vs FD (best h) | other check |
|---|---|---|
| King ∂lnρ/∂W0, eager (scipy+graft) | 4.5e-8 – 1.2e-7 (max over r, abs) | jit vs eager ≤ 2.3e-7; torch vs jax eager ≤ 1.3e-14 |
| King ∂lnρ/∂W0, jax.jit (diffrax) | 6.7e-8 – 1.9e-7 | — |
| OM sample, per particle dv_r/da | median 4.9e-11 (h=1e-5) | 99.9th pct 3.8e-7 (h=1e-5): kinks |
| OM sample, binned dσ_r/da | 6.5e-8 (h=1e-6); 2.2e-6 (h=1e-4) | sample vs Jeans: max pull 2.1, median 1.1 |

Reproduce: delete `data/gD4_spherical_df_param.json` and run
`python -m paper.figures._harness run gD4_spherical_df_param` (CPU only, ~12 min on 4 cores).
