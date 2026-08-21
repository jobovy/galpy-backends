# Appendix F — Why the C adiabatic actions lost ~4 digits (gh#1354 / gh#1356)

**TL;DR.** `calcJzAdiabatic` and `calcJRAdiabatic` applied a fixed-order Gauss–Legendre rule
directly to integrands that vanish like a *square root* at the ends of the integration
interval. Gauss–Legendre is spectrally accurate only for integrands that are analytic on the
interval; against an algebraic endpoint branch point it degrades to **algebraic** convergence,
here $O(n^{-3})$. At the order-10 default that is a relative error of $\sim 10^{-4}$, while the
pure-Python path (adaptive quadrature) is accurate to $\sim 10^{-14}$. A change of variable
that maps the branch point to an analytic point restores geometric convergence **at the same
number of nodes, i.e. for free**.

---

## 1. The integrals

In the adiabatic approximation the vertical and radial motions decouple. Write the vertical
potential at fixed $R$ as

$$\Phi_z(R,z) \;=\; \Phi(R,z)-\Phi(R,0),\qquad E_z \;=\; \Phi_z(R,z)+\tfrac12 v_z^2 .$$

galpy's C computes (matching `actionAngleAdiabatic.c` exactly, including the prefactors):

$$J_z \;=\; \frac{2\sqrt2}{\pi}\int_0^{z_{\max}}\!\sqrt{F_z(z)}\;\mathrm dz,
\qquad F_z(z)\;=\;E_z-\Phi_z(R,z),$$

$$J_R \;=\; \frac{\sqrt2}{\pi}\int_{r_p}^{r_a}\!\sqrt{F_R(r)}\;\mathrm dr,
\qquad F_R(r)\;=\;E_R-\Phi(r,0)-\frac{L_z^{2}}{2r^{2}} .$$

The limits are *defined* as the turning points: $F_z(z_{\max})=0$, and $F_R(r_p)=F_R(r_a)=0$.
That definition is the whole problem.

## 2. The endpoint structure

At a generic (non-degenerate) turning point the derivative does not vanish, so $F$ has a
**simple zero** there. Taylor-expanding about $z_{\max}$:

$$F_z(z)\;=\;(z_{\max}-z)\,h(z),\qquad h \text{ analytic},\; h(z_{\max})=|F_z'(z_{\max})|\neq 0 .$$

Hence the integrand behaves like

$$\sqrt{F_z(z)}\;\sim\;C\,(z_{\max}-z)^{1/2}\qquad (z\to z_{\max}^-).$$

It is continuous and vanishing — but **not analytic**: its derivative diverges at the endpoint.
For $J_R$ the same happens at *both* limits, $ (r-r_p)^{1/2}$ and $(r_a-r)^{1/2}$.

## 3. Why Gauss–Legendre degrades here

An $n$-point Gauss–Legendre rule is exact for polynomials of degree $\le 2n-1$. For an
integrand analytic on a neighbourhood of the interval, the error decays **geometrically**,
$O(\rho^{-2n})$, where $\rho$ is the parameter of the largest Bernstein ellipse free of
singularities. That is the usual "spectral accuracy" intuition — and it does not apply here.

For an integrand with an algebraic endpoint singularity $(1-x)^{\alpha}$ times an analytic
function, with $\alpha>-1$ not a non-negative integer, the classical result (Donaldson &
Elliott 1972; see also Trefethen, *SIAM Review* 2008) is that the $n$-point Gauss–Legendre
error decays only **algebraically**:

$$\varepsilon_n \;=\; O\!\left(n^{-(2\alpha+2)}\right).$$

For a square-root endpoint, $\alpha=\tfrac12$, this predicts

$$\boxed{\varepsilon_n = O(n^{-3})}$$

i.e. a factor $2^3=8$ per doubling of the order. **The measurement confirms the exponent**
(MWPotential2014, one representative point, error against an independent reference):

| order $n$ | $J_z$ plain GL | ratio | $J_R$ plain GL | ratio |
|---:|---:|---:|---:|---:|
| 10 | 1.438e-04 | — | 4.605e-04 | — |
| 20 | 1.925e-05 | 7.5 | 6.147e-05 | 7.5 |
| 40 | 2.494e-06 | 7.7 | 7.958e-06 | 7.7 |
| 80 | 3.175e-07 | 7.9 | | |
| 160 | 4.005e-08 | 7.9 | | |
| 320 | 5.030e-09 | 8.0 | | |

Ratio $\to 8$, i.e. $n^{-3.0}$ — the predicted rate, not a coincidence of one potential.
So the ~$10^{-4}$ error at $n=10$ was not a tuning choice; it is what this rule *must* give on
this integrand.

## 4. The fix: move the branch point

We want a substitution after which the integrand is analytic at the endpoint. Both are
classical, and — the reason this was worth filing rather than debating — **galpy already used
both of them a few lines further down**, in the sibling derivative routines
`calcdJzAdiabatic` / `calcdJRAdiabatic`, which face a $1/\sqrt{F}$ singularity at the same
turning points.

### Vertical: $z=z_{\max}\sin\varphi$, $\varphi\in[0,\pi/2]$

Then $\mathrm dz = z_{\max}\cos\varphi\,\mathrm d\varphi$, and writing $\varphi=\pi/2-\epsilon$:

$$z_{\max}-z \;=\; z_{\max}(1-\sin\varphi)\;=\;z_{\max}(1-\cos\epsilon)\;\simeq\;\tfrac12 z_{\max}\epsilon^{2},$$

$$\sqrt{F_z}\;\simeq\;\sqrt{h\,z_{\max}/2}\;\epsilon,
\qquad \cos\varphi=\sin\epsilon\simeq\epsilon .$$

The transformed integrand is therefore $\propto \epsilon^{2}$ near the endpoint — an **even,
analytic** function of $\epsilon$. The square root has been absorbed: what was a branch point
is now an ordinary zero.

### Radial: $r = c - \rho\cos\theta$, $\theta\in[0,\pi]$, with $c=\frac{r_a+r_p}{2},\ \rho=\frac{r_a-r_p}{2}$

Then $\mathrm dr = \rho\sin\theta\,\mathrm d\theta$, and near $\theta=0$:

$$r-r_p=\rho(1-\cos\theta)\simeq\tfrac12\rho\theta^{2}
\;\Rightarrow\;\sqrt{F_R}\simeq\text{const}\cdot\theta,
\qquad \sin\theta\simeq\theta,$$

so the integrand is again $\propto\theta^{2}$, and symmetrically at $\theta=\pi$. This is the
Chebyshev-flavoured map that turns a two-sided square-root profile into an analytic one — the
same reason $\int\sqrt{1-x^2}\,\mathrm dx$ is trivial in the angle variable.

### Why it is free

The rule still evaluates $n$ nodes; only the abscissae change. The cost of these routines is
one potential evaluation per node, so **the substituted rule costs exactly what the plain rule
cost**. This is not a speed/accuracy trade.

## 5. What the rate becomes

With the endpoint regularized, convergence returns to geometric, $O(\rho^{-2n})$. The rate is
now set by the *potential's own* complex singularities — for a Miyamoto–Nagai disk the branch
point at $z=\pm i b$ — rather than by the endpoint, so it is fast but not instant:

| order $n$ | $J_z$ substituted | $J_R$ substituted |
|---:|---:|---:|
| 10 | 1.240e-07 | 2.147e-11 |
| 20 | 1.221e-12 | 1.403e-14 |
| 40 | 1.424e-15 | 1.403e-14 |

Roughly five orders of magnitude per doubling, versus a factor of eight — geometric versus
algebraic, exactly as the theory says.

## 6. Choosing the order

The single point above is favourable. Over 40 random phase-space points the **worst case** is
what matters:

| | $J_R$ max / median | $J_z$ max / median | cost / 200k actions |
|---|---|---|---|
| before (plain, $n{=}10$) | 7.88e-04 / 4.97e-04 | 1.52e-04 / 1.47e-04 | — |
| substituted, $n{=}10$ | 1.90e-06 / 4.43e-08 | 1.48e-06 / 2.56e-07 | 2.48 s |
| substituted, $n{=}20$ | **6.83e-10 / 1.14e-12** | **9.31e-10 / 5.49e-12** | 3.30 s |

Doubling the order costs **+33%, not +100%**, because the turning points $z_{\max}, r_p, r_a$
come from a *root solve*, whose cost is independent of the quadrature order. Solving

$$F+Q=2.48,\qquad F+2Q=3.30 \;\Longrightarrow\; Q\approx0.82\ \text{s},\;\; F\approx1.66\ \text{s},$$

so the quadrature is only ~33% of the call and root-finding ~67%. Order 40 would be +99% for
little further gain, which is why $n=20$ is the knee.

## 7. A useful non-worry: the turning points do not limit accuracy

One might expect the root solve for $z_{\max}$ to cap the achievable accuracy. It does not, and
the reason is pretty: by the Leibniz rule,

$$\frac{\partial J_z}{\partial z_{\max}}\;\propto\;\sqrt{F_z(z_{\max})}\;=\;0 ,$$

since $F_z$ vanishes at the turning point *by definition*. The action is **stationary** with
respect to the turning-point location, so an error $\delta$ in $z_{\max}$ enters only at
$O(\delta^{2})$. The quadrature was the only first-order error source — which is why fixing it
moved the answer by three orders of magnitude.

## 8. Provenance / how to reproduce

- Before changing anything, the existing rule was reproduced in numpy and matched the C output
  to **1.1e-14 ($J_z$) / 2.3e-14 ($J_R$)**, confirming that the rule being analysed is the one
  C actually ran.
- The reference is an independent adaptive quadrature (`scipy.integrate.quad`,
  `epsabs=epsrel=1e-14`) applied to the *substituted* integrand, cross-checked against galpy's
  own `c=False` path, which agrees with it to 7e-15. Three independent routes agree.
- Timings are min-of-7, measured interleaved A/B/A/B, in a tree asserted to import its own
  `galpy` and to have actually loaded `libgalpy` — an earlier attempt silently compared one
  build against itself because `python /abs/path/script.py` puts the *script's* directory on
  `sys.path`, not the cwd.

## 9. Status

- **gh#1354** — issue, with the measurements above.
- **gh#1356** — fix, in two separately reviewable commits: (1) the substitution, which costs
  nothing; (2) order $10\to20$, which costs +33%. Splitting them keeps the cost decision
  revertable on its own.
- This **changes `c=True` output** at the $10^{-4}$ level, so it is deliberately not
  byte-identical; `HISTORY.txt` records it. Anything built on adiabatic actions (e.g.
  `quasiisothermaldf`) inherits the improvement.
- The `gsl_function`/`params` machinery removed alongside was *not* dead code: GSL's callback
  interface forces state through a `void *params` pointer, and that state is written inside an
  OpenMP loop, hence the per-thread `malloc` and `omp_get_thread_num()` indexing. Inlining the
  integrand removes the callback, so the state becomes loop-body locals that OpenMP privatizes
  per iteration. Verified rather than argued: $J_R,J_z$ are **bitwise identical** (sha256) at
  `OMP_NUM_THREADS` 1/2/8/16, scaling 10.04 s → 1.15 s.

## 10. Does the same bug sit in Staeckel? (measured — yes; NOT actioned)

`actionAngleStaeckel.c` has the identical structure: `gsl_integration_glfixed` over
`[u_{\min},u_{\max}]` and `[v_{\min},\pi/2]`, integrands `sqrt(...Squared)`, **no
substitution**. The Python/backend path deliberately reproduces C's rule
(`actionAngleStaeckel.py:92`: *"Matches the C gsl_glfixed exactly: plain GL of `order` points
... the J integrands VANISH at the turning points, so no t^2-substitution is needed"*) — which
is the same inference this appendix refutes: vanishing is not sufficient, it vanishes like a
square root.

Note the trap this creates: comparing `c=True` against `c=False` shows agreement to ~1e-12,
which proves nothing, because both run the same rule. The discriminating test is convergence
**in the order**:

| order | $\Delta J_R$ | $\Delta J_z$ | ratio |
|---:|---:|---:|---:|
| 10→20 | 3.97e-04 | 1.25e-04 | — |
| 20→40 | 5.32e-05 | 1.68e-05 | 7.46 |
| 40→80 | 6.90e-06 | 2.18e-06 | 7.71 |
| 80→160 | 8.80e-07 | 2.78e-07 | 7.84 |
| 160→320 | 1.11e-07 | 3.51e-08 | 7.93 |
| 320→640 | 1.40e-08 | 4.41e-09 | 7.93 |

Factor $\to 8$ per doubling: $O(n^{-3})$ again. Extrapolated, the default order 10 carries
**~4.6e-4 in $J_R$ and ~1.4e-4 in $J_z$** — the same magnitudes as the adiabatic defect. The
module already contains `_staeckel_t2_action` (used for the derivative panels), so the machinery
for the fix exists.

**Deliberately not actioned here** (2026-08-20, Jo's call): a separate agent is working on
Staeckel, so this is recorded rather than fixed, to avoid two people editing the same code.

A paste-ready brief for that agent is at [`HANDOFF_staeckel_quadrature.md`](../HANDOFF_staeckel_quadrature.md).

## 11. The transferable lesson

The same shape of bug is the largest remaining cluster in the xfail ledger: three entries
(`poisson_surfdens[AnyAxisym]` ×2 and `test_eccentricity`) are one root cause — AnyAxisym's
*traced* GL forces carrying ~$10^{-6}$, again from integrating across an endpoint singularity
with a fixed-order rule. **Whenever a quadrature's limits are defined by the vanishing of its
own integrand, the integrand has a branch point at the limit, and a fixed-order polynomial rule
will converge algebraically there.** The fix is nearly always a change of variable, not more
nodes.
