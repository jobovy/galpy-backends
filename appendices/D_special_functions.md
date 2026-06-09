# Appendix D — Pspecial: pure-backend special functions (spec & scope)

## Why / when
The Track A potential migrations (#893–896) made every *non-special* method backend-agnostic.
What remains un-migrated is exactly the **special-function-backed** methods. Pspecial is the
single track that unblocks them — and it is high-leverage: each function unblocks several
potentials, and a migrated potential's **forces** immediately become usable for differentiable
orbit integration (diffrax/torchdiffeq, Track D) and downstream action-angle / df work.

**Priority lens: unblock FORCES first.** Orbit integration needs `_Rforce/_zforce/_phitorque`
backend-agnostic; `_mass/_surfdens/_dens` matter less for orbit-int (they gate sampling, some df).
So a function that sits on a force path is higher priority than one that only sits on `_mass`.

## Design (settled — see 00_overview / track0_infra)
A **native-preferring router** in `galpy/backend/special/`:
- For each function, dispatch to the **native** implementation when present
  (`jax.scipy.special` / `torch.special` / `scipy.special` for numpy), else fall back to a
  **pure-backend, AD-friendly** implementation in `galpy/backend/special/_fallback/<fn>.py`.
- A capability test pins the (shrinking) fallback list and asserts native-vs-fallback agreement,
  so native versions are auto-adopted as the backends ship them and each fallback is trivially
  deletable.
- Everything must be **differentiable** (grad-vs-FD) and **vmap/jit-safe**; validated to ~1e-12
  vs scipy on a grid covering the argument ranges galpy actually uses.

## Function inventory

Native availability key: J = jax.scipy.special, T = torch.special, S = scipy.special (numpy path).

### Tier 1 — native-routable, easy, several potentials (mostly mass/dens; some forces via routing)
| fn | needed by | native | fallback if needed |
|----|-----------|--------|--------------------|
| `gammaln`, `gamma` | ~everywhere (44×); gamma=exp(gammaln) w/ reflection for neg args | J,T,S | Lanczos (AD-friendly) |
| `gammainc`, `gammaincc` (regularized P,Q) | **PowerSphericalwCutoff forces** (via `_mass`), **Einasto**, spherical DFs | J,T,S | series (x<a+1) + continued fraction (x≥a+1) |
| `erf`, `erfc` | **TriaxialGaussian `_mass`**, **dynamical friction forces** (#897), several | J,T,S | rational/continued-fraction (or via gammainc) |
| `i0`, `i1` (modified Bessel) | RazorThin / DoubleExp force paths | J,T,S | — |
| `xlogy` | TwoPower/NFW evals | trivial | `xp.where(x==0,0,x*log(y))` |
| `binom` | a couple | derive | gamma-based |

### Tier 2 — pure-backend, high force-unblock impact
| fn | needed by | native | fallback |
|----|-----------|--------|----------|
| **`hyp2f1`** (Gauss ₂F₁) | **TwoPowerSpherical forces+eval+mass** (15×), PowerSpherical `_surfdens`, TwoPowerTriaxial `_mass`/`_psi` | none in J/T (S only) | power series + Pfaff/Euler transformations for |z|→1; the galpy args are real, |z|<1 mostly — restricted-domain series is enough |
| **`ellipk`, `ellipe`** (complete elliptic K,E) | disk potentials (25× combined) — several **force** paths | none in J/T | **AGM** (arithmetic-geometric mean) — fast, AD-friendly, the standard backend-agnostic route |
| `hyp1f1` (confluent ₁F₁) | PowerSphericalwCutoff `_mass` | none in J/T | series + Kummer transformation |

### Tier 3 — Bessel family (force paths of the disk potentials)
| fn | needed by | native | fallback |
|----|-----------|--------|----------|
| `k0`, `k1` (modified Bessel 2nd kind) | **RazorThinExponentialDisk forces/eval/R2deriv** | maybe J; **torch lacks** → need fallback | series + asymptotic (or via integral); pair with i0/i1 |
| `kn`, `iv` (general order) | RazorThin/DoubleExp | none | upward recurrence from k0/k1, i0/i1 |
| `j0`, `j1`, `jv`, `jn_zeros` | **DoubleExponentialDisk** (Hankel-transform forces) | none in J/T | series + asymptotic; `jn_zeros` are **constants** → precompute a table |

### Tier 4 — SCF / Multipole / DiskSCF (P2.5; the angular+radial harmonic machinery)
| fn | needed by | native | fallback |
|----|-----------|--------|----------|
| **associated Legendre `P_l^m(x)` (all orders + 1st/2nd derivs)** | **SCFPotential & MultipoleExpansionPotential** (replaces `util/special.compute_legendre` → scipy `assoc_legendre_p_all`/`lpmn`) | none | **Bonnet / forward recurrence** in (l,m); derivatives by the standard recurrences (AD also gives derivs for free) |
| **Gegenbauer / ultraspherical `C_n^α(x)`** | **SCFPotential radial basis** | none | three-term recurrence in n |
| `sici` (Si, Ci) | dynamical friction (#897) | none | series (small x) + asymptotic (large x) |

### Quadrature (not a "special function" but the same blocker)
A **fixed-order Gauss–Legendre quadrature** (backend-agnostic; nodes/weights precomputed as
constants, integrand evaluated with `xp`) to replace the residual `scipy.integrate.quad` calls:
`EllipsoidalPotential._mass`, MultipoleExpansion radial integrals, DoubleExp, and the spherical-DF
nested quads (Track F). The ellipsoidal **GL angle integral is already backend-agnostic** — this is
the same pattern, just for the radial direction.

## Suggested sequencing (max force-unblock per unit effort)
1. **Tier 1** (native-routable): `gammaln/gamma`, `gammainc/gammaincc`, `erf/erfc`, `i0/i1`, `xlogy`,
   `binom`. Mostly wiring the router to native. Unblocks Einasto, PowerSphericalwCutoff (with…),
   TriaxialGaussian, and the gamma-only mass methods.
2. **`hyp2f1`** + **`ellipk/ellipe`** (Tier 2): the two biggest force-unblockers — TwoPowerSpherical
   (NFW/Hernquist/Jaffe/Dehnen general forms) and the disk elliptic-integral forces. `hyp1f1`
   finishes PowerSphericalwCutoff.
3. **Bessel** (`k0/k1/kn/iv`, then `j0/j1/jv`+zeros): RazorThin then DoubleExponentialDisk.
4. **assoc_legendre + Gegenbauer + GL-quadrature**: SCFPotential, MultipoleExpansionPotential,
   DiskSCF (P2.5) — the largest single unblock (whole potentials, not single methods).
5. **`sici`** + the dissipative-friction `erf`: finishes #897 (dynamical friction in all backends).

Note: **TwoPowerTriaxial forces are already migrated** (pure `_mdens`); only its `_mass`/`_psi` need
hyp2f1/gamma — so the triaxial NFW/Hernquist/Jaffe are orbit-integ-ready *now*.

## Validation (per function)
- native-vs-fallback agreement (the capability test) — pins which fallbacks are still live.
- vs `scipy.special` to ~1e-12 on a grid spanning galpy's actual argument ranges.
- grad-vs-finite-difference (and jax `jacrev==jacfwd` where forward-mode is available; torch
  `gradcheck`) — the functions must be differentiable for the whole point of the backend layer.
- `jit`/`vmap` survival; device/dtype preservation.

## Downstream unblocked
Each migrated potential's forces → usable in the diffrax/torchdiffeq orbit integrator (Track D)
and as test cases for differentiable orbit integration; then action-angle (Track E) and df
(Track F, esp. spherical DFs which also need gamma/hyp2f1) build on top. So Pspecial is the
gate to "most of galpy differentiable end-to-end."

---

## Implementation log — Tiers 1-2 (PR #916, `feat/pspecial` → `feat/backends`)

**Router contract.** `galpy/backend/special/{__init__,_router}.py` + `_fallback/<fn>.py`.
`_dispatch(fnname, args, fallback, ns_args=None)` resolves the backend from `ns_args` (the
array arguments only, so scalar potential params like `(a,b,c)` don't confuse the resolver),
then routes to native iff `fnname not in _NEEDS_FALLBACK[backend]` and the native attr exists.
`_NEEDS_FALLBACK = _NATIVE_MISSING (absent) | _NATIVE_UNRELIABLE (present-but-wrong)`. A
capability test pins `_NATIVE_MISSING` to `hasattr` reality; tripwire tests pin the
`_NATIVE_UNRELIABLE` entries so they auto-promote when the backend fixes them.

**Tier 1 (native-routable).** `gammaln, gamma, gammainc, gammaincc, erf, erfc, i0, i1, xlogy`.
Only torch lacks `gamma` → `gamma_fallback` = exp(gammaln) for x>0, reflection for x<=0, with
each branch's dead side clamped so gammaln's poles can't `0*inf`-NaN-poison gradients.

**Tier 2 (force-unblockers).**
- `ellipk/ellipe` (scipy parameter-m convention): **AGM** — `K=pi/(2·AGM(1,sqrt(1-m)))`,
  `E=K·(1-sum 2^{n-1}c_n^2)`. 16 fixed iters → ~1e-15. Native on neither jax nor torch.
- `hyp2f1(a,b;c;z)`, real z<=0 (galpy: z=-r/a, -a/r, -(z/R)^2): **Euler integral + boundary-layer
  GL**. `2F1 = Γ(c)/(Γ(B)Γ(c-B)) ∫_0^1 t^{B-1}(1-t)^{c-B-1}(1-zt)^{-A} dt` with {A,B}={a,b}
  chosen so B>0, c-B>=1. Two substitutions make 128-pt Gauss-Legendre spectral: (1)
  `1+|z|t=(1+|z|)^X` resolves the `(1+|z|t)^{-A}` boundary layer near t=0 for large |z|; (2)
  `X=xi^k`, `k=min(12,ceil(6/B))`, lifts the residual `X^{B-1}` fractional-power endpoint to a
  GL-friendly high integer power. Floor `|z|>=1e-10` (prevents `X·log1p` underflow → `T^{B-1}=inf`
  for B<1; gives 2F1(z=0)=1 to ~1e-10). **~1e-13 at r/a<~50, ~1e-6 at r/a~500, never diverges.**
- `hyp1f1(a,b;z)`, z<=0: galpy's b=a+1 case is **exact** — `1F1(a,a+1;-X)=a·X^{-a}·Γ(a)·P(a,X)`
  via native `gammainc` (pass `a` as an array — torch.special.gammainc needs both args tensors).
  General b via the same Euler-GL with an exponential boundary-layer map.

**KEY FINDING — jax native hyp2f1 AND hyp1f1 are unusable for galpy.**
`jax.scipy.special.hyp2f1(a,b,c,z)` and `hyp1f1(a,b,z)` are **catastrophically wrong for z<-1**:
hyp2f1 returns +-inf with grad ~5e110 (FD ~0.05) at z=-3; hyp1f1 rel-err ~1e10 by z~-64. They
happen to be exact only when the series terminates (e.g. c=b). Since galpy's z=-r/a, -(R/rc)^2
routinely exceed -1 (r>a), **both are forced to the fallback on jax** (`_NATIVE_UNRELIABLE`),
not just torch (which lacks them entirely). torch.special also has neither, nor ellipk/ellipe.

**Accuracy ceiling note (for P2.5 wiring).** The hyp2f1 fallback is ~1e-6 at extreme r/a~500.
If any existing TwoPowerSpherical test evaluates forces at such radii, jax/torch parity vs the
numpy(scipy) path may need more GL nodes there; revisit when wiring P2.5.

## Implementation log — Tier 3 (PR #917, modified Bessel K0/K1/Kn)

Unblocks RazorThinExponentialDiskPotential forces (k0, k1, kn(2,.)). **Fallback
on BOTH jax and torch:** jax.scipy.special has no k0/k1/kn; torch.special has
modified_bessel_k0/k1 but they are **not differentiable** (no autograd backward)
and there is no kn -- so we don't alias them (a tripwire test pins this). The
router classifies them via the usual hasattr check (no torch.special.k0).

k0/k1 (`_k01`), ~1e-15 vs scipy, AD-friendly, two regimes with dead-branch guards:
- x <= 2: A&S ascending series (9.6.13/9.6.11) on native i0/i1.
- x  > 2: trapezoidal rule on K_nu(x)=int_0^inf e^{-x cosh t} cosh(nu t) dt. The
  double-exponential decay -> geometric trapezoidal convergence; the e^{-x(cosh t-1)}
  peak has width ~1/sqrt(x), so the nodes are SCALED by 1/sqrt(x) (a fixed step
  undersamples the peak and degrades past x~12). h=0.25, N=64 -> uniform ~1e-15
  over x in [2, 400] (K0 underflows to 0 beyond, as in scipy).
- guards: series branch clamps x to 1.0 in its dead region (else i0 overflows for
  x>~700 -> inf -> NaN-poison); trapezoid branch clamps x to 2.0 in its dead region.
kn: stable UPWARD recurrence K_{m+1}=K_{m-1}+(2m/x)K_m from K0,K1.

KEY FINDING: torch.special.modified_bessel_k0/k1 are accurate (~1e-16 vs scipy)
but have NO autograd backward -> unusable for the differentiable backend layer
(same pattern as jax's broken hyp2f1/hyp1f1, but here it's missing-gradient not
wrong-value). Use the fallback. Asymptotic K_nu ~ sqrt(pi/2x)e^{-x}(1+...) was
tried but only reaches 1e-12 for x>~15 (soft gap at x in [10,15]); the scaled
trapezoidal is uniform and simpler, so the asymptotic was not needed.

## Implementation log — Tier 4 (PR #917, assoc_legendre + Gegenbauer)

Unblocks SCFPotential + MultipoleExpansionPotential.
- assoc_legendre(L,M,x[,deriv]): P_l^m with Condon-Shortley phase (scipy
  assoc_legendre_p_all branch_cut=2 convention) + optional 1st/2nd x-derivs.
  Forward (Bonnet) recurrences: diagonal P_m^m=(-1)^m(2m-1)!!(1-x^2)^{m/2},
  P_{m+1}^m=x(2m+1)P_m^m, (l-m)P_l^m=x(2l-1)P_{l-1}^m-(l+m-1)P_{l-2}^m. Derivs
  via (x^2-1)P'=lxP-(l+m)P_{l-1} and the Legendre ODE for P''. Built with
  lists+xp.stack (NO in-place mutation -> jax-immutable/torch-autograd safe);
  AD through it == the analytic derivs (test-pinned). **numpy routes to scipy
  (byte-identical)**; jax/torch use the recurrence (~1e-14 vs scipy).
- gegenbauer(N,alpha,x): C_n^alpha 3-term recurrence. SCF._C already used this
  exact recurrence (not scipy), so it runs on every backend and reproduces SCF
  byte-for-byte; validated ~1e-12 vs scipy.eval_gegenbauer.
- The (L,M)/(N,) array primitives don't fit the single-array _dispatch, so they
  are routed directly in the router (not via _NATIVE_MISSING).

DEFERRED: sici (Si/Ci) -> only used by FDMDynamicalFrictionForce; lands with the
dissipative-force migration (its consumer). Tiers 1-4 (minus sici) now provide
the special functions for ALL force-path potentials + SCF/Multipole/DiskSCF.
