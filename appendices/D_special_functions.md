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
