# Numpy speed audit: feat/backends vs main (2026-10-25)

Jo, 2026-10-25: "we should match main numpy speed, and we shouldn't need jit
for it"; "maybe we need to audit all main-line tests to check for slowdowns for
numpy on the backend branch".

## Method

CI runs `pytest --durations=0`. I compared the numpy jobs of the rebase check
PR #1750 (tip 055f142df) with main's push run at the same main SHA (057dfee16),
job name to job name. Runner speed varies 0.5-2.8x between the two runs of a
job, so `robust.py` divides each test's tip/main ratio by its job's median ratio
and flags a test only if it is slower in **every** job that runs it.
`robust_full.txt` is the output over all 53 numpy jobs.

## Findings

- `test_sphericaldf.py::test_constantbeta_dehnencore_in_nfw_dens_spherically_symmetric`:
  106 s on main, 2883 s on the tip (it builds the module fixture's constantbetadf
  fE spline). Locally, the fE-spline build for that fixture's DF:

  | | jax 0.10.1 | jax 0.12 (CI) |
  |---|---|---|
  | main | 233 s | 223 s |
  | feat/backends | 632 s | **6251 s** |

- Orbit tests that evaluate potentials in Python many times are 1.6-3.6x slower
  in 10-12 jobs: `test_energy_jacobi_conservation` (bars, nested lists,
  ExpTruncNFW, wrapped MWP14), `test_liouville_3d` (RotateAndTilt +30 s each,
  ExpTruncNFW, DehnenBar), `test_analytic_ecc_rperi_rap[ExpTruncNFW]`.
- The cause of the broad part: per-call numpy potential evaluation is 2-7x slower
  than main's (scalar calls, us):

  | | Phi main -> feat | Rforce | R2deriv |
  |---|---|---|---|
  | Hernquist | 15 -> 45 | 19 -> 41 | 26 -> 55 |
  | MiyamotoNagai | 16 -> 36 | 19 -> 40 | 26 -> 60 |
  | MWPotential2014 | 46 -> 234 | 99 -> 106 | 143 -> 224 |
  | ExpTruncNFW | 38 -> 147 | 32 -> 147 | 50 -> 234 |
  | DehnenBar | 18 -> 135 | 22 -> 126 | 29 -> 161 |
  | RotateAndTilt(MWP14) | 72 -> 370 | 225 -> 363 | 872 -> 1600 |

  cProfile: 7-10 `get_namespace`/`namespace_from_arrays`/`is_compiling` calls per
  evaluation, the `@backend_input` wrapper, `scalar_like`, the special-function
  router (5 per PowerSphericalwCutoff call), 45-60 `isinstance`; DehnenBar's
  `_evaluate` spends 60 us in `ones_like`/`where` on scalars. Next: a numpy fast
  path in the shared dispatch.

## constantbetadf numpy fE: cause and fix

The numpy fE path computes its d^m(rho r^2beta)/dPsi^m chain with eager jax
autodiff; `quadpack.quadrature` calls it with sizes 1, 2, 3, ... (rising GL
order), and eager jax compiles every primitive once per new size (95% of the
time on main as well). feat's chain got its force from the generic
`evaluateRforces` (more primitives: Hernquist chain 89 vs main's 58), so each
size compiled more.

Fix (branch `backend/constantbeta-numpy-speed`):

1. pad the chain's input to `bucket_size` (bit-identical);
2. main's hand-written `_rforce_jax` back for the numpy path (Jo: "give the
   numpy path its own hand-written derivative again for the profiles that had
   one"); NFW and ExpTruncNFW precise at small r (Jo chose precise; main's
   closed forms give fE 23% off within 1e-8 of Phi_min, main itself 60%);
3. elementwise `jax.grad(sum(f))` instead of `vmap(grad(f))` on the numpy path
   (Jo: yes; to an ulp, 1.3-1.4x);
4. TruncatedTwoPower `_rho_g` back to main's plain arithmetic.

PR: [#1752](https://github.com/jobovy/galpy/pull/1752) (commit e1d59053c on
055f142df). Construct + fE spline, x64, numpy path (each row concurrent):

| DF | main | feat | #1752 |
|---|---|---|---|
| CI fixture (NFW pot, DehnenCore dens, beta=0.25), jax 0.10.1 | 250 s | 636 s | 312 s |
| same, jax 0.12 | 235 s | 6256 s | 279 s |
| same, quiet box | 197 s | | 188 s |
| Hernquist, beta=-0.3 | 145 s | 234 s | 75 s |
| TruncatedNFW (rc=5a, n=2), beta=-0.3 | 1249 s | >9000 s | 939 s |
| TruncNFW + TruncHern, float32, build+sample+eval | 1793 s | | 1499 s |

The NFW fixture is 1.0-1.25x main: the precise small-r series is what runs at
deep energies (all nodes < 0.25a), and nested autodiff through it costs more
than main's closed form (which would give 1.75x faster than main, but fE 23% off
within 1e-8 of Phi_min).

Dead ends: padding alone is not enough for a full 202-energy build (463 s vs
main's 233 s): 4-energy probes are compile-dominated and overstate it. The
precise NFW form costs ~2x under `vmap` (both branches evaluate on mixed
batches), and still ~1.6x with elementwise grad (the outer-integral nodes
straddle the series switch).

## Main bugs found on the way

- galpy never enables jax x64: unless the user does, constantbetadf's chain runs
  in float32, and fE is 25% off at the most-bound energy (main and feat).
- main's NFW/ExpTruncNFW `_rforce_jax` cancel at small r (NFW dF/dr 290x off at
  r = 1e-6 a).

## Shared dispatch: #1753

Per-layer timing (each decorator's `__wrapped__`) of a scalar `evaluatePotentials`:
`@backend_input` ~10 us (main has none); the unit decorators cost the same on both
trees in isolation. #1753 (74a2d19ce): `@backend_input` early return for host
coordinates + cached no-backend-parameter flag + no non-numpy default; inline host
checks in `namespace_from_arrays`/`is_backend_array`; `scalar_like` host early
return; cached `torch.compiler.is_compiling`; router -> scipy for numpy.
Public calls (us, main / feat / #1753): Hernquist Phi 15/34/23, MWP14 Rforce
95/104/74, DehnenBar Phi 19/130/106. Internal calls over 44 default-constructible
potentials: median scalar 1.89x -> 1.63x main.

Remaining: branch-free bodies. numpy.where / ones_like on a scalar ~3-4 us each and
return a 0-d array (each later op ~1 us). Worst internal scalar calls (main -> feat):
DehnenBar Phi 5 -> 90 us, Kepler 1.7 -> 26, SphericalShell 2.5 -> 35, NFW 5.8 -> 60,
ExpTruncNFW 13 -> 100, HomogeneousSphere 4.7 -> 27, Einasto 11 -> 56, Burkert 12 -> 58.
Prototype on DehnenBar with `where(xp, c, a, b)` / `ones_where(xp, c, x)` helpers
(numpy-scalar `a if c else b`, else the exact original expression): 70 -> 22 us,
bitwise identical results and types (scalars, edges, arrays). Awaiting Jo on the
approach.
