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
bitwise identical results and types (scalars, edges, arrays).

## Bodies: #1756 (stack #1757 on #1753)

`where`/`fill_where` in `galpy.backend._coerce` (numpy-scalar `a if c else b`, else the
same `xp.where`), applied to DehnenBar, NFW Phi, PowerSpherical, SphericalShell,
HomogeneousSphere, ExpTruncNFW (numpy `_F` -> branch_where), Burkert, Einasto,
PowerSphericalwCutoff, RotateAndTilt's R=inf guard; router: module-level scipy.special,
fallbacks as "module.name" strings. Internal scalar Phi (us, main / #1753 / #1756):
DehnenBar 5/72/23, NFW 5.9/52/11, Kepler 1.7/23/4.4, SphericalShell 2.4/33/6.9,
HomogeneousSphere 4.6/21/4.5, ExpTruncNFW 19/80/34, MWP14 24/147/54 (forces 66/45/37).
Flagged tests (s): analytic_ecc[ExpTruncNFW] 8.6/28.4/15.7, liouville_3d[DehnenBar]
2.3/7.9/4.0; liouville_3d[RotateAndTilt_offset] 32/64/65 (2nd derivs, not touched).
Bitwise check: `eq_harness.py`/`eq_compare.py` (internal methods, 4023 records) and
`pub_types.py`/`pub_compare.py` (public functions, 2200 records): 0 value changes; 30
public types numpy.float64 -> float, all = main's. `bodies_all.py`: per-potential timing.

## ubuntu-26.04 CI (found on these PRs): #1754 (main) / #1755 (feat)

GCC 15 on the new image: (1) coverage build: undeclared __gcov_dump/__gcov_reset;
(2) Torus build: omp.h inside integrateFullOrbit.h's extern "C" ("template with C
linkage"); (3) test_harmonic_planar_stm_vs_fd_of_flow: forward FD eps=1e-6 of an
adaptive integration = integrator error/eps ~1e-4, machine-dependent (5.53e-4 on some
26.04 runners vs a 5e-4 bound). Central FD eps=1e-4: 1.06e-5 (non-axi Multipole),
2.5-4.6e-8 others; bounds tightened to 5e-5 / 5e-7. Not #1753: the failure follows
the runner (#1755 without #1753 fails identically).

2026-10-26: #1754 (dd6e0fb06) 105 pass / 3 skipping, #1755 (2f6f784a6) 152 pass, codecov
patch+project pass, both MERGEABLE CLEAN. Every ubuntu-latest + run-notebooks job ran on
ubuntu-26.04 (26/26 and 33/33 by the job logs' "Image:" line), so all three fixes were
exercised. #1754's earlier Windows test_actionAngle red did not recur.

## Coverage 15 vs 14 since the rebase: #1758 (stack #1759 on #1755)

Jo: "15 in 1755". #1755 adds none (codecov base 055f142df 15, head 15). Per-file misses
6ac49fcc1 (pre-rebase tip) 14 vs 055f142df 15: the new one is conversion.py:1065, the
rebase commit's `as_backend_constant` move of per-orbit ro/vo scales onto the output's
backend inside main's #1740 shape branch. #1758 adds
test_numpy_orbit_per_orbit_scales_under_forced_backend (#1740's shape/time/dontreshape
grid, forced jax/torch, quantity on/off, 1e-13 vs numpy). A/B under CI's
-W error::DeprecationWarning: 4/4 with the line, both torch cases fail without
(Tensor * ndarray -> numpy -> torch __array_wrap__ deprecation). Whole file 239 passed
(13 min); new test passes in --backend jax/torch, +--jit.
#1758 CI: codecov base (2f6f784a6) 15 -> head (981db67b9) 14 misses, 20/20 sessions.
3 reds, all SIMBAD (from_name: DALServiceError / SSL-read timeout); CDS unreachable from
the dev box too (000 after 30 s). Waiter reruns the failed jobs + all-clear once SIMBAD
answers 200.
#1758 green 18:42Z (152 pass, MERGEABLE CLEAN) after SIMBAD reruns: CDS down until ~16:10Z,
then flaky (1 in 3 local from_name runs fully passed); 4 rerun rounds of the from_name jobs.

2026-10-26: Jo merged stack #1759 (#1755 -> 89ecb4651, #1758 -> 40507855b on feat/backends).
#1752: empty commit 723b6d0fb. Stack #1757 rebased by hand onto 40507855b: #1753
74a2d19ce -> 652853a92, #1756 1a94ee3c0 -> 662defbe8; trees == trunk+old-layer merges,
each layer's diff byte-identical to before; pushed with per-branch leases.

2026-10-27: Jo merged #1753 (392cdf29a) and #1752 (34abedf03). GitHub's stack rebase put
#1756 on 392cdf29a (e2cba220f, tree == my 662defbe8). #1756 had 15 misses vs 14:
_router.py `return sp.exp1(x)` went dead behind #1756's numpy early return in exp1 (torch has
no exp1, jax returns earlier). Fix 2e333522f: numpy selects scipy.special and returns through
that line; 1.82 us/scalar call unchanged; special+exptruncnfw tests 322 (CI flags, _router.py
100%) and 328 in jax/torch/+jit. Old runs cancelled on push (Jo: "Don't wait for CI to finish,
just cancel when you push"). All other reds on #1752/#1753/#1756 were SIMBAD (from_name tests +
notebooks) and codecov phantoms from the failed shards' missing uploads.

2026-10-27: #1756 green at 2e333522f after one SIMBAD rerun round (3 from_name shards). Codecov:
14 misses head = 14 base (392cdf29a), patch 109/109. Merged-tree check (#1756 + #1752 at
34abedf03): sphericaldf rforce_jax 4, test_potential profiles 450, exptruncnfw+constantbetadf 135.
#1754 (main) merged 2026-10-26 as cde125a78.

2026-10-27: #1760 (main): NFW/ExpTruncNFW _rforce_jax use the numpy split (_nfw_h, _F_series)
at small r. float64 at r=1e-6: F_r 1.2e-4/1.2e-3 -> 0, dF_r/dr 290x/2100x -> 2e-10/8e-10. fE at
E=Phi(1e-5) moves 2e-6..5e-5. test_rforce_jax_matches_forces (#1752's, unforced) A/B fails on each
potential separately. Full test_sphericaldf green (one xdist-only fixture-order failure; serial pass).

2026-10-27: #1756 merged (db1849984). #1760 green (codecov 24=24, patch 5/5). Jo: "let's do the
temp x64 in the code as a small fix" (not global x64 for users). #1762 (main): constantbetadf's
_gradfunc runs under jax.enable_x64(True) (jax>=0.8; jax.experimental's before -- checked on a jax
0.6.2 venv) and returns numpy. Plain main without x64: NFW beta=-0.6 fE(Phi(1e-3)) 1.70e11 vs
1.34e12, ExpTruncNFW twobeta=-1 NEGATIVE (-2.9e9 vs 9.6e10), 7% at Phi(0.01). Bit-identical with x64
on (3 DFs x 5 E). New test turns x64 off around build+fE; A/B (jax.enable_x64 directly) fails on
main. Full test_sphericaldf serial CI flags: 251 passed, constantbetadf.py 100%. Merges cleanly
with #1760. fb twin in /var/tmp/x64-fb (_numpy_gradfunc only; jax potentials keep user dtype).

2026-10-27: #1763 (feat/backends twin of #1762) at 322ac41a5: _numpy_gradfunc evaluates inside
_enable_x64(True) (same module-level alias as main); jax potentials (_raw_gradfunc) and the backend
fE path keep the user's dtype. fb without x64: NFW fE(Phi(1e-3)) 1.8e4 vs 1.34e12. Bit-identical
with x64 on; A/B fails on fb HEAD. Full test_sphericaldf serial CI flags 255 passed (new lines
covered); test_backend_constantbetadf 82; constantbeta forced jax / jax --jit 51 each. Torch-forced
local failures identical without the change (-W error __array_wrap__ DeprecationWarning; local
torch 2.12+cu126 dynamo is_leaf assertion under --jit; fb CI torch-jit passes them). HISTORY entry
verbatim main's, same place.

2026-10-27: #1762 green at f45e03409 (105 pass, no SIMBAD rerun needed); codecov misses 24 = 24, all
modified lines covered.

2026-10-27: #1763 green at 322ac41a5 (152 pass after one SIMBAD round). Backend rows (|| true) read from
logs: test_sphericaldf 255 passed, 0 failed under jax, jax-jit, torch, torch-jit (the local torch
failures do not occur on CI). Codecov misses 14 = 14, all modified lines covered. Jobs API 502s at
per_page=100 on this run; per_page=30 works.

2026-10-27: merged: #1760 (main, f2f6071d7), #1762 (main, 9c129abcb), #1763 (feat/backends,
1ae1f80f9). feat/backends sync onto main prepared, not applied: main 3 ahead (#1754, #1760, #1762),
fb 605. Reference tree = fb's tree + #1760's HISTORY entry (#1754 is in via #1755, #1762 via #1763;
merge-tree -X ours alone would leak a dead _enable_x64 block into _make_gradfunc and a second
test_rforce_jax_matches_forces). -X theirs replay: 2 min, #1755 skipped as #1754's equivalent,
#1760's HISTORY entry dropped at #1634's HISTORY hunk -> fix-up commit; tip tree == reference, vs old
fb +7 HISTORY lines, 0 deletions. Scratch branch rebase/feat-backends-onto-main-2026-10-27 at
db33e5fcf (no CI: code identical to #1763's green tree). Jo applies.
