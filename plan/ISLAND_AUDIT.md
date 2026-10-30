# Island audit: how much of the existing suite runs on the backend (2026-10-16)

Question (Jo, 2026-10-16): how much of galpy's existing test suite actually runs on the
backend under `--backend torch/jax`? Ideally 100.00 %.

## Instrument

`paper/ms/stats/island_audit/` (pytest plugin `island_audit.py`, loaded with
`-p island_audit`). Per test it records every galpy call site (first frame in
`galpy/` outside `galpy/backend/` and `tests/`) where

- implicit: numpy/scipy converts a backend array (`torch.Tensor.__array__`,
  jax `ArrayImpl.__array__`);
- explicit: `as_numpy` / `to_host` / `host_eval` is given backend data;
- pin: galpy enters `use("numpy", force=True)`.

`run_audit.sh` runs the existing suite (no `test_backend*`, no torus) in per-file /
per-CI-group processes, as CI does (`--dist load` broke order-dependent tests);
`merge.py` merges, `classify.py` assigns each site a category and computes the
tiers, `groups.py` groups the genuine islands.

Run on feat/backends 8d4e9ff2d (the stack tip; #1720's GPU fixes only move sites
between implicit and explicit).

## Categories

| cat | meaning |
|---|---|
| C | galpy's C extensions: argument packing, C bridges (C-STM / custom-VJP derivatives), C actions |
| CTL | host control and bookkeeping: validation, tolerances, root brackets, step counts, time grids, memo keys, IC bookkeeping |
| OUT | astropy Quantity / SkyCoord, matplotlib, numexpr, NEMO strings |
| USER | a numpy function supplied by the test (densities, df functions) called with backend nodes |
| TAB | constant tables built at construction from non-differentiated parameters (a differentiated parameter takes the backend path; the param sweep checks it) |
| RNG | sampling with numpy's global RNG (a backend key runs the backend sampler) |
| PYINT | the pure-Python integrators (numpy by design; warn for backend ICs) |
| EXC | documented exceptions: numpy-built streamdf fixtures' lazy setups (#1726); until 2026-10-29 also evolveddiskdf and AnyAxisymmetricRazorThinDisk (removed in #1774, #1775) |
| ISL | genuine island: numpy computation that could run on the backend |

## torch (4,181 tests ran)

| tier | tests | share |
|---|---|---|
| no numpy contact at all | 1,850 | 44.25 % |
| + C extensions | 2,119 | 50.68 % |
| + host control/bookkeeping | 2,838 | 67.88 % |
| + output | 2,969 | 71.01 % |
| + test-supplied numpy functions | 3,036 | 72.61 % |
| + constant setup tables | 3,550 | 84.91 % |
| + numpy-RNG sampling | 3,678 | 87.97 % |
| + Python integrators | 3,902 | 93.33 % |
| + documented exceptions (= no genuine island) | 3,961 | 94.74 % |

220 tests touch a genuine island (tests can be in several families):

| family | tests | sites |
|---|---|---|
| IsochroneApprox orbit-input path (`_h`: numpy buffers when R is numpy) | 55 | data-guard `is_backend_array(R)` |
| streamdf / streamgapdf numpy track tables | 53 | `_interpolate_stream_track`, `_progenitor_setup`, `_offset_setup`, `calcaAJac`, `_determine_stream_spreadLB`, ... |
| scipy root finding | 49 | `Potential.rl/rE/zvc/zvc_range` scipy branches, `lindbladR`, `rhalf`, `rvir`; `Orbit.rguiding/rE/LcE` (non-differentiated E) |
| diskdf surface/sigma profiles | 39 | `diskdf._ssp` (numpy consumers) |
| `Orbit.E` fallback loop | 12 | the except path for potentials that do not vectorize |
| `coords.align_to_orbit` | 12 | numpy rotation |
| qdf `estimate_hr/hz/hsr/hsz`, `surfacemass_z` | 9 | numpy fits of backend densities |
| MultipoleExpansion at-point helpers | 8 | `_compute_spher_forces_at_point`, `_eval_R_lm`, ... (tests call them directly) |
| MovingObjectPotential | 4 | numpy pin for numpy queries with a numpy object orbit |
| other | 17 | SpiralArms `_K/_dK_dR` (ndarray*Tensor), Irrgang13 halo dens (`numpy.tanh`), Chandrasekhar `lnLambda`, `LinShuReductionFactor`, interpRZ out-of-grid, `Orbit.Jacobi`, `Orbit.from_fit`, `streamTrack.__call__` |

## jax

Not used: the jax run recorded 0 implicit conversions. numpy reads a CPU jax array
through the buffer protocol, before it ever calls `__array__`, so patching
`ArrayImpl.__array__` intercepts nothing (explicit conversions and pins were recorded:
tiers 48.7 % untouched / 94.9 % island-free on 4,066 tests, a lower bound on contact).
The backend code paths are shared, so the torch run is the measurement.

## Status

Measured and reported in the paper (process.tex, `sec:process-islands`, 2026-10-16).

**Final re-audit (2026-10-18, feat/backends 108b69b21, forced torch, 4,181 tests):**
no numpy contact 1,871 (44.75%); + C 51.78%; + host control 70.15%; + output 73.93%;
+ test-supplied 75.53%; + constant tables 87.97%; + numpy RNG 91.25%; + Python
integrators 96.75%; island-free 98.18%. Genuine islands 220 -> 76 tests: diskdf
`_ssp` 29, `align_to_orbit` 12, streamdf `sample` (numpy-RNG output) 9,
MultipoleExpansion at-point helpers ~6-14, MovingObject 4, streamdf lazy-setup
re-runs on numpy fixtures 3+3+2+1, qdf `_rl_host` 3, interpRZ out-of-grid 3,
`compute_legendre` 2, `length(ang=True)` 1, `calcaAJac(lb=True)` 1,
`surfacemass_z` 1. New by-design sites classified: IsochroneApprox `_parse_args`
host IC for the C backward orbits (C), from_fit's host copy of the data for L-BFGS
(CTL), the backend root's bracket-width read and lindbladR's sign test (CTL).
In the paper (process.tex, sec:process-islands).

**Retired (2026-10-18, Jo):** streamdf.sample(key=None) -> RNG (numpy-RNG draws propagated on
the stream's backend, output cast to numpy, as every key=None sampler returns numpy; the
spherical DFs copy their one table to the host instead, streamdf would need its whole track).
MovingObject's numpy pin replaced by @backend_input coercion (#1730): its Python-integrator
queries compute on the backend; targeted audit of its 4 tests: only C/CTL/PYINT sites. Tiers
with streamdf.sample as RNG (isl3b run): ... + numpy RNG 92.82%; + Python integrators 98.45%;
island-free 100.00% (4,181 tests, 0 genuine islands, 0 unclassified).

**Post-merge audits (2026-10-29).** After the round that removed the two documented
exceptions (#1774 evolveddiskdf, #1775 AnyAxisymmetricRazorThinDisk) and merged the
pspline stream track (#1731/#1732): feat/backends e807264ec (forced torch; orbit group 5
was killed by the 12 GB memory cap, so 4,106 tests) found two new islands, the pspline
fit's numpy namespace for numpy particles (`streamTrack._fit_track_pspline`, 12 tests) and
the backend marginalizations' `calcRotcurve` (2 tests); #1777 removed both. Re-audit of
feat/backends 72143f658 (#1777 merged; orbit group 5 at a 32 GB cap; 4,353 tests,
0 unclassified sites, 0 genuine islands): no numpy contact 2,012 (46.22%); + C 53.34%;
+ host control 72.46%; + output 76.34%; + test-supplied 77.88%; + constant tables 90.72%;
+ numpy RNG 94.14%; + Python integrators 99.91%; + the one documented exception 100.00%
(the numpy-built streamdf fixtures' lazy setups, 4 tests). The razor-thin disk's remaining
`_on_numpy` pin is its constructor's unit probes and Phi(0) integral (TAB). Classifier: the
evolveddiskdf host-integrator lines moved by one (3368/3369 -> PYINT). Failures in that run:
SIMBAD outage (test_from_name_values), nemo x2 (local install); test_amuse not collected
(amuse not installed). Final re-audit at the tip a61a7f31f (#1778, #1779 merged; 4,353 tests,
0 unclassified, 0 genuine islands; in the paper's Table 1): no numpy contact 2,011 (46.20%);
+ C 53.32%; + host control 72.43%; + output 76.32%; + test-supplied 77.85%; + constant tables
90.72%; + numpy RNG 94.14%; + Python integrators 99.91%; + the one documented exception
100.00%. Against 72143f658: #1779's per-table `as_numpy` in
`OblateStaeckelWrapperPotential._tabulate` is a new site (TAB; the per-knot `float()` it replaced
is invisible to the instrument, which does not patch `Tensor.__float__`), moving
test_actionAngle_oblatestaeckelwrapper_cache_c from host control to constant tables, and
test_from_name_values, which failed early in the SIMBAD outage, now passes through a host-control
site. Failures: nemo x2, test_orbit_c_sigint_planar (timing under load). Paper updated
(process.tex, Table 1).

**Islands part 4 (2026-10-18): #1728 (align_to_orbit + backend custom-frame T through
Orbit/StreamTrack/pmrapmdec_to_custom/sky_to_customsky_jac; MultipoleExpansion at-point
helpers) and #1729 (diskdf _ssp + private helpers + eval under forcing; qdf _rg coerced,
_rl_host removed; surfacemass_z(fixed_quad=False) as scipy's smoothing spline = the
least-squares cubic, 7e-14; calcaAJac(lb=True) by AD, 100x closer to central differences
than numpy's forward difference).** Re-audit of feat/backends 108b69b21 + both PRs (forced
torch, 4,181 tests, 0 unclassified): no numpy contact 1,902 (45.49%); + C 52.55%; + host
control 71.01%; + output 74.96%; + test-supplied 76.56%; + constant tables 89.33%; + numpy
RNG 92.61%; + Python integrators 98.23%; island-free 99.78%. Genuine islands 76 -> 9, all
streamdf.sample with key=None (numpy-RNG output cast to numpy; Jo's exception). Reclassified
after checking the call stacks: MovingObject's numpy pin (entered only from the pure-Python
leapfrog/dop853 integrators: PYINT), interpRZ's out-of-grid numpy branch (StaeckelGrid
tables: TAB; planar Python integrator: PYINT), streamdf `_parse_track_dim` (plot accessor:
OUT), and the numpy-built-fixture streamdf lazy setups `_determine_stream_spreadLB`,
`_interpolate_stream_track`, `_lb_track`, `length` (EXC, as kept in #1726). Failures in that
run: SIGINT timing x2, test_fixedstepsize (timing under load), nemo x2 (local install);
the earlier run also hit test_streamgapdf_impulse::test_impulse_deltav_general_hernquist,
an unseeded-RNG flake that fails on the base for 1 of 40 seeds. Paper updated (process.tex).

**Fixes (2026-10-17).** #1722 (diskdf LOS, StreamTrack.__call__) and #1723 (root
finders `rl`/`rE`/`zvc`/`zvc_range`/`rhalf`/`lindbladR`/`rvir`, LinShu, lnLambda,
SpiralArms, Irrgang13, qdf `estimate_*`) merged; forced plain scalars solve on the
backend too (Jo: forcing means backend use), via `brentq(..., elementwise=True)`'s
512-section concrete root. Re-audit (forced torch) after #1723: 220 -> 176 island
tests (the audit map now classes the backend root's bracket-width read and
lindbladR's sign test as CTL). Remaining in that area, accepted as no-impact (Jo: "numpy islands that have no
practical impact are fine"): qdf `_rg`'s numpy branch (reached under forcing only by
direct calls of the private `_rg` with numpy `lz`; `qdf(...)` coerces), qdf's
untraced-jax host solve of out-of-table points (nothing to differentiate, same
values), and the constant rg(Lz) setup table. A tried-and-dropped patch for the
first two: /var/tmp/claude-probes/qdf_rg/dropped_followup.diff. #1724 (from_fit on the backend) merged.
Open as a stack: #1725 (Orbit methods, IsochroneApprox) -> #1726 (streamdf/streamgapdf).

The original sizing: Rough sizes: trivial (SpiralArms
`_K`, Irrgang13, Chandrasekhar `lnLambda`, `Orbit.Jacobi`), small (interpRZ out-of-grid,
qdf `estimate_*`, `align_to_orbit`, MultipoleExpansion at-point helpers, `LinShu`),
medium (scipy roots -> backend brentq under forcing; IsochroneApprox orbit input;
diskdf `_ssp` consumers; MovingObject's numpy object orbit), large (streamdf's numpy
track tables).
