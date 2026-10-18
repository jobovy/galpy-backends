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
| EXC | documented exceptions: evolveddiskdf (forcing exception, CI time), AnyAxisymmetricRazorThinDisk (scipy quad for numpy inputs) |
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

**Fixes (2026-10-17).** #1722 (diskdf LOS, StreamTrack.__call__) and #1723 (root
finders `rl`/`rE`/`zvc`/`zvc_range`/`rhalf`/`lindbladR`/`rvir`, LinShu, lnLambda,
SpiralArms, Irrgang13, qdf `estimate_*`) merged; forced plain scalars solve on the
backend too (Jo: forcing means backend use), via `brentq(..., elementwise=True)`'s
512-section concrete root. Re-audit (forced torch) after #1723: 220 -> 176 island
tests (the audit map now classes the backend root's bracket-width read and
lindbladR's sign test as CTL). Remaining in that area: qdf `_rg`'s numpy branch
returning `rl` to numpy callers (3 tests), and qdf's older untraced-jax host solve
of out-of-table points. #1724 (from_fit on the backend) merged.
Open as a stack: #1725 (Orbit methods, IsochroneApprox) -> #1726 (streamdf/streamgapdf).

The original sizing: Rough sizes: trivial (SpiralArms
`_K`, Irrgang13, Chandrasekhar `lnLambda`, `Orbit.Jacobi`), small (interpRZ out-of-grid,
qdf `estimate_*`, `align_to_orbit`, MultipoleExpansion at-point helpers, `LinShu`),
medium (scipy roots -> backend brentq under forcing; IsochroneApprox orbit input;
diskdf `_ssp` consumers; MovingObject's numpy object orbit), large (streamdf's numpy
track tables).
