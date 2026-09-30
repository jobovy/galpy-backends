# Track F — df module (ALL DFs + streams)

Branch: `feat/backends`. The largest, mostly-downstream track (df depends on
potential + orbit + action-angle backends). ~23k lines. Detail in master plan §"Track F".

Batches by difficulty:
- **Pdf.1 easy:** df base, surfaceSigmaProfile, jeans — establishes the DF backend-swap
  + backend-agnostic quadrature pattern.
- **Pdf.2 spherical DFs:** sphericaldf + isotropic/constantbeta/osipkovmerritt/eddington/
  king variants — nested `quad`→backend Gauss-Legendre; `gamma/hyp2f1`→Pspecial;
  CDF-inversion sampling → `jax.random`/implicit-diff. (needs Pspecial)
- **Pdf.3 disk DFs:** diskdf, evolveddiskdf — actionAngleAdiabatic backend,
  `integrate`/`integrate_dxdv` backend, scipy.optimize→backend root-find, spline
  caches→functional. (needs Track E Adiabatic + Track D)
- **Pdf.4 quasiisothermaldf:** actionAngleIsochrone/Staeckel backend, tplquad→backend,
  RectBivariateSpline→backend interp, `_rgInterp` cache→functional.
- **Pdf.5 streams** (ultra-hard): streamspraydf, streamgapdf, streamTrack, then streamdf.
  Sampling/RNG, orbit integration, action-angle (IsochroneApprox/Staeckel/Spherical —
  **not** Torus), `make_smoothing_spline`/`cKDTree`, and streamdf's iterative track
  assembly (refactor ~60 loops + in-place mutations + eigen-slerp to functional/vectorized).
  streamdf last; several sub-PRs.
  - **`useTM=True` is OUT OF SCOPE for non-numpy backends** — it routes the track
    through `actionAngleTorus` (external C++ Torus library, never backend-agnostic).
    **Requirement:** when migrating streamdf, RAISE a clear error (e.g. NotImplementedError
    with a message naming Torus) if `useTM=True` AND the active backend is not numpy,
    rather than letting it fail deep inside the Torus C call. The 4 streamdf `useTM`
    tests (`test_bovy14_useTM[_poterror/_useTMHessian/_approxConstTrackFreq]`) are
    DESELECTED from the all-backend CI shard (`-k 'not useTM'`, harness PR #956) and the
    backend job no longer builds Torus at all — so this error-raise is the only backend
    behavior `useTM` should have. Numpy keeps full useTM support unchanged.

Each batch: full existing DF tests pass under all backends + new grad-vs-FD tests on
DF evaluations, moments, and stream-track outputs (w.r.t. inputs and potential params).

## ⛔ CI obligation — RE-ENABLE the deferred jax spherical-DF tests when Pdf.2 lands

The all-backend CI harness (PR #956, `feat/pillar1-backend-ci` → `feat/backends`)
**defers 26 slow jax `test_sphericaldf.py` tests to this track (Pdf.2)**. They
sample (300k CDF inversions) + integrate the DF by scipy at scalar points; under
forced-jax every scalar eval dispatches an XLA graph, so each runs minutes (the
shard brushed the 4500 s session cap until they were skipped). The **Pdf.2**
spherical-DF migration — nested `quad` → backend Gauss-Legendre + CDF-inversion
sampling → `jax.random`/implicit-diff — is exactly what makes them fast.

**Deferral mechanism:** per-nodeid lines in `tests/backend_slow_skip.txt`:
```
jax tests/test_sphericaldf.py::<testname>      # 26 entries, selected by CI
                                               # duration >= 90 s (runs 27500392935
                                               # + 27503988401)
```
matched tests are `pytest.mark.skip("backend-slow-skip: …")`, counted in the
status report's separate **`deferred`** burndown bucket (NOT xfail). One genuine
fast jax correctness failure, `test_isotropic_hernquist_unitsofsamples`, is in
the xfail-LEDGER (not deferred) — that one is a real backend bug to fix, not a
slowness deferral.

**To re-enable (DO THIS as the closing step of Pdf.2):**
1. Delete the 26 `jax tests/test_sphericaldf.py::…` lines from
   `tests/backend_slow_skip.txt` (the whole "jax spherical-DF" section).
2. `GALPY_BACKEND_XFAIL_REGEN=1` regen the now-fast jax sphericaldf shard; add any
   genuine remaining failures to `tests/backend_xfail.txt`. Most should pass.
3. Confirm the sphericaldf jax shard completes well under the session cap and is
   green; the `deferred` count for sphericaldf drops to 0.

(Counterpart obligation, different track: the **whole jax orbit shard** is
deferred to **Track D** via two file-level slow-skip lines — see
`trackD_autodiff.md` "CI obligation".)

---

## EXECUTION PLAN (scouted 2026-07-01, workflow wf_77a2f588-cbc, verified vs code)

**Scale:** ~22.8K lines, 0% migrated — the largest remaining block (> potentials +
AA combined). **All upstream deps LANDED → fully unblocked.** Shared-helper layer
(`galpy.backend.quadrature/interpolate/special/optimize`) ~90% sufficient: only
`binom` missing (Pdf.5); RNG / multivariate-optimize / cKDTree intentionally
numpy-guarded (not made backend-native).

**PR sequence** (Pdf.2/3/4 have disjoint files → parallelizable; Pdf.5 last):
| PR | Files | Lines | Effort | Notes |
|----|-------|-------|--------|-------|
| Pdf.1 foundation | df.py, surfaceSigmaProfile.py, jeans.py | 480 | M | warm-up; establishes df dispatch + quad-swap pattern |
| Pdf.2 sphericaldf | sphericaldf + 9 subclasses | 2778 | XL | special(gamma/hyp2f1)+nested quad+pvr/cmf interp+sampling |
| Pdf.3 diskdf | diskdf, evolveddiskdf | 6776 | XL | DFcorrection spline cache, numpylog(28×), Orbit.integrate; slow_skip under jax |
| Pdf.4 qdf | quasiisothermaldf | 3121 | L | _rg/_maxVT interp caches, tplquad, Staeckel/Isochrone dep |
| Pdf.5 streams | streamdf/spraydf/gapdf/Track | 9597 | XL | LAST; 4 sub-PRs; iterative track loop + eigen-slerp = hardest |

**Pdf.1 concrete (the immediate PR):**
- `df.py` (base): units only, no compute → no changes.
- `surfaceSigmaProfile.py`: 4 methods (surfacemass/…Derivative/sigma2/…Derivative) — add
  `from ..backend import get_namespace`; `xp = get_namespace(R)` first line; `numpy.exp/log→xp.*`
  (L150/175/198/200-201/230). params are Python floats → broadcast, no coercion.
- `jeans.py`: `sigmar`, `sigmalos` — add `from ..backend.quadrature import quad, fixed_quad_semiinfinite`;
  `xp = get_namespace(r/R)`; the 4 `integrate.quad(...)[0]` → `quad(...)` (finite) /
  `fixed_quad_semiinfinite(xp, ..., lo)` (semi-inf); `numpy.sqrt→xp.sqrt`. **Non-mechanical:** the
  `numpy.isnan(called_surfdens)` at L125 is a "was surfdens supplied" sentinel — keep the
  supplied-vs-not as a Python `None`-check (static), only `xp.isnan` for genuine array tests.
- **Byte-identity:** quad/fixed_quad_semiinfinite delegate to scipy on numpy → byte-identical; VERIFY
  the numpy branch is literally scipy (flag-numpy-byte-identical-changes). jax/torch = fixed-order GL
  (smooth mass/density integrands → matches the accepted Ellipsoidal/Spherical tradeoff).
- **Coverage:** add `tests/test_backend_df_*.py` exercising the `if xp is not numpy` lines or
  codecov/patch fails (no-hiding-uncovered-code).

**Hard parts (Pdf.5):** streamdf `_determine_stream_track` (~60-iter refinement loop, rebuild full
array/iter, fixed loop count, no value-branch); eigen-slerp needs a deterministic eigenvector sign
convention (else AD discontinuities); sampling RNG stays numpy-seeded (bitstream divergence accepted,
tests check shape/dtype only); spline caches = dual-path (scipy at init for numpy byte-id + pickles,
`spline_to_ppoly` coeffs for backend eval); `numpylog` (scimath.log) helper for diskdf.

**Milestones:** ledger prune after Pdf.1 (smoke), Pdf.2, Pdf.3+Pdf.4 (mid), Pdf.5 (final). ~11-15
person-weeks total; ~6-8wk wall-clock if Pdf.2/3/4 run concurrently. After Pdf.5 = last breadth block
→ then the deferred speed-up roadmap (logs/backend_speedup_audit/).

**2026-09-29 status.** eddingtondf f(E) exact near Emin on both paths (#1586
main, #1578 feat). #1598 (open): osipkovmerrittdf.sample under jax.jit (traced
r(Phi) now log-r bisection, exact near Emin); constantbetadf backend fE needs
no calibrated lower limit (7.7e-12 vs Hernquist's closed form). #1593 (open):
qdf constructor and streamTrack fit compile under torch.compile. Next: numpy
islands in streamspraydf; vectorized stream track; kingdf under torch.compile.
