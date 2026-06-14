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
