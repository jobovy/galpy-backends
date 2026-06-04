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
