# Backend-θ differentiable stream sampling (generative d(track)/dθ)

## Approach
Make `streamspraydf.streamTrack()` (no `particles=`) give the PHYSICAL `d(track)/dθ`
where the spray sample `xv(θ)` FLOWS — every member star's orbit integrated on the
backend — not the earlier (wrong) fixed-`xv` gradient. Target: fully `jax.jit`-able,
GPU-capable, via jax + diffrax. See [[stream-track-two-gradients]].

## Dead-ends
- **Fixed-`xv` "fitting gradient"** (Track G artifact): non-physical — `xv` is a theory
  intermediate `xv(θ)`, not data. Freezing it drops `∂track/∂xv·dxv/dθ` (the dominant term).
- **Eager torchdiffeq generative sampler** (WIP branch `feat/backend-theta-sampler`,
  streamspraydf.py `_backend_sampling`): correct in principle but hangs/OOMs — adaptive
  torchdiffeq over tdisrupt≈126 units (20 orbital periods) × 600 stars is impractical
  (~26 s progenitor alone). Parked, not a PR.

## Results (2026-07-11)
Foundation + M1 prototype validated (`m1_generative_prototype.py`, `probe_jax_diffrax_scale.py`):
- **jax+diffrax batch integration is jittable**: θ-grad matches eager exactly; cached
  re-run ~0 s (compile is one-time). `Orbit.integrate(method="diffrax")` traces under jit;
  `o.orbit[...]` works but **`o.x(t)` does NOT** (numpy `__array__` on the tracer) — read
  `o.orbit` directly (integrate ON the grid you need).
- **Scale (GPU, TITAN Xp, n=600, span 126)**: 13.2 s compile, **~2.5 s/step** cached
  value+grad. (Eager torchdiffeq: hung.) → the "just jit it" path works.
- **M1 physical gradient VALIDATED**: fully-jitted generative pipeline (sample → integrate
  all N orbits via diffrax → jax-native reconstruction: brute-force argmin closest-point,
  histogram binning, fixed Gaussian smoother; discrete indices `stop_gradient`, values flow).
  Frozen-structure grad-vs-FD (identical operator, xv flows): AD −1134.468 vs FD −1134.441,
  **rel 2.3e-5**. Moving-structure secant −301 differs by discrete assignment/bin jumps
  (coarse 40-bin histogram; galpy's smooth spline reconstruction will narrow this — M2).

## M2 done (2026-07-12): fully-jittable generative pipeline, gradient VALIDATED
`m2_generative.py` -- sample (reparam noise) -> integrate progenitor + all N stars
via diffrax -> jax-native reconstruction (brute-force argmin closest-point +
raw-data P-spline smoother: `s=(B'B+lam D'D)^-1 B'off`, GCV lambda). Fully under
`jax.jit`; xv(theta) flows; only discrete structure (assignment, P-spline basis,
lambda) stop_gradient'd. **Track gradient VALIDATED to REL 6.4e-7** (lam=1.0).

**CRITICAL VALIDATION LESSON**: streams are SENSITIVE probes -> sample orbits over
tdisrupt (~20 orbital periods) have huge higher derivatives, so a single-h FD is
UNRELIABLE. Must use **h-convergence**: at h=1e-3/1e-4 the FD is nonlinear garbage
(REL 2-30), only h=1e-5 lands in the linear regime (REL 6e-7). Two phantom "bugs"
(M1 histogram, M2 GCV) were actually just bad single-h FDs -- the AD was correct
all along. Every piece (rtide/vc/frame/integration/IC-sensitivity) individually
grad-vs-FD passes at h=1e-6. `o.x(t)` jittability fix committed (f7ac8b5d).

Open tuning: GCV picks lambda~2e-5 (under-smooths -> wiggly, over-sensitive track);
fixed lam=1.0 used for validation. Also the full-tdisrupt extent (20 periods) makes
the old stream end very sensitive -> M3 should bound track extent like galpy's
track_time_range. Neither is a correctness issue.

## Payoff demo (2026-07-12): the gradient WORKS for inference
`m2_fit_demo.py` -- gradient-descent halo-mass fit on the GENERATIVE track.
Younger coherent stream (STREAM_TD=25 ~4 periods): **amp 1.08 -> 0.998** (true 1.0),
loss 9.3 -> 8e-3, hand-rolled Adam, 60 steps, 107 s on CPU. Full-tdisrupt (20 periods)
does NOT fit cleanly: chaotic sensitivity of the old stream end -> rugged loss
landscape (sign-oscillating gradient). Physical, not a bug -- real stream analyses
bound the extent (galpy `track_time_range`); M3 must do the same.

### Multi-parameter proof (`m2_multiparam_gradients.py`)
The generative track is differentiable w.r.t. ALL parameters through the jittable
sampler, each h-converged: d/d(amp) 4.7e-8, d/d(q) 1.8e-6, and all 6 progenitor ICs
d/d(R,vR,vT,z,vz,phi) all REL < 6e-6. Halo mass + halo shape + full progenitor phase
space -- everything flows.

### STATUS: core goal PROVEN in prototype
fully-jittable + GPU + correctly-differentiable (ALL params) + useful generative
streamspray. Everything the user asked for (backend / differentiable / GPU, no
compromises) is demonstrated end-to-end in `m2_generative_prototype.py` +
`m2_multiparam_gradients.py` + `m2_fit_demo.py`. Remaining = integrate into the
actual galpy `streamspraydf` (M3, delicate/collaborative).

## M3 IN PROGRESS (2026-07-12): Option A — jit the existing class (user chose this)
`streamspraydf.sample()` is now FULLY JITTABLE + differentiable in real galpy:
- **`sample()` committed `fe5764f1`**: `jax.jit(fardal15spraydf(...).sample())` compiles,
  runs CPU+GPU, grad-vs-FD h-converged REL 2e-6, eager==jit. numpy BYTE-IDENTICAL
  (maxdiff 0.0 vs 66837a7f). Construction routes the progenitor to the in-backend ODE
  (`_integrate_progenitor`, detection jit-safe), on a NUMPY-time grid (self.t concrete
  so o.x/L interpolate under jit; state differentiable). `_sample_tail`/`_setup_rot`
  keep dt on-backend + query at backend `qt`. `_progenitor_mass_fn` backend-agnostic.
  **spray_df draws via the jax key** (reparameterized) — a MUST: a numpy key / numpy.random
  bakes a per-trace constant → AD≠FD (chased this: the test's `grandom.key(0)` was a NUMPY
  key; `key(0, backend="jax")` fixed it).
- **`o.x(t)` jit fix committed `f7ac8b5d`**; **streamTrack key-threading `b7d5e468`**.

### DONE — `streamTrack()` fully jittable (commit `b8c9d946`) — OPTION A COMPLETE
`jax.jit(fardal15spraydf(...).streamTrack())` compiles + runs CPU+GPU, differentiable,
grad-vs-FD h-converged **REL 3.8e-5**; numpy **BYTE-IDENTICAL** (maxdiff 0.0); all 35
tests/test_backend_streamspraydf.py pass (no #1102/#1106 regression).
- streamspraydf.py: streamTrack curve-build reuses `self._bsamp` (not a float() probe on
  self._progenitor, now a backend orbit). **Eager C-STM PRESERVED** (user: "Preserve"):
  `_integrate_progenitor` -> in-backend ODE for backend-theta or a TRACED (jit) IC, but
  `dop853_c` for a CONCRETE backend IC (#1102 intact). base/chen24 spray_df take the key.
- streamTrack.py: `_fit_track_backend_jit` -- jittable jax/torch-native reconstruction
  (M2 recipe: static-shape 6D velocity-weighted argmin closest-point + raw-data hat-basis
  P-spline; structure stop_gradient'd, offset values flow). Dispatched when the curve is a
  TRACER (as_numpy fails); eager/numpy keep the scipy/cKDTree path.
- Caveats/follow-ups: fixed P-spline lambda (GCV TODO); tp_grid over the full arm [0,T].

### DONE (2026-07-12b) — jittable track_time_range auto-estimate + covariance
- **track_time_range auto-estimate jittable** (`_auto_track_time_range`): eager keeps the
  particle-extent estimate; under jit (particles/prog TRACED, concrete grid impossible)
  falls back to a CONCRETE physics estimate from the numpy IC (~3 radial periods
  2piR/|vT|, clipped to [1,tdisrupt]; tdisrupt for a backend IC). streamTrack() jittable
  WITHOUT explicit track_time_range now: grad-vs-FD h-conv **REL 9.5e-6**.
  (A traced particle-extent estimate would need StreamTrack to store a backend tp_grid +
  a backend-grid interp everywhere — deferred; interp_linear does numpy.asarray(grid).)
- **jittable covariance** in `_fit_track_backend_jit` (order>=2): weighted residual
  scatter per node `cov_node = einsum("nk,ni,nj->kij",B,resid,resid)/wsum`, interpolated
  to tp_grid. **NO psd_project** -- cov_node is PSD BY CONSTRUCTION (B>=0 -> non-neg
  weighted sum of outer products; linear interp preserves PSD), and psd_project's frozen
  eigenvectors would CORRUPT the gradient (dropped rotation term -> AD 0.104 vs true
  0.0118). Without it: PSD (min eig -4e-21) + grad-vs-FD h-conv **REL 8.3e-5**.
  KEY LESSON: only psd_project a matrix that isn't already PSD; a construction-PSD cov
  should flow ungated.
- numpy byte-identical (maxdiff 0.0); suites re-run.

### DONE (2026-07-12c) — EXACT-match: traced particle-extent track_time_range + normalized StreamTrack
User: "we want to exactly match functionality." Under jit the auto track_time_range now
uses the TRUE particle-extent estimate as a TRACED scalar (not the concrete IC proxy),
so d(track_time_range)/dtheta flows too (commit `70315842`).
- Blocker: interp grids must be concrete (cubic_spline_coeffs assembles its tridiagonal
  matrix with a numpy loop over the grid) -> a traced tp_grid is impossible as the spline
  axis. SOLUTION: NORMALIZED curve coordinates. Integrate the progenitor over physical
  times T*u (traced) but the reconstruction axis is the CONCRETE normalized u in [-1,1];
  carry T into StreamTrack as **tp_scale** (physical tp = u * tp_scale).
- **StreamTrack tp_scale mode**: cubic spline built on the concrete normalized grid;
  accessors (_eval_cart/_cart_eval/cov) map physical->normalized (tp/tp_scale) at query
  (a traced query r is fine; only the GRID x must be concrete); tp_grid() returns u*scale.
  tp_scale=None -> physical axis (eager, byte-identical). from_particles threads tp_scale.
- **_backend_curve reads o.orbit** (cyl->cart) instead of o.x(ts): o.x needs a concrete
  self.t, but the jit integration grid T*u is traced. Same knot values for eager.
- Validated: jit streamTrack track_time_range=None grad-vs-FD **REL 3.3e-7** (CPU),
  **1.0e-7** (GPU); cov with auto extent **REL 3.95e-6** PSD; numpy BYTE-IDENTICAL
  (maxdiff 0.0); 35 backend + 38 numpy tests pass. StreamTrack tp_scale unit test
  (test_st_scale.py): d(x)/d(scale) AD=FD REL 2e-11.
- **streamspray now EXACTLY matches eager functionality under jit** (sample + streamTrack
  + auto track_time_range + covariance, all differentiable + GPU + byte-identical).
  Only remaining follow-up: fixed P-spline lambda (GCV); C param-diffs (Track C) NOT yet.

### DONE (2026-07-12d) — jittable GCV lambda (commit `a595af23`)
`_fit_track_backend_jit` now GCV-selects the P-spline lambda (matching eager
make_smoothing_spline auto-tuning) instead of a fixed value: minimize
N||(I-H)off||^2/(N-tr H)^2 over a log-spaced concrete grid (the grid-search + argmin
unroll under jit; backend-agnostic; H=B A^-1 B', tr H=tr(A^-1 B'B)). lambda chosen on
the stop_gradient offsets -> a FROZEN structural hyperparameter, so the track gradient
is unaffected. Validated: jit streamTrack grad-vs-FD REL 4.9e-7; GCV track SMOOTH
(2nd-diff RMS 1.5e-7, not near-interpolation). jit-only -> numpy byte-identical.

### OPTION A + all follow-ups COMPLETE
streamspray FULLY jittable + GPU + differentiable + byte-identical, EXACTLY matching
eager functionality: sample(), streamTrack() (mean + PSD covariance), auto
track_time_range (true particle-extent), GCV smoothing. Nothing left on the jit-parity
list. C param-diffs (Track C, the fast eager path) remains user-deferred.

### (historical) Remaining before this: streamTrack() reconstruction
`jax.jit(streamTrack())` fails in `_fit_track_from_particles` (streamTrack.py): cKDTree
+ make_smoothing_spline on traced particles, AND dynamic-shape ops (`tp_assign[interior]`
boolean filter, `numpy.percentile` trim) that jit can't do. Needs a SEPARATE jax-native
path (M2 recipe: static-shape argmin closest-point + raw-data P-spline, where-mask instead
of boolean-filter, sorted-quantile instead of percentile) + a jittable `StreamTrack.x(tp)`.
Design-laden (fidelity-vs-M2, dynamic-shape handling, StreamTrack storage) — good place for
user input. numpy/eager paths already work + are byte-identical.

## Next (milestones)
- **M2**: jax-native reconstruction matching galpy fidelity (covariance-weighted smoothing,
  velocity-weight, smooth spline instead of histogram) so the frozen-vs-moving gap shrinks.
- **M3**: wire into `streamspraydf.streamTrack()` as the jax generative path; API-compatible;
  tests (physical grad-vs-FD, jit, GPU). Likely reshapes #1106's dispatch (held open for this).

## LANDED into PR #1106 (2026-07-12)
Folded all 9 commits (f7ac8b5d..b26e5121) into the open #1106 branch
`feat/cintegrator-backend-param` (fast-forward from the old 66837a7f base, which
kept the `_finalize_pot_args` byte-identical C-path piece). Retitled #1106 →
"streamspraydf: fully jittable + differentiable stream track/spray (jax/torch,
GPU) — generative ∂track/∂θ" and rewrote the body for the expanded scope
(generative jittable sample()/streamTrack(), auto extent, cov, GCV, tp_scale,
o.x fix, C-STM preserved, numpy byte-identical, all grad-vs-FD h-converged).
Pushed b26e5121; CI queued. The old 66837a7f All-backend "failure" was a GitHub
artifact-upload infra timeout (tests were green: 179 passed, 0 un-ledgered), not
a real red. New jit tests run in build.yml's `test_backend*.py` jobs (REQUIRES_JAX
→ jax+diffrax installed); not in the forced all-backend shard.
