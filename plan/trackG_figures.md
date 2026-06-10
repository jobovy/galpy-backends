# Track G — Figures & diagnostics for the paper(s)

> Summary/diagnostic plots that go into the eventual paper(s): backend
> comparisons, speed/scaling benchmarks (CPU now, **GPU once #891/#898 land**),
> worked autodiff examples, and derivative visualizations (orbits → stream
> tracks). **Hard requirement: every figure ships with version-controlled code
> that re-runs deterministically and reproduces the identical plot**, so against
> the *final* galpy code we can re-render and diff every figure in the paper.

This track is cross-cutting: it consumes the other tracks' deliverables (backends,
variational STM, autodiff wrappers, action-angle, df/streams) and turns them into
paper-ready evidence. It runs *continuously* alongside the implementation tracks —
each track produces its figures as it lands, not in a big batch at the end.

---

## G0 — Figure harness (do this first; everything else plugs into it)

A tiny, dependency-light harness so "every figure has reproducible code" is
enforced structurally, not by discipline.

Layout (in `galpy-jax/`):
```
paper/figures/
  _harness/
    figure.py        # @figure(name, ...) decorator + registry + run/verify CLI
    style.py         # shared matplotlib rcParams (fonts, sizes, colorblind palette)
    backends.py      # helpers: run_under(backend, fn), available_backends(), device label
    bench.py         # timing helpers (warmup, repeats, jax block_until_ready / torch sync)
    manifest.json    # generated: name -> {script, sha256(pdf), data hash, env, git sha}
  <figname>/
    make.py          # imports the harness; produces <figname>.pdf (+ optional .png)
    data/            # cached inputs/results (only if expensive); hashed in manifest
    <figname>.pdf    # the committed artifact
    README.md        # one-paragraph: what it shows, which track/PR, how to re-run
```

Harness contract:
- `@figure(name="...", track="G/B/...", reproducible_seed=0)` wraps a `make()` that
  returns a `matplotlib.Figure`. The decorator: seeds numpy/jax/torch deterministically,
  applies `style.py`, saves `<figname>.pdf` with **fixed metadata** (no embedded
  timestamps — set `matplotlib` `metadata={'CreationDate': None}` and a pinned
  `Producer`) so the bytes are stable, and records `sha256` + git sha + env into
  `manifest.json`.
- CLI:
  - `python -m paper.figures._harness.figure run [name...]` — (re)generate.
  - `python -m paper.figures._harness.figure verify [name...]` — regenerate to a temp
    file and assert the sha256 matches the manifest → **this is the reproducibility
    gate** we run against the final code. Numerical-but-not-byte-stable plots fall
    back to an image-diff tolerance (SSIM/`matplotlib` `compare_images`) recorded in
    the manifest.
  - `python -m paper.figures._harness.figure list` — table of figures × track × status.
- Determinism rules (mirror the tolerance policy): jax `enable_x64`; fixed RNG seeds
  threaded everywhere; pin benchmark machine/env in the caption metadata (timings are
  reported with the env, and the *plot-from-saved-data* path is byte-stable even if a
  re-timed run differs). Benchmarks therefore separate **measure** (writes `data/*.json`,
  machine-dependent) from **plot** (reads `data/*.json`, byte-stable). `verify` checks
  the plot step; a `--remeasure` flag re-runs timings.
- A CI-able `verify-all` target (later, once the final code exists) re-renders every
  figure and fails on any mismatch — the guarantee the user asked for.

**Deliverable:** the harness + one trivial smoke figure (`g0_smoke`: a parabola) that
exercises `run`/`verify`/`manifest`.

---

## G1 — Backend comparison figures (numpy / jax / torch / array-API)
*Consumes: Track 0, Track A. Available now (CPU).*

- **Value parity grid.** For a representative set of potentials × methods
  (`_evaluate`, forces, all 6 second derivatives) over a grid of (R, z, φ): heatmap /
  scatter of `|numpy − jax|`, `|numpy − torch|` (should be ≲1e-13 fp64). Demonstrates
  "same answer, every backend." Pairs with the tolerance ledger (appendix E).
- **Identity-consistency panel.** `d(_evaluate)/dR + _Rforce`, `d(−_Rforce)/dR −
  _R2deriv`, etc., backend autodiff vs analytic — residuals near machine epsilon.
- **Device/dtype preservation.** Small table-figure: input dtype/device → output
  dtype/device unchanged, across backends.

## G2 — Speed / scaling benchmarks (CPU now; **GPU after #891/#898 land**)
*Consumes: Track 0/A/D. CPU figures now; GPU figures when moved to the GPU box.*

- **Throughput vs N** (vectorized potential/force eval): numpy/C vs jax(CPU) vs
  torch(CPU) vs **jax(GPU)/torch(GPU)**; log-log, crossover annotated. This is the
  headline "GPU wins at large N" plot.
- **Orbit-integration speed**: C-STM vs in-backend diffrax/torchdiffeq, single vs
  batched (many orbits), CPU vs GPU. Wall-clock + per-orbit.
- **Autodiff overhead**: forward eval vs grad (jacrev/jacfwd, torch backward) cost
  ratio vs N and vs #parameters.
- **jit/compile amortization**: first-call vs steady-state for jax `jit` / torch
  compile.
- Methodology: `bench.py` does warmup + median-of-k with `block_until_ready` /
  `torch.cuda.synchronize`; raw timings cached to `data/*.json` (machine-stamped);
  the plotted figure is byte-stable from that JSON. **GPU runs are gated behind a
  `--device` selector and re-measured on the GPU machine; the plotting code is
  identical.**

## G3 — Autodiff worked examples (the "look, it differentiates" figures)
*Consumes: Track A/D (orbits), Track E (action-angle), Track F (df/streams).*

- **Potential parameter gradients**: e.g. ∂(rotation curve)/∂(a,b,amp) for
  MiyamotoNagai; ∂(force field)/∂θ quiver/heatmap — analytic vs AD overlaid.
- **Orbit sensitivity to ICs**: the STM `M(t)=∂x(t)/∂x₀` columns visualized along an
  orbit (fan of perturbed orbits + the linearized prediction), axisymmetric and
  triaxial. Ties directly to Track B validation.
- **Orbit sensitivity to potential parameters**: ∂x(t)/∂θ vs finite-difference;
  gradient-descent "fit an orbit to data" toy demo (loss vs iteration) to show the
  stack end-to-end.
- **Action-angle gradients**: ∂J/∂(x,v), ∂Ω/∂θ — analytic-rule vs AD vs FD.
- **Stream-track derivatives** (later, the showpiece): ∂(stream track)/∂(potential
  params) — e.g. how the GD-1-like track shifts with halo flattening/mass; gradient
  arrows on the track. The "differentiable streams" figure the project is for.

## G4 — Variational / dynamics diagnostics
*Consumes: Track B (Pvar-rhs, Pvar-lyapunov).*

- **Liouville/symplecticity drift**: `det M(t)−1` and `‖MᵀΩM−Ω‖` vs t across
  integrators — the conservation evidence (companion to the test battery).
- **Lyapunov**: integrable→0 (power-law deviation growth) vs chaotic (exponential);
  Hénon–Heiles / flattened-log SALI-style or finite-time-Lyapunov maps reproducing
  literature.
- **2D-reduction bridge**: planar-restricted 3D STM vs trusted planar STM (residual).

---

## Sequencing & ownership
- **Now (CPU):** G0 harness, G1 (backends landing in #892 + P2.*), G4 (Track B; #898
  STM already validated → first real figures can be drafted from the K-validator
  data). These also double as living regression evidence for the PRs.
- **After #891/#898 land + GPU machine:** G2 GPU benchmarks (re-measure on GPU; plot
  code unchanged), then G3 as Track D/E/F land.
- Each implementation PR that produces a paper-worthy result should drop its figure
  under `paper/figures/<name>/` with `make.py` + README and register it — so the
  figure set grows with the code and `verify-all` against the final code is the
  acceptance test.

## Paper mapping
Figures map to paper sections 1:1 with the tracks: G1→backends, G2→performance,
G3/G4→autodiff & variational/dynamics, G3(stream)→the streams showcase.

---

## 2026-06-10 — G4 delivered: variational-equations figures g9/g10/g11

Three paper figures for the variational section, all following the g8
cross-environment pattern (standalone `measure/` scripts run against the
Track-B galpy worktrees writing machine/worktree-stamped `data/*.json`;
plot-only `make.py` registered with the harness, byte-stable from the JSONs;
`verify` passes for all three):

- **g9_variational_validity** (sha256 `74c2b3f68633…`) — measured against
  `/var/tmp/pvar-lyap` (feat/pvar-lyapunov @7ca49a1b + local fix, see below).
  (a) Liouville `|det M(t)−1|` ≤ 2.6e-8 (3D 6×6) / 1.3e-6 (planar 4×4) over
  t=0–100, MWPotential2014, dop853_c 1e-12; (b) symplecticity
  `‖MᵀΩM−Ω‖max` ≤ 1.0e-6 (3D) / 1.3e-6 (planar); (c) FD-of-the-flow V-curve
  at t=10: forward arm ∝ε, central arm ∝ε², roundoff arm ∝1/ε, STM matching
  the best central FD at the 1.3e-8 floor (ε=1e-6) — the ground-truth panel
  (det/symp are necessary-not-sufficient); (d) 2D-reduction bridge ≤ 1.0e-8
  over t=0–100 with exactly zero out-of-plane leakage.
- **g10_lyapunov** (sha256 `4231b0708ec7…`) — same worktree. Hénon–Heiles
  E=1/8, t→3e4: chaotic orbit F converges to λ=0.0456 (0.0425 at t=1.5e4),
  inside the 0.04–0.08 literature band; regular orbit E at the same energy
  decays ~ln(t)/t to 2.7e-4 (×170 separation). MWPotential2014 Sun-like 3D
  orbit, t→3e3: λ(t) tracks ln(t)/t to 2.8e-3 (≈0.08/Gyr upper limit).
- **g11_dissipative_phase_volume** (sha256 `7825e333ee59…`) — measured against
  `/var/tmp/pvar-dissip` (@43451e7a) and `/var/tmp/pvt-fdm` (@36615802).
  det M(t) from the 6×6 STM vs exp(∫tr(∂F/∂v)dt′) from the C
  `evalRectForceJacobian` trace, decaying orbit in MWPotential2014 + friction
  (GMs=0.008): Chandrasekhar contracts 5+ decades (det M(10)=6.1e-6,
  r→0.05), FDM at identical GMs (mℏ⁻¹=10, dispersion regime,
  C_FDM/C_CDM∈[0.48,0.59]) only to 0.52 (r→0.79). Max relative difference of
  the two sides: 5.7e-6 (Chandrasekhar; trapezoid-quadrature-limited O(h²)) /
  3.5e-7 (FDM).

**Bug found & locally fixed while making g10** (needs landing on
feat/pvar-lyapunov, + the same latent bug in the dissip/fdm branches):
`integrateFullOrbit_dxdv` corrupts the returned *base orbit* — `rect_to_cyl`
passes `Z` through **by reference**, so `Zout` aliases `out[...,2]`, which is
overwritten with `vT` before `out[...,3] = Zout` runs ⇒ the returned z-column
equals vT (and with `rectOut=False` the deviation Jacobian is evaluated at
the wrong z too). The deviation columns with `rectOut=True` (all of
g9/g11) are unaffected, but `Orbit.lyapunov` restarts every renormalization
segment from the corrupted orbit: unfixed, 3D λ estimates plateau at a
spurious ~0.05 *even for the integrable isochrone* (and depend on the
renormalization interval). The existing test battery missed it because no
test consumed the base-orbit columns of the 3D dxdv output, and
test_lyapunov_integrable's fine grid happened to pass its loose threshold.
Fix: copy `Zout`/`vzout` before the in-place column assignments
(uncommitted in /var/tmp/pvar-lyap; data stamped `galpy_dirty: true`).
Post-fix validation: segmented vs single-shot deviation propagation agrees to
7e-13; isochrone λ identical for renorm intervals 1 and 10 (0.006414);
MW2014 regular orbits decay ~ln(t)/t as they must. Suggested regression
tests: base-orbit columns of 3D integrate_dxdv vs plain integrate;
renorm-interval independence of lyapunov.
