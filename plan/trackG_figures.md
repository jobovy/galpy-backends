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
