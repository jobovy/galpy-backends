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

## 2026-06-16 — g13 backend-port burndown (progress vs time)

- **g13_ledger_burndown** (NEW; sha256 `cff702301097…`) — the migration's
  PROGRESS metric: # tests still in the `xfail(strict=False)` ledger (still
  failing under jax / torch) vs time. Reconstructed **exactly from git history**
  of `tests/backend_xfail.txt` + `backend_slow_skip.txt` on `feat/backends`
  (`measure/burndown_from_git.py`) — no manual logging, auto-grows one point per
  ledger-touching commit. Baseline 2026-06-14: jax 355, torch 1670; #959 drops
  jax → 324. Falls to 0 when the full suite passes under both backends at numpy
  tolerances. Dashed lines = milestones (`data/milestones.json`, hand-annotated).
  Re-run the measure script + `run` at each milestone review (see plan Pillar 1
  ledger-burndown policy) to extend the series. The torch≫jax early gap is the
  systemic scalar-rejection that #960's central coercion fixes (first milestone).

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

---

## 2026-06-13 — g12 same-work jax-Dopri5 vs torch-dopri5 (+work counts) + torch rise to 1e6

g12's jax-GPU-vs-torch-GPU crossover was NOT same-work: jax SHIPS diffrax **Dopri8**
(8th order), torch SHIPS torchdiffeq **dopri5** (5th order — chosen because
torchdiffeq's dopri8 *backward* is buggy, ~1e-5 grad error; see
`galpy.backend._torch.orbit_ode`). Different methods → different work/orbit, so the
shipped lines can't be read as a backend kernel-speed comparison. Fixed by quantifying
the work and adding a matched 5th-order jax line.

**Work counts** (`measure/work_counts.py` → `data/work_counts.json`; nfev via a
`jax.debug.callback` that fires once per RHS call under diffrax, and a plain Python
counter on the eager torchdiffeq func). ONE Sun-like orbit, rtol=atol=1e-8
(accepted steps / nfev):
- jax  diffrax **Dopri8**: **189 / 3056**  (8th order: few big steps, ~16 evals/acc-step incl. rejections)
- jax  diffrax **Dopri5**: **545 / 3547**  (5th order: ~3× more steps, ~6.5 evals/acc-step)
- torch torchdiffeq **dopri5**: **~578 total steps / 3470**
→ **jax-Dopri5 and torch-dopri5 are genuinely same-work** (matching step + nfev
counts; the small Δ is rejections vs the torch batch's shared controller). Dopri8 nets
a little *lower* nfev than dopri5 (fewer steps × more evals).

**New same-work jax-Dopri5 GPU line** (`measure/jax_dopri5_measure.py`, local to g12,
NOT shared — the shared `jax_*.json` stay the shipped Dopri8 curve g8 also uses).
Same orbit/tol/nt convention, jit+vmap, GPU, serial. lowN (nt=1000) +
hiN (nt=16) → `data/jax_dopri5_gpu[_hiN].json`. Reaches ~0.13 ms/orbit at N=3e5
(cheaper-per-step Dopri5 beats Dopri8's ~0.21 on the GPU despite more steps).

**Same-work verdict at large N:** jax-Dopri5 ~0.13 ms/orbit vs torch-dopri5 ~0.069
ms/orbit at N=3e5. Torch's edge is therefore NOT the method (same nfev/steps) — it's
torchdiffeq's single *shared* (6,N) adaptive controller (one dt sequence, one fused
per-step kernel for the whole batch) vs jax's per-orbit vmap-of-solve (N independent
controllers). Identical to g8's galax batched-vs-per-orbit story; galpy-jax keeps
per-orbit stepping for per-orbit error guarantees.

**torch rise extended to N=1e6** (`measure/torch_measure.py`, hiN nt=16). Added
intermediate 5e5, 7e5 and retried 1e6 — run with **NO jax import** + 
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reclaim the ~322 MiB that OOMed
1e6 before. **1e6 now FITS** (67.3 s, 0.0673 ms/orbit; no `stopped_at`). The
compute-bound rise is now well-resolved: 20.8 → 34.4 → 47.9 → 67.3 s for
3e5/5e5/7e5/1e6, ms/orbit flat ~0.069.

**g8 unchanged** (checked): its galpy-jax AND galax lines are BOTH diffrax Dopri8, and
the caption says so — g8's same-work claim is already honest, no annotation needed.

Re-rendered + manifest updated: **g12 sha256 `d102b7aa8c05…`**, `verify` OK
(byte-stable from JSON). All GPU jobs run STRICTLY SERIALLY (verified GPU idle via
nvidia-smi between each). g7 pre-existing-red (untouched).

---

## 2026-06-13 — g8/g12 made cross-consistent (shared orbit+C+jax) + torch large-N + legend fix

g8 and g12 both integrate MWPotential2014 for 2 Gyr but were INCONSISTENT on their
shared series — g8's galpy-C used nt=1000 dense / no pinned tol (galpy's 1e-10
default) → ~6.8 ms/orbit, while g12's used nt=16 / rtol=atol=1e-8 → ~2.1 ms/orbit.
Same integrator, different sampling+tol, so the two figures could not be read
together. Fixed by establishing ONE canonical parameterization for the series that
appear in both figures and having BOTH load the same JSONs.

**Unified parameterization (the now-shared convention):**
- Same orbit: MWPotential2014, Sun-like IC [1,0,1,0.0125,0.036,0] (internal units),
  per-N-seeded Gaussian perturbation, 2 Gyr = linspace(0, 56.2, nt).
- Same tolerance: **rtol = atol = 1e-8** for ALL lines (galpy-C, galpy-jax CPU/GPU,
  galax) — the galax OrbitSolver default; galpy's C and in-backend integrators
  called with it instead of their tighter 1e-10 default.
- Same nt convention: **lowN regime nt=1000 dense saves** (resolved small-N
  crossover); **hiN regime nt=16 sparse saves** (fits the batch in 12 GiB at large
  N). Integration is adaptive-step-bound not save-bound, so per-orbit cost matches
  across regimes and the overlap points agree → one consistent curve each.

**Where the shared JSONs live:** new `paper/figures/_shared/orbit_speed/` dir with
`galpy_c_measure.py` + `galpy_jax_measure.py` (lowN/hiN × CPU/GPU) writing
`data/{galpy_c_cpu[_hiN],jax_{cpu,gpu}[_hiN]}.json`. BOTH g8 and g12 `_load_one`
search `_shared/orbit_speed/data` first, then the figure-local `data/` (galax stays
local to g8, torch local to g12). The old per-figure galpy/jax JSONs+scripts were
deleted. Confirmed: g8 and g12 load **byte-identical** galpy-C (6 pts),
galpy-jax-CPU (5 pts), galpy-jax-GPU (8 pts). The single backend worktree
`/var/tmp/backends-merge` (feat/backends @36c92e8f) now ships libgalpy, so the C
baseline AND the in-backend jax/torch lines all run from the SAME galpy.

Canonical numbers (TITAN Xp, py313 env, jax 0.10.1 + torch 2.12.0+cu126, GPU jobs
run STRICTLY SERIALLY):
- **galpy-C**: ~2.1 ms/orbit constant (2.10 lowN / 2.07 hiN at N≥1e3 overlap).
- **galpy-jax-GPU**: 1.65 s for one orbit (launch floor) → 0.291 ms/orbit @ N=1e4,
  0.242 @ N=1e5, **0.212 @ N=3e5** (headline peak); lowN dense reaches N=3e4, hiN
  reaches N=3e5 (N=1e5 dense still OOMs in 12 GiB → why hiN exists).
- **galpy-jax-CPU** (NEW in g8, was g12-only): ~6–8 ms/orbit at large N (N=1..1e4;
  CPU N=3e4 dense capped — impractically slow, not needed for the device reference).

**g12 torch extended to LARGE N** (`measure/torch_measure.py`, hiN nt=16 convention,
200 s/run wall budget): pushed N from the old ~1e4 stop out to OOM. torchdiffeq runs
eager, so the batch is *overhead-bound* (~9.6–10.3 s wall, GPU-starved, ms/orbit ∝
1/N) for N=1..1e5, then crosses into the *compute-bound* regime — **wall time turns
up at N=3e5 (21.7 s)** and ms/orbit flattens toward torch's per-orbit asymptote
(0.103 @ 1e5 → **0.072 ms/orbit @ 3e5**). **N=1e6 OOMs** (tried to allocate 322 MiB,
only 286 MiB free on the 11.9 GiB card) — recorded as `stopped_at` in the JSON. This
captures the eager overhead→compute transition the figure previously missed.

**Legend fix:** both g8 and g12 legends moved OUTSIDE the axes (upper-left anchored at
bbox (1.01, 1.0)) so they overlap no data — notably g12's near-flat eager torch line
across the top that the old in-axes upper-left legend covered. Env-stamp annotations
moved to the bottom-left corner.

Re-rendered + manifest updated (matplotlib 3.10.8): **g8 sha256 `50e0f526c33b…`**,
**g12 sha256 `4adce1e02114…`**; both `verify` OK (byte-stable from JSON). Docstrings
+ captions updated to state the shared orbit/tol/nt convention and the torch
eager-vs-compute story. NOTE: g7_orbit_sensitivity has a PRE-EXISTING verify mismatch
(expected edb8f55f52b3, got a741150dc35c) unrelated to this work — left untouched,
still red.

## 2026-06-12 — GPU measurements: g2/g6 re-measured + new g12 (in-backend orbit GPU)

Re-ran the GPU benchmark figures on the TITAN Xp (sm_61) against the rebased
backend worktree (`/var/tmp/backends-merge`, feat/backends @36c92e8f), in the
`py313` env (jax 0.10.1 + diffrax 0.7.2, torch 2.12.0+cu126 + torchdiffeq 0.2.5).
The pure-backend jax/torch paths don't need the C extension (libgalpy-not-loaded
warning ignored); the galpy-C baseline used the C-built main env
(`/home/bovy/Repos/galpy`).

- **g2_throughput** (sha256 `bc47b1552dbc…`) — re-measured Plummer `F_R` throughput
  vs N. jax-GPU compute-only peaks ~2.3e9 pts/s at N=1e7 vs numpy-CPU ~3.6e7
  (~64×); torch-GPU ~1.5e9. End-to-end (host↔GPU transfer) saturates an order of
  magnitude below compute-only (PCIe-bound). CPU baseline kept; structure
  (compute-only solid / +transfer dashed) unchanged.
- **g6_jit_amortization** (sha256 `c93b6b93034d…`) — re-measured jit compile-once/
  run-many for Plummer `F_R`. Compile ~95 ms (N=1e4) / ~122 ms (N=1e6); steady-state
  0.12 / 0.91 ms; eager (un-jit) 1.60 / 2.22 ms ⇒ jit-vs-eager break-even ≈ 65 / 94
  calls.
- **g12_inbackend_orbit_gpu** (NEW; sha256 `0e0e0f56e91f…`) — galpy's OWN in-backend
  differentiable integrator (`galpy.backend._reference.integrate_orbit`, #906), the
  analogue of g8 but galpy-only: MWPotential2014, Sun-like orbit, 2 Gyr, batch of N,
  rtol=atol=1e-8, per-orbit adaptive stepping. Four series (cross-env `measure/`
  scripts → per-backend machine/version-stamped `data/*.json`; plot byte-stable):
  - **galpy (C, CPU; sequential)** dop853_c — ~2.1 ms/orbit, constant; 2.06 s @ N=1e3,
    21.2 s @ N=1e4, 67.8 s @ N=3e4.
  - **galpy in-backend (JAX, GPU)** diffrax Dopri8 (jit+vmap) — 1.56 s for one orbit
    (launch/compile floor), amortising to **0.28 ms/orbit @ N=1e4, 0.25 @ N=1e5,
    0.21 @ N=3e5** (the clean peak); compile (excluded from steady wall) ~5 s small-N
    → ~28 s @ N=1e5. Crossover with galpy-C between N=1e3 (C 2.06 s < GPU 2.49 s) and
    N=1e4 (C 21.2 s ≫ GPU 2.82 s) → **~7× faster per orbit at N=1e4**. lowN (1000
    dense saves) **OOMs at N=1e5** in 12 GiB (same cap g8 hit); hiN (16 saves) reaches
    N=3e5 (N=1e6 attempted but XLA compile of the 1e6-orbit vmap thrashed >13 min and
    was abandoned — 3e5 is the headline peak). NB: the GPU benchmarks were run STRICTLY
    SERIALLY (one device-timed job at a time) — an initial N=1e5 point timed while a
    torch job was co-resident read 0.34 ms/orbit (contention); re-measured GPU-exclusive
    it is 0.25, consistent with the trend.
  - **galpy in-backend (JAX, CPU)** — same diffrax path on CPU, ~5–8 ms/orbit (device
    contribution isolated).
  - **galpy in-backend (torch, GPU; eager)** torchdiffeq dopri5, state (6,N) batched —
    runs EAGER (no jit-fuse): the whole batch takes a near-constant ~12–15 s
    regardless of N (Python-overhead/step-loop-bound, GPU-starved), so per-orbit only
    reaches ~1.2 ms @ N=1e4 — ~4× slower than the jit-fused JAX-GPU line and ~40× more
    wall time. Correct + differentiable (the #906 in-backend tests pass vs C and FD
    gradients), but not the batched-throughput path — exactly why g8 omits torch.
  POTENTIAL SUPPORT: as of #949 (36c92e8f) MWPotential2014 (PowerSphericalCutoff +
  MiyamotoNagai + NFW) is fully in-backend-integrable on jax AND torch — the earlier
  "Plummer/Isochrone only" status no longer holds, so g12 uses the same MW model as g8.

## g15 — cost of the IC derivative ∂(orbit)/∂(x0): C-STM vs in-backend AD (CPU+GPU)

Companion to g12, but for *differentiating* the orbit w.r.t. its initial conditions.
`measure/measure_icderiv.py` (single-orbit) + `measure/measure_scaling.py` +
`measure/measure_largeN.py` → `data/icderiv.json`; plot byte-stable.

- Mechanisms (full 6-vector gradient of the summed final R): **C-STM** (the C
  variational integrator; M=∂x(t)/∂x0 is a byproduct of the variational forward solve);
  **in-backend AD** diffrax (jax, vmap-batched) and torchdiffeq (batched as the g12
  `(6,N)` shared-controller state, then backprop). Finite-difference dropped (the user:
  "not reliable"). MWPotential2014, nt=100, CPU float64 + TITAN Xp.
- **Single orbit (CPU):** C-STM ≈ 66 ms ≈ diffrax-CPU 61 ms ≪ diffrax-GPU 2.1 s ≪
  torch-CPU 4.5 s ≪ torch-GPU 13.4 s. (Note: "forward vs grad" was dropped — not
  apples-to-apples; C-STM's *variational* forward already builds M, so grad≈forward.)
- **Batch scaling:** C-STM-CPU **linear** (loops over orbits; 93 ms→13.8 s over N=1→256,
  capped). diffrax-CPU (vmap) sub-linear (109 ms→6.4 s @ N=1024). diffrax-GPU **flat
  ~2 s** (launch-overhead-bound) → **crosses diffrax-CPU at N≈300–500**, ~3× faster by
  N=1024 — the batch crossover. Batched torch flat (CPU ~2.7 s, GPU ~7.5 s; high eager
  floor). Large-N sweep (measure_largeN.py) pushes the flat lines to the
  compute-bound bend / memory ceiling.
- **Two C-STM optimisations surfaced (tasks #59, #60):** (#59) the current
  `c_stm_forward` builds M one column at a time = **6 `integrate_dxdv` solves**; a single
  C solve of Ṁ=J·M (J once/step, all 6 columns) ≈ 6× faster — for the full Jacobian.
  (#60) C-STM is natively forward-sensitivity (1 solve = 1 column = 1 JVP); a
  `custom_jvp` would give a **single** directional sensitivity (e.g. ∂R/∂R0) in 1 solve
  instead of materialising the full M. Both will drop the C-STM line; re-measure after.

## g14 — "The Great Escape" hypervelocity-star potential/IC fit (PLANNED, hybrid)

S5-HVS1 (Koposov+19, arXiv:1907.11725): a star ejected by Sgr A* must, integrated
*backward*, return to the Galactic Centre. Worked example composing BOTH gradient
mechanisms (user-confirmed scope = full hybrid in one figure):
- **C-STM IC-fit (fixed potential):** find the HVS phase-space within its observational
  uncertainties that minimises closest approach to the GC — an **IC**-gradient problem,
  C-STM's wheelhouse (verified: finite ∂(min r²)/∂IC via the STM, ∂/∂φ≈0 by symmetry).
- **In-backend AD potential fit:** add a potential knob (halo flattening / solar V_y)
  and fit it — a **parameter** gradient, which C-STM cannot do (verified: finite
  ∂(min r²)/∂q through the backward diffrax solve; GD converges).
Data: a mock star ejected from the GC in a fiducial MW potential (self-validating —
recovers a known truth), made S5-HVS1-like; real S5-HVS1 observables as a follow-on
variant. Ships `skeleton.py` (the minimal runnable core) that make.py imports, so the
paper listing is exactly the code that ran — the reproducible-skeleton convention.
