# Investigation: galpy-jax-GPU vs galax-GPU orbit-integration speed gap

Date: 2026-06-09. Machine: NVIDIA TITAN Xp (sm_61). jax 0.10.1 (galpy, py313 env) /
jax 0.8.0 (galax, its own uv-locked venv). All numbers ms / orbit at **N = 1e5**
(per-orbit cost is flat in the GPU-saturated regime). Solver: diffrax `Dopri8`
with a `PIDController` in *both* codes. Scripts: `scripts/{galpy_jax_ladder,galax_ladder}.py`.

## Question
The g8 figure showed galpy-jax-GPU ~0.31 ms/orbit on MWPotential2014 vs galax-GPU
~0.07 — a ~4.5x gap. Is the expensive `PowerSphericalPotentialwCutoff` (PSCw) bulge
to blame? Is galpy's per-call backend dispatch (`get_namespace`) a runtime cost? Can
a lower-overhead implementation close it?

## Findings (summary)
The original ~4.5x decomposes into three independent pieces; none is "galpy framework
overhead", and the single biggest *real* contributor is exactly the PSCw bulge.

### 1. Tolerance mismatch — ~1.6x, a BENCHMARK ARTIFACT
galpy's in-backend integrator defaults to `rtol=atol=1e-10`; galax's `OrbitSolver`
defaults to `1e-8`. Same solver (Dopri8), same controller type — so galpy was simply
taking more adaptive steps. At matched `1e-8`:

| potential        | galpy 1e-10 | galpy 1e-8 |
|------------------|------------:|-----------:|
| Hernquist        | 0.075       | 0.046      |
| NFW              | 0.076       | 0.046      |
| MiyamotoNagai    | 0.163       | 0.096      |
| PSCwCutoff       | 0.095       | 0.058      |
| MWPotential2014  | 0.357       | 0.246      |

→ **Action: the g8 benchmark must match tolerances (1e-8).** That alone takes the
MW gap from ~4.5x to ~3.1x and is the honest comparison.

### 2. Backend dispatch (`get_namespace`) — ZERO runtime cost
`jax.make_jaxpr` of a migrated `_Rforce` is pure arithmetic (`pow/add/sqrt/div/neg`,
~14 eqns, **no callbacks, no dispatch primitives**). `array_namespace(*inputs)` runs
at *trace* time and is constant-folded out of the compiled XLA graph. The "choose the
backend every call" cost is entirely jitted away — not a factor.

### 3. PSCw `gammainc` bulge — CONFIRMED disproportionately expensive
Clean bulge-swap using the **real** MWPotential2014 components (same Sun orbit, 1e-8):

| MWPotential2014 variant      | ms/orbit |
|------------------------------|---------:|
| full (PSCw bulge)            | 0.241    |
| disk+halo only (no bulge)    | 0.152    |
| **Hernquist** bulge+disk+halo| 0.154    |

The PSCw bulge adds **0.089 ms/orbit (~37% of the total)** despite being the lightest
component (amp 0.030 vs disk 0.76, halo 4.85). A Hernquist bulge adds ~nothing (0.002).
The cost is the incomplete-gamma (`gammainc`) in the enclosed-mass / force. galax pays
a similar but ~2x smaller penalty for its `PowerLawCutoffPotential` bulge: galax
BovyMW2014 = 0.080, galax MW-with-Hernquist-bulge = 0.039, i.e. galax's cutoff bulge
adds ~0.041.

→ **Action: galpy's `gammainc` (Pspecial) is the optimization target** — it is ~2x
galax's. Swapping the bulge for a non-special-function form would cut galpy's MW
orbit cost by ~1.6x (0.241 -> 0.154).

### 4. Force-layer structure is NOT the problem
Hypothesis: galpy's separate `Rforce`/`zforce`/`phitorque` per component (vs galax's
single fused Cartesian gradient) wastes work. Tested by replacing the EOM with a fused
`accel = -jax.grad(Phi_total)` (one autodiff pass, à la galax): **0.241 vs 0.246** —
no change. So the per-step cost is the raw cost of galpy's force *expressions* + the
cyl<->rect handling, not the aggregation structure. A fused gradient would not help.

### 5. Matched full-MW residual
At matched tol/orbit/solver, galpy MWPotential2014 (0.241) vs galax BovyMW2014 (0.080)
is ~3x. Roughly: ~0.089 of galpy's is the gammainc bulge (galax ~0.041); the remaining
disk+halo evaluation is also somewhat heavier in galpy than galax. (A fully matched
cross-code disk+halo number was not isolated — galax's standalone-potential runs used
galax-native parameters, so their orbits/step-counts differ; only the full-MW
comparison is orbit-matched, because galpy MWPotential2014 over t=56.2 natural units
*is* the physical 2-Gyr Sun orbit in BovyMW2014.)

## Bottom line
- ~1.6x of the headline gap was a **tolerance artifact** (fix the benchmark).
- Backend dispatch is **free**.
- The PSCw **`gammainc` bulge is the single biggest real contributor (~37%)** and is
  ~2x more expensive in galpy than in galax — confirming the original hypothesis. It
  is the concrete optimization target (Pspecial `gammainc`), not the framework.
- For a simple potential at matched tolerance galpy is within ~1.5x of galax (and
  ~parity for spherical), so there is no large fixed framework penalty.

## Dead-ends
- Fused `jax.grad(Phi)` EOM: no speedup (ruled out force-layer structure).
- Cross-code per-potential ratios from standalone potentials: confounded by
  parameter/step-count differences; only the full-MW comparison is orbit-matched.

## Reproduce
- galpy ladder: `python scripts/galpy_jax_ladder.py` (py313 env, jax-GPU; PSCw/MW need
  the migrated PSCw from feat/backends / #922).
- galax ladder: run `scripts/galax_ladder.py` in galax's uv venv (see
  `../../paper/figures/_setup/galax_gpu_setup.md`).

## Follow-up: WHY is galax's cutoff bulge ~2x faster — same `gammainc`?

Yes, it is literally the same function. galpy's special-function router does **not**
fall back for `gammainc` on jax (it is in neither `_NATIVE_MISSING` nor
`_NATIVE_UNRELIABLE`), so it routes to native `jax.scipy.special.gammainc` — the same
XLA `igamma` kernel galax calls. Identical per-call cost.

The difference is the **number of `igamma` evaluations per RHS step** (counted via
`jax.make_jaxpr`):

| code  | igamma per step | why |
|-------|----------------:|-----|
| galpy | **2** | the EOM calls `_Rforce` and `_zforce` separately; each calls the spherical `_rforce(r)` (which contains the `igamma`), and XLA does **not** CSE the duplicate custom `igamma`. |
| galax | **1** | acceleration = `-grad(Phi)`; `Phi` has one `igamma`, and autodiff differentiates that single call into all three Cartesian components (the `d/dx gammainc` is a closed-form exp/pow, not another `igamma`). |

This is why a cheap potential (Hernquist) is at parity — recomputing its `_rforce`
twice is free — but PSCw, whose `_rforce` carries an expensive `igamma`, pays ~2x.
It also explains why the fused `grad(Phi)` experiment didn't help: galpy's *potential*
`_evaluate` itself has 2 `igamma` (two different `a` parameters), so differentiating it
is no cheaper.

**Fix:** have the spherical force layer compute `_rforce(r)` **once** per step and
project to (R, z) instead of `_Rforce`/`_zforce` each recomputing it. Confirmed this
collapses the bulge to **1 igamma** (jaxpr count). Expected to remove roughly half of
the bulge's ~0.089 ms/orbit, and it generalises to every special-function-backed
spherical potential (Einasto, etc.). (A clean in-integrator wall-time demo still needs
the real force layer — a quick hand-rolled EOM gave a bogus number because it dropped
the `_amp` factor.)

NOTE: the radial-force-once fix addresses the bulge only; galpy's disk+halo evaluation
is also somewhat heavier than galax's, so it narrows but does not fully close the
matched-MW gap.

## CORRECTION + final root cause (supersedes the two sections above)

The "galpy evaluates igamma 2x/step and radial-force-once would fix it" claim above
is **wrong** — it was based on the *jaxpr* (pre-optimization) count. Verified three
independent ways:

1. **Micro**: `gammainc(a,x)+gammainc(a,x)` costs exactly the same as a single call
   (10.81 vs 10.84 ms @1e5; 597.7 vs 597.9 @1e7) and compiles to the same optimized
   HLO (117 `while` strings vs 234 for genuinely-different arguments).
2. **Real path**: galpy's `(Rforce, zforce)` pair = `Rforce` alone = manual
   radial-once (2.27/2.28/2.30 ms on (1e5,) arrays; identical optimized HLO).
3. **Source-level** (jax 0.8 AND 0.10, verified in `jax/_src/lax/special.py` +
   openxla `hlo_cse.cc`): jax expands igamma at lowering into inline while-loops
   (no `chlo.igamma`), and XLA's HloCSE merges structurally identical while loops
   with identical inputs. Identical behavior in both versions.

**XLA CSE already dedupes the duplicate igamma. There is nothing to optimize in
galpy's spherical-potential special-function pattern, and subclassing PSCw under
SphericalPotential is performance-neutral** (PSCw already hand-implements the
identical `_rforce(r)*R/r` pattern; the traced graph is the same either way).

### The actual residual factor: batching architecture, not special functions

Further controlled experiments:
- galax controlled bulge-swap (BovyMW's own disk+halo): full 0.080 / disk+halo
  0.036 / Hernquist-bulge 0.038 → galax's cutoff bulge genuinely adds 0.044
  (so the igamma bulge is intrinsically expensive in BOTH codes).
- dtype: galax solution + gradient verified float64 (not an f32-vs-f64 sm_61 artifact).
- jaxlib version: an identical pure-jax MW-like benchmark (no galpy/galax imports;
  fused-grad RHS + diffrax Dopri8 + PID 1e-8, vmap N=1e5) gives **0.283** ms/orbit
  on jax 0.10.1 and **0.280** on jax 0.8.0 (galax's env) — version is irrelevant.
  igamma kernel: 14.2 vs 13.7 ms @1e6 — same.
- step counts @1e-8: galpy MW 235 steps; galax BovyMW **312** steps (galax takes
  MORE steps yet is 3x faster) → the gap is per-RHS-eval throughput, not stepping.

**Root cause: vmap-of-solve vs batch-as-one-system.** galpy's in-backend
`integrate_orbit` vmaps a per-orbit `diffeqsolve` (each orbit its own adaptive
controller; the batched while-loop runs max-over-batch iterations — measured
per-orbit steps across the perturbed batch: median 243, p95 319, max 382+ — with
every iteration processing all lanes). galax integrates the whole batch as ONE ODE
system with a SHARED PIDController (one global error norm -> one dt sequence,
**207 steps total** in the pure-jax replica). Rewriting the same pure-jax benchmark
batch-as-one-system: **0.283 -> 0.097 ms/orbit (2.9x)** — landing at galax's 0.080.

Accuracy trade-off of shared stepping (vs per-orbit, against an rtol=1e-12
reference, N=1e3 batch, 2 Gyr): worst-case |err| 9.2e-6 (shared) vs 3.3e-6
(per-orbit) at nominal 1e-8 — same order, ~3x looser at the batch tail.

### Final decomposition of the original ~4.5x
| factor | size | nature |
|---|---|---|
| tolerance defaults (1e-10 vs 1e-8) | ~1.5x | benchmark artifact — match in g8 |
| vmap-of-solve vs shared-controller batch | ~2.9x | architecture choice; opt-in batch mode would close it |
| backend dispatch / igamma duplication / jaxlib version / dtype | ~1.0x | all ruled out |
| PSCw igamma bulge | intrinsic in both codes | same kernel, same cost (~0.04 ms/orbit shared-mode) |

**Actionable (surgical, additive):** an opt-in shared-controller batched mode in
`galpy.backend._reference.integrate_orbit` (state (N,6) through one diffeqsolve)
— no potential-class or force-layer changes — recovers ~3x on GPU batches at a
modest, tolerance-tunable accuracy cost. Per-orbit mode stays the default.

## Apples-to-apples re-run (matched tolerance + matched stepping) — g8 updated

Re-measured the full g8 grids with everything matched (Dopri8, rtol=atol=1e-8,
same orbit/potential), adding a vmapped per-orbit galax line (single-orbit
`OrbitSolver.solve` under `jax.vmap` — galax supports this fine, it is just not
its native mode):

| N    | galpy per-orbit | galax per-orbit | galax batched (native) |
|------|----------------:|----------------:|-----------------------:|
| 1e4  | 0.262           | 0.223           | --                      |
| 1e5  | 0.244           | 0.232           | ~0.08                   |
| 3e5  | 0.211           | 0.209           | ~0.07                   |
(ms/orbit, TITAN Xp)

**At matched semantics galpy == galax to within ~1-5%.** The entire original
"galax ~5x faster" observation decomposes as (tolerance defaults ~1.5x) x
(batched vs per-orbit stepping ~2.9x) x (framework difference ~1.0x).

Decision (user): galpy keeps per-orbit stepping in its in-backend integrator —
per-orbit error guarantees and batch-independent trajectories (shared-controller
mode only guarantees error in the batch-RMS sense, and the same IC integrated
inside a different batch shifts by ~1e-6 at 1e-8 tolerance). The g8 figure now
shows three GPU lines (galpy per-orbit / galax per-orbit / galax native batched)
with the matched-tolerance note; data in
paper/figures/g8_galax_orbit_speed/data/{galpy-jax_gpu*,galax_gpu_perorbit*}.json,
measure scripts measure/{galpy_jax_measure_tol8,galax_perorbit_measure}.py.
