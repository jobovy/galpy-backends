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
