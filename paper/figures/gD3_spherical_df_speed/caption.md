# gD3 — spherical-DF speed: sampling throughput and construction cost

**Caption.** Cost of sampling and building galpy's spherical distribution
functions on each backend (float64). (a–d) Wall time of `df.sample(n=N)`
(positions and velocities) for four DFs whose tables are already built: numpy
(galpy's historical path: global RNG + scipy spline inverse CDF), and the
backend-native path taken when a jax/torch random key is passed (eager jax,
the caller's `jax.jit` with compile excluded, eager torch; GPU series where
available). Every CPU series ran on ONE core of a Xeon E5-2697 v4: numpy
(scipy) is single-threaded, torch ran one intra-op thread, and jax was pinned to a
single core with `taskset` (its XLA:CPU thread pool spans the CPU affinity;
re-measured 2026-10-08), so the CPU lines compare like with like. The native
bilinear inverse CDF makes every backend competitive with numpy at large $N$;
jitted jax is the fastest CPU path at every $N$ (1.4–6× numpy at $N=10^7$, ~2–7×
at $N=10^3$), while eager jax pays a ~15–20 ms dispatch floor below $N\sim10^4$
and eager torch is within ~1.7× of numpy at $10^7$ (faster for the tabulated
Eddington and Osipkov–Merritt DFs). On the GPU the jitted sampler draws $10^7$
particles in 39–68 ms on an NVIDIA TITAN Xp (~200–800× one CPU core running numpy)
and 3.9–7.2 ms on an NVIDIA H100 (~2000–7000×).
(e) Constructing the DF from a potential parameter and drawing its first
1000 particles (which builds the sampling tables), first call in a fresh
process (outline) vs warm (filled). For the Eddington-type inversions the
vectorised backend construction is 16–1500× faster than numpy when warm under
`jax.jit` — constantbetadf drops from 1.4–1.8 min to 0.07 s — whereas for
kingdf numpy's scipy ODE solve (0.05 s warm) is already as fast as any backend.
Eager jax pays 9–19 s of one-time per-op compilation and `jax.jit` 4.5–14 s of
compile (2–6 s on the H100 node). Warm, a jitted Eddington/Osipkov–Merritt/constant-β
construction takes 2–3.5 ms on the TITAN Xp and 1.1 ms on the H100. Colour =
backend × device (numpy black; jax sky blue / blue / navy and torch orange /
vermillion / dark red for CPU / TITAN Xp / H100); in (a–d) jit solid, eager dashed;
in (e) eager bars hatched.

## CPU numbers (Xeon E5-2697 v4, one core per series; shared box)
numpy re-measured 2026-10-07 (single-threaded, 4 cores pinned), jax eager/jit
2026-10-08 pinned to one core (`taskset -c 16`), torch (one intra-op thread)
2026-10-06 on a more heavily loaded box. Sampling, wall time of `sample(n=N)`:

| DF | N | numpy | jax eager | jax jit | torch |
|---|---|---|---|---|---|
| isotropicHernquistdf | 1e3 | 1.2 ms | 15 ms | 0.55 ms | 4.7 ms |
| | 1e5 | 0.071 s | 0.070 s | 0.038 s | 0.086 s |
| | 1e7 | 8.5 s | 12.0 s | 5.9 s | 14.4 s |
| eddingtondf (NFW) | 1e7 | 22.4 s | 15.7 s | 7.2 s | 19.9 s |
| osipkovmerrittdf | 1e7 | 47.4 s | 17.6 s | 7.8 s | 19.7 s |
| kingdf | 1e7 | 12.6 s | 16.6 s | 8.0 s | 17.9 s |

jit compile of `sample` (one core): 1.3–1.6 s for N ≤ 3e5, 7–9.5 s at N = 1e7 (per shape).

Construction + first 1000 draws (first call / warm):

| DF | numpy | jax eager | jax jit | torch |
|---|---|---|---|---|
| eddingtondf (NFW) | 2.9 / 2.7 s | 19 / 0.42 s | 7.7 / 0.099 s | 0.49 / 0.31 s |
| osipkovmerrittdf | 1.2 / 1.3 s | 17 / 0.38 s | 6.4 / 0.080 s | 0.34 / 0.22 s |
| constantbetadf | 83 / 108 s | 15 / 0.52 s | 4.5 / 0.073 s | 3.7 / 0.26 s |
| kingdf (W0) | 1.1 / 0.047 s | 9.2 / 0.28 s | 14 / 0.080 s | 3.3 / 0.22 s |

## GPU numbers (NVIDIA TITAN Xp, jax 0.10.1 / torch 2.12.0+cu126; NVIDIA H100 80GB on Trillium, jax 0.11.2 / torch 2.14.1+cu130)

| | TITAN Xp | H100 |
|---|---|---|
| `sample(n=1e7)`, jax jit | 39–68 ms | 3.9–7.2 ms |
| `sample(n=1e7)`, torch eager | 1.0–1.2 s | 0.50–0.57 s |
| `sample`, jax eager floor (N ≤ 1e6) | ~40–70 ms | ~17–25 ms |
| construct warm, jax jit (Edd./OM/β) | 2.3–3.5 ms | 1.1 ms |
| construct warm, jax jit (kingdf) | 0.15 s | 0.11 s |
| construct warm, torch eager | 0.07–0.13 s | 0.025–0.042 s |
| construct first call, jax jit | 5.7–8.4 s (king 21 s) | 2.0–2.8 s (king 6.1 s) |

The H100 series are drawn next to the TITAN Xp ones (navy / dark red), from
`data/h100/` (same script).

Reproduce (CPU): delete `data/gD3_spherical_df_speed.json`, then
`CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu taskset -c <one core> python -m paper.figures._harness run gD3_spherical_df_speed`
(pin to ONE core: jax's XLA:CPU pool otherwise uses every core in the affinity).
GPU (merges into the JSON): `python paper/figures/gD3_spherical_df_speed/make.py --gpu`,
then re-run the harness. Single modes can be re-measured with
`make.py --modes torch` (comma-separated).
