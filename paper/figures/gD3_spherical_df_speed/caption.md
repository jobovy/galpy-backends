# gD3 — spherical-DF speed: sampling throughput and construction cost

**Caption.** Cost of sampling and building galpy's spherical distribution
functions on each backend (float64). (a–d) Wall time of `df.sample(n=N)`
(positions and velocities) for four DFs whose tables are already built: numpy
(galpy's historical path: global RNG + scipy spline inverse CDF), and the
backend-native path taken when a jax/torch random key is passed (eager jax,
the caller's `jax.jit` with compile excluded, eager torch; GPU series where
available). The native bilinear inverse CDF makes every backend faster than
numpy at large $N$; jitted jax is the fastest (or tied) CPU path at every $N$
(3–17× numpy at $N=10^7$), while eager jax pays a ~30 ms dispatch floor below
$N\sim10^5$. On the GPU the jitted sampler draws $10^7$ particles in 39–68 ms on an
NVIDIA TITAN Xp and 3.9–7.2 ms on an NVIDIA H100 (~1000× numpy).
(e) Constructing the DF from a potential parameter and drawing its first
1000 particles (which builds the sampling tables), first call in a fresh
process (outline) vs warm (filled). For the Eddington-type inversions the
vectorised backend construction is 10–1200× faster than numpy when warm —
constantbetadf drops from 2–2.6 min to 0.13 s under `jax.jit` — whereas for
kingdf numpy's scipy ODE solve (0.1 s) is already as fast as any backend. Eager
jax pays 20–40 s of one-time per-op compilation and `jax.jit` 6–23 s of compile
(2–6 s on the H100 node). Warm, a jitted Eddington/Osipkov–Merritt/constant-β
construction takes 2–3.5 ms on the TITAN Xp and 1.1 ms on the H100. Colour =
backend × device (numpy black; jax sky blue / blue / navy and torch orange /
vermillion / dark red for CPU / TITAN Xp / H100); in (a–d) jit solid, eager dashed;
in (e) eager bars hatched.

## CPU numbers (Xeon E5-2697 v4, 4 pinned cores of a loaded shared box; torch 1 thread)
Sampling, wall time of `sample(n=N)`:

| DF | N | numpy | jax eager | jax jit | torch |
|---|---|---|---|---|---|
| isotropicHernquistdf | 1e3 | 2.0 ms | 30 ms | 1.8 ms | 4.7 ms |
| | 1e5 | 0.17 s | 0.09 s | 0.08 s | 0.09 s |
| | 1e7 | 17.8 s | 12.9 s | 6.2 s | 14.5 s |
| eddingtondf (NFW) | 1e7 | 54 s | 16.1 s | 6.5 s | 19.9 s |
| osipkovmerrittdf | 1e7 | 108 s | 16.3 s | 6.4 s | 19.7 s |
| kingdf | 1e7 | 24.6 s | 15.6 s | 5.8 s | 17.9 s |

jit compile of `sample`: 1.5–2.3 s for N ≤ 1e6, 8–10 s at N = 1e7 (per shape).

Construction + first 1000 draws (first call / warm):

| DF | numpy | jax eager | jax jit | torch |
|---|---|---|---|---|
| eddingtondf (NFW) | 5.3 / 5.3 s | 40 / 0.72 s | 7.7 / 0.13 s | 0.49 / 0.31 s |
| osipkovmerrittdf | 2.9 / 2.9 s | 33 / 0.54 s | 7.4 / 0.076 s | 0.34 / 0.22 s |
| constantbetadf | 121 / 157 s | 29 / 0.79 s | 5.8 / 0.13 s | 3.7 / 0.26 s |
| kingdf (W0) | 2.2 / 0.098 s | 18 / 0.47 s | 23 / 0.13 s | 3.3 / 0.22 s |

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
`CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu taskset -c <4 cores> python -m paper.figures._harness run gD3_spherical_df_speed` (~45 min).
GPU (merges into the JSON): `python paper/figures/gD3_spherical_df_speed/make.py --gpu`,
then re-run the harness. Single modes can be re-measured with
`make.py --modes torch` (comma-separated).
