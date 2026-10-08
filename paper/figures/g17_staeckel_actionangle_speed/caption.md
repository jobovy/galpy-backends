# g17 — Staeckel action-angle throughput: C vs numpy vs jax/torch (CPU & GPU)

**Speed of the Staeckel action-angle map.** Wall time (a) and throughput (b) of
`actionAngleStaeckel.actionsFreqsAngles` (the full $J_R, L_z, J_z, \Omega_{R,\phi,z},
\theta_{R,\phi,z}$ for $N$ phase-space points) against batch size $N$, for
MWPotential2014 with $\delta=0.45$ and Gauss–Legendre order 10, in float64. The
figure is split into a jax version and a torch version; both carry the same C and
numpy reference lines. Colour encodes framework and device (galpy C grey, numpy
black; jax sky blue/blue/navy and torch orange/vermillion/dark red for CPU/TITAN
Xp/H100). Jitted series are solid and eager ones dashed. Eager GPU series that
include the host↔device transfer are dotted, as is the old per-object numpy loop
(black crosses). C on 16 threads is dash-dot. Lines are labelled directly.

CPU series ran on an otherwise idle Xeon E5-2697 v4. Each ran on **one core**: the
C, numpy, torch and old per-object series single-threaded (`OMP_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`, one torch thread), and the jax series in a process
pinned to one core. The extra C line uses 16 OpenMP threads on 16 pinned cores.
GPU series are on an NVIDIA TITAN Xp (12 GB; jax 0.10.1, torch 2.12.0+cu126) and
an NVIDIA H100 80GB (Trillium; jax 0.11.2, torch 2.14.1+cu130), both with galpy
#1642. Every point is the median of repeated warm calls with device
synchronisation; jit compile time is excluded. On the TITAN Xp, points that run out
of GPU memory are omitted: eager jax beyond $N=10^5$ (with transfer, beyond
$3\times10^4$) and torch at $N=10^6$.

## Headline numbers (wall time per call)
| series | $N=1$ | $N=10^3$ | $N=10^5$ | $N=10^6$ |
|---|---|---|---|---|
| galpy C, 1 thread | 1.5 ms | 0.33 s | 30 s | 310 s |
| galpy C, 16 threads | 5.9 ms | 35 ms | 3.0 s | 26 s |
| numpy, vectorised (1 core) | 91 ms | 0.77 s | 92 s | 761 s |
| numpy, old per-object loop (1 core) | 4.0 ms | 4.0 s | — | — |
| torch CPU (1 core) | 0.34 s | 0.80 s | 86 s | — |
| jax CPU eager (1 core) | 3.7 s | 9.0 s | — | — |
| jax CPU jit (1 core) | 5.1 ms | — | — | — |
| torch TITAN Xp | 1.4 s | 1.4 s | 5.0 s | (12.9 s at $3\times10^5$) |
| jax TITAN Xp eager | 8.7 s | 10.3 s | 16 s | — |
| jax TITAN Xp jit | 0.71 s | — | 4.6 s | — |
| torch H100 | 0.47 s | 0.50 s | 0.78 s | 3.5 s |
| jax H100 eager | 4.1 s | 4.3 s | 5.0 s | 12.5 s |
| jax H100 jit | 0.80 s | — | 0.16 s | — |

(— : not measured at that $N$. jax jit is measured at $N=1, 100, 3000$ and
$3\times10^4$ on the CPU and up to $10^5$ on the GPU; jax CPU jit takes 44 s at
$N=3\times10^4$.)

## What it shows
- **On one core, compiled C is fastest at every $N$**: 0.31 ms per point at
  large $N$. The vectorised numpy path costs 2.5–3× more per point at large $N$
  (0.76 ms per point at $N=10^6$), and up to 60× more at $N=1$, where its fixed
  per-call array overhead dominates.
- **The vectorisation win**: the vectorised numpy path is 5–6× faster than the
  old per-object loop for $N\gtrsim10^3$ (2.1 s vs 11.8 s at $N=3000$). The old
  loop is faster only below $N\approx20$.
- **The CPU backends**: eager torch matches numpy at large $N$ (86 vs 92 s at
  $N=10^5$). Eager jax is dispatch-bound (a ≈4 s floor). Single-core jitted jax
  takes 5 ms at $N=1$ but is slower than numpy at larger $N$ (44 vs 24 s at
  $N=3\times10^4$).
- **16 threads**: C on 16 cores is 10–12× faster than on one (26 s at $N=10^6$).
- **GPUs**: after a fixed launch/dispatch floor (torch 1.4 s on the TITAN Xp,
  0.47 s on the H100), the GPU lines rise slowly. On the H100, eager torch
  overtakes single-thread C at $N\approx2000$ and 16-thread C at
  $N\approx2\times10^4$. At $N=10^6$ it takes 3.5 s: 89× faster than single-thread
  C and 7.5× faster than 16-thread C. Jitted jax on the H100 takes 0.16 s at
  $N=10^5$: 190× faster than one C thread and 19× faster than 16. The
  float64-limited TITAN Xp beats single-thread C by 6–8× at large $N$ (torch
  12.9 s vs 102 s at $N=3\times10^5$; jitted jax 4.6 s vs 30 s at $N=10^5$) but
  not 16-thread C (8.6 s and 3.0 s).
- The backends are there to make the action-angle map **differentiable and
  composable** inside jax/torch models; on a modern GPU they are also the fastest
  way to evaluate it for large batches.

GPU device-correctness for every backend-migrated action-angle method (Staeckel,
Spherical, Vertical, Harmonic, Isochrone) was verified on the TITAN Xp as part of
producing this figure: see `logs/trackE_actionangle/gpu_verify_aa.py` (17/17 torch
CUDA + 6/6 jax GPU at machine precision). That pass surfaced and fixed the GL-node
and δ device-anchoring bugs (CI has no GPU, so they were latent).

Reproduce: `OMP_NUM_THREADS=1 python make.py` in the GPU env (PYTHONPATH includes
the galpy worktree): every series runs in its own process and is checkpointed, so an
interrupted run resumes. `measure/torch_gpu_only.py` re-measures the torch GPU
series alone, and `OMP_NUM_THREADS=16 taskset -c 16-31 python
measure/c_omp_threads.py` gives the 16-thread C line (`data/g17_c_16threads.json`).
The H100 series are in `data/h100/`. The plot is byte-stable from the JSONs;
`verify` checks only the deterministic plot.
