# g17 — Staeckel action-angle throughput: C vs numpy vs jax/torch (CPU & GPU)

Wall-clock throughput (points/s) of `actionAngleStaeckel.actionsFreqsAngles`
(the full jr,Lz,jz,Ω_r,Ω_φ,Ω_z,θ_r,θ_φ,θ_z for N phase-space points) vs batch
size N, for the compiled C path and the now-vectorised, differentiable c=False
path under numpy/jax/torch on CPU and an NVIDIA TITAN Xp. MWPotential2014,
δ=0.45, GL order 10; median of repeated calls with backend warmup + device sync.

## Headline numbers (this machine; timings are machine-dependent)
| backend | N=1 | N=10³ | N=10⁵ |
|---|---|---|---|
| C (c=True, 1 thread) | 1.2 ms | 0.29 s | 25 s |
| numpy (vectorised) | 53 ms | 0.31 s | 26 s |
| numpy (per-object, OLD) | 5.8 ms | 4.2 s | — |
| torch CPU | 0.24 s | 0.57 s | (≈numpy @1e4) |
| jax CPU (eager) | 3.2 s | 6.8 s | — |
| **torch GPU** | 0.94 s | 0.97 s | **1.4 s** |
| jax GPU (eager) | 7.7 s | 7.0 s | 12 s |

## What it shows
- **Vectorised numpy ≈ single-threaded C** at every N (26 vs 25 s at 1e5) — going
  fully array-wise closed the per-object-dispatch gap with compiled C.
- **The vectorisation win**: the new numpy path is **13–17× faster than the old
  per-object loop** for batches (N≳100); the old loop only wins at N=1 (no
  fixed array overhead). Crossover ≈ N=10.
- **torch GPU is the batch winner**: a ~1 s launch/dispatch floor that amortises to
  **~18× faster than C/numpy by N=1e5**. The GPU only pays off once N is large
  enough to hide the fixed launch cost (N≳3000 here).
- **jax eager is dispatch-bound** (~7 s flat, CPU and GPU alike): galpy adds no
  internal jit — it is made jit-*compatible*, so the user `jit`s their galpy-using
  code to collapse the hundreds of per-op dispatches. Eager jax is the slowest.
- The backends are not here to beat C on raw throughput; they make the
  action-angle map **differentiable and composable** inside jax/torch models.
  This figure quantifies the cost of that.

GPU device-correctness for every backend-migrated action-angle method (Staeckel,
Spherical, Vertical, Harmonic, Isochrone) was verified on the TITAN Xp as part of
producing this figure: see `logs/trackE_actionangle/gpu_verify_aa.py` (17/17 torch
CUDA + 6/6 jax GPU at machine precision). That pass surfaced + fixed the GL-node
and δ device-anchoring bugs (CI has no GPU, so they were latent).

Reproduce: delete `data/g17_staeckel_actionangle_speed.json` and re-run `make` in
the GPU env (PYTHONPATH includes the galpy worktree). The plot is byte-stable from
the JSON; `verify` checks only the deterministic plot.
