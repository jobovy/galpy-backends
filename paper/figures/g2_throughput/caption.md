**Figure G2 — Force-evaluation throughput, CPU vs GPU.**

Throughput (points evaluated per second, $= N/\text{wall-time}$) versus array length $N$ (number of points per call) for a vectorized Plummer $F_R$ evaluation, on numpy (CPU) and jax/torch (NVIDIA TITAN Xp GPU). Each marker is the median over repeated calls. The timed wall-time is **compute-only**: the input array is resident on the GPU before timing and the result stays on-device (`block_until_ready` / `cuda.synchronize` wait for the kernel, not a copy-back), so host$\leftrightarrow$GPU transfers are *not* included. Dotted slope-1 lines mark constant wall-time per call (0.1, 1, 10, 100 ms); on a log–log throughput-vs-$N$ plot a fixed wall-time $t_0$ is the line $N/t_0$.

**Take-aways:**
- At small $N$ the GPU backends are **launch/overhead-bound** — the wall-time per call is roughly constant, so the curves hug the $\sim$0.1–0.3 ms reference lines and throughput rises only because $N$ does. The CPU, with negligible per-call overhead, is faster here.
- The GPU overtakes the CPU around $N\sim 10^{4}$ and is $\sim\!100\times$ faster at $N=10^{7}$ ($\approx$0.65 s on CPU vs $\approx$6 ms on GPU).
- Because transfers are excluded, this is the **best-case compute throughput**; an end-to-end, transfer-inclusive measurement would shift the crossover to larger $N$ (host$\leftrightarrow$GPU copies are bandwidth-bound and add a per-call cost that scales with $N$).
