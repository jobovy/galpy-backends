**Figure G2 — Force-evaluation throughput, CPU vs GPU: compute-only vs end-to-end.**

Throughput (points evaluated per second, $= N/\text{wall-time}$) versus array length $N$ (number of points per call) for a vectorized Plummer $F_R$ evaluation, on numpy (CPU) and jax/torch (NVIDIA TITAN Xp GPU). Each marker is the median over repeated calls. Two regimes are shown for each GPU backend:

- **Solid (compute-only):** the input array is resident on the GPU before timing and the result stays on-device — the timed wait (`block_until_ready` / `cuda.synchronize`) is for the kernel, not a copy-back. This is the best-case GPU throughput, with host$\leftrightarrow$GPU transfers *excluded*.
- **Dashed (end-to-end, $+$transfer):** each call starts from a host (numpy) array and returns a host array, so it includes the host$\rightarrow$device and device$\rightarrow$host copies — the realistic cost of calling galpy on the GPU from a numpy workflow.

Dotted slope-1 lines mark constant wall-time per call (0.1, 1, 10, 100 ms); on a log–log throughput-vs-$N$ plot a fixed wall-time $t_0$ is the line $N/t_0$.

**Take-aways:**
- At small $N$ the GPU backends are **launch/overhead-bound** — wall-time per call is roughly constant, so the curves hug the $\sim$0.1–0.5 ms reference lines and throughput rises only because $N$ does. The CPU, with negligible per-call overhead, wins here.
- **Compute-only**, the GPU overtakes the CPU around $N\sim10^{4}$ and is $\sim\!100\times$ faster at $N=10^{7}$ ($\sim\!10^{9}$ vs $\sim\!10^{7}$ pts/s).
- **End-to-end**, the round-trip copy is PCIe-bandwidth-bound (cost $\propto N$ while the kernel cost is sub-linear), so it saturates GPU throughput at $\sim\!10^{8}$ pts/s — **a full order of magnitude below compute-only** — and the jax curve even turns over at $N=10^{7}$. The honest GPU win for one-shot host calls is a modest $\sim\!4$–$6\times$, and the CPU crossover moves to larger $N$.
- **Implication:** the GPU pays off when data *stays* on-device across many chained operations (a full differentiable pipeline — forces, orbit integration, action-angle, DF — without round-trips), not for isolated calls from a host workflow. That on-device pipeline is exactly what the multi-backend layer is built to enable.
