**Figure G6 — jit compile-once / run-many amortization (jax, GPU, float64).**

A jit-compiled backend trades a one-time XLA compilation for fast steady-state execution. Measured for a vectorized Plummer $F_R$ on the NVIDIA TITAN Xp.

*(a)* Wall-time per call vs call index for a freshly-jitted function at two array sizes. **Call 0 includes compilation** — a $\sim\!120$ ms spike — after which every call runs the cached executable at the steady-state cost ($\sim\!1$ ms at $N=10^6$, $\sim\!0.15$ ms at $N=10^4$). The dotted lines mark the eager (un-jit) per-call time ($\sim\!2.4$ ms, dispatch-bound and nearly $N$-independent): once compiled, jit is several times faster per call, and much more so at small $N$ where eager is dominated by Python/dispatch overhead.

*(b)* Cumulative wall-time vs number of calls at $N=10^6$. The jit curve starts a compile-cost behind the eager curve but has a shallower slope (lower per-call cost), so it overtakes eager after a **break-even of $\approx\!86$ calls**. 

**Take-away:** compilation is a fixed up-front cost amortized over calls; for any workload that evaluates the same shape repeatedly — an orbit integration makes millions of force calls — jit is far past break-even and the compile time is negligible. The break-even count is the relevant figure of merit for short, one-off evaluations.
