**Figure G5 — The cost of differentiability: gradient vs value throughput (GPU, float64).**

For an autodiff library the key performance question is how much more expensive the *gradient* is than the *value*. Reverse-mode AD computes the gradient of a scalar with respect to all $N$ inputs in a single backward pass whose cost is a small constant multiple of the forward pass, *independent of $N$* — so differentiability is cheap. Measured here for a vectorized Plummer evaluation on the NVIDIA TITAN Xp: the **value** is $\Phi(R,z)$ over $N$ points; the **gradient** is $\partial/\partial R\sum_i\Phi(R_i,z_i)$, i.e. the per-point $\partial\Phi/\partial R$ from one reverse-mode pass.

*(a)* Throughput (points/s) vs $N$, value (solid) vs gradient (dashed), for jax and torch. *(b)* The overhead ratio (gradient wall-time / value wall-time) vs $N$.

**Take-aways:**
- Under **jax the gradient is nearly free** — overhead $\approx\!1.1\times$ across most of the range (the dashed jax curve sits almost on top of the solid one). XLA fuses the forward and backward computation into a single kernel, and for a pointwise map $\partial\Phi/\partial R$ is itself a pointwise expression of comparable cost.
- Under **torch (eager) the overhead is $\sim\!3$–$4\times$** — a separate backward pass with its own kernel launches and intermediate allocations — still a modest, roughly $N$-independent constant.
- In both cases the ratio does not grow with $N$: the price of differentiability is a small constant factor, not an asymptotic penalty. This is what makes embedding galpy inside gradient-based inference/ML tractable at scale.
