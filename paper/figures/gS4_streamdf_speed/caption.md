# gS4 — streamdf (action-angle stream model) cost per backend

Cost of galpy's action-angle stream model `streamdf` (a progenitor on an inclined
orbit in a q=0.9 logarithmic halo, 4.5 Gyr of disruption, actionAngleIsochroneApprox)
on each backend. (a) Construction including the stream track vs the number of
track chunks: the numpy/C build grows linearly (4–27 s for 4–32 chunks), whereas
the differentiable jax build (diffrax orbits, exact AD Jacobians, chunk loop under
`vmap`) is flat in the chunk count but costs ~90–110 s warm (~200–230 s on first
call, dominated by eager-op compilation); jitting the whole construction costs
15–24 min of compile and then 53–195 s per build; eager torch construction takes
~22 min and >100 GB of host memory and is not plotted. (b) Drawing N stream stars
and (c) evaluating the log DF at N phase-space points, from a model built once:
under `jax.jit` both are ~100× faster than the numpy path on the CPU (N=10⁵: 0.51 s
vs 77 s to sample, 0.13 s vs 17 s to evaluate), because the backend linearised
track maps are vectorised where the numpy ones loop over points; eager jax and
eager torch (on the numpy-built model) sit in between. On the GPU the same code is
faster again: jitted DF evaluation at N=10⁶ takes 67 ms on an NVIDIA TITAN Xp and
1.7 ms on an NVIDIA H100 (vs 149 s for numpy), and the eager jax construction
drops to ~53 s warm on the H100 (~130 s on the TITAN, a consumer card with float64
at 1/32 rate). Colour = backend × device (numpy black; jax sky blue / blue / navy
and torch orange / vermillion / dark red for CPU / TITAN Xp / H100); jit solid,
eager dashed, first (cold) calls dotted / dash-dotted. Medians of 3 warm calls (construction: single
shots); CPU series pinned to 4 cores of a shared, heavily loaded Xeon E5-2697 v4,
so absolute CPU times are inflated: numpy (whose C orbit calls integrate one orbit
at a time, i.e. one OpenMP thread) and torch (one intra-op thread) on one core
each, jax with XLA:CPU's thread pool over up to all 4 cores (CPU labels state the
cores instead of the device); H100 series on Trillium (jax 0.11.2, torch
2.14.1+cu130).
