# gS4 — streamdf (action-angle stream model) cost per backend

Cost of galpy's action-angle stream model `streamdf` (a progenitor on an inclined
orbit in a q=0.9 logarithmic halo, 4.5 Gyr of disruption, actionAngleIsochroneApprox)
on each backend. (a) Construction including the stream track vs the number of
track chunks: the numpy/C build grows linearly (4–27 s for 4–32 chunks), whereas
the differentiable jax build (diffrax orbits, exact AD Jacobians, chunk loop under
`vmap`) is flat in the chunk count but costs ~90–110 s warm (~200–230 s on first
call, dominated by eager-op compilation); torch construction and a jitted
construction are not yet possible (see the bug notes). (b) Drawing N stream stars
and (c) evaluating the log DF at N phase-space points, from a model built once:
under `jax.jit` both are ~100× faster than the numpy path (N=10⁵: 0.51 s vs 77 s
to sample, 0.13 s vs 17 s to evaluate), because the backend linearised track maps
are vectorised where the numpy ones loop over points; eager jax and eager torch
(on the numpy-built model) sit in between. GPU timings use the same code on an
NVIDIA TITAN Xp [to be added]. Medians of 3 warm calls (construction: single
shots); CPU series on 4 cores of a shared, heavily loaded Xeon E5-2697 v4, so
absolute CPU times are inflated.
