# gS5 — particle-spray stream generation speed (numpy/C vs differentiable jax/torch)

Wall time (a) and cost per particle (b) to generate a particle-spray stream with
`fardal15spraydf.sample(N)` — N stripping times drawn, N particles sprayed off a
2×10⁴ M☉ progenitor on an inclined orbit in a flattened (q=0.9) logarithmic halo
and each integrated over up to 4.5 Gyr of stripping to the present. The numpy path
uses galpy's C integrator (OpenMP) and is not differentiable; the jax and torch
paths make the halo flattening a backend scalar, so the whole stream is
differentiable in q (cf. gS2/gS3), and integrate all particles in-backend,
vectorised (jax: diffrax Dopri8 under `vmap`; torch: torchode Dopri5), at
rtol=atol=10⁻¹². The jitted jax sampler (`jax.jit` of `sample`, compile excluded)
costs ~1–1.5 ms per particle on the CPU with no fixed overhead — as fast as, or
faster than, the non-differentiable C path (~2.1 ms/particle) — while eager jax
and eager torch carry ~6 s and ~12 s per-call floors from dispatching the
adaptive-step loop, amortised only for N ≳ 10⁴. On the GPU the same code runs the
particles in parallel [GPU numbers to be added]. Medians of 3 warm calls; CPU
series pinned to 4 cores of a Xeon E5-2697 v4 (a shared, heavily loaded machine,
so absolute CPU times are inflated).
