**galpy (C) vs galax (JAX, CPU & GPU): orbit-integration throughput.**
MWPotential2014 / BovyMWPotential2014, a Sun-like orbit integrated for 2 Gyr
(1000 output points) over a batch of N initial conditions. galpy's compiled
`dop853_c` (single-thread C loop) has ~zero startup and wins decisively at small
N — a single orbit takes ~10 ms vs ~1.6 s of kernel-launch overhead for galax on
the GPU — but its per-orbit cost is constant. galax (modern `OrbitSolver`, diffrax
Dopri8) amortises the vectorised solve across the batch: on the TITAN Xp GPU down
to ~0.09 ms/orbit at N=3×10⁴ (~70× faster per orbit than galpy-C), crossing galpy
near N≈300 (GPU) / N≈40 (CPU). The GPU OOMs at N=10⁵ (12 GiB; 1000 dense steps ×
N × 6 doubles). NOTE: galpy's own JAX/torch in-backend GPU integrators are not yet
shown — they need MWPotential2014's bulge (PowerSphericalPotentialwCutoff) migrated
for jit (its force still uses scipy.special.hyp1f1); the disk (MN) and halo (NFW)
are already jit-clean.
