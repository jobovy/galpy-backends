# DIFFERENTIABLE_DYNAMICS_USES.md -- published/preprint science uses of differentiable (jax/torch) galactic dynamics

Date compiled: 2026-10-07. Web-only research (no code run against galpy).

## How this was assembled and verified

Sources actually consulted (what each one yielded):

- galax README (`github.com/GalacticDynamics/galax`): only a "Citation" section pointing at Zenodo (concept DOI 10.5281/zenodo.11553324); no "used by" list. No galax paper exists (arXiv `all:galax` returns only unrelated "GALAX"/"PyGALAX" ML papers and Chen et al. 2025, consistent with BIB_NOTES.md).
- GitHub dependents page of galax: `nstarman/potamides` (and fork `xggs-dev/potamides`), `GalacticDynamics/phasecurvefit`, `GalacticDynamics/galactic_dynamics_interoperability`, `CCA-Software-Group/short_tutorials`, `alphavector/all`, `Lyn4ever29/pipy_server`. Only potamides/phasecurvefit are science codes.
- Semantic Scholar Graph API: `paper/search` for galax (4 hits, only Chen2025 and Odisseo relevant) and `paper/arXiv:<id>/citations` for StreamSculptor (2410.21174, 16 citing), Deep Potential (2011.04673 / 2205.02244, 19 each), albatross (2304.02032, 19), Hermans+21 (2011.14923, 33), An+21 (2106.05981, 14), Buckley+23 (2205.01129, 15). Heavy 429 rate-limiting; retried with back-off.
- arXiv API listing searches: `all:galax`, `all:StreamSculptor`, `abs:differentiable AND abs:stream`, `abs:JAX AND abs:stream`, `abs:differentiable AND abs:orbit`, `abs:JAX AND abs:gravitational AND abs:potential`, `all:diffrax`, `abs:GPU AND abs:orbits AND abs:potential`, `abs:"Hamiltonian Monte Carlo" AND abs:stream` (0 hits), all restricted to astro-ph.GA.
- arXiv HTML full texts (`arxiv.org/html/<id>`) of every candidate, grepped for `galax`, `gala`, `JAX`, `streamsculptor`, `diffrax`, `GPU`, `numpyro|blackjax`, `automatic differentiation`. This is what decides the "which code / what the differentiability bought" columns below; the contexts are quoted.
- Bibliographic verification: arXiv API `id_list=` for every arXiv id (title, authors, DOI/journal-ref where present) and Crossref `works/<doi>` for every DOI (title, authors, journal, volume, pages, dates). Codes used below: **X** = arXiv API record, **C** = Crossref DOI record, **H** = arXiv HTML full text (for the usage claim).

Papers that use galax directly (full-text confirmed): Wu2026/Wu2026b (Potamides), Starkman2026b, Euclid2026, AnauMontel2026, Viterbo2026 (fiducial MWPotential2014 parameters taken from galax), Chen2025 (particle-spray algorithm implemented in galax). All other stream papers from the Nibauer/Price-Whelan/Starkman group use StreamSculptor (JAX + diffrax; built independently of galax) or gala.

## The list: concrete science enabled by gradients / compilation / GPUs

Format: **key** -- citation [verification]. *What they did.* *What differentiability/compilation/GPU bought.*

### A. galax-based

1. **Wu2026 + Wu2026b** -- Wu, Starkman, Nibauer & Pearson 2026, JOSS 11(126), 10712, doi:10.21105/joss.10712 [C]; Wu, Starkman, Pearson, Nibauer, Miro-Carretero & Martinez-Delgado 2026, arXiv:2605.07809 [X,H].
   *Potamides: constrain projected halo axis ratio and orientation of 15 Stellar Stream Legacy Survey hosts from the curvature of the stream track alone (no kinematics), using a triaxial logarithmic potential implemented via `galax`.*
   *JAX/galax give analytic, auto-differentiated accelerations and a JIT-compiled curvature-vs-acceleration likelihood; mock streams generated with galax in flattened potentials validated the method (posteriors bounded away from q=1).*

2. **Starkman2026b** -- Starkman, Nibauer, Pearson, Wu & Necib 2026, arXiv:2609.40057 [X,H].
   *A "convexity" diagnostic: any stream segment whose curvature points away from the host centre falsifies spherical (and broad classes of) potentials; the convex-segment rate across a population constrains halo triaxiality and so CDM vs SIDM (forecast up to 5 sigma). Hosts are `galax.TriaxialNFWPotential`; tracks built with potamides + phasecurvefit.*
   *A fully vectorised JAX pipeline makes the population-level forecast (many hosts x many viewing directions x many mock streams) tractable; the paper notes GPU acceleration is what lets particle-spray searches over halo parameters run for external galaxies.*

3. **Euclid2026** -- Euclid Collaboration: Starkman, Nibauer, Pearson, Wu, Walmsley, Necib, Bovy, et al. 2026, arXiv:2606.21774 [X,H].
   *Euclid Q1 extragalactic streams: segmentation maps -> smooth curvature-preserving tracks -> shape and centre-of-mass constraints for 13 host halos (q = 0.95 +0.05/-0.10 stacked).*
   *"The pipeline is implemented in JAX, using the custom libraries galax, coordinax, and unxt, for high performance and auto-differentiation"; ~15 min end-to-end per galaxy on a laptop, <~1 min for the dynamical steps on a GPU, and "on a cluster with 100 GPU cores, 1000 stream segments could be analysed in under 10 minutes".*

4. **AnauMontel2026** -- Anau Montel & Schmidt 2026, arXiv:2606.09629 [X,H].
   *A differentiable forward model for streams heated by many weak subhalo encounters (diffusion regime), with the substructure entering only through its power spectrum; forecasts for a GD-1-like stream show kinematics tighten the free-streaming-cutoff constraint from ~1.2 dex to ~0.25 dex.*
   *Orbits of all star particles are integrated with `galax` (via diffrax), parallelised with `vmap` on a GPU (A100 timings given), and JAX autodiff supplies exact derivatives for Fisher forecasts and optimal data compression.*

5. **Viterbo2025 + Viterbo2026** -- Viterbo & Buck 2025, arXiv:2511.22468 (Odisseo) [X,H]; Viterbo & Buck 2026, A&A 707, A363, doi:10.1051/0004-6361/202558358 [C,X,H].
   *Odisseo: a differentiable N-body code in JAX; demonstration fits four parameters (accretion time, progenitor mass, NFW and Miyamoto-Nagai masses) of a mock GD-1 stream by gradient descent on a maximum-mean-discrepancy loss, then (2026) joint flow-matching SBI of the GD-1 progenitor and the Milky Way potential, with the fiducial MWPotential2014 parameters taken from galax.*
   *Autodiff through the full N-body integration (diffrax Tsit5) gives parameter gradients and Fisher contours; near-linear multi-GPU scaling and JIT let the mock-stream training set be generated in ~7.5 GPU-hours on an H200.*

6. **Chen2025** (already in refs.bib) -- Chen, Valluri, Gnedin & Ash 2025, ApJS 276, 32 [C,H].
   *Improved particle-spray algorithm, "implemented in galactic dynamics codes agama, gala, galax, and galpy".* (Not a differentiability use case; listed as the only other paper that touches galax directly.)

### B. StreamSculptor (JAX + diffrax; Nibauer et al.)

7. **Nibauer2025** (already in refs.bib) -- Nibauer, Bonaca, Spergel, Price-Whelan, Greene, Starkman & Johnston 2025, ApJ 983, 68 [C,X,H].
   *Hamiltonian perturbation theory for subhalo impacts on streams in arbitrary, time-dependent potentials (bar, LMC), operating in observable coordinates; forecasts GD-1's velocity dispersion vs subhalo statistics down to ~1e5 Msun.*
   *The perturbative response is obtained by forward-mode automatic differentiation of Hamilton's equations (jvp through the diffrax integrator), so arbitrary subhalo potentials can be treated without action-angle transforms; GPU batching over realisations.*

8. **Nibauer2025b** (already in refs.bib) -- Nibauer & Bonaca 2025, ApJL 985, L22 [C,X,H].
   *Measures the Galactic acceleration field along GD-1 by directly differentiating a spline fit to the 6D stream track (a_R, a_z, a_phi at R=11.9, z=7.3 kpc; M(<14 kpc) = 1.4e11 Msun; tilted halo).*
   *Spline-knot posteriors are sampled with the NUTS sampler from BlackJAX in JAX, i.e. gradient-based HMC over the track model; accelerations are analytic derivatives of the differentiable track.*

9. **Nibauer2026** (already in refs.bib) -- Nibauer, Bonaca, Price-Whelan, Spergel & Greene 2026, ApJ 1004, 62 [C,X,H].
   *160-member radial-velocity dispersion of GD-1 (2-5 km/s, a 4 sigma hot region) converted into constraints on the number density and concentration of 1e5-1e9 Msun subhalos.*
   *Hundreds of subhalo impacts per realisation are modelled perturbatively in StreamSculptor; the dispersion/binarity model is fit with NumPyro NUTS.*

10. **Nibauer2026b** -- Nibauer & Pearson 2026, ApJ 1002, 204, doi:10.3847/1538-4357/ae5bb0 [C,X,H].
    *X-Stream: nested sampling over thousands of particle-spray stream realisations to infer the full radial dark-matter density profile (inner and outer slopes) of external galaxies from stream imaging alone.*
    *"The GPU support of our simulator allows us to efficiently generate stream models in batches of 500", each in a different trial potential -- the brute-force imaging-only inference is only feasible with GPU batching.*

11. **Holm2026** -- Holm, Pearson, Nibauer, Sand, Price-Whelan, Starkenburg, Hendel & Fielder 2026, Nature 656, 843-847, doi:10.1038/s41586-026-10878-w [C,X,H].
    *First evidence for a globular-cluster stream outside the Milky Way (UGC 9050-Dw1, HST imaging), used to constrain the dark-matter content of an ultra-diffuse galaxy.*
    *Host-halo constraints come from the X-Stream sampler running particle-spray models "implemented in the GPU code streamsculptor".*

12. **Zaritsky2026** -- Zaritsky, Nibauer, Golini, et al. 2026, ApJL 1000, L6, doi:10.3847/2041-8213/ae45a4 [C,X,H].
    *A 28.7 mag/arcsec^2 stream wrapping >180 deg around NGC 3938 (LIGHTS survey); its apparently off-centred orbit is reproduced with standard host assumptions.*
    *Simulations "using the streamsculptor python package" -- the fast JAX simulator makes the morphology exploration cheap. (Peripheral: the paper does not use gradients explicitly.)*

### C. Other JAX stream / orbit codes

13. **Chemaly2026** -- Chemaly, Sola, Belokurov, Koposov, Meyong, Zhang & Erkal 2026, MNRAS 550(2), stag1161, doi:10.1093/mnras/stag1161 [C,X,H].
    *StreaMAX, a JAX particle-spray package, plus a hierarchical Bayesian framework that infers the population distribution of halo flattening from projected (photometric-only) extragalactic stream tracks; oblate/spherical/prolate populations are cleanly separated on mocks.*
    *JIT + GPU (NVIDIA L4) give "orders of magnitude faster stream generation" than gala/Agama with a nearly flat GPU scaling curve, which is what makes fitting hundreds of thousands of streams for the hierarchical step affordable.*

14. **SpurioMancini2026** -- Spurio Mancini 2026, arXiv:2609.28458 [X,H].
    *GRADSOLVE: one adaptive ODE integration per GPU thread with parameter derivatives in the same pass; the first of three worked examples is stellar orbits in the Galactic potential and their sensitivity to its parameters (stream modelling).*
    *A million orbits-with-gradients run 8.5-1500x faster than a 128-core CPU run of gala/scipy (gala differentiating by finite differences) and 11-15x faster than diffrax on the same GPU, "so gradient-based analyses that need ensembles of this size become routine".*

15. **Tavangar2025** -- Tavangar & Price-Whelan 2025, ApJ 988, 45, doi:10.3847/1538-4357/addd1c [C,X,H].
    *Flexible density/membership model of GD-1 in Gaia DR3 (1689 members; density and kinematic tracks, velocity dispersion, initial and present mass).*
    *The `stream-membership` package is built on JAX + NumPyro: autodiff, JIT and vmap over the many spline-knot parameters enable NUTS sampling of the full density model.*

16. **PriceWhelan2025 + Oeur2025** -- Price-Whelan, Hunt, Horta, Oeur, Hogg, Johnston & Widrow 2025, ApJ 979, 115, doi:10.3847/1538-4357/ad969a [C,X,H]; Oeur, Loebman, Price-Whelan, Arora, Necib & Horta 2025, ApJ 993, 29, doi:10.3847/1538-4357/ae0584 [C,X,H].
    *Orbital Torus Imaging: a flexible foliation of the vertical phase space by orbits, fit to element-abundance gradients to get the vertical acceleration profile; then tested on the FIRE-2 m12i simulation (26-parameter model).*
    *"Implemented ... using JAX to accelerate the model evaluation and to use automatic differentiation to compute the gradients of the model with respect to the (many) model parameters" (JAXopt optimisation; BlackJAX sampling in Oeur+25).*

### D. Learning the potential from a phase-space snapshot (normalizing flows + differentiable potential networks)

17. **Kalda2025 + Kalda2026** -- Kalda & Green 2025, ApJ 992, 84, doi:10.3847/1538-4357/adf8ea [C,X,H]; Kalda & Green 2026, ApJL 1002, L47, doi:10.3847/2041-8213/ae626c [C,X,H].
    *Deep Potential on 5.6 million Gaia DR3 upper-main-sequence stars within 1 kpc: full 3D potential plus local pattern speed Omega_p = 28.2 +/- 0.1 km/s/kpc; then a "conditional" variant modelling p(v|x) that needs no spatial selection function.*
    *The potential is a neural network trained to satisfy the collisionless Boltzmann equation on a normalizing-flow DF (gradients of Phi and f by autodiff). Kalda2025 is TensorFlow 2; Kalda2026 moved to JAX (flowjax, diffrax, equinox, optax) and reports a >100x training speed-up that made optuna hyperparameter searches possible.*

18. **Lim2025** (already in refs.bib) + **Putney2024 + Putney2025** -- Lim, Putney, Buckley & Shih 2025, JCAP 01, 021 [C,X,H]; Putney, Shih, Lim & Buckley 2024, arXiv:2412.14236 [X,H]; Putney, Shih, Lim & Buckley 2025, arXiv:2512.09989 [X,H].
    *Model-free, unbinned local acceleration and mass-density fields from 6 million Gaia DR3 stars (3 kpc sphere); then a learned "dust efficiency" selection function; then ClearPotential, a fully 3D neural-network potential within 4 kpc including the dust-obscured disc.*
    *PyTorch + nflows masked autoregressive flows give a differentiable DF, and the potential is a neural network whose gradients (acceleration, Laplacian -> density) come from autodiff; "our fully-differentiable model of the gravitational potential allows us to map the acceleration and mass density". (No GPU/JAX claims found in the HTML; this is the torch branch of the field.)*

### E. Simulation-based inference with fast simulators (precursors / adjacent)

19. **Hermans2021, Alvey2023** (already in refs.bib) -- neural ratio estimation for WDM from stream density (Hermans+), and albatross, the scalable TMNRE pipeline for full stream analyses (Alvey+). Neither uses autodiff through the simulator; they motivate the "fast differentiable simulator" direction and are cited as such by StreamSculptor, Odisseo and AnauMontel2026.

20. **Widmark2025** -- Widmark & Johnston 2025, ApJ 987, 120, doi:10.3847/1538-4357/add887 [C,X,H].
    *Implicit-likelihood inference (forward simulation + data compression + neural density estimation) of the orbit and total mass density of the tidally perturbed satellite NGC 205 from its projected internal velocity field.*
    *An SBI use case, but the HTML full text has no JAX/GPU/autodiff mentions -- include only as "SBI with a fast simulator", not as a differentiable-code use.*

### Considered and excluded

- Arora et al. 2026 (ApJ 1008, 91; arXiv:2605.16200), Pelissier et al. 2026 (arXiv:2609.10897), Phillips et al. 2026, Mohammed et al. 2026, Weatherford & Bonaca 2025, Panithanpaisal et al. 2025: cite StreamSculptor but do not use it or any differentiable code for their results.
- Starkman2023 (Stream Members Only; already in refs.bib): mixture-density networks for stream membership, not a dynamics-gradient use; mention only as the data-driven membership input to Nibauer2025b.
- Cook et al. 2026 (KRIOS basis-expansion N-body, arXiv:2602.13385): GPU-adjacent (compared to NBODY6++GPU) but not differentiable/jax.
- Lim et al. 2025 JFlow (arXiv:2505.00763): equivariant continuous normalizing flows for spherical Jeans analysis of dSphs -- differentiable DF but no galactic-dynamics code; omitted to keep the list focused.
- Silva, Valluri & Vasiliev 2024/2025 (minimum-entropy potentials), Das et al. 2026 (closed-form local potential): no autodiff/GPU claims found.

## Flags / things not verified

- **Chemaly2026** has no page number yet in Crossref (only the article id `stag1161`, vol. 550 issue 2, online 2026-06-19); `pages` is set to the article id. Re-check at submission.
- **Euclid2026** author list: taken from the arXiv record ("Euclid Collaboration: N. Starkman, J. Nibauer, ..."); truncated to 12 + `and others` per the refs.bib convention. The Euclid style normally wants `{Euclid Collaboration}: Starkman, N., ...`; adjust to the journal's preference.
- **Viterbo2026**: Crossref's title is only "The dynamical memory of tidal stellar streams" (the subtitle is a separate field); the arXiv title with subtitle is used.
- **Wu2026** (JOSS): `number = {126}`, `volume = {11}`, `pages = {10712}` from Crossref (issued 2026-10-03, i.e. four days before this note).
- Preprints (Wu2026b, Starkman2026b, Euclid2026, AnauMontel2026, Viterbo2025, SpurioMancini2026, Putney2024, Putney2025) have no DOI on their arXiv records as of 2026-10-07; cited as arXiv e-prints.
- galax has no paper; cite the Zenodo software record (Starkman2026 in refs.bib) -- confirmed again from the README.
- No usable ADS or Google Scholar access, so "cites galax" was established by grepping arXiv HTML full texts, not by an indexed full-text search; papers without an arXiv HTML rendering (none among the candidates, all rendered) or outside the StreamSculptor/Deep Potential/albatross citation graphs could have been missed.
- Semantic Scholar returned at most 100 citations per query; the StreamSculptor list had 16 and Deep Potential 19, so no truncation occurred.

## BibTeX for entries NOT already in refs.bib (all verified as marked above; author initials style matches refs.bib)

```bibtex
@article{Wu2026,
  author = {Wu, S. and Starkman, N. and Nibauer, J. and Pearson, S.},
  title = {{{Potamides}: {JAX} tools for curvature-based inference from stellar streams}},
  journal = {Journal of Open Source Software},
  year = {2026},
  volume = {11},
  pages = {10712},
  doi = {10.21105/joss.10712},
  number = {126}
}

@article{Wu2026b,
  author = {Wu, S. and Starkman, N. and Pearson, S. and Nibauer, J. and Miro-Carretero, J. and Martinez-Delgado, D.},
  title = {{{Potamides}: Mapping Dark Matter Halo Shapes from Stellar Stream Tracks in the Local Universe}},
  journal = {arXiv e-prints},
  year = {2026},
  pages = {arXiv:2605.07809},
  archivePrefix = {arXiv},
  eprint = {2605.07809}
}

@article{Starkman2026b,
  author = {Starkman, N. and Nibauer, J. and Pearson, S. and Wu, S. and Necib, L.},
  title = {{When Streams Curve Away: a Test of Dark Matter from Extragalactic Stellar Stream Populations}},
  journal = {arXiv e-prints},
  year = {2026},
  pages = {arXiv:2609.40057},
  archivePrefix = {arXiv},
  eprint = {2609.40057}
}

@article{Euclid2026,
  author = {{Euclid Collaboration} and Starkman, N. and Nibauer, J. and Pearson, S. and Wu, S. and Walmsley, M. and Necib, L. and Bovy, J. and Marleau, F.~R. and Sola, E. and Scott, D. and Altieri, B. and others},
  title = {{{Euclid} Quick Data Release ({Q}1): The geometry of dark matter halos from extragalactic streams}},
  journal = {arXiv e-prints},
  year = {2026},
  pages = {arXiv:2606.21774},
  archivePrefix = {arXiv},
  eprint = {2606.21774}
}

@article{AnauMontel2026,
  author = {Anau Montel, N. and Schmidt, F.},
  title = {{A differentiable forward model for weakly perturbed stellar streams: substructure forecasts from density and kinematics spectra}},
  journal = {arXiv e-prints},
  year = {2026},
  pages = {arXiv:2606.09629},
  archivePrefix = {arXiv},
  eprint = {2606.09629}
}

@article{Viterbo2025,
  author = {Viterbo, G. and Buck, T.},
  title = {{Differentiable {N}-body code for Galactic Dynamics -- {Odisseo}}},
  journal = {arXiv e-prints},
  year = {2025},
  pages = {arXiv:2511.22468},
  archivePrefix = {arXiv},
  eprint = {2511.22468}
}

@article{Viterbo2026,
  author = {Viterbo, G. and Buck, T.},
  title = {{The dynamical memory of tidal stellar streams: Joint inference of the Galactic potential and the progenitor of {GD}-1 with flow matching}},
  journal = {\aap},
  year = {2026},
  volume = {707},
  pages = {A363},
  doi = {10.1051/0004-6361/202558358},
  archivePrefix = {arXiv},
  eprint = {2512.04600}
}

@article{Nibauer2026b,
  author = {Nibauer, J. and Pearson, S.},
  title = {{Testing Dark Matter with Generative Models for Extragalactic Stellar Streams}},
  journal = {\apj},
  year = {2026},
  volume = {1002},
  pages = {204},
  doi = {10.3847/1538-4357/ae5bb0},
  archivePrefix = {arXiv},
  eprint = {2508.02666}
}

@article{Holm2026,
  author = {Holm, J.~K. and Pearson, S. and Nibauer, J. and Sand, D.~J. and Price-Whelan, A.~M. and Starkenburg, T. and Hendel, D. and Fielder, C.},
  title = {{Evidence for the first globular cluster stellar stream beyond the {M}ilky {W}ay}},
  journal = {\nat},
  year = {2026},
  volume = {656},
  pages = {843-847},
  doi = {10.1038/s41586-026-10878-w},
  archivePrefix = {arXiv},
  eprint = {2608.12254}
}

@article{Zaritsky2026,
  author = {Zaritsky, D. and Nibauer, J. and Golini, G. and Ruiz Cejudo, I. and Trujillo, I. and Pearson, S. and Chamba, N. and Chuang, C.-Y. and D'Onofrio, M. and Eskandarlou, S. and Guerra Arencibia, S. and Hosseini-ShahiSavandi, S.~Z. and others},
  title = {{{LIGHTS}. The Thin Encircling Stellar Stream of {NGC} 3938}},
  journal = {\apjl},
  year = {2026},
  volume = {1000},
  pages = {L6},
  doi = {10.3847/2041-8213/ae45a4},
  archivePrefix = {arXiv},
  eprint = {2602.14192}
}

@article{Chemaly2026,
  author = {Chemaly, D. and Sola, E. and Belokurov, V. and Koposov, S.~E. and Meyong, G. and Zhang, H. and Erkal, D.},
  title = {{Constraining the population of dark matter halo shapes using hierarchical inference with extragalactic stellar streams}},
  journal = {\mnras},
  year = {2026},
  volume = {550},
  pages = {stag1161},
  doi = {10.1093/mnras/stag1161},
  archivePrefix = {arXiv},
  eprint = {2601.15373}
}

@article{SpurioMancini2026,
  author = {Spurio Mancini, A.},
  title = {{Differentiable astrophysics at scale: solving and differentiating {ODE} ensembles on the {GPU}}},
  journal = {arXiv e-prints},
  year = {2026},
  pages = {arXiv:2609.28458},
  archivePrefix = {arXiv},
  eprint = {2609.28458}
}

@article{Tavangar2025,
  author = {Tavangar, K. and Price-Whelan, A.~M.},
  title = {{Inferring the Density and Membership of Stellar Streams with Flexible Models: The {GD}-1 Stream in {G}aia Data Release 3}},
  journal = {\apj},
  year = {2025},
  volume = {988},
  pages = {45},
  doi = {10.3847/1538-4357/addd1c},
  archivePrefix = {arXiv},
  eprint = {2502.13236}
}

@article{PriceWhelan2025,
  author = {Price-Whelan, A.~M. and Hunt, J.~A.~S. and Horta, D. and Oeur, M. and Hogg, D.~W. and Johnston, K. and Widrow, L.},
  title = {{Data-driven Dynamics with Orbital Torus Imaging: A Flexible Model of the Vertical Phase Space of the Galaxy}},
  journal = {\apj},
  year = {2025},
  volume = {979},
  pages = {115},
  doi = {10.3847/1538-4357/ad969a},
  archivePrefix = {arXiv},
  eprint = {2401.07903}
}

@article{Oeur2025,
  author = {Oeur, M. and Loebman, S.~R. and Price-Whelan, A.~M. and Arora, A. and Necib, L. and Horta, D.},
  title = {{{OTI} on {FIRE}: Testing the Efficacy of Orbital Torus Imaging to Recover the Galactic Potential}},
  journal = {\apj},
  year = {2025},
  volume = {993},
  pages = {29},
  doi = {10.3847/1538-4357/ae0584},
  archivePrefix = {arXiv},
  eprint = {2505.05590}
}

@article{Kalda2025,
  author = {Kalda, T. and Green, G.~M.},
  title = {{Deep Potential: Recovering the Gravitational Potential and Local Pattern Speed in the Solar Neighborhood with {GDR}3 Using Normalizing Flows}},
  journal = {\apj},
  year = {2025},
  volume = {992},
  pages = {84},
  doi = {10.3847/1538-4357/adf8ea},
  archivePrefix = {arXiv},
  eprint = {2507.03742}
}

@article{Kalda2026,
  author = {Kalda, T. and Green, G.~M.},
  title = {{Determining the {M}ilky {W}ay Gravitational Potential Without Selection Functions}},
  journal = {\apjl},
  year = {2026},
  volume = {1002},
  pages = {L47},
  doi = {10.3847/2041-8213/ae626c},
  archivePrefix = {arXiv},
  eprint = {2512.02115}
}

@article{Putney2024,
  author = {Putney, E. and Shih, D. and Lim, S.~H. and Buckley, M.~R.},
  title = {{Sweeping the Dust Away -- Correcting the Phase Space Density of the {M}ilky {W}ay with Unsupervised Machine Learning}},
  journal = {arXiv e-prints},
  year = {2024},
  pages = {arXiv:2412.14236},
  archivePrefix = {arXiv},
  eprint = {2412.14236}
}

@article{Putney2025,
  author = {Putney, E. and Shih, D. and Lim, S.~H. and Buckley, M.~R.},
  title = {{{ClearPotential}: Revealing Local Dark Matter in Three Dimensions}},
  journal = {arXiv e-prints},
  year = {2025},
  pages = {arXiv:2512.09989},
  archivePrefix = {arXiv},
  eprint = {2512.09989}
}

@article{Widmark2025,
  author = {Widmark, A. and Johnston, K.~V.},
  title = {{Using Simulation-based Inference on Tidally Perturbed Dwarf Galaxies: The Dynamics of {NGC} 205}},
  journal = {\apj},
  year = {2025},
  volume = {987},
  pages = {120},
  doi = {10.3847/1538-4357/add887},
  archivePrefix = {arXiv},
  eprint = {2501.13148}
}
```
