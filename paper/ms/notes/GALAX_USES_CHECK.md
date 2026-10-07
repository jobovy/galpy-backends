# GALAX_USES_CHECK.md -- fact-check of the "Codes built on these frameworks" paragraph (introduction.tex)

Date: 2026-10-07. Every cited paper was read in full text (arXiv PDF via `curl -sL -A Mozilla https://arxiv.org/pdf/<id>` + `pdftotext`).
Quotes below are verbatim from that text (pdftotext line-breaks/hyphenation repaired); section numbers are the papers' own.
Nothing in `introduction.tex` or `refs.bib` was edited.

## Verdict on the author's doubt

The doubt is justified. Of the four papers cited for galax, only **AnauMontel2026** uses galax's
differentiability (Fisher matrices by AD through a galax/diffrax stream simulation) and GPU (A100) for its main result.
**Wu2026b** and **Euclid2026** use galax only to *evaluate an analytic (logarithmic) halo acceleration field* whose
direction is compared with the curvature of a spline-fitted stream track; the halo likelihood is a non-differentiable
sign count evaluated on a parameter grid. There are **no stream models** and **no GPU fits** in either; the JAX
gradients they do use (Optax + BlackJAX NUTS) fit the *spline track to the image*, not the dynamics.
**Starkman2026b** uses galax as a plain orbit integrator (TriaxialNFWPotential) for a Monte-Carlo forecast; no
gradients, no GPU, no data fit. The sentence "measure the shapes of dark-matter halos ... by fitting large batches of
stream models on GPUs" is therefore not supported by any of the three papers it cites.

Two further claims outside the galax sentence are also wrong: **Nibauer2025b does not use Nibauer2025** (StreamSculptor)
-- it differentiates a spline fit to the GD-1 track (method of Nibauer et al. 2022), and its mass models use agama + emcee;
and **Chemaly2026 does not use gradients** (dynesty nested sampling; "we currently use JAX primarily for computational speed
rather than for gradient-based sampling"; main analyses on CPUs). Green2023 and Kalda2025 are **TensorFlow 2**, not
jax/torch, which matters because the paragraph opens with "Codes built on these frameworks" (= jax, torch).

## Classification table

| Paper | Dynamics code in main analysis | Gradients / jit / GPU used for main result? | Data | Class |
|---|---|---|---|---|
| Starkman2026 | (galax itself, Zenodo record) | -- | -- | n/a; "re-implementation of gala in jax" is fair (README: "if you've used gala, then galax should be familiar") |
| Wu2026b | galax (triaxial logarithmic potential, evaluated on the sky plane) | No AD of dynamics, no GPU; likelihood = sign count over (q, theta); galax also generates validation mocks | 15 SSLS streams (real) | PARTIAL |
| Euclid2026 | galax (logarithmic halo acceleration field) | AD only for the spline-track fit (Optax + BlackJAX NUTS); halo likelihood on a dense 10^6-point grid; benchmarks on a laptop CPU, GPU only "tested" | 13 Euclid Q1 hosts (real) | PARTIAL |
| Starkman2026b | galax (TriaxialNFWPotential orbit integration, 5000 orbits/potential) | No gradients, no GPU; forward Monte Carlo only | forecast + 2 simulated streams | PARTIAL |
| AnauMontel2026 | galax (via diffrax), in a JAX forward model; vmap on A100 GPU | Yes: Fisher matrices by JAX AD through the full simulator (w.r.t. subhalo-population parameters) | forecast, GD-1-like mock | CENTRAL |
| Viterbo2025 | own code Odisseo (JAX N-body); MW potential = galpy MWPotential2014 | Yes: gradient descent on MMD through the N-body run; Fisher via jax.hessian; multi-GPU scaling | mock GD-1 (4-page EurIPS workshop paper) | CENTRAL (mock data) |
| Chemaly2026 | own code StreaMAX (JAX particle spray) | JIT/vmap for speed only; dynesty nested sampling, HMC tried and abandoned; main analyses on CPUs | mocks only | PARTIAL |
| Nibauer2025 | own code StreamSculptor (JAX + diffrax), GPU | AD for forces, tidal tensors, release-function Jacobian; perturbations from *analytically derived* linearized EOM along unperturbed orbits (pure AD judged too costly); no fit -- a forecast of GD-1's dispersion | forecast vs. GD-1 dispersion | PARTIAL |
| Nibauer2025b | none in JAX: spline track (interpax) + BlackJAX NUTS; mass models with agama + emcee | HMC (NUTS) on spline knots, yes; no dynamics code differentiated; does **not** use Nibauer2025 | GD-1 (real) | NOT SUPPORTED as written (link to Nibauer2025 false); rephrased claim supported |
| Green2023 | own code Deep Potential, **TensorFlow 2** | Yes (AD of flow + NN potential trained on CBE) | mocks only | PARTIAL (framework; mock) |
| An2021 | nflows (**PyTorch**) MAFs for the DF; accelerations by a semi-analytic direct CBE solution (no potential network) | AD derivatives of the flow DF, yes | mocks only | PARTIAL (mostly a uniqueness-theory paper) |
| Buckley2023 | nflows/**PyTorch** MAFs; accelerations and density computed directly from the CBE | AD derivatives of the flow DF, yes | simulated galaxy h277 with Gaia-like errors | PARTIAL (simulation; real-data follow-up is Lim2025) |
| Kalda2025 | Deep Potential, **TensorFlow 2** | Yes (NF DF + NN potential, CBE loss) | 5.6 M Gaia DR3 stars (real) | CENTRAL for the method claim, but not jax/torch |
| PriceWhelan2025 | own code torusimaging (JAX) | Yes: JAX AD, JAXopt L-BFGS-B, BlackJAX NUTS | simulated data only | PARTIAL (fits a vertical phase-space orbit foliation / a_z(z), not "the Galactic potential"; no real-data fit) |

## Per-paper evidence

### Wu2026b -- arXiv:2605.07809 (Potamides) -- PARTIAL
(a) galax evaluates the projected acceleration field of a triaxial logarithmic halo.
(b) No stream models, no AD of dynamics, no GPU. The word "JAX" occurs only in the Software line; "GPU" never occurs.
- §3.3 Gravitational Potential: "In practice, we implemented this 2D projected field using the galax dynamics package, which provides performant implementations of all the common analytic potentials, specifically the triaxial logarithmic parametrization introduced by Law et al. (2009)."
- §3.3: "logarithmic potentials yield fully analytic gradients" (the acceleration is analytic, Eq. 2; only its direction is used).
- §3.4 Likelihood: pass/fail fractions f1 = (1/N) Σ ½[1 + sgn(â_xy,i · κ̂_i)], log-likelihood N(f1 ln f1 + f2 ln f2) -- a non-differentiable sign count; results are "Normalized profile likelihood L(q, θ)/Lmax" maps (Figs. 3-8).
- §5.2.2: "we ran tests using mock streams generated with galax in known flattened potentials and successfully recovered posteriors bounded strictly away from unity" (side check).
- Appendix A: the spline track is optimized with a cost using "a differentiable approximation to the sign function" (the only place where differentiability enters; it is the track, not the dynamics).

### Euclid2026 -- arXiv:2606.21774 -- PARTIAL
(a) galax (with coordinax, unxt) inside a JAX pipeline; the potential is a single-component logarithmic halo (Eq. 5).
(b) Gradients fit the spline track, not the potential; halo likelihood on a grid; laptop CPU benchmark.
- §3.1.3 Posterior exploration: "First, we apply gradient-based optimisation using the Optax (DeepMind 2020) library to minimise the total loss function and identify a good initialisation for the knot positions ... We then explore the full posterior distribution over the spline parameters using Hamiltonian Monte Carlo (HMC), specifically No-U-Turn ... implemented with BlackJAX".
- §3.3 Likelihood of a stream: Eq. (7) is an indicator, 1 if |θ_T,i| < π/2 else 0 ("this likelihood expresses the survival requirement that a stellar stream is sufficiently aligned with the local acceleration field").
- §5.4 Computational speed: "Potential likelihood evaluation – Finally, we evaluate the curvature likelihood over a dense 10^6-point grid of parameters while also marginalising over about 50 realisations of the spline. This step takes only around 15 s per segment." ... "These benchmarks are based on a 2023 M2 MacBook Pro ... The pipeline is also GPU-compatible – tests show runtimes for steps 2 and 3 combined ≲ 1 minute ... On a cluster with 100 GPU cores, 1000 stream segments could be analysed in under 10 minutes." ... "the pipeline is implemented in JAX ..., using the custom libraries galax ..., coordinax ..., and unxt ..., for high performance and auto-differentiation features."
- Abstract: "From our study of 13 galaxies ... q = 0.95 +0.05/−0.10".

### Starkman2026b -- arXiv:2609.40057 -- PARTIAL
(a) galax as an orbit integrator in a triaxial NFW potential; potamides + phasecurvefit for tracks of two simulated streams.
(b) Forward Monte Carlo only; no gradients; GPU is mentioned only for *other* groups' particle-spray work.
- §5.1 The Stream Convexity Rate: "For each potential, we integrate 5000 orbits for 5 Gyr ... Viewing directions are isotropic, with 7200 external views ... Each host uses a triaxial NFW density model (galax.TriaxialNFWPotential; N. Starkman et al. 2024a), with potential calculated through the Chandrasekhar ellipsoidal formalism".
- §1: "particle-spray models ... and with GPU acceleration can search over halo parameters for external galaxies (J. Nibauer & S. Pearson 2026; D. Chemaly et al. 2026a)" -- this is a statement about other papers, not this one.
- Abstract: "We forecast that upcoming survey-scale stream morphology catalogs can discriminate between CDM and SIDM at up to 5σ."

### AnauMontel2026 -- arXiv:2606.09629 -- CENTRAL
(a) galax (diffrax) for all orbit integration, inside a jit-compiled, vmapped JAX simulator.
(b) Yes: AD through the whole simulator for the Fisher forecasts (main result); GPU (A100).
- §III A Implementation: "for the gravitational evolution we use the open-source galactic dynamics library galax [37], which solves the equations of motion ẍ = −∇Φ(x) for each star particle via diffrax [38]"; "JAX's vmap transformation allows this to be easily executed on GPU"; AD is "a key ingredient for gradient-based inference, for computing Fisher matrices, and for optimal data compression."
- §III A: "The full simulation is implemented as a jax.lax.scan over Nk kick times, where each step consists of two sub-steps: (i) orbital integration of all star particles from the previous kick time to the current one via galax, and (ii) application of the velocity kick."
- §IV C Fisher forecast results: "Derivatives are computed via JAX automatic differentiation through the full forward model, including the stream simulator, the projection onto stream-frame observables, and the power spectrum estimators; we have verified that the resulting gradients agree with finite-difference derivatives".
- Appendix A: timings "on a single NVIDIA A100 GPU". Parameters differentiated: θ = (ρ̄, M_hm, α) of the subhalo population (not the smooth potential). Forecast on a GD-1-like mock.

### Viterbo2025 -- arXiv:2511.22468 (Odisseo) -- CENTRAL (mock, workshop paper)
(a) Own JAX N-body code Odisseo; no galax. Potential: "we adopt the MWPotential2014 model from galpy [3] as the analytic Milky Way potential" (§2.2.1). (The A&A follow-up Viterbo2026 takes "the default fiducial parameters in Galax ... for MWPotential2014" -- a parameter-value use, not a code use.)
(b) Yes, AD through the N-body integration is the demonstration.
- Abstract: "we present a case study using a mock GD-1 stellar stream simulation, where we optimize four physical parameters via gradient descent: the accretion time and progenitor mass, as well as the masses of the host galaxy's [NFW halo and MN disk]".
- §2.2.1: "The resulting simulated stream is compared to the observed stream using the Maximum Mean Discrepancy (MMD), and the gradient of the loss is computed. 5. We perform 10 steps of vanilla Gradient Descent".
- §2.2.2 Orbit fitting and Fisher contours: "The Fisher matrix is trivially obtained using jax.hessian."
Note: the paragraph's "allow stream progenitors and their orbits to be fit by gradient descent" omits that the host-potential masses are also fit, and that the data are a mock.

### Chemaly2026 -- arXiv:2601.15373 (MNRAS 550, stag1161) -- PARTIAL
(a) Own JAX particle-spray code StreaMAX.
(b) Compilation/vectorization for speed; no gradients; main analyses on CPUs; mock data only.
- §2.1 Stream Generation: "we built StreaMAX, a fully JAX ... accelerated package that exactly reproduces conventional spray generators, such as the ones in GALA ... and AGAMA ..., while gaining orders of magnitude of speed-up through compilation and efficient parallelism." ... "Although GPUs provide the best performance, our main analyses use CPUs due to wider availability on our computing resources."
- §3.1 Likelihood: "we use dynamic nested sampling with Dynesty ... Typical single-stream fits on a 44-core CPU take ∼10 hours. Although StreaMAX is fully differentiable, we currently use JAX primarily for computational speed rather than for gradient-based sampling. We have experimented with Hamiltonian Monte Carlo (HMC) but found that it struggles with the multi-modality present in our posteriors".
- Abstract: "Using mock data, we show that individual fits recover the correct flattening ...".

### Nibauer2025 -- arXiv:2410.21174 (StreamSculptor) -- PARTIAL
(a) Own JAX + diffrax code StreamSculptor (independent of galax); GPU.
(b) AD is used for forces/tidal tensor and the release-function Jacobian; the perturbative response is from analytically derived forward-mode (linearized) equations integrated along unperturbed orbits. The main result is a forecast compared to GD-1's measured dispersion, not a fit.
- §2.7 Code Implementation: "We implement our code ... in the python library Jax ..., using the differential equation solvers from Diffrax ... All dynamics calculations presented in the main text ... will be carried out on a single GPU." ... "once a gravitational potential is supplied, AD can be used to rapidly calculate the force, tidal tensor, or even higher order derivatives. Additionally, we use AD to compute the Jacobian of the particle-spray release function" ... "While AD can even be used to obtain the perturbative corrections ..., doing so is computationally (and memory) intensive ... We exploit this feature of the problem, and compute unperturbed stream trajectories once".
- Abstract: "we develop a direct and efficient forward mode differentiation of Hamilton's equations of motion ... We forecast the velocity dispersion of the GD-1 stream, and find that observations are in agreement with a CDM subhalo population".
So "fit the perturbations of streams ... by differentiating Hamilton's equations" -> "compute (model) the response of streams to subhalos with a forward-mode linearization of Hamilton's equations".

### Nibauer2025b -- arXiv:2504.07187 -- NOT SUPPORTED as written
(a) No dynamics code in JAX. Track: interpax splines; mass models: AGAMA + emcee.
(b) HMC (BlackJAX NUTS) on the spline knots; accelerations are derivatives of the track. Builds on Nibauer et al. 2022 (N22), not on StreamSculptor.
- §1: "We use a direct differentiation of the GD-1 stream track in phase-space (Nibauer et al. 2022, hereafter N22) to infer the 3D Galactic acceleration field along the stream."
- §2: "Splines are implemented using the package Interpax" ... "Sampling of the spline knots is performed in JAX (Bradbury et al. 2018) using the NUTS sampler from Blackjax".
- §4: "We use AGAMA (Vasiliev 2019a) to derive a potential model from the density function, Eq. 5. Sampling is performed using emcee".
StreamSculptor / Nibauer et al. 2025a is not used. "which Nibauer2025b use" must go.

### Green2023 -- arXiv:2205.02244 (ApJ 942, 26) -- PARTIAL
- §2.4 Implementation: "In this paper, we implement Deep Potential in Tensorflow 2."
- Abstract: "Using the Collisionless Boltzmann Equation, we then find the gravitational potential – represented by a feed-forward neural network – that renders this distribution function stationary ... We demonstrate Deep Potential on mock datasets".
Method description accurate; framework is TensorFlow; mock data.

### An2021 -- arXiv:2106.05981 (MNRAS 506, 5721) -- PARTIAL
- §1: "Whereas Green & Ting (2020) proposed a neural network, we instead utilize an efficient and accurate semi-analytic method, based on a direct solution of the collisionless Boltzmann equation. We demonstrate the efficacy of our method on mock datasets".
- §6.1: "We use the implementation of MAFs in the publicly available software package nflows" (PyTorch); "the learned DF is everywhere exactly differentiable ... we can efficiently calculate exact derivatives, obviating the need for potentially noisy finite difference schemes."
Mostly a theory paper (uniqueness conditions); flow + CBE is accurate; accelerations are solved for directly, not via a trained potential.

### Buckley2023 -- arXiv:2205.01129 (MNRAS 521, 5100) -- PARTIAL
- Abstract: "we apply autoregressive flows to stars from a realistic simulation of a Milky Way-type galaxy to learn ... the stellar phase space density and its derivatives. With these as inputs, and under the assumption of dynamic equilibrium, the gravitational acceleration field and mass density can be calculated directly from the Boltzmann Equation".
- MAF Implementation and Training: "Our MAFs are built in nflows and PyTorch".
Simulation (h277 with Gaia-like errors) only; the Gaia DR3 application is Lim2025.

### Kalda2025 -- arXiv:2507.03742 (ApJ 992, 84) -- CENTRAL for the claim, framework caveat
- Abstract: "We make use of a sample of 5.6 million upper-main-sequence stars to map the full 3D gravitational potential ... We model the distribution of stars in 6D phase space using a normalizing flow and the gravitational [potential] using a neural network."
- §2.4 Implementation: "We implement Deep Potential in TensorFlow 2 ... and use TensorFlow Distributions". (Kalda2026 moved to JAX.)

### PriceWhelan2025 -- arXiv:2401.07903 (ApJ 979, 115) -- PARTIAL
(a) Own JAX code torusimaging.
(b) Yes: AD + JAXopt L-BFGS-B + BlackJAX NUTS. But simulated data only, and the model is a foliation of the vertical (z, v_z) phase space yielding a_z(z), not a fit of "the Galactic potential".
- §3.2: "We have implemented this new Orbital Torus Imaging (OTI) framework in Python using JAX ... to accelerate the model evaluation and to use automatic differentiation to compute the gradients of the model with respect to the (many) model parameters."
- §4.1: "We use a standard L-BFGS-B optimizer ... as implemented in JAXopt" ... "Hamiltonian Monte Carlo 'No U-turn Sampler' (NUTS) implemented in the blackjax ... Python package".
- §4 is titled "APPLICATIONS TO SIMULATED DATA"; abstract: "does a good job of fitting orbits to simulated stellar abundance data with varying degrees of realism."

## Proposed replacement (from "Codes built on these frameworks" to the end of the paragraph; ~200 words)

```latex
Codes built on these frameworks are now in use in cosmology
\citep[\eg,][]{Modi2021,Campagne2023,Li2024} and, over the past two
years, in galactic dynamics. \texttt{galax} \citep{Starkman2026}, a
\jax\ code modelled on \texttt{gala}, provides the halo potentials whose
acceleration directions are compared with the curvature of stellar
streams around other galaxies to constrain the shapes of their halos
\citep{Wu2026b,Euclid2026}, and the orbits used to forecast how well
such curvature tests separate dark-matter models \citep{Starkman2026b}.
\citet{AnauMontel2026} integrate streams perturbed by subhalos with
\texttt{galax} on a GPU and obtain Fisher forecasts by differentiating
the full simulation. Other \jax\ codes fit the progenitor and the host
potential of a mock stream by gradient descent through an $N$-body
simulation \citep{Viterbo2025}, generate the many particle-spray models
needed to infer the distribution of halo flattenings from a population
of streams \citep{Chemaly2026}, and compute the response of streams to
subhalos by integrating linearized equations of motion
\citep{Nibauer2025}. Gradient-based samplers in \jax\ fit a spline
model of the track of GD-1 to measure the accelerations along it
\citep{Nibauer2025b}, and a flexible model of vertical orbits to the
variation of stellar abundances \citep{PriceWhelan2025}. Normalizing
flows for the DF, differentiated automatically, recover the potential
from phase-space snapshots of simulated galaxies
\citep{Green2023,An2021,Buckley2023} and of 5.6 million \textit{Gaia}
stars \citep{Kalda2025}.
```

Changes vs. the current text, and why:
- galax sentence: "by fitting large batches of stream models on GPUs" removed -- none of Wu2026b/Euclid2026/Starkman2026b fits stream models or relies on GPUs; galax supplies the potential (Wu2026b, Euclid2026) or the orbits for a forecast (Starkman2026b).
- AnauMontel2026: kept; now says galax + GPU + differentiating the full simulation, which is exactly what the paper does.
- Viterbo2025: now says the host potential is fit too, and that the stream is a mock.
- Chemaly2026: no longer implies differentiability is used (it is not); the JAX code provides the many fast stream models.
- Nibauer2025: "fit ... by differentiating Hamilton's equations" -> "compute the response ... by integrating linearized equations of motion" (it is a forecast, and the linearization is derived analytically).
- Nibauer2025b: the false dependence on Nibauer2025 removed; the HMC is on a spline track model.
- Normalizing-flow papers: "differentiated automatically" without naming jax/torch, since Green2023 and Kalda2025 are TensorFlow 2; "simulated" vs. "Gaia" distinguishes the mock-only papers from Kalda2025.
- PriceWhelan2025: "fits the Galactic potential to the abundance gradients across action space" -> "fit a flexible model of vertical orbits to the variation of stellar abundances" (the paper fits only simulated data; if that should be explicit, add "in simulated disks", +3 words).
- The closing sentence ("Analyses such as these ... were not practical with codes that only return values") is dropped: most of the cited analyses (Wu2026b, Euclid2026, Starkman2026b, Chemaly2026) do not need derivatives. If a closing sentence is wanted, this one is supported (+~25 words):
  `Differentiating an entire simulation \citep{Viterbo2025,AnauMontel2026}, or fitting a neural-network potential to millions of stars \citep{Kalda2025}, is not practical with codes that only return values.`
- Side note for the galpy paper: Viterbo2025 uses galpy's MWPotential2014 as its Milky Way potential.
