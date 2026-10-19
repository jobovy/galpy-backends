# BIB_NOTES.md -- verification record for refs.bib

Date of verification: 2026-10-06 (two passes). Entries in `refs.bib`: 143. Compiles with `aasjournalv7.bst` (bibtex: 0 warnings, 143 bibitems).

## How entries were verified

Rule applied: nothing is in `refs.bib` unless title, authors, year and journal/volume/pages (or arXiv id, or software release) were confirmed from a primary source. No ADS token was available, so ADS was not used.

| code | source | what it confirmed |
|---|---|---|
| C | Crossref record of the DOI (`https://api.crossref.org/works/<doi>`) | title, author list, year, journal, volume, pages/article number |
| X | arXiv API (`https://export.arxiv.org/api/query?id_list=<id>`) | title, authors, arXiv id, and (where present) the publisher DOI / journal-ref |
| Z | Zenodo REST API record | software title, version, release date, DOI, creators |
| G | GitHub REST API (repository + latest release) | software name, URL, current version |
| J | JMLR paper page (`jmlr.org/papers/...`) | volume(number):pages, year |
| S | Springer Nature Link book/chapter page | authors (Hairer) and series volume (Skokos) missing from the Crossref record |
| E | ESO `doi.eso.org` landing page (schema.org metadata) | The Messenger issue 175, pages 3-11 (Crossref API returned no record for this DOI) |
| O | OpenAlex | cross-check of the 4MOST record (title/authors/DOI) |
| S2 | Semantic Scholar | **secondary** source; used only for Rossi2017's volume/pages (see caveats) |
| CQ | Crossref bibliographic query | used to *locate* a DOI absent from the arXiv record (title+authors+journal); the entry was then fetched by DOI (C) |
| B | the galpy paper's own bibliography (Bovy 2015 source, `scratchpad/galpy_paper/ms.in`) | Teuben1995 only: not in Crossref/arXiv; taken verbatim from the published galpy paper's thebibliography |

Author lists were taken programmatically from the Crossref/arXiv records (Crossref `family, given`; arXiv `First Last` split on the last token), not retyped. Lists longer than 15 names are truncated to 12 + `and others`. Titles were hand-braced for capitalisation. Collaboration author-less records (Astropy, Gaia) were given `{Astropy Collaboration}` / `{Gaia Collaboration}` as first author.

### Caveats on specific entries
- **Rossi2017**: Crossref has the DOI 10.1093/mnras/stx098 (title, authors, journal, year) but its record is the advance-access version with no volume/pages; the arXiv record (1608.02000) confirms the DOI. Volume 467, pages 1844-1856 come from Semantic Scholar only. Drop the volume/pages or re-check against OUP if you want strictly primary data.
- **deJong2019** (4MOST): the Crossref API returns nothing for 10.18727/0722-6691/5117, but doi.org resolves it to the ESO landing page whose schema.org metadata gives title, authors, issue 175 and pages 3-11; OpenAlex agrees. Author list truncated to three + `and others` (the full list is long and was not transcribed).
- **Bovy2011**: Crossref lacks pages; pages 1657-1677 (AoAS 5, No. 2B) come from the arXiv journal-ref of 0905.2979.
- **Campagne2023** (jax-cosmo): OJAp article number 15 is not in Crossref/OpenAlex; it is what OJAp assigns (vol. 6, 2023, id 15). Keep or drop `pages = {15}` as you prefer; DOI is verified.
- **Chen2025** (chen24spraydf): the published version is Chen, Valluri, Gnedin & Ash 2025, ApJS 276, 32 (Crossref), arXiv:2408.01496; galpy's docs still cite it as 'Chen et al. (2024)'.
- **Petts2016**: Crossref lists the authors as Petts, Read & Gualandris (galpy's docstring adds Bovy; the Crossref record does not).
- **BinneyMcMillan2016, Vasiliev2019, Vasiliev2021, McMillan2017, SchonrichBinney2012**: Crossref 'issued' year is the online year (2015/2018/2020/2016/2011); the print-volume year is used, matching ADS.
- **King1966**: Crossref title contains an OCR error ('dvriamical'); corrected to 'dynamical'.
- **Hairer1993**: the Crossref record for the book DOI has no author field; authors (Hairer, Norsett, Wanner) confirmed from the Springer page. Series volume 8 is from the Springer series listing.
- **Skokos2010**: Lecture Notes in Physics vol. 790 confirmed from the Springer chapter page (Crossref lacks the volume).
- **Neal2011**: chapter DOI 10.1201/b10905-6 (pages 113-162) verified in Crossref; editors from the handbook record 10.1201/b10905.
- **Software entries** (Bradbury2018/JAX, PriceWhelan2026/gala, Starkman2026/galax, Teuben2026/NEMO): @misc with URL and version from the GitHub/Zenodo API (JAX jax-v0.11.2 2026-09-17; gala v1.12.0 2026-09-27, DOI 10.5281/zenodo.23003195, concept 10.5281/zenodo.593786; galax v0.0.3 2026-08-17, DOI 10.5281/zenodo.21985205, concept 10.5281/zenodo.11553324). The JAX author list is the one in the project's own citation block; no release paper exists. Update versions at submission time.
- **arXiv-only entries** (Bovy2020, Starkman2023, GreenTing2020, Li2022, KidgerGarcia2021, LienenGunnemann2022, Phan2019, Cabezas2024, Kollmeier2017): no DOI found on the arXiv record as of 2026-10-06; cited as arXiv e-prints. Paszke2019 and Chen2018 are NeurIPS proceedings papers cited via their arXiv ids; Kidger2021 is an Oxford DPhil thesis cited via arXiv:2202.02435.
- **galax**: no JOSS/arXiv paper could be found (arXiv searches `ti:galax AND au:Starkman`, `all:galax AND all:JAX AND all:dynamics`, `au:Starkman_N AND all:galax` returned nothing; the repository README points only to Zenodo), so galax is a software entry.

## Unverified / not found (NOT in refs.bib)

- **Osipkov 1979**, Pis'ma Astron. Zh. 5, 77 (Sov. Astron. Lett. 5, 42): not in Crossref or OpenAlex. Merritt1985 is the verified reference for the Osipkov-Merritt DFs. Add by hand from ADS (1979PAZh....5...77O) if needed.
- **Kollmeier et al. 2025** (SDSS-V overview): no such paper found on arXiv (search `ti:SDSS-V AND au:Kollmeier`); only the 2017 white paper is included.
- **Webb et al. 2019a** (MNRAS 488, 5748; galpy's ttensor/rtide reference): Crossref (10.1093/mnras/stz2118) and OpenAlex have the DOI but no volume/pages; not needed for this paper, so left out.
- **Binney 2014** (torus mapping) and **Binney & McMillan 2016** alternative 'Binney 2014': the torus-mapper references used are McMillanBinney2008 and BinneyMcMillan2016; no separate 'Binney 2014' torus paper was sought.
- **Bovy 2014 action-angle paper**: does not exist as a separate paper (coordinator's correction); Bovy2014 is the streams/isochrone-approximation paper.
- **Bovy 2017 MWPotential2014 update**: skipped as instructed.
- **Lim et al. 2023**: the MNRAS/JCAP 'Mapping dark matter ... normalizing flows and Gaia DR3' paper is Lim2025 (JCAP 01(2025)021); its 2023 arXiv version is the same work.
- Items listed as 'skip' or '?' in the request (Hernitschek, Kipper, Sellwood 2013, Lindegren, Sanderson, Tamm, jax-galsim, 'dynamical friction in jax') were not sought.

## Per-entry verification and one-sentence description

Format: `key` -- how -- URL(s) -- description (for use in the text).

### galpy and the methods it implements

- `Bovy2015` -- C -- https://api.crossref.org/works/10.1088/0067-0049/216/2/29  
  Introduces galpy, the Python library for galactic dynamics whose jax/PyTorch backends this paper describes.
- `Bovy2014` -- C+X -- https://api.crossref.org/works/10.1088/0004-637X/795/1/95 ; https://export.arxiv.org/api/query?id_list=1401.2985  
  Builds the action-angle framework for modeling tidal streams (streamdf) and the isochrone-approximation action solver (actionAngleIsochroneApprox).
- `BovyRix2013` -- C+X -- https://api.crossref.org/works/10.1088/0004-637X/779/2/115 ; https://export.arxiv.org/api/query?id_list=1309.0809  
  Measures the Milky Way disk surface-density profile with qDF-based dynamical modeling; its appendix documents galpy's Staeckel-fudge implementation.
- `Binney2010` -- C -- https://api.crossref.org/works/10.1111/j.1365-2966.2009.15845.x  
  Introduces the quasi-isothermal distribution function and the adiabatic approximation for disk actions (actionAngleAdiabatic, quasiisothermaldf).
- `Binney2012` -- C -- https://api.crossref.org/works/10.1111/j.1365-2966.2012.21757.x  
  Presents the Staeckel fudge for estimating actions in general axisymmetric potentials (actionAngleStaeckel).
- `SchonrichBinney2012` -- C -- https://api.crossref.org/works/10.1111/j.1365-2966.2011.19816.x  
  Derives a new formula for disc kinematics within the adiabatic (separable) approximation used for disk actions.
- `BinneyMcMillan2011` -- C -- https://api.crossref.org/works/10.1111/j.1365-2966.2011.18268.x  
  Builds dynamical models of the Galaxy from quasi-isothermal DFs, the second paper in the qDF Galaxy-model series.
- `SandersBinney2016` -- C -- https://api.crossref.org/works/10.1093/mnras/stw106  
  Reviews and benchmarks action-estimation methods for galactic dynamics (including the TACT code).
- `McMillanBinney2008` -- C -- https://api.crossref.org/works/10.1111/j.1365-2966.2008.13767.x  
  Uses torus-mapping angle-action coordinates to disassemble the Galaxy (basis of actionAngleTorus).
- `BinneyMcMillan2016` -- C -- https://api.crossref.org/works/10.1093/mnras/stv2734  
  Describes the Torus Mapper code that galpy's actionAngleTorus wraps.
- `Sanders2016` -- C -- https://api.crossref.org/works/10.1093/mnras/stw232  
  Derives the dynamics of stream-subhalo interactions in action-angle space (streamgapdf).
- `Fardal2015` -- C -- https://api.crossref.org/works/10.1093/mnras/stv1198  
  Introduces the particle-spray method for generating mock tidal streams (fardal15spraydf).
- `Qian2022` -- C -- https://api.crossref.org/works/10.1093/mnras/stac238  
  Studies the structure of accreted streams and describes galpy's implementation of the Fardal particle-spray model.
- `Chen2025` -- C+X -- https://api.crossref.org/works/10.3847/1538-4365/ad9904 ; https://export.arxiv.org/api/query?id_list=2408.01496  
  Presents an improved particle-spray algorithm for globular-cluster streams (chen24spraydf).
- `Petts2015` -- C -- https://api.crossref.org/works/10.1093/mnras/stv2235  
  Develops a semi-analytic dynamical-friction model that reproduces core stalling, the basis for galpy's variable-Coulomb-logarithm friction force.
- `Petts2016` -- C -- https://api.crossref.org/works/10.1093/mnras/stw2011  
  Extends the semi-analytic dynamical-friction model to cored galaxies (ChandrasekharDynamicalFrictionForce).
- `Chandrasekhar1943` -- C -- https://api.crossref.org/works/10.1086/144517  
  Derives the classical dynamical-friction formula.
- `Bovy2011` -- C+X -- https://api.crossref.org/works/10.1214/10-AOAS439 ; https://export.arxiv.org/api/query?id_list=0905.2979  
  Introduces extreme deconvolution for fitting Gaussian mixtures to noisy, heterogeneous data.
- `Trick2016` -- C -- https://api.crossref.org/works/10.3847/0004-637X/830/2/97  
  Presents RoadMapping, action-based dynamical modeling of the Milky Way disk built on galpy's qDF machinery.
- `MackerethBovy2018` -- C -- https://api.crossref.org/works/10.1088/1538-3873/aadcdd  
  Describes the fast Staeckel-based estimator of orbital parameters (eccentricity, zmax, rperi, rap) in galpy.
- `Mackereth2019` -- C -- https://api.crossref.org/works/10.1093/mnras/stz1521  
  Measures dynamical heating across the disc with APOGEE and Gaia using galpy actions.
- `Bovy2016` -- C+X -- https://api.crossref.org/works/10.3847/1538-4357/833/1/31 ; https://export.arxiv.org/api/query?id_list=1609.01298  
  Constrains the shape of the inner Milky Way halo from the Pal 5 and GD-1 streams with streamdf.
- `Bovy2016b` -- C+X -- https://api.crossref.org/works/10.1103/PhysRevLett.116.121301 ; https://export.arxiv.org/api/query?id_list=1512.00452  
  Shows how stream density perturbations detect the disruption of dark-matter subhalos.
- `Bovy2017a` -- C+X -- https://api.crossref.org/works/10.1093/mnras/stx1277 ; https://export.arxiv.org/api/query?id_list=1704.05063  
  Performs a stellar inventory of the solar neighbourhood with Gaia DR1.
- `Bovy2017b` -- C+X -- https://api.crossref.org/works/10.1093/mnrasl/slx027 ; https://export.arxiv.org/api/query?id_list=1610.07610  
  Measures Galactic rotation (the Oort constants) from Gaia DR1.
- `Bovy2020` -- X -- https://export.arxiv.org/api/query?id_list=2012.02169  
  Derives fundamental Galactic parameters from accelerations alone, i.e. from gradients of the potential.
- `BovyTremaine2012` -- C+X -- https://api.crossref.org/works/10.1088/0004-637X/756/1/89 ; https://export.arxiv.org/api/query?id_list=1205.4033  
  Re-derives the local dark-matter density with a careful dynamical analysis.
- `BovyRixHogg2012` -- C+X -- https://api.crossref.org/works/10.1088/0004-637X/751/2/131 ; https://export.arxiv.org/api/query?id_list=1111.6585  
  Shows that the Milky Way has no distinct thick disk, using mono-abundance populations.
- `McMillan2017` -- C -- https://api.crossref.org/works/10.1093/mnras/stw2759  
  Provides the McMillan (2017) Milky Way mass model that galpy ships as McMillan17.
- `BinneyTremaine2008` -- C -- https://api.crossref.org/works/10.1515/9781400828722  
  Standard textbook on galactic dynamics (actions, DFs, dynamical friction, orbit integration).

### Other galactic-dynamics codes; ML and differentiable dynamics/cosmology

- `PriceWhelan2017` -- C -- https://api.crossref.org/works/10.21105/joss.00388  
  Introduces Gala, a Python package for galactic dynamics (JOSS).
- `PriceWhelan2026` -- Z -- https://zenodo.org/api/records/23003195  
  Zenodo software record for the current Gala release (v1.12.0).
- `Vasiliev2019` -- C -- https://api.crossref.org/works/10.1093/mnras/sty2672  
  Introduces AGAMA, the action-based galaxy-modelling architecture.
- `Starkman2026` -- Z+G -- https://zenodo.org/api/records/21985205 ; https://api.github.com/repos/GalacticDynamics/galax ; .../releases/latest  
  Zenodo/GitHub software record for galax, a JAX-based galactic-dynamics package with GPU and autodiff support (no paper found on arXiv as of 2026-10-06).
- `Dehnen2000` -- C -- https://api.crossref.org/works/10.1086/312724  
  Introduces the fast, momentum-conserving tree code behind gyrfalcON.
- `Dehnen2002` -- C -- https://api.crossref.org/works/10.1006/jcph.2002.7026  
  Describes the hierarchical O(N) force-calculation algorithm used by gyrfalcON.
- `Teuben2025` -- C -- https://api.crossref.org/works/10.26624/qvuj7010  
  Gives a retrospective view of the NEMO stellar-dynamics toolbox (ADASS XXXII).
- `Teuben2026` -- G -- https://api.github.com/repos/teuben/nemo  
  GitHub record for the NEMO stellar-dynamics toolbox software.
- `Nibauer2022` -- C+X -- https://api.crossref.org/works/10.3847/1538-4357/ac93ee ; https://export.arxiv.org/api/query?id_list=2205.11767  
  Charts Galactic accelerations from stellar streams with machine learning (differentiable, JAX-based stream modeling).
- `Nibauer2023` -- C+X -- https://api.crossref.org/works/10.3847/1538-4357/ace9bc ; https://export.arxiv.org/api/query?id_list=2303.17406  
  Constrains the gravitational potential from the projected morphology of extragalactic tidal streams.
- `Nibauer2025` -- C+X+CQ -- https://api.crossref.org/works/10.3847/1538-4357/adb8e8 ; https://export.arxiv.org/api/query?id_list=2410.21174 ; https://api.crossref.org/works?query.bibliographic=<title+authors+journal> (DOI located by search, then record fetched by DOI)  
  Presents StreamSculptor, Hamiltonian perturbation theory for streams in flexible potentials with differentiable (JAX) simulations.
- `Nibauer2026` -- C+X -- https://api.crossref.org/works/10.3847/1538-4357/ae6776 ; https://export.arxiv.org/api/query?id_list=2510.02247  
  Measures dark-matter substructure from the kinematics of the GD-1 stream.
- `Starkman2023` -- X -- https://export.arxiv.org/api/query?id_list=2311.16960  
  Characterizes stellar streams in a data-driven way with mixture density networks (Stream Members Only).
- `GreenTing2020` -- X -- https://export.arxiv.org/api/query?id_list=2011.04673  
  Introduces Deep Potential: recovering the gravitational potential from a phase-space snapshot with normalizing flows (NeurIPS 2020 workshop).
- `Green2023` -- C -- https://api.crossref.org/works/10.3847/1538-4357/aca3a7  
  Full journal version of Deep Potential: potential recovery from a phase-space snapshot via normalizing flows and a neural potential.
- `Buckley2023` -- C -- https://api.crossref.org/works/10.1093/mnras/stad843  
  Measures the Galactic dark-matter distribution with unsupervised machine learning (normalizing-flow DFs).
- `Lim2025` -- C+X -- https://api.crossref.org/works/10.1088/1475-7516/2025/01/021 ; https://export.arxiv.org/api/query?id_list=2305.13358  
  Maps dark matter in the Milky Way using normalizing flows applied to Gaia DR3.
- `An2021` -- C -- https://api.crossref.org/works/10.1093/mnras/stab2049  
  Establishes when a unique potential can be extracted from a distribution function (Charting galactic accelerations I).
- `Naik2022` -- C -- https://api.crossref.org/works/10.1093/mnras/stac153  
  Shows how to learn accelerations in the solar neighbourhood from normalizing-flow DFs (Charting galactic accelerations II).
- `Alvey2023` -- C -- https://api.crossref.org/works/10.1093/mnras/stad2458  
  Presents Albatross, a scalable simulation-based inference pipeline for stellar streams.
- `Hermans2021` -- C -- https://api.crossref.org/works/10.1093/mnras/stab2181  
  Constrains warm dark matter with GD-1 through neural simulation-based inference.
- `Li2022` -- X -- https://export.arxiv.org/api/query?id_list=2211.09958  
  Introduces pmwd, a differentiable cosmological particle-mesh N-body library in JAX.
- `Li2024` -- C+X -- https://api.crossref.org/works/10.3847/1538-4365/ad0ce7 ; https://export.arxiv.org/api/query?id_list=2211.09815  
  Develops differentiable cosmological simulation with the adjoint method (the pmwd methods paper).
- `Modi2021` -- C -- https://api.crossref.org/works/10.1016/j.ascom.2021.100505  
  Introduces FlowPM, a differentiable TensorFlow implementation of the FastPM N-body solver.
- `Campagne2023` -- C+X -- https://api.crossref.org/works/10.21105/astro.2302.05163 ; https://export.arxiv.org/api/query?id_list=2302.05163  
  Introduces jax-cosmo, an end-to-end differentiable, GPU-accelerated cosmology library.
- `Hahn2024` -- C -- https://api.crossref.org/works/10.1088/1475-7516/2024/06/063  
  Presents DISCO-DJ, a differentiable Einstein-Boltzmann solver in JAX.
- `JascheWandelt2013` -- C -- https://api.crossref.org/works/10.1093/mnras/stt449  
  Performs Bayesian (HMC) reconstruction of cosmological initial conditions (BORG).
- `JascheLavaux2019` -- C -- https://api.crossref.org/works/10.1051/0004-6361/201833710  
  Physical Bayesian modelling of the non-linear matter distribution with BORG.

### Frameworks, solvers, inference, automatic differentiation

- `Bradbury2018` -- G -- https://api.github.com/repos/jax-ml/jax ; .../releases/latest  
  Software reference for JAX, composable transformations (autodiff, vectorization, JIT) of Python+NumPy programs.
- `Paszke2019` -- X -- https://export.arxiv.org/api/query?id_list=1912.01703  
  Describes PyTorch, the imperative-style deep-learning library (NeurIPS 2019).
- `Kidger2021` -- X -- https://export.arxiv.org/api/query?id_list=2202.02435  
  Doctoral thesis on neural differential equations that introduces the Diffrax solver library.
- `KidgerGarcia2021` -- X -- https://export.arxiv.org/api/query?id_list=2111.00254  
  Introduces Equinox, neural networks and PyTree-based modules in JAX.
- `Chen2018` -- X -- https://export.arxiv.org/api/query?id_list=1806.07366  
  Introduces neural ODEs and the torchdiffeq adjoint-based differentiable ODE solvers (NeurIPS 2018).
- `LienenGunnemann2022` -- X -- https://export.arxiv.org/api/query?id_list=2210.12375  
  Introduces torchode, a parallel, batch-aware ODE solver for PyTorch.
- `Harris2020` -- C -- https://api.crossref.org/works/10.1038/s41586-020-2649-2  
  Describes the NumPy array-programming library.
- `Virtanen2020` -- C -- https://api.crossref.org/works/10.1038/s41592-019-0686-2  
  Describes SciPy 1.0 and its fundamental scientific-computing algorithms.
- `Hunter2007` -- C -- https://api.crossref.org/works/10.1109/MCSE.2007.55  
  Introduces Matplotlib, the 2D plotting library.
- `Astropy2013` -- C -- https://api.crossref.org/works/10.1051/0004-6361/201322068  
  Introduces the Astropy community Python package for astronomy.
- `Astropy2018` -- C -- https://api.crossref.org/works/10.3847/1538-3881/aabc4f  
  Describes the Astropy Project and the v2.0 core package.
- `Astropy2022` -- C -- https://api.crossref.org/works/10.3847/1538-4357/ac7c74  
  Describes the Astropy Project's sustainability and the v5.0 core package.
- `Phan2019` -- X -- https://export.arxiv.org/api/query?id_list=1912.11554  
  Introduces NumPyro, a JAX-backed probabilistic programming library with NUTS.
- `Bingham2019` -- X+J -- https://export.arxiv.org/api/query?id_list=1810.09538 ; http://jmlr.org/papers/v20/18-403.html  
  Introduces Pyro, deep universal probabilistic programming on PyTorch (JMLR).
- `Cabezas2024` -- X -- https://export.arxiv.org/api/query?id_list=2402.10797  
  Introduces BlackJAX, composable Bayesian inference (HMC/NUTS and more) in JAX.
- `HoffmanGelman2014` -- X+J -- https://export.arxiv.org/api/query?id_list=1111.4246 ; http://jmlr.org/papers/v15/hoffman14a.html  
  Introduces the No-U-Turn Sampler, the adaptive Hamiltonian Monte Carlo method used by Stan, NumPyro and BlackJAX (JMLR).
- `Duane1987` -- C -- https://api.crossref.org/works/10.1016/0370-2693(87)91197-X  
  Introduces Hybrid (Hamiltonian) Monte Carlo.
- `Neal2011` -- C+X -- https://api.crossref.org/works/10.1201/b10905-6 ; https://export.arxiv.org/api/query?id_list=1206.1901  
  Reviews Markov chain Monte Carlo using Hamiltonian dynamics.
- `Baydin2018` -- X+J -- https://export.arxiv.org/api/query?id_list=1502.05767 ; http://jmlr.org/papers/v18/17-468.html  
  Surveys automatic differentiation in machine learning (forward/reverse mode, implementations).
- `GriewankWalther2008` -- C -- https://api.crossref.org/works/10.1137/1.9780898717761  
  Textbook on the principles and techniques of algorithmic differentiation.
- `Hairer1993` -- C+S -- https://api.crossref.org/works/10.1007/978-3-540-78862-1 ; https://link.springer.com/book/10.1007/978-3-540-78862-1  
  Textbook on non-stiff ODE solvers, including the DOP853 Dormand-Prince 8(5,3) method.
- `DormandPrince1980` -- C -- https://api.crossref.org/works/10.1016/0771-050X(80)90013-3  
  Introduces the embedded Dormand-Prince Runge-Kutta formulae (dopr54).
- `Yoshida1990` -- C -- https://api.crossref.org/works/10.1016/0375-9601(90)90092-3  
  Constructs higher-order (e.g. fourth-order) symplectic integrators (symplec4).
- `LeimkuhlerReich2004` -- C -- https://api.crossref.org/works/10.1017/CBO9780511614118  
  Textbook on symplectic and geometric integration of Hamiltonian systems.

### Potentials and distribution functions

- `Hernquist1990` -- C -- https://api.crossref.org/works/10.1086/168845  
  Introduces the Hernquist potential-density pair for spheroids and bulges.
- `Navarro1997` -- C -- https://api.crossref.org/works/10.1086/304888  
  Introduces the universal NFW dark-matter halo density profile.
- `MiyamotoNagai1975` -- C -- https://api.crossref.org/works/10.1093/pasj/27.4.533  
  Introduces the Miyamoto-Nagai disk potential.
- `Plummer1911` -- C -- https://api.crossref.org/works/10.1093/mnras/71.5.460  
  Introduces the Plummer sphere.
- `Dehnen1993` -- C -- https://api.crossref.org/works/10.1093/mnras/265.1.250  
  Introduces the Dehnen family of two-power spherical potential-density pairs.
- `King1966` -- C -- https://api.crossref.org/works/10.1086/109857  
  Introduces the King lowered-isothermal models of star clusters (kingdf).
- `Eddington1916` -- C -- https://api.crossref.org/works/10.1093/mnras/76.7.572  
  Derives the Eddington inversion from density to an isotropic DF (eddingtondf).
- `Merritt1985` -- C -- https://api.crossref.org/works/10.1086/113810  
  Introduces the Osipkov-Merritt family of anisotropic spherical DFs (osipkovmerrittdf).
- `Cuddeford1991` -- C -- https://api.crossref.org/works/10.1093/mnras/253.3.414  
  Derives the analytic inversion for constant-anisotropy (constant-beta) spherical DFs (constantbetadf).

### Milky Way context and surveys

- `Vasiliev2021` -- C -- https://api.crossref.org/works/10.1093/mnras/staa3673  
  Models the Sagittarius stream with the LMC and a non-inertial Milky Way frame (time-dependent, moving potentials).
- `Erkal2019` -- C -- https://api.crossref.org/works/10.1093/mnras/stz1371  
  Measures the LMC mass from its perturbation of the Orphan stream.
- `Gaia2016` -- C -- https://api.crossref.org/works/10.1051/0004-6361/201629272  
  Describes the Gaia mission.
- `Gaia2018` -- C -- https://api.crossref.org/works/10.1051/0004-6361/201833051  
  Summarizes Gaia Data Release 2.
- `Gaia2023` -- C -- https://api.crossref.org/works/10.1051/0004-6361/202243940  
  Summarizes Gaia Data Release 3.
- `Ivezic2019` -- C -- https://api.crossref.org/works/10.3847/1538-4357/ab042c  
  Describes LSST's science drivers, reference design and data products.
- `Kollmeier2017` -- X -- https://export.arxiv.org/api/query?id_list=1711.03234  
  Describes the SDSS-V survey (white paper).
- `deJong2019` -- E+O -- https://doi.org/10.18727/0722-6691/5117 -> https://doi.eso.org/10.18727/0722-6691/5117 (schema.org metadata: issue 175, pages 3-11) ; https://api.openalex.org/works?search=4MOST%20Project%20overview%20and%20information%20for%20the%20First%20Call%20for%20Proposals  
  Gives the 4MOST project overview (The Messenger 175).
- `Jin2024` -- C -- https://api.crossref.org/works/10.1093/mnras/stad557  
  Describes the WEAVE spectroscopic survey design and overview.

### Chaos and Lyapunov exponents

- `Benettin1980a` -- C -- https://api.crossref.org/works/10.1007/BF02128236  
  Theory of computing all Lyapunov characteristic exponents via repeated renormalization (Part 1).
- `Benettin1980b` -- C -- https://api.crossref.org/works/10.1007/BF02128237  
  Numerical application of the renormalization method for Lyapunov exponents (Part 2); basis of Orbit.lyapunov.
- `Skokos2010` -- C+S -- https://api.crossref.org/works/10.1007/978-3-642-04458-8_2 ; https://link.springer.com/chapter/10.1007/978-3-642-04458-8_2  
  Reviews the Lyapunov characteristic exponents and their computation via variational equations.
- `LichtenbergLieberman1992` -- C -- https://api.crossref.org/works/10.1007/978-1-4757-2184-3  
  Textbook on regular and chaotic dynamics in Hamiltonian systems.

### Hypervelocity stars

- `Brown2015` -- C -- https://api.crossref.org/works/10.1146/annurev-astro-082214-122230  
  Reviews hypervelocity stars.
- `Hills1988` -- C -- https://api.crossref.org/works/10.1038/331687a0  
  Predicts hypervelocity stars from binaries disrupted by the Galactic-centre black hole (Hills mechanism).
- `Brown2005` -- C -- https://api.crossref.org/works/10.1086/429378  
  Reports the discovery of the first unbound hypervelocity star.
- `Gnedin2005` -- C -- https://api.crossref.org/works/10.1086/496958  
  Shows that hypervelocity-star trajectories probe the shape of the Galactic halo.
- `Rossi2017` -- C+X+S2 -- https://api.crossref.org/works/10.1093/mnras/stx098 ; https://export.arxiv.org/api/query?id_list=1608.02000 ; https://api.semanticscholar.org/graph/v1/paper/DOI:10.1093/mnras/stx098?fields=title,year,journal  
  Jointly constrains the Galactic dark-matter halo and Galactic Centre from hypervelocity stars.
- `Contigiani2019` -- C -- https://api.crossref.org/works/10.1093/mnras/stz1547  
  Assesses how well hypervelocity stars can measure the Galactic dark-matter halo.
- `Hattori2018` -- C -- https://api.crossref.org/works/10.3847/1538-4357/aadee5  
  Identifies old, metal-poor extreme-velocity stars in the solar neighbourhood with Gaia DR2.

### Papers using galpy cited in the introduction

- `Frankel2020` -- C+X -- https://api.crossref.org/works/10.3847/1538-4357/ab910c ; https://export.arxiv.org/api/query?id_list=2002.04622  
  Measures strong radial orbit migration but little heating in the Galactic disk from APOGEE red clump stars.
- `Xiang2022` -- C+X -- https://api.crossref.org/works/10.1038/s41586-022-04496-5 ; https://export.arxiv.org/api/query?id_list=2203.12110  
  Gives a time-resolved picture of the Milky Way's early formation history from ages and orbits of subgiants.
- `Belokurov2020` -- C+X -- https://api.crossref.org/works/10.1093/mnras/staa876 ; https://export.arxiv.org/api/query?id_list=1909.04679  
  Identifies the 'Splash', the heated proto-disc population produced by the Gaia-Sausage/Enceladus merger.
- `Lane2023` -- C+X+CQ -- https://api.crossref.org/works/10.1093/mnras/stad2834 ; https://export.arxiv.org/api/query?id_list=2306.03084 ; https://api.crossref.org/works?query.bibliographic=<title+authors+journal> (DOI located by search, then record fetched by DOI)  
  Measures the stellar mass of the Gaia-Sausage/Enceladus accretion remnant with APOGEE and Gaia using galpy DF modelling.
- `Fritz2018` -- C+X -- https://api.crossref.org/works/10.1051/0004-6361/201833343 ; https://export.arxiv.org/api/query?id_list=1805.00908  
  Derives Gaia DR2 proper motions and orbits for dwarf galaxies within 420 kpc and the implied Milky Way mass.
- `Pace2022` -- C+X -- https://api.crossref.org/works/10.3847/1538-4357/ac997b ; https://export.arxiv.org/api/query?id_list=2205.05699  
  Measures proper motions, orbits and tidal influences of Milky Way dwarf spheroidals with Gaia EDR3.
- `Boubert2018` -- C+X -- https://api.crossref.org/works/10.1093/mnras/sty1601 ; https://export.arxiv.org/api/query?id_list=1804.10179  
  Revisits hypervelocity-star candidates with Gaia DR2 and finds most are bound.
- `Marchetti2022` -- C+X -- https://api.crossref.org/works/10.1093/mnras/stac1777 ; https://export.arxiv.org/api/query?id_list=2206.06962  
  Searches Gaia DR3 6D data for hypervelocity stars and constrains the Galactic Centre environment.
- `ODoherty2023` -- C+X -- https://api.crossref.org/works/10.1093/mnras/stad680 ; https://export.arxiv.org/api/query?id_list=2303.01059  
  Derives an observational natal-kick distribution for neutron stars in binaries from their Galactic orbits.
- `Zucker2022` -- C+X -- https://api.crossref.org/works/10.1038/s41586-021-04286-5 ; https://export.arxiv.org/api/query?id_list=2201.05124  
  Shows that nearby star formation is driven by the expansion of the Local Bubble, tracing clusters back in time.
- `Swiggum2024` -- C+X -- https://api.crossref.org/works/10.1038/s41586-024-07496-9 ; https://export.arxiv.org/api/query?id_list=2406.06510  
  Shows that most nearby young star clusters formed in three massive complexes by integrating their orbits back in time.
- `Stefansson2023` -- C+X+CQ -- https://api.crossref.org/works/10.1126/science.abo0233 ; https://export.arxiv.org/api/query?id_list=2303.13321 ; https://api.crossref.org/works?query.bibliographic=<title+authors+journal> (DOI located by search, then record fetched by DOI)  
  Reports a Neptune-mass exoplanet around a very low-mass star that challenges formation models (Galactic orbit used for population context).
- `Forbes2025` -- C+X+CQ -- https://api.crossref.org/works/10.3847/1538-4357/adc9ac ; https://export.arxiv.org/api/query?id_list=2411.14577 ; https://api.crossref.org/works?query.bibliographic=<title+authors+journal> (DOI located by search, then record fetched by DOI)  
  Models the tidal streams of interstellar objects (He Awa Whiria) with galpy stream tools.
- `Tonnesen2019` -- C+X -- https://api.crossref.org/works/10.3847/1538-4357/ab0960 ; https://export.arxiv.org/api/query?id_list=1903.08178  
  Shows that realistic orbits matter when simulating ram-pressure stripping of satellite galaxies.
- `Malhan2019` -- C+X -- https://api.crossref.org/works/10.1093/mnras/stz1035 ; https://export.arxiv.org/api/query?id_list=1807.05994  
  Constrains the Milky Way halo potential with the GD-1 stream.
- `Banik2021` -- C+X -- https://api.crossref.org/works/10.1093/mnras/stab210 ; https://export.arxiv.org/api/query?id_list=1911.02662  
  Finds evidence for a population of dark subhaloes from Gaia and Pan-STARRS observations of GD-1 (streamgapdf/streampepperdf).
- `Hunt2018` -- C+X -- https://api.crossref.org/works/10.1093/mnras/sty921 ; https://export.arxiv.org/api/query?id_list=1803.02358  
  Explains the Hercules stream with the 4:1 outer Lindblad resonance of a long, slow bar.
- `Lucey2023` -- C+X -- https://api.crossref.org/works/10.1093/mnras/stad406 ; https://export.arxiv.org/api/query?id_list=2206.01798  
  Dynamically constrains the length of the Milky Way bar.
- `Cautun2020` -- C+X -- https://api.crossref.org/works/10.1093/mnras/staa1017 ; https://export.arxiv.org/api/query?id_list=1911.04557  
  Infers the Milky Way's total mass profile from Gaia DR2 (the Cautun20 potential in galpy).
- `Nibauer2025b` -- C+X+CQ -- https://api.crossref.org/works/10.3847/2041-8213/add0a9 ; https://export.arxiv.org/api/query?id_list=2504.07187 ; https://api.crossref.org/works?query.bibliographic=<title+authors+journal> (DOI located by search, then record fetched by DOI)  
  Infers Galactic accelerations from the GD-1 stream and finds they favour a tilted dark-matter halo.
- `Teuben1995` -- B -- galpy paper source: scratchpad/galpy_paper/ms.in, \bibitem[Teuben(1995)]{Teuben95a} (Teuben, P. J. 1995, in ADASS IV, ed. R. Shaw, H. E. Payne & J. J. E. Hayes, ASP Conf. Ser. 77, 398)  
  Introduces NEMO, the stellar-dynamics toolbox (ADASS IV proceedings).

### Stellar streams

- `Johnston1998` -- C -- https://api.crossref.org/works/10.1086/305273  
  Gives a prescription for building the stellar halo from disrupted satellites (stream formation).
- `HelmiWhite1999` -- C -- https://api.crossref.org/works/10.1046/j.1365-8711.1999.02616.x  
  Describes the build-up of the stellar halo and the phase-space structure of debris streams.
- `EyreBinney2011` -- C -- https://api.crossref.org/works/10.1111/j.1365-2966.2011.18270.x  
  Analyses the mechanics of tidal streams in angle-action space.
- `SandersBinney2013a` -- C -- https://api.crossref.org/works/10.1093/mnras/stt806  
  Shows the dangers of fitting orbits to streams because of stream-orbit misalignment.
- `SandersBinney2013b` -- C -- https://api.crossref.org/works/10.1093/mnras/stt816  
  Introduces an action-space algorithm to constrain the Galactic potential with streams.
- `BonacaHogg2018` -- C -- https://api.crossref.org/works/10.3847/1538-4357/aae4da  
  Quantifies the information content of cold stellar streams for the Galactic potential (Fisher/gradient-based).
- `Koposov2010` -- C -- https://api.crossref.org/works/10.1088/0004-637X/712/1/260  
  Constrains the Milky Way potential with a 6D phase-space map of the GD-1 stream.
- `PriceWhelanBonaca2018` -- C -- https://api.crossref.org/works/10.3847/2041-8213/aad7b5  
  Uses Gaia to reveal GD-1 stars off the main stream track (spur and gaps).
- `Bonaca2019` -- C -- https://api.crossref.org/works/10.3847/1538-4357/ab2873  
  Interprets the GD-1 spur and gap as dynamical evidence for a dark perturber.
- `ErkalBelokurov2015` -- C -- https://api.crossref.org/works/10.1093/mnras/stv655  
  Describes the three phases of gap growth in subhalo-stream encounters.
- `Pearson2017` -- C -- https://api.crossref.org/works/10.1038/s41550-017-0220-3  
  Shows that gaps and length asymmetry in Pal 5 can arise from the rotating Galactic bar.

## Additions of 2026-10-07 (third pass, by the coordinator)

Verified by DOI through the Crossref API (`https://api.crossref.org/works/<doi>`):
- `Shu1969` 10.1086/150214 — ApJ 158, 505; the Shu disk DF.
- `Dehnen1999` 10.1086/301010 — AJ 118, 1201 (not MNRAS); the Dehnen disk DF.
- `Evans1997` 10.1093/mnras/286.2.315 — MNRAS 286, 315, Evans, Häfner & de Zeeuw; the scale-free (power-law) models referenced by galpy's HISTORY for the analytic power-law spherical DFs (the exact reference galpy intends should be confirmed in the galpy docs).
- `Galassi2009` — GSL manual, ISBN 0954612078, as in the galpy paper's bibliography.

## Additions of 2026-10-07 (galpy component citations, as listed in galpy's docs "Acknowledging galpy")
- `Webb2019` 10.1093/mnras/stz2118 (Crossref; volume/page 488, 5748 from the ADS bibcode 2019MNRAS.488.5748W given in galpy's docs) — ttensor/rtide.
- `Webb2020` 10.1093/mnras/staa788 (Crossref: MNRAS 494, 2268) — to_amuse (galpy docs cite it as Webb et al. 2019b, arXiv:1910.01646).
- actionAngleVerticalInverse: cited in the text as "(J. Bovy, in prep.)", no reference-list entry (AAS style).

## ADS verification (2026-10-19)

Every refs.bib entry was looked up on ADS (by DOI, else arXiv ID, else title;
script and raw records: /var/tmp/claude-probes/ads/check_ads.py, compare.py, ads_raw.json)
and its first author, author list, year, title, volume, first page and DOI compared.
80 of the 171 entries are cited in the paper.

- 129 entries match ADS exactly.
- Fixed (both UNcited, so the rendered bibliography did not change): Starkman2023 is
  now published (ApJ 980, 253, 2025; doi 10.3847/1538-4357/ad94f2), and Ivezic2019's
  author "P. Angel, James Roger" -> "Angel, J. Roger P.".
- Benign differences: arXiv-only preprints without a DOI (ADS assigns 10.48550 arXiv DOIs;
  none of them has a published version on ADS); collaboration authors (Astropy, Gaia,
  Euclid); LaTeX accents/ADS markup (Zucker2022 Großschedl, Dehnen2002 title); ADS
  mis-parsing "Ngai Pok Kwan" (Li2022); Benettin1980a's ADS title truncated before
  "; a method for computing all of them. Part 1: Theory"; Kidger2021 is the 2021 Oxford
  thesis (arXiv posting 2022); published ML-venue versions that ADS only has as the
  arXiv preprint (Bingham2019 JMLR, HoffmanGelman2014 JMLR, Baydin2018 JMLR, Ponce2019
  PEARC, Neal2011 chapter DOI).
- Not on ADS (checked by DOI resolution or as software/books): Bradbury2018 (JAX),
  Galassi2009 (GSL manual), Starkman2026 (galax, Zenodo), PriceWhelan2026 (gala, Zenodo),
  Wu2026 (JOSS 10712), DormandPrince1980, Hairer1993, GriewankWalther2008; all DOIs resolve.
  Teuben2026 (NEMO's GitHub) matched ADS's ASCL record for NEMO (Barnes, Hut & Teuben 2010,
  ascl:1010.051) instead; uncited.
