# galpy: history (2010–2026) and applications — background notes for the backend paper's introduction

Notes, not prose. Sources are the galpy checkout at /var/tmp/fbp (branch feat/backends,
`galpy.__version__ = "1.12.1.dev0"`, galpy/__init__.py:1), its HISTORY.txt, doc/source/whatsnew.rst,
doc/source/index.rst, README.md, `git -C /var/tmp/fbp log`, and the live "papers using galpy" JSON that
the docs page loads from galpy.org (see Part B). All counts were produced on 2026-10-06; commands are
given so they can be re-run. Line numbers refer to the feat/backends checkout.

---------------------------------------------------------------------------------------------------
## Part A — galpy's history, by era, with trends

### A.0 Pre-history (2010–2014): a one-person research code becomes a package
- First commit 2010-07-10 ("pygd: start"; `git log --reverse`), GitHub repo created 2011-09-13 (GitHub API).
- C orbit integration + potential C extension started 2011-10-02; the *planar* variational equations
  (`integrate_dxdv`) existed by 2011-10-18 (`git log --grep=dxdv`); OpenMP 2012-12-18.
- quasiisothermaldf added 2012-07-25; streamdf 2013-09-16 (`git log --diff-filter=A`). Travis CI 2013-11, coveralls 2014-05.
- v0.1 2014-01-09; **v1.0 2014-12-10** (HISTORY.txt:1348), coincident with the method paper Bovy (2015),
  ApJS 216, 29, arXiv:1412.3451 (index.rst:404). v1.0 set the defaults still in use: MWPotential2014,
  `symplec4_c`, ro=8 kpc / vo=220 km/s, physical-output scaling, Orbit.fit (HISTORY.txt:1351–1363).
- State at v1.0 (`git show v1.0:galpy/potential.py`, `git ls-tree v1.0`): 22 potentials documented
  (21 classes + MWPotential), 13 potential modules with a C implementation (`self.hasC = True`); 9 actionAngle
  names (Adiabatic[+Grid], Staeckel[+Grid], Isochrone, IsochroneApprox, Spherical, Axi); DFs: diskdf
  (dehnendf, shudf), evolveddiskdf, quasiisothermaldf, streamdf. 36.3k Python LOC + 7.8k C LOC in galpy/;
  502 `def test_` in 14 nose files (12.9k LOC). Python 2.7 only (setup.py classifiers at v1.0 list none; v1.2 lists 2.7/3.3/3.4).

### A.1 Era 1 — 2015–2018 (v1.1 2015-06-30 … v1.4 2018-09-09): generality and community contributions
- v1.1: Python 3 (HISTORY.txt:1292); N-body snapshot potentials `SnapshotRZPotential`/`InterpSnapshotRZPotential`
  (Rok Roskar; :1296); NEMO export; Plummer, MN3ExponentialDisk, KuzminKutuzovStaeckel.
- v1.2 (2016-09-06): **astropy Quantity in/out everywhere** (PR #258; :1206); **SCFPotential** basis-function
  Poisson solver (Aladdin Seaifan; :1220) and TwoPowerTriaxial with C (:1210); experimental TorusMapper
  interface `actionAngleTorus` (:1234); `streamgapdf` for subhalo impacts (Sanders, Bovy & Erkal 2016; :1246);
  actionAngleIsochroneApprox generalised to triaxial potentials.
- v1.3 (2018-02-06): **potential wrappers** (DehnenSmooth, SolidBodyRotation; :1091) — the mechanism that later
  carries time dependence, rotation, tilting, adiabatic contraction; DiskSCFPotential (:1103); bars and spirals
  (SoftenedNeedleBar, Ferrers, 3D DehnenBar, SpiralArmsPotential; :1116–1135); Hénon–Heiles; Staeckel-based
  fast e/zmax/rperi/rap (Mackereth & Bovy 2018; :1085); nose→pytest (:1188); schwarzschilddf.
- v1.4 (2018-09-09): **dissipative forces**: `ChandrasekharDynamicalFrictionForce` and the Force/
  DissipativeForce class split (:972); `galpy.df.jeans` (:977); general `EllipsoidalPotential` (+C) and
  PerfectEllipsoid (:993); `Orbit.from_name` via SIMBAD (:1014) and SkyCoord initialisation (:1026);
  rtide/ttensor; Windows/MSVC builds; package layout galpy/potential/ etc.
- Trend: potential library grows 22 → 50 documented entries by v1.5 (`git show v1.5.0:doc/source/reference/potential.rst`),
  with the growth shifting from fixed analytic profiles to *general solvers* (SCF, DiskSCF, Ellipsoidal) and wrappers.
  AUTHORS.txt names ~18 direct code contributors in this and the next era (Seaifan, Trick, Hong, Oh/Juranova,
  Mackereth, Webb, Bub, Wong, Lane, Leung, Starkman, Bennett, Cautun, Dovgal…).

### A.2 Era 2 — 2019–2022 (v1.5 2019-09-12 … v1.8.1 2022-10-31): vectorised orbits, time dependence, spherical DFs, infrastructure
- v1.5 (2019-09-12): **full rewrite of Orbit as an arbitrary-shape array of orbits with parallel integration**
  (PR #384, merged 2019-06-05; HISTORY.txt:873) — "millions of objects" (whatsnew.rst:522ff); 1D C integration;
  `galpy.potential.mwpotentials` (McMillan17, Dehnen & Binney 1998 models, Irrgang13; :886); JSON of known
  objects for `from_name` with tab completion; `to_amuse` (:892); potential arithmetic `pot1+pot2`, `2*pot`;
  Dormand–Prince 8(5,3) (Henry Leung; :958); MovingObjectPotential with general mass distributions.
- v1.6 (2020-04-24): Chandrasekhar DF in C (100–1000x; :840); time dependence in all Potential functions;
  single `libgalpy` shared object (:864); **binary wheels built by GitHub Actions and uploaded on release** (:867).
- v1.7 (2021-07-04): **spherical DF framework** — Eddington, Osipkov–Merritt, Cuddeford/constant-β inversions
  for any density/potential pair, plus Hernquist/Plummer/NFW/King analytic DFs (:709). Constant-β DFs
  *required JAX* — JAX first entered the code base 2021-02-25 (`git log --grep=JAX`), as an optional dependency
  for one DF (README.md: "Constant-anisotropy DFs in galpy.df.constantbetadf require JAX"). Also: SCF
  coefficients from N-body, interpSphericalPotential, AdiabaticContractionWrapper, RotateAndTiltWrapper,
  AnySpherical / AnyAxisymmetricRazorThinDisk, TriaxialGaussian, PowerTriaxial; inverse action-angle for
  isochrone and harmonic oscillator (:785); bovy_* module renames; Travis → GitHub Actions + Windows CI (:822); Py3.9.
- v1.7.2 (2022-04-15): **NonInertialFrameForce** (arbitrary rotating/accelerating frames; motivated by the LMC's
  barycentre acceleration; :671; whatsnew.rst:250ff); TimeDependentAmplitudeWrapper; NullPotential; rE/LcE.
- v1.8.0 (2022-07-04): **streamspraydf** particle-spray (from jobovy/streamtools; :631); **pyodide/emscripten/WASM
  wheels and the "Try galpy" in-browser session** (:663); Python 2 fully dropped (:666); phiforce→phitorque;
  start of the *yearly release cycle* (major ~July 1, minors ~Nov 1 and Mar 1; whatsnew.rst:250ff).
  v1.8.1: cibuildwheel for Linux/Mac(arm64)/Windows, pre-commit, dependabot (:608–623).

### A.3 Era 3 — 2023–2026 (v1.8.2 2023-02-28 … v1.12.0 2026-07-01, v1.12.1 dev): chaos, inverse AA, higher-order integrators, derivatives
- v1.8.2 (2023-02-28): `actionAngleVertical` public + **`actionAngleVerticalInverse`** (custom torus machinery
  with point transformations; :578); Python 3.7 dropped; Orbit.animate3d.
- v1.9.0 (2023-07-02): **surfaces of section** (`Orbit.SOS`, `bruteSOS`; :533); `dMdE`; 2D dissipative/
  non-inertial forces; black formatting. v1.9.1: numpy-style docstrings "with the help of GitHub Copilot" (:521ff).
  v1.9.2: KuzminLikeWrapperPotential.
- v1.10.0 (2024-07-07): OpenMP with clang / Mac wheels, Ruff. v1.10.1 (2024-11-01): `chen24spraydf` and
  `fardal15spraydf` on a `basestreamspraydf` base (:492). v1.10.2 (2025-03-03): **IAS15** adaptive integrator (:465).
- v1.11.0 (2025-07-02): Py3.8 dropped; v1.11.1 (2025-11-02): EinastoPotential, **FDM dynamical friction**,
  OblateStaeckelWrapperPotential, Py3.14 wheels, multi-version pyodide wheels (:431–444).
- v1.11.2 (2026-02-27): `CompositePotential` classes replacing lists (:367); continued/extended orbit integration
  (:377); `integrate(pot)` with automatic time (10 t_dyn; :400); C two-power potentials via vendored `xsf`
  hypergeometrics; CylindricallySeparablePotentialWrapper; Py3.14, Py3.9 dropped.
- v1.12.0 (2026-07-01): **3D variational equations (state-transition matrix) in C for most of the library**,
  incl. dissipative forces, "lays the groundwork for differentiable orbit integration" (:247); **`Orbit.lyapunov`**
  (Benettin et al. 1980); `MultipoleExpansionPotential` (+Disk variant; time-dependent); `StreamTrack`; ExpTruncNFW;
  analytic power-law spherical DFs (Evans et al. 1997); per-orbit time arrays; `tail=` for spray DFs;
  tutorials rewritten as executed Jupyter notebooks (whatsnew.rst:8–145).
- v1.12.1 (in progress, HISTORY.txt:1–243): accuracy overhaul of the action-angle quadratures (Staeckel/Adiabatic
  regularised at turning points, near-circular `actionAngleSpherical`), momentum-matched canonical map for
  `actionAngleVerticalInverse`, time-dependent SCF, SCF↔multipole translation.
- **feat/backends** (this paper): 509 commits since 2026-06-04 not on main (`git log origin/main..HEAD`), adding
  `galpy.backend` — "multi-backend (numpy / jax / torch / array-API) dispatch … the same code runs and
  differentiates under numpy, JAX, and PyTorch. Backend selection follows the data first" (galpy/backend/__init__.py:1–12).
  Installation docs: `pip install galpy[jax]` / `galpy[torch]` with `array-api-compat` (installation.rst:62) and a
  `[backend] default = numpy|jax|torch` config option overridable with `galpy.backend.use(...)` (installation.rst:585–609).

### A.4 Quantitative anchors (how obtained)
| quantity | v1.0 (2014-12) | v1.5.0 (2019-09) | main (v1.12.x, 2026-10) | feat/backends (2026-10) | method |
|---|---|---|---|---|---|
| documented potentials/forces/wrappers | 22 | 50 | 72 (55 3D + 7 2D + 10 wrappers) | 72 | toctree entries in doc/source/reference/potential.rst (`git show <tag>:…`) |
| potential modules with C (`hasC = True`) | 13 | — | 42 (40 with `hasC_dxdv`) | 42 | grep over galpy/potential/*.py |
| actionAngle classes exported | 9 | — | 15 | 15 | `galpy/actionAngle/__init__.py` |
| DF classes exported | 6 | — | ~25 | ~25 | `galpy/df/__init__.py` |
| Python LOC in galpy/ | 36.3k | 54.4k | 93.2k | 121.2k | `git ls-tree -r <ref> -- galpy \| grep .py \| xargs git show \| wc -l` |
| C/H LOC in galpy/ | 7.8k | 13.6k | 25.8k | 27.8k | same, `\.(c|h)$` |
| test functions / files | 502 / 14 (nose) | 1167 / — | 2313 / 37 | 3862 / 127 | `grep -c '^def test_'` over tests/ (nose/ at v1.0) |
| test LOC | 12.9k | 34.3k | 114.7k | 169.8k | as above |
| coverage | — | — | 100% (codecov badge, main) | — | `curl codecov.io/gh/jobovy/galpy/branch/main/graph/badge.svg` |
| Python versions | 2.7 | 2.7, 3.3–3.7 | 3.10–3.15 (README.md:49; pyproject classifiers) | same | setup.py / pyproject classifiers per tag |
Release dates (`git for-each-ref refs/tags`): v1.0 2014-12-10, v1.1 2015-06-30, v1.2 2016-09-06, v1.3.0 2018-02-06,
v1.4.0 2018-09-08, v1.5.0 2019-09-12, v1.6.0 2020-04-24, v1.7.0 2021-07-04, v1.8.0 2022-07-03, v1.9.0 2023-07-02,
v1.10.0 2024-07-07, v1.11.0 2025-07-01, v1.12.0 2026-07-01. Repository: 8647 commits (8306 non-merge), 56 author
strings, Jo Bovy 7112 non-merge commits (`git shortlog -sn --no-merges`); per-year commit counts peak in 2014 (1346),
2016 (1089) and 2026 (746 so far, of which 309+19 by agent accounts). GitHub: 284 stars, 116 forks (API, 2026-10-06).

### A.5 Trends to use in the intro (one line each)
1. Potential library: count saturates (~70) while *generality* grows — SCF/multipole/Any*/wrapper Poisson solvers replace bespoke profiles.
2. Orbits: scalar Orbit (2010) → arrays of millions of orbits with multi-core parallelism (2019) → per-orbit times and continued integrations (2026).
3. Physics: static axisymmetric → non-axisymmetric bars/spirals (2014–18) → time-dependent, rotating, non-inertial, dissipative (2018–22) → chaos diagnostics (SOS 2023, Lyapunov 2026).
4. DFs: razor-thin disc DFs (2010) → quasi-isothermal 3D disc (2012) → streams (2013/2016/2022/2024) → spherical DFs (2021/2026).
5. Action-angle: forward transforms (2012–14) → inverse transforms (Torus 2016, isochrone/harmonic 2021, 1D 2023) → quadrature-accuracy overhaul (2026).
6. Derivatives: planar `dxdv` since 2011; full 3D state-transition matrix in C in 2026 — the explicit groundwork for differentiable galpy (HISTORY.txt:247–251).
7. Infrastructure: Travis+coveralls (2013/14) → GitHub Actions wheels (2020) → WASM/pyodide (2022) → 100% coverage, yearly cadence, notebooks as tests (2026).
8. JAX was an *optional island* (constantbetadf, 2021) for five years before becoming a first-class backend.

---------------------------------------------------------------------------------------------------
## Part B — "Papers using galpy"

### B.0 Where the list lives, and its size
- doc/source/index.rst no longer holds the list inline: index.rst:425–435 has a `papers-gallery` div and a
  hard-coded placeholder "more than <span id=span-number-of-papers-using-galpy>200</span>" that JavaScript
  overwrites at load (doc/source/_static/papers-gallery.js:4 fetches
  `https://d2g2jw00kypyq.cloudfront.net/data/papers-using-galpy.json`). The inline list existed in older tags
  (e.g. `git show v1.5.0:doc/source/index.rst` has ~200 arXiv/ADS links).
- JSON fetched 2026-10-06: **1057 keys, 1056 papers** (one `_template`). Fields: author, title, year, journal,
  volume, pages, url, img. URLs: 971 arXiv, 85 ADS, 1 other.
- **Year span 2012–2025** (no 2026 entries yet). Counts: 2012:4, 2013:3, 2014:2, 2015:17, 2016:21, 2017:30,
  2018:71, 2019:73, 2020:89, 2021:118, 2022:144, 2023:145, 2024:170, 2025:168. (Three malformed year fields:
  one blank, one "2021; ", one journal-in-year.) → ~70/yr post-Gaia DR2 (2018), ~150–170/yr since 2022.
- Journals: MNRAS 350, ApJ 241, A&A 194, AJ 73, ApJS 18, Nature 10, Nature Astronomy 8, Science 1, PRL 3+2,
  PRD 7+1, JCAP 2, Turkish J. Astron. Astrophys. 6 (Turkish-language theses/summaries), 47 with blank journal (arXiv-only).
- Bovy is a co-author on 59/1056 (6%), falling from 75–100% (2012–14) to 24% (2015–16), ~8–12% (2018–20),
  ≤4% since 2021 (0/170 in 2024): the list is overwhelmingly *independent* use.

### B.1 Classification method and caveats
First-match keyword classifier on *titles only* (regex buckets applied in the order below; script not retained,
rules summarised here). Precision per bucket is roughly 75–90% by spot check of 9–12 samples each; treat numbers
as ±20%. Order: streams → MW mass/potential → bar/bulge/spiral → HVS/runaway/compact objects → GCs/dwarfs/
satellites → halo/accretion/Gaia-substructure → disc chemodynamics → open clusters/young stars → planets/
encounters/solar system → extragalactic/simulations → stellar-astrophysics context → methods/software/teaching.

| bucket | first-match N | any-match N | 2015–18 | 2019–22 | 2023–25 |
|---|---|---|---|---|---|
| Galactic disc dynamics & chemodynamics (APOGEE/GALAH/LAMOST/Gaia-ESO, migration, heating, phase spiral, ages) | 233 | 492 | 33 | 92 | 106 |
| Stellar halo, accretion history, Gaia substructure (GSE, action-space clustering, metal-poor stars) | 186 | 311 | 21 | 78 | 87 |
| Hypervelocity/runaway stars, pulsars, NS/BH/WD kinematics & natal kicks | 154 | 166 | 24 | 50 | 77 |
| Globular clusters, dwarf galaxies, satellites (orbits, tides, LMC/Sgr association) | 138 | 199 | 18 | 52 | 68 |
| Stellar streams (modelling, gaps, subhaloes, potential constraints) | 99 | 99 | 22 | 42 | 34 |
| Open clusters, associations, young stars, tidal tails of clusters | 58 | 191 | 3 | 22 | 33 |
| Bar, bulge, spiral structure, resonances (Hercules etc.) | 55 | 60 | 7 | 24 | 24 |
| MW mass / potential / rotation curve / local DM / Oort constants / acceleration | 47 | 63 | 3 | 26 | 14 |
| Exoplanet hosts, stellar encounters, ISOs, Oort cloud, solar system | 31 | 80 | 2 | 15 | 14 |
| Extragalactic / cosmological simulations / other galaxies | 21 | 85 | 0 | 6 | 15 |
| Stellar astrophysics with a kinematic side-result (binaries, abundances, benchmark stars) | 17 | 485 | 2 | 8 | 7 |
| Methods / software / teaching | 6 | 71 | 0 | 3 | 3 |
| unclassified | 11 | — | 4 | 6 | 1 |
(Era totals: 2012–14: 9; 2015–18: 139; 2019–22: 424; 2023–25: 483.) The "any-match" column shows how much of the
list is stellar astrophysics whose title mentions stars/kinematics (485) — galpy's orbit integration is used
as a *side tool* to place an object in the Galaxy.

### B.2 Representative citations per topic (exact strings from the JSON; key in brackets)
**Milky Way mass / potential / local dark matter**
- [BovyRix13] Jo Bovy & Hans-Walter Rix (2013), "A direct dynamical measurement of the Milky Way's disk surface density profile, disk scale length, and dark matter profile at 4 kpc < R < 9 kpc", Astrophys. J., 779, 115, https://arxiv.org/abs/1309.0809
- [Bovy16c] Jo Bovy, Anita Bahmanyar, Tobias K. Fritz, & Nitya Kallivayalil (2016), "The shape of the inner Milky Way halo from observations of the Pal 5 and GD-1 stellar streams", Astrophys. J., 833, 31, https://arxiv.org/abs/1609.01298
- [Cautun20a] Marius Cautun, Alejandro Benítez-Llambay, Alis J. Deason, Carlos S. Frenk, Azadeh Fattahi, et al. (2020), "The milky way total mass profile as inferred from Gaia DR2", Mon. Not. Roy. Astron. Soc., 494, 4291, https://arxiv.org/abs/1911.04557
- [Bovy20a] Jo Bovy (2020), "A purely acceleration-based measurement of the fundamental Galactic parameters", Phys. Rev. Lett., submitted, https://arxiv.org/abs/2012.02169
- [Green22a] Gregory M. Green, Yuan-Sen Ting, & Harshil Kamdar (2023), "Deep Potential: Recovering the Gravitational Potential from a Snapshot of Phase Space", Astrophys. J., 942, 26, https://arxiv.org/abs/2205.02244
- [Buckley19a] Matthew R. Buckley, David W. Hogg, & Adrian M. Price-Whelan (2019), "Applying Liouville's Theorem to Gaia Data", Phys. Rev. D., https://arxiv.org/abs/1907.00987
- [Necib21a] Lina Necib & Tongyan Lin (2022), "Substructure at High Speed. II. The Local Escape Velocity and Milky Way Mass with Gaia eDR3", Astrophys. J., 926, 189, https://arxiv.org/abs/2102.02211

**Stellar streams (modelling, gaps, dark-matter subhaloes)**
- [Bovy14a] Jo Bovy (2014), "Dynamical modeling of tidal streams", Astrophys. J., 795, 95, https://arxiv.org/abs/1401.2985
- [Sanders15a] Jason L. Sanders, Jo Bovy, & Denis Erkal (2015), "Dynamics of stream-subhalo interactions", Mon. Not. Roy. Astron. Soc., 457, 3817, http://adsabs.harvard.edu/abs/2016MNRAS.457.3817S
- [Bovy16a] Jo Bovy (2016), "Detecting the disruption of dark-matter halos with stellar streams", Phys. Rev. Lett., 116, 121301, http://adsabs.harvard.edu/abs/2016PhRvL.116l1301B
- [Bovy16b] Jo Bovy, Denis Erkal, & Jason L. Sanders (2016), "Linear perturbation theory for tidal streams and the small-scale CDM power spectrum", Mon. Not. Roy. Astron. Soc., 466, 628, https://arxiv.org/abs/1606.03470
- [Banik19a] Nilanjan Banik, Jo Bovy, Gianfranco Bertone, Denis Erkal, & T. J. L. de Boer (2021), "Evidence of a population of dark subhaloes from Gaia and Pan-STARRS observations of the GD-1 stream", Mon. Not. Roy. Astron. Soc., 502, 2364, https://arxiv.org/abs/1911.02662
- [Banik21b] Nilanjan Banik, Jo Bovy, Gianfranco Bertone, Denis Erkal, & T. J. L. de Boer (2021), "Novel constraints on the particle nature of dark matter from stellar streams", Journal of Cosmology and Astroparticle Physics, 2021, 043, https://arxiv.org/abs/1911.02663
- [Malhan18a] Khyati Malhan & Rodrigo A. Ibata (2018), "Constraining the Milky Way Halo Potential with the GD-1 stellar stream", Mon. Not. Roy. Astron. Soc., 486, 2995, http://arxiv.org/abs/1807.05994
- [Reino21a] Stella Reino, Elena M. Rossi, Robyn E. Sanderson, Elena Sellentin, Amina Helmi, et al. (2021), "Galactic potential constraints from clustering in action space of combined stellar stream data", Mon. Not. Roy. Astron. Soc., 502, 4170, https://arxiv.org/abs/2007.00356
- [Qian22a] Yansong Qian, Yumna Arshad, & Jo Bovy (2022), "The structure of accreted stellar streams", Mon. Not. Roy. Astron. Soc., 511, 2339, https://arxiv.org/abs/2201.11045
- [Starkman23a] Nathaniel Starkman, Jo Bovy, Jeremy J. Webb, Daniela Calvetti, & Erkki Somersalo (2023), "On the fast track: Rapid construction of stellar stream paths", Mon. Not. Roy. Astron. Soc., 522, 5022, https://arxiv.org/abs/2212.00949
- [Nibauer24a] Jacob Nibauer, Ana Bonaca, David N. Spergel, Adrian M. Price-Whelan, Jenny E. Greene, et al. (2025), "StreamSculptor: Hamiltonian Perturbation Theory for Stellar Streams in Flexible Potentials with Differentiable Simulations", Astrophys. J., 983, 68, https://arxiv.org/abs/2410.21174   ← differentiable-stream work *in JAX* that cites/uses galpy; good hook for the paper
- [Nibauer25a] Jacob Nibauer & Ana Bonaca (2025), "Galactic Accelerations from the GD-1 Stream Suggest a Tilted Dark Matter Halo", Astrophys. J., 985, L22, https://arxiv.org/abs/2504.07187

**Galactic disc dynamics, chemodynamics, migration (APOGEE/GALAH/Gaia)**
- [Trick16a] Wilma H. Trick, Jo Bovy, & Hans-Walter Rix (2016), "Action-based Dynamical Modelling for the Milky Way Disk", Astrophys. J., 830, 97, http://arxiv.org/abs/1605.08601
- [Trick17a] Wilma H. Trick, Jo Bovy, Elena D'Onghia, & Hans-Walter Rix (2017), "Action-based Dynamical Modeling for the Milky Way Disk: The Influence of Spiral Arms", Astrophys. J., 839, 61, https://arxiv.org/abs/1703.05970
- [Mackereth19a] J. Ted Mackereth, Jo Bovy, Henry W. Leung, et al. (2019), "Dynamical heating across the Milky Way disc using APOGEE and Gaia", Mon. Not. Roy. Astron. Soc., 489, 176, http://arxiv.org/abs/1901.04502
- [Frankel20a] Neige Frankel, Jason Sanders, Yuan-Sen Ting, & Hans-Walter Rix (2020), "Keeping It Cool: Much Orbit Migration, yet Little Heating, in the Galactic Disk", Astrophys. J., 896, 15, https://arxiv.org/abs/2002.04622
- [Frankel23a] Neige Frankel, Jo Bovy, Scott Tremaine, & David W. Hogg (2023), "Vertical motion in the Galactic disc: unwinding the snail", Mon. Not. Roy. Astron. Soc., 521, 5917, https://arxiv.org/abs/2212.11991
- [Bennett19a] Morgan Bennett & Jo Bovy (2019), "Vertical waves in the solar neighbourhood in Gaia DR2", Mon. Not. Roy. Astron. Soc., 482, 1417, https://arxiv.org/abs/1809.03507
- [PriceWhelan20a] Adrian M. Price-Whelan, David W. Hogg, Kathryn V. Johnston, Melissa K. Ness, Hans-Walter Rix, et al. (2021), "Orbital Torus Imaging: Using Element Abundances to Map Orbits and Mass in the Milky Way", Astrophys. J., 910, 17, https://arxiv.org/abs/2012.00015
- [Ness19a] M. K. Ness, K. V. Johnston, K. Blancato, H. -W. Rix, A. Beane, et al. (2019), "In the Galactic Disk, Stellar [Fe/H] and Age Predict Orbits and Precise [X/Fe]", Astrophys. J., 883, 177, https://arxiv.org/abs/1907.10606
- [GaiaCollaboration22a] Gaia Collaboration, A. Recio-Blanco, G. Kordopatis, P. de Laverny, P. A. Palicio, et al. (2023), "Gaia Data Release 3. Chemical cartography of the Milky Way", Astron. & Astrophys., 674, A38, https://arxiv.org/abs/2206.05534
- [Xiang22a] Maosheng Xiang & Hans-Walter Rix (2022), "A time-resolved picture of our Milky Way's early formation history", Nature, 603, 599, https://arxiv.org/abs/2203.12110

**Galactic bar / bulge / spiral structure**
- [Hunt18a] Jason A. S. Hunt & Jo Bovy (2018), "The 4:1 Outer Lindblad Resonance of a long slow bar as a potential explanation for the Hercules stream", Mon. Not. Roy. Astron. Soc., 477, 3945, http://arxiv.org/abs/1803.02358
- [Hunt18b] Jason A. S. Hunt, Jack Hong, Jo Bovy, Daisuke Kawata, Robert J. J. Grand (2018), "Transient spiral structure and the disc velocity substructure in Gaia DR2", Mon. Not. Roy. Astron. Soc., 481, 3794, http://arxiv.org/abs/1806.02832
- [Trick19a] Wilma H. Trick, Francesca Fragkoudi, Jason A. S. Hunt, J. Ted Mackereth, & Simon D. M. White (2021), "Identifying resonances of the Galactic bar in Gaia DR2: I. Clues from action space", Mon. Not. Roy. Astron. Soc., 500, 2645, https://arxiv.org/abs/1906.04786
- [PrezVillegas17a] Angeles Pérez-Villegas, Matthieu Portail, Christopher Wegg, & Ortwin Gerhard (2017), "Revisiting the Tale of Hercules: How Stars Orbiting the Lagrange Points Visit the Sun", Astrophys. J., 840, L2, https://arxiv.org/abs/1702.06541
- [Lucey22a] Madeline Lucey, Sarah Pearson, Jason A. S. Hunt, Keith Hawkins, Melissa Ness, et al. (2023), "Dynamically constraining the length of the Milky way bar", Mon. Not. Roy. Astron. Soc., 520, 4779, https://arxiv.org/abs/2206.01798

**Stellar halo, accretion history, Gaia substructure**
- [Mackereth18b] J. Ted Mackereth, Ricardo P. Schiavon, Joel Pfeffer, et al. (2018), "The origin of accreted stellar halo populations in the Milky Way using APOGEE, Gaia, and the EAGLE simulations", Mon. Not. Roy. Astron. Soc., 482, 3426, http://arxiv.org/abs/1808.00968
- [Myeong17a] G. C. Myeong, N. W. Evans, V. Belokurov, S. E. Koposov, & J. L. Sanders (2017), "A Halo Substructure in Gaia Data Release 1", Mon. Not. Roy. Astron. Soc., 469, L78, https://arxiv.org/abs/1704.01363
- [Belokurov19a] Vasily Belokurov, Jason L. Sanders, Azadeh Fattahi, Martin C. Smith, Alis J. Deason, et al. (2020), "The biggest splash", Mon. Not. Roy. Astron. Soc., 494, 3880, https://arxiv.org/abs/1909.04679
- [Horta20b] Danny Horta, Ricardo P. Schiavon, J. Ted Mackereth, et al. (2020), "Evidence from APOGEE for the presence of a major building block of the halo buried in the inner Galaxy", Mon. Not. Roy. Astron. Soc., 500, 1385, https://arxiv.org/abs/2007.10374
- [Lane23a] James M. M. Lane, Jo Bovy, & J. Ted Mackereth (2023), "The stellar mass of the Gaia-Sausage/Enceladus accretion remnant", Mon. Not. Roy. Astron. Soc., 526, 1209, https://arxiv.org/abs/2306.03084
- [Montalban20a] Josefina Montalbán, J. Ted Mackereth, Andrea Miglio, Fiorenzo Vincenzo, Cristina Chiappini, et al. (2021), "Chronologically dating the early assembly of the Milky Way", Nature Astronomy, 5, 640, https://arxiv.org/abs/2006.01783
- [Rix22a] Hans-Walter Rix, Vedant Chandra, René Andrae, Adrian M. Price-Whelan, David H. Weinberg, et al. (2022), "The Poor Old Heart of the Milky Way", Astrophys. J., 941, 45, https://arxiv.org/abs/2209.02722

**Globular clusters, dwarf galaxies, satellites (and the LMC's effect)**
- [Fritz18a] T. K. Fritz, G. Battaglia, M. S. Pawlowski, N. Kallivayalil, R. van der Marel, T. S. Sohn, C. Brook, & G. Besla (2018), "Gaia DR2 Proper Motions of Dwarf Galaxies within 420 kpc: Orbits, Milky Way Mass, Tidal Influences, Planar Alignments, and Group Infall", Astron. & Astrophys., 619, A103, http://arxiv.org/abs/1805.00908
- [Simon18a] Joshua D. Simon (2018), "Gaia Proper Motions and Orbits of the Ultra-Faint Milky Way Satellites", Astrophys. J., 863, 89, http://arxiv.org/abs/1804.10230
- [Pace22a] Andrew B. Pace, Denis Erkal, & Ting S. Li (2022), "Proper Motions, Orbits, and Tidal Influences of Milky Way Dwarf Spheroidal Galaxies", Astrophys. J., 940, 136, https://arxiv.org/abs/2205.05699
- [Jethwa16a] P. Jethwa, D. Erkal, & V. Belokurov (2016), "A Magellanic Origin of the DES Dwarfs", Mon. Not. Roy. Astron. Soc., 461, 2212, http://arxiv.org/abs/1603.04420
- [Garrow20a] Turner Garrow, Jeremy J. Webb, & Jo Bovy (2020), "The Effects of Dwarf Galaxies on the Orbital Evolution of Galactic Globular Clusters", Mon. Not. Roy. Astron. Soc., 499, 804, https://arxiv.org/abs/2007.13752
- [Webb19a] Jeremy J. Webb, Jo Bovy, Raymond G. Carlberg, & Mark Gieles (2019), "Modelling the Effects of Dark Matter Substructure on Globular Cluster Evolution with the Tidal Approximation", Mon. Not. Roy. Astron. Soc., 488, 5748, http://arxiv.org/abs/1907.13132
- [Sawala22a] Till Sawala, Marius Cautun, Carlos Frenk, John Helly, Jens Jasche, et al. (2023), "The Milky Way's plane of satellites is consistent with ΛCDM", Nature Astronomy, 7, 481, https://arxiv.org/abs/2205.02860
- [Malhan22a] Khyati Malhan, Rodrigo A. Ibata, Sanjib Sharma, Benoit Famaey, Michele Bellazzini, et al. (2022), "The Global Dynamical Atlas of the Milky Way Mergers: Constraints from Gaia EDR3-based Orbits of Globular Clusters, Stellar Streams, and Satellite Galaxies", Astrophys. J., 926, 107, https://arxiv.org/abs/2202.07660

**Hypervelocity / runaway stars, pulsars, compact objects**
- [Boubert18b] Douglas Boubert, James Guillochon, Keith Hawkins, Idan Ginsburg, & N. Wyn Evans (2018), "Revisiting hypervelocity stars after Gaia DR2", Mon. Not. Roy. Astron. Soc., 479, 2789, http://arxiv.org/abs/1804.10179
- [Marchetti22a] Tommaso Marchetti, Fraser A. Evans, & Elena Maria Rossi (2022), "Gaia DR3 in 6D: the search for fast hypervelocity stars and constraints on the galactic centre environment", Mon. Not. Roy. Astron. Soc., 515, 767, https://arxiv.org/abs/2206.06962
- [Repetto17a.png] Serena Repetto, Andrei P. Igoshev, & Gijs Nelemans (2017), "The Galactic distribution of X-ray binaries and its implications for compact object formation and natal kicks", Mon. Not. Roy. Astron. Soc., 467, 298, https://arxiv.org/abs/1701.01347
- [ODoherty23a] Tyrone N. O'Doherty, Arash Bahramian, James C. A. Miller-Jones, Adelle J. Goodwin, Ilya Mandel, et al. (2023), "An observationally derived kick distribution for neutron stars in binary systems", Mon. Not. Roy. Astron. Soc., 521, 2504, https://arxiv.org/abs/2303.01059
- [Disberg25a] Paul Disberg & Ilya Mandel (2025), "The Kick Velocity Distribution of Isolated Neutron Stars", Astrophys. J., 989, L8, https://arxiv.org/abs/2505.22102
- [Burdge19a] Kevin B. Burdge, Michael W. Coughlin, Jim Fuller, et al. (2019), "General relativistic orbital decay in a seven-minute-orbital-period eclipsing binary system", Nature, 571, 528, http://arxiv.org/abs/1907.11291
- [ElBadry23a] Kareem El-Badry, Hans-Walter Rix, Yvette Cendes, Antonio C. Rodriguez, Charlie Conroy, et al. (2023), "A red giant orbiting a black hole", Mon. Not. Roy. Astron. Soc., 521, 4323, https://arxiv.org/abs/2302.07880
- [Archibald18a] Anne M. Archibald, Nina V. Gusinskaia, Jason W. T. Hessels, Adam T. Deller, David L. Kaplan, et al. (2018), "Testing the universality of free fall by tracking a pulsar in a stellar triple system", Nature, 559, 73, https://arxiv.org/abs/1807.02059

**Open clusters, associations, young stars (Gaia DR2/DR3 kinematics)**
- [Tarricq20a] Y. Tarricq, C. Soubiran, L. Casamiquela, T. Cantat-Gaudin, L. Chemin, et al. (2021), "3D kinematics and age distribution of the open cluster population", Astron. & Astrophys., 647, A19, https://arxiv.org/abs/2012.04017
- [Coronado20a] Johanna Coronado, Hans-Walter Rix, Wilma H. Trick, Kareem El-Badry, Jan Rybizki, et al. (2020), "From birth associations to field stars: mapping the small-scale orbit distribution in the Galactic disc", Mon. Not. Roy. Astron. Soc., 495, 4098, https://arxiv.org/abs/2002.09496
- [Swiggum24a] Cameren Swiggum, João Alves, Robert Benjamin, Sebastian Ratzenböck, Núria Miret-Roig, et al. (2024), "Most nearby young star clusters formed in three massive complexes", Nature, 631, 49, https://arxiv.org/abs/2406.06510
- [Zucker22a] Catherine Zucker, Alyssa A. Goodman, João Alves, Shmuel Bialy, Michael Foley, et al. (2022), "Star formation near the Sun is driven by expansion of the Local Bubble", Nature, 601, 334, https://arxiv.org/abs/2201.05124
- [Konietzka24a] Ralf Konietzka, Alyssa A. Goodman, Catherine Zucker, Andreas Burkert, João Alves, et al. (2024), "The Radcliffe Wave is oscillating", Nature, 628, 62, https://arxiv.org/abs/2402.12596

**Exoplanets, stellar encounters, interstellar objects, solar system**
- [Almeida-Fernandes18a] F. Almeida-Fernandes & H. J. Rocha-Pinto (2018), "A kinematical age for the interstellar object 1I/'Oumuamua", Mon. Not. Roy. Astron. Soc., 480, 4903, http://arxiv.org/abs/1808.03637
- [Forbes24a] John C. Forbes, Michele T. Bannister, Chris Lintott, Angus Forrest, Simon Portegies Zwart, et al. (2025), "He Awa Whiria: The Tidal Streams of Interstellar Objects", Astrophys. J., 988, 121, https://arxiv.org/abs/2411.14577
- [Webb19b] Jeremy J. Webb, Natalie Price-Jones, Jo Bovy, Simon Portegies Zwart, Jason A. S. Hunt, et al. (2020), "Searching for solar siblings in APOGEE and Gaia DR2 with N-body simulations", Mon. Not. Roy. Astron. Soc., 494, 2268, https://arxiv.org/abs/1910.01646
- [Stefnsson23a] Guðmundur Stefánsson, Suvrath Mahadevan, Yamila Miguel, Paul Robertson, Megan Delamer, et al. (2023), "A Neptune-mass exoplanet in close orbit around a very low-mass star challenges formation models", Science, 382, 1031, https://arxiv.org/abs/2303.13321
- [Spitoni25a] E. Spitoni, M. Palla, L. Magrini, F. Matteucci, C. Danielski, et al. (2025), "Shaping Galactic habitability: Impact of stellar migration and gas giants", Astron. & Astrophys., 700, A58, https://arxiv.org/abs/2506.19981
- [Chen24b] Huiling Chen, Yang Huang, Wei Zhu, Timothy C. Beers, Renjing Xie, et al. (2024), "The Kinematic and Chemical Properties of the Close-in Planet Host Star 8 UMi", Astrophys. J., 966, L27, https://arxiv.org/abs/2401.05120

**Extragalactic / other galaxies / simulations**
- [Tonnesen19a] Stephanie Tonnesen (2019), "The Journey Counts: The Importance of Including Orbits when Simulating Ram Pressure Stripping", Astrophys. J., 874, 161, http://arxiv.org/abs/1903.08178
- [Ghosh23a] Soumavo Ghosh, Wilma H. Trick, & Gregory M. Green (2023), "Quantifying the influence of bars on action-based dynamical modelling of disc galaxies", Mon. Not. Roy. Astron. Soc., 523, 991, https://arxiv.org/abs/2212.06184
- [Lane24a] James M. M. Lane & Jo Bovy (2025), "Distribution functions for the modelling of accretion remnants in Milky Way-like galaxies: insights from IllustrisTNG", Mon. Not. Roy. Astron. Soc., 538, 553, https://arxiv.org/abs/2405.10945
- [PortegiesZwart20a] Simon Portegies Zwart, Inti Pelupessy, Carmen Martínez-Barbosa, Arjen van Elteren, & Steve McMillan (2020), "Non-intrusive hierarchical coupling strategies for multi-scale simulations in gravitational dynamics", Communications in Nonlinear Science and Numerical Simulations, 85, 105240, https://arxiv.org/abs/2002.11206 (AMUSE bridge; cf. `to_amuse`)
- [Hamilton19a] Chris Hamilton & Roman R. Rafikov (2019), "Secular dynamics of binaries in stellar clusters I: general formulation and dependence on cluster potential", Mon. Not. Roy. Astron. Soc., 488, 5489, http://arxiv.org/abs/1902.01344

**Methods / software / teaching**
- [Mackereth18a] J. Ted Mackereth & Jo Bovy (2018), "Fast estimation of orbital parameters in Milky-Way-like potentials", Pub. Astron.Soc. Pac., 130, 114501, https://arxiv.org/abs/1802.02592
- [Hadden24a] Sam Hadden (2024), "Action-angle Variables for Axisymmetric Potentials via Birkhoff Normalization", Astrophys. J., 972, 64, https://arxiv.org/abs/2404.16941
- [Hyman24a] Sóley Ó. Hyman, Kathryne J. Daniel, & David A. Schaffner (2025), "PECCARY: A Novel Approach for Characterizing Orbital Complexity, Stochasticity, and Regularity", Astrophys. J., 987, 195, https://arxiv.org/abs/2407.11970
- [Kalda25a] Taavet Kalda & Gregory M. Green (2025), "Deep Potential: Recovering the gravitational potential and local pattern speed in the solar neighborhood with GDR3 using normalizing flows", https://arxiv.org/abs/2507.03742
- Teaching/training evidence: six Turkish-language entries in Turkish J. Astron. Astrophys. vol. 4 (2023; e.g. [Plevne23a] Olcay Plevne (2023), "Bir Samanyolu Tarifi", TJAA, 4, 225; [akmak23a] Hikmet Çakmak (2023), "Samanyolu Galaksisinin I. ve III. Çeyreğindeki Açık Kümelerin Dinamik Evrimi", TJAA, 4, 178), a textbook ([Gizis24a] John Gizis (2024), "An Introduction to Brown Dwarfs; From very-low-mass stars to super-Jupiters", IOP ebooks), and one workshop proceeding ("Star Clusters: the Gaia Revolution. Online workshop"). No explicit pedagogy papers in the list; the docs' own "Try galpy" WASM shell (index.rst:33–80) and schwarzschilddf "(useful for teaching)" (whatsnew.rst:681) are the teaching hooks.

### B.3 "Tool" vs "framework" mix
- **galpy as a tool (orbit integration / fast orbital parameters / action estimates for a sample of objects)**:
  the bulk — HVS/compact (154), GC/dwarf (138), open clusters (58), planets/encounters (31), stellar context (17),
  and most of disc chemodynamics (233) and halo/accretion (186) use `Orbit.integrate`, `e/zmax/rperi/rap`
  (Mackereth & Bovy 2018 Staeckel estimates), or `actionAngleStaeckel` actions in MWPotential2014/McMillan17 to
  *classify* or *contextualise* objects. Order of magnitude: 70–80% of the list.
- **galpy as a modelling framework (fit a DF/stream/potential to data)**: streams via streamdf/streamgapdf/
  streamspraydf (Bovy14a, Sanders15a, Bovy16a/b/c, Banik18–21, Qian22a, Starkman23a), action-based disc DF
  modelling with quasiisothermaldf (Trick16a/17a; Mackereth19a heating), spherical DF/Jeans halo modelling
  (Lane21a/23a/24a; Bird22a), potential inference from clustering/Liouville/accelerations (Reino21a,
  Buckley19a, Yang19a = Tianyi Yang, Supranta S. Boruah, & Niayesh Afshordi (2020), "Gravitational potential from small-scale clustering in action space: application to Gaia Data Release 2", MNRAS 493, 3061, arXiv:1908.02336; Bovy20a; Craig23a = Peter Craig, Sukanya Chakrabarti, Robyn E. Sanderson, & Farnik Nikakhtar (2023), "Building an Acceleration Ladder with Tidal Streams and Pulsar Timing", ApJ 945, L32, arXiv:2211.00613; Bird22a = Sarah A. Bird et al. (2022), "Milky Way mass with K giants and BHB stars using LAMOST, SDSS/SEGUE, and Gaia: 3D spherical Jeans equation and tracer mass estimator", MNRAS 516, 731, arXiv:2207.08839). Order of magnitude: 15–20%, concentrated among Bovy's group and
  its alumni early on, now spreading (Malhan18a, Reino21a, Necib21a, Nibauer24a/25a).
- Trend relevant to the paper: the "framework" papers are exactly the ones for which gradients matter
  (likelihood fitting of streams/DFs/potentials); recent entries doing this with differentiable JAX codes
  (Nibauer24a StreamSculptor; Green22a/Kalda25a Deep Potential; Starkman25a) still use galpy for
  potentials/comparisons — the gap the backends close.
- Context phrase in the docs (index.rst:430): "more than 200 scientific publications … properties of planetary systems
  around distant stars, the kinematics of pulsars and stars ejected from the Milky Way by supernova explosions, binary
  evolution in stellar clusters, chemo-dynamical modeling of stellar populations in the Milky Way, and the dynamics of
  satellites around external galaxies" — the number is stale (JSON: 1056) but the topic spread matches B.1.

---------------------------------------------------------------------------------------------------
## Part C — galpy's own method papers, as the docs give them

From doc/source/index.rst:402–419 ("Acknowledging galpy"), verbatim link text/targets:
- Main citation (index.rst:404): *galpy: A Python Library for Galactic Dynamics*, Jo Bovy (2015), *Astrophys. J. Supp.*, **216**, 29 (arXiv/1412.3451, http://arxiv.org/abs/1412.3451). README.md adds ADS 2015ApJS..216...29B and DOI 10.1088/0067-0049/216/2/29 (badge).
- `EccZmaxRperiRap` / `Orbit(..., analytic=True)` (index.rst:409): Mackereth & Bovy (2018), https://arxiv.org/abs/1802.02592  [JSON: Pub. Astron. Soc. Pac., 130, 114501]
- `actionAngleAdiabatic` (index.rst:410): Binney (2010), http://adsabs.harvard.edu/abs/2010MNRAS.401.2318B
- `actionAngleStaeckel` (index.rst:411): Bovy & Rix (2013), http://adsabs.harvard.edu/abs/2013ApJ...779..115B  and  Binney (2012), http://adsabs.harvard.edu/abs/2012MNRAS.426.1324B
- `actionAngleIsochroneApprox` (index.rst:412) and `streamdf` (index.rst:413): Bovy (2014), http://adsabs.harvard.edu/abs/2014ApJ...795...95B  [JSON: "Dynamical modeling of tidal streams", ApJ 795, 95, arXiv:1401.2985]
- `streamgapdf` (index.rst:414): Sanders, Bovy, & Erkal (2016), http://adsabs.harvard.edu/abs/2016MNRAS.457.3817S  [JSON: "Dynamics of stream-subhalo interactions", MNRAS 457, 3817]
- `chen24spraydf` (index.rst:415): Chen et al. (2024), https://ui.adsabs.harvard.edu/abs/2024arXiv240801496C/abstract  [JSON: Yingtian Chen, Monica Valluri, Oleg Y. Gnedin, & Neil Ash (2025), "Improved Particle Spray Algorithm for Modeling Globular Cluster Streams", ApJS 276, 32, arXiv:2408.01496]
- `fardal15spraydf` (index.rst:416): Fardal et al. (2015), https://ui.adsabs.harvard.edu/abs/2015MNRAS.452..301F/abstract  [JSON: "Generation of mock tidal streams", MNRAS 452, 301] for the method and Qian et al. (2022), https://ui.adsabs.harvard.edu/abs/2022MNRAS.511.2339Q/abstract for the galpy implementation
- `ttensor` / `rtide` (index.rst:417): Webb et al. (2019a), https://ui.adsabs.harvard.edu/abs/2019MNRAS.488.5748W/abstract
- `to_amuse` (index.rst:419): Webb et al. (2019b), http://arxiv.org/abs/1910.01646

Other method references named in HISTORY.txt / whatsnew.rst (for the feature list, strings as given there):
- Benettin et al. (1980) — Lyapunov renormalisation; https://ui.adsabs.harvard.edu/abs/1980Mecc...15....9B/abstract (whatsnew.rst:21)
- Rein & Spiegel (2014/2015) — IAS15; https://ui.adsabs.harvard.edu/abs/2015MNRAS.446.1424R/abstract (whatsnew.rst:152; HISTORY.txt:465 says "Rein & Spiegel 2015")
- Hernquist & Ostriker (1992) — SCF (HISTORY.txt:1220); Kuijken & Dubinski (1995) — DiskSCF trick (:1103)
- Dehnen (2000) — DehnenSmoothWrapper (:1091ff); Monari et al. (2016) — 3D bar (:1135); Cox & Gomez (2002), https://arxiv.org/abs/astro-ph/0207635v1 — SpiralArmsPotential (:1120)
- Long & Murali (1992) — SoftenedNeedleBar; Binney & Tremaine (2008) Ch 2.5 — Ferrers; de Zeeuw (1985) — PerfectEllipsoid, http://adsabs.harvard.edu/abs/1985MNRAS.216..273D (whatsnew.rst:607; HISTORY.txt:1002)
- Merritt & Fridman (1996) — TwoPowerTriaxial (:1210); Emsellem et al. (1994) — TriaxialGaussian, https://ui.adsabs.harvard.edu/abs/1994A%26A...285..723E/abstract (whatsnew.rst:386; HISTORY.txt:760)
- Dehnen (1993) — DehnenSpherical, https://ui.adsabs.harvard.edu/abs/1993MNRAS.265..250D (whatsnew.rst:494; HISTORY.txt:833); Widrow (2000) — NFW DF approx (:709ff); Evans et al. (1997) — power-law DFs (:299); Ogata (2005) — double-exponential Bessel quadrature (:797)
- Batsleer & Dejonghe (1994) — KuzminKutuzovStaeckel, http://adsabs.harvard.edu/abs/1994A%26A...287...43B; Smith et al. (2015) — MN3ExponentialDisk, arXiv:1502.00627, http://adsabs.harvard.edu/abs/2015MNRAS.448.2934S
- MW potentials: McMillan (2017); Dehnen & Binney (1998); Irrgang et al. (2013) (:886); Cautun20 (AUTHORS.txt; tutorials index.rst:139)
- Solar parameters: Schoenrich, Binney, & Dehnen (2010), http://adsabs.harvard.edu/abs/2010MNRAS.403.1829S; Bennett & Bovy (2019), http://adsabs.harvard.edu/abs/2019MNRAS.482.1417B (zo = 20.8 pc)
- Known-object catalogues: Pace et al. (2022; 2205.05699); Baumgardt et al. (2019; 1811.01507); Vasiliev & Baumgardt (2021; 2102.09568); Baumgardt & Vasiliev (2021; 2105.09526) (HISTORY.txt:638–644; whatsnew.rst:323–328)
- Variational equations ref in docs: "3D variational equations … Benettin et al. (1980)"; the pre-existing planar machinery dates to 2011 (git log).
