# Writing like Jo Bovy: a working style guide for the paper

Built from the LaTeX sources of Bovy's first-author papers (list at the end).
The primary template is the galpy paper (Bovy 2015, ApJS 216, 29; arXiv:1412.3451,
`ms.in`). Every rule below is backed by a verbatim quote; when the paper is not
named, the quote is from the galpy paper. Numbers in brackets are corpus counts.

---

## 0. The voice in one paragraph

Plain, declarative, unhurried American-English prose that explains *what the code
does and why it is built that way*, in sentences of ~22 words (median) that lean on
`because`, `such that`, `that is,` and `therefore` rather than on adjectives. The
single-author papers use **I** for the author's decisions, claims, and the paper's
own structure, and **we** for the derivation the reader is walked through. Claims
are stated flatly, hedged with `approximately`, `likely`, `typically`, `to within`,
and quantified to two significant figures with units attached by thin-space macros.
Code is shown as REPL transcripts with the printed output included, API names are in
`\texttt{}`, and figures carry long self-contained captions whose first "sentence"
is a noun-phrase title. Nothing is sold: the strongest adjectives in the galpy paper
are "powerful", "exquisite", and "state-of-the-art", each used once or twice, and the
closest thing to hype is "It is clear from Figure 23 that…".

---

## 1. Structure of a methods/software paper

### 1.1 Section order (galpy paper, 29 pp.)

1. `Introduction` (6 paragraphs + outline paragraph + code-availability paragraph)
2. `General galpy overview` → `Package structure`, `Units in galpy`
3. `galpy's Potential framework` → `General framework`, `C implementations…`,
   `Axisymmetric potentials`, `Non-axisymmetric potentials`,
   `Example: Milky-Way-like potentials`
4. `Orbit integration` → `General framework`, `Supported integrators`
5. `Action-angle coordinates` → `Generalities`, three method sections,
   `Example: Adiabatic invariance of the actions`
6. `Distribution functions for disks` → 2D, `Example: Oort functions…`, 3D
7. `The response of a kinematically-warm disk…` (a full science application)
8. `galpy development` → `Source code`, `Automated testing and code coverage`
9. `Future outlook` (no "Conclusions" section; the outlook *is* the close)
10. `\acknowledgements`, `\appendix` (`Coordinate transformations in galpy`)

Pattern: *framework section → how to extend it → concrete science example*, repeated
for each subpackage. Every component gets a `General framework`/`Generalities`
subsection first and an `Example:` subsection last. The engineering (git, tests,
coverage) is a real section with numbers, not a footnote.

### 1.2 The introduction, paragraph by paragraph

The galpy introduction is a fixed sequence; reuse it.

- **P1 — the field and the gap, stated as fact, with the competitor named.**
  > "Galactic dynamics is an old and venerable subject in astrophysics and a
  > mainstay in the Astronomy and Astrophysics graduate-school curriculum. While
  > many good textbooks are devoted to the topic \citep[\eg,][]{binneytremaine},
  > very few software tools designed to aid in galactic-dynamics computations are
  > currently available (with the notable exception of $N$-body codes, …). One of
  > the only such software packages is the NEMO Stellar Dynamics Toolbox
  > \citep{Teuben95a}, which is …"
- **P2 — "In this paper, I introduce …" + the design in one breath + dependencies.**
  > "In this paper, I introduce a new, modern software toolbox for galactic dynamics
  > that is wholly focused on the dynamics and distributions of orbits in external
  > gravitational potentials: \galpy. The \galpy\ package is largely written in the
  > \python\ programming language and combines the flexibility and ease-of-use of a
  > high-level, object-oriented language with the speed of the lower level \cc\
  > language to resolve speed bottlenecks. The basic functionality of \galpy\ only
  > depends on … and is therefore straightforward to install."
- **P3 — the core objects and why they matter now (one forward-looking sentence).**
  > "At its core \galpy\ contains \pot\ objects and \orb\ objects that can represent
  > a variety of galactic potentials and orbits in those potentials. … \galpy's
  > powerful framework will be especially useful in aiding in the interpretation of
  > the exquisite kinematic data from the astrometric Gaia satellite …"
- **P4 — scope/version and where the code lives (URL centered on its own line).**
  > "This paper describes the basic functionality of \galpy\ contained in the
  > \flag{v1.0} release. As with most software, \galpy's development is ongoing and
  > features are being added regularly."
- **P5–6 — the outline paragraph(s), always present, always this opener.**
  > "The outline of this paper is as follows. In \sectionname~\ref{sec:overview}, I
  > give a general overview of the structure and design of \galpy. In
  > \sectionname~\ref{sec:units} I discuss …"
  Variants in other papers: "This paper is outlined as follows." (2014),
  "The structure of this paper is as follows." (2016, 2017). Bovy & Rix 2013 adds
  a reading guide: "We recommend that readers not interested in the technical
  details … read the introduction of \sectionname~\ref{sec:method}, … and then …".
- **P7 — reproducibility paragraph.**
  > "Self-contained code to reproduce all of the plots in this paper … is available
  > in a \git\ repository associated with this particular paper. This repository is
  > separate from the \galpy\ repository and can be found on \github\ as well at …"

The science-paper introductions (2014, 2016) follow the same skeleton with a
different P1–P3: *why the problem matters* → *what has hampered progress* (numbered
"First, … Second, …") → *the natural coordinates/approach* → "In this paper, I
present …" → *how this differs from prior approaches, author by author* → outline.

### 1.3 What the conclusion does

Four moves, in order (2014 streams paper, 2016 halo paper, 2013 Bovy & Rix):

1. Restate what was done in the present perfect: "I have presented a new method for
   modeling the dynamics of tidal streams, making extensive use of action--angle
   variables." / "In this paper we have performed the first proper stream fits to
   observational data…"
2. Restate the key numbers with the figure/equation they come from: "The combined
   constraint from Pal 5 and GD--1 is $c/a= 1.05\pm0.14$." (2016)
3. Say what it is good for next, concretely: "it should be immediately useful for
   the proper analysis of data on the GD-1 \citep{Koposov10a}, Orphan …, Pal 5, and
   other streams."
4. Code availability, URL centered: "All code used in this paper is made publicly
   available at \\ \centerline{\url{http://github.com/jobovy/mwhalo-shape-2016}~.}"

The galpy paper instead closes with `Future outlook`: what is coming, who is working
on it ("Work has been started, primarily by Rok Ro\v{s}kar, on implementing a
\texttt{SnapshotPotential} class…"), and "Further small and large possible extensions
of the \galpy\ codebase are described on \galpy's \github\ page." No summary table,
no bullet list of contributions.

### 1.4 Abstract

First person, concrete, with numbers and the URL; the galpy abstract is one paragraph
of 11 sentences. Opening sentence names the deliverable: "I describe the design,
implementation, and usage of \galpy, a \python\ package for galactic-dynamics
calculations." It ends with hard counts: "Overall, \galpy\ consists of about 54,000
lines, including 23,000 lines of code in the module, 11,000 lines of test code, and
about 20,000 lines of documentation. The test suite covers $99.6\,\%$ of the code."

---

## 2. Paragraph architecture

**Length.** Median 3–4 sentences, ~85 words [galpy: median 4 sentences/84 words;
p90 = 6 sentences]. Short 2-sentence paragraphs are common and fine; 8+ sentences
is rare outside the response-section methodology.

**Topic sentence = a flat statement of what the thing is or what this section does.**
> "At the heart of \galpy\ are \pot\ objects. These come in three flavors: one
> dimensional, two dimensional, and three dimensional."

> "The integration and characterization of orbits is an essential part of \galpy."

> "Three classes of DFs for axisymmetric disks are currently included in \galpy."

**Development = the mechanism, then the consequence, then the caveat.** The caveat
very often ends the paragraph; there is no wrap-up sentence.
> "This is implemented simply by interpolating separately the potential itself, the
> forces and second derivatives, and the circular velocity curve, … That is, no
> effort is made to calculate the forces using derivatives of the splines
> interpolating the potentials, etc. Therefore care should be taken to build a dense
> enough interpolation grid to avoid large inconsistencies between the potential
> and its forces and derivatives."

**Transitions** are made with ordinary connectives at the start of the sentence, not
with signposting phrases. Sentence-initial counts over five papers: `However` [59],
`As` [50], `Because` [39], `Therefore` [34], `Thus` [33], `While` [31],
`For example` [15], `In particular` [10], `That is` [8], `Finally` [3], `Overall` [3].
Paragraph-initial words: `The` [116], `We` [47], `To` [29], `In` [28], `As` [13],
`Finally` [9], `Because` [9]. "Finally," opens the last of a run of examples:
> "Finally, note that \galpy\ contains an older version of a similar potential:
> \texttt{MWPotential}. This model was \emph{not} fit to data on the Milky Way
> (although it is close to a good model) and is now superseded by
> \texttt{MWPotential2014}."

**Forward/backward references** replace transitional prose: "(see below)", "described
further in \sectionname~\ref{sec:aa}", "As mentioned before,", "As described above,",
"As discussed in \sectionname~…".

**Hedging** is moderate and specific. Preferred hedges: `approximately`/`$\approx$`,
`typically`, `likely`, `to within`, `quite`, `somewhat`, `relatively`, `in principle`
vs `in practice`. `may` is rare [22 in five papers], `might` [3], `perhaps` [3],
`arguably` [0]. When something is uncertain he says what would settle it:
> "While this model oversimplifies the real stripping process, I hypothesize that at
> least for constraining the gravitational potential with stream data, this simple
> model is likely unbiased." (2014)

Strong claims from figures use "It is clear that/from" [32]:
> "It is clear from \figurename~\ref{fig:axi_oort} that both $A$ and $B$ depend
> linearly on the asymmetric drift and that $A$ is much more affected by the
> kinematic temperature of the population than $B$."

Emphasis is "we stress that" [5] or an `\emph{}` on one word ("\emph{not} fit",
"\emph{any} \pot\ instance", "\emph{natural units}"); never "importantly".

---

## 3. Sentence-level habits

### 3.1 I vs. we (the galpy paper is single-author and uses BOTH)

Counts in the galpy paper: I/my [29], we/our [81]. In the 2014 streams paper (also
single-author): I [55], we [85]. In the short 2016 Gaia letter: I [23], we [3].

Rule actually followed:
- **I** = authorial actions, decisions, and claims: "I describe", "I introduce",
  "I present", "I give an example of a new \pot\ class", "I compute the Oort
  functions", "I compare", "I investigate", "I do not attempt as detailed a
  characterization", "I briefly discuss", "I test energy conservation", "Here I have
  explicitly demonstrated that…", "I will employ the terminology of \citet{Binney12a}".
- **we** = the reader and author doing a derivation or a worked example together,
  and the inclusive "we find/expect/can understand": "we note that \orb\ instances
  can be initialized as", "We build a regular grid in $L_z$", "We then employ
  third-degree bivariate spline interpolation", "We can understand all of this
  behavior by calculating…", "we expect a linear relation", "We will focus the
  discussion in this section on…".
- Both in one paragraph is normal:
  > "To interpolate the St\"{a}ckel method, we roughly follow the procedure described
  > in \sectionname~2.2 of \citet{Binney12a} with some tweaks for efficiency. Because
  > it is beyond the scope of this paper to fully describe the St\"{a}ckel method, I
  > will employ the terminology of \citet{Binney12a}; the reader should refer to that
  > paper for full details. We build a regular three-dimensional grid, …"
- Multi-author papers are "we" throughout (Bovy & Rix: I [1], we [397]).
- "the author" appears once: "These simple relations are to the author's knowledge
  not present in the literature."

### 3.2 Length and shape

Median 22 words, mean 24, 90th percentile 39 (galpy); science papers run longer
(median 25–27). Long sentences are built by subordination (`because`, `such that`,
`while`, `which`, `where`), not by stacking clauses with commas. Typical shape:
*claim, because mechanism, such that consequence.*
> "Corrections are stored, because the iterative procedure is quite slow."

> "The stability of the \texttt{actionAngleIsochroneApprox} algorithm makes it ideal
> for investigating the behavior of the angles over many orbital periods, such as for
> tidal streams \citep{Bovy14a}."

Short sentences are used for verdicts: "The latter in particular performs poorly."
"This is a not a serious limitation." "There is no private development version."

### 3.3 Punctuation

- **Semicolons** [60 in galpy, ~1 per 10 sentences]: joining two tightly coupled
  statements, and separating list items that contain commas.
  > "the latter uses Gaussian quadrature and is much faster."
  > "…; these two are also implemented in \cc, while the more general version
  > currently is not."
- **Parentheses** are the main aside device: "(see \sectionname~\ref{sec:units})",
  "(\eg, \texttt{vcirc})", "(that is, …)", "(although it is close to a good model)",
  "(note the different range on the $y$ axis for \texttt{odeint})".
- **Em-dashes** `---` without spaces, ~1.4 per 1000 words, for a mid-sentence
  definition or interjection, not for rhythm:
  > "$J_R$ only depends on $L_z$ and $E_R$---the radial energy, see the caption of
  > \figurename~\ref{fig:orbmethods}---in the adiabatic method."
- **Colons** introduce a named thing or a list: "…galactic dynamics…: \galpy." /
  "has the following methods: (a) …, (b) …, and (c) …".
- **Footnotes** [4 in galpy; 1–8 per paper]: technical asides, caveats, URLs, and
  "to the author's knowledge" remarks. The footnote mark goes *before* the sentence's
  final period: `the iterative procedure is quite slow\footnote{Note that this is
  partially because …}.` Openers: "Note that…", "Basically, …", "Because…", "For
  simplicity, …", "Available at …".
- **Inline enumerations** "(a) …, (b) …, and (c) …" are used constantly, including
  inside sentences ("using (a) that … and (b) that …").
- Quotes: LaTeX ``double'' for scare quotes and coined terms ("fudge" factor, "pull
  requests", "bow tie" structure); `single' for labels and informal terms
  (`vertical' coordinate, `Oort constants' as a misnomer, `+' or `-').

### 3.4 Spelling and hyphenation (American)

`center`, `modeling`/`modeled`, `behavior`, `color`, `neighborhood`, `analyze`,
`parameterize(d)` (once `parametrized`), `toward(s)` both. Never `centre`,
`modelling`, `behaviour`, `whilst`.

Compound modifiers are hyphenated aggressively: `kinematically-warm disk`,
`close-to-circular orbits`, `two-dimensional`, `three-dimensional`, `Milky-Way-like`,
`dark-matter halo` (2016 paper: `dark--matter`), `mid-plane`, `line-of-sight
velocity`, `surface-density profile`, `radial-velocity-dispersion profile`,
`orbit-integration-based method`, `grid-based-interpolation version`,
`phase--space-volume-conserving integrators`. Coordinate *pairs* get an en-dash:
`action--angle`, `frequency--angle`, `position--velocity`, `phase--space` (also
`phase-space`; both occur), `Runge--Kutta`, `Dormand--Prince`, `color--magnitude`.
Capitalisation: `Galactic`/`Galactocentric` for the Milky Way, `galactic dynamics`
generic; `the Sun`, `the Galactic center`; `$N$-body`; `St\"{a}ckel`, `Poincar\'{e}`.

### 3.5 Equations

- Few in a software paper (14 numbered in galpy) and only where the definition is
  the point (the Shu, Dehnen, and quasi-isothermal DFs; the Oort/Jeans derivation).
- Lead-in is a clause ending in `as`, `is given by`, `has the following form`,
  `can be written as`, `defined as`, `This gives`; a colon is rare (`in
  \texttt{DehnenBarPotential}:` once). The equation is a grammatical part of the
  sentence: it ends in `\,,` or `\,.` (thin space before the punctuation), and the
  next line starts with `where` [40 of ~110 equations], `Here,`, `and`, or `This`.
  > "The classical Shu DF is given by
  > \begin{equation}\label{eq:fshu}
  >   f_{\mathrm{Shu}}(E,L_z) = … \,,
  > \end{equation}
  > where $\Omega$ is the rotational frequency, $\kappa$ is the epicycle frequency, …"
- Referenced in running text as `\equationname~(\ref{eq:oortAapprox})` → "equation
  (7)" mid-sentence, `\Eqnname~(\ref{…})` → "Equation (7)" at sentence start,
  `\equationname s~(\ref{a})--(\ref{b})` for ranges; later papers use `\eqref`.
  Equations in cited papers: "\equationname~(9) in \citet{Sanders12a}".
- `\begin{align}` for systems; `\begin{split}` inside `equation` for long lines;
  `\dd` (upright d), `\vec{x}` → `\mathbf{x}`, `\mathrm{}` for word subscripts
  (`R_{\mathrm{max}}`, `f_{\mathrm{na}}`, `\Sigma_{\mathrm{out}}`).

### 3.6 Numbers and units

- Units via macros carrying a leading thin space: `$8\kpc$`, `$220\kms$`,
  `$1.1\kpc$`, `$\mas\yr\inv$`, `$10^{-13}\,\mathrm{km\,s}^{-2}$`,
  `$M_\odot\pc^{-2}$`. Percentages `$99.6\,\%$`, `$\lesssim 5\,\%$`; in prose "a few
  percent", "less than one percent", "80\,percent".
- Uncertainties `$2.15\pm0.14\kpc$`, `$0.95\pm0.04$`, `$68\pm4\,M_\odot\pc^{-2}$`;
  asymmetric `$1.3^{+0.5}_{-0.3}$`; systematic `$\pm1.5\,(\mathrm{syst.})$`.
  Two significant figures on the uncertainty, central value matched to it.
- Approximation always in math: `$\approx 5\times10^{-5}$`, `$\delta \approx 0.3$ to
  $0.6$`, `$\lesssim10^{-2}$`; in words "about 54,000 lines", "about 20 and 5 times
  better". Precision statements: "conserved to 1 part in $10^{-8}$", "to better than
  1 part in $10^{-3}$", "to a few percent".
- Thousands separators in counts: "23,000 lines", "304,267 main-sequence stars",
  "14,166 of 14,220 relevant lines". Big-number factors "$4\times10^6\,M_{\odot}$".
- Natural-unit values are quoted raw and the unit system is pointed at: "(in
  \emph{natural coordinates})", "All of the outputs are in natural units (see
  \sectionname~\ref{sec:units})".

---

## 4. How software is described

### 4.1 Level of detail

Describe the *contract* (what a method takes, returns, assumes) and the *reason for
the design*, not the implementation line by line. Limitations are stated in the same
breath as features, flatly, often with "currently":

> "The philosophy behind this implementation is that it allows users to implement
> additional potentials with a minimum of effort, only requiring the basic
> properties of a potential to be implemented."

> "While the forces and second derivatives could be obtained by taking numerical
> derivatives of the potential, such that a new potential could be defined using
> only the \texttt{\_evaluate} function, this is not currently supported."

> "\galpy\ currently does not contain any methods for calculating positions and
> velocities for a given set of action--angle coordinates in a given potential."

> "Note that whether or not the potential is actually spherical is not explicitly
> checked by the code, so care must be taken not to use it with axisymmetric
> potentials."

Performance is quantified by *what* was measured, and "much faster" is allowed only
with the reason: "the latter uses Gaussian quadrature and is much faster". Code-size
and test statistics are given as plain counts ("500 test functions consisting of
about 11,000 lines of code … about 20,000 assertions"; "coverage of the code is
currently 99.6\,\%, that is, 14,166 of 14,220 relevant lines are touched"). What is
*not* tested is said too: "The plotting routines in \texttt{bovy\_plot} are not
tested, because it is hard to meaningfully test plotting routines."

The engineering story is told in a dedicated `galpy development` section with
subsections `Source code` and `Automated testing and code coverage`, written in the
same voice: "There is no private development version." "Users are encouraged to
report bugs through this issue tracker." "Much effort has been put into creating a
useful and meaningful test suite…"

### 4.2 How code is shown

Two forms, both `minted` (`\usepackage{minted}`, compiled with `-shell-escape`;
the arXiv submission is a wrapper `article` that `\includepdf`s the compiled PDF and
ships `ms.in` as the real source).

**(a) Inline snippets, 1–6 lines, unnumbered, no frame**, introduced by an ordinary
clause ending in `as`/`by running`/`looks like` (no colon), and followed by a
sentence that says what the snippet did:
```latex
which can be imported and used as
\begin{minted}{python}
  from galpy.potential import NFWPotential
  np= NFWPotential(normalize=1.)
\end{minted}
This initialization is explained further below. \pot\ objects form the
basis of almost all of the \galpy\ functionality.
```
```latex
An example \orb\ instantiation from observed coordinates looks like
\begin{minted}[fontsize=\small]{python}
  from galpy.orbit import Orbit
  o= Orbit([25.,10.,2.,5.,-2.,50.],radec=True,ro=8.,
           vo=220.,solarmotion=[-11.1,25.,7.25])
\end{minted}
where the solar motion is such that the Sun's rotational velocity with
respect to Sgr A$^*$ is $245\kms$ \citep{BovyVc}.
```

**(b) Code listings as numbered Figures** with captions (there is no "Listing"
environment; they are `\figurename`s and are referenced as such):
```latex
\begin{figure}[t!]
\begin{minted}[mathescape,
               linenos,
               numbersep=5pt,
               gobble=2,
               frame=lines,
               framesep=2mm]{python}
  from galpy.orbit import Orbit
  from galpy.potential import MWPotential2014
  o= Orbit([0.8,0.3,0.75,0.,0.2,0.]) # setup R,vR,vT,z,vz,phi
  times= numpy.linspace(0.,10.,1001) # Output times
  o.integrate(times,MWPotential2014) # Integrate
  o.E() # Energy
  -1.2547650648697966
  o.rperi(), o.rap(), o.zmax() # Peri-/apocenter r, max. |z|
  (0.44231993168097, 0.87769030382105, 0.077452357289016)
\end{minted}
\caption{Methods of \orb\ instances, illustrated using an orbit
  similar to the one at the right in \figurename~\ref{fig:orbits}. …
  All outputs are in natural units (see \sectionname~\ref{sec:units}), …
  \orb\ instances have many more methods similar to \texttt{o.R()},
  \texttt{o.ra()}, and \texttt{o.jr()}.}\label{fig:orbmethods}
\end{figure}
```
Conventions visible in every listing:
- REPL style: the call on one line, the *actual printed output* on the next, full
  float precision, arrays as `array([ 0.00980542])`. "Figure 2" is literally a
  transcript of imports. Outputs are tested: "tests are run that all of the code
  examples given in this paper produce the output given here".
- Every line that is not obvious gets a short trailing `# comment` in telegraphic
  English: `# Rad. and vert. E at end`, `# Quasi-iso df w/ hr=1/3, hsr/z=1,
  sr(1)=0.2, sz(1)=0.1`, `# check that these have the right mean radius = 2hr=2/3`.
- Spacing idiom `name= value` (space after `=`, none before) everywhere.
- Sanity checks are written as `assert numpy.fabs(numpy.mean(rs)-2./3.) < 0.1`
  inside the example itself.
- A listing may define a whole class (`TimeInterpPotential`, Figure 4) that later
  figures reuse ("using \texttt{TimeInterpPotential} defined in
  \figurename~\ref{fig:newpotclass}"), or combine plot + code in one figure
  ("the code used to generate them is shown at the bottom").
- A listing can span languages (`{c}` then `{python}`, with `\centerline{\cc\ code:}`
  headers and `firstnumber=61` to continue numbering) when showing the glue between
  layers; the caption then refers to specific line numbers ("declared in the general
  \texttt{galpy\_potentials.h} header file on line 42").

### 4.3 Typesetting of names

- API names, modules, classes, methods, keywords, files, and attributes: `\texttt{}`
  with escaped underscores — `\texttt{galpy.util.bovy\_conversion}`,
  `\texttt{integrate\_dxdv}`, `\texttt{hasC}`, `\texttt{normalize=1.}`,
  `\texttt{ro=}` (keywords keep the trailing `=`), `\texttt{\_\_init\_\_}`.
  [340 `\texttt` in the galpy paper.]
- Macros for the recurring ones: `\newcommand{\galpy}{\flag{galpy}}` with
  `\flag[1]{\texttt{\lowercase{#1}}}`; `\python`, `\numpy`, `\scipy`, `\matplotlib`;
  `\cc` = `\texttt{C}`; `\pot` = `\texttt{Potential}`, `\orb` = `\texttt{Orbit}`;
  `\git`, `\github`, `\travis`, `\sphinx`. `\galpy's` is written as `\galpy's`.
  Section titles containing the macro use `\lowercase{\galpy's}`.
- Plurals of code names are formed in roman type: `\texttt{PowerSphericalPotential}s`,
  `\texttt{linearOrbit}s`, `\texttt{diskdf}s`.
- Prose nouns, not code font, for concepts: "instance methods", "member functions",
  "subclasses", "keyword", "glue", "flavors", "the codebase", "routines", "the code".
- Program/service names in `\texttt{}` (Travis CI, Coveralls, Read the Docs) the
  first time, roman afterwards in the acknowledgements.

---

## 5. Figures and captions

- **Captions are long and self-contained**: galpy median 92 words (12–322); science
  papers 100–130. A reader should get parameters, what each curve/panel/color is,
  and the takeaway from the caption alone.
- **First sentence is a noun-phrase title ending in a period, no finite verb**:
  "Rotation curves of some of the axisymmetric potentials included in \galpy."
  "Adiabatic invariance of the actions." "Fractional energy error as a function of
  the length of the integration for the orbit on the right in
  \figurename~\ref{fig:orbits} in \texttt{MWPotential2014} for the eight orbit
  integrators contained in \galpy." "Methods of \pot\ instances in \galpy,
  illustrated using a double-exponential disk potential with …". Variant: "Same as
  \figurename~\ref{fig:barresponse_ad}, but for a bar that is grown over only two
  bar periods…" / "Like \figurename~\ref{fig:pal5post}, but for the GD--1 analysis."
- **Then full declarative sentences**: what is plotted and how ("The top panel shows
  …", "The bottom panel displays …", "Different curves demonstrate the dependence of
  the response on $\sigma_R$."), definitions needed to read the axes ("The error is
  computed as $\Delta\theta = \theta(t) - \theta(t=0)-\Omega\,t$"), then the
  interpretive sentence: "The energy error remains constant for the symplectic
  integrators while it increases in time for the other integrators." "Remarkably,
  $K$ is independent of $\sigma_R$." "This figure can be used to establish an
  approximate focal length for using the St\"{a}ckel approximation for
  \texttt{MWPotential2014}."
- **Provenance** goes in the caption ("This figure was made by calling each \pot\
  instance's \texttt{plotRotcurve} method.") and in a LaTeX comment after it
  (`%python plot_energy_conservation.py ../tex/energy.ps`), with timings kept as
  comments.
- **Text references**: `\figurename~\ref{fig:x}` → "Figure 7" (never "Fig."), as
  the sentence subject with a verb from {shows, displays, demonstrates, illustrates,
  compares}: "\figurename~\ref{fig:orbits} demonstrates orbit integration in \galpy:
  the two orbits displayed in the top panel of this figure…". Plural via the
  `\figurename s~\ref{a} and \ref{b}` hack. Text and caption *both* carry the
  interpretation; the text adds the "why" and the comparison to literature ("This
  figure is similar to figure 3.4 in \citet{binneytremaine}."; "(note that the
  captions of their figures 6 and 7 appear to be switched)").
- Widths `0.48\textwidth` single column, `\textwidth` or `\textheight`-based for
  `figure*`; `[t!]` placement; stacked panels with `\\`.

---

## 6. Citations

- natbib: `\citet` for authors as sentence subjects ("the St\"{a}ckel approximation
  of \citet{Binney12a}"; "This was first done by \citet{Kuijken94a}"), `\citep` for
  support, `\citealt` inside parentheses: "(NFW; \citealt{Navarro97a})", "(see the
  appendix of \citealt{Bovy14a} for full details)", "(see also \citealt{Olling03a}
  who reach a similar conclusion)". galpy paper: `\citet` [64] > `\citep` [30] >
  `\citealt` [10]; science papers are closer to 1:1.
- `\citep[\eg,][]{binneytremaine}` is the house form for "e.g." citations [17 `\eg`
  in galpy]; `\eg`/`\ie` are macros expanding to "e.g."/"i.e." and are *followed by a
  comma* in prose: "\eg, for tracing the orbits of stars or pulsars kicked out from
  the Galactic center \citep[\eg,][]{Dexter13a}".
- One to three citations per claim; long lists only for "simulations predict"
  statements in introductions ("\citep[\eg,][]{Frenk88a,Dubinski91a,Warren92a,
  Cole96a,Jing02a,VeraCiro11a}") or to list stream-modelling methods.
- Software is cited the first time it is named: "\numpy\ \citep{Oliphant06a}, \scipy\
  \citep{Jones01a}, and \matplotlib\ \citep{Hunter07a}"; integrators are cited to
  the original algorithm paper ("a fifth-order Dormand--Prince 5(4) method
  \texttt{dopr54\_c} \citep{Dormand80a}"; "the SI6A sixth-order symplectic integrator
  from \citet{Kinoshita91a}").
- Self-citations are matter-of-fact and frequent ("as described in \sectionname~5 of
  \citet{Bovy13a}"); a shorthand is defined when reused: "\citet{Bovy14a} (hereafter
  B14)"; "we use \bo\ to collectively refer to the set of … papers".
- Pointing into a cited paper is precise: "\equationname~(9) in \citet{Sanders12a}",
  "\sectionname~3.2 of \citet{Dehnen99a}", "figure 3.4 in \citet{binneytremaine}".
- Deferral formula: "We refer the reader to \citet{Dehnen99a} for a detailed
  discussion of…" / "the reader should refer to that paper for full details."

---

## 7. Vocabulary

**Verbs for showing results**: demonstrate [35], display [30], show, illustrate [18],
present, compute/calculate, determine, find, obtain, evaluate, investigate, employ,
make use of, perform (a fit). "displays"/"demonstrates" are the Bovy fingerprints
for figures. Code "contains", "consists of", "is implemented as", "is available as",
"supports", "allows one to", "is provided", "is included".

**Connectives and fillers he actually uses** (five-paper counts): because [178],
therefore [94], thus [71], however [69], such that [48], "it is clear" [32],
"that is," [27], overall [27], essentially [25], "in particular" [22],
straightforward [21], "for example" [20], "we expect" [18], "in detail" [17],
"see below" [15], simply [14], "note that" [12], "in what follows" [12],
"as expected" [7], "we stress" [5].

**Characteristic phrases** (quote-ready):
- "It is a pleasure to thank …" (every acknowledgements)
- "The outline of this paper is as follows."
- "In what follows, I will …" / "As usual, we work in natural units"
- "That is, …" to restate in plainer terms
- "to within these errors", "to 1 part in $10^{-8}$", "to a few percent"
- "is beyond the scope of this paper", "we do not repeat it here"
- "to the author's knowledge not present in the literature"
- "it is important to keep in mind that this model is merely provided for convenience"
- "care should be taken", "care must be taken not to"
- "This turns out to be …", "It turns out that …"
- "Remarkably," (sparingly: twice in galpy), "Surprisingly, we find" (once, 2017)
- "innocuous", "venerable", "mainstay", "wholly focused", "weighs in at 283 pages",
  "superseded", "fudge factor", "glue", "flavors", "(if you will)", "for better or
  worse", "Suffice it to say", "But it's fun and illustrative to include it." (the
  last three from the 2020 PRL-style letter — informality is allowed in small doses)

**Words he does NOT use** (zero or near-zero in ~130k words of first-author text):
delve, leverage, utilize/utilise, facilitate, seamless, showcase, underscore,
highlight (verb), enable [0], ensure [1], harness, unlock, landscape, paradigm [1],
pivotal, vital, holistic, nuanced, intricate, myriad, plethora, "cutting-edge",
comprehensive, "game-changing", "state-of-the-art" [1, in the galpy abstract],
moreover [3], furthermore [10 across 13 papers], notably [1], importantly [4],
interestingly [1]. Words he uses but *only* in a technical sense: "robust" [20] as in
"robustly measure the total surface density" or "$A$ is robust to all changes to the
DF except for $h_\sigma$" — never as a decorative adjective for code; "novel" [14]
only for the action-angle method and always with the noun it modifies; "crucial" only
as "crucially depends on"; "exquisite" only for Gaia data/stream sensitivity.

---

## 8. Negative list: AI-isms and the Bovy alternative

| Avoid | Why / what Bovy does instead |
|---|---|
| "delve into", "explore in depth" | "I describe … in detail", "discussed in some detail in the Appendix" |
| "leverage", "harness", "utilize" | "use", "make use of", "employ", "take advantage of" ("Taking advantage of the essentially one-dimensional nature of tidal streams, …") |
| "robust" as praise ("a robust framework") | Say what it withstands: "robust to all changes to the DF except for $h_\sigma$"; otherwise "reliable", or give the test that shows it |
| "seamless(ly)", "effortlessly" | "can be used in exactly the same way as", "automatically", "without any need for the user to do anything" |
| "game-changing", "revolutionary", "transformative" | State the consequence: "This clears the way for proper probabilistic inference of…". (He wrote "a revolutionary moment" exactly once, for Gaia's acceleration measurement.) |
| "crucial", "critical", "key", "vital", "essential" (as hype) | "important", "an integral part of \galpy", "an essential part of \galpy" (once each); or drop the adjective and let the number talk |
| "It is worth noting that", "It should be noted that" | "Note that …" or just say it; "we stress that" when it is genuinely a warning |
| "Importantly,", "Notably,", "Interestingly," | Delete. If needed: "In particular,", "Remarkably," (rare), "Surprisingly, we find" (rare) |
| "In this paper we present a novel …" | "In this paper, I introduce a new, modern software toolbox for galactic dynamics that is wholly focused on …" — "new" plus the actual property |
| Triadic adjective lists ("fast, flexible, and powerful") | He uses triads of *nouns/facts* ("simple, intuitive, and easily extensible" is his one adjective triad; otherwise "(a) …, (b) …, and (c) …" with content) |
| Em-dash-driven rhythm, dashes for drama | `---` only for an inline definition or interjection, ~1 per 700 words; parentheses are the default aside |
| Every paragraph ending on a summary/"takeaway" sentence | Paragraphs end on the caveat, the limitation, or a forward reference ("…and is planned for a future version of the code.") |
| Bulleted prose, bold run-in headers | Running prose with inline "(a) … (b) …"; `enumerate`/`itemize` only for a list of data constraints or model ingredients, each item a full sentence |
| "comprehensive", "holistic", "end-to-end" | "general", "a large number of", "all of the", with the count ("about 23,000 lines of code") |
| "enable(s)", "allow(s) for" overuse | "allows one to", "can be used to", "makes it possible to", "supports"; he never writes "enable" |
| "significantly" as an intensifier | Only with a measured quantity ("changes … below $\lesssim0.2\kms\kpc\inv$"); otherwise "much", "strongly", "substantially" |
| "state-of-the-art" | Name the method and cite it: "the St\"{a}ckel approximation of \citet{Binney12a}" |
| "ensures correctness" | "The test suite … automatically checks … that the forces are the negative derivative of the potential" |
| "seamlessly integrates with JAX" etc. | "can be used wherever \pot\ instances can be used anywhere in \galpy. This includes functions that use \cc\ to speed up computations" |
| Rhetorical questions, "Why does this matter?" | Only as a section title in a discussion ("At what progenitor mass does the framework break down?", 2014) |
| "We believe", "We feel" | Used once in 13 papers ("We believe that this paper presents the first dynamical measurement…"). Prefer "I hypothesize that", "it is likely that", "we expect" |
| "Overall, this demonstrates the power of…" | "Overall, \galpy\ consists of about 54,000 lines" — "Overall" introduces a total, not a moral |
| "cutting-edge", "modern" as hype | "modern" appears once, qualified: "a new, modern software toolbox … that is wholly focused on…" |
| Passive-voice evasions ("it was decided") | "I" for decisions: "I do not attempt as detailed a characterization as in the previous section" |
| "the user is able to" | "the user needs to", "one can", "can be specified using" |
| British spellings | American throughout |

---

## 9. Acknowledgements, software, facilities, data statements

- There is **no `\software{}`, `\facilities{}`, or data-availability macro** in any of
  the papers (emulateapj/mn2e/revtex era). Software is credited (i) by `\citep` at
  first mention in the text and (ii) again in the acknowledgements:
  > "This work was further made possibly by the open-source codes \numpy\
  > \citep{Oliphant06a}, \scipy\ \citep{Jones01a}, \matplotlib\ \citep{Hunter07a},
  > and the GNU Scientific Library \citep{Galassi09a}. Version-control,
  > continuous-integration, test-statistics, and documentation services are provided
  > by GitHub, Travis CI, Coveralls, and Read the Docs and are gratefully
  > acknowledged."
  > "The MCMC analyses in this work were run using \emph{emcee} \citep{Foreman13a}."
- Acknowledgements open with people, in this order: direct code contributors, bug
  reporters, discussants, referee:
  > "\acknowledgements It is a pleasure to thank Mark Fardal and Denis Erkal for some
  > direct contributions to the code, Eduardo Balbinot, …, and Wilma Trick for bug
  > reports, and the anonymous referee for helpful comments. J.B. was supported by
  > NASA through Hubble Fellowship grant HST-HF-51285.01 …"
  Later papers: "JB received support from the Natural Sciences and Engineering
  Research Council of Canada. JB also received partial support from an Alfred P.
  Sloan Fellowship and from the Simons Foundation." / "JB acknowledges financial
  support from NSERC (funding reference number RGPIN-2020-04712) and an Ontario Early
  Researcher Award (ER16-12-061)." Initials without periods in MNRAS-era papers (JB),
  with periods in ApJ-era (J.B.).
- Survey boilerplate (Gaia/DPAC, SDSS) is pasted verbatim as separate paragraphs
  after the personal acknowledgements.
- Code/data availability is **a sentence in the conclusion (or last section), with
  the URL centered on its own line and a tilde before the period**:
  > "All code used in this paper is made publicly available at
  > \\ \centerline{\url{http://github.com/jobovy/mwhalo-shape-2016}~.}"
  > "All of the code used to perform the analysis presented in this paper is
  > available at \\ \centerline{\url{https://github.com/jobovy/tgas-completeness}~,}
  > and can serve as an example of how to use the selection function."
  In the galpy paper the repository and docs URLs appear in the abstract, the
  introduction (twice, centered), and `Source code`; the figure-reproduction
  repository is announced in the introduction.
- Versions are pinned when it matters: "This paper describes the basic functionality
  of \galpy\ contained in the \flag{v1.0} release." Footnote form for an external
  table: "Specifically, we use version \texttt{2016.08.21} downloaded from \url{…}~."
- License is stated flatly: "\galpy\ is released under the three-clause BSD license."

---

## 10. Fifteen-item checklist for a draft

1. Does the introduction run: field/gap with the competitor named → "In this paper,
   I introduce/present …" → core design in one paragraph → version + URL → "The
   outline of this paper is as follows." → reproducibility paragraph?
2. Is every pronoun right: **I** for decisions/claims/structure, **we** for
   derivations and worked examples the reader follows? (Single author: both occur.)
3. Does each section go *general framework → how to extend/limitations → Example:*?
4. Is every limitation stated in the same paragraph as the feature, with "currently"
   or "is not currently supported", and no apology?
5. Are all claims about accuracy or speed quantified ("to 1 part in $10^{-8}$",
   "$\approx 5$ times better", "much faster" + the reason)?
6. Are all numbers formatted as `$X\pm Y\unit$`, `$\approx$`, `\,\%`, two significant
   figures, thousands separators, units via thin-space macros?
7. Is every code example a transcript with the real printed output, `# comments`,
   `name= value` spacing, and (if long) a numbered Figure with a caption?
8. Is every API name in `\texttt{}` (underscores escaped), with macros for the
   recurring ones, and keywords written with the trailing `=`?
9. Does every caption begin with a noun-phrase title ending in a period, then say
   what each panel/curve is, then give the takeaway, and is it readable on its own?
10. Is the figure referenced in the text as the subject of a sentence
    ("\figurename~\ref{…} shows/displays/demonstrates …") and interpreted there too?
11. Are citations `\citet` for subjects, `\citep` for support, `\citealt` inside
    parentheses, `\citep[\eg,][]{…}` for examples, with 1–3 per claim and precise
    pointers ("\equationname~(9) in \citet{…}")?
12. Grep for the banned list: delve, leverage, utilize, seamless, robust (non-
    technical), crucial, importantly, notably, novel (non-technical), ensure, enable,
    comprehensive, "it is worth noting", "state-of-the-art". Replace per §8.
13. Count em-dashes (target ≲ 1.5 per 1000 words) and check no paragraph ends on a
    summary sentence that merely restates the paragraph.
14. Is spelling American, are compound modifiers hyphenated, coordinate pairs
    en-dashed (`action--angle`), footnote marks placed before the period?
15. Does the paper close with: present-perfect recap, the key numbers, what it is
    good for next, and "All code used in this paper is made publicly available at
    \\ \centerline{\url{…}~.}"? Do the acknowledgements open "It is a pleasure to
    thank …" and credit the open-source codes with citations?

---

## 11. LaTeX conventions found in the sources

- **Classes**: `\documentclass{emulateapj}` for the ApJ/ApJS papers (galpy, 2013,
  2014; 2016 with `[numberedappendix]`); `\documentclass[useAMS,usenatbib,letterpaper]
  {mn2e}` for the MNRAS papers (2016 letter, 2017); `revtex4-1` for PRL; the
  commented-out alternative `%\documentclass[12pt,preprint]{aastex}` heads every
  ApJ file. No aastex6 `\software`/`\facilities`.
- **Packages (galpy)**: `amssymb,amsmath`, `verbatim`, `color`,
  `\PassOptionsToPackage{hyphens}{url}\usepackage{hyperref}`, `breakurl`, `minted`,
  `needspace`, `enumitem`. hyperref: `colorlinks=true` with all link colours
  `\definecolor{linkcolor}{rgb}{0,0,0.25}` ("% hypertex insanity").
- **Layout tweaks**: `\setlength{\emergencystretch}{5em}%No overflowing references`,
  `\allowdisplaybreaks`, `\raggedbottom`, `\needspace{8ex}` before subsections that
  must not strand, `\submitted{}`.
- **Naming macros** (identical across 11 files):
  `\newcommand{\sectionname}{$\mathsection$}` (sections are referred to as "§ 3"),
  `\renewcommand{\figurename}{Figure}`, `\renewcommand{\tablename}{Table}`,
  `\newcommand{\figurenames}{\figurename~s}` (plural hack, also typed inline as
  `\figurename s~\ref{a} and \ref{b}`), `\newcommand{\eqnname}{equation}`,
  `\newcommand{\Eqnname}{Equation}`, `\newcommand{\equationname}{\eqnname}`,
  `\newcommand{\appendixname}{Appendix}`.
- **Text macros**: `\newcommand{\ie}{i.e.}`, `\newcommand{\eg}{e.g.}`,
  `\newcommand{\etal}{et al.}`, `\newcommand{\dd}{\mathrm{d}}`,
  `\newcommand{\normal}{\ensuremath{\mathcal{N}}}`,
  `\renewcommand{\vec}[1]{\ensuremath{\mathbf{#1}}}`,
  `\newcommand{\unitvec}[1]{\ensuremath{\mathbf{\hat{#1}}}}`, `\vecx`, `\vecv`,
  `\vecj`, `\veco`, `\veca` (= `\boldsymbol\theta`), `\feh`, `\afe`, `\Ro`.
- **Unit macros, all with a leading thin space and `\ensuremath`**:
  `\newcommand{\kpc}{\ensuremath{\,\mathrm{kpc}}}`, `\pc`, `\kms` (=
  `\,\mathrm{km\ s}^{-1}`), `\msun` (= `\,\mathrm{M}_{\odot}`), `\Gyr`, `\Myr`,
  `\yr`, `\mas`, `\muas`, `\dex`, `\inv` (= `^{-1}`), `\magunit`. Usage:
  `$1.1\kpc$`, `$\mas\yr\inv$`, `$\kms\kpc\inv$`.
- **Code-name macros**: `\newcommand{\flag}[1]{\texttt{\lowercase{#1}}}`,
  `\newcommand{\galpy}{\flag{galpy}}`, `\numpy`, `\scipy`, `\matplotlib`,
  `\python` (= `\flag{Python}`), `\cc` (= `\texttt{C}`), `\pot`, `\orb`, `\git`,
  `\github`, `\travis`, `\nose`, `\sphinx`, `\readthedocs`.
- **Equations**: `\label{eq:shortname}`; referenced as `\equationname~(\ref{eq:x})`
  / `\Eqnname~(\ref{eq:x})` / `\equationname s~(\ref{a})--(\ref{b})` in 2013–2015,
  `\eqref{eq:x}` from 2016 on; square brackets inside captions/parentheses:
  `(see \equationname s~[\ref{eq:dpml}])`. Punctuation inside the equation as `\,,`
  / `\,.`.
- **Figures**: `\begin{figure}[t!]`, `\includegraphics[width=0.48\textwidth,clip=]{x.eps}`,
  `\caption{…}\label{fig:x}` on the same line, provenance `%python …` comment after
  the caption. Listings are figures with the `minted` options
  `[mathescape, linenos, numbersep=5pt, gobble=2, frame=lines, framesep=2mm]`;
  inline snippets use `\begin{minted}{python}` or `[fontsize=\small]`.
- **Tables**: `deluxetable` with `\tablecaption`, `\tablehead{\colhead{…}}`,
  `\startdata \input{table.txt}`, `\tablecomments{…}`, `\label{table:x}`.
- **Lists**: `\begin{enumerate}[label=\emph{\alph*)}]` for data constraints;
  `itemize` for model ingredients (2013); each item a full sentence ending in `;`
  or `.`.
- **URLs**: `\\ \centerline{\url{http://github.com/jobovy/galpy}~.}` (tilde before
  the final period/comma; `\scriptsize` or `\small` for long ones; `\burl` for
  URLs with `#`); in footnotes `\protect{\url{…}}~.`.
- **Bibliography**: hand-written `\begin{thebibliography}{}` with
  `\bibitem[Author(Year)]{Key}` and keys like `Binney12a`, `BovyVc`,
  `binneytremaine`; journal macros `\mnras`, `\apj`, `\aj`, `\aap`, `\apjl`.
- **Authoring scaffolding to strip before submission**: `\newcommand{\response}[1]
  {{\bf #1}}` wraps referee-response changes; `%BOVY: reference point` marks
  revision anchors; timing data and figure-generation commands live in `%` comments.
- **arXiv packaging for minted**: `ms.tex` is a 16-line `article` wrapper that
  `\includepdf[pages=1-last]{galpy.pdf}` with a comment "The actual tex is included
  as ms.in and makes use of minted (which arxiv does not support); it needs to be
  compiled with `latex -shell-escape`".

---

## 12. Corpus statistics (for calibration)

| paper | words | sent. | median w/sent | I | we | `;`/1k w | `---`/1k w | footnotes | citet | citep |
|---|---|---|---|---|---|---|---|---|---|---|
| galpy 2015 | 13.7k | 574 | 22 | 29 | 81 | 4.4 | 1.4 | 4 | 64 | 30 |
| streams 2014 | 11.7k | 444 | 25 | 55 | 85 | 2.0 | 1.7 | 3 | 18 | 33 |
| Bovy & Rix 2013 | 18.2k | 623 | 27 | 1 | 397 | 3.0 | 0.8 | 8 | 74 | 53 |
| halo shape 2016 | 10.1k | 361 | 26 | 0 | 222 | 1.7 | 1.9 | 1 | 70 | 41 |
| inventory 2017 | 11.5k | 412 | 27 | 1 | 205 | 1.5 | 1.1 | 4 | 40 | 38 |
| Gaia rotation 2016 | 2.7k | 102 | 26 | 23 | 3 | 0.7 | 3.7 | 1 | 6 | 18 |
| acceleration 2020 | 2.8k | 89 | 28 | 24 | 22 | 1.4 | 3.9 | 0 | – | – |

Paragraphs: median 3–4 sentences, 84–122 words; 10th percentile 2 sentences.

## Sources read

- arXiv:1412.3451 — *galpy: A Python Library for Galactic Dynamics* (ApJS 2015), full `ms.in`
- arXiv:1401.2985 — *Dynamical modeling of tidal streams* (ApJ 2014)
- arXiv:1309.0809 — Bovy & Rix, *A direct dynamical measurement of the Milky Way's disk surface density profile…* (ApJ 2013)
- arXiv:1609.01298 — Bovy, Bahmanyar, Fritz & Kallivayalil, *The shape of the inner Milky Way halo from … Pal 5 and GD-1* (ApJ 2016)
- arXiv:1704.05063 — *Stellar inventory of the solar neighborhood using Gaia DR1* (MNRAS 2017)
- arXiv:1610.07610 — *Galactic rotation in Gaia DR1* (MNRAS Letters 2017), full text
- arXiv:2012.02169 — *A purely acceleration-based measurement of the fundamental Galactic parameters* (2020), full text
- arXiv:1512.00452 — *Detecting the disruption of dark-matter halos with stellar streams* (PRL 2016)
- arXiv:0905.2979 — Bovy, Hogg & Roweis, *Extreme deconvolution* (AoAS 2011), intro/conclusion
- Also scanned for statistics: 1510.06745, 1606.03470, 1704.03884, 1205.4033, 1111.6585
- galaxiesbook.org, chapters 0-02 and I-03 (voice check only)
