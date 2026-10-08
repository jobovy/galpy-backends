# Style pass (2026-10-07)

A sentence-level revision of `sections/*.tex` toward the voice of Bovy's
first-author papers (calibrated on the galpy paper, `ms.in`, and
*Dynamical modeling of tidal streams*, arXiv:1401.2985), following
`STYLE.md`. No number, citation, equation, label, code block, figure path,
or `%TODO` line was changed; a script compared these between the old and
new sources for every file, and `make` was rerun after each section with
no undefined references.

## Kinds of changes

1. **Active voice with "we"** where the agent of a passive was us:
   "were fixed / was flagged / were replaced / is enforced by" became "we
   fixed / we flagged / we replaced / we enforce". Passives that describe
   what the code does ("is implemented as", "is evaluated") were kept, as
   in the galpy paper. In the process section, the actor is named (JB or
   the agent) where it was clear who did what.
2. **Runs of short sentences merged by subordination** (because, while,
   which, such that, a parenthesis, or a colon), e.g. the stand-alone
   "Under a jax trace the bisection is a rolled loop rather than an
   unrolled one."
3. **Take-away sentences in figure captions** where a speed or accuracy
   figure had none (jitamort, adoverhead, throughput, orbitsens, lyapunov,
   icderivcost, galax, orbitspeed, dfspeed, streamtrack, spraysp,
   neuralvertical, neural, dissipative). Each restates a claim that the
   text already makes (or, for spraysp, that the figure's data show at
   every N); none adds a new number.
4. **AI-isms removed**: "Sampling deserves a word.", "A few things are worth
   stating plainly.", "it deserves a description of its own", "is worth
   knowing", "A point ... is general enough to state here.", punchy
   stand-alone openers ("The backend follows the data.", "Some things are
   not done.").
5. **Small fixes**: "matched mine" -> "matched JB's own" (the paper is in
   "we"); plain "jax and torch" -> `\jax\ and \torch\`; a stray blank line
   after the `%TODO` comment in `dfs.tex` that started a new paragraph
   mid-sentence; a doubled "and ... and" in the process rules; one of two
   back-to-back references to fig:extrapolation in `streams.tex`.

## Before / after

1. (design, root finding)
   *Before:* "Root finding is provided by a drop-in replacement for scipy's
   brentq. On backend data it runs a fixed number of sign-preserving
   bisection steps, ... Under a jax trace the bisection is a rolled loop
   rather than an unrolled one. One shared root finder serves every
   action--angle and DF calculation, where galpy previously had several."
   *After:* "For root finding, galpy.backend provides a drop-in replacement
   for scipy's brentq, which on backend data runs a fixed number of
   sign-preserving bisection steps (a rolled rather than an unrolled loop
   under a jax trace), set by ... This one root finder now serves every
   action--angle and DF calculation, where galpy previously had several."

2. (design, principles)
   *Before:* "Six rules were fixed before any code was written, and the
   rest of this paper is largely a description of how they were kept."
   *After:* "We fixed six rules before writing any code, and the rest of
   this paper is largely a description of how we kept them."

3. (design, compilation)
   *Before:* "galpy does not call jax.jit or torch.compile on its own code.
   The reason is that compilation is most effective at the level of the
   user's whole computation, ..."
   *After:* "galpy does not call jax.jit or torch.compile on its own code,
   because compilation is most effective at the level of the user's whole
   computation, ..."

4. (potentials)
   *Before:* "The test suite checks these identities for every potential,
   under both backends and in both modes. For the user, the relevant
   derivatives are those with respect to the parameters. A potential
   constructed with a backend array as a parameter is differentiable ..."
   *After:* "While the test suite checks these identities for every
   potential, under both backends and in both modes, the derivatives that
   matter to the user are those with respect to the parameters: a
   potential constructed with a backend array as a parameter is
   differentiable ..."

5. (dfs)
   *Before:* "Sampling deserves a word. Every sampler in galpy takes a key=
   argument; key=None reproduces ..."
   *After:* "Every sampler in galpy takes a key= argument: key=None
   reproduces ..."

6. (streams)
   *Before:* "A point about these derivatives is general enough to state
   here. A stream model is a generative model: the stream's stars are a
   sample from the model's DF. The derivative of the track ... The first
   is not a physical quantity, and the two differ."
   *After:* "The following point applies to the derivatives of all of
   galpy's stream models. Because a stream model is a generative model, in
   which the stream's stars are a sample from the model's DF, the
   derivative of the track ... can be taken either with the sample of
   stars held fixed, ... or by letting the sample move with the parameter,
   ... The two differ, and the first is not a physical quantity; ..."

7. (process, fixes)
   *Before:* "About 50 PRs fixing galpy's numpy and C code were merged into
   the main branch as a result of the backend work. Some of these were
   found by the three-dimensional variational equations, ..."
   *After:* "... is an effective way to find bugs in the first, and the
   backend work led to about 50 PRs that fixed galpy's numpy and C code
   and were merged into the main branch. The three-dimensional variational
   equations, which require correct second derivatives of the potential,
   found some of these: ..."

8. (conclusion)
   *Before:* "Some things are not done. The torus-mapping code is not
   covered and will not be. astropy units cannot be used inside compiled
   programs. The construction of ..."
   *After:* "Some things are not done: the torus-mapping code is not covered
   and will not be, astropy units cannot be used inside compiled programs,
   and the construction of ..."

9. (caption, fig:dfspeed; take-away added)
   *After:* "... hatched bars are eager. Compiling with jax.jit or running
   on a GPU makes sampling much faster than with numpy, by up to three
   orders of magnitude for large samples."

10. (caption, fig:galax; take-away added)
   *After:* "... with per-orbit control as in galpy (dash-dotted). With the
   same per-orbit step-size control, the two codes take the same time to
   within a few percent."

## Deliberately left alone

- Paragraphs whose numbers are pending re-measurement (the St\"ackel
  speed paragraph and the inverse-transformation speed paragraph in
  `actionangle.tex`, the gS4 caption in `streams.tex`), apart from turning
  "The one-dimensional inverse transformation is shown in Figure" into
  "Figure ... shows ...". No take-away was added to fig:vinvspeed or
  fig:streamdfspeed for the same reason.
- The AI-statement sentence on who drafted the text (it carries a `%TODO`).
- Short verdict sentences that end a paragraph ("galpy keeps the
  per-orbit control."), which the galpy paper also uses.
