# JB's round-5 comments

From `galpy-backends-paper_latexdiff-vs-673b47a_revision2_2026-11-02_f11ce46.pdf` (the latexdiff of f11ce46 against 673b47a; page and column of the annotation, highlighted text quoted). Answered in `response.md` Part I.

General instructions (chat, 2026-11-03): "More paper comments, getting pretty close! Then give me a latexdiff with these fixes and also the GPU stuff that you're doing now (so wait until that's all done)". The GPU stuff: the device-aware incomplete-gamma switch (galpy PR #1797), Fig. 15 re-measured on both GPUs, and the orbit figures' compiled-jax GPU lines.

### [R5.1] p.1R Highlight
> The backend
**JB:** Replace with: The user-facing design is kept simple: the active backend
[or if you think this is clunky, rephrase]

### [R5.2] p.1R Caret
> [caret after "the" in] the interface is unchanged
**JB:** galpy

### [R5.3] p.1R Caret
> [caret at] We give examples of fitting the Galactic potential ...
**JB:** Replace the statt of this sentence with: To illustrate the new capabilities, we present simple examples of fitting

### [R5.4] p.1L Caret
> [caret after "the" in] under the direction and review of the human author
**JB:** detailed

### [R5.5] p.1R Highlight (the abstract's last sentence)
> galpy is available at https://github.com/jobovy/galpy, with the documentation of the backends at https://docs.galpy.org/en/v2/.
**JB:** I think this last sentence should say that this is the start of 'galpy v2'

### [R5.6] p.9L Text (Fig. 2, left panel)
**JB:** Need a bit less transparency here (is there any applied?) and a bit more in the right panel, where the lines are hard to follow. But this style is great!

### [R5.7] p.9R Text (Fig. 2, right panel)
**JB:** Need a bit less transparency here so the lines don't cross the legend as much (is there any?)

### [R5.8] p.15R Highlight
> the problem,
**JB:** the specifics of the problem

### [R5.9] p.15R Highlight
> ones.
**JB:** trends

### [R5.10] p.16L Highlight (Fig. 9, left panel)
> λ(3 × 10 4 ) = 0.046
**JB:** Can we move this label slightly to the right so it doesn't overlap the dashed line?

### [R5.11] p.18R Text (Fig. 10, the label columns)
**JB:** So the line color helps, but it's confusing that some lines start at the very left edge of the legend and some from the middle and then go through other legend items and seem to emerge from them. Why not, in all these types of plots (including Figs. 1 and 2 where I made comments) have them start from the very left edge always, so the transparency doesn't really matter anymore?

### [R5.12] p.18L Text (Fig. 10, the page)
**JB:** There is still too little margin below the figure and its caption, so it will have to shrink some more. It should have a normal margin at the bottom. But I reviewed the caption and it looks good

### [R5.13] p.20L Highlight (title of §5.3)
> The inverse transformation
**JB:** Inverse transformations

### [R5.14] p.20L Text (end of §5.3)
**JB:** We should add a closing sentence to 5.3: Other galpy-native inverse action-angle transformations are currently under active development and will be added with full backend support soon.

### [R5.15] p.23R Text (Fig. 15)
**JB:** This is good labeling, I think we should probably do this style for all the plots that are in this style (legend on the right, with many lines being labeled). Make sure to look at all figures to get them all!

### [R5.16] p.31R Text (Fig. 22, right panel)
**JB:** Again, lines connecting legend to curves overlap the legend too much, see other comments.

### [R5.17] p.32L Highlight (§8.1)
> [the backend branch] ... and which **is** periodically rebased onto the main branch
**JB:** was [make sure to use a consistent tense in this discussion]

### [R5.18] p.33L Highlight (§8.1)
> The **far** larger cost was the continuous integration
**JB:** delete far

### [R5.19] p.41R Highlight (§9)
> the start of galpy's version 2 series, which **is** distributed as pre-releases
**JB:** for now is
