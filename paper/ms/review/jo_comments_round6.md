# JB's round-6 comments

From `galpy-backends-paper_latexdiff-vs-f11ce46_revision3_2026-11-03_b1c3c99.pdf` (the latexdiff of b1c3c99 against f11ce46; page and column of the annotation, highlighted text quoted). Answered in `response.md` Part I.

General instruction (chat, 2026-11-03): "Some more comments on the latest diff, we're probably done after this."

### [R6.1] p.9 Text (Fig. 2, right panel)
**JB:** Left panel looks good, but for the right panel: best to remove the fading of the line as it passes through a legend and just extend the legend lines a bit further into the figure to go to the curves. They are hard to see here.

### [R6.2] p.15L Highlight (Fig. 9a)
> λ(3 × 10⁴) = 0.046
**JB:** You moved it too far! It can't go out of the box. So try to go left until you hit the dashed curve, but better to be on top of the dashed curve a bit than to be out of the box

### [R6.3] p.17 Highlight (Fig. 10, the in-plot label)
> galpy C (36 threads)
**JB:** Add a little line from this label to the curve it is labeling for consistency with the other legend labels. In all panels

### [R6.4] p.17 Highlight (Fig. 10a, label column)
> galpy C (16 threads)
**JB:** Can we use the same grayscale for the label and line label as for the curve it is labeling?

### [R6.5] p.18 Highlight (Fig. 11b)
> 0.01, 0.1, 1, 10, 100 ms
**JB:** Not good that the label goes under the line, fix

### [R6.6] p.18 Highlight (Fig. 11a)
> hatched = eager, solid = compiled
**JB:** Not good that the label extends outside of the box, why not just a line break after "eager,"?

### [R6.7] p.19L Highlight (§5.3)
> Other galpy-native inverse action–angle transformations are under active development and will be added with full backend support soon.
**JB:** New paragraph for this

### [R6.8] p.20 Highlight (§5.4, the estimateDeltaStaeckel code example, split between the two columns at the foot of the Fig. 13 page)
> delta= estimateDeltaStaeckel(MWPotential2014, o.R(ts),o.z(ts)) ... jr,lz,jz= aAS(...)
**JB:** I'd rather it not be the case that a code example renders split between columns at the end of the page. Maybe cut a bit of text just before to make it go on the previous page?

### [R6.9] p.22 Highlight (Fig. 15b)
> jax H100 jit
**JB:** Can we not extend this line to 10^6? And I'm sure I've asked this before, but no torch H100 compiled?

### [R6.10] p.29 Text (Fig. 20, panels b/c)
**JB:** Are the curves that stop at 10^5 not able to go higher?

### [R6.11] p.30 Text (Fig. 22)
**JB:** Are the curves that stop at 10^5 not able to go higher?

### [R6.12] p.37L Highlight (§9.1, the objective/adam_descent code example at the top of the page)
> objective= jax.jit(jax.value_and_grad(closest_approach)) q= adam_descent(objective,q0=1.1) # a few lines of optax
**JB:** Ideally would also not be after a page change
