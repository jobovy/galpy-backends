# JB's round-4 comments

From `galpy-backends-paper_latexdiff-vs-4515cf6_revision1_2026-10-30_673b47a.pdf` (the latexdiff of 673b47a against 4515cf6; page and column of the annotation, highlighted text quoted). Answered in `response.md` Part H.

General instructions (chat, 2026-11-02): fully re-write the abstract in the style of JB's first-author papers and of the galpy paper (too long and detailed; no numbers like a few parts in 10^10; the essentials). Overall shortening: a few pages shorter in the end. All plots must fit nicely on the page. v2 docs are now up. Deliver: overview of major changes, a latexdiff against 673b47a, and the final paper.

### [R4.1] p.1L Text
**JB:** The intro need to be a half a page shorter, so shorten it. I thnk there's some stuff you can drop and tighten

### [R4.2] p.2R Highlight
> In 2026, as the first step of the work described in this paper, we implemented the three-dimensional variational equations that propagate the derivatives of an orbit with respect to its initial conditions in C for most of the potential library, and on top of them the computation of Lyapunov exponents (§ 4.3).
**JB:** Can drop this, part of this paper

### [R4.3] p.3L Text (the paragraph on other jax codes: "Other jax-based codes were used to ...", "Gradient-based samplers in jax were used to ...", "Normalizing flows ... were used to ...")
**JB:** Please write this much more in the active tense, too much passive tense here.

### [R4.4] p.3L Highlight
> We support two frameworks rather than one on purpose: doing so forced the implementation to be agnostic of the backend, with solutions that do not depend on the particular framework or that are at least easy to generalize, and because the ecosystem of array frameworks is in flux, this should make it easier in the future to add a backend or to replace one that falls out of use.
**JB:** I think this is almost verbatim what I had suggested, but please re-write in a less awkward and long way, conveying the same points, in the style of the paper

### [R4.5] p.8L Highlight
> a changed amplitude causes the re-trace that it must and a freshly created but identical wrapper does not cause a needless one.
**JB:** So when you change the amplitude of a potential, you have to re-trace? But it just changes a number, not a shape? No need really to answer in the paper, can be just to me

### [R4.6] p.10L Highlight
> adiabatic-contraction wrapper
**JB:** maybe should be 'adiabatic-dark-matter-halo-contraction wrapper' and then cite the Cautun et al. paper on which implementatio it is based

### [R4.7] p.13R Highlight
> A leapfrog for many orbits
**JB:** Should be 'Backend symplectic integrators for many orbits'?

### [R4.8] p.15L Highlight
> fourth- and sixth-order symplectic integrators of
**JB:** insert "backend versions of the '

### [R4.9] p.15L Highlight (same place)
**JB:** Should cite the original method papers here, galpy paper probably has them

### [R4.10] p.15L Highlight
> 3250
**JB:** steps or force evaluations?

### [R4.11] p.16R Highlight
> evaluations against the adaptive method's ≈3250), is the
**JB:** This repeats info from before, so just refer back to this (also solves same confusion about the latter number: steps or force evals? Anyway, should be resolved through my other comment above)

### [R4.12] p.16L Highlight
> 4.5. Speed
**JB:** The bsolute times in this section depend strongly on the exact problem, so de-emphasize them and focus on relative duration of different methods, which is more robust (although not fully). Too many details mentioned is also very LLM-y, which we want to avoid

### [R4.13] p.18L Highlight (Figure 10 caption)
**JB:** does this caption fit on the page in the paper itself (not the latexdiff)? Probably not. It should! EDIT: I just checked and it indeed doesn't fit. So you need to shrink the figure a bit and make sure the caption fits within the margins. Maybe bring the CPU, Titan Xp, H100 labels inside the plots. And remind me to review this caption since I can't read it!

### [R4.14] p.18R Text (Figure 10 labels)
**JB:** Maybe for even more clarity, it would be good to indicate the linestyle for each lgend item on its right, because it can still be a bit confusing which curve a label applies to

### [R4.15] p.18L Text (Figure 10a, the jax jit symplec6 line)
**JB:** Do you discuss this funny jax jit symplec6 line in the text?

### [R4.16] p.18R Text (Figure 10c)
**JB:** We can drop the 0.1 mus per orbit line here, because the fastest one gets is 1 mus (which is impressive!)

### [R4.17] p.18L Highlight
> NVIDIA TITAN Xp
**JB:** Maybe for the dense reader add 'GPU' also for the H100 panel

### [R4.18] p.19L Highlight
> all of which are the building blocks of § 2.4: root finding for the turning points of
**JB:** This should be more like "all of which were discussed in paragraph 2.4". Building blocks is odd in this context

### [R4.19] p.20L Highlight
> The inverse transformation, from actions and an-
**JB:** Maybe these inverse transforms should be their own short subsection.

### [R4.20] p.24R Text (Figure 15b, the jax H100 jit line dropping at N >= 10^4)
**JB:** Is this point discussed? Because it looks odd!!!

### [R4.21] p.24R Text (Figure 15b, the "+transfer" lines)
**JB:** These transfer curves are just on top of the no-transfer ones? SO maybe just have one and then say that the transfer is negligible?

### [R4.22] p.25L Highlight
> construction is differentiable for every DF whose construction is itself a numerical computation—the
**JB:** Are there still exceptions? Because it not we don't need the 'for every DF whose cobstruction is itself a numerical computation" caveat

### [R4.23] p.26L Highlight
> Schwarzschild
**JB:** Need a reference, galaxiesbook.org probably has it in its Chapter 7, 8, or 9 (the disk DF chapter)

### [R4.24] p.26L Highlight
> the energy derivative of their mass profiles are likewise differentiable
**JB:** I still don't understand what the 'energy derivative of the mass profile' is? Do you dMdE? That's the different energy distribution (I think, you can check galaxesbook.org's definition in its chapter on spherical DFs, near where it talks about the density of states). Or look at the galpy docs for this function.

### [R4.25] p.31R Highlight
> 8. HOW THE BACKENDS WERE BUILT
**JB:** This section is quite long. Try to shorten it, at least in part by not repeating information. But in general try to shorten

### [R4.26] p.32R Text (Figure 20)
**JB:** This kind of direct line labeling is much better than the broken line version where you go to the axis and then go horizontal, which you use in other figures. Reconsider all other figures that use direct line labeling to move towards the style in this figure if you can (but make sure it doesn't get messy!).

### [R4.27] p.38L Highlight (Table 1)
> + one documented exception 4,353
**JB:** Still confused why one documented exception adds 4 tests...
