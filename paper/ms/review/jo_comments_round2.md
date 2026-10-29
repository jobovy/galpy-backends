# JB's round-2 comments (sections 8 onward) on the 2026-10-28 revision (galpy-backends-paper_2026-10-28_revision1_8bf622a.pdf), in reading order

### [R2.1] p.31L Highlight
> as the acceptance test.
**JB:** similar to before, you should make it clear that this is an important goal and acceptance test, but not the only goal. The other goal, which I think we discussed, is to keep the coverage as high or higher than before in terms of numbers of lines NOT covered. And in fact we are now much lower (from 24 to 14, should be mentioned somewhere, probably not right here)

### [R2.2] p.31R Highlight
> continues
**JB:** Add "for now"

### [R2.3] p.31R Highlight
> October 18,
**JB:** Needs to be updated, also PR numbers etc. (cost is harder, the lmc project has been using CI significantly, I guess I can give you galpy specific numbers once the current set of PRs is merged)

### [R2.4] p.31R Highlight
> this branch
**JB:** Readers will not know what 'this branch' means. I think you need to discuss the branch organization

### [R2.5] p.31R Highlight
> 49 against galpy’s main branch,
**JB:** I know we've checked and re-checked this, but this still seems low to me. It has felt that we've done more than 49 related main PRs.

### [R2.6] p.33L Highlight
> cause a PR could not be merged before its matrix had run, the shards queued behind each other when several PRs were open, and a flaky job meant a two-hour wait. We made several of the engineering decisions—stacked be-
**JB:** Important to stress here though how crucual the CI was to establishing the correctness of the PRs, as CI often turned up unexpected errors

### [R2.7] p.33L Highlight
> stacks of dependent PRs
**JB:** Probably worth pointing out that GitHub only released PR stacks as a feature in mid July (you can look up the exact date), so this only became possible then. Do you discuss the rebase hell that otherwise ensued when multiple PRs were open that touched the same files (don't use the word hell though)? That was solved by stacks.

### [R2.8] p.33L Highlight
> ledger,
**JB:** maybe add 'test' to ledger or explain slightly more because readers might not remember what the ledger is about

### [R2.9] p.33L Highlight
> added 107.
**JB:** Probably needs to be updated

### [R2.10] p.33L Highlight
> after- 5× wards the tier at about CAD 140 per month.
**JB:** note that since two weeks we're back on 20x Max, so could mention that for the end stretch

### [R2.11] p.33L Highlight
> was used only to draft this paper.
**JB:** add 'and its major revisions' because I used Fable for the major revision and will turn it on again for this one and any subsequent ones

### [R2.12] p.33L Highlight
> $5,800 in October 2026 through the 18th.
**JB:** Ask me for updated numbers once the final PRs merge

### [R2.13] p.33L Highlight
> backends.
**JB:** add "project" or so? "to the backends" isn't very clear

### [R2.14] p.33R Highlight
> has
**JB:** had

### [R2.15] p.33R Highlight
> (there are four: a tolerance in the King DF §s and the three cases noted in 6 and 7)
**JB:** Still up to date?

### [R2.16] p.33R Highlight
> have
**JB:** had

### [R2.17] p.33R StrikeOut
> not
**JB:** (no text)

### [R2.18] p.33R Highlight
> may
**JB:** was not allowed to

### [R2.19] p.33R Highlight
> Tests must make mean- ingful assertions, such as a gradient matching a finite difference that is shown to converge, and never merely that a result is finite.
**JB:** Also important to add that tolerances generally had to be very tight: rtol 1e-13 or better, often doing 1e-14. Basically only sampling tests that have Monte Carlo noise were allowed to have looser tolerances set by the MC noise

### [R2.20] p.33R Highlight
> ex- § cept during the stretch described in 8.3.
**JB:** We should add (but note that that stretch was reviewed by JB before being merged into the main backends branch)

### [R2.21] p.34L Highlight
> It could not merge into the main backend branch or into galpy’s main branch, and it could not change the continuous-integration configuration.
**JB:** I thnk it's important to describe how this is done: through the branch rulesets on GitHub and through the use of the jobovy-agent GitHub account which lacks the permission to update workflows. It's important to stress these hard guardrails against agent misbehavior (sorry, I'm sure you would never try to misbehave, but you might make mistakes!)

### [R2.22] p.34L Highlight
> regular intervals,
**JB:** maybe add "to ensure that it kept going"

### [R2.23] p.34L Highlight
> and 145 PRs in total went through the develop- ment branch before this branch was merged into the backend branch on 2026 September 2,
**JB:** You need to add that JB was so happy with the ways things were going when he returned from vacation that he let the unsupervised procedure continue for a few more weeks (I guess until Sept 2)

### [R2.24] p.34R Highlight
> it once left conflict markers in a working tree;
**JB:** maybe add (the type of error that a human review would have easily spotted)

### [R2.25] p.34R Highlight
> We have kept this account in the paper because it is the kind of information that a reader deciding whether to work this way needs.
**JB:** Further conclusion to: On the whole, the autonomous stretch was a success in keeping the agent working productively while JB was away, but only because the agent had a clear, limited, and verifiable goal.
[or something similar, I want to get across that you, the agent, did great work during this stretch, but it would have been harder to do more general work towards the project rather than just burning down the ledger].

### [R2.26] p.34R Highlight
> The acceptance test
**JB:** One of the main acceptance tests

### [R2.27] p.34R Highlight
> which grew from 2,466 to 4,315 tests over the course of this work,
**JB:** unless you say this elswhere, you should probably add in parenthesis or as a footnote: mainly due to the splitting up of existing tests, e.g., tests that internally looped over all galpy potentials were changed to have a single test per potential so failures and success could be traced better
 (feel free to rephrase!)

### [R2.28] p.34R Highlight
> We do not delete or modify tests that fail under a backend, but list them in a ledger, one line per test and mode, that marks them as expected failures, such that the size of the ledger is the measure of what re- mains to be done. A second ledger lists tests that pass under a backend but exceed the time limit of the contin- uous integration, and a third lists the tests of code that is permanently out of scope. We regenerate the ledgers from the actual test results on demand, such that they cannot drift from reality, and in addition we run a dedi- cated set of tests of the backend code itself, now 17,995 tests in 93 files.
**JB:** because we are explicitly describing what we did, this should be in the past tense

### [R2.29] p.34R Text
**JB:** Two more comments on this first paragraph of Section 8.4: we should say that we determined the time limit ourselves and what it is (and why); we should maybe discuss the ledger report that is posted on each PR?

### [R2.30] p.34R Highlight
> which is out of scope.
**JB:** which as an externally-wrapped code is out of scope

### [R2.31] p.35L Text
**JB:** The bottom y label is cut-off in this figure

### [R2.32] p.35L Highlight
> number of tests in the dedicated backend test modules.
**JB:** Why the big incraese near the end?

### [R2.33] p.35L Highlight
> takes
**JB:** maybe this should be "at first took"

### [R2.34] p.35R Highlight
> 4311 / 4315
**JB:** The four here are the four that are permanently out of scope? Are there really only four actionAngleTorus tests? ANd shouldn't we just leave them out of this tally? I know, I know, it's more honest to include them...

### [R2.35] p.36L Highlight
> 73 %
**JB:** 73% doesn't seem quite high. So 27% goes off the backend when it shouldn't? That's what this reads like

### [R2.36] p.36L Highlight
> the razor-thin disk of arbitrary surface density, whose adaptive quadrature some tests need for their ac- scipy curacy, and the DF of an evolving disk, whose grids of orbits, evaluated eagerly on the backend, made one shard of the forced run nine times slower. (A stream that the tests build with before the backend is forced numpy also keeps the state that it computes later on numpy.)
**JB:** Is this all still the case??? I feel like it shouldn't

### [R2.37] p.36R Highlight
> of a single concrete value now evaluates the function at 513 points of the bracket in one call and keeps the sub- interval with the sign change, which reduces the bracket 512-fold per call and the number of calls to about seven, followed by a secant step on the final bracket. An orbit whose initial condition is a concrete backend array, with nothing to differentiate, is integrated with the plain C integrator, and the state-transition matrix is computed only for a traced or gradient-tracking initial condition. The streams build their tracks on the backend; there, the first eager construction was dominated by com- jax pilation, because compiles every operation anew at jax each new array shape and the vectorized map over the track’s chunks created shapes that nothing else used, so a concrete construction loops over the chunks instead. The profiles of the disk DFs, the alignment of a coor- dinate frame with an orbit, and the evaluation of the multipole expansion at single points
**JB:** I feel like there's way too much detail here, much more detail on this conversions than given elsewhere in the paper (in this entire paragraph, I only highlighted some of it)

### [R2.38] p.36R Highlight
> 77 %
**JB:** So what's the 23%??? It is legit?

### [R2.39] p.36R Highlight
> and we count it with the sampling.
**JB:** At the end of this paragraph, I'm still left confused about whether all or all-but-legit runs on the backend or not? The 77% number seems like the main one and is very far from 100%

### [R2.40] p.36R Highlight
> 40 PRs
**JB:** didnb't you say 49 before?

### [R2.41] p.37L Highlight
> We always filed pre-existing defects of this kind as issues
**JB:** I don't think we generally explicitly filed them as issues, we generally just fixed them, no?

### [R2.42] p.37L Highlight
> a decade
**JB:** more than a decade

### [R2.43] p.37L Highlight
> listed.
**JB:** listed where?

### [R2.44] p.37R Highlight
> The one remaining ex- ception is by design: under compilation, the anisotropy of a constant-anisotropy distribution function cannot be traced, because it selects the order of a derivative in the inversion.
**JB:** I think this is still the case, but just checking to be sure!

### [R2.45] p.37R Highlight
> of any one of them.
**JB:** add something like "only based on the combination of them combined with human review of the code itself"

### [R2.46] p.37R Highlight
> is the shape of the
**JB:** is to illustrate the

### [R2.47] p.38L Highlight
> S5-HVS1
**JB:** Need to cite the discovery

### [R2.48] p.38L Highlight
> are limited
**JB:** are currently limited (the fast-orbits agent is working on extending to spherical and staeckel and beyond)

### [R2.49] p.38L Highlight
> the torus machinery.
**JB:** probably need to add "of Binney & McMillan (XXXX)"

### [R2.50] p.38R Highlight
> case with the St¨ackel approximation, where no reference exists.
**JB:** need to specify whcih of the figures this is (before looking I assume 27)

### [R2.51] p.38R Highlight
> add 17,995 more.
**JB:** maybe add a whimsical (AI agents \emph{love} to write tests)

### [R2.52] p.39L Text
**JB:** Not good that the legend overflows the plot. Why no torch compile, just to show options (fine)?

### [R2.53] p.39L Text
**JB:** I feel like this legend takes up too much space, maybe label orbits directly? Or also "who cares what the exact Jz is!"

### [R2.54] p.39L Highlight
> about fifty
**JB:** back to fifty... need to make sure this number is consistent throughout the text

### [R2.55] p.40L Text
**JB:** Again, maybe it's just not necessary to state the radial and vertical actions of these orbits, the true vs. network is the pokint of the panel

### [R2.56] p.40L Highlight
> Fit- ting a stream model to data with a Hamiltonian Monte § Carlo sampler, which the derivatives of 7 allow, is the obvious next application, and we will present it sepa- rately.
**JB:** I don't think we have a concrete plan (maybe you're writing a paper on your own!), so let's drop this statement

### [R2.57] p.40R Highlight
> fixed Lz,
**JB:** So this is just at a single Lz? Maybe discuss in the text

### [R2.58] p.40R Highlight
> reconstructs orbits that it has not seen in training.
**JB:** what was not included in training? Just interpolated? Fine

### [R2.59] p.40R Highlight
> the torus-mapping code
**JB:** again, emphasize that this is simply wrapped code from Binney & McMillan (...)

### [R2.60] p.40R Highlight
> the interpolated veloc- ity sampler of the quasi-isothermal disk DF is a rejection sampler that is not differentiable,
**JB:** Still the case?

### [R2.61] p.41L Highlight
> and the test that checks this will say when they are no longer needed.
**JB:** As an example, you can give the torch 2.15 special function stuff from PR 1772

### [R2.62] p.41L Highlight
> is
**JB:** is the start of

### [R2.63] p.41L Highlight
**JB:** series,

### [R2.64] p.41L Highlight
> https://docs.galpy.org .
**JB:** Is there a way to make this link go to v2.xx by default? Otherwise need to say to select the latest v2 version there

### [R2.65] p.41L Highlight
> We also thank the many users of whose bug reports and questions over galpy the past decade shaped the code that is extended here.
**JB:** Also need to thank the contributors, could grab these from GitHub and list in order of contributions

### [R2.66] p.41R Highlight
> October 18;
**JB:** Update (moving target, I know)

### [R2.67] p.42L Highlight
> both are implemented for galpy’s dynamical-friction forces.
**JB:** there's also NonInertialFrameForce? Nothing else right?

### [R2.68] p.42L Highlight
> galpy
**JB:** cite FDM friction paper (see comment on the figure)

### [R2.69] p.43L Highlight
**JB:** add M (state-transition matrix M)

### [R2.70] p.44L Highlight
> fuzzy-dark-matter friction,
**JB:** do we cite the Spilzfidel paper that introduced this (I'm an author, it's from 2025 or 2026, Adrian is his first name, I don't know how to spell his name offhand)
