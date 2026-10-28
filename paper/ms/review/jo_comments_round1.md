### [1] p.1L Highlight
> The rea- son that such codes are useful is simple: the motion of a star in a given gravitational potential is a solved problem, but one that requires careful numerical work, and a well-tested implementation that is shared saves
**JB:** I would add something along the lines of that it allows developers to build more complex models within a well-defined framework that can then be used by users to use more sophisticated models. And worth to emphasize how useful the action-angle implementations have been, not trivial at all to do and used by many people in many analyses that would not otherwise have happened.

### [2] p.1R StrikeOut
> and from the mis- takes made along the way.
**JB:** (no text)

### [3] p.1R Highlight
> orbits
**JB:** and also actions [maybe you give examples later]

### [4] p.1R Highlight
> time-critical
**JB:** 'computationally-expensive' instead?

### [5] p.1R Highlight
> (1,056 at the time of writ- ing, from four in 2012 to about 170 per year since 2024).
**JB:** I need to update these numbers, for now I wouldn't be so precise because the 1056 number is a year old or so

### [6] p.2L Highlight
> A smaller set of papers, perhaps one in six, uses as a modeling framework in which a DF or galpy a potential is fit to data:
**JB:** It would be good to include some more examples that don't include me as an author...

### [7] p.2L Highlight
> has changed much since 2015, but the trends galpy are simple to state.
**JB:** I think this paragraph includes the main changes since 2015, but do another audit

### [8] p.2L Highlight
> particle-spray models
**JB:** Need to cite the Qian et al. paper in the galpy "How to cite" section for the streamspray implementation

### [9] p.2L Highlight
> general framework for spherical DFs appeared.
**JB:** Need to cite the Lane, Bovy, & Mackereth paper for the implementation

### [10] p.2L Highlight
> gained inverse transformations
**JB:** Need to cite Bovy (2027, in preparation) here

### [11] p.2L Highlight
> Lyapunov exponents.
**JB:** This is really part of this work

### [12] p.2L Highlight
> In 2026, we implemented the three-dimensional varia- tional equations that propagate the derivatives of an or- bit with respect to its initial conditions in for most C of the potential library.
**JB:** This should include the Lyapunov addition. And this is really part of this work's push, to get the derivatives of orbit integrations. Probably worth mentioning or just reconsidering this.

### [13] p.2L Highlight
> Over the same period the code grew from 36,000 to 93,000 lines of Python
**JB:** Is this correct? Coverage reports report many fewer I think?

### [14] p.2R Highlight
> that cover all of the code.
**JB:** Mention the % covered. Maybe also the number of lines NOT covered

### [15] p.2R Highlight
> a code modeled on gala, jax provides the halo potentials whose acceleration direc- tions are compared with the curvature of stellar streams around other galaxies to constrain the shapes of their halos (S. Wu et al. 2026; Euclid Collaboration et al. 2026), and the orbits used to forecast how well such curvature tests separate dark-matter models (N. Stark- man et al. 2026a).
**JB:** I think it's a bit weird to immediately jump into galax applications. Maybe a quick sentence giving a general overview of galax, then discuss its applications

### [16] p.2R Highlight
> Other codes jax
**JB:** Should this be "Other jax-based codes are used to"?

### [17] p.2R Highlight
> fit
**JB:** were used to fit

### [18] p.2R Highlight
> Normalizing flows for the DF, differentiated automatically, recover the po- tential from phase-space snapshots of
**JB:** In general I don't like this construction "A METHOD .... does ... [CITE PEOPLE]", it should be more "METHOD ... was used to ... [CITE]" or "CITE used METHOD to ...". For all of the clauses in this paragraph

### [19] p.3L Highlight
> In this paper, we describe an extension of that galpy adds and as backends to its existing jax PyTorch numpy and implementation. C
**JB:** i'm not sure whether you will discuss this elsewhere, but part of the reason to support both jax and torch is to make sure the new backends implementation was somewhat backend-agnostic, requiring us to come up with solutions that as much as possible don't care about the specific backend or that are at least easy to generalize when we want to add more backends. We believe the backend ecosystem to be fluid, especially in the new AI era, so this will hopefully future-proof this code by making it easier in the future to add additional backends or replace existing ones in case they fall out of use

### [20] p.3L Highlight
> numpy
**JB:** Maybe here it would be good to add in parentheses (and scipy) to emphasize that a lot actually comes from scipy. And it's really numpy+scipy that's replaced by the backends

### [21] p.3L Highlight
> with the same tolerances.
**JB:** To be precise we probably have to add "(with very few exceptions)" or stronger

### [22] p.3L Highlight
> Most
**JB:** Should we say "All" here? Most is a bit waffly, leaves room for lots of exceptions when there really aren't any at this point?

### [23] p.3L Highlight
> the use of astropy units inside compiled programs.
**JB:** Starkman is working on a jax astropy units package, allowing units in jax programs (look at his GitHub, not sure what the status is); might be worth mentioning briefly

### [24] p.3L Highlight
> the torus-mapping code
**JB:** I think this is such a special case, the only case where galpy is just wrapping an external code, that I think we can just omit it here and just say astropy is missing. We can make sure to mention it later on, but I think this gives it too much prominence. It's really not an important gap.

### [25] p.3R Highlight
> behind the orbit derivatives.
**JB:** maybe add (which are also useful outside of the backend frameworks for computing orbit sensitivities and Lyapunov exponents) or something similar and better written.

### [26] p.3R Highlight
> separate from the repository, at galpy https://github.com/jobovy/galpy-jax
**JB:** Maybe we should rename this repos to galpy-backends

### [27] p.3R Highlight
> one test tolerance, numpy
**JB:** really only one? I feel like I approved more than 1

### [28] p.3R Highlight
> is the acceptance test.
**JB:** I'm not sure what this means and I don't think it's correct. I think what you mean is that one aim was to make the existing test suite fully pass under all backends, but that's not the only acceptance test, just an important part of it. Other acceptance is that we want new tests to pass, which can test functioality that didn't exist before, like autodiff.

### [29] p.4L Highlight
> from galpy.potential import (MWPotential2014, evaluateRforces) R= jnp.linspace(0.5,2.,1001) FR= evaluateRforces(MWPotential2014,R,0.) # jax array
**JB:** Since the introduction of CompositePotential, this is outdated code, you should just do MWPotential2014.Rforce(R,0.) and in all code examples in this paper don't use the evaluateX functions in potentials, use the method-based approach (I don't think it has gaps, test any updated code examples)

### [30] p.4L Highlight
> The
**JB:** We need to say something like "Backends are infectious [I don't think infectious is the right word here, but something like it to get the point across]: any use of a backend array in any input turns on the backend computation and output" Maybe requires a bit of a re-write

### [31] p.4L Highlight
**JB:** Again, don't use evaluatePotentials, just call MWPotential2014

### [32] p.4L Highlight
> (the default integrators run on the host), C
**JB:** Maybe this doesn't require saying? Because obviously C code runs on the CPU? Or at least say "the default C integrators run on the CPU" because I don't think all readers will equate host with CPU and that's really what you mean

### [33] p.4L Highlight
> The resolution order is therefore: an explicit argu- xp= ment,
**JB:** You haven't said yet that you can pass xp= as an argument? Is this generally the case? I thought this is more of an external capability that users generally wouldn't use? So maybe it needs to be stated that there is also the xp= possibility for some/all functions but that's really to ensure consistent use internally and not supposed to be used by users

ALso, the 'therefore' here seems like it's not quite appropriate, because the resolution order hasn't been discussed clearly yet. I would say above that the context manager and the default from the config file still defers to the data type so that the force=True statement makes more sense later and so the 'therefore' makes more sense.

### [34] p.4R Highlight
> Because cannot trace jax.jit objects, Quantity
**JB:** I know they can't natively, but they can't even with Starkman's library (see earlier comment)? Not suggesting we do this, but if it's possible we should discuss and discuss that we might add this later

### [35] p.4R Text
**JB:** Maybe an opening sentence repeating the principle: "Code in the physics modules is written as much as possible in a backend-agnostic manner". To achieve this, every migrated compute method ...

### [36] p.4R Highlight
> def _evaluate(self,R,z,phi=0.,t=0.): xp= get_namespace(R,z,phi,t) return -self._amp/xp.sqrt(R**2+z**2+self._b2)
**JB:** Briefly say what this is (Plummer?)

### [37] p.4R Highlight
> input(’R’,’z’,’phi’,’t’), @backend
**JB:** This is just an example no? Sometimes it looks different? So make that clear

### [38] p.5L Highlight
> host.
**JB:** Again to avoid non-expert confusing maybe "host CPU"

### [39] p.5L Highlight
> such that we could migrate the functions one at a time.
**JB:** They are all migrated now thought, right? No need to mention it here, but I want to make sure!

### [40] p.5L Highlight
> return
**JB:** maybe we should say "currently return" in case this gets fixed?

### [41] p.5L Highlight
> where its algorithm switches.
**JB:** switches to what?

### [42] p.5R Text
**JB:** I added a small explanation of what this code does, please check my new text

### [43] p.5R Highlight
> Figure 1
**JB:** I feel like it's a bit weird to start discussing a GPU speed up figure before discussing the GPUs, so maybe reconsider this? Seems like this figure should really be used to show off the GPU capabilities. Maybe there's a figure for CPU compilation benefits? But they aren't large for this right, or maybe even small! Figure 1 also feels a bit redundant with Figure 3 as is.

### [44] p.6L Highlight
> Figure 2
**JB:** Again a GPU figure I think, but we haven't introduced GPUs really yet (that's in 2;7) and the cost of a derivative on CPU is also interesting. The paper is quite long, so I'm loath to suggest new figures, but maybe something to consider

### [45] p.6L Highlight
> For the test suite, has an optional galpy global trace mode that traces every call at the coercion boundary described above; we use it to run the suite in compiled mode, and it is not needed in normal use.
**JB:** This feels a bit out of place? Maybe just discuss where tests are discussed in more detail? Or at least explain better

### [46] p.6L Highlight
> the sixty or so iterations with which a stream track is refined.
**JB:** Is this discussed elsewhere? I'm not actually sure what this means! But I get what you mean in general

### [47] p.6L Highlight
> such that a potential whose amplitude is changed causes a re-trace while a freshly created but identical wrapper does not.
**JB:** I don't quite understand this

### [48] p.6L Highlight
> a thread-local mirror while is PyTorch tracing.
**JB:** Maybe a closing remark to end the subsection? Feels like a very abrupt end here

### [49] p.6R Highlight
> § 8.5
**JB:** Hopefully it doesn't repeat to much of what's here, otherwise consolidate

### [50] p.6R Highlight
> 2.7. GPUs
**JB:** Since we'll be mentioning them constantly, we should discuss the two types of GPU that we use in the paper, but of course state that these just stand in for different types of GPU that users might use. Titan Xp is a relatively low-cost consumer-grade GPU while the H100 is a data-center-grade GPU that is quite expensive (maybe add some estimated costs)

### [51] p.7L Highlight
> 104 N =
**JB:** Is the 10^4 stuff adding much here? I find it hard to read, so maybe just doing 10^6 is better? That's all that's in (b) anyway. Also, in general better labeling of the lines would be good in the figures, often it's quite hard to trace the line connecting the label with the line in the figure

### [52] p.7R Highlight
> jax jit
**JB:** Why no torch compile here?

### [53] p.8L Highlight
> 100 μs,
**JB:** I think you have to add 10 mus for the H100 results to lie between gray curves

### [54] p.8L Highlight
> pynbody
**JB:** cite pynbody here

### [55] p.8L Highlight
> (J. J. Webb et al. 2020),
**JB:** Also cite AMUSE

### [56] p.8L Highlight
> the adiabatic-contraction wrapper, whose construction is a root-find, declares itself incom- scipy patible and is computed on numpy.
**JB:** I don't think this is true anymore? So maybe you need a full audit of such outdated statements in the paper

### [57] p.8L Highlight
> J. J. Webb et al. 2019)
**JB:** Maybe just remove this ref as it seems like it's for the whole list when it's really just for the last entries

### [58] p.8R Highlight
> Forty-nine of the fifty potentials that can be constructed from parameters alone are differentiable with respect to their amplitude in this way, both eagerly and under com- pilation; the one exception is the disk basis-function ex- pansion under jax.jit.
**JB:** Is that still the case? If it is, we need to make it not the case! And then say "All current fifty potentials ..." And isn't there a DiskSCF and DiskMultipole potential?

### [59] p.9L Highlight
> carry the derivative through the expansion coefficients.
**JB:** Also for DiskSCF and DiskMultipole? Should be!

### [60] p.9L Highlight
> but not in the tabulated values.
**JB:** Not even when they are set up to be evaluated? I'm thinking of a interpSphericalPotential derived from an input potential or procedure, we'd like the interpSphericalPotential to be differentiable wrt the parameters that went into the thing that goes into interpSphericalPotential (get it?). And for the other tabulated potentials as well.

### [61] p.9L Highlight
> A potential that moves along a precomputed orbit, as the MovingObjectPotential, keeps that orbit on numpy and computes the field at backend coordinates.
**JB:** What if we want the derivative wrt the orbit (maybe integrated in a potential itself)? And does it do GPU?

### [62] p.9R Highlight
> AD.
**JB:** Unless it's used a ton, I think it's best to spell out automatic differentiation

### [63] p.10L Highlight
> Figure 4.
**JB:** Figure placement: previous page

### [64] p.10L Highlight
> on axis: identical (0)
**JB:** Need to explain this in the caption

### [65] p.11R Highlight
> because the variational integration dou- bles their cost.
**JB:** But isn't the cost much higher for the non-symplectic ones as well?

### [66] p.11R Highlight
**JB:** I changed on of these earlier but maybe I shouldn't have

### [67] p.11R Highlight
> The third way is the old one: an orbit with a con- crete backend initial condition and a method that is not routed is integrated in and returned as a backend array C that is not differentiable, with a warning.
**JB:** I'm confused, doesn't this conflict with the first statement? Make more clear!

### [68] p.11R Highlight
> Initializing an orbit from observed coordinates with backend arrays works as well, such that a derivative with respect to a star’s distance or proper motion is available.
**JB:** Do you mention anywhere how to use from_name to initialize a backend orbit?

### [69] p.11R Highlight
> 4.2. A leapfrog for many orbits
**JB:** I think this subsection needs a motivation: why are we doing this? Rework this whole subsection with this in mind.

### [70] p.11R Highlight
> For each of 2000 orbits in the Milky-Way model with a moving satellite and the acceleration of a non-inertial frame, the error against the integrator falls by a factor C of four per halving of the step, and in a static potential the energy error oscillates without the secular drift that a non-symplectic second-order method with the same steps shows.
**JB:** I don't understand why you say "For each of 2000 orbits..." because these statements are generic properties of leapfrog. I think you are thinking of an example, but that should come later, as an example of its speed.

### [71] p.11R Highlight
> the error against the integrator falls by a factor C of four per halving of the step,
**JB:** You mean against the DOP853_C one? Maybe just say 'the error' instead of 'the error against...'?

### [72] p.12L Highlight
> Velocity-dependent forces,
**JB:** I think you just need to say 'non-conservative' because that's actually the check in the code and it's what matters

### [73] p.12R Highlight
> for
**JB:** for example, for...

### [74] p.13L Highlight
> as in N-body simu- lations.
**JB:** Of course, we don't do N-body galpy so this is irrelevant

### [75] p.13L Highlight
> that do not run on the backends.
**JB:** but they can? So then say 'do not require the backends'

### [76] p.13L Highlight
> The remedy is to integrate with a fixed number of steps when differentiating a batch,
**JB:** Do we or can we check that the user is using this correctly? We know when something is traced for a derivative, right?

### [77] p.13L Highlight
> and the eighth-order method of 10−5 carries a gradient error of about in torchdiffeq its backward pass, such that uses its fifth-order galpy method.
**JB:** Unless it's directly related to the first statement in this sentence, should be a separate sentence

### [78] p.13R Highlight
> Figure 12
**JB:** Figures should appear in the order in which they are first discussed (generally, unless explicitly referencing forward). I don't think Figures 10 and 11 have been discussed yet?

### [79] p.13R Highlight
> in-backend solvers on one CPU core
**JB:** Is there a torch CPU line?

### [80] p.13R Highlight
> the lines at a PyTorch few thousand orbits.
**JB:** The current figure is too messy for me to check this statement, but please check all of these statements again (and see my comment on the figure itself, about breaking it into panels)

### [81] p.13R Highlight
> its OpenMP loop speeds up almost linearly up to the 36 physical cores of our node (22 to 26 times on 36 cores; hyper-threads add nothing),
**JB:** I think we should probably just do 36 threads at most, not 72, if hyperthreading doesn't help

### [82] p.13R Highlight
> compiled jax, whose runtime keeps no more than a few cores busy, gains nothing.
**JB:** And there is really no easy way to make jax use multiple cores here? Fine if not, no need to discuss then!

### [83] p.13R Highlight
> with galpy’s own arithmetic
**JB:** I think you should say something like "when directly compiling galpy's C potential code to CUDA" because I think readers will find "own arithmetic" confusing. Or maybe that's not quite right because you had to port the GSL special functions, so then maybe just say "using direct ports of the necessary special functions" to distinguish it from the optimized special functions mentioned right after.

### [84] p.14L Highlight
> galpy’s sixth-order symplectic integrator with one common step for the whole batch reaches the same accuracy (the worst-case position error over 300 orbits after 2 Gyr) with 15% fewer force evaluations and a frac- tion of the state, and runs in 0.78 µs (galpy’s arithmetic) and 0.25 µs (fast special functions) per orbit, 3.0 to 3.7 times faster than dop853.
**JB:** This is also not part of galpy? But doesn't that imply that we should perhaps also implement the symplectic integrator directly in the backends like we did for leapfrog, so we can actually get this speed-up within galpy? Seems manageable

### [85] p.14R Highlight
> the same orbits,
**JB:** same orbits as what?

### [86] p.14R Highlight
> such that every orbit takes the steps that the most demanding orbit requires; this is cheaper per orbit for large batches, by about
**JB:** Previously you had told me that it took some median step, so that was wrong, it actually uses the smallest step (-ish?)?

### [87] p.14R Highlight
> turn- ing points of an orbit, quadrature of the action inte- grals between them, and, for the approximate methods, auxiliary orbit integrations.
**JB:** Need to briefly refer back to the section 2.x that discusses the numerical utilities on the backends (very briefly)

### [88] p.14R Highlight
> methods,
**JB:** Isn't this just actionAngleIsochroneApprox? (from Bovy 2014). If so, just one method and identify it

### [89] p.14R Highlight
> runs on
**JB:** are migrated onto

### [90] p.14R Highlight
> external torus-mapping code,
**JB:** Need to cite it properly. Is it mentioned more? I don't think so, we can just make a quick statement that this is an external code wrapped for convenience and not considered to be a core part of galpy.

### [91] p.15L Text
**JB:** Need a bit more space between the panels, not too much

### [92] p.15L Highlight
> hatched = eager
**JB:** Any reason why jax is jitted but torch isn't compiled? Just to keep the number of bars down? That's fine!

### [93] p.15L Highlight
> hatched bars are eager.
**JB:** And then also say what is compiled

### [94] p.15L Highlight
> actionAngle
**JB:** This is misleading, because you don't set up an actionAngle instance, you set up one of its sub-classes like actionAngleStaeckel. So rephrase

### [95] p.15L Highlight
> St¨ackel, adiabatic, and spheri- cal
**JB:** I would order them spherical, adiabatic, and Staeckel (and you don't mention the others? Which is fine, but maybe say, e.g.,.

### [96] p.15L Highlight
> it adjusts the potential to match measured actions.
**JB:** its model includes actions (that's a more interesting use case than matching actions)

### [97] p.16L Text
**JB:** Here we should also perhaps split into a CPU and a GPU panel like I suggest for Figure 12

### [98] p.16L Highlight
> Purple:
**JB:** In purple we display the experimental... (re-write)

### [99] p.16L Highlight
> rebuilt from q
**JB:** If you follow my suggestion in the caption of that figure, will need to update this description

### [100] p.16R Highlight
> 5.3.
**JB:** I feel like this and the previous subsection should be swapped unless there's a reason not to

### [101] p.17L Text
**JB:** There are a lot of lines in this figure. because it's such an important one, one option is to split it into two or three panels with a reference curve (like the galpy C 72 threads one) repeated for reference: maybe CPU should be one panel and GPU another or maybe two GPU panels if there are a lot of lines and the CUDA kernel one...

### [102] p.17L Highlight
> On the H100, galpy’s sixth-order symplectic integrator with one common step (not shown) reaches the same worst-case accuracy 3.0 (§ to 3.7 times faster, because these disc orbits share a dynamical timescale 4.5).
**JB:** Once we have more panels, maybe we should show it?

### [103] p.17L Highlight
> in prep.),
**JB:** I think AAS wants a year, so put 2027

### [104] p.17R Text
**JB:** Maybe we need to add little diagonal arrows "Faster" "Slower" to these types of plots of speed because it always takes me a few seconds to figure out which way is faster and it's not consistent between figures which way it is

### [105] p.17R Highlight
> including the copies between host and
**JB:** Say what we are supposed to see for it

### [106] p.18L Highlight
> 8e −09
**JB:** 8 x 10^-9 instead

### [107] p.18L Highlight
> in a logarithmic halo as a function
**JB:** maybe we should use MWPotential2014 and do the derivative wrt the mass of the disk? Or more fun: with respect to the ratio b/a!

### [108] p.18L Highlight
> return aAVI.xvFreqs(J,angles)[:2] # (z,v_z) z,vz= torus(0.5) dz_dsigma,dvz_dsigma= jax.jacfwd(torus)(0.5)
**JB:** Not great that this code example is spread over multiple pages!

### [109] p.19L Highlight
> 0.3 ms
**JB:** This is on one CPU or more?

### [110] p.19L Highlight
> Eager PyTorch on the CPU runs at the speed of numpy, while eager jax is bound by dispatching hundreds of small operations
**JB:** We should discuss compiled/jitted as well or people will think the CPU backends are terrible

### [111] p.19L Highlight
> than on 16 OpenMP threads, and compiled eval- C jax 105 uates points in 0.16 s. On the TITAN Xp, with its slow double-precision arithmetic, eager beats PyTorch single-threaded by a factor of 5–6 at large N, but C not on 16 threads. Thus the backends that make the C action–angle transformation differentiable are also the fastest way to evaluate it for large samples when a mod- ern GPU is available.
**JB:** Again, we need to discuss jitted results on the GPU. Or are they not faster?

### [112] p.19L Highlight
> Figure 17 shows the speed of the one-dimensional in- verse transformation. Evaluating the torus map at N angles is fastest with at every N on one CPU numpy core: compiled ties it for ten angles and is two to jax eight times slower for more angles (49 s against 10.9 s at 106), N = while for many tori the compiled over vmap tori is about 1.5 times faster than the loop over numpy tori. On a GPU the picture reverses: on the H100 the 106 evaluation at N = takes 11 ms, about a thousand times faster than numpy. The backends find the angle with the same safeguarded Newton iteration as numpy, started at the requested angle. Constructing the map for a family of nE energies is faster on every backend than on numpy, by a factor of three to ten at nE = 256, because the batched Gauss–Newton solve replaces
**JB:** Maybe we should drop this figure and its discussion and keep it for the in prep paper. You're not working on that, so message the fast-orbits-develop agent about this, that agent is working on that (they may be asleep). This will shorten the paper a bit which is good

### [113] p.19R Highlight
> it is done by inverse transformation,
**JB:** Just checking that this caveat is still necessary?

### [114] p.19R Highlight
> and construction is differentiable for the DFs whose construction is it- self a numerical computation that could be migrated— the spherical DFs obtained by inverting a density, the § King model, and the stream models of 7—and re- mains a computation for the others, chiefly the numpy quasi-isothermal disk DF, whose construction tabulates a guiding-center relation with scipy.
**JB:** Isn't this outdated? Are there any DFs that can't be constructed on the backends?

### [115] p.19R Highlight
> Figure 19
**JB:** Again, figures should be discussed in order: Figure 18 has not been discussed yet. I'm also wondering why there is a discussion of the spherical DF results already when this is not yet the sub-section on spherical DFs. I think you are trying to illustrate a general point about DFs

### [116] p.19R Highlight
> We checked
**JB:** Maybe "Figure 18 shows an explicit check that..." (and then drop the (Figure 18) later in that sentence.

### [117] p.20L Highlight
> +z turning points
**JB:** Not ideal that it spills out of the box

### [118] p.20R Text
**JB:** I'm still not happy about this legend overflowing so much. Maybe the info in it needs to be re-thought to shrink it, maybe through not mentioning h or just using a single h for all so it can be stated in the caption

### [119] p.21L Highlight
> dotted: 1 ms–1 s per call
**JB:** Not consistent with other similar legends, which normally always mention all of the numbers for each line explicitly

### [120] p.21R Text
**JB:** No lines of constant time per point?

### [121] p.21R Text
**JB:** Much better line-labeling here than in some other plots, more like this please!

### [122] p.21R Highlight
> integrand is now obtained from the generic force routine rather than a private one, which reorders floating-point operations.
**JB:** I think this has changed again after the PR where we reinstated rforce_jax for speed?

### [123] p.23L Highlight
> Not everything is differentiable:
**JB:** Why not??? Certainly the power-law DFs should be on the backend (I assumed they were) and I don't know what this energy derivative stuff is about, but it sounds like it should be on the backend!

### [124] p.23L Highlight
> The two-dimensional disk DFs of W.
**JB:** Did you also bring the schwarzschilddf onto the backend? You should have!

### [125] p.23L Highlight
> J. Binney (2010)
**JB:** I've re-written this a bit, but even in the re-write you need to cite this DF better, because it also requires citations to some Binney and McMillann papers from 2011/2012 I think that refined the 2010 proposal

### [126] p.23R Highlight
> Its construction is a computation, numpy
**JB:** Is this still the case? I know the many position sampler is on numpy

### [127] p.23R Highlight
> We did not mi- grate the corrections of the disk DFs for the difference between the input and output profiles.
**JB:** This should be disucssed in the 2D paragraph (I've split this section into two paragraphs)

### [128] p.24L Highlight
> compiled is faster than jax numpy at every size,
**JB:** Why is it that this is so much faster? It's just a computation that is suitable to the GPU?

### [129] p.24R Highlight
> Tidal streams are the application for which galpy’s DFs are most often fit to data,
**JB:** I don't think this is true: are there many papers that fit galpy tidal stream models to data? I don't know of many. So maybe we should start this section in a different way

### [130] p.25R Text
**JB:** Maybe this should sit between the panels and then connect to the lines on each side panel. And do the same in the top row

### [131] p.26R Highlight
> sample from the model’s DF, the derivative of the track with respect to a parameter can be taken either with the sample of stars held fixed, which is what one obtains by differentiating the track code alone, or by letting the sample move with the parameter, which is what a finite difference of the full model computes. The two differ, and the first is not a physical quantity; where galpy provides derivatives of stream models, they are of the second kind, and we test them against finite differences whose step is shown to be converged.
**JB:** This is all a good point, but it just seems to float here for no reason, because it is not followed by an example of a sample or something. So maybe it should be moved to the particle-spray section where this type of derivative is discussed more?

### [132] p.26R Highlight
> streamTrack()
**JB:** explain very briefly what this does, because it's also a relatively new feature

### [133] p.26R Highlight
> pot=lp,tdisrupt=td,
**JB:** we should probably add arm='leading' or whatever even if leading is the default, to make it clear how the arm is defined

### [134] p.27L Text
**JB:** This just shows the track points without trying to interpolate between them the way streamdf does to get a smooth track? It would be nicer to have a smooth track

### [135] p.27L Highlight
> the 6000-particle stream of Figure 23.
**JB:** I think Figure 23 should be introduced in a bit less of an off-handed manner

### [136] p.27R Highlight
> 10 µs on the H100 per particle (Figure 24).
**JB:** Again, it's a bit surprising that this is so much faster especially since orbit integration is such a big part of the process. Do you think these results are consistent with the speed of orbit integration with the different backends / GPUs etc? I suppose it must be!

### [137] p.27R Text
**JB:** add "of \texttt{streamdf}"

### [138] p.27R Highlight
> The construction of the model also runs on the backends and can be compiled: with a backend progenitor and an that integrates with actionAngleIsochroneApprox diffrax, every
**JB:** Also possible for torch I assume?

### [139] p.29R Highlight
> order and the track is re-fitted.
**JB:** add a short conclusion about that (that it doesn't work as well, although from the figure not so much worse either)

### [140] p.38L Highlight
> https://github.com/jobovy/galpy-jax .
**JB:** As I mentioned above, we should rename to galpy-backends. I should probably do that...
