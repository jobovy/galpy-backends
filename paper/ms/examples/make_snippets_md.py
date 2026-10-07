"""Regenerate SNIPPETS.md: pairs each example with the CURRENT .tex snippet (run before the .tex is updated)."""
import glob, os, re, textwrap
MS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EX = MS + '/examples'
B, E = "# --- paper snippet begins ---", "# --- paper snippet ends ---"
ORDER = {  # example prefix -> tex file
    'design': 'design', 'potentials': 'potentials', 'orbits': 'orbits',
    'actionangle': 'actionangle', 'dfs': 'dfs', 'streams': 'streams',
    'examples': 'examples'}
NOTES = {
 'design_01': ("reformat", "Long import split with parentheses; trailing comment shortened to fit 56 characters (the device claim stays in the text and is checked by the example)."),
 'design_02': ("WRONG", "An Orbit does not follow the `use()` context: `Orbit([floats])` inside `use('jax')` keeps a numpy IC and is integrated in C, returning numpy arrays. A potential evaluated at Python floats does follow the context (returns a jax array), so the snippet now shows that. The sentence before it (\"as when a user sets up an \\orb\\ instance from \\python\\ floats\") needs to change accordingly, e.g. \"as when a potential is evaluated at \\python\\ floats\"."),
 'design_03': ("WRONG", "Without `method=`, galpy's default `symplec4_c` does not route a concrete torch IC to the in-backend solver: the orbit is integrated in C on the host (with a warning), not with torchode on the GPU. Added `method='torchode'`; long line split. Also: `ts` must be a CUDA tensor here; with numpy times torchode raises (torchode's same_device assertion; galpy moves numpy times to the IC's device for torchdiffeq but not for torchode), a galpy defect the example records."),
 'design_04': ("unchanged", "Fits and runs as is."),
 'design_05': ("reformat", "The potential is built on its own line so the constructor fits; comments realigned."),
 'design_06': ("unchanged", "Fits and runs as is; the jitted gradient matches a five-point finite difference to 5e-9."),
 'potentials_01': ("unchanged", "Fits and runs as is; matches the analytic derivative to 1e-14."),
 'potentials_02': ("reformat", "The `Rforce(...).backward()` line was 57 characters; split at the comma."),
 'potentials_03': ("reformat", "codefig: the `__init__` signature (57 characters) split over two lines, and the potential of the last lines built on its own line (the `integrate` call was 58 characters). Runs as is; d z(t_end)/d a matches a five-point finite difference of the C orbit in galpy's own MiyamotoNagaiPotential to 1.5e-10."),
 'orbits_01': ("reformat", "The potential is built on its own line (`pp=`, as in the torch snippet) so the `integrate` call fits."),
 'orbits_02': ("unchanged", "Fits and runs as is (with `b` a float32 tensor, the derivative is float32-accurate: 5e-7 relative)."),
 'orbits_03': ("reformat", "The `integrate` call split and the comments shortened. Runs as is: the C state-transition-matrix path is used, and the Jacobian agrees with a finite difference of tightly converged C orbits to 3e-10."),
 'orbits_04': ("reformat", "Comment shortened to fit on one line. `jax.vmap` over the C-STM `jacrev` works and equals per-orbit `jacrev` to 1e-12."),
 'orbits_05': ("reformat", "Long calls split; the comment on the deviation vector dropped (it no longer fits)."),
 'actionangle_01': ("reformat", "The potential is built on its own line so the constructor fits on one line."),
 'actionangle_02': ("reformat", "Constructor and the jit/grad call split. Runs as is; the compiled derivative goes through a C callback and matches the c=False AD derivative to 1e-8."),
 'actionangle_03': ("reformat", "Long calls split at commas. Runs as is, and the same chain (orbit, delta, actions) runs inside `jax.jit`."),
 'actionangle_04': ("reformat", "The `actionAngleVerticalInverse` call (59 characters) split with a hanging indent; the disk fits on one line. The derivative matches a five-point finite difference that rebuilds the family to 3e-10 (the eager snippet takes ~50 s on one CPU core)."),
 'dfs_01': ("WRONG", "Two problems with the claim that the derivative is r_i/a to 1e-13: (1) with `rmin=0.1` the radii do not scale with `a` (the truncation radius is fixed), and the derivative differs from r_i/a by 2--4%; (2) `torch.tensor(1.,requires_grad=True)` is float32, so the derivative is float32 (5e-8). Without `rmin` and with a float64 `a`, d(sum r_i)/da = sum r_i/a to 2e-16, per particle to 1e-13. The potential is built on its own line."),
 'dfs_02': ("WRONG", "As written, `kingdf(W0=<tracer>)` fails both eagerly and under `jax.jit` (TracerArrayConversionError): the constructor evaluates its potential at Python floats (`self._pot(self.rt,0.)`), which go to numpy and meet the traced W0. Inside `galpy.backend.use('jax')` both work. The eager line was dropped because it takes ~2 min on a CPU (the jitted one ~1 min, nearly all compilation); the jitted derivative matches a five-point finite difference of the numpy model to 2e-7."),
 'dfs_03': ("reformat", "The tracer is built on its own line. Runs as is. But the text's claim that the DF is differentiable w.r.t. the parameters of BOTH densities only holds inside `galpy.backend.use('jax')`: d/d(potential parameter), e.g. NFW `a` or `amp`, fails without it (the constructor evaluates the potential at r=0 as a Python float). With the context, it matches a finite difference to 4e-10."),
 'dfs_04': ("WRONG", "`qdf.sampleV(...,key=<jax key>)` on a qdf built from Python floats raises a TypeError (numpy and jax arrays mixed in the inverse-CDF draw) unless it runs inside `galpy.backend.use('jax')`; wrapped. The comment \"differentiable in qdf's parameters\" was removed: for `MWPotential2014` the derivative of the sample w.r.t. any DF parameter (hr, sr, sz, hsr, hsz) is NaN (jax_debug_nans points at `_cumcdf` in `_sampleV_icdf`); for `MWPotential` it is finite and matches a finite difference. This is a galpy defect; the text's claim needs a caveat or a fix in galpy."),
 'streams_01': ("WRONG", "`time_in_Gyr()` takes `(vo,ro)`: now `time_in_Gyr(220.,8.)`. Long lines split; the `integrate_method` with a hanging indent."),
 'streams_02': ("WRONG", "(1) `jax.jvp` through the default (checkpointed) diffrax adjoint raises \"can't apply forward-mode autodiff (jvp) to a custom_vjp function\": `integrate_kwargs={'adjoint':'direct'}` is required. (2) With `progenitor=Orbit(ic)` from a list the derivative disagrees with a finite difference of the same sample at 7e-5 (and the jvp primal with a plain call at 1.5e-5); with a jax progenitor, `Orbit(jnp.asarray(ic))`, it agrees to ~1e-6 as the text says. (3) `time_in_Gyr()` needs `(vo,ro)`; replaced by `td` as in the next snippet. The output is now unpacked as `xv,dxv` (shorter, and it is a sample, not a track)."),
 'streams_03': ("WRONG", "`streamTrack()` returns a `StreamTrack` object, which is not a jax pytree, so `jax.jacfwd(track)` cannot differentiate it; the function now returns the track evaluated at fixed progenitor times, `tr.x(tps)`. Same forward-mode fixes as the previous snippet (direct adjoint, jax progenitor). Caveat: the fitted track is not a smooth function of q (the closest-node assignment and the GCV smoothing level are discrete choices, frozen in the derivative but re-made at q+-h): a finite difference does not converge onto `jacfwd` (relative differences 1--3 for h=1e-3..1e-5, also with fixed smoothing and ntp). The example records this rather than checking a value."),
 'examples_01': ("reformat", "The coordinate-order comment moved to its own line and the `jacrev` call split. Runs as is: the C state-transition-matrix path is used, through an Orbit initialized from (ra,dec,d,pmra,pmdec,vlos), and the gradient matches a five-point finite difference to 1e-6."),
 'examples_02': ("reformat", "Only the potential needed splitting. `adam_descent` (a dozen lines of optax in the example) recovers q=0.8485 for a mock generated with q=0.85."),
 'examples_03': ("reformat", "Comments shortened and the return split over two lines in parentheses. Runs as is (with a small MLP class with an `apply(params,J,theta)` method); the gradient matches a finite difference along a random direction in weight space to 1e-7."),
}
WRONG_ORDER = []
def tex_snippets(texfile):
    s = open(f'{MS}/sections/{texfile}.tex').read()
    out = []
    for m in re.finditer(r'\\begin\{(code|codefig)\}\n(.*?)\\end\{\1\}', s, re.S):
        out.append((m.group(1), m.group(2).rstrip('\n'), s[:m.start()].count('\n') + 1))
    return out
def example_snippet(path):
    s = open(path).read()
    m = re.search(re.escape(B) + r"\n(.*?)\n[ ]*" + re.escape(E), s, re.S)
    return textwrap.dedent(m.group(1))
lines = ["# Paper code snippets: corrected and reformatted", "",
 "Each snippet of `sections/*.tex` (`\\begin{code}` blocks and the one `\\begin{codefig}`), its CURRENT text, and the REPLACEMENT text to paste between the `\\begin{...}`/`\\end{...}` lines. The replacement is exactly the snippet that `examples/<name>.py` runs (between the `# --- paper snippet begins/ends ---` markers), so it is tested by `tests_paper_examples.py`. Every replacement line is at most 56 characters, including indentation and comments.", "",
 "Status: **WRONG** = the snippet failed or contradicted its claim and was fixed; **reformat** = layout only (lines > 56 characters, spacing, comments); **unchanged** = already fine.", ""]
summary = []
body = []
counts = {}
for pre, tex in ORDER.items():
    files = sorted(glob.glob(f'{EX}/{pre}_[0-9][0-9]_*.py'))
    texs = tex_snippets(tex)
    assert len(files) == len(texs), (pre, len(files), len(texs))
    for f, (env, cur, lineno) in zip(files, texs):
        name = os.path.basename(f)[:-3]
        key = '_'.join(name.split('_')[:2])
        status, note = NOTES[key]
        counts[status] = counts.get(status, 0) + 1
        new = example_snippet(f)
        if status == 'unchanged':
            assert new.strip() == cur.strip(), name
        else:
            assert new.strip() != cur.strip(), name
        summary.append(f"| `{name}` | sections/{tex}.tex:{lineno} | {env} | {status} |")
        body += [f"## `{name}` — sections/{tex}.tex, line {lineno} (`{env}`)", "",
                 f"**{status}**: {note}", ""]
        if status == 'unchanged':
            body += ["Text (no change):", "", "```python", cur, "```", ""]
        else:
            body += ["Current:", "", "```python", cur, "```", "",
                     "Replacement:", "", "```python", new, "```", ""]
lines += ["| example | location | env | status |", "|---|---|---|---|"] + summary + [""]
lines += [f"Totals: " + ", ".join(f"{v} {k}" for k, v in counts.items()), ""]
lines += ["## Claims in the surrounding text that the examples contradict", "",
 "- design.tex (Choosing a backend): \"When there is no array to dispatch on, as when a user sets up an \\orb\\ instance from \\python\\ floats, the default backend is used\". False for Orbit: the IC stays numpy and is integrated in C inside `use('jax')`. True for potentials (and other functions) evaluated at Python floats.",
 "- design.tex: \"an \\orb\\ set up from a tensor on a GPU is integrated there\" holds only with an in-backend method (`method='torchode'`/`'torchdiffeq'`); the default method integrates in C on the host. With torchode the times must also be on the GPU.",
 "- dfs.tex (Spherical DFs): \"this derivative is $r_i/a$, which the sampler reproduces to $10^{-13}$\" holds without `rmin` (with `rmin=0.1` it is off by 2--4%) and for a float64 `a` (float32: 5e-8).",
 "- dfs.tex: the King d/dW0 and the Eddington d/d(potential parameter) run only inside `galpy.backend.use('jax')` (both eagerly and under `jax.jit`); without it, the constructors evaluate the potential at Python floats on numpy and fail on the traced parameter. \"differentiable with respect to the parameters of both\" therefore needs that qualifier (the tracer-parameter derivative works without it).",
 "- dfs.tex (Disk DFs): \"a sample is differentiable in the DF's parameters for a fixed key\": for `MWPotential2014` the derivative is NaN (any of hr, sr, sz, hsr, hsz); for `MWPotential` it is finite and correct. `sampleV(key=<jax key>)` also needs `use('jax')` when the qdf was built from floats.",
 "- streams.tex (Particle-spray models): \"The whole computation compiles and is differentiable with respect to every parameter\": forward mode (jvp/jacfwd) needs `integrate_kwargs={'adjoint':'direct'}`, and the 1e-6 agreement with finite differences needs a jax progenitor Orbit (a numpy progenitor gives 7e-5). `streamTrack()` returns an object, so it is differentiated through its accessors (e.g. `.x(tp)`); and \"The fitted track ... is obtained in the same way\" holds for running jacfwd but the result is the derivative with the track fit's discrete choices frozen, which a finite difference of the track does not reproduce (it does not converge). The 1e-6 agreement is a property of the particles, not of the fitted track.",
 "- No contradiction found for: the C-STM Jacobian (agrees with finite differences of tightly converged C orbits to 3e-10--1.3e-9, consistent with \"a few parts in $10^{10}$\"), `jax.vmap` over the C-STM, the c=True Staeckel derivative through C under `jax.jit`, `estimateDeltaStaeckel` inside `jax.jit`, the vertical-inverse torus derivative (3e-10 vs. finite differences), the HVS q fit (recovers 0.8485 for 0.85).",
 ""]
open(f'{EX}/SNIPPETS.md', 'w').write('\n'.join(lines + body))
print(counts)
