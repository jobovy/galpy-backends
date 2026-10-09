"""Regenerate SNIPPETS.md: pairs each example with the CURRENT .tex snippet (run before the .tex is updated)."""
import glob, os, re, textwrap
MS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EX = MS + '/examples'
B, E = "# --- paper snippet begins ---", "# --- paper snippet ends ---"
ORDER = {  # example prefix -> tex file
    'design': 'design', 'potentials': 'potentials', 'orbits': 'orbits',
    'actionangle': 'actionangle', 'dfs': 'dfs', 'streams': 'streams',
    'examples': 'examples'}
NOTES = {  # relative to the CURRENT .tex
 'design_01': ('unchanged', 'Matches the .tex.'),
 'design_02': ('unchanged', 'Matches the .tex.'),
 'design_03': ('unchanged', 'Matches the .tex.'),
 'design_04': ('unchanged', 'Matches the .tex.'),
 'design_05': ('unchanged', 'Matches the .tex.'),
 'design_06': ('unchanged', 'Matches the .tex.'),
 'potentials_01': ('unchanged', 'Matches the .tex.'),
 'potentials_02': ('unchanged', 'Matches the .tex.'),
 'potentials_03': ('unchanged', 'Matches the .tex.'),
 'orbits_01': ('unchanged', 'Matches the .tex.'),
 'orbits_02': ('unchanged', 'Matches the .tex.'),
 'orbits_03': ('unchanged', 'Matches the .tex.'),
 'orbits_04': ('unchanged', 'Matches the .tex.'),
 'orbits_05': ('unchanged', 'Matches the .tex.'),
 'actionangle_01': ('unchanged', 'Matches the .tex.'),
 'actionangle_02': ('unchanged', 'Matches the .tex.'),
 'actionangle_03': ('unchanged', 'Matches the .tex.'),
 'actionangle_04': ('unchanged', 'Matches the .tex.'),
 'dfs_01': ('simplify', "The `galpy.backend.use('torch')` block is no longer needed: since galpy PR #1656 a differentiated potential parameter is the data for the constant-beta DF's $f(E)$ inversion and its sampler. Dropped and dedented; the derivative is unchanged (r_i/a to 1e-13)."),
 'dfs_02': ('simplify', "The `galpy.backend.use('jax')` block is no longer needed (galpy PR #1646: the King constructor follows the traced W0), and `sigmar` takes a Python-float radius under a traced parameter (galpy PR #1656), so `jnp.asarray(0.5)` becomes `0.5`. The jitted derivative is unchanged (matches a five-point finite difference to 2e-7, and the old snippet to 1e-12)."),
 'dfs_03': ('simplify', '`df.sigmar(jnp.asarray(1.))` becomes `df.sigmar(1.)`: since galpy PR #1656 the moments take a Python-float radius under a traced parameter. The derivative w.r.t. the potential\'s parameter also runs without a backend context and at a float radius (checked in the example), so "differentiable with respect to the parameters of the density" could again say "of both".'),
 'dfs_04': ('simplify', "The `galpy.backend.use('jax')` block is no longer needed (galpy PR #1646: the draw follows the key); the same draw, and its derivative w.r.t. the DF's parameters matches a finite difference for MWPotential2014 and MWPotential."),
 'streams_01': ('unchanged', 'Matches the .tex.'),
 'streams_02': ('simplify', "The progenitor can be set up from the list, `progenitor=Orbit(ic)` (galpy PR #1646: the numpy and backend samplers agree to ~1e-8), saving a line. `integrate_kwargs={'adjoint':'direct'}` stays: recent jax lets galpy select the direct adjoint under jvp itself (checked in the example), but jax 0.6, the last jax for Python 3.10, does not."),
 'streams_03': ('simplify', 'Same progenitor simplification as the previous snippet. The text after the snippet ("a finite difference of the fitted track does not converge onto it") no longer holds: finite differences of the same, traced fit (`jax.jit(track)`) with steps h = 1e-6 and 3e-7 agree with each other and with `jacfwd` (the fit\'s discrete choices change only for larger steps); the example now checks this instead of the non-convergence. (A finite difference of eager calls at float q does not converge because those take the NumPy fit, a different estimator.)'),
 'examples_01': ('unchanged', 'Matches the .tex.'),
 'examples_02': ('unchanged', 'Matches the .tex.'),
 'examples_03': ('unchanged', 'Matches the .tex.'),
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
 "Status: **simplify** = runs as is, but a workaround in it is no longer needed after a galpy fix; **WRONG** = the snippet failed or contradicted its claim and was fixed; **reformat** = layout only (lines > 56 characters, spacing, comments); **unchanged** = already fine.", ""]
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
lines += ["## Text around the snippets", "",
 "- dfs.tex (Spherical DFs): \"A DF built from a density that is not the one generating the potential is differentiable with respect to the parameters of the density\" can again say \"of both\": the derivative w.r.t. the potential's parameter runs without a backend context (galpy #1646) and at a float radius (galpy #1656).",
 "- streams.tex (Particle-spray models), after the `track` snippet: \"but this derivative holds the discrete choices of the track fit (its binning in stripping time and the knots of its spline) fixed, and a finite difference of the fitted track does not converge onto it. For a likelihood, smooth statistics of the particles themselves are the better target.\" The first half holds; the second does not: finite differences of the same fit with steps small enough that no discrete choice changes (h = 1e-6 and 3e-7 for this stream) agree with each other and with `jacfwd` (see `streams_03`). Suggested: \"where the derivative holds the discrete choices of the track fit (each particle's closest point on the progenitor's orbit and the amount of smoothing) fixed; it agrees with finite differences of the fitted track for steps small enough that none of these choices changes.\" (and drop the likelihood sentence, or keep it as advice about smoothness for optimizers).",
 "",
 ""]
open(f'{EX}/SNIPPETS.md', 'w').write('\n'.join(lines + body))
print(counts)
