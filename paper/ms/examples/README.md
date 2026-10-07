# Runnable, tested code examples of the paper

Every code snippet in `../sections/*.tex` (the `\begin{code}` blocks and the
one `\begin{codefig}` listing) has a complete script here, named
`<section>_<NN>_<topic>.py` in the order in which the snippets appear. Each
script defines what the paper leaves undefined (imports, `ts`, `ic`, `pot`,
...) with small sizes, runs the snippet **verbatim** between the markers

    # --- paper snippet begins ---
    ...
    # --- paper snippet ends ---

and then asserts what the surrounding text claims (backend array types and
devices, derivatives against finite differences, the C state-transition-matrix
path being used, ...). A script prints `OK` at the end when all checks pass.

`SNIPPETS.md` lists, for every snippet, the current text in the .tex, the
replacement text (identical to what the script runs), and what changed and why.
It is generated from the scripts and the .tex by `python make_snippets_md.py`
(run it before the .tex is updated: it pairs each script with the CURRENT text).

## Running

Requirements: galpy with the backends, jax (+ diffrax, optax), torch (+
torchode, torchdiffeq), pytest. Run on the CPU in double precision:

    cd paper/ms/examples
    CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONPATH=/path/to/galpy \
        python -m pytest -q tests_paper_examples.py

`tests_paper_examples.py` runs each script in a fresh interpreter (the scripts
set jax's x64 flag and some patch galpy to count calls) and checks that every
snippet line fits the 56-character column. The one example that needs a GPU
(`design_03_cuda_orbit.py`, `device='cuda'`) is skipped when CUDA is not
available; run it on a GPU machine without `CUDA_VISIBLE_DEVICES=`. A single
example runs as a plain script, e.g. `python orbits_03_cstm_jacobian.py`.

The whole set takes about 25 minutes on 12 CPU cores; the slowest examples
(the stream models, the qdf sampler, the King-model derivative under
`jax.jit`) take one to four minutes each.

Some scripts also check a known galpy defect and fail loudly when it is fixed
(so the paper text can be updated): the qdf sample derivative being NaN for
`MWPotential2014` (`dfs_04`), and DF construction with a traced parameter
needing `galpy.backend.use('jax')` (`dfs_02`, `dfs_03`), numpy times with
a CUDA IC failing under torchode (`design_03`, printed only), and the fitted stream track not
being smooth in the potential (`streams_03`).
