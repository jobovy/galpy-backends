# galpy-backends

This repository accompanies the paper

> Bovy, J. (2026), *galpy with differentiable, accelerated backends: galactic
> dynamics in jax and PyTorch* (arXiv link to follow)

It contains the manuscript, the code and data behind every figure in it, and
the working notes, plans, and logs of the development of
[galpy](https://github.com/jobovy/galpy)'s jax and PyTorch backends. The
backends themselves are part of galpy: they are developed on galpy's `v2`
branch and released as galpy 2.0 pre-releases (`pip install --pre galpy`).

The backends were built by Claude, Anthropic's AI coding agent, working in
Claude Code under JB's direction, as described in the paper's section "How
the backends were built". Most of the notes and logs in this repository were
written by the agent as the work progressed.

## The paper

`paper/ms/` holds the manuscript (AASTeX 7). Build it with `make` in that
directory; the code listings use `minted`, so this needs `pdflatex
-shell-escape` and `pygments`.

## The figures

Each figure has a directory `paper/figures/<name>/` with a `make.py` that
draws it. Figures that report measurements (timings, GPU runs) are drawn from
the data in `data/`, produced by the measurement code next to it (`measure/`,
`measure_*.py`, `make.py` itself, or `paper/figures/_shared/` for the shared
orbit timings) on the machines described in the paper; redrawing them is fast
and byte-reproducible, while re-measuring needs comparable hardware. The neural-network figures
(`g19_neural_aa_inverse`, `g20_neural_staeckel_inverse`) have their own
training and plotting scripts, described in their `README.md`.

From the top of the repository, with galpy 2.0, `jax`, `diffrax`, `PyTorch`,
and `matplotlib` installed:

```
python -m paper.figures._harness list             # the figures
python -m paper.figures._harness run <name>       # (re)draw a figure
python -m paper.figures._harness verify [<name>]  # redraw and compare to the committed PDF
```

`paper/figures/_harness/manifest.json` records, for every figure, the galpy
commit and package versions it was made with and the SHA-256 of its PDF.
`plan/trackG_figures.md` describes the figure harness.

## The development record

These are working documents. They record plans that changed, dead ends, and
intermediate results that were later revised, and their paths refer to the
machines on which the work was done. The paper and galpy's documentation
describe the final state.

- `master_plan.md`, `00_overview.md`, `galpy_backend_autodiff_discussion.md`:
  the initial plan and the design discussion that settled its scope.
- `plan/`: the per-track plans, and audits such as `BACKEND_GAP_AUDIT.md` and
  `ISLAND_AUDIT.md`.
- `logs/`: dated logs of the work, per pull request or per stretch of work.
  `logs/vacation_2026-07/log.md` is the agent's log of the summer (about
  60,000 lines), including the two unsupervised weeks.
- `STATUS.md`: the running project status and pull-request tables kept during
  the development.
- `RESTART.md`, `RESUME_2026-07-09.md`, `HANDOFF_staeckel_quadrature.md`,
  `pr4_smoother_recipe.md`: hand-off notes written between agent sessions.
- `experiments/`, `benchmarks/`, `prototypes/`, `prepared/`: experiments
  (e.g., a hand-written CUDA orbit integrator), GPU benchmarks, and
  prototypes.
- `appendices/`: technical notes (special functions, test tolerances,
  quadrature) and the scripts behind them.
- `tools/`: instructions and scripts for the runs on an H100 GPU cluster, and
  maintenance scripts.

## License

MIT; see `LICENSE`.
