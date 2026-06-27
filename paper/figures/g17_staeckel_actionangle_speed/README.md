# g17 — Staeckel action-angle speed (split by backend)

This directory holds the shared **measurement + data** (`make.py`, `data/`): one
`_measure()` runs all backends, and `make.py` registers **two** figures (too many
curves for one panel). The rendered figures live in sibling dirs:

- `../g17_staeckel_actionangle_speed_jax/`   — jax (eager vs jit, CPU & GPU) vs C/numpy
- `../g17_staeckel_actionangle_speed_torch/` — torch (eager, CPU & GPU) vs C/numpy

Re-render: `python -m paper.figures._harness run g17_staeckel_actionangle_speed_jax g17_staeckel_actionangle_speed_torch`
Re-measure (GPU box): delete `data/*.json` and re-run.
