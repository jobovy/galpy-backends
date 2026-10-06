# Instructions for a Claude session: measure galpy-jax's speed figures on an H100

You are a Claude Code session on a machine with an NVIDIA H100. Your job is to
re-measure the **GPU timing series** of the galpy-jax paper figures on this GPU,
commit the measured data to a branch, and report. Another session (on the dev
box with a TITAN Xp) owns the figures and will fold your H100 series into them
as extra lines. You do NOT change figure code, galpy, or galpy-jax's `main`.

## Background (read this first)

- **galpy** (github.com/jobovy/galpy) is a galactic-dynamics library. Its
  `feat/backends` branch makes galpy run, differentiate and jit under **jax**
  and **PyTorch**, on CPU and GPU, alongside the original numpy/C code.
- **galpy-jax** (github.com/jobovy/galpy-jax) holds the paper's figures. Each
  figure lives in `paper/figures/<name>/`. Its `make.py` either measures the
  timings into `data/*.json` (machine-dependent, stamped with GPU name and
  library versions) or plots from that JSON (byte-stable). Some figures have
  standalone measure scripts in `paper/figures/<name>/measure/`.
- The existing GPU numbers are from a **TITAN Xp**, a consumer card that runs
  float64 at 1/32 of its float32 rate (~0.38 TFLOPS). galpy computes in
  float64. The H100 runs float64 at ~34 TFLOPS with ~6x the memory bandwidth,
  so the large-N GPU timings should drop by one to two orders of magnitude,
  while the small-N, launch- and dispatch-bound end changes little. That
  contrast is the point of this run.

## Hard rules

1. **The GPU must be otherwise idle for every measurement.** Check
   `nvidia-smi --query-compute-apps=pid --format=csv,noheader` is empty before
   each step. Never run two GPU jobs at once, including test suites; timings
   from a shared GPU are useless. `tools/rerun_figures_gpu.sh` waits for an
   idle GPU itself.
2. **jax must not preallocate the whole GPU:** `export
   XLA_PYTHON_CLIENT_PREALLOCATE=false` for everything.
3. Work on a **branch** `gpu-h100` of galpy-jax. Never push to galpy-jax `main`,
   and never push anything to galpy.
4. Don't edit figure code (`make.py`, `measure/*.py`). If a script fails
   because of the machine (paths, versions), make the smallest fix, commit it
   separately on the branch, and say so in the report. If it fails because of a
   galpy bug, record the traceback and move on; don't work around it silently.
5. Before killing a process, look up its PID in one command and kill it by PID
   in a separate command. `pkill -f <pattern>` / `pgrep -f` match your own
   shell, which carries the pattern in its command line.
6. Keep host-memory use modest: one measurement at a time, at most 4 pytest
   workers.

## Setup

```bash
# galpy at the commit to measure. Use feat/backends' tip and RECORD the SHA.
# (The TITAN data was measured on feat/backends 9f6c9d6f7 plus the fixes in
# galpy #1634-#1637 and #1639; the first four are merged, and #1639 may be by
# the time you run. If #1639 is still open, cherry-pick it: gammainc on CUDA.)
git clone -b feat/backends https://github.com/jobovy/galpy.git galpy-fb
cd galpy-fb && git log --oneline -1
# optional, only if PR #1639 is not merged yet:
#   git fetch origin pull/1639/head && git cherry-pick FETCH_HEAD
python setup.py build_ext --inplace     # needs GSL (libgsl-dev) and a C compiler
cd ..

git clone https://github.com/jobovy/galpy-jax.git && cd galpy-jax
git checkout -b gpu-h100

python -m venv ../venv && . ../venv/bin/activate
pip install "jax[cuda12]" torch torchode torchdiffeq diffrax equinox \
    numpy scipy matplotlib astropy array-api-compat pytest pytest-xdist
python -c "import jax, torch; print(jax.devices(), torch.cuda.get_device_name(0))"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

The H100 is sm_90: current `jax[cuda12]` and CUDA-12 torch wheels are right.
Ignore the old-version pins in `paper/figures/_setup/`; those are for the TITAN
Xp's sm_61.

## Step 0: smoke test, then the CUDA test suites (about 1 h)

```bash
cd ../galpy-fb
python -m pytest -q -n 2 --backend jax --device cuda tests/test_backend_force.py
python -m pytest -q -n 2 --backend torch --device cuda tests/test_backend_force.py
```

If both pass, also run the broader CUDA suites, which catch device bugs: every
hidden CPU/GPU mixing raises on CUDA. Note that `--device cuda` sets torch's
*default* device to cuda, which hides one class of bug, so also run
`tests/test_backend_special.py` and
`tests/test_backend_actionAngleVerticalInverse.py` without `--backend`. Their
GPU-only tests keep the default device on the CPU deliberately.

```bash
for bk in jax torch; do
  python -m pytest -q -n 2 --backend $bk --device cuda \
    tests/test_backend_*.py -p no:cacheprovider 2>&1 | tail -3
done
python -m pytest -q tests/test_backend_special.py tests/test_backend_actionAngleVerticalInverse.py -k cuda
```

Record pass/fail counts and every failure's test id and error line. Don't try
to fix galpy.

## Step 1: the existing timing figures (about 5-7 h)

From the galpy-jax root:

```bash
tools/rerun_figures_gpu.sh ../galpy-fb ../logs-h100
```

This runs, strictly serially: g2, g5, g6 (force/autodiff/jit throughput), g17
(Staeckel action-angle), the shared orbit-speed data behind g8/g12, g12, g15,
`benchmarks/gpu/run_all.sh` (writes `benchmarks/gpu/results_<date>/`), and
re-renders. It also re-measures their CPU series, on this machine's CPU. That's
fine: only the GPU series get folded in, and the CPU ones give context. Expect
some large-N points to run out of GPU memory; they are recorded as NaN, which
is intended. Look through `../logs-h100/*.log` for "FAILED" when it finishes.

g17's "numpy per-object OLD" series needs an old galpy build at
`/var/tmp/staeckel-old`. Without it that series is skipped. That's fine; say
so in the report.

## Step 2: the newer figures' GPU series (about 1-2 h)

These measure only their GPU series and merge them into the existing JSON.

```bash
P="PYTHONPATH=../galpy-fb:."
env $P python paper/figures/gV2_vertical_inverse_speed/make.py --devices gpu
env $P python paper/figures/gD3_spherical_df_speed/make.py --gpu
env $P python -m paper.figures.gS5_streamspray_speed.make --measure gpu
env $P python -m paper.figures.gS4_streamdf_speed.make --measure gpu
env $P python -m paper.figures._harness run gV2_vertical_inverse_speed \
    gD3_spherical_df_speed gS5_streamspray_speed gS4_streamdf_speed
```

## Step 3 (optional, if time allows): galax on the H100 for g8

g8 compares galpy's orbit integration with galax (github.com/GalacticDynamics/galax).
Recipe: `paper/figures/_setup/galax_gpu_setup.md`, without the sm_61 caveats.
Use galax main's own `uv.lock` (`uv sync --frozen --no-dev`), then add the
jax-cuda12 plugin **matching the locked jaxlib** with `--no-deps`. Then run
`paper/figures/g8_galax_orbit_speed/measure/galax_measure_modern.py`,
`galax_extended.py`, and `galax_perorbit_measure.py lowN` / `hiN` with that
environment's python. They write to `/var/tmp/galax-bench/`; copy those JSONs
into `paper/figures/g8_galax_orbit_speed/data/`. Record galax's git SHA.

## Deliverables

1. On branch `gpu-h100`: all changed `data/*.json`, the re-rendered PDFs,
   `benchmarks/gpu/results_<date>/`, and `logs/h100/report.md` containing:
   - machine (GPU, driver, CUDA, CPU model), galpy SHA, jax/torch/diffrax
     versions, galax SHA if run;
   - Step 0 test counts and failures;
   - per figure: measured OK, failed (with the error), or skipped (why);
   - a short table of headline GPU numbers next to the TITAN Xp values from
     `main`, e.g. g12 jax GPU ms/orbit at the largest N, g2 forces at N=1e7,
     gS5 jax GPU jit ms/particle, and gV2 jax GPU jit at N=1e6;
   - anything surprising (a GPU line slower than the TITAN's, OOM thresholds,
     numerical mismatches).
2. Commit with a clear message and `git push -u origin gpu-h100`. Do NOT merge.
   Then tell the user it's done; the other session folds the data in.
