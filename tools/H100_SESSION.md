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

## Where you are running: Trillium (Alliance / SciNet)

- **You run on the GPU login node** `trillium-gpu.alliancecan.ca`, inside the
  user's tmux. Login nodes have internet (for you, git, pip and conda);
  **compute nodes do not**, so you can never run on a compute node yourself.
  All GPU work goes to compute nodes as batch jobs that you submit and poll.
  Don't run heavy computation on the login node: building galpy and installing
  packages is fine, but measurements and tests are not.
- **GPU nodes:** 4x H100 80 GB and 96 cores each. Jobs get **1 GPU** (a quarter
  node) or a whole node. Max walltime is 24 h. A one-GPU job shares the node
  with other users' jobs on the other GPUs; that's fine for us.
- **Filesystems:** `$HOME` is **read-only on compute nodes**. Put everything
  (clones, environment caches, outputs, logs) under `$SCRATCH/galpy-h100`;
  scratch files expire after 2 months, which is plenty.
  `tools/trillium_stage.sh` sets the cache variables (XDG, matplotlib,
  inductor) to scratch.
- **Python:** the user installed **miniconda**; use a conda environment (in
  `$HOME` is fine, since compute nodes can read it). Install jax and torch from
  PyPI with their pip-packaged CUDA libraries, which need only the node's
  driver. The Alliance wheelhouse (`pip install --no-index`, `avail_wheels`)
  is the alternative; don't mix the two in one environment.
- **Slurm account:** GPU jobs need `--account=<...>`. Find the user's with
  `sacctmgr show assoc user=$USER format=account%30` and ask the user which to
  use if there's more than one.
- **Batch jobs:** `tools/trillium_job.sbatch` runs one stage of
  `tools/trillium_stage.sh` on one H100. Submit one stage at a time, poll with
  `squeue -u $USER` / `sacct -j <id>`, read `$SCRATCH/galpy-h100/logs/<stage>.log`,
  and commit + push from the login node after each stage, so nothing is lost if
  a later one fails. A debugjob (`debugjob -g 1`, interactive, 60 min) is not
  usable from your tool, since it opens an interactive shell.

## Hard rules

1. **One measurement at a time on the GPU.** A batch job's GPU is exclusive
   to it, so never run two jobs that use the GPU concurrently, and don't run
   tests while a measurement stage runs. Timings from a shared GPU are useless.
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
6. Keep host-memory use modest: at most 4 pytest workers.
7. If you need something only the user can provide (a Slurm account choice, git
   credentials, a quota problem), stop and ask; don't work around it.

## Setup (on the login node)

```bash
mkdir -p $SCRATCH/galpy-h100 && cd $SCRATCH/galpy-h100
# galpy at the commit to measure: feat/backends' tip; RECORD the SHA.
# (The TITAN data was measured on feat/backends 9f6c9d6f7 plus galpy #1634-#1637
# and #1639; the first four are merged. If #1639 is still open, cherry-pick it.)
git clone -b feat/backends https://github.com/jobovy/galpy.git galpy-fb
git -C galpy-fb log --oneline -1
#   only if #1639 is not merged:  (cd galpy-fb && git fetch origin pull/1639/head && git cherry-pick FETCH_HEAD)
git clone https://github.com/jobovy/galpy-jax.git && git -C galpy-jax checkout -b gpu-h100

conda create -y -n galpy-h100 -c conda-forge python=3.13 gsl compilers
conda activate galpy-h100
pip install "jax[cuda12]" torch torchode torchdiffeq diffrax equinox \
    numpy scipy matplotlib astropy array-api-compat pytest pytest-xdist
(cd galpy-fb && python setup.py build_ext --inplace)   # gsl-config from the env
export CONDA_SH=$(conda info --base)/etc/profile.d/conda.sh CONDA_ENV=galpy-h100
```

The H100 is sm_90: current `jax[cuda12]` and CUDA-12 torch wheels are right.
Ignore the old-version pins in `paper/figures/_setup/`, which are for the
TITAN Xp's sm_61. jax can't see a GPU on the login node; check on the GPU in
the smoke stage.

## Running the stages

```bash
cd $SCRATCH/galpy-h100/galpy-jax
sbatch --account=<ACCOUNT> --time=00:20:00 --export=ALL,STAGE=smoke tools/trillium_job.sbatch
```

Then, after each one finishes and its log looks right (`grep FAILED`), submit
the next:

| stage | walltime to request | what |
|---|---|---|
| `smoke` | 00:20:00 | jax/torch see the H100; tiny CUDA test |
| `tests` | 02:00:00 | galpy's CUDA test suites (record counts + failures) |
| `g2g5g6` | 01:00:00 | g2, g5, g6 |
| `orbits` | 02:00:00 | GPU orbit speeds behind g8/g12; g12's Dopri5 + torch lines |
| `bench` | 01:00:00 | `benchmarks/gpu`, jax + torch on the GPU |
| `newfigs` | 04:00:00 | GPU series of gV2, gD3, gS5, gS4 |
| `g15` | 08:00:00 | g15 (includes CPU parts) |
| `g17` | 10:00:00 | g17 (one measurement, including CPU series) |

The walltimes are generous estimates from the TITAN box; H100 GPU parts will be
faster. If a stage times out, resubmit it with a longer walltime; every stage
starts its measurements afresh. After each stage, commit its changed
`data/*.json`, PDFs and logs to `gpu-h100` and push.

## Optional, if time allows: galax on the H100 for g8

g8 compares galpy's orbit integration with galax (github.com/GalacticDynamics/galax).
Recipe: `paper/figures/_setup/galax_gpu_setup.md`, without the sm_61 caveats.
Use galax main's own `uv.lock` (`uv sync --frozen --no-dev`), then add the
jax-cuda12 plugin **matching the locked jaxlib** with `--no-deps`. Then run
`paper/figures/g8_galax_orbit_speed/measure/galax_measure_modern.py`,
`galax_extended.py`, and `galax_perorbit_measure.py lowN` / `hiN` with that
environment's python, in a batch job of their own (a copy of
`tools/trillium_job.sbatch` that runs those four commands; build the galax
environment on the login node first, in `$SCRATCH`). They write to `/var/tmp/galax-bench/`; copy those JSONs
into `paper/figures/g8_galax_orbit_speed/data/`. Record galax's git SHA.

## Deliverables

1. On branch `gpu-h100`: all changed `data/*.json`, the re-rendered PDFs,
   `benchmarks/gpu/results_<date>/`, and `logs/h100/report.md` containing:
   - machine (GPU, driver, CUDA, CPU model), galpy SHA, jax/torch/diffrax
     versions, galax SHA if run;
   - the `tests` stage counts and failures;
   - per figure: measured OK, failed (with the error), or skipped (why);
   - a short table of headline GPU numbers next to the TITAN Xp values from
     `main`, e.g. g12 jax GPU ms/orbit at the largest N, g2 forces at N=1e7,
     gS5 jax GPU jit ms/particle, and gV2 jax GPU jit at N=1e6;
   - anything surprising (a GPU line slower than the TITAN's, OOM thresholds,
     numerical mismatches).
2. Commit with a clear message and `git push -u origin gpu-h100`. Do NOT merge.
   Then tell the user it's done; the other session folds the data in.
