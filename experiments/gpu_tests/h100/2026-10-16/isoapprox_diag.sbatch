#!/bin/bash
# h100_isoapprox_diag.py cpu cuda from the galpy 50df80f3b checkout (not a repo stage)
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --job-name=isoapprox-diag
set -u
G=$SCRATCH/galpy-h100; WT=$G/galpy-gap0i; OUT=$G/logs/gaptest_2026-10-16; mkdir -p $OUT
source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate galpy-h100
L=${SLURM_TMPDIR:-/tmp}/isodiag_$SLURM_JOB_ID; mkdir -p $L
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
       TORCHINDUCTOR_CACHE_DIR=$L/inductor TRITON_CACHE_DIR=$L/triton TRITON_HOME=$L CUDA_CACHE_PATH=$L/nv \
       XDG_CACHE_HOME=$G/cache MPLCONFIGDIR=$G/cache/mpl LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
unset XLA_FLAGS
cd $WT
{ echo "galpy $(git rev-parse HEAD) dirty=[$(git status --porcelain -uno)]  galpy-jax $(git -C $G/galpy-jax rev-parse HEAD)  host $(hostname)  $(date)"
  echo "== nvidia-smi before"; nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
  PYTHONPATH=. python $G/galpy-jax/experiments/gpu_tests/diag/h100_isoapprox_diag.py cpu cuda; echo "diag exit $?"
  echo "== nvidia-smi after"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv; date; } > $OUT/isoapprox_diag.txt 2>&1
rm -rf $L
