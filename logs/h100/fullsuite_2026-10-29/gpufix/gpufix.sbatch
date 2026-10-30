#!/bin/bash
# PR #1780 (71721b13b) device-regression re-run: torch then jax, --device cuda, test ids from jobs/gpufix/ids_<bk>.txt
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --job-name=gpufix
set -u
source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate galpy-h100
G=$SCRATCH/galpy-h100; WT=$G/galpy-gpufix; OUT=$G/logs/fullsuite_2026-10-29/gpufix; mkdir -p $OUT
export PYTHONPATH=$WT XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONDONTWRITEBYTECODE=1 \
       XDG_CACHE_HOME=$G/cache MPLCONFIGDIR=$G/cache/mpl TORCHINDUCTOR_CACHE_DIR=$G/cache/inductor \
       TRITON_HOME=$G/cache TRITON_CACHE_DIR=$G/cache/triton CUDA_CACHE_PATH=$G/cache/nv \
       LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
cd $WT
for BK in torch jax; do
  { nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; echo "galpy $(git rev-parse HEAD) dirty=[$(git status --porcelain -uno)]"
    python -c "import jax, torch; print('jax', jax.__version__, 'torch', torch.__version__, jax.devices())"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv; date; } > $OUT/gpufix_$BK.env 2>&1
  python -m pytest -p no:cacheprovider -n 2 --backend $BK --device cuda -rA --tb=short --junitxml=$OUT/gpufix_$BK.xml @$G/jobs/gpufix/ids_$BK.txt > $OUT/gpufix_$BK.txt 2>&1
  echo "pytest exit $? at $(date)" >> $OUT/gpufix_$BK.env
done
