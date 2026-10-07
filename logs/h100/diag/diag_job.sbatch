#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --job-name=g17-diag
set -u
source "$CONDA_SH" && conda activate "$CONDA_ENV"
G=$SCRATCH/galpy-h100; cd $G/galpy-jax
export PYTHONPATH=$G/galpy-fb:$G/galpy-jax XLA_PYTHON_CLIENT_PREALLOCATE=false XDG_CACHE_HOME=$G/cache MPLCONFIGDIR=$G/cache/mpl \
       TRITON_HOME=$G/cache TRITON_CACHE_DIR=$G/cache/triton CUDA_CACHE_PATH=$G/cache/nv PYTHONDONTWRITEBYTECODE=1 LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
for m in torch jax seq; do python $G/diag/g17_gpu_oom.py $m 2>&1 | grep -E "^(mode|torch|jax|before)"; done
