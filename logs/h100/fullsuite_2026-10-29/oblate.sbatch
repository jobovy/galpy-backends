#!/bin/bash
# #1779 delta: OblateStaeckel tests at feat/backends a61a7f31f2, --device cuda, torch then jax
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --job-name=oblate
set -u
source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate galpy-h100
G=$SCRATCH/galpy-h100; WT=$G/galpy-fb-a61; OUT=$G/logs/fullsuite_2026-10-29; mkdir -p $OUT
export PYTHONPATH=$WT XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONDONTWRITEBYTECODE=1 \
       XDG_CACHE_HOME=$G/cache MPLCONFIGDIR=$G/cache/mpl TORCHINDUCTOR_CACHE_DIR=$G/cache/inductor \
       TRITON_HOME=$G/cache TRITON_CACHE_DIR=$G/cache/triton CUDA_CACHE_PATH=$G/cache/nv \
       LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
cd $WT
for BK in torch jax; do
  { nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; echo "galpy $(git rev-parse HEAD) dirty=[$(git status --porcelain -uno)]"
    python -c "import jax, torch; print('jax', jax.__version__, 'torch', torch.__version__, jax.devices())"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv; date; } > $OUT/oblate_$BK.env 2>&1
  python -m pytest -p no:cacheprovider --backend $BK --device cuda -rA --tb=short --junitxml=$OUT/oblate_$BK.xml \
    tests/test_orbit.py tests/test_potential.py tests/test_backend_wrappers.py -k "oblatestaeckel or OblateStaeckel" > $OUT/oblate_$BK.txt 2>&1
  echo "pytest exit $? at $(date)" >> $OUT/oblate_$BK.env
done
