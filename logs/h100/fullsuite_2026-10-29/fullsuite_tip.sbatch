#!/bin/bash
# Full galpy CUDA test suite for one backend on one H100 (not a repo stage).
#   sbatch --account=def-bovy --time=16:00:00 --export=ALL,BK=torch fullsuite.sbatch
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --job-name=fullsuite
set -u
source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate galpy-h100
G=$SCRATCH/galpy-h100; WT=$G/galpy-fb-tip; OUT=$G/logs/fullsuite_2026-10-29; mkdir -p $OUT
export PYTHONPATH=$WT XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONDONTWRITEBYTECODE=1 \
       XDG_CACHE_HOME=$G/cache MPLCONFIGDIR=$G/cache/mpl TORCHINDUCTOR_CACHE_DIR=$G/cache/inductor \
       TRITON_HOME=$G/cache TRITON_CACHE_DIR=$G/cache/triton CUDA_CACHE_PATH=$G/cache/nv \
       LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
cd $WT
{ nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; echo "galpy $(git rev-parse HEAD)"
  python -c "import jax, torch; print('jax', jax.__version__, 'torch', torch.__version__)"; date; } > $OUT/$BK.env
python -m pytest -p no:cacheprovider -n 2 --backend $BK --device cuda -rA --tb=short \
  --junitxml=$OUT/$BK.xml --ignore=tests/test_amuse.py --ignore=tests/test_snapshotpotential.py \
  tests/ > $OUT/$BK.txt 2>&1
echo "pytest exit $? at $(date)" >> $OUT/$BK.env
