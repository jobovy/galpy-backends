#!/bin/bash
# galpy backend tests on one H100: RUN=<label> BK=<torch|jax> SPEC=<file with pytest args> (not a repo stage)
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --job-name=gaptest
set -u
G=$SCRATCH/galpy-h100; WT=$G/galpy-gap0i; OUT=$G/logs/gaptest_2026-10-16; mkdir -p $OUT
source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate galpy-h100
L=${SLURM_TMPDIR:-/tmp}/gaptest_$SLURM_JOB_ID; rm -rf $L; mkdir -p $L/inductor $L/triton
export PYTHONPATH=$WT GALPY_WORKTREE=$WT XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
       TORCHINDUCTOR_CACHE_DIR=$L/inductor TRITON_CACHE_DIR=$L/triton TRITON_HOME=$L CUDA_CACHE_PATH=$L/nv \
       XDG_CACHE_HOME=$G/cache MPLCONFIGDIR=$G/cache/mpl LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
unset XLA_FLAGS
cd $WT
E=$OUT/$RUN.env.txt
{ echo "run $RUN backend $BK job $SLURM_JOB_ID host $(hostname) $(date)"; echo "galpy $(git rev-parse HEAD) dirty=[$(git status --porcelain -uno)]"
  echo "node-local cache $L"; pip list 2>/dev/null | grep -iE "torch|jax|diffrax|torchdiffeq|torchode|numpy|scipy|pytest"
  python -c "import jax, torch; print('jax.devices()', jax.devices()); print('torch', torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
  echo "== nvidia-smi before"; nvidia-smi; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv; } > $E 2>&1
python -m pytest -p no:cacheprovider --timeout 1800 -W error::DeprecationWarning -W error::FutureWarning -n 2 -rA --tb=short \
  --backend $BK --device cuda --junitxml=$OUT/$RUN.xml $(cat $SPEC) > $OUT/$RUN.log 2>&1
echo "pytest exit $? at $(date)" >> $E
{ echo "== nvidia-smi after"; nvidia-smi; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv; } >> $E 2>&1
rm -rf $L
