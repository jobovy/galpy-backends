#!/bin/bash
# symplec6_jax GPU at N=1e6 (recheck), 3e6, 1e7; protocol of job 1184895 (galpy 00527ee63), measure script copy with only Ns changed
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --job-name=symp6c
set -u
G=$SCRATCH/galpy-h100; WT=$G/galpy-symp; D=$G/jobs/symp6c; O=$G/logs/symp6c; mkdir -p $O
source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate galpy-h100
export XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
       XDG_CACHE_HOME=$G/cache MPLCONFIGDIR=$G/cache/mpl CUDA_CACHE_PATH=$G/cache/nv LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
unset XLA_FLAGS
{ echo "galpy $(git -C $WT rev-parse HEAD) dirty=[$(git -C $WT status --porcelain -uno)]  script from galpy-backends $(cat $D/SOURCE_COMMIT)  host $(hostname) job $SLURM_JOB_ID $(date)"
  echo "== nvidia-smi before"; nvidia-smi --query-gpu=name,driver_version,memory.used,clocks.sm,power.draw --format=csv; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
  nvidia-smi --query-gpu=timestamp,clocks.sm,memory.used,power.draw,clocks_throttle_reasons.active --format=csv -l 2 > $O/smi.csv & SMI=$!
  cd $D && GALPY_WORKTREE=$WT PYTHONPATH=$WT python galpy_symplec_measure.py gpu lowN; echo "exit $?"
  kill $SMI; cp $D/data/symplec6_jax_gpu.json $O/symplec6_jax_gpu.hiN_extension.json
  echo "== nvidia-smi after"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv; date; } > $O/run.log 2>&1
