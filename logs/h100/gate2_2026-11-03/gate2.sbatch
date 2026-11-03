#!/bin/bash
# Gate (galpy PR #1797 7b2f84c88) re-measure of the compiled-jax GPU orbit lines (g8/g12 shared jax, g12 Dopri5, symplec6) + g15 diffrax GPU
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --job-name=gate2
set -u
source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate galpy-h100
G=$SCRATCH/galpy-h100; WT=$G/galpy-gate; R=$G/galpy-jax; J=$G/jobs/gate2; O=$G/logs/gate2; mkdir -p $O
F=paper/figures; SD=$F/_shared/orbit_speed/data; GD=$F/g12_inbackend_orbit_gpu/data
unset OMP_NUM_THREADS XLA_FLAGS
export PYTHONPATH=$WT GALPY_WORKTREE=$WT XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
       XDG_CACHE_HOME=$G/cache MPLCONFIGDIR=$G/cache/mpl CUDA_CACHE_PATH=$G/cache/nv LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
cd $R
step() { echo ">>> $(date +%T) $*"; "$@"; echo "<<< exit $? $(date +%T)"; }
{ echo "galpy $(git -C $WT rev-parse HEAD) dirty=[$(git -C $WT status --porcelain -uno)]  galpy-backends $(git rev-parse HEAD)  host $(hostname) job $SLURM_JOB_ID cpus $(python3 -c 'import os;print(len(os.sched_getaffinity(0)))') $(date)"
  nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
  step python $F/_shared/orbit_speed/galpy_jax_measure.py gpu lowN && cp $SD/jax_gpu.json $SD/h100/jax_gpu.json
  step python $F/_shared/orbit_speed/galpy_jax_measure.py gpu hiN
  step python $F/_shared/orbit_speed/galpy_jax_measure.py gpu hiN --Ns 1e6,3e6 && cp $SD/jax_gpu_hiN.json $SD/h100/jax_gpu_hiN.json
  step python $F/_shared/orbit_speed/galpy_symplec_measure.py gpu && cp $SD/symplec6_jax_gpu.json $O/symplec6_jax_gpu.default_grid.json
  (cd $J/symp_hi && step python galpy_symplec_measure.py gpu lowN) && cp $J/symp_hi/data/symplec6_jax_gpu.json $O/symplec6_jax_gpu.hi_3e6_1e7.json
  step python $F/g12_inbackend_orbit_gpu/measure/jax_dopri5_measure.py gpu lowN && cp $GD/jax_dopri5_gpu.json $GD/h100/jax_dopri5_gpu.json
  step python $F/g12_inbackend_orbit_gpu/measure/jax_dopri5_measure.py gpu hiN
  step python $F/g12_inbackend_orbit_gpu/measure/jax_dopri5_measure.py gpu hiN --Ns 1e6,3e6 && cp $GD/jax_dopri5_gpu_hiN.json $GD/h100/jax_dopri5_gpu_hiN.json
  step python $F/g15_icderiv_speed/measure/jax_gpu_only.py $F/g15_icderiv_speed/data/h100/icderiv.json
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv; date; } > $O/run.log 2>&1
