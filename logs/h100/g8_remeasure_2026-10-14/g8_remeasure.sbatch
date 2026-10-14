#!/bin/bash
# g8 H100 re-measure: galpy jax (lowN, hiN) then galax default (modern, extended) then galax per-orbit (lowN, hiN), back to back
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --job-name=g8-remeasure
set -u
G=$SCRATCH/galpy-h100; R=$G/galpy-jax; OUT=$G/logs/g8_remeasure_2026-10-14; mkdir -p $OUT /var/tmp/galax-bench
smi() { echo "== nvidia-smi $1 $(date +%T)"; nvidia-smi; nvidia-smi --query-gpu=name,driver_version,memory.used,utilization.gpu --format=csv; echo "compute apps:"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv; }
export XLA_PYTHON_CLIENT_PREALLOCATE=false OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
       XDG_CACHE_HOME=$G/cache MPLCONFIGDIR=$G/cache/mpl CUDA_CACHE_PATH=$G/cache/nv
echo "host $(hostname)  cpu_affinity $(python3 -c 'import os; print(len(os.sched_getaffinity(0)))')"
smi before
# 1. galpy jax GPU
source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate galpy-h100
export PYTHONPATH=$G/galpy-fb:$R GALPY_WORKTREE=$G/galpy-fb LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
echo "galpy $(git -C $G/galpy-fb rev-parse HEAD) dirty=[$(git -C $G/galpy-fb status --porcelain -uno)]  galpy-jax $(git -C $R rev-parse HEAD)"
cd $R
for m in lowN hiN; do echo ">>> $(date +%T) galpy jax gpu $m"; python paper/figures/_shared/orbit_speed/galpy_jax_measure.py gpu $m; echo "<<< exit $? $(date +%T)"; done
cp paper/figures/_shared/orbit_speed/data/jax_gpu.json paper/figures/_shared/orbit_speed/data/jax_gpu_hiN.json $OUT/
conda deactivate
smi "after galpy"
# 2-3. galax (its own uv venv; jax 0.8.0)
unset PYTHONPATH GALPY_WORKTREE
PY=$G/galax-src/.venv/bin/python; M=$R/paper/figures/g8_galax_orbit_speed/measure; cd $M
echo "galax $(git -C $G/galax-src rev-parse HEAD)  $($PY -c 'import jax; print("jax", jax.__version__, jax.devices())')"
echo ">>> $(date +%T) galax default modern"; GALAX_OUT=$OUT/galax_gpu.json $PY galax_measure_modern.py; echo "<<< exit $? $(date +%T)"
echo ">>> $(date +%T) galax default extended (hiN)"; $PY galax_extended.py; echo "<<< exit $? $(date +%T)"; cp /var/tmp/galax-bench/galax_gpu_hiN.json $OUT/ 2>/dev/null
for m in lowN hiN; do echo ">>> $(date +%T) galax perorbit $m"; $PY galax_perorbit_measure.py $m; echo "<<< exit $? $(date +%T)"; done
cp /var/tmp/galax-bench/galax_gpu_perorbit.json /var/tmp/galax-bench/galax_gpu_perorbit_hiN.json $OUT/ 2>/dev/null
smi after
ls -la $OUT; echo "done $(date)"
