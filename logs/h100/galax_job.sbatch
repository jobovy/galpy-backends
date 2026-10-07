#!/bin/bash
# g8 galax GPU measurements on one H100 (copy of tools/trillium_job.sbatch for the galax env).
#   sbatch --account=def-bovy --time=01:00:00 galax_job.sbatch
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --job-name=galax-h100
set -u
G=$SCRATCH/galpy-h100; PY=$G/galax-src/.venv/bin/python; M=$G/galpy-jax/paper/figures/g8_galax_orbit_speed/measure
LOG=$G/logs/galax.log; OUT=$G/galax-bench; mkdir -p $OUT /var/tmp/galax-bench
unset PYTHONPATH; export XLA_PYTHON_CLIENT_PREALLOCATE=false XDG_CACHE_HOME=$G/cache MPLCONFIGDIR=$G/cache/mpl PYTHONDONTWRITEBYTECODE=1
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
cd $M
run() { echo ">>> $(date +%T) [galax] $*"; ( eval "$@" ) >> $LOG 2>&1 && echo "    ok $(date +%T)" || echo "!!! FAILED [galax] $* (see $LOG)"; cp /var/tmp/galax-bench/*.json $OUT/ 2>/dev/null; }
run "$PY -c 'import jax; print(jax.devices())'"
run "$PY galax_measure_modern.py"
run "$PY galax_extended.py"
run "$PY galax_perorbit_measure.py lowN"
run "$PY galax_perorbit_measure.py hiN"
echo ">>> $(date +%T) [galax] done"
