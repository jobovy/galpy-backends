#!/bin/bash
# Steps 3+4: earlier CUDA failures as a group + named files; XLA-flag nondeterminism check.
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --job-name=group-xla
set -u
source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate galpy-h100
G=$SCRATCH/galpy-h100; WT=$G/galpy-fb; OUT=$G/logs/fullsuite_2026-10-08; mkdir -p $OUT/group $OUT/xla
TL=$G/galpy-jax/logs/h100/triage
export PYTHONPATH=$WT GALPY_WORKTREE=$WT XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONDONTWRITEBYTECODE=1 \
       XDG_CACHE_HOME=$G/cache MPLCONFIGDIR=$G/cache/mpl TORCHINDUCTOR_CACHE_DIR=$G/cache/inductor \
       TRITON_HOME=$G/cache TRITON_CACHE_DIR=$G/cache/triton CUDA_CACHE_PATH=$G/cache/nv \
       LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
cd $WT
P="python -m pytest -p no:cacheprovider --device cuda -rA --tb=short"
echo "galpy $(git rev-parse HEAD)  $(date)"
$P --backend torch --junitxml=$OUT/group/torch_failed.xml $(cat $TL/torch_failed.txt) > $OUT/group/torch_failed.txt 2>&1; echo "torch_failed exit $?"
for i in 1 2 3; do $P --backend jax --junitxml=$OUT/group/jax_failed_$i.xml $(cat $TL/jax_failed.txt) > $OUT/group/jax_failed_$i.txt 2>&1; echo "jax_failed $i exit $?"; done
$P --backend torch --junitxml=$OUT/group/compile_workflows_inductor.xml tests/test_backend_compile_workflows.py -k torch-inductor > $OUT/group/compile_workflows_inductor.txt 2>&1; echo "inductor exit $?"
$P --backend torch --junitxml=$OUT/group/torchode.xml tests/test_backend_torchode.py > $OUT/group/torchode.txt 2>&1; echo "torchode exit $?"
$P --backend torch --junitxml=$OUT/group/kingdf_compile.xml tests/test_backend_kingdf.py -k "torchode or compile" > $OUT/group/kingdf_compile.txt 2>&1; echo "kingdf exit $?"
$P --backend torch --junitxml=$OUT/group/tracectx_forced.xml "tests/test_backend_tracectx.py::test_forced_backend_survives_a_fullgraph_compile" > $OUT/group/tracectx_forced.txt 2>&1; echo "tracectx exit $?"
for flag in "" "--xla_gpu_deterministic_ops=true" "--xla_gpu_autotune_level=0"; do
  for i in 1 2 3; do XLA_FLAGS="$flag" python $G/jobs/edf_grad.py >> $OUT/xla/edf_grad.txt 2>&1; echo "edf_grad [$flag] $i exit $?"; done
done
echo "done $(date)"
