#!/bin/bash
# galpy PR #1658 (backend/cuda-backend-tests e402a035b) checks on one H100 (worktree galpy-pr1658)
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --job-name=pr1658
set -u
source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate galpy-h100
G=$SCRATCH/galpy-h100; WT=$G/galpy-pr1658; OUT=$G/logs/pr1658; mkdir -p $OUT/item1 $OUT/item2 $OUT/item3 $OUT/item4
unset XLA_FLAGS
export PYTHONPATH=$WT GALPY_WORKTREE=$WT XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
       XDG_CACHE_HOME=$G/cache MPLCONFIGDIR=$G/cache/mpl TORCHINDUCTOR_CACHE_DIR=$G/cache/inductor \
       TRITON_HOME=$G/cache TRITON_CACHE_DIR=$G/cache/triton CUDA_CACHE_PATH=$G/cache/nv \
       LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
cd $WT
{ nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; echo "galpy $(git rev-parse HEAD)"
  python -c "import jax, torch; print('jax', jax.__version__, 'torch', torch.__version__)"; date; } | tee $OUT/env.txt
P="python -m pytest -p no:cacheprovider -rA --tb=long"
F="tests/test_backend_diskdf.py tests/test_backend_multi.py tests/test_backend_optimize.py tests/test_backend_quadrature.py tests/test_backend_streamgapdf.py tests/test_backend_evolveddiskdf.py tests/test_backend_actionAngleVerticalInverse.py tests/test_streamdf.py::test_plotting"
for bk in torch jax; do $P -n 2 --backend $bk --device cuda --junitxml=$OUT/item1/$bk.xml $F > $OUT/item1/$bk.txt 2>&1; echo "item1 $bk exit $?"; done
for i in 1 2 3 4 5; do $P --backend jax --device cuda --junitxml=$OUT/item2/run_$i.xml "tests/test_backend_evolveddiskdf.py::test_moment_grad_through_evolution_vs_fd" > $OUT/item2/run_$i.txt 2>&1; echo "item2 run $i exit $?"; done
for i in 1 2 3 4 5; do XLA_FLAGS="--xla_gpu_dump_autotune_results_to=$OUT/item3/at_$i.txt --xla_dump_to=$OUT/item3/hlo_$i --xla_dump_hlo_as_text" python $G/jobs/edf_grad.py > $OUT/item3/edf_grad_$i.txt 2>&1; echo "item3 run $i exit $?"; done
$P --backend torch --device cuda --junitxml=$OUT/item4/inductor.xml tests/test_backend_compile_workflows.py -k torch-inductor > $OUT/item4/inductor.txt 2>&1; echo "item4 inductor exit $?"
for bk in torch jax; do $P --backend $bk --device cuda --junitxml=$OUT/item4/diskdf_jit_$bk.xml "tests/test_backend_diskdf.py::test_moment_array_R_under_jit" > $OUT/item4/diskdf_jit_$bk.txt 2>&1; echo "item4 diskdf_jit $bk exit $?"; done
echo "done $(date)"
