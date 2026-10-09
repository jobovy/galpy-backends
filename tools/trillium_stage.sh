#!/bin/bash
# Run one stage of the H100 re-measurement on a Trillium GPU compute node.
#   tools/trillium_stage.sh <stage>      (normally via tools/trillium_job.sbatch)
# Everything is written under $SCRATCH: $HOME is read-only on compute nodes.
# Stages (GPU series only; the CPU series of these figures stay the TITAN box's):
#   smoke   : 2 tiny CUDA checks (minutes)
#   tests   : galpy CUDA test suites, jax + torch (~1 h)
#   g2g5g6  : g2_throughput, g5_autodiff_overhead, g6_jit_amortization (minutes; they re-measure all series)
#   g17     : g17 Staeckel AA (one measurement incl. its CPU series; hours)
#   orbits  : shared jax GPU orbit speeds (g8/g12) + g12 jax-Dopri5 GPU + g12 torch GPU (~40 min)
#   g15     : g15 IC-gradient speed, all five scripts (hours, CPU parts included)
#   g15gpu  : g15 GPU series only (G15_GPU_ONLY=1; no C-STM / CPU series)
#   g17gpu  : g17 GPU series only (make.py --gpu-only, ~15 min)
#   bench   : benchmarks/gpu bench.py, jax gpu + torch gpu (~20 min)
#   newfigs : GPU series of gV2, gD3, gS5, gS4 (~1-2 h)
#   hiN2    : larger N on the 80 GB card, MERGED into the existing JSONs (--Ns): g12 jax Dopri8 +
#             Dopri5 at N=1e6, 3e6, g12 torch at 3e6, gV2 GPU eval at 1e7, gD3 GPU sampling at 1e8 (~1 h)
#   g17torch: g17 torch GPU series alone (torch_gpu_only.py) + nvidia-smi memory log
#   gV2     : gV2 GPU series only
#   gS4     : gS4 GPU series only (merged) + nvidia-smi memory log
#   cudaexp : experiments/cuda_orbits/run_h100.sh (CUDA dop853 experiment; loads the cuda module)
#   cudav2  : experiments/cuda_orbits/run_h100_v2.sh (diagnosis + fixed kernel scans + ncu)
#   triage  : only the failing CUDA tests, in the activated env (TRIAGE_TAG, JAXREP)
set -u
STAGE=$1
: "${G13ROOT:=$SCRATCH/galpy-h100}"           # clones, logs, caches live here
R=$G13ROOT/galpy-jax; WT=$G13ROOT/galpy-fb; LOG=$G13ROOT/logs; mkdir -p $LOG
export PYTHONPATH=$WT:$R GALPY_WORKTREE=$WT XLA_PYTHON_CLIENT_PREALLOCATE=false
export XDG_CACHE_HOME=$G13ROOT/cache MPLCONFIGDIR=$G13ROOT/cache/mpl \
       TORCHINDUCTOR_CACHE_DIR=$G13ROOT/cache/inductor PYTHONDONTWRITEBYTECODE=1 \
       TRITON_HOME=$G13ROOT/cache TRITON_CACHE_DIR=$G13ROOT/cache/triton CUDA_CACHE_PATH=$G13ROOT/cache/nv  # default ~/.triton, ~/.nv: read-only
[ -n "${CONDA_PREFIX:-}" ] && export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}  # libgsl
cd $R
run() { echo ">>> $(date +%T) [$STAGE] $*"; ( eval "$@" ) >> $LOG/$STAGE.log 2>&1 && echo "    ok $(date +%T)" || echo "!!! FAILED [$STAGE] $* (see $LOG/$STAGE.log)"; }
F=$R/paper/figures; M12=$F/g12_inbackend_orbit_gpu/measure; M15=$F/g15_icderiv_speed/measure
H="python -m paper.figures._harness run"
case $STAGE in
  smoke)   run "python -c 'import jax, torch; print(jax.devices(), torch.cuda.get_device_name(0))'"
           run "cd $WT && python -m pytest -q -p no:cacheprovider --backend jax --device cuda tests/test_backend_force.py -x"
           run "cd $WT && python -m pytest -q -p no:cacheprovider --backend torch --device cuda tests/test_backend_force.py -x" ;;
  tests)   for bk in jax torch; do run "cd $WT && python -m pytest -q -p no:cacheprovider -n 2 --backend $bk --device cuda tests/test_backend_*.py"; done
           run "cd $WT && python -m pytest -q -p no:cacheprovider tests/test_backend_special.py tests/test_backend_actionAngleVerticalInverse.py -k cuda" ;;
  g2g5g6)  for n in g2_throughput g5_autodiff_overhead g6_jit_amortization; do run "rm -f $F/$n/data/*.json && $H $n"; done ;;
  g17)     run "rm -f $F/g17_staeckel_actionangle_speed/data/*.json && $H g17_staeckel_actionangle_speed_jax g17_staeckel_actionangle_speed_torch" ;;
  g17gpu)  # GPU series only (the paper's CPU series come from the dev box); ~15 min
           run "rm -f $F/g17_staeckel_actionangle_speed/data/g17_staeckel_actionangle_speed.json.partial && python $F/g17_staeckel_actionangle_speed/make.py --gpu-only" ;;
  orbits)  run "python $F/_shared/orbit_speed/galpy_jax_measure.py gpu lowN"
           run "python $F/_shared/orbit_speed/galpy_jax_measure.py gpu hiN"
           run "python $M12/jax_dopri5_measure.py gpu lowN"
           run "python $M12/jax_dopri5_measure.py gpu hiN"
           run "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python $M12/torch_measure.py"
           run "$H g8_galax_orbit_speed g12_inbackend_orbit_gpu" ;;
  g15)     for sc in measure_icderiv measure_scaling measure_largeN measure_torch_hiN measure_cstm_largeN; do run "python $M15/$sc.py"; done
           run "$H g15_icderiv_speed" ;;
  g15gpu)  # GPU series only (C-STM and the CPU series come from the dev box)
           for sc in measure_icderiv measure_scaling measure_largeN measure_torch_hiN; do run "G15_GPU_ONLY=1 python $M15/$sc.py"; done ;;
  bench)   D=$R/benchmarks/gpu; O=$D/results_h100_$(date +%F); mkdir -p $O
           run "python $D/bench.py jax gpu $O/jax_gpu.json"
           run "python $D/bench.py torch gpu $O/torch_gpu.json" ;;
  newfigs) run "python $F/gV2_vertical_inverse_speed/make.py --devices gpu"
           run "python $F/gD3_spherical_df_speed/make.py --gpu"
           run "python -m paper.figures.gS5_streamspray_speed.make --measure gpu"
           run "python -m paper.figures.gS4_streamdf_speed.make --measure gpu"
           run "$H gV2_vertical_inverse_speed gD3_spherical_df_speed gS5_streamspray_speed gS4_streamdf_speed" ;;
  hiN2)    # each N in its own process (an OOM fails only that run); --Ns keeps all other points
           for n in 1e6 3e6; do
             run "python $F/_shared/orbit_speed/galpy_jax_measure.py gpu hiN --Ns $n"
             run "python $M12/jax_dopri5_measure.py gpu hiN --Ns $n"
           done
           run "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python $M12/torch_measure.py --Ns 3e6"
           run "python $F/gV2_vertical_inverse_speed/make.py --devices gpu --Ns 1e7"
           run "python $F/gD3_spherical_df_speed/make.py --gpu --Ns 1e8"
           run "$H g8_galax_orbit_speed g12_inbackend_orbit_gpu gV2_vertical_inverse_speed gD3_spherical_df_speed" ;;
  g17torch) J17=$F/g17_staeckel_actionangle_speed/data/g17_staeckel_actionangle_speed.json
           nvidia-smi --query-gpu=timestamp,memory.used,memory.total --format=csv -l 5 > $LOG/g17torch_mem.csv & SMI=$!
           run "python $F/g17_staeckel_actionangle_speed/measure/torch_gpu_only.py $J17"
           kill $SMI
           run "$H g17_staeckel_actionangle_speed_jax g17_staeckel_actionangle_speed_torch" ;;
  gV2)     run "python $F/gV2_vertical_inverse_speed/make.py --devices gpu"
           run "$H gV2_vertical_inverse_speed" ;;
  triage)  # only the failing CUDA tests (logs/h100/triage/*_failed.txt), in the env the job activated;
           # TRIAGE_TAG names the output dir (e.g. t214, t212); JAXREP=3 repeats the jax list
           T=$LOG/triage_${TRIAGE_TAG:-x}; mkdir -p $T; TL=$R/logs/h100/triage
           run "python -c 'import torch, jax; print(torch.__version__, torch.version.cuda, jax.__version__)'"
           run "cd $WT && python -m pytest -p no:cacheprovider --backend torch --device cuda -rA --tb=long --junitxml=$T/torch.xml \$(cat $TL/torch_failed.txt) > $T/torch.txt 2>&1"
           for i in $(seq 1 ${JAXREP:-0}); do run "cd $WT && python -m pytest -p no:cacheprovider --backend jax --device cuda -rA --tb=long --junitxml=$T/jax_$i.xml \$(cat $TL/jax_failed.txt) > $T/jax_$i.txt 2>&1"; done ;;
  gS4)     nvidia-smi --query-gpu=timestamp,memory.used,memory.total --format=csv -l 5 > $LOG/gS4_mem.csv & SMI=$!
           run "python -m paper.figures.gS4_streamdf_speed.make --measure gpu"
           kill $SMI
           run "$H gS4_streamdf_speed" ;;
  cudaexp) # experiments/cuda_orbits: CUDA MWPotential2014 + dop853, correctness vs galpy C, timings to 1e7
           module load StdEnv/2023 gcc/12.3 cuda/12.9   # driver 580 = CUDA 13.0; cuda/13.x is newer than it
           command -v nvcc > /dev/null || { echo "!!! no nvcc after module load"; exit 3; }
           run "nvcc --version | tail -1"
           run "cd $R/experiments/cuda_orbits && PY=$CONDA_PREFIX/bin/python ./run_h100.sh" ;;
  cudav2)  # experiments/cuda_orbits/run_h100_v2.sh: diagnosis matrix, pinned N scans, ncu (if permitted)
           module load StdEnv/2023 gcc/12.3 cuda/12.9
           command -v nvcc > /dev/null || { echo "!!! no nvcc after module load"; exit 3; }
           run "grep RmProfilingAdminOnly /proc/driver/nvidia/params || true"
           run "cd $R/experiments/cuda_orbits && mkdir -p data/h100/v2 && PY=$CONDA_PREFIX/bin/python ./run_h100_v2.sh 2>&1 | tee data/h100/v2/run.log" ;;
  *) echo "unknown stage $STAGE"; exit 2 ;;
esac
echo ">>> $(date +%T) [$STAGE] done"
