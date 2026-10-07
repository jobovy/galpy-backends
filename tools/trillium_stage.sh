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
#   bench   : benchmarks/gpu bench.py, jax gpu + torch gpu (~20 min)
#   newfigs : GPU series of gV2, gD3, gS5, gS4 (~1-2 h)
#   hiN2    : larger N on the 80 GB card, MERGED into the existing JSONs (--Ns): g12 jax Dopri8 +
#             Dopri5 at N=1e6, 3e6, g12 torch at 3e6, gV2 GPU eval at 1e7, gD3 GPU sampling at 1e8 (~1 h)
set -u
STAGE=$1
: "${G13ROOT:=$SCRATCH/galpy-h100}"           # clones, logs, caches live here
R=$G13ROOT/galpy-jax; WT=$G13ROOT/galpy-fb; LOG=$G13ROOT/logs; mkdir -p $LOG
export PYTHONPATH=$WT:$R GALPY_WORKTREE=$WT XLA_PYTHON_CLIENT_PREALLOCATE=false
export XDG_CACHE_HOME=$G13ROOT/cache MPLCONFIGDIR=$G13ROOT/cache/mpl \
       TORCHINDUCTOR_CACHE_DIR=$G13ROOT/cache/inductor PYTHONDONTWRITEBYTECODE=1
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
  orbits)  run "python $F/_shared/orbit_speed/galpy_jax_measure.py gpu lowN"
           run "python $F/_shared/orbit_speed/galpy_jax_measure.py gpu hiN"
           run "python $M12/jax_dopri5_measure.py gpu lowN"
           run "python $M12/jax_dopri5_measure.py gpu hiN"
           run "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python $M12/torch_measure.py"
           run "$H g8_galax_orbit_speed g12_inbackend_orbit_gpu" ;;
  g15)     for sc in measure_icderiv measure_scaling measure_largeN measure_torch_hiN measure_cstm_largeN; do run "python $M15/$sc.py"; done
           run "$H g15_icderiv_speed" ;;
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
  *) echo "unknown stage $STAGE"; exit 2 ;;
esac
echo ">>> $(date +%T) [$STAGE] done"
