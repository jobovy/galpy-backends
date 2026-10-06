#!/bin/bash
# Re-measure every TIMING / GPU paper figure against a galpy worktree, strictly
# serially, each step waiting for the GPU to be free of other users' processes.
#   tools/rerun_figures_gpu.sh /var/tmp/fbp <logdir> [step ...]
# Steps (default all): g2 g5 g6 g17 shared g12 g15 bench render
# galax (g8's galax lines) is external code and is not re-measured here.
set -u
WT=$(realpath "$1"); LOG=$(realpath -m "$2"); shift 2; mkdir -p "$LOG"
STEPS=${*:-"g2 g5 g6 g17 shared g12 g15 bench render"}
REPO=$(cd "$(dirname "$0")/.." && pwd); F=$REPO/paper/figures
export PYTHONPATH=$WT:$REPO GALPY_WORKTREE=$WT XLA_PYTHON_CLIENT_PREALLOCATE=false
H="python -m paper.figures._harness run"
cd "$REPO"
gpu_wait() { while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 60; done; }
step() { local name=$1; shift; gpu_wait; echo ">>> $(date +%T) [$name] $*"; ( eval "$@" ) >> "$LOG/$name.log" 2>&1 && echo "    ok $(date +%T)" || echo "!!! FAILED [$name] (see $LOG/$name.log)"; }
for s in $STEPS; do case $s in
  g2)  rm -f $F/g2_throughput/data/*.json; step g2 "$H g2_throughput" ;;
  g5)  rm -f $F/g5_autodiff_overhead/data/*.json; step g5 "$H g5_autodiff_overhead" ;;
  g6)  rm -f $F/g6_jit_amortization/data/*.json; step g6 "$H g6_jit_amortization" ;;
  g17) rm -f $F/g17_staeckel_actionangle_speed/data/*.json
       step g17 "$H g17_staeckel_actionangle_speed_jax g17_staeckel_actionangle_speed_torch" ;;
  shared) S=$F/_shared/orbit_speed
       step shared "CUDA_VISIBLE_DEVICES= python $S/galpy_c_measure.py lowN"
       step shared "CUDA_VISIBLE_DEVICES= python $S/galpy_c_measure.py hiN"
       step shared "python $S/galpy_jax_measure.py gpu lowN"
       step shared "python $S/galpy_jax_measure.py gpu hiN"
       step shared "CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu python $S/galpy_jax_measure.py cpu lowN"
       step shared "CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu timeout 7200 python $S/galpy_jax_measure.py cpu hiN" ;;
  g12) M=$F/g12_inbackend_orbit_gpu/measure
       step g12 "python $M/jax_dopri5_measure.py gpu lowN"
       step g12 "python $M/jax_dopri5_measure.py gpu hiN"
       step g12 "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python $M/torch_measure.py" ;;
  g15) M=$F/g15_icderiv_speed/measure
       for sc in measure_icderiv measure_scaling measure_largeN measure_torch_hiN measure_cstm_largeN; do
         step g15 "python $M/$sc.py"; done ;;
  bench) step bench "$REPO/benchmarks/gpu/run_all.sh $WT $REPO/benchmarks/gpu/results_$(date +%F)" ;;
  render) step render "$H g8_galax_orbit_speed g12_inbackend_orbit_gpu g15_icderiv_speed" ;;
esac; done
echo ">>> $(date +%T) all done"
