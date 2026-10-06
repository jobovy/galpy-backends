#!/bin/bash
# Re-measure + re-render every CPU (non-timing) paper figure against a galpy
# worktree. Four groups run in parallel, each pinned to 3 cores (shared box).
#   tools/rerun_figures_cpu.sh /var/tmp/fbp <logdir>
# Timing/GPU figures: tools/rerun_figures_gpu.sh (strictly serial, quiet box).
set -u
WT=$(realpath "$1"); LOG=$(realpath -m "$2"); mkdir -p "$LOG"
REPO=$(cd "$(dirname "$0")/.." && pwd); F=$REPO/paper/figures
export PYTHONPATH=$WT:$REPO GALPY_WORKTREE=$WT CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu
H="python -m paper.figures._harness run"
cd "$REPO"
run() { local g=$1 cores=$2; shift 2; ( for c in "$@"; do echo ">>> $(date +%T) $c"; ( eval "taskset -c $cores $c" ) || echo "!!! FAILED: $c"; done; echo ">>> $(date +%T) done" ) > "$LOG/$g.log" 2>&1 & }
rm -f $F/gF1_autodiff_forces/data/*.json $F/gD1_df_sensitivity/data/*.json \
      $F/gD2_df_sample_autodiff/data/*.json $F/gS1_stream_track_sensitivity/data/*.json \
      $F/gS2_streamspray_sensitivity/data/*.json $F/gS3_track_extrapolation/data/*.json
run inline 60-62 "$H g1_backend_parity g3_param_gradients g7_orbit_sensitivity g18_actionangle_sensitivity gF1_autodiff_forces gD1_df_sensitivity"
run dfspray 63-65 "$H gD2_df_sample_autodiff gS3_track_extrapolation gS2_streamspray_sensitivity"
run track 66-68 "$H gS1_stream_track_sensitivity"
run measured 69-71 \
  "python $F/g9_variational_validity/measure/variational_validity_measure.py" \
  "python $F/g10_lyapunov/measure/lyapunov_measure.py" \
  "python $F/g11_dissipative_phase_volume/measure/dissipative_measure.py cdf" \
  "python $F/g11_dissipative_phase_volume/measure/dissipative_measure.py fdm" \
  "python $F/g16_icjacobian_precision/measure/measure_precision.py" \
  "python $F/g14_hvs_potential_fit/measure/measure_fit.py" \
  "python $F/g13_ledger_burndown/measure/burndown_from_git.py" \
  "python $F/g12_inbackend_orbit_gpu/measure/work_counts.py" \
  "$H g9_variational_validity g10_lyapunov g11_dissipative_phase_volume g16_icjacobian_precision g14_hvs_potential_fit g13_ledger_burndown"
wait
