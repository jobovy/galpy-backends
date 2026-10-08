#!/usr/bin/env bash
# Second H100 job: diagnosis matrix + the fixed kernel's N scan.  One command:
#   module load StdEnv/2023 gcc/12.3 cuda/12.9     # Trillium (cuda/13.x too new for driver 580)
#   ./run_h100_v2.sh
# Needs nvcc (>= 11.8), gcc, python3 + numpy.  ~1 h.  Writes data/h100/v2/*.json.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python3}
OUT=data/h100/v2
mkdir -p $OUT
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
ARCHS="${ARCHS:-90}" ./build.sh | tee $OUT/build.txt
echo "== correctness vs galpy dop853_c (300 orbits, 2 Gyr; includes refill + sorted schedules) =="
for v in "" _noinl _fast_noinl; do $PY check_correctness.py ref_N300.npz --gpu --variant "$v"; done \
  | tee $OUT/correctness.txt
echo "== diagnosis matrix at N=1e6 =="
PY=$PY REP=2 ./run_diag.sh $OUT/diag 1000000 || echo 'run_diag FAILED (continuing)'
echo "== N scan, pinned host buffers (wall = H2D + kernel + D2H; alloc of pinned buffer untimed) =="
# galpy-exact arithmetic, one out-of-line force, refill kernel
$PY bench.py --Ns 1e3,1e4,1e5,1e6,3e6,1e7 --variant _noinl --refill --pinned --modes faithful \
  --out $OUT/scan_noinl_refill_pinned.json || echo FAILED
# + division-free gamma_inc, sqrt forms, cost-sorted schedule (fastest configuration)
$PY bench.py --Ns 1e3,1e4,1e5,1e6,3e6,1e7 --variant _fast_noinl --refill --sort --pinned \
  --modes faithful --out $OUT/scan_fast_noinl_refill_sort_pinned.json || echo FAILED
# v1 kernel with pinned buffers only (isolates the D2H fix)
$PY bench.py --Ns 1e5,1e6,1e7 --pinned --modes faithful --out $OUT/scan_v1_pinned.json || echo FAILED
echo "== Nsight Compute (if available and counters are permitted): stall reasons, occupancy, i-cache =="
if command -v ncu >/dev/null; then
  for cfg in "faithful '' 0 v1_exact_faithful" "lazy '' 0 v1_exact_lazy" "faithful _fast_noinl 1 fast_noinl_refill"; do
    eval set -- $cfg
    ncu --set full --kernel-name regex:k_integrate --launch-count 1 -f -o $OUT/ncu_$4 \
      $PY profile_run.py 2e5 $1 "$2" $3 > $OUT/ncu_$4.log 2>&1 \
      && ncu --import $OUT/ncu_$4.ncu-rep --page details > $OUT/ncu_$4.txt 2>&1 \
      || echo "ncu $4 failed (see $OUT/ncu_$4.log)"
  done
fi
echo "done: $(ls $OUT)"
