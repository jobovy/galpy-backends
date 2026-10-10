#!/usr/bin/env bash
# Short H100 follow-up to v3 (~15 min): the two best variants at N = 1e7 (+ 1e6 as a
# same-job reference) and Nsight Compute on them.
#   module load StdEnv/2023 gcc/12.3 cuda/12.9
#   ./run_h100_v3b.sh
# Writes data/h100/v3b/ (timing JSONs record gpu_exclusive; summary.md; ncu_*.txt).
set -uo pipefail
cd "$(dirname "$0")"
PY=${PY:-python3}
OUT=data/h100/v3b
mkdir -p $OUT
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | tee $OUT/gpu.txt
ARCHS="${ARCHS:-90}" VARIANTS="base noinl fast_noinl" ./build.sh > $OUT/build.txt 2>&1 || { cat $OUT/build.txt; exit 1; }
./resource_usage.sh libcudaorbits_noinl.so libcudaorbits_fast_noinl.so > $OUT/resource_usage.txt
echo "== correctness =="
for v in _noinl _fast_noinl; do
  $PY check_correctness.py ref_N300.npz --gpu --variant $v   # includes refill and smem (10,3)
  $PY check_symplectic.py --gpu --variant $v --minb 8
done 2>&1 | tee $OUT/correctness.txt
echo "== timings (N = 1e6, 1e7) =="
B="$PY bench.py --modes faithful --repeats 2 --pinned --Ns 1e6,1e7"
S="$PY bench_symp.py --repeats 2 --pinned --Ns 1e6,1e7"
for v in _noinl _fast_noinl; do
  $B --variant $v --refill --out $OUT/dop853_${v#_}_refill.json || echo FAILED
  $B --variant $v --nsh 10,3 --out $OUT/dop853_${v#_}_smem10_3.json || echo FAILED
  $S --variant $v --scheme symplec6 --n 28 --minb 8 --out $OUT/s6_${v#_}_n28_minb8.json || echo FAILED
done 2>&1 | tee $OUT/timings.txt
echo "== Nsight Compute (N = 2e5, fast arithmetic) =="
if command -v ncu >/dev/null; then
  ncu --set full --kernel-name regex:k_integrate_smem --launch-count 1 -f -o $OUT/ncu_smem10_3 \
    $PY profile_run.py 2e5 faithful _fast_noinl 1 10,3 > $OUT/ncu_smem10_3.log 2>&1 \
    && ncu --import $OUT/ncu_smem10_3.ncu-rep --page details > $OUT/ncu_smem10_3.txt 2>&1 || echo "ncu smem failed"
  ncu --set full --kernel-name regex:k_symp --launch-count 1 -f -o $OUT/ncu_symp6_n28_minb8 \
    $PY profile_symp.py 2e5 symplec6 28 _fast_noinl 8 > $OUT/ncu_symp6_n28_minb8.log 2>&1 \
    && ncu --import $OUT/ncu_symp6_n28_minb8.ncu-rep --page details > $OUT/ncu_symp6_n28_minb8.txt 2>&1 || echo "ncu symp failed"
else
  echo "ncu not found" | tee $OUT/ncu_missing.txt
fi
$PY summarize_v3.py $OUT --md > $OUT/summary.md 2>&1 || true
echo "done: $(ls $OUT)"
