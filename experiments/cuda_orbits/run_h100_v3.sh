#!/usr/bin/env bash
# Third H100 job (~30-60 min): (A) dop853 with its stage vectors in shared memory
# (occupancy), (B) galpy's fixed-step symplectic integrators at MATCHED ACCURACY vs
# dop853.  One command:
#   module load StdEnv/2023 gcc/12.3 cuda/12.9     # Trillium (cuda/13.x too new for driver 580)
#   ./run_h100_v3.sh
# Needs nvcc (>= 11.8), gcc, python3 + numpy.  Writes data/h100/v3/* (each timing JSON
# records the GPU's other compute processes at start/end: gpu_exclusive).
#
# Matched accuracy (accuracy_scan.py on the CPU build = galpy's arithmetic; 300 reference
# orbits, MWPotential2014, 2 Gyr; error = max over orbits of |dx|/|x| at 2 Gyr vs galpy
# dop853_c at 1e-13):
#   target   dop853 rtol=atol (nfev)   symplec6 n (nfev)   symplec4 n (nfev)   leapfrog n (nfev)
#   1.4e-5   1e-8   (3472)             28 (2940)           112 (5040)          3072 (~46k)
#   1e-4     5e-8   (2974)             20 (2100)            68 (~3060)         1152 (~17k)
#   1e-3     2e-6   (1913)             14 (1470)            38 (~1710)          368 (~5.5k)
# (n = common steps per output interval of 56.2/15; dt = 3.7467/n)
set -uo pipefail
cd "$(dirname "$0")"
PY=${PY:-python3}
OUT=data/h100/v3
mkdir -p $OUT
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | tee $OUT/gpu.txt
ARCHS="${ARCHS:-90}" VARIANTS="base noinl fast_noinl" ./build.sh > $OUT/build.txt 2>&1 || { cat $OUT/build.txt; exit 1; }
./resource_usage.sh libcudaorbits_noinl.so libcudaorbits_fast_noinl.so | tee $OUT/resource_usage.txt
echo "== correctness (dop853 incl. shared-memory kernels; symplectic vs galpy C) =="
for v in _noinl _fast_noinl; do
  $PY check_correctness.py ref_N300.npz --gpu --variant $v
  $PY check_symplectic.py --gpu --variant $v --minb 4
  $PY check_symplectic.py --gpu --variant $v --minb 2 --per-thread2
done 2>&1 | tee $OUT/correctness.txt
B="$PY bench.py --modes faithful --repeats 2 --pinned"
S="$PY bench_symp.py --repeats 2 --pinned"
echo "== A: dop853 refill, stages in registers vs shared memory =="
for v in _noinl _fast_noinl; do
  $B --Ns 1e5,1e6,1e7 --variant $v --refill --out $OUT/A_${v#_}_refill.json || echo FAILED
  for s in 10,2 10,3 14,3 14,4 18,3 18,4; do
    $B --Ns 1e5,1e6 --variant $v --nsh $s --out $OUT/A_${v#_}_smem${s/,/_}.json || echo FAILED
  done
done
echo "== B: symplectic (common dt, lockstep) at matched accuracy, register caps =="
for v in _noinl _fast_noinl; do
  for m in 2 4 8; do
    $S --Ns 1e5,1e6,1e7 --variant $v --scheme symplec6 --n 28 --minb $m --out $OUT/B_${v#_}_s6_n28_minb$m.json || echo FAILED
  done
  for m in 1 2; do  # two orbits per thread (paired force call)
    $S --Ns 1e5,1e6,1e7 --variant $v --scheme symplec6 --n 28 --per-thread 2 --minb $m --out $OUT/B_${v#_}_s6_n28_2pt_minb$m.json || echo FAILED
  done
  for n in 20 14; do $S --Ns 1e5,1e6 --variant $v --scheme symplec6 --n $n --out $OUT/B_${v#_}_s6_n$n.json || echo FAILED; done
  for n in 112 68 38; do $S --Ns 1e5,1e6 --variant $v --scheme symplec4 --n $n --out $OUT/B_${v#_}_s4_n$n.json || echo FAILED; done
  for n in 3072 1152 368; do $S --Ns 1e5,1e6 --variant $v --scheme leapfrog --n $n --out $OUT/B_${v#_}_lf_n$n.json || echo FAILED; done
  $S --Ns 1e5,1e6 --variant $v --scheme symplec6 --tol 1e-8 --out $OUT/B_${v#_}_s6_tol1e-8.json || echo FAILED
  $S --Ns 1e5,1e6 --variant $v --scheme symplec4 --tol 1e-8 --out $OUT/B_${v#_}_s4_tol1e-8.json || echo FAILED
  for t in 1e-8 5e-8 2e-6; do
    $B --Ns 1e5,1e6,1e7 --variant $v --refill --tol $t --out $OUT/B_${v#_}_dop853_tol$t.json || echo FAILED
  done
done
echo "== Nsight Compute (best variants) =="
if command -v ncu >/dev/null; then
  ncu --set full --kernel-name regex:k_symp --launch-count 1 -f -o $OUT/ncu_symp6_n28 \
    $PY profile_symp.py 2e5 symplec6 28 _fast_noinl 4 > $OUT/ncu_symp6_n28.log 2>&1 \
    && ncu --import $OUT/ncu_symp6_n28.ncu-rep --page details > $OUT/ncu_symp6_n28.txt 2>&1 || echo "ncu symp failed"
  ncu --set full --kernel-name regex:k_integrate_smem --launch-count 1 -f -o $OUT/ncu_smem18_3 \
    $PY profile_run.py 2e5 faithful _fast_noinl 1 18,3 > $OUT/ncu_smem18_3.log 2>&1 \
    && ncu --import $OUT/ncu_smem18_3.ncu-rep --page details > $OUT/ncu_smem18_3.txt 2>&1 || echo "ncu smem failed"
fi
$PY summarize_v3.py $OUT --md > $OUT/summary.md 2>&1 || true
echo "done: $(ls $OUT)"
