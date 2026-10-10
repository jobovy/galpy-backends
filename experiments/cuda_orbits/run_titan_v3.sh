#!/usr/bin/env bash
# TITAN Xp (dev box) v3 experiments: occupancy (dop853 stages in shared memory) and
# fixed-step symplectic integrators at matched accuracy.  Every run is gated on
# ./gpu_idle.sh timing (no running cF_*gpu step, no compute apps) and logged.
set -uo pipefail
cd "$(dirname "$0")"
PY="taskset -c 68-71 nice -n 19 /tmp/claude-21516/-home-bovy-Repos-galpy/e83d6165-0836-4adf-9f62-52251f942f9d/scratchpad/memcap.sh 16 /var/tmp/galpy-gpu/bin/python"
OUT=data/titan/v3; mkdir -p $OUT
LOG=$OUT/run.log
g() { until ./gpu_idle.sh >/dev/null; do sleep 60; done
      echo "$(date '+%F %T') RUN $*" | tee -a $LOG gpu_usage.log >/dev/null
      $PY "$@" 2>&1 | tee -a $LOG | grep -E "us/orbit|regs|GPU|\\[gpu|FAIL|Error" ; }
N=${N:-30000}
# correctness (seconds)
g check_correctness.py ref_N300.npz --gpu --variant _noinl
g check_correctness.py ref_N300.npz --gpu --variant _fast_noinl
g check_symplectic.py --gpu --variant _noinl --minb 4
g check_symplectic.py --gpu --variant _fast_noinl --minb 8
# A: dop853 refill vs stages in shared memory (faithful dense output)
for v in _noinl _fast_noinl; do
  g bench.py --Ns $N --variant $v --refill --modes faithful --repeats 2 --out $OUT/A_${v#_}_refill.json
  for s in 10,3 14,4 18,3 18,4; do
    g bench.py --Ns $N --variant $v --nsh $s --modes faithful --repeats 2 --out $OUT/A_${v#_}_smem${s/,/_}.json
  done
done
# B: symplectic at matched accuracy (common dt; n = substeps per output interval, from
# accuracy_scan: pos_err ~1.4e-5 / 1e-4 / 1e-3 at symplec6 n=28/20/14, symplec4 n=112/68/38,
# leapfrog n=3072/1152/368) and dop853 at the tolerances with the same accuracy
for v in _noinl _fast_noinl; do
  for m in 2 4 8; do g bench_symp.py --Ns $N --variant $v --scheme symplec6 --n 28 --minb $m --repeats 2 --out $OUT/B_${v#_}_s6_n28_minb$m.json; done
  g bench_symp.py --Ns $N --variant $v --scheme symplec6 --n 20 --repeats 2 --out $OUT/B_${v#_}_s6_n20.json
  g bench_symp.py --Ns $N --variant $v --scheme symplec6 --n 14 --repeats 2 --out $OUT/B_${v#_}_s6_n14.json
  g bench_symp.py --Ns $N --variant $v --scheme symplec4 --n 112 --repeats 2 --out $OUT/B_${v#_}_s4_n112.json
  g bench_symp.py --Ns $N --variant $v --scheme symplec4 --n 68 --repeats 2 --out $OUT/B_${v#_}_s4_n68.json
  g bench_symp.py --Ns $N --variant $v --scheme symplec4 --n 38 --repeats 2 --out $OUT/B_${v#_}_s4_n38.json
  g bench_symp.py --Ns $N --variant $v --scheme leapfrog --n 3072 --repeats 1 --out $OUT/B_${v#_}_lf_n3072.json
  g bench_symp.py --Ns $N --variant $v --scheme leapfrog --n 1152 --repeats 1 --out $OUT/B_${v#_}_lf_n1152.json
  g bench_symp.py --Ns $N --variant $v --scheme leapfrog --n 368 --repeats 2 --out $OUT/B_${v#_}_lf_n368.json
  g bench_symp.py --Ns $N --variant $v --scheme symplec6 --tol 1e-8 --repeats 2 --out $OUT/B_${v#_}_s6_tol1e-8.json
  for t in ${DOP_TOLS:-1e-8 5e-8 2e-6}; do
    g bench.py --Ns $N --variant $v --refill --modes faithful --tol $t --repeats 2 --out $OUT/B_${v#_}_dop853_tol$t.json
  done
done
echo "$(date '+%F %T') DONE" | tee -a $LOG
