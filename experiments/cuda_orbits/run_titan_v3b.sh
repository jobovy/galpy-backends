#!/usr/bin/env bash
# TITAN Xp v3b: A (dop853 stages in shared memory) and B (symplectic at matched accuracy)
# at N = 1e5 (and 1e6 for the headline configurations).  Gated on ./gpu_idle.sh; logged.
set -uo pipefail
cd "$(dirname "$0")"
PY="taskset -c 44-45 nice -n 19 /tmp/claude-21516/-home-bovy-Repos-galpy/e83d6165-0836-4adf-9f62-52251f942f9d/scratchpad/memcap.sh 16 /var/tmp/galpy-gpu/bin/python"
OUT=data/titan/v3b; mkdir -p $OUT
LOG=$OUT/run.log
# timing only when no other process is on the GPU and no cF_ GPU step runs (gpu_idle.sh)
g() { until ./gpu_idle.sh >/dev/null; do sleep 60; done
      echo "$(date '+%F %T') RUN $*" | tee -a $LOG gpu_usage.log >/dev/null
      $PY "$@" 2>&1 | tee -a $LOG | grep -E "us/orbit|regs|smem|\\[gpu|FAIL|Error" ; }
g check_correctness.py ref_N300.npz --gpu --variant _noinl
g check_correctness.py ref_N300.npz --gpu --variant _fast_noinl
# A
for v in _noinl _fast_noinl; do
  g bench.py --Ns 1e5 --variant $v --refill --pinned --modes faithful --repeats 2 --out $OUT/A_${v#_}_refill.json
  for s in 10,2 10,3 14,3 14,4 18,3 18,4; do
    g bench.py --Ns 1e5 --variant $v --nsh $s --pinned --modes faithful --repeats 2 --out $OUT/A_${v#_}_smem${s/,/_}.json
  done
done
# B (matched accuracy; see run_h100_v3.sh header for the table)
for v in _noinl _fast_noinl; do
  for t in 1e-8 5e-8 2e-6; do
    g bench.py --Ns 1e5 --variant $v --refill --pinned --modes faithful --tol $t --repeats 2 --out $OUT/B_${v#_}_dop853_tol$t.json
  done
  for n in 28 20 14; do g bench_symp.py --Ns 1e5 --pinned --variant $v --scheme symplec6 --n $n --repeats 2 --out $OUT/B_${v#_}_s6_n$n.json; done
  for n in 112 68 38; do g bench_symp.py --Ns 1e5 --pinned --variant $v --scheme symplec4 --n $n --repeats 2 --out $OUT/B_${v#_}_s4_n$n.json; done
  for n in 1152 368; do g bench_symp.py --Ns 1e5 --pinned --variant $v --scheme leapfrog --n $n --repeats 1 --out $OUT/B_${v#_}_lf_n$n.json; done
  # galpy's per-orbit estimate_step at the tolerance that matches 1.4e-5 (symplec6 1e-8: 6.3e-6;
  # symplec4 1e-8: 7.6e-6; leapfrog 1e-10: 1.6e-5)
  g bench_symp.py --Ns 1e5 --pinned --variant $v --scheme symplec6 --tol 1e-8 --repeats 2 --out $OUT/B_${v#_}_s6_tol1e-8.json
  g bench_symp.py --Ns 1e5 --pinned --variant $v --scheme symplec4 --tol 1e-8 --repeats 2 --out $OUT/B_${v#_}_s4_tol1e-8.json
  # two orbits per thread (paired force call), symplec6 n=28
  for m in 1 2; do
    g bench_symp.py --Ns 1e5 --pinned --variant $v --scheme symplec6 --n 28 --per-thread 2 --minb $m --repeats 2 --out $OUT/B_${v#_}_s6_n28_2pt_minb$m.json
  done
  g bench_symp.py --Ns 1e5 --pinned --variant $v --scheme symplec6 --n 28 --minb 8 --repeats 2 --out $OUT/B_${v#_}_s6_n28_minb8.json
done
# headline at 1e6
g bench.py --Ns 1e6 --variant _noinl --refill --pinned --modes faithful --repeats 2 --out $OUT/H_noinl_dop853_tol1e-8_1e6.json
g bench.py --Ns 1e6 --variant _fast_noinl --refill --pinned --modes faithful --repeats 2 --out $OUT/H_fast_noinl_dop853_tol1e-8_1e6.json
g bench_symp.py --Ns 1e6 --pinned --variant _noinl --scheme symplec6 --n 28 --repeats 2 --out $OUT/H_noinl_s6_n28_1e6.json
g bench_symp.py --Ns 1e6 --pinned --variant _fast_noinl --scheme symplec6 --n 28 --repeats 2 --out $OUT/H_fast_noinl_s6_n28_1e6.json
echo "$(date '+%F %T') DONE" | tee -a $LOG
