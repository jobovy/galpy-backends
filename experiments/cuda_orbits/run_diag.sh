#!/usr/bin/env bash
# Controlled experiments isolating the kernel's inefficiencies (one variable at a time).
#   ./run_diag.sh <outdir> [N]     e.g. ./run_diag.sh data/titan_diag 20000
# Each line is one bench.py call; all use nt=16, rtol=atol=1e-8 unless noted.
set -uo pipefail  # keep going if one configuration fails
cd "$(dirname "$0")"
OUT=${1:?outdir}; N=${2:-100000}; PY=${PY:-python3}; REP=${REP:-2}
# GATE (optional): command that must succeed before each run (shared dev box)
b() { if [[ -n "${GATE:-}" ]]; then until $GATE >/dev/null; do sleep 30; done; fi
      $PY bench.py --Ns $N --repeats $REP "$@" || echo "FAILED: $*"; }
mkdir -p "$OUT"
# 1. baseline v1 kernel (one thread = one whole orbit), both dense-output modes
b --modes faithful,lazy                              --out $OUT/base.json
# 2. warp divergence: lockstep fixed steps (250 ~ mean steps/orbit), no imbalance
b --modes faithful,lazy --fixed-nstep 250            --out $OUT/fixed250.json
# 3. imbalance fixes: cost-sorted schedule; persistent-thread refill; both
b --modes faithful,lazy --sort                       --out $OUT/sort.json
b --modes faithful,lazy --refill                     --out $OUT/refill.json
b --modes faithful,lazy --refill --sort              --out $OUT/refill_sort.json
# 4. code size / arithmetic cost of the force: one out-of-line force copy (kernel
#    700 kB -> 107 kB of SASS); division-free gamma_inc; sqrt instead of pow; all
for v in noinl fastgam fastpow fast fast_noinl; do
  b --modes faithful,lazy --variant _$v              --out $OUT/$v.json
  b --modes faithful,lazy --variant _$v --refill     --out $OUT/${v}_refill.json
done
b --modes faithful --variant _fast_noinl --fixed-nstep 250 --out $OUT/fast_noinl_fixed250.json
b --modes faithful --variant _fast_noinl --refill --sort --out $OUT/fast_noinl_refill_sort.json
# 5. occupancy: 128-register cap (2x resident warps, more spills)
b --modes faithful,lazy --variant _r128 --refill     --out $OUT/r128_refill.json
b --modes faithful,lazy --variant _r128 --refill     --block 64 --out $OUT/r128_refill_b64.json
# 6. FP64 penalty at equal tolerance (1e-5): float32 vs float64, fast forms, refill
b --modes faithful --refill --tol 1e-5                --out $OUT/refill_f64_tol1e-5.json
b --modes faithful --variant _f32 --refill --tol 1e-5  --out $OUT/f32_refill_tol1e-5.json
b --modes faithful --variant _noinl --refill --tol 1e-5 --out $OUT/noinl_refill_f64_tol1e-5.json
b --modes faithful --variant _f32_noinl --refill --tol 1e-5 --out $OUT/f32_noinl_refill_tol1e-5.json
