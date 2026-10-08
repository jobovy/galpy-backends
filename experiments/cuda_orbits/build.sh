#!/usr/bin/env bash
# Build the libcudaorbits*.so variants for the listed SM architectures.
#   ./build.sh            -> sm_61 (TITAN Xp) + sm_90 (H100)
#   ARCHS="90" ./build.sh -> H100 only
# Needs only nvcc (CUDA >= 11.8 for sm_90) and a host gcc; no GSL, no Python pkgs.
# On Trillium: module load StdEnv/2023 gcc/12.3 cuda/12.9
# Variants (suffix: flags):
#   ""        exact galpy arithmetic (GSL gamma_inc_P port, libm pow); device FMA on
#   _nofma    as "" with --fmad=false (device rounds like galpy's non-FMA C)
#   _fastgam  division-free incomplete-gamma series (table of 1/(a+n))
#   _fastpow  pow(x,0.5|1.5|-1.5) -> sqrt forms, MN/NFW R/z forces fused
#   _fast     _fastgam + _fastpow
#   _noinl    as "" but ONE out-of-line force function (kernel ~17x smaller)
#   _fast_noinl  _fast + _noinl
#   _r128     as "" with registers capped at 128/thread (2x resident warps)
#   _f32      float32 (ONLY to show the FP64 penalty; galpy is float64)
set -euo pipefail
cd "$(dirname "$0")"
NVCC=${NVCC:-$(command -v nvcc || echo /usr/local/cuda-12.4/bin/nvcc)}
ARCHS=${ARCHS:-"61 90"}
VARIANTS=${VARIANTS:-"base nofma fastgam fastpow fast noinl fast_noinl r128 f32 f32_noinl"}
GENCODE=""
for a in $ARCHS; do GENCODE+=" -gencode arch=compute_${a},code=sm_${a}"; done
COMMON="-O3 -std=c++17 -shared -Xcompiler -fPIC,-O2 $GENCODE -Xptxas -v"
declare -A FLAGS=(
  [base]="" [nofma]="--fmad=false" [fastgam]="-DCO_FASTGAM" [fastpow]="-DCO_FASTPOW"
  [fast]="-DCO_FASTGAM -DCO_FASTPOW" [r128]="-DCO_MINBLOCKS=2" [f32]="-DCO_FLOAT32"
  [noinl]="-DCO_NOINLINE_FORCE" [fast_noinl]="-DCO_FASTGAM -DCO_FASTPOW -DCO_NOINLINE_FORCE"
  [f32_noinl]="-DCO_FLOAT32 -DCO_NOINLINE_FORCE"
)
pids=()
for v in $VARIANTS; do
  sfx=$([[ $v == base ]] && echo "" || echo "_$v")
  $NVCC $COMMON ${FLAGS[$v]} cuda_orbits.cu -o libcudaorbits${sfx}.so 2> build_ptxas${sfx}.log &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait $p || fail=1; done
if [[ $fail == 1 ]]; then grep -l "error" build_ptxas*.log | xargs -r -n1 sh -c 'echo "== $0"; grep -m5 error "$0"'; exit 1; fi
for f in build_ptxas*.log; do  # registers / spills per kernel
  echo "== $f"
  awk '/Compiling entry function/ {match($0,/k_integrate[a-z_]*ILb[01]/); k=substr($0,RSTART,RLENGTH);
       match($0,/sm_[0-9]+/); a=substr($0,RSTART,RLENGTH)}
       /Function properties for .*co_dev_accel/ {k="co_dev_accel"; a=""}
       /spill stores/ && k!="" {sp=$0} /Used [0-9]+ registers/ && k!="" {match($0,/Used [0-9]+ registers/);
       r=substr($0,RSTART,RLENGTH); gsub(/^ +/,"",sp); print "  " k, a, r ";", sp; k=""}' $f || true
done
echo "built: $(ls lib*.so | tr '\n' ' ')"
