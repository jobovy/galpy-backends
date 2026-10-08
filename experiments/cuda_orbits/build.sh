#!/usr/bin/env bash
# Build libcudaorbits[_nofma].so for the listed SM architectures.
#   ./build.sh            -> sm_61 (TITAN Xp) + sm_90 (H100)
#   ARCHS="90" ./build.sh -> H100 only
# Needs only nvcc (CUDA >= 11.8 for sm_90) and a host gcc; no GSL, no Python pkgs.
set -euo pipefail
cd "$(dirname "$0")"
NVCC=${NVCC:-$(command -v nvcc || echo /usr/local/cuda-12.4/bin/nvcc)}
ARCHS=${ARCHS:-"61 90"}
GENCODE=""
for a in $ARCHS; do GENCODE+=" -gencode arch=compute_${a},code=sm_${a}"; done
COMMON="-O3 -std=c++17 -shared -Xcompiler -fPIC,-O2 $GENCODE -Xptxas -v"
# default: nvcc contracts a*b+c into FMA on the device (faster, ~1 ulp different)
$NVCC $COMMON cuda_orbits.cu -o libcudaorbits.so 2> build_ptxas.log
# no device FMA contraction: device arithmetic rounds like galpy's (non-FMA) C
$NVCC $COMMON --fmad=false cuda_orbits.cu -o libcudaorbits_nofma.so 2> build_ptxas_nofma.log
# float32 variant: ONLY to show the FP64 penalty (galpy is float64)
$NVCC $COMMON -DCO_FLOAT32 cuda_orbits.cu -o libcudaorbits_f32.so 2> build_ptxas_f32.log
# register-capped variant (128 regs/thread -> 2x the resident warps, more spills)
$NVCC $COMMON -DCO_MINBLOCKS=2 cuda_orbits.cu -o libcudaorbits_r128.so 2> build_ptxas_r128.log
grep -E "Compiling entry|registers|spill" build_ptxas.log | sed 's/^ptxas info *: //' | grep -A2 k_integrate || true
echo "built: $(ls lib*.so)"
