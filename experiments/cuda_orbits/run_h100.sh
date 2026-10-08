#!/usr/bin/env bash
# One command for the H100 session:  ./run_h100.sh   (from any directory)
# Needs: nvcc (CUDA >= 11.8 for sm_90; uses $NVCC or nvcc on PATH), a host gcc,
# python3 with numpy.  No galpy / GSL / jax needed: the potential parameters
# (mw2014_params.json) and the galpy-C reference orbits (ref_N300.npz) ship here.
# Writes data/h100/cuda_gpu_hiN[_nofma].json and prints the correctness check.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python3}
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
ARCHS="${ARCHS:-90}" ./build.sh
echo "== correctness vs galpy dop853_c (300 orbits, 2 Gyr) =="
$PY check_correctness.py ref_N300.npz --gpu
$PY check_correctness.py ref_N300.npz --gpu --variant _nofma
echo "== timing (hiN: nt=16 saves; wall = alloc + H2D + kernel + D2H) =="
$PY bench.py --Ns 1e3,1e4,1e5,1e6,3e6,1e7 --out data/h100/cuda_gpu_hiN.json
$PY bench.py --Ns 1e5,1e6 --modes lazy --variant _r128 --repeats 2 --out data/h100/cuda_gpu_hiN_r128.json
$PY bench.py --Ns 1e5,1e6 --modes lazy --block 64 --repeats 2 --out data/h100/cuda_gpu_hiN_block64.json
$PY bench.py --Ns 1e5,1e6 --modes lazy --block 256 --repeats 2 --out data/h100/cuda_gpu_hiN_block256.json
$PY bench.py --Ns 1e6 --variant _f32 --modes lazy --tol 1e-5 --out data/h100/cuda_gpu_hiN_f32.json || true
$PY bench.py --Ns 1e6 --modes lazy --tol 1e-5 --out data/h100/cuda_gpu_hiN_f64_tol1e-5.json
