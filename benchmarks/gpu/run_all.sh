#!/bin/bash
# usage: run_all.sh <galpy worktree> <outdir>
# Runs each (backend, device) in its own process with the GPU otherwise idle.
WT=$1; OUT=$2; mkdir -p $OUT
GPUPY=/var/tmp/galpy-gpu/bin/python   # jax 0.10.1 + torch 2.12.0+cu126 (TITAN Xp)
cd $WT
export PYTHONPATH=$WT XLA_PYTHON_CLIENT_PREALLOCATE=false
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu $GPUPY $(dirname $0)/bench.py numpy cpu $OUT/numpy_cpu.json
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu $GPUPY $(dirname $0)/bench.py jax cpu $OUT/jax_cpu.json
$GPUPY $(dirname $0)/bench.py jax gpu $OUT/jax_gpu.json
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu $GPUPY $(dirname $0)/bench.py torch cpu $OUT/torch_cpu.json
$GPUPY $(dirname $0)/bench.py torch gpu $OUT/torch_gpu.json
python $(dirname $0)/report.py $OUT > $OUT/report.md
