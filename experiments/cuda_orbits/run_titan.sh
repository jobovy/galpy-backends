#!/usr/bin/env bash
# Dev-box (TITAN Xp) timing sequence; run only when ./gpu_idle.sh timing passes.
set -euo pipefail
cd "$(dirname "$0")"
PY=/var/tmp/galpy-gpu/bin/python
CAP=/tmp/claude-21516/-home-bovy-Repos-galpy/e83d6165-0836-4adf-9f62-52251f942f9d/scratchpad/memcap.sh
run() { ./gpu_idle.sh >/dev/null || { echo "GPU no longer idle, abort"; exit 3; }
        taskset -c 24-35 nice -n 19 $CAP 16 $PY "$@"; }
run bench.py --Ns 1e3,1e4,1e5,1e6,3e6 --out data/cuda_gpu_hiN.json
run bench.py --Ns 1e5 --modes lazy --block 64 --repeats 2 --out data/cuda_gpu_hiN_block64.json
run bench.py --Ns 1e5 --modes lazy --block 256 --repeats 2 --out data/cuda_gpu_hiN_block256.json
run bench.py --Ns 1e5 --modes lazy --variant _r128 --repeats 2 --out data/cuda_gpu_hiN_r128.json
run bench.py --Ns 1e5,1e6 --modes lazy --variant _nofma --repeats 2 --out data/cuda_gpu_hiN_nofma.json
run bench.py --Ns 1e5,1e6 --modes lazy --tol 1e-5 --repeats 2 --out data/cuda_gpu_hiN_f64_tol1e-5.json
run bench.py --Ns 1e5,1e6 --modes lazy --variant _f32 --tol 1e-5 --repeats 2 --out data/cuda_gpu_hiN_f32.json
echo "TITAN timing done"
