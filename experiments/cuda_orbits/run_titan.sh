#!/usr/bin/env bash
# Dev-box (TITAN Xp) N scans; run only when ./gpu_idle.sh timing passes (after CHAIN5 DONE).
set -euo pipefail
cd "$(dirname "$0")"
PY=/var/tmp/galpy-gpu/bin/python
CAP=/tmp/claude-21516/-home-bovy-Repos-galpy/e83d6165-0836-4adf-9f62-52251f942f9d/scratchpad/memcap.sh
run() { ./gpu_idle.sh >/dev/null || { echo "GPU no longer idle, abort"; exit 3; }
        taskset -c 24-35 nice -n 19 $CAP 16 $PY bench.py "$@"; }
mkdir -p data/titan
run --Ns 1e3,1e4,1e5,1e6 --modes faithful,lazy --repeats 2 --out data/titan/scan_v1.json
run --Ns 1e3,1e4,1e5,1e6 --modes faithful --variant _noinl --refill --pinned --repeats 2 \
    --out data/titan/scan_noinl_refill_pinned.json
run --Ns 1e3,1e4,1e5,1e6,3e6 --modes faithful --variant _fast_noinl --refill --sort --pinned \
    --repeats 2 --out data/titan/scan_fast_noinl_refill_sort_pinned.json
echo "TITAN timing done"
