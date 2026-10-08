#!/usr/bin/env bash
# Shared-box GPU gate (dev box only; the H100 script does not use it).
#   ./gpu_idle.sh          -> 0 iff OK for a SHORT correctness run: no compute apps
#                             and the last chain.log line is not START c3_gS4/c3_g17
#   ./gpu_idle.sh timing   -> additionally requires "CHAIN5 DONE" in chain.log
CHAIN=/tmp/claude-21516/-home-bovy-Repos-galpy/e83d6165-0836-4adf-9f62-52251f942f9d/scratchpad/chain.log
n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
last=$(tail -n 1 "$CHAIN" 2>/dev/null || true)
[[ "$n" -eq 0 ]] || { echo "busy: $n compute apps"; exit 1; }
[[ "$last" =~ START\ (c[0-9]_gS4|c[0-9]_g17|s6|s7) ]] && { echo "busy: $last"; exit 1; }
if [[ "${1:-}" == timing ]]; then
  grep -q "CHAIN5 DONE" "$CHAIN" || { echo "wait: no CHAIN5 DONE yet ($last)"; exit 1; }
fi
echo idle; exit 0
