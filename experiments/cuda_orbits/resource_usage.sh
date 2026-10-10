#!/usr/bin/env bash
# Static per-kernel resources (registers/thread, stack bytes) of the built libraries; no GPU needed.
cd "$(dirname "$0")"
CUOBJDUMP=${CUOBJDUMP:-$(command -v cuobjdump || echo /usr/local/cuda-12.4/bin/cuobjdump)}
for f in ${@:-libcudaorbits*.so}; do
  echo "== $f"
  $CUOBJDUMP --dump-resource-usage "$f" 2>/dev/null | awk '
    /Function/ {fn=$2; sub(":","",fn)} /^ *Fatbin|arch = / {}
    /REG:/ && fn ~ /k_(integrate|symp)/ {print fn, $1, $2}' | c++filt -t 2>/dev/null \
    | sed -E 's/^_Z[0-9]+([a-z_]+)I(L[bi])?([0-9]+)E.*(REG:[0-9]+) (STACK:[0-9]+)/  \1<\3> \4 \5/'
done
