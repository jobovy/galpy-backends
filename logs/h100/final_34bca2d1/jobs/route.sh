#!/bin/bash
# route.sh pre|post <stage>: run a repo stage against the H100 copies (data/h100/) instead of
# main's data/*.json (the dev box's). pre: data/h100/X -> data/X; post: data/X -> data/h100/X
# (+ galpy_commit if absent), then main's files under paper/ restored (PDFs, manifest, data/X).
set -euo pipefail
R=$SCRATCH/galpy-h100/galpy-jax; F=paper/figures; cd $R
case $2 in
  g2g5g6)  L="g2_throughput/g2_throughput.json g5_autodiff_overhead/g5_autodiff_overhead.json g6_jit_amortization/g6_jit_amortization.json" ;;
  orbits)  L="_shared/orbit_speed/jax_gpu.json _shared/orbit_speed/jax_gpu_hiN.json g12_inbackend_orbit_gpu/jax_dopri5_gpu.json g12_inbackend_orbit_gpu/jax_dopri5_gpu_hiN.json g12_inbackend_orbit_gpu/torch_cuda.json" ;;
  hiN2)    L="_shared/orbit_speed/jax_gpu_hiN.json g12_inbackend_orbit_gpu/jax_dopri5_gpu_hiN.json g12_inbackend_orbit_gpu/torch_cuda.json gV2_vertical_inverse_speed/gV2_vertical_inverse_speed.json gD3_spherical_df_speed/gD3_spherical_df_speed.json" ;;
  newfigs) L="gV2_vertical_inverse_speed/gV2_vertical_inverse_speed.json gD3_spherical_df_speed/gD3_spherical_df_speed.json gS5_streamspray_speed/gS5_streamspray_speed.json gS4_streamdf_speed/gS4_streamdf_speed.json" ;;
  g15|g15gpu) L="g15_icderiv_speed/icderiv.json" ;;
  g17|g17torch|g17gpu) L="g17_staeckel_actionangle_speed/g17_staeckel_actionangle_speed.json" ;;
  *) echo "unknown stage $2"; exit 2 ;;
esac
SHA=$(git -C $SCRATCH/galpy-h100/galpy-fb rev-parse HEAD)
for x in $L; do d=$F/$(dirname $x)/data; b=$(basename $x)
  if [ $1 = pre ]; then cp $d/h100/$b $d/$b; echo "pre  $d/h100/$b -> $d/$b"
  else cp $d/$b $d/h100/$b
    python3 - "$d/h100/$b" "$SHA" <<'PY'
import json, sys
p, sha = sys.argv[1], sys.argv[2]; d = json.load(open(p))
if '"galpy_commit"' not in json.dumps(d): d["galpy_commit"] = sha; json.dump(d, open(p, "w"), indent=2); print("    + galpy_commit", sha[:9], p)
else: print("    galpy_commit already present", p)
PY
  fi
done
if [ $1 = post ]; then git checkout -q origin/main -- paper/ ':(exclude)paper/**/data/h100/**'; git clean -fdq -- paper/ ':(exclude)paper/**/data/h100/**'; fi
git status --short -- paper/ | grep -v "data/h100/" || true
