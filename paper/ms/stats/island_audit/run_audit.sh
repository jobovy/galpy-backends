#!/bin/bash
# per-file / per-CI-group processes, file order preserved (as CI runs them)
cd $GALPY
export CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 PYTHONPATH=$GALPY:/var/tmp/claude-probes/audit
A=/var/tmp/claude-probes/audit
for bk in torch jax; do
  rm -rf $A/out2_$bk $A/logs2_$bk; mkdir -p $A/logs2_$bk
  JOBS=$A/jobs_$bk.txt; : > $JOBS
  for g in 1 2 3 4 5; do echo "tests/test_potential.py --splits 5 --group $g" >> $JOBS; done
  for g in 1 2 3 4 5; do echo "tests/test_orbit.py tests/test_orbits.py --splits 5 --group $g --durations-path tests/.test_orbit_durations" >> $JOBS; done
  for f in $(ls tests/test_*.py | grep -v "test_backend\|test_actionAngleTorus\|test_potential.py\|test_orbit.py\|test_orbits.py"); do echo "$f" >> $JOBS; done
  s=$(date +%s)
  cat $JOBS | nl -nln | xargs -P 18 -L 1 bash -c 'i=$0; MC=12; case "$*" in *"group 5 --dur"*) MC=32;; esac; ISLAND_AUDIT_OUT='$A'/out2_'$bk' timeout 7200 taskset -c 12-47 /var/tmp/claude-scratch-restore/scratchpad/memcap.sh $MC python -m pytest "$@" -p island_audit --backend '$bk' -q -p no:cacheprovider --timeout 3600 -rf > '$A'/logs2_'$bk'/job_$i.log 2>&1'
  echo "$bk $(( $(date +%s) - s ))s $(cat $A/logs2_$bk/*.log | grep -cE '^(FAILED|ERROR)') failed lines" >> $A/summary2.txt
done
echo DONE >> $A/summary2.txt
