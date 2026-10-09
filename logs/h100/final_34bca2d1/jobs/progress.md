# H100 final campaign progress (galpy 34bca2d1d, galpy-jax gpu-h100 = origin/main 2b3668c)
- 2026-10-08: galpy pinned + rebuilt; gpu-h100 reset to origin/main + force-pushed; 14 data/h100 JSONs backed up (.bak_pre_final); old stage logs moved to logs/pre_final/.
- Chain submitted (chain_ids.txt). Test job script: jobs/fullsuite.sbatch (-n 2, --ignore amuse/snapshotpotential: modules not installed). Group+XLA: jobs/group.sbatch, jobs/edf_grad.py. Stages: jobs/measure.sbatch -> route.sh pre / tools/trillium_stage.sh / route.sh post.
- Notes for report: tests/test_backend_jit_workflows.py does not exist -> used test_backend_compile_workflows.py -k torch-inductor. g17 has --cpu-only but no GPU-only switch -> g17 stage re-measures CPU series too. No pytest-timeout installed (none used before either).
- 2026-10-09 00:40 check-in: torch full suite (1060557) COMPLETED 4:19:09, pytest exit 1: 10913 passed / 1925 failed / 217 skipped / 0 xfail
  (ledger tests/backend_xfail.txt has no eager torch/jax entries left, so nothing xfails; CI's CPU torch run is green => classic failures are CUDA-only).
  Classification in logs/fullsuite_2026-10-08/torch_failures.tsv (jobs/classify.py):
  * 1908 classic-suite (non test_backend_*): 1536 = the TEST calls numpy on a CUDA result (deepest own frame in tests/*, e.g. test_potential.py:2243 test_amp_mult_divide 106x);
    219 = galpy's deliberate guard Orbits.py:312 _refuse_numpy_integrator_on_cuda (dop853/odeint/leapfrog/rk*_c/symplec*_c with a forced torch CUDA backend);
    19 actionAngleTorus _reject_backend; ~80 galpy scipy-integrand/lambda paths on CUDA tensors (SCFPotential.py:2184/2012/2588/2624/917, KuijkenDubinski...:72);
    ~100 test-side Quantity/unit/.to() on tensors; few misc (astroquery/astropy not installed).
  * 17 in test_backend_* (the meaningful ones):
    - 9 diskdf test_moment_array_R_grad_vs_fd + profile_parameter_grad_numpy_array_R: TEST-side .numpy() on CUDA (tests/test_backend_diskdf.py:325, :347) [new tests, #1653]
    - diskdf test_moment_array_R_under_jit[torch]: galpy numpy constant under compile, surfaceSigmaProfile.py:223 sigma2 (mul FakeTensor*2.0 -> ndarray) [#1653, galpy CUDA bug, same family as #1645]
    - VerticalInverse test_cuda_potential_parameters_cpu_default_device[True/False]: galpy/backend/_torch/optimize.py:106 _newton_root device mismatch (min on cuda, other on cpu) [new since #1641 Newton]
    - multi test_forked_child_is_capped_at_one_torch_thread: children got 24 torch threads, expected 1 (likely #1645's parallel_map-after-CUDA path)
    - streamgapdf test_backend_impact_angle_value: 37/5220 mismatched at rtol 1e-12 [new test, #1652]
    - jax nondeterministic: evolveddiskdf meanvT/vertexdev, finite_part_quad[jax] (as before)
- jax full suite 1061096 RUNNING since ~00:20.
- 2026-10-09 05:50: jax full suite 1061096 slow: 77% at 4:26, 78% at 5:26 (limit 11:00, started ~00:20 -> killed ~11:20). scontrol TimeLimit increase: permission denied.
  junitxml/-rA summary are written only at session end -> a timeout loses everything. PLAN: if still running at >= ~10:20 elapsed (~10:40 wall),
  `scancel --signal=INT --full 1061096` (pytest KeyboardInterrupt -> writes junit + summary of completed tests), then submit a jax job for the
  not-yet-run nodeids only (collected minus junit testcases), before/inside the chain (group 1061097 starts on afterany when 1061096 ends).
- 2026-10-09 08:40 check-in:
  * jax full suite 1061096 COMPLETED 7:32:43: 12689 passed / 142 failed / 224 skipped / 0 xfail. Classified -> logs/fullsuite_2026-10-08/jax_failures.tsv.
    Backend-file failures (4): multi test_forked_child_is_capped_at_one_torch_thread (24 threads), VerticalInverse test_cuda_potential_parameters_cpu_default_device[True/False]
    (_torch/optimize.py:106 _newton_root device mismatch), streamgapdf test_backend_impact_angle_value. evolveddiskdf + finite_part_quad PASSED this run.
    Classic 138: mostly test-side Quantity .to/.value on jax arrays (~75), actionAngleTorus reject (19), astroquery missing (3), 'a#s' FileNotFoundError (8, check), plot parsing (4).
  * group/XLA 1061097 COMPLETED 0:31: torch_failed list 47/47 PASS; jax pair 2/2 x3 PASS; compile_workflows -k torch-inductor 5/5 PASS (inductor check of #1647);
    test_backend_torchode.py 18/18; kingdf torchode/compile 2/2; tracectx forced_backend PASS.
    XLA: no flag 1/3 runs correct (meanvT 0.08852278281994709, rel 5.6e-8 vs FD), 2/3 wrong (0.08857196664931519, rel 5.56e-4); vertexdev same pattern
    (0.3111197362136613 vs 0.31093183514607714). --xla_gpu_deterministic_ops=true: 3/3 identical correct (0.08852278281994708 / 0.3111197362136612).
    --xla_gpu_autotune_level=0: 3/3 identical correct (same values). => autotuner-selected kernel gives the bad gradient; either flag fixes it.
  * g2g5g6 1061098 COMPLETED 1:26: data/h100 g2/g5/g6 rewritten (H100 stamp, galpy_commit 34bca2d1d added), main's files restored. g2 1e7 jax GPU 0.1457->0.1386 ms, torch 0.5873->0.5833;
    g5 jax grad 0.1447->0.1412, torch grad 1.222->1.213; g6 see report.
  * orbits 1061099 RUNNING.
- 2026-10-09 08:50: orbits 1061099 COMPLETED 9:18, all 5 scripts ok, 5 data/h100 JSONs rewritten (H100, galpy_commit added). us/orbit old->new:
  jax Dopri8 hiN 3e5 24.68->24.45; Dopri5 hiN 3e5 18.23->18.07; torch 1e6 9.22->9.89 (3e5 15.25->17.47); jax lowN 3e4 81.47->81.25; Dopri5 lowN 3e4 66.14->67.29.
  hiN files now end at 3e5 (fresh write); hiN2 re-adds 1e6/3e6 (and torch 3e6). newfigs 1061100 RUNNING.
- 2026-10-09 10:30: newfigs 1061100 COMPLETED 34:56; hiN2 1061101 COMPLETED 14:04 (to verify). g15 1061102 + g17 1061103 CANCELLED by root after 3-4 s on trig0021 (empty out; node transient, MIXED now);
  g17torch 1061104 then ran on the OLD g17 data (torch series merged into old file) -> superseded by the rerun. Resubmitted with --exclude=trig0021: g15 1062736 -> g17 1062737 -> g17torch 1062738 (chain_ids.txt updated).
  newfigs/hiN2: all measure commands ok; only the in-stage harness render failed (gV2: x/y 12 vs 11 = routing artefact, the H100 file sits in data/X.json during the
  stage while the figure expects main's file there). Verified on the login node: main's data/*.json + new data/h100 render all 9 figures (gV2 gD3 gS5 gS4 g8 g12 g2 g5 g6) fine.
  New gV2 h100 file internally consistent (eval series 12 = Ns). newfigs: gS4 eager 1e6 OOM skips as before (jax eval 22.4 GiB, torch sample/eval 16.3/22.4 GiB).
- USER RULE (2026-10-09 ~10:50): if a job sits PENDING for resources for a long time (Reason=Resources/Priority, not Dependency), switch it to account
  ctb-ripperda: `scontrol update JobId=<id> Account=ctb-ripperda` (pending job), or cancel + resubmit with --account=ctb-ripperda (keep the afterany chain).
- 2026-10-09 11:50: g15 1062736 ended OUT_OF_MEMORY (0:125) after 59:34 on trig0003 -- same as run 1: measure_largeN.py host-OOM-killed (quarter-node 187.5 GB)
  at N=1048576 after the GPU series' own CUDA-OOM stops (torchdiffeq_gpu at 262144, diffrax_gpu at 1048576); incremental saves keep all points below.
  Other 4 scripts ok; route post ran (galpy_commit added). Not resubmitted (same limit would hit again). g17 1062737 RUNNING (trig0003).
- 2026-10-09 12:40: user: other agent added g17gpu/g15gpu (main 2233ac1). g17 1062737 (1:07 in, still on torch CPU) + g17torch 1062738 CANCELLED by me.
  Tree restored (main's paper files; h100 g17 json reset to .bak_pre_final). gpu-h100 fast-forwarded to origin/main (12 commits, none touching data/h100); h100 changes stashed+popped.
  g15 keeps its run (GPU series fine; CPU in it unused). Submitted g17gpu 1063179 (route.sh knows g17gpu). g17torch only if g17gpu's torch GPU points come back NaN.
