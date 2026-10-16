# H100 backend tests: galpy backend/gap-0i-forced-backend @ 50df80f3b (PR #1718)

**Preliminary: torch runs 1 and 2 only; jax (run 3) still running and will be added here.**

## Environment

- GPU: NVIDIA H100 80GB HBM3 (driver 580.178.04), exclusive (nvidia-smi before/after: no other process; see `*.env.txt`).
- galpy 50df80f3b3f2c62ef21cfc1e587c16f62c77dc85 (clean worktree, C extension built in place, PYTHONPATH at the checkout).
- **torch 2.14.1+cu130 (CUDA 13.0)**, jax 0.11.2 (jax-cuda12-plugin 0.11.2), diffrax 0.7.2, torchdiffeq 0.2.5, torchode 1.0.1,
  numpy 2.5.3, scipy 1.18.1, pytest 9.1.1, pytest-xdist 3.8.0, pytest-timeout 2.4.0 (added for --timeout).
  Note: the `galpy-h100` env has torch 2.14.1, not 2.12 (2.12 was a separate, since-deleted triage env).
- `pytest -p no:cacheprovider --timeout 1800 -W error::DeprecationWarning -W error::FutureWarning -n 2 -rA --tb=short --device cuda`;
  XLA_PYTHON_CLIENT_PREALLOCATE=false, OMP_NUM_THREADS=1, fresh TORCHINDUCTOR_CACHE_DIR / TRITON_CACHE_DIR per job in node-local /dev/shm.
- Job script `gaptest.sbatch`, test lists in `spec/`, classifier `gapclass.py`.

Classes: **known** = listed as known from the TITAN round; **REAL GPU defect** = numpy conversion of a CUDA tensor,
device mismatch, inductor/Triton compile error or wrong value; **tolerance miss**; **infrastructure** (OOM, timeout, missing package).

## run 1: torch, compile files (compile_workflows, torchode, tracectx, kingdf, inbackend_ode, orbit_stm, jit_mode)

Counts: {'failed': 7, 'passed': 483, 'skipped': 16}

- 7 known: dynamo traces numpy under set_default_device

| test | classification | error (last line) |
|---|---|---|
| `tests/test_backend_compile_workflows.py::test_workflow_compiled_matches_eager[torch-orbit_c_integrator]` | known: dynamo traces numpy under set_default_device (upstream) | Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo" |
| `tests/test_backend_compile_workflows.py::test_workflow_compiled_matches_eager[torch-spray_sample]` | known: dynamo traces numpy under set_default_device (upstream) | Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo" |
| `tests/test_backend_compile_workflows.py::test_workflow_compiled_matches_eager[torch-spray_track]` | known: dynamo traces numpy under set_default_device (upstream) | Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo" |
| `tests/test_backend_compile_workflows.py::test_workflow_compiled_matches_eager[torch-inductor-spray_sample]` | known: dynamo traces numpy under set_default_device (upstream) | Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo" |
| `tests/test_backend_compile_workflows.py::test_workflow_compiled_matches_eager[torch-inductor-spray_track]` | known: dynamo traces numpy under set_default_device (upstream) | Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo" |
| `tests/test_backend_orbit_stm.py::test_torch_compile_orbit_integration_matches_eager[dop853_c]` | known: dynamo traces numpy under set_default_device (upstream) | Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo" |
| `tests/test_backend_orbit_stm.py::test_torch_compile_orbit_integration_matches_eager[symplec4_c]` | known: dynamo traces numpy under set_default_device (upstream) | Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo" |

## run 2: torch, all 92 tests/test_backend_*.py (includes param_sweep and run 1's files)

Counts: {'passed': 17702, 'failed': 86, 'skipped': 127, 'xfailed': 1}

- 45 REAL GPU defect: numpy conversion of a CUDA tensor -- 37 inside galpy (Orbits.py:7610 _call_internal 16; sphericaldf.py:1667 _setup_rphi_interpolator 13;
  actionAngleStaeckel.py:870/957 and actionAngleStaeckel_c.py:641, 8), 8 in the test code (test_backend_scf.py:1204, 2; test_backend_streamgapdf.py:638, 6)
- 21 known: numpy conversion of a CUDA tensor
- 9 REAL GPU defect: device mismatch
- 7 known: dynamo traces numpy under set_default_device
- 3 known: last-digit miss in test_analytic_actions_*
- 1 REAL GPU defect: wrong value (singular AᵀA, see traceback below)

| test | classification | error (last line) |
|---|---|---|
| `tests/test_backend_eddingtondf.py::test_plain_backend_potential_differentiated_denspot[torch-sigmar]` | known: numpy conversion of a CUDA tensor (fixed on unpushed follow-up) | TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first. |
| `tests/test_backend_eddingtondf.py::test_plain_backend_potential_differentiated_denspot[torch-vmomentdensity]` | known: numpy conversion of a CUDA tensor (fixed on unpushed follow-up) | TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first. |
| `tests/test_backend_adiabaticcontraction.py::test_torch_values_vs_numpy[cautun-True]` | known: numpy conversion of a CUDA tensor (fixed on unpushed follow-up) | TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first. |
| `tests/test_backend_adiabaticcontraction.py::test_torch_values_vs_numpy[cautun-False]` | known: numpy conversion of a CUDA tensor (fixed on unpushed follow-up) | TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first. |
| `tests/test_backend_adiabaticcontraction.py::test_torch_values_vs_numpy[blumenthal-True]` | known: numpy conversion of a CUDA tensor (fixed on unpushed follow-up) | TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first. |
| `tests/test_backend_adiabaticcontraction.py::test_torch_values_vs_numpy[blumenthal-False]` | known: numpy conversion of a CUDA tensor (fixed on unpushed follow-up) | TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first. |
| `tests/test_backend_adiabaticcontraction.py::test_torch_values_vs_numpy[gnedin-True]` | known: numpy conversion of a CUDA tensor (fixed on unpushed follow-up) | TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first. |
| `tests/test_backend_adiabaticcontraction.py::test_torch_values_vs_numpy[gnedin-False]` | known: numpy conversion of a CUDA tensor (fixed on unpushed follow-up) | TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first. |
| `tests/test_backend_adiabaticcontraction.py::test_torch_backward_vs_finite_difference[cautun]` | known: numpy conversion of a CUDA tensor (fixed on unpushed follow-up) | TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first. |
| `tests/test_backend_adiabaticcontraction.py::test_torch_backward_vs_finite_difference[blumenthal]` | known: numpy conversion of a CUDA tensor (fixed on unpushed follow-up) | TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first. |
| `tests/test_backend_adiabaticcontraction.py::test_torch_backward_vs_finite_difference[gnedin]` | known: numpy conversion of a CUDA tensor (fixed on unpushed follow-up) | TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first. |
| `tests/test_backend_adiabaticcontraction.py::test_torch_compile_evaluation` | known: numpy conversion of a CUDA tensor (fixed on unpushed follow-up) | TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first. |
| `tests/test_backend_adiabaticcontraction.py::test_amp_honoured_on_the_backend[torch]` | known: numpy conversion of a CUDA tensor (fixed on unpushed follow-up) | TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first. |
| `tests/test_backend_coerce.py::test_lazy_cond_takes_one_arm[torch]` | known: numpy conversion of a CUDA tensor (fixed on unpushed follow-up) | TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first. |
| `tests/test_backend_compile_workflows.py::test_workflow_compiled_matches_eager[torch-orbit_c_integrator]` | known: dynamo traces numpy under set_default_device (upstream) | Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo" |
| `tests/test_backend_compile_workflows.py::test_workflow_compiled_matches_eager[torch-spray_sample]` | known: dynamo traces numpy under set_default_device (upstream) | Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo" |
| `tests/test_backend_compile_workflows.py::test_workflow_compiled_matches_eager[torch-spray_track]` | known: dynamo traces numpy under set_default_device (upstream) | Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo" |
| `tests/test_backend_linalg.py::test_real_eig_torch_vmap_grad_matches_eager` | known: numpy conversion of a CUDA tensor (fixed on unpushed follow-up) | TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first. |
| `tests/test_backend_movingobject.py::test_eager_returns_backend_array[torch]` | REAL GPU defect: numpy conversion of a CUDA tensor (galpy/orbit/Orbits.py:7610) | TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first. |
| `tests/test_backend_movingobject.py::test_grad_vs_fd_R[torch]` | REAL GPU defect: numpy conversion of a CUDA tensor (galpy/orbit/Orbits.py:7610) | TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first. |
| `tests/test_backend_movingobject.py::test_grad_vs_fd_phi[torch]` | REAL GPU defect: numpy conversion of a CUDA tensor (galpy/orbit/Orbits.py:7610) | TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first. |
| `tests/test_backend_movingobject.py::test_object_orbit_and_kernel_grad_vs_fd[torch-b-data]` | known: numpy conversion of a CUDA tensor (fixed on unpushed follow-up) | TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first. |
| `tests/test_backend_movingobject.py::test_object_orbit_and_kernel_grad_vs_fd[torch-b-forced]` | known: numpy conversion of a CUDA tensor (fixed on unpushed follow-up) | TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first. |
| `tests/test_backend_movingobject.py::test_object_orbit_and_kernel_grad_vs_fd[torch-amp-data]` | known: numpy conversion of a CUDA tensor (fixed on unpushed follow-up) | TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first. |
| `tests/test_backend_movingobject.py::test_object_orbit_and_kernel_grad_vs_fd[torch-amp-forced]` | known: numpy conversion of a CUDA tensor (fixed on unpushed follow-up) | TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first. |
| `tests/test_backend_movingobject.py::test_forced_float_query_returns_backend_array[torch]` | REAL GPU defect: numpy conversion of a CUDA tensor (galpy/orbit/Orbits.py:7610) | TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first. |
| `tests/test_backend_compile_workflows.py::test_workflow_compiled_matches_eager[torch-inductor-spray_sample]` | known: dynamo traces numpy under set_default_device (upstream) | Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo" |
| `tests/test_backend_compile_workflows.py::test_workflow_compiled_matches_eager[torch-inductor-spray_track]` | known: dynamo traces numpy under set_default_device (upstream) | Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo" |
| `tests/test_backend_orbit_accessors.py::test_analytic_actions_backend_orbit_values[jax-staeckel-noc-delta]` | known: last-digit miss in test_analytic_actions_* (fixed on follow-up) | DESIRED: array(0.602894) |
| `tests/test_backend_orbit_accessors.py::test_analytic_actions_backend_orbit_values[torch-staeckel-noc-delta]` | known: last-digit miss in test_analytic_actions_* (fixed on follow-up) | DESIRED: array(1.170418) |
| `tests/test_backend_orbit_accessors.py::test_analytic_actions_unbound_backend_orbits[jax]` | known: last-digit miss in test_analytic_actions_* (fixed on follow-up) | DESIRED: array([1.707444,      nan, 2.58046 ]) |
| `tests/test_backend_orbit_accessors.py::test_backend_orbit_read_under_a_numpy_pin[torch]` | known: numpy conversion of a CUDA tensor (fixed on unpushed follow-up) | TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first. |
| `tests/test_backend_orbit_stm.py::test_torch_compile_orbit_integration_matches_eager[dop853_c]` | known: dynamo traces numpy under set_default_device (upstream) | Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo" |
| `tests/test_backend_orbit_stm.py::test_torch_compile_orbit_integration_matches_eager[symplec4_c]` | known: dynamo traces numpy under set_default_device (upstream) | Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo" |
| `tests/test_backend_paramgrad.py::test_forced_numpy_beats_a_parameter_backend` | known: numpy conversion of a CUDA tensor (fixed on unpushed follow-up) | TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first. |
| `tests/test_backend_param_sweep.py::test_param_grad[MovingObjectPotential-amp-torch-data-eager]` | REAL GPU defect: numpy conversion of a CUDA tensor (galpy/orbit/Orbits.py:7610) | TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first. |
| `tests/test_backend_param_sweep.py::test_param_grad[MovingObjectPotential-R-torch-floatcoords-eager]` | REAL GPU defect: numpy conversion of a CUDA tensor (galpy/orbit/Orbits.py:7610) | TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first. |
| `tests/test_backend_param_sweep.py::test_param_value_on_backend[RotateAndTiltWrapperPotential[galaxy_pa]-galaxy_pa-torch-data]` | REAL GPU defect: device mismatch | RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! |
| `tests/test_backend_param_sweep.py::test_param_value_on_backend[RotateAndTiltWrapperPotential[galaxy_pa]-galaxy_pa-torch-floatcoords]` | REAL GPU defect: device mismatch | RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! |
| `tests/test_backend_param_sweep.py::test_param_value_on_backend[RotateAndTiltWrapperPotential[galaxy_pa]-galaxy_pa-torch-forced]` | REAL GPU defect: device mismatch | RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! |
| `tests/test_backend_param_sweep.py::test_param_value_on_backend[RotateAndTiltWrapperPotential[inclination]-inclination-torch-data]` | REAL GPU defect: device mismatch | RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! |
| `tests/test_backend_param_sweep.py::test_param_value_on_backend[RotateAndTiltWrapperPotential[inclination]-inclination-torch-floatcoords]` | REAL GPU defect: device mismatch | RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! |
| `tests/test_backend_param_sweep.py::test_param_value_on_backend[RotateAndTiltWrapperPotential[inclination]-inclination-torch-forced]` | REAL GPU defect: device mismatch | RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! |
| `tests/test_backend_param_sweep.py::test_param_value_on_backend[RotateAndTiltWrapperPotential[inclination]-sky_pa-torch-data]` | REAL GPU defect: device mismatch | RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! |
| `tests/test_backend_param_sweep.py::test_param_value_on_backend[RotateAndTiltWrapperPotential[inclination]-sky_pa-torch-floatcoords]` | REAL GPU defect: device mismatch | RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! |
| `tests/test_backend_param_sweep.py::test_param_value_on_backend[RotateAndTiltWrapperPotential[inclination]-sky_pa-torch-forced]` | REAL GPU defect: device mismatch | RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! |
| `tests/test_backend_param_sweep.py::test_param_value_on_backend[MovingObjectPotential-amp-torch-data]` | REAL GPU defect: numpy conversion of a CUDA tensor (galpy/orbit/Orbits.py:7610) | TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first. |
| `tests/test_backend_param_sweep.py::test_param_value_on_backend[MovingObjectPotential-amp-torch-floatcoords]` | REAL GPU defect: numpy conversion of a CUDA tensor (galpy/orbit/Orbits.py:7610) | TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first. |
| `tests/test_backend_param_sweep.py::test_param_value_on_backend[MovingObjectPotential-amp-torch-forced]` | REAL GPU defect: numpy conversion of a CUDA tensor (galpy/orbit/Orbits.py:7610) | TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first. |
| `tests/test_backend_param_sweep.py::test_param_value_on_backend[MovingObjectPotential-pb-torch-data]` | REAL GPU defect: numpy conversion of a CUDA tensor (galpy/orbit/Orbits.py:7610) | TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first. |
| `tests/test_backend_param_sweep.py::test_param_value_on_backend[MovingObjectPotential-pb-torch-floatcoords]` | REAL GPU defect: numpy conversion of a CUDA tensor (galpy/orbit/Orbits.py:7610) | TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first. |
| `tests/test_backend_param_sweep.py::test_param_value_on_backend[MovingObjectPotential-pb-torch-forced]` | REAL GPU defect: numpy conversion of a CUDA tensor (galpy/orbit/Orbits.py:7610) | TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first. |
| `tests/test_backend_param_sweep.py::test_param_value_on_backend[MovingObjectPotential-oR-torch-forced]` | REAL GPU defect: numpy conversion of a CUDA tensor (galpy/orbit/Orbits.py:7610) | TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first. |
| `tests/test_backend_param_sweep.py::test_param_value_on_backend[MovingObjectPotential-R-torch-data]` | REAL GPU defect: numpy conversion of a CUDA tensor (galpy/orbit/Orbits.py:7610) | TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first. |
| `tests/test_backend_param_sweep.py::test_param_value_on_backend[MovingObjectPotential-R-torch-floatcoords]` | REAL GPU defect: numpy conversion of a CUDA tensor (galpy/orbit/Orbits.py:7610) | TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first. |
| `tests/test_backend_param_sweep.py::test_param_value_on_backend[MovingObjectPotential-R-torch-forced]` | REAL GPU defect: numpy conversion of a CUDA tensor (galpy/orbit/Orbits.py:7610) | TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first. |
| `tests/test_backend_param_sweep.py::test_param_value_on_backend[actionAngleStaeckel[c=True]-delta-torch-data]` | REAL GPU defect: numpy conversion of a CUDA tensor (galpy/actionAngle/actionAngleStaeckel.py:957) | TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first. |
| `tests/test_backend_param_sweep.py::test_param_value_on_backend[actionAngleStaeckel[c=True]-delta-torch-floatcoords]` | REAL GPU defect: numpy conversion of a CUDA tensor (galpy/actionAngle/actionAngleStaeckel.py:957) | TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first. |
| `tests/test_backend_param_sweep.py::test_param_value_on_backend[actionAngleStaeckel[c=True]-delta-torch-forced]` | REAL GPU defect: numpy conversion of a CUDA tensor (galpy/actionAngle/actionAngleStaeckel.py:957) | TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first. |
| `tests/test_backend_param_sweep.py::test_param_value_on_backend[eddingtondf-a-torch-data]` | REAL GPU defect: numpy conversion of a CUDA tensor (galpy/df/sphericaldf.py:1667) | TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first. |
| `tests/test_backend_param_sweep.py::test_param_value_on_backend[eddingtondf-a-torch-floatcoords]` | REAL GPU defect: numpy conversion of a CUDA tensor (galpy/df/sphericaldf.py:1667) | TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first. |
| `tests/test_backend_param_sweep.py::test_param_value_on_backend[eddingtondf-a-torch-forced]` | REAL GPU defect: numpy conversion of a CUDA tensor (galpy/df/sphericaldf.py:1667) | TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first. |
| `tests/test_backend_param_sweep.py::test_param_value_on_backend[eddingtondf[denspot]-a-torch-data]` | REAL GPU defect: numpy conversion of a CUDA tensor (galpy/df/sphericaldf.py:1667) | TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first. |
| `tests/test_backend_param_sweep.py::test_param_value_on_backend[eddingtondf[denspot]-a-torch-floatcoords]` | REAL GPU defect: numpy conversion of a CUDA tensor (galpy/df/sphericaldf.py:1667) | TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first. |
| `tests/test_backend_param_sweep.py::test_param_value_on_backend[eddingtondf[denspot]-a-torch-forced]` | REAL GPU defect: numpy conversion of a CUDA tensor (galpy/df/sphericaldf.py:1667) | TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first. |
| `tests/test_backend_param_sweep.py::test_param_value_on_backend[eddingtondf[Hernquist]-a-torch-data]` | REAL GPU defect: numpy conversion of a CUDA tensor (galpy/df/sphericaldf.py:1667) | TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first. |
| `tests/test_backend_param_sweep.py::test_param_value_on_backend[eddingtondf[Hernquist]-a-torch-floatcoords]` | REAL GPU defect: numpy conversion of a CUDA tensor (galpy/df/sphericaldf.py:1667) | TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first. |
| `tests/test_backend_param_sweep.py::test_param_value_on_backend[eddingtondf[Hernquist]-a-torch-forced]` | REAL GPU defect: numpy conversion of a CUDA tensor (galpy/df/sphericaldf.py:1667) | TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first. |
| `tests/test_backend_param_sweep.py::test_param_value_on_backend[actionAngleStaeckelGrid-delta-torch-data]` | REAL GPU defect: numpy conversion of a CUDA tensor (galpy/actionAngle/actionAngleStaeckel_c.py:641) | TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first. |
| `tests/test_backend_param_sweep.py::test_param_value_on_backend[actionAngleStaeckelGrid-delta-torch-floatcoords]` | REAL GPU defect: numpy conversion of a CUDA tensor (galpy/actionAngle/actionAngleStaeckel_c.py:641) | TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first. |
| `tests/test_backend_param_sweep.py::test_param_value_on_backend[actionAngleStaeckelGrid-delta-torch-forced]` | REAL GPU defect: numpy conversion of a CUDA tensor (galpy/actionAngle/actionAngleStaeckel_c.py:641) | TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first. |
| `tests/test_backend_param_sweep.py::test_param_value_on_backend[osipkovmerrittdf-a-torch-data]` | REAL GPU defect: numpy conversion of a CUDA tensor (galpy/df/sphericaldf.py:1667) | TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first. |
| `tests/test_backend_param_sweep.py::test_param_value_on_backend[osipkovmerrittdf-a-torch-floatcoords]` | REAL GPU defect: numpy conversion of a CUDA tensor (galpy/df/sphericaldf.py:1667) | TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first. |
| `tests/test_backend_param_sweep.py::test_param_value_on_backend[osipkovmerrittdf-a-torch-forced]` | REAL GPU defect: numpy conversion of a CUDA tensor (galpy/df/sphericaldf.py:1667) | TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first. |
| `tests/test_backend_param_sweep.py::test_param_value_on_backend[constantbetadf-a-torch-forced]` | REAL GPU defect: numpy conversion of a CUDA tensor (galpy/df/sphericaldf.py:1667) | TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first. |
| `tests/test_backend_param_sweep.py::test_param_value_on_backend[Orbit.analytic[staeckel]-delta-torch-data]` | REAL GPU defect: numpy conversion of a CUDA tensor (galpy/actionAngle/actionAngleStaeckel.py:870) | TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first. |
| `tests/test_backend_param_sweep.py::test_param_value_on_backend[Orbit.analytic[staeckel]-delta-torch-floatcoords]` | REAL GPU defect: numpy conversion of a CUDA tensor (galpy/actionAngle/actionAngleStaeckel.py:870) | TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first. |
| `tests/test_backend_scf.py::test_from_density_timedep_backend_matches_numpy[const-torch]` | REAL GPU defect: numpy conversion of a CUDA tensor (tests/test_backend_scf.py:1204) | TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first. |
| `tests/test_backend_scf.py::test_from_density_timedep_backend_matches_numpy[tdep-torch]` | REAL GPU defect: numpy conversion of a CUDA tensor (tests/test_backend_scf.py:1204) | TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first. |
| `tests/test_backend_streamdf.py::test_streamdf_progenitor_grad_with_c_aA_is_loud` | REAL GPU defect: wrong value (singular AᵀA in actionAngleIsochroneApprox._actionsFreqsAngles_backend, actionAngleIsochroneApprox.py:1236; full traceback below) | torch._C._LinAlgError: linalg.inv: (Batch element 0): The diagonal element 3 is zero, the inversion could not be completed because the input matrix is singular. |
| `tests/test_backend_streamgapdf.py::test_gapdf_eval_grad_vs_fd[torch-approx-density]` | REAL GPU defect: numpy conversion of a CUDA tensor (tests/test_backend_streamgapdf.py:638) | TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first. |
| `tests/test_backend_streamgapdf.py::test_gapdf_eval_grad_vs_fd[torch-approx-meanOmega]` | REAL GPU defect: numpy conversion of a CUDA tensor (tests/test_backend_streamgapdf.py:638) | TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first. |
| `tests/test_backend_streamgapdf.py::test_gapdf_eval_grad_vs_fd[torch-higherorder-density]` | REAL GPU defect: numpy conversion of a CUDA tensor (tests/test_backend_streamgapdf.py:638) | TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first. |
| `tests/test_backend_streamgapdf.py::test_gapdf_eval_grad_vs_fd[torch-higherorder-meanOmega]` | REAL GPU defect: numpy conversion of a CUDA tensor (tests/test_backend_streamgapdf.py:638) | TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first. |
| `tests/test_backend_streamgapdf.py::test_gapdf_eval_grad_vs_fd[torch-direct-density]` | REAL GPU defect: numpy conversion of a CUDA tensor (tests/test_backend_streamgapdf.py:638) | TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first. |
| `tests/test_backend_streamgapdf.py::test_gapdf_eval_grad_vs_fd[torch-direct-meanOmega]` | REAL GPU defect: numpy conversion of a CUDA tensor (tests/test_backend_streamgapdf.py:638) | TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first. |

## Notes

- run 2b (param_sweep alone, job 1108802) was cancelled after 2 min: run 2a already included param_sweep
  (`--ignore` does not drop a file that the glob lists explicitly). Its partial log/env are kept for the record.
- run 1's 7 known failures appear again in run 2 (run 2 covers run 1's files).

## Full traceback: tests/test_backend_streamdf.py::test_streamdf_progenitor_grad_with_c_aA_is_loud

Construction of the torch streamdf fails on CUDA before the expected `RuntimeError('second derivatives through the C ...')`:
the normal matrix AᵀA of actionAngleIsochroneApprox's angle fit is singular on the GPU.

```
tests/test_backend_streamdf.py:831: in test_streamdf_progenitor_grad_with_c_aA_is_loud
    sdf = streamdf(
galpy/df/streamdf.py:454: in __init__
    self._determine_stream_track(nTrackChunks)
galpy/df/streamdf.py:1549: in _determine_stream_track
    return self._determine_stream_track_backend()
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
galpy/df/streamdf.py:1883: in _determine_stream_track_backend
    outs = _vmap_track_chunks(xp, single_at(xv0_all), xv0_all, thetasTrack)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
galpy/df/streamdf.py:5357: in _vmap_track_chunks
    outs = [single(xv0_all[ii], thetasTrack[ii]) for ii in range(xv0_all.shape[0])]
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
galpy/df/streamdf.py:1871: in single
    return _determine_stream_track_single_backend(
galpy/df/streamdf.py:5461: in _determine_stream_track_single_backend
    tacfs = aA.actionsFreqsAngles(xv0[0], xv0[1], xv0[2], xv0[3], xv0[4], xv0[5])
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
galpy/util/conversion.py:1305: in wrapper
    return method(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^
galpy/backend/_namespaces.py:840: in wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
galpy/util/conversion.py:1260: in wrapped
    return method(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^
galpy/actionAngle/actionAngle.py:395: in actionsFreqsAngles
    return _forward(self, method, args, kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
galpy/actionAngle/actionAngle.py:131: in _forward
    out = method(*_point_1d(_lift_args(obj, args, point_1d=True)), **kwargs)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
galpy/actionAngle/actionAngleIsochroneApprox.py:329: in _actionsFreqsAngles
    return self._actionsFreqsAngles_backend(
galpy/actionAngle/actionAngleIsochroneApprox.py:1236: in _actionsFreqsAngles_backend
    atainv = xp.linalg.inv(xp.matmul(ATt, A))  # (no, 2+nn, 2+nn)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/home/bovy/miniconda3/envs/galpy-h100/lib/python3.13/site-packages/torch/utils/_device.py:122: in __torch_function__
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
E   torch._C._LinAlgError: linalg.inv: (Batch element 0): The diagonal element 3 is zero, the inversion could not be completed because the input matrix is singular.
```
