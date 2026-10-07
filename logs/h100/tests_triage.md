# Triage of the CUDA test failures on the H100 (tests run 2: 47 torch-suite + 2 jax-suite)

**Method.** I reran only the failing tests (`*_failed.txt`) on the H100, with
`--device cuda` and the same `--backend` as the suite they failed in. That was
done in two conda environments that are identical except for torch:

| env | torch | jax | galpy |
|---|---|---|---|
| `galpy-h100` | 2.14.1+cu130 | 0.11.2 | `feat/backends` 3d9ac516 (#1641 merged) + #1642 cherry-picked (0465ad32) |
| `galpy-h100-t212` (clone + `pip install torch==2.12.0 --index-url .../whl/cu126`) | 2.12.0+cu126 | 0.11.2 | same |

Note that the galpy under test is newer than in tests run 2 (70165d10 + #1639).
The two jax-suite tests were each run 3× in each env (`JAXREP=3`), and once
more inside each torch-list run. Stage: `tools/trillium_stage.sh triage`
(TRIAGE_TAG t214 / t212). Outputs are in `t214/`, `t212/` (pytest `-rA --tb=long`
text, gzipped), and `outcomes_t214_t212.tsv` (outcome and first error line in
both envs). `parse.py` and `frames.py` extract these from the junit XML.

## Result

- **Not a single torch-2.14 regression.** All 42 torch failures fail identically
  under torch 2.12 and 2.14. One test's message differs: `test_forced_backend_survives_a_fullgraph_compile`
  is `Unsupported: Observed exception` on 2.14 and `fake tensor call` on 2.12;
  both are the numpy-constant problem below.
- **2 fixed upstream:** `test_misalignment_traces[False/True]`. #1641's tip
  loosens the atol to 1e-14, so #1642's duplicate of that change cherry-picked
  as an empty commit and was skipped.
- **3 jax tests are nondeterministic on CUDA,** with the same jax in both envs
  (details below).

| class | count | what |
|---|---|---|
| (b) galpy CUDA bug: numpy operand under `torch.compile` | 26 | A numpy ndarray operand reaches dynamo next to a CUDA FakeTensor (`'ndarray' object has no attribute 'mul'/'sub'/'neg'`). The operand is a precomputed attribute or a numpy/scipy constant that was not passed through `galpy.backend._namespaces.like()`. |
| (b) galpy CUDA bug, other | 2 | `parallel_map` forks after CUDA init; the dop853_c handoff calls `.numpy()` on a grad-tracking tensor under compile |
| (b') test-suite bug | 7 | The test itself calls `.numpy()` on a CUDA result. galpy's value is fine; only `--device cuda` exposes it. |
| third-party: torchode + `torch.compile` on CUDA | 7 | `torchode/adjoints.py:125` mutates a graph input in place, then `:199` records a CUDA event; dynamo refuses it. Not galpy code; galpy only reaches it via `method="torchode"` under compile. |
| (c) jax nondeterminism / last-bit tolerance | 3 | see below |
| fixed upstream | 2 | `test_misalignment_traces` |

## The 2 (+1) jax tests: nondeterministic, not a stable tolerance miss

`test_backend_evolveddiskdf.py::test_moment_grad_through_evolution_vs_fd[meanvT-jax|vertexdev-jax]`
compares `jax.grad` through the orbit evolution with a central finite
difference at rtol 2e-4. The finite difference is identical every time. The
autodiff value changes from run to run:

| run | meanvT grad (FD 0.088523) | vertexdev grad (FD 0.31112) | result |
|---|---|---|---|
| tests run 2, jax suite | 0.088474 | 0.311308 | fail |
| triage t214, torch-list run | (within rtol) | (within rtol) | pass |
| triage t214, jax ×3 | pass, pass, 0.088572 | pass, pass, 0.310932 | 2 pass / 1 fail |
| triage t212, torch-list run | 0.088572 | 0.310932 | fail |
| triage t212, jax ×3 | pass ×3 | pass ×3 | pass |

So the gradient takes at least three distinct values on the H100 (two of them
±4.9e-5 around FD). The scatter is ~5.6e-4 relative, far above round-off. That
points to a discrete choice in the traced evolution, such as an adaptive step
count or a reduction order, that flips between runs on the GPU. It is
classified as **nondeterminism (c)**, not a tolerance miss. Not investigated
further (no galpy edits).

`test_backend_quadrature.py::test_finite_part_quad_batches_over_its_limits[jax]`
uses atol=0 and fails on the last bit (10.740408962032504 vs ...506).
Outcomes: failed in tests runs 1 and 2 and in t212, passed in t214. This is a
nondeterministic last ulp in the GPU reduction.

## Per test

| test | torch 2.14 | torch 2.12 | class | where | diagnosis |
|---|---|---|---|---|---|
| `test_backend_compile_workflows.py::test_workflow_compiled_matches_eager[torch-inductor-kingdf_W0]` | failed | failed | galpy CUDA bug (b): numpy constant under compile | galpy/df/kingdf.py:470 in _dens_W | a numpy ndarray operand meets a CUDA FakeTensor in `mul`: `out = xp.exp(Ws) * _bspecial.erf(sqW) - _TWOOVERSQRTPI * sqW * (` |
| `test_backend_compile_workflows.py::test_workflow_compiled_matches_eager[torch-inductor-orbit_potential_parameter]` | failed | failed | torchode + compile on CUDA (third-party) | torchode/adjoints.py:125,199 | torchode mutates a graph input in place (`y_eval[...] = ...`), then records a CUDA event; dynamo rejects it under compile |
| `test_backend_compile_workflows.py::test_workflow_compiled_matches_eager[torch-inductor-qdf_constructor_parameters]` | failed | failed | galpy CUDA bug (b): numpy constant under compile | galpy/potential/PowerSphericalPotentialwCutoff.py:117 in torch_dynamo_resume_in__rforce_at_120 | a numpy ndarray operand meets a CUDA FakeTensor in `mul`: `-2.0` |
| `test_backend_compile_workflows.py::test_workflow_compiled_matches_eager[torch-inductor-spray_sample]` | failed | failed | torchode + compile on CUDA (third-party) | torchode/adjoints.py:125,199 | torchode mutates a graph input in place (`y_eval[...] = ...`), then records a CUDA event; dynamo rejects it under compile |
| `test_backend_compile_workflows.py::test_workflow_compiled_matches_eager[torch-inductor-spray_track]` | failed | failed | torchode + compile on CUDA (third-party) | torchode/adjoints.py:125,199 | torchode mutates a graph input in place (`y_eval[...] = ...`), then records a CUDA event; dynamo rejects it under compile |
| `test_backend_compile_workflows.py::test_workflow_compiled_matches_eager[torch-kingdf_W0]` | failed | failed | galpy CUDA bug (b): numpy constant under compile | galpy/df/kingdf.py:470 in _dens_W | a numpy ndarray operand meets a CUDA FakeTensor in `mul`: `out = xp.exp(Ws) * _bspecial.erf(sqW) - _TWOOVERSQRTPI * sqW * (` |
| `test_backend_compile_workflows.py::test_workflow_compiled_matches_eager[torch-orbit_c_integrator]` | failed | failed | galpy CUDA bug (b) | dop853_c handoff (frame stripped by the test's `raise ... from None`) | a grad-tracking tensor reaches numpy (`.numpy()` without detach) on the C-integrator path under compile |
| `test_backend_compile_workflows.py::test_workflow_compiled_matches_eager[torch-orbit_potential_parameter]` | failed | failed | torchode + compile on CUDA (third-party) | torchode/adjoints.py:125,199 | torchode mutates a graph input in place (`y_eval[...] = ...`), then records a CUDA event; dynamo rejects it under compile |
| `test_backend_compile_workflows.py::test_workflow_compiled_matches_eager[torch-qdf_constructor_parameters]` | failed | failed | galpy CUDA bug (b): numpy constant under compile | galpy/potential/PowerSphericalPotentialwCutoff.py:117 in torch_dynamo_resume_in__rforce_at_120 | a numpy ndarray operand meets a CUDA FakeTensor in `mul`: `-2.0` |
| `test_backend_compile_workflows.py::test_workflow_compiled_matches_eager[torch-qdf_density]` | failed | failed | galpy CUDA bug (b): numpy constant under compile | galpy/potential/PowerSphericalPotentialwCutoff.py:246 in torch_dynamo_resume_in__mass_at_249 | a numpy ndarray operand meets a CUDA FakeTensor in `mul`: `2.0` |
| `test_backend_compile_workflows.py::test_workflow_compiled_matches_eager[torch-spray_sample]` | failed | failed | torchode + compile on CUDA (third-party) | torchode/adjoints.py:125,199 | torchode mutates a graph input in place (`y_eval[...] = ...`), then records a CUDA event; dynamo rejects it under compile |
| `test_backend_compile_workflows.py::test_workflow_compiled_matches_eager[torch-spray_track]` | failed | failed | torchode + compile on CUDA (third-party) | torchode/adjoints.py:125,199 | torchode mutates a graph input in place (`y_eval[...] = ...`), then records a CUDA event; dynamo rejects it under compile |
| `test_backend_dtype_policy.py::test_float32_default_internals_run_in_float64` | failed | failed | test bug (b') |  | the TEST calls `.numpy()` on a CUDA result (no `.cpu()` / `as_numpy`); galpy's value is fine |
| `test_backend_dtype_policy.py::test_float32_default_orbit_and_df_match_float64_default` | failed | failed | test bug (b') |  | the TEST calls `.numpy()` on a CUDA result (no `.cpu()` / `as_numpy`); galpy's value is fine |
| `test_backend_evolveddiskdf.py::test_moment_grad_through_evolution_vs_fd[meanvT-jax]` | passed | failed | jax nondeterminism (c) | - | autodiff/quadrature value varies run to run on CUDA (see below) |
| `test_backend_evolveddiskdf.py::test_moment_grad_through_evolution_vs_fd[vertexdev-jax]` | passed | failed | jax nondeterminism (c) | - | autodiff/quadrature value varies run to run on CUDA (see below) |
| `test_backend_input.py::test_traced_call_traces_exactly_the_matching_namespace[torch]` | failed | failed | galpy CUDA bug (b): numpy constant under compile | (no galpy frame in message) | a numpy ndarray operand meets a CUDA FakeTensor in `mul` |
| `test_backend_interpolate.py::test_map_coordinates_quintic_and_narrow_grids_vs_scipy[torch]` | failed | failed | test bug (b') |  | the TEST calls `.numpy()` on a CUDA result (no `.cpu()` / `as_numpy`); galpy's value is fine |
| `test_backend_kingdf.py::test_kingdf_W0_under_torch_compile_torchode` | failed | failed | galpy CUDA bug (b): numpy constant under compile | galpy/df/kingdf.py:470 in _dens_W | a numpy ndarray operand meets a CUDA FakeTensor in `mul`: `out = xp.exp(Ws) * _bspecial.erf(sqW) - _TWOOVERSQRTPI * sqW * (` |
| `test_backend_multi.py::test_parallel_map_survives_torch_work_in_the_child` | failed | failed | galpy CUDA bug (b) | galpy/util/multi.py:239 | `parallel_map` forks (`get_context('fork')`) after the parent initialized CUDA; only jax gets the serial fallback |
| `test_backend_namespaces.py::test_get_namespace_all_torch_fast_path` | failed | failed | galpy CUDA bug (b): numpy constant under compile | galpy/potential/Potential.py:410 in _call_nodecorator | a numpy ndarray operand meets a CUDA FakeTensor in `mul`: `return self._amp * rawOut if not rawOut is None else rawOut` |
| `test_backend_namespaces.py::test_set_at_casts_values_to_the_destination_dtype` | failed | failed | test bug (b') |  | the TEST calls `.numpy()` on a CUDA result (no `.cpu()` / `as_numpy`); galpy's value is fine |
| `test_backend_potential_jit.py::test_torch_compile_traces_public_entry_point[DehnenBarPotential-__call__]` | failed | failed | galpy CUDA bug (b): numpy constant under compile | galpy/potential/DehnenBarPotential.py:159 in _smooth | a numpy ndarray operand meets a CUDA FakeTensor in `sub`: `xi = 2.0 * deltat / (self._tsteady - self._tform) - 1.0` |
| `test_backend_potential_jit.py::test_torch_compile_traces_public_entry_point[DehnenBarPotential-dens]` | failed | failed | galpy CUDA bug (b): numpy constant under compile | galpy/potential/DehnenBarPotential.py:159 in _smooth | a numpy ndarray operand meets a CUDA FakeTensor in `sub`: `xi = 2.0 * deltat / (self._tsteady - self._tform) - 1.0` |
| `test_backend_potential_jit.py::test_torch_compile_traces_public_entry_point[DehnenBarPotential-Rforce]` | failed | failed | galpy CUDA bug (b): numpy constant under compile | galpy/potential/DehnenBarPotential.py:159 in _smooth | a numpy ndarray operand meets a CUDA FakeTensor in `sub`: `xi = 2.0 * deltat / (self._tsteady - self._tform) - 1.0` |
| `test_backend_potential_jit.py::test_torch_compile_traces_public_entry_point[DehnenBarPotential-zforce]` | failed | failed | galpy CUDA bug (b): numpy constant under compile | galpy/potential/DehnenBarPotential.py:159 in _smooth | a numpy ndarray operand meets a CUDA FakeTensor in `sub`: `xi = 2.0 * deltat / (self._tsteady - self._tform) - 1.0` |
| `test_backend_potential_jit.py::test_torch_compile_traces_public_entry_point[EinastoPotential-__call__]` | failed | failed | galpy CUDA bug (b): numpy constant under compile | galpy/potential/EinastoPotential.py:150 in torch_dynamo_resume_in__revaluate_body_at_144 | a numpy ndarray operand meets a CUDA FakeTensor in `mul`: `out = -(4 * numpy.pi * (self.h**2) * self.n * gamma_3n) * (` |
| `test_backend_potential_jit.py::test_torch_compile_traces_public_entry_point[ExpTruncNFWPotential-__call__]` | failed | failed | galpy CUDA bug (b): numpy constant under compile | galpy/potential/ExpTruncNFWPotential.py:313 in _F_closed | a numpy ndarray operand meets a CUDA FakeTensor in `mul`: `self._exp_alpha * (1.0 + self._alpha) * (self._E1_alpha - exp1(beta))` |
| `test_backend_potential_jit.py::test_torch_compile_traces_public_entry_point[ExpTruncNFWPotential-Rforce]` | failed | failed | galpy CUDA bug (b): numpy constant under compile | galpy/potential/ExpTruncNFWPotential.py:313 in _F_closed | a numpy ndarray operand meets a CUDA FakeTensor in `mul`: `self._exp_alpha * (1.0 + self._alpha) * (self._E1_alpha - exp1(beta))` |
| `test_backend_potential_jit.py::test_torch_compile_traces_public_entry_point[ExpTruncNFWPotential-zforce]` | failed | failed | galpy CUDA bug (b): numpy constant under compile | galpy/potential/ExpTruncNFWPotential.py:313 in _F_closed | a numpy ndarray operand meets a CUDA FakeTensor in `mul`: `self._exp_alpha * (1.0 + self._alpha) * (self._E1_alpha - exp1(beta))` |
| `test_backend_potential_jit.py::test_torch_compile_traces_public_entry_point[FerrersPotential-__call__]` | failed | failed | galpy CUDA bug (b): numpy constant under compile | galpy/potential/FerrersPotential.py:150 in _evaluate_xyz | a numpy ndarray operand meets a CUDA FakeTensor in `mul`: `-math.pi` |
| `test_backend_potential_jit.py::test_torch_compile_traces_public_entry_point[FerrersPotential-dens]` | failed | failed | galpy CUDA bug (b): numpy constant under compile | galpy/potential/FerrersPotential.py:404 in _dens | a numpy ndarray operand meets a CUDA FakeTensor in `mul`: `return xp.where(m2 < 1, self._rhoc_M * base_safe**self.n, 0.0)` |
| `test_backend_potential_jit.py::test_torch_compile_traces_public_entry_point[FerrersPotential-Rforce]` | failed | failed | galpy CUDA bug (b): numpy constant under compile | galpy/potential/FerrersPotential.py:220 in _xforce_xyz | a numpy ndarray operand meets a CUDA FakeTensor in `mul`: `-2.0` |
| `test_backend_potential_jit.py::test_torch_compile_traces_public_entry_point[FerrersPotential-zforce]` | failed | failed | galpy CUDA bug (b): numpy constant under compile | galpy/potential/FerrersPotential.py:220 in _xforce_xyz | a numpy ndarray operand meets a CUDA FakeTensor in `mul`: `-2.0` |
| `test_backend_potential_jit.py::test_torch_compile_traces_public_entry_point[PowerSphericalPotentialwCutoff-__call__]` | failed | failed | galpy CUDA bug (b): numpy constant under compile | galpy/potential/PowerSphericalPotentialwCutoff.py:83 in torch_dynamo_resume_in__evaluate_at_86 | a numpy ndarray operand meets a CUDA FakeTensor in `mul`: `atinf=2.0` |
| `test_backend_potential_jit.py::test_torch_compile_traces_public_entry_point[PowerSphericalPotentialwCutoff-Rforce]` | failed | failed | galpy CUDA bug (b): numpy constant under compile | galpy/potential/PowerSphericalPotentialwCutoff.py:117 in torch_dynamo_resume_in__rforce_at_120 | a numpy ndarray operand meets a CUDA FakeTensor in `mul`: `-2.0` |
| `test_backend_potential_jit.py::test_torch_compile_traces_public_entry_point[PowerSphericalPotentialwCutoff-zforce]` | failed | failed | galpy CUDA bug (b): numpy constant under compile | galpy/potential/PowerSphericalPotentialwCutoff.py:117 in torch_dynamo_resume_in__rforce_at_120 | a numpy ndarray operand meets a CUDA FakeTensor in `mul`: `-2.0` |
| `test_backend_potential_jit.py::test_torch_compile_traces_public_entry_point[TwoPowerSphericalPotential-__call__]` | failed | failed | galpy CUDA bug (b): numpy constant under compile | galpy/potential/TwoPowerSphericalPotential.py:335 in torch_dynamo_resume_in__evaluate_at_335 | a numpy ndarray operand meets a CUDA FakeTensor in `neg`: `-special.beta(2.0 - self.alpha, self.beta - 2.0) / self.a` |
| `test_backend_potential_jit.py::test_torch_compile_traces_public_entry_point[TwoPowerTriaxialPotential-__call__]` | failed | failed | galpy CUDA bug (b): numpy constant under compile | galpy/potential/TwoPowerTriaxialPotential.py:177 in torch_dynamo_resume_in__psi_at_180 | a numpy ndarray operand meets a CUDA FakeTensor in `sub`: `self.psi_inf` |
| `test_backend_quadrature.py::test_finite_part_quad_batches_over_its_limits[jax]` | passed | failed | jax nondeterminism (c) | - | autodiff/quadrature value varies run to run on CUDA (see below) |
| `test_backend_streamdf.py::test_approxaAInv_float32_query_points` | failed | failed | test bug (b') |  | the TEST calls `.numpy()` on a CUDA result (no `.cpu()` / `as_numpy`); galpy's value is fine |
| `test_backend_streamdf.py::test_misalignment_traces[False]` | passed | passed | fixed upstream | - | tolerance loosened 1e-15 -> 1e-14 on the feat/backends tip (#1641) |
| `test_backend_streamdf.py::test_misalignment_traces[True]` | passed | passed | fixed upstream | - | tolerance loosened 1e-15 -> 1e-14 on the feat/backends tip (#1641) |
| `test_backend_torchode.py::test_float32_ic_integrates_in_float64[torchdiffeq]` | failed | failed | test bug (b') |  | the TEST calls `.numpy()` on a CUDA result (no `.cpu()` / `as_numpy`); galpy's value is fine |
| `test_backend_torchode.py::test_float32_ic_integrates_in_float64[torchode]` | failed | failed | test bug (b') |  | the TEST calls `.numpy()` on a CUDA result (no `.cpu()` / `as_numpy`); galpy's value is fine |
| `test_backend_torchode.py::test_torch_compile_torchode_orbit_matches_eager` | failed | failed | torchode + compile on CUDA (third-party) | torchode/adjoints.py:125,199 | torchode mutates a graph input in place (`y_eval[...] = ...`), then records a CUDA event; dynamo rejects it under compile |
| `test_backend_tracectx.py::test_forced_backend_survives_a_fullgraph_compile` | failed | failed | galpy CUDA bug (b): numpy constant under compile | galpy/potential/Force.py:289 in _Rforce_nodecorator | a numpy ndarray operand meets a CUDA FakeTensor in `mul`: `return self._amp * self._Rforce(R, z, **kwargs)` |
