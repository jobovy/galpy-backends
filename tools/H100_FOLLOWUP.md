# Follow-up instructions for the H100 (Trillium) Claude session

Context: your first run is folded into galpy-jax `main` (branch `gpu-h100` stays as
the record). Same ground rules as `tools/H100_SESSION.md` (one GPU job at a time,
`XLA_PYTHON_CLIENT_PREALLOCATE=false`, work on branch `gpu-h100`, never push to
galpy-jax `main` or to galpy, no figure-code edits beyond what is listed here, caches
on scratch, one measurement per job). Rebase `gpu-h100` onto `origin/main` of
galpy-jax first (`git fetch && git rebase origin/main`; keep your data files where
they conflict, our copies live in `data/h100/`).

## 1. galpy version

Update the galpy checkout to the current `feat/backends` tip and cherry-pick the
open PRs **#1641** (VerticalInverse Newton polish) and **#1642** (Staeckel c=False
chi-anomaly mesh: it makes the pure-Python Staeckel path ~4x faster, so every g17
series measured before it is stale) if they are not merged yet:
```
cd $SCRATCH/galpy-h100/galpy-fb && git fetch origin && git checkout feat/backends && git reset --hard origin/feat/backends
for n in 1641 1642; do gh pr view $n --json state -q .state | grep -q MERGED || (git fetch origin pull/$n/head && git cherry-pick FETCH_HEAD); done
python setup.py build_ext --inplace -j 8
```
Record the SHA and which PRs were cherry-picked in the report.

## 2. g17: GPU series, torch alone (stage `g17`, then the torch-only script)

Re-run the `g17` stage (both GPU frameworks; the CPU series it also produces are
not used). Then, in a SEPARATE job, fill torch's OOM points by running torch alone:
```
PYTHONPATH=$SCRATCH/galpy-h100/galpy-fb python paper/figures/g17_staeckel_actionangle_speed/measure/torch_gpu_only.py \
    paper/figures/g17_staeckel_actionangle_speed/data/g17_staeckel_actionangle_speed.json
```
(it overwrites only the two torch GPU series in that JSON). Report the memory peak if
you can (`nvidia-smi --query-gpu=memory.used --format=csv -l 5` in the job script).

## 3. gV2: GPU series (stage `newfigs`, gV2 only)

#1641 changes the VerticalInverse angle solve (Newton instead of bisection), so the
gV2 GPU series change. Re-run only gV2's GPU measurement (the `newfigs` stage runs
gV2, gD3, gS5, gS4; if it cannot be restricted, running all four is fine, ~4 h).

## 4. The 47 torch and 2 jax CUDA test failures (stage `tests`, split)

Goal: classify each failure as (a) a torch-2.14 regression, (b) a galpy-on-CUDA
bug, or (c) a tolerance miss. Method: create a SECOND conda env identical to
`galpy-h100` but with `torch==2.12.0` from the cu126 index
(`pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu126`;
sm_90 is supported by cu126) and the same torchode/torchdiffeq, and run ONLY the
failing tests (list in `logs/h100/report.md`) in both envs:
```
python -m pytest -q --backend torch --device cuda -p no:cacheprovider <nodeids...>
```
A test that passes with 2.12 and fails with 2.14 is (a); one that fails in both is
(b) or (c). For each (b), save the full traceback and a one-line diagnosis (which
galpy line converts a CUDA tensor to numpy, or which object reaches dynamo as an
ndarray). For the 2 jax failures (`test_moment_grad_through_evolution_vs_fd`, 5.6e-4 vs
rtol 2e-4), run them 3 times to see whether the value is stable (then it is a tolerance
miss) or varies (then a nondeterminism). Write the classification table into
`logs/h100/tests_triage.md`. Do not fix anything in galpy.

## 5. Optional: larger N on the 80 GB card (new stage `hiN2`)

On the TITAN Xp the largest N were set by its 12 GB. If time allows, extend the GPU
series to show where the H100 saturates: g12 jax Dopri8 and torch to N = 1e6 and 3e6
(the `_shared/orbit_speed` and `g12` hiN measure scripts take `--Ns`-style arguments;
check `python <script> --help`, do not edit the scripts), gV2 evaluation to N = 1e7,
gD3 sampling to N = 1e8 (memory permitting; a 1e8 x float64 sample is 0.8 GB per
coordinate). Keep the existing points; add the new ones to the same JSONs.

## Deliverables

As before: changed `data/*.json`, re-rendered PDFs, logs in `logs/h100/`, and an
updated `logs/h100/report.md` with: galpy SHA + cherry-picks, the g17/gV2 headline
numbers vs the previous run, the triage table, and the hiN2 numbers if run. Commit
and push `gpu-h100`; do not merge.
