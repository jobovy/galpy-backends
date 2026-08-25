# Vacation status snapshot — 2026-08-25 (midday)

**This file is OVERWRITTEN, not appended.** "Where are things now"; `log.md` is
the trace.

---

## Merged into `feat/backends-develop` this cycle

* **gh#1381** `fix/gammainc-endpoint-grad` → `206ccd78`
* **gh#1382** `fix/quadrature-batching-anyaxisym-units` → `06267d1d`
  (merged only after the duplication audit: the `on_nodes` closure was replaced
  by the pre-existing module-level `node_axis()` helper.)

## The one open PR (mission cap = 2, using 1)

* **gh#1383** `fix/coords-inverse-transforms`, worktree `/var/tmp/rev1`,
  head **`1ff63b7e`** — review findings 3 + 4 + 6 (finding 6 complete, 7 of 7),
  plus `promote_common_dtype` hoisted into `galpy.backend`, the
  `scalarDecorator` python-scalar/float32 fix, and a torch `.T`-on-1-D guard.
  **Ledger delta: 0.**

  Two red classes found on the previous head and BOTH now fixed and pushed:

  1. *torch stream shard.* A/B in the same worktree — base (merge-base
     `206ccd78`) 161 passed / 0 failed, my branch 2 failures, so both were mine.
     `test_calcaAJacLB` needed `coerce_coords` in
     `actionAngleIsochroneApprox._parse_args_backend` (mixed numpy/backend
     coordinate set); `test_streamTrack_sample_consistency` needed `as_numpy`
     at the assertion site. Re-run of the same three files under `--backend
     torch`: **161 passed, 0 failed (751 s)** — matches the base arm exactly.
  2. *codecov/project + codecov/patch.* A REAL drop (99.96 → 99.90, 28
     sessions), not the stale-base phantom. 30 missed lines, all in
     `coords.py`, all backend branches added by this PR. Four new
     `_CHAIN_CASES` in `tests/test_backend_coords.py` cover them; a coverage
     run confirms **all 30 lines now execute** (82 passed, was 67).

  **`1ff63b7e` reached 154/154 SUCCESS, mergeStateStatus CLEAN** — codecov
  cleared itself at 18:42Z once the second `tests/test_backend*.py` upload
  landed, confirming the incomplete-upload reading.

  Then the pre-merge audit found a **duplicate**: the forward
  `XYZ_to_galcenrect` / `vxvyvz_to_galcenrect` extra-rot blocks were identical
  three-liners, and the same rotation `_apply_extra_rot` already did for the
  inverse transforms. Hoisted to `_rotate_components`; pushed as **`0c69715a`**,
  which is re-running the full ~154-check cycle. Gate for the change: 180 passed
  on numpy, 98 under torch, 98 under jax.

## Queued behind gh#1383 (approved sequencing)

1. `fix/mapcoords-grid-dispatch` (worktree `/var/tmp/mapfix`, head
   **`f05987fa`**, 2 commits) — backend `map_coordinates` silently returned an
   ndarray under torch/jax eager and raised under torch+grad and jax-jit.
   **READY**: pre-open audit removed a duplicated `device_of`, and the branch is
   gated on a throwaway probe (gh#1383 head + this branch cherry-picked):
   `tests/test_backend_interpolate.py` **165 passed on numpy, 165 under
   `--backend torch`**. Rebase onto develop and open the moment gh#1383 merges.
2. Review batch-2 items 3 AND 4 — **BUILT AND COMMITTED LOCALLY, not
   pushed**: branch `backend/constructor-jit-safety`, worktree `/var/tmp/jitsafe`,
   head **`48a68247`** (4 commits) on develop `06267d1d`. PR body drafted at
   `$SCRATCH/pr_body_jitsafe.md`. `git merge-tree` says it merges cleanly onto
   the post-gh#1383 tree, so no rebase. Also carries the hyp2f1 + hyp1f1
   parameter-gradient fixes and the silent-detachment audit. Traced-green
   measured: 245 passed under `--backend jax --jit` AND `--backend torch --jit`. Adds
   `galpy.backend.concretely_true` and applies it to `ExpTruncNFWPotential`
   `min(a, rc)`, `TwoPowerTriaxialPotential`'s alpha/beta validation and its
   two alpha == 2 branches, and `constantbetaPowerLawdf`'s four asserts; plus
   hoists `sphericaldf._handle_rmin`'s explicit-rmin early return above the
   non-traceable Phi(0) probe. Gates: jit(grad)-vs-FD 1e-11..1e-7 on four
   parameters, torch autograd agreeing to ~1e-14, numpy byte-identity by
   sha256 with a 1-ulp negative control, A/B 7-failed-on-base vs 14-passed,
   numpy regressions 394 + 1288 passed. Forced-backend gate still running.
3. Review batch-2 item 4 — `hyp2f1.py:51/92/156` scalar params. **Confirmed
   still open by item 2's work**: `TwoPowerTriaxialPotential.__call__`
   (`_psi` -> hyp2f1) dies at `_in_regime`'s `a > 0 and (c - a) >= 1.0` under
   jit(grad), and at `float(math.ceil(6.0 / B))` under eager grad. Route
   selection cannot be a concreteness guard, so this needs its own design:
   either evaluate both routes and select (NaN-safety on the dead branch), or
   pin the Euler quadrature's grading `k` to its finest value while traced.
4. `_jit.py` docstring: "trace every boundary call" overstates it. Next review
   PR, not gh#1383.

## Closed / retracted

* **finding 5 (`set_jit`) — CLOSED as working-as-intended** on Jo's
  instruction ("leave set_jit as is and drop the branch"). Branch and worktree
  deleted; the reasoning is posted as PR gh#1204 comment 5411848615: `set_jit`
  runs before the data exists, so validating a backend there checks the
  *ambient* backend, which is the wrong thing when the pattern is data-driven.
* **RETRACTED: "pre-existing develop red in `test_galpypaper::test_orbmethods`".**
  There was no red. `/var/tmp/rev1` had **no built C extension at all**, so the
  test ran a different code path. Built it; passes locally under numpy and
  torch, and CI passes it too. [[worktree-missing-c-ext-fakes-passes]] cuts
  both ways — a missing `.so` also fakes FAILURES.

## Still open, not started

* **findings 8 & 9** — traced constructor params + `hyp2f1` scalar params
  (= queue items 2 and 3 above); all four finding-8 sites reproduce
  `TracerBoolConversionError`.
* **finding 11 (mine, reported not fixed)** — no torch gradient through
  `surfdens`.
* **To-do, deferred by Jo**: systematic f32 / default-dtype sweep.
  `tests/conftest.py:485` sets `torch.set_default_dtype(torch.float64)`, which
  masks the entire class from every test. Jo: "for now we can leave it, but put
  it on a to-do list."
* **Ledger**: 137 entries, 97 exercised, **0 stale** — emptying it needs real
  fixes, not pruning. Jo: "the goal is eventually a literal zero ledger, but
  not for gh#1204."

## Housekeeping

* Cron `5451d57e` (hourly at :13) verified alive 2026-08-25; created
  2026-08-24, expires ~2026-08-31.
* Root filesystem hit 100% on 2026-08-25 and wedged every command with ENOSPC.
  Freed ~10G from never-GC'd `torchinductor_*` caches; expect recurrence.
* `coverage run -m pytest` is BROKEN in this env (numpy re-import dies with
  "cannot load module more than once per process"). Workaround that works:
  start `coverage.Coverage()` in-process AFTER pre-importing numpy/scipy/
  pytest/jax/torch, with `sys.path.insert(0, os.getcwd())`, then call
  `pytest.main([...])`. Driver kept at `$SCRATCH/covdrv.py`.
