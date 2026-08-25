# Vacation status snapshot — 2026-08-25 (midday)

**This file is OVERWRITTEN, not appended.** "Where are things now"; `log.md` is
the trace.

---

## Merged into `feat/backends-develop` this cycle

* **gh#1381** `fix/gammainc-endpoint-grad` -> `206ccd78`
* **gh#1382** `fix/quadrature-batching-anyaxisym-units` -> `06267d1d`
* **gh#1383** `fix/coords-inverse-transforms` -> **`4e2cc322`** at 154/154 CLEAN.
  Review findings 3, 4 and 6 (finding 6 complete, 7 of 7), plus
  `promote_common_dtype` + `prefer_backend_namespace` hoisted into
  `galpy.backend`, the `scalarDecorator` float32 fix, the `_quad_needs_backend`
  all-coordinates fix, and the `actionAngleIsochroneApprox` mixed-set coercion.
  Its last red was a **19-second race**: `all-clear` gave up at 22:16:40Z while
  codecov flipped green at 22:16:58Z. Re-ran that one job. Ledger delta: 0.

## The open PR

* **gh#1388** `fix/mapcoords-grid-dispatch`, worktree `/var/tmp/mapfix`, head
  `e7dcb342` (2 commits) on develop `4e2cc322`. Review batch-2 item 2:
  `map_coordinates` dispatched on the query coordinates alone, so a backend
  prefiltered GRID queried at fixed numpy points fell into scipy and raised
  ("Can't call numpy() on Tensor that requires grad" / TracerArrayConversion).
  Gates measured BEFORE opening: value parity 3.3e-16, grad-vs-FD 3.0e-10 at
  every one of 12 grid nodes, `tests/test_backend_interpolate.py` 165 passed on
  numpy / 165 under jax / 165 under torch. Ledger delta: 0. CI starting.

## Queued behind gh#1388 (approved sequencing)

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
* **finding 11 (mine, reported not fixed) — SHARPENED, and worse than
  reported.** It is not "no torch gradient": with `forcepoisson=True` surfdens
  returns a Tensor with `requires_grad=True` and a `grad_fn` whose gradient is
  **23% wrong in R and 72% wrong in z** (MiyamotoNagai a=0.5 b=0.1 at R=1,
  z=0.3; FD anchored by jax AD matching it to 1.6e-10 on both routes). With the
  default `forcepoisson=False` it returns a bare Python float, severing the
  graph. Cause: `_quad_needs_backend` gates on `under_trace`, False for eager
  torch autograd, so scipy runs and the integrated piece detaches while the
  algebraic prefactor keeps a graph. **Needs Jo's decision** (behaviour change):
  for a grad-tracking eager tensor, use the backend GL rule when
  `_force_accepts_arrays(pot)` and RAISE otherwise. Own PR.
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
