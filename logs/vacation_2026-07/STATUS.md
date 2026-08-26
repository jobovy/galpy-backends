# Vacation status snapshot — 2026-08-25 (midday)

**This file is OVERWRITTEN, not appended.** "Where are things now"; `log.md` is
the trace.

---

## The gh#1204 review is fully addressed — five PRs, ledger delta 0

    batch 1 #1        -> gh#1381  206ccd78   gammainc endpoint gradient
    batch 1 #2, #10   -> gh#1382  06267d1d   quadrature batching + AnyAxisym torch
    batch 1 #3/#4/#6  -> gh#1383  4e2cc322   coords inverse galcen + f64/f32 promotion
    batch 1 #5        -> CLOSED as working-as-intended (comment 5411848615)
    batch 2 #1        -> gh#1383
    batch 2 #2        -> gh#1388  107a8060   map_coordinates dispatches on the GRID
    batch 2 #3, #4    -> gh#1391  cfe23abe   traceable constructors + hyp2f1/hyp1f1 params

Status comment posted to gh#1204: **comment 5422012558**.

**No PR is open.** develop is at `cfe23abe`.

## FOR JO — three decisions, none of which I made unilaterally

1. **Finding 11, sharpened.** `surfdens` under torch is not "no gradient": with
   `forcepoisson=True` it returns a Tensor with `requires_grad=True` and a
   `grad_fn` whose gradient is **23% wrong in R, 72% wrong in z** (jax AD matches
   the FD reference to 1.6e-10, so torch is the wrong one); with the default
   `forcepoisson=False` it returns a bare float. Proposal: for a grad-tracking
   eager tensor use the backend GL rule when `_force_accepts_arrays(pot)` and
   RAISE otherwise. Behaviour change -> your call.
2. **hyp2f1 direct-Euler value accuracy**, pre-existing and invisible to the
   test grid: ~1e-6..1e-7 when `c - B` is not an integer, ~1e-14 when it is.
   Gauss-Jacobi is the real fix.
3. **actionAngleAdiabaticGrid off-grid fallback** (the parked opt-in decision) —
   what its two xfail entries encode.

## Queued behind gh#1391

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
* **Ledger** (measured on `107a8060`, non-comment lines): **8 xfail + 57
  slow-skip = 65 active**, not the 137 I had been repeating. All **four eager
  xfail entries are BY DESIGN**, confirmed from the code's own docstrings:
  - `test_qdf.py::test_call_diffinoutputs` (jax+torch) — backend classifies an
    orbit numpy calls unbound as bound;
  - `test_actionAngleAdiabaticGrid_outsidegrid_c` (jax+torch) — the class
    docstring says the backend path "evaluates the interpolation grid only (no
    off-grid fallback)", and the test probes R=3.0 against an Rmax=2.0 grid.
    Lifting it is the parked "opt-in off-grid fallback" design decision, which
    costs the vectorisation the backend path exists for — **Jo's call**.

  So the eager ledger cannot be burned down without a design change; the other
  four entries are jit-only. Jo: "the goal is eventually a literal zero ledger,
  but not for gh#1204."

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
