# Vacation status snapshot — 2026-08-24 (Codex review of #1204: fix PRs 1 & 2)

**This file is OVERWRITTEN, not appended.** It is the "where are things now"
answer. `log.md` is the trace; read this first and go there only for the why.

---

## Nothing is blocked on Jo except (a) the main->develop rebase (parked) and
## (b) review finding 5, which needs a direction call (see below).

## Branch state

    main                   5d7b7e76
    feat/backends          982b1781
    feat/backends-develop  068de984   (ledger xfail 8 / slow_skip 60 / skip 72)

## Open PRs (2 = my cap)

* **gh#1380** `ledger/group1-measured` @ c2744b79 — Group-1 ledger. All 10
  previously-unmeasured jax deferrals measured; 3 un-deferred (25.2s / 83.4s /
  139.6s), 7 kept with measured notes. jax slow_skip 33 -> 30, UNMEASURED 10 -> 0.
  CI: 61 green / 74 queued / 17 running under heavy repo-wide CONTENTION —
  wait, do NOT re-run ([[all-clear-queued-rerun-and-ledger-rebase]]).
  `codecov/project` red = PHANTOM, established not assumed: the PR changes
  exactly ONE file, `tests/backend_slow_skip.txt`, with no Python in the diff,
  so a real project drop is impossible. Self-heals as uploads land.
* **gh#1381** `fix/gammainc-endpoint-grad` @ e6f63275 — review finding 1.
  d/dx gammainc was NaN at the x=0 endpoint. CI just dispatched.

## Pushed, PR deliberately HELD (would be a 3rd open PR)

* **`fix/quadrature-batching-anyaxisym-units`** @ e4ac1e07 — review finding 2
  (finite_part_quad honours its documented array signature; 4 sites, not 1)
  PLUS a torch construction bug the byte-identity A/B turned up on its own
  (`.to()` duck-typing collides with `torch.Tensor.to`). Based directly on
  develop, NOT stacked on gh#1381 — the two touch disjoint files.
  Local gate green: 132 passed / 1 skipped across both touched test files.
  Branch is pushed; opening the PR waits on gh#1380 merging, both to respect
  the two-PR cap and because a 3rd PR would add ~150 jobs to a queue that is
  already the thing delaying gh#1380.

## Codex review of #1204 — 5 findings, all reproduced

| # | Where | State |
|---|-------|-------|
| 1 | `gammainc.py:196` NaN d/dx at x=0 | **gh#1381 open** |
| 2 | `quadrature.py:622` scalar-only `finite_part_quad` | **committed, pushing next** |
| 3 | `Potential.py:128/:817/:914` surfdens gate misses traced `R` | next PR |
| 4 | `coords.py:100/:1217/:1591` `_galcen_rot` hard-codes numpy | next PR (with 3) |
| 5 | `_jit.py:39` + `_input.py:179` `set_jit("jax")` alone doesn't trace | **NEEDS JO**: fix the docs or fix the behaviour? |

Finding 2 is worth stating precisely: it is **not reachable today**. The only
caller is AnyAxisym, whose forces are `@check_potential_inputs_not_arrays`, so
R and z are guaranteed scalar. It is a general `galpy.backend` helper failing to
honour its own docstring, not a user-visible defect.

## Remaining burndown (unchanged — the review is the priority)

The 30 residual jax deferrals are ONE root cause: jax eager per-call overhead
(~5 ms flat) — Python-loop orbit integration and per-pixel grid loops. That is
the vectorization project (tasks #74/#79/#130), weeks not days.

## Deferred / next up

* **main -> develop rebase** — PARKED, Jo applies. MEASURED (merge-base
  87256016): main +13, feat/backends +225, develop +107; 16 overlapping files;
  `git merge-tree` = **7 conflicting**, a LOWER BOUND. I prepare on a scratch
  branch, Jo applies. Unblocks the 2 `AdiabaticGrid_outsidegrid_c` xfails +
  task #103.
* **AdiabaticContractionWrapperPotential** — NOT small: two
  `scipy.optimize.fixed_point` calls, Blumenthal1986 iterates on an ARRAY with
  `interp1d` inside. Needs a fixed-point helper + backend interpolation.
* Off-grid AA fallback — gated on the rebase.

## Standing corrections to how I work

* **Never decrease coverage, not even one line** (Jo).
* **Read the docstring before guessing the contract.** Finding 2 cost two wrong
  turns — a one-line "fix" that moved the error, and a test asserting 1.08 while
  getting 54734.5 — purely because I passed `peak_width=b` when the finite part
  needs `peak_width=0`. The docstring says so explicitly.
* **A reference implementation is only an oracle where it is correct.** jax's
  native gammainc returns NaN for d/dx at a=1,x=0 where the answer is plainly 1.
  "Match jax" would have pinned the wrong value.
* **Duck-typing on a method name is unsafe once tensor libraries are in play** —
  `.to`, `.T`, `.item`, `.shape` all collide with astropy/numpy meanings.
* **A byte-identity A/B is a reachability probe too** — it runs paths the suite
  does not, which is how the torch construction bug surfaced.
* **A value-only assertion cannot see a numpy fallback.** Assert
  `is_backend_array` FIRST, then compare.
* **Check the same-SHA twin FIRST** on any shard red.
* **A failed shard uploads no coverage**, so a flake presents AS a codecov drop.

## Cron

`5451d57e`, hourly at :13, session-only, **created 2026-08-24 -> expires
~2026-08-31**. Verified alive this wake via CronList.
