# Vacation status snapshot — 2026-08-09

**This file is OVERWRITTEN, not appended.** It is the "where are things now"
answer. `log.md` is the trace; read this first and go there only for the why.

---

## PR pipeline

| state | PR | effect |
|---|---|---|
| **merged since the last snapshot (08-04)** | ~19 PRs, `#1256` → `#1278` | develop is now `f963eb42` |
| **open** | **gh#1279** `feat/jit-slow-skip-own-list` @ `d3545330` | `--jit` slow-skip stops inheriting eager; 11 measured traced entries; 3 tolerance entries burned |
| **open, stacked on #1279** | **gh#1280** `feat/streamdf-inplace-writes` @ `5ff4464c` | streamdf out-of-place LB jacobian; **10 jax-jit ledger entries cleared** |
| **open, FOR JO** | **gh#1274** → `main` | **151 green.** Targets `main`, so it is yours to merge, not mine |
| **filed as issues** | gh#1276, gh#1277 | two pre-existing numpy accuracy defects, mpmath-validated |

**CI right now:** #1279 110 green / 11 running / 31 queued; #1280 11 green /
24 running / 72 queued. The only red on either is `codecov/project`, which is a
**quantified upload phantom** — the API reports head at 21 of base's 28 upload
sessions, and coverage has climbed 70.83% → 76.12% → **99.47%** as they land
(base is 99.95%). `lines` is identical on both sides. No action; if it is still
red at session parity, re-run all-clear.

**Merge path is rehearsed** (dry-run in a throwaway worktree, zero conflicts):
squash #1279 → retarget #1280 to develop → rebase → push.

## Ledger and deferrals

```
xfail ledger      develop  17   →  after #1279  24   →  after #1280  14
                  (jax 5, jax-jit 6,      (the +10 is streamdf      (jax 4, jax-jit 6,
                   torch 5, torch-jit 1)   immutability, surfaced    torch 3, torch-jit 1)
                                           then fixed by the pair)

slow-skip         develop  jax 75, torch 15, torch-jit 2
                  #1279 adds jax-jit 11 (measured, not inherited)
```

The pair takes the ledger **17 → 14** net. The intermediate 24 is the point of
the pair, not a regression: #1279 stops a *performance* deferral from hiding a
real jax defect, which makes 10 failures visible; #1280 fixes them at one site.

## Awaiting Jo

**One decision:**

* **Should CI gain a `--jit` dimension?** No workflow runs `--jit` today
  (checked across all 10 files in `.github/workflows/`), so the traced ledger,
  the traced slow-skip list and `assert_jit_matches_eager`'s premise are
  validated on the dev box and **nowhere else**. Options: a nightly shard on the
  two measured files / fold into `backend-tests.yml` / accept and document.
  Needs `workflow` scope, which my token lacks.

**One action:** merge **gh#1274** (green, targets `main`).

**One thing worth knowing:** `build.yml` triggers on both `push:` (no branch
filter) and `pull_request:`, so **every in-repo PR runs ~52 build jobs where 26
would do**. Observed directly: develop's `f963eb42` carried two complete,
identical job sets created 3 s apart. A branch filter on the push trigger halves
the CI bill. Also a workflow edit.

## In flight locally

A serial traced sweep of the **10 files** whose eager-jax slow-skip entries stop
being inherited under #1279 but were never measured traced (54 entries). Serial
because concurrent jax oversubscribes the XLA CPU pool and corrupts the timings.

**4 of 10 done:**

| file | tests | eager entries | → jax-jit | un-deferred |
|---|---|---|---|---|
| `test_dynamfric` | 7 | 1 | 1 | — |
| `test_FDMdynamfric` | 9 | 3 | 2 | `test_dynamfric_c` (226.6 s) |
| `test_galpypaper` | 15 | 1 | 0 | `test_diskdf` |
| `test_streamspraydf` | 38 | 1 | 0 | `test_center` (81.5 s) |
| | | **6** | **3** | **3** |

**Half the eager deferrals do not survive a traced run**, and in the FDM file
inheritance was wrong in *both* directions at once. **Zero defect candidates**
across 69 traced tests so far. Remaining: `test_interp_potential`, `test_qdf`,
`test_orbits`, `test_actionAngle`, `test_potential`, `test_orbit` — the last
three are large and will take hours.

## Queued next, in order

1. Merge #1279, retarget+merge #1280 (path rehearsed).
2. File the sweep's measured `jax-jit` entries — into #1279 if it is still open,
   else a follow-up.
3. **#211(B)** `interpRZPotential` grid-build vectorisation — Jo chose (B). All
   **7** sites are structurally identical, so it is one module-private helper
   used 7×. Two gates before it lands: per-potential numpy byte-identity by
   sha256 (vectorised evaluation can take a *different* code path inside some
   potentials), and an explicit decision for the 2 array-unsafe potentials.
   Blocked on #1274 merging (shared call sites) and a free PR slot.

## Standing caveats

* **numpy stays byte-identical.** Every deviation is flagged, never quietly
  shipped. #1280's claim is backed by sha256 of all five arrays its function
  produces, in a patched vs unpatched tree.
* **Traced green is asserted by me, not enforced by CI** — see the decision above.
* `actionAngleVerticalInverse` and `actionAngleTorus` are permanently out of
  scope (external C++ wrapper / prior decision).
* A repo-wide AST inventory found **98** in-place augmented array writes
  (83 in backend-aware files). It is a **watchlist, not a defect list** — a site
  only matters if a backend array reaches it, and the honest filter is a traced
  test result. streamdf had 27 such sites; exactly 3 were reachable.
