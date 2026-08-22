# Vacation status snapshot — 2026-08-22 (gh#1363 + gh#1364 MERGED; gh#1365 sole open PR)

**This file is OVERWRITTEN, not appended.** It is the "where are things now"
answer. `log.md` is the trace; read this first and go there only for the why.

---

## Nothing is blocked on Jo except the main->develop rebase (parked, Jo applies)

## Branch state

    main                   5d7b7e76
    feat/backends          982b1781
    feat/backends-develop  a4193e84   (gh#1364 merged; ledger xfail 8 / slow_skip 69 / skip 68)

## Open PRs — ONE, mine

**gh#1363 MERGED** `19f31a54` (154/154). Windows red was `test_from_name_values`,
a live SIMBAD query -- same-SHA push twin passed the identical shard, and a
ledger-only diff cannot reach a numpy `build` job. codecov self-healed.

**gh#1364 MERGED** `a4193e84` (154/154). `as_backend_constant` no longer names the
default device (galpy #1300's guard, missing from the sibling helper); 144 call
sites. Self-audit before merge found the SAME predicate open-coded twice -- which
IS the defect class -- so it was hoisted to `effective_device(xp, device)` in
`_namespaces.py` and pushed as a second commit, at the cost of a full CI re-run.
Measured: `_force` 5069.7 -> 3064.0 us (1.65x).

**gh#1365 OPEN -> develop** `ledger/noninertial-undefer`, head `43caac60`.
Ledger-only. Un-defers 8 noninertial entries measured under the #1364 guard:
jax eager 8->5, jax --jit 9->4, **slow_skip 69 -> 61**. Both halves are
CONTROLLED A/Bs (BASE = develop without the guard, same box/command); BASE
reproduces this ledger's own historical numbers to 0.85-1.02, and the speedup is
heterogeneous 1.07x-1.64x so the two slowest tests stay deferred on their own
merits. **Mine to merge once green.**

## Ledger (at develop f15a522a)

    backend_xfail        8
    backend_slow_skip   68     (-> 69 entries / 43 tests un-skipped via gh#1363)
    backend_skip        68

Largest slow_skip blocks: test_orbit 20 (9 jax + 7 jax-jit + 3 torch-jit + 1
torch), **test_noninertial 17** (8 jax + 9 jax-jit), test_potential 6.

## Deferred / next up

* **Noninertial un-defer — MEASURED 2026-08-22, ledger edit PREPARED AND HELD.**
  Whole file, jax eager, WITH gh#1364's guard: 53 passed in 5145.98 s (ledger's
  own baseline for the same file: 6308 s). All 8 jax-eager entries pass; none
  stale. Under the ledger's 240 s CI margin, **3 become un-deferrable**
  (CI~213.7 / 199.1 / 176.7 s), 5 stay. Prospective delta **slow_skip 69 -> 66**.
  **Cannot be opened until gh#1364 merges** — without the guard those three run
  ~415 s local (CI~294) and would flake. The 9 jax-jit noninertial entries are
  untouched: under trace the guard runs once per trace, so it buys far less.
* **main -> develop rebase** — PARKED, Jo applies. MEASURED difficulty
  (merge-base 87256016): main +13, feat/backends +225, develop +107; 16
  overlapping files; **git merge-tree = 7 conflicting**. 7 is a LOWER BOUND.
  Protocol: I prepare on a scratch branch, Jo applies. Unblocks the 2 remaining
  fixable xfail entries (`AdiabaticGrid_outsidegrid_c`) + task #103.
* Off-grid AA fallback — gated on the rebase.

## Two hypotheses that died on 2026-08-22 (both recorded in log.md)

1. **REFUTED** — the md5 single-entry cache in `_force` is 0.8% of the jax call
   (66 us of 8207 us), not the bottleneck.
2. **I read the source from the stale `feat/backends` tree** (107 commits behind)
   and reported a defect that develop had already fixed. The measurement was
   sound (it ran in a develop worktree); only the reading was stale.

## Standing corrections to how I work

* **Never decrease coverage, not even one line** (Jo).
* **`--author @me` is not "mine"** — check the `Claude-Session` trailer.
* **Check the same-SHA twin FIRST** on any shard red — a push/pull_request split
  at one SHA settles flake-vs-real for free.
* **A failed shard uploads no coverage**, so a flake presents AS a codecov drop.
* **Read the source in the tree you MEASURED in**, not the main repo checkout.
* **An accepted `gh run cancel` is not a completed one.**

## Cron

`e935a60c`, hourly at :13, session-only, **created 2026-08-19 -> expires
~2026-08-26**. Verified alive this wake via CronList.
