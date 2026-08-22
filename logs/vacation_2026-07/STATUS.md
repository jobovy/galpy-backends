# Vacation status snapshot — 2026-08-22 (gh#1363-1367 MERGED; NO open PR; next = task #221 Einasto)

**This file is OVERWRITTEN, not appended.** It is the "where are things now"
answer. `log.md` is the trace; read this first and go there only for the why.

---

## Nothing is blocked on Jo except the main->develop rebase (parked, Jo applies)

## Branch state

    main                   5d7b7e76
    feat/backends          982b1781
    feat/backends-develop  08db5d90   (gh#1367 merged; ledger xfail 8 / slow_skip 61 / skip 68)

## Open PRs — NONE. Next PR = task #221 (Einasto _calculate_dn), recipe ready.

gh#1363 `19f31a54` ledger split (43 streamTrack tests back in jax CI)
gh#1364 `a4193e84` as_backend_constant default-device guard + effective_device
gh#1365 `46ea6ade` un-defer 8 noninertial entries, slow_skip 69 -> 61
gh#1366 `47875227` ledger annotations: censored bound -> measurement
gh#1367 `08db5d90` from_nfw mass -> rc differentiable (8e-12 vs FD)

**CI never runs `--jit`** (verified). jax-jit/torch-jit ledger rows are LOCAL-only:
no CI risk, no CI validation. Report traced progress as locally-measured.

**Routed helpers follow the AMBIENT forced namespace, not their argument.**
`backend.special.*` resolves the forced namespace even for a Python float, while
`backend.optimize.brentq` follows the DATA. On a scipy path derive EVERY piece
from one `numpy_path = not is_backend_array(...)` predicate. Cost gh#1367 two
round-trips. Only torch surfaces it; jax accepts float args and stays green.

## Measured and CLOSED this wake

* **streamTrack burndown: NO.** The #1364 guard gives 0.96x-1.11x there (both
  eager and jit, whole file, post-guard develop). All 4 entries stay. The guard
  pays only where `as_backend_constant` sits in a hot per-evaluation loop
  (noninertial `_force`), not in track SETUP. Do not assume it un-blocks other
  clusters -- measure each, FIXED-arm first.

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
