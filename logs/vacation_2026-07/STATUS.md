# Vacation status snapshot — 2026-08-22 (gh#1363 + gh#1364 open, both mine)

**This file is OVERWRITTEN, not appended.** It is the "where are things now"
answer. `log.md` is the trace; read this first and go there only for the why.

---

## Nothing is blocked on Jo except the main->develop rebase (parked, Jo applies)

## Branch state

    main                   5d7b7e76
    feat/backends          982b1781
    feat/backends-develop  f15a522a   (gh#1360 merged; ledger 68/8/68)

## Open PRs — TWO, both mine

**gh#1363 -> develop** `backend/split-streamtrack-filedefer`, head `3950a8d2`.
Ledger-only (`tests/backend_slow_skip.txt`). Replaces the whole-file
`jax tests/test_streamTrack.py` defer with two per-test entries
(`plot_spread_non_cartesian` 408.0 s, `with_progpot` 331.7 s) — entry count
68 -> 69 but **43 tests un-skipped**; the count is the wrong metric and the PR
says so. Also reclassifies `jax tests/test_streamdf.py` as **DEADLOCK, not slow**
(two runs hung, xdist 7186 s and serial >4 h, both at load ~0.05).
CI 102 green, only `codecov/project` red (incomplete-upload phantom, self-heals).
The one real red — windows-latest 3.14 orbit shard — was
`test_from_name_values`, a **live SIMBAD network query**; the same-SHA push twin
PASSED the identical shard, and a ledger-only diff cannot reach a numpy `build`
job (conftest returns before reading the ledger when `--backend` is numpy).
Re-run, now queued. **Mine to merge once green.**

**gh#1364 -> develop** `backend/coerce-default-device-guard`, head `cecfd453`.
`as_backend_constant` never got the galpy #1300 default-device guard that
`coerce_coords` has 60 lines up in the same file, so every stored constant named
its device explicitly — which COMMITS placement (397.6 us vs 134.6 us on jax CPU).
Attributed by patching `jax._src.pjit.with_sharding_constraint`: **7.00 calls per
`NonInertialFrameForce._force`, 100% from `_coerce.py:164`**.
Measured A/B (arms verified to differ): `_force` **5069.7 -> 3064.0 us (1.65x)**;
`test_arbitraryaxisrotation_omegafunc_nullpotential` eager **414.5 -> 273.6 s
(1.51x)**. **144 call sites** (streamdf 66, streamgapdf 20, sphericaldf 11,
NonInertialFrameForce 10, RotateAndTilt 9, ...) — a repo-wide eager win, not a
noninertial one. numpy untouched (early return); GPU non-default anchors still
named. Gate green: numpy 708 / jax 694 / torch 694. **No ledger change.**

## Ledger (at develop f15a522a)

    backend_xfail        8
    backend_slow_skip   68     (-> 69 entries / 43 tests un-skipped via gh#1363)
    backend_skip        68

Largest slow_skip blocks: test_orbit 20 (9 jax + 7 jax-jit + 3 torch-jit + 1
torch), **test_noninertial 17** (8 jax + 9 jax-jit), test_potential 6.

## Deferred / next up

* **Noninertial un-defer** — gh#1364 makes the measured test 273.6 s local
  (~194 s CI at the 0.71 CI/local factor), under the 300 s cap. Un-deferring needs
  all 17 entries re-measured ON TOP of gh#1364; that is the next ledger PR, and
  it must wait for gh#1364 to land or the numbers are wrong.
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
