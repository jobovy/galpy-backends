# Vacation status snapshot — 2026-08-22 (gh#1363 MERGED; gh#1364 sole open PR)

**This file is OVERWRITTEN, not appended.** It is the "where are things now"
answer. `log.md` is the trace; read this first and go there only for the why.

---

## Nothing is blocked on Jo except the main->develop rebase (parked, Jo applies)

## Branch state

    main                   5d7b7e76
    feat/backends          982b1781
    feat/backends-develop  19f31a54   (gh#1363 merged; ledger xfail 8 / slow_skip 69 / skip 68)

## Open PRs — ONE, mine

**gh#1363 MERGED** as `19f31a54` — fully green 154/154. The shard that had to
prove it (`backend jax ... test_streamTrack.py`) passed with the 43 un-skipped
tests running. Both reds cleared with no code change: the windows red was
`test_from_name_values`, a live SIMBAD query (same-SHA push twin PASSED the
identical shard, and a ledger-only diff cannot reach a numpy `build` job);
codecov/project self-healed as shards uploaded.

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
