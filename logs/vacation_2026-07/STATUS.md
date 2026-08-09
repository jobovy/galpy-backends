# Vacation status snapshot — 2026-08-09 (05:30 EDT)

**This file is OVERWRITTEN, not appended.** It is the "where are things now"
answer. `log.md` is the trace; read this first and go there only for the why.

---

## PR pipeline

| state | PR | effect |
|---|---|---|
| **merged since the last snapshot (08-04)** | ~19 PRs, `#1256` → `#1278` | develop is now `f963eb42` |
| **MERGED** | **gh#1279** | `--jit` slow-skip stops inheriting eager; 11 measured traced entries; 3 tolerance burned. develop `f963eb42` → **`d2ad93c9`** |
| **open** | **gh#1280** `feat/streamdf-inplace-writes` @ `ec7e67e2` | streamdf out-of-place LB jacobian; **10 jax-jit ledger entries cleared**. Retargeted to develop + rebased clean after #1279 merged |
| **open, FOR JO** | **gh#1274** → `main` | **151 green.** Targets `main`, so it is yours to merge. It was still marked **draft** until 2026-08-09 — i.e. un-mergeable while I was asking you to merge it; now `ready`, blocked only on required review |
| **local, not yet opened** | `backend-jit/sweep-measured-entries` @ `1fc19ede` | the sweep's measured entries + 8 diagnosed traced defects; waits on `test_orbit` |
| **filed as issues** | gh#1276, gh#1277 | two pre-existing numpy accuracy defects, mpmath-validated |

**CI right now: no reds anywhere.** gh#1280 5 green / 12 running / 134 queued
on its new base (a rebase changes every SHA, so it restarted a full cycle).

Every codecov red on both PRs was an **upload phantom** and every one cleared on
its own — #1279 went 70.83% → 76.12% → 99.47% → **99.95%** as sessions went
9 → 22 (base 99.95%), and #1280's patch then project cleared in upload order.
Nothing was "fixed"; the discriminator was the API's `sessions` count.

One shard on #1279 ran **3× its 15m36s historical norm** and then failed at 53
min — a **runner eviction**, not a test failure, on four signatures: the pytest
step frozen at `in_progress`, the artifact-upload steps never reached, the job
log absent (`BlobNotFound`), and well under the 90-min cap. A targeted single-job
re-run passed inside the normal 15m36s, `all-clear` was re-run after it, and the
PR merged at 154/154.

The merge path was rehearsed in a throwaway worktree first (zero conflicts) and
then executed exactly: squash #1279 → retarget #1280 → rebase → push, with the
end state verified by value (`0` streamdf jax-jit entries, `4` torch useTM
skips, out-of-place line present, no conflict markers).

## Ledger and deferrals

```
xfail ledger   develop pre-#1279   17
               #1279  +10 -3   ->  24     (+10 = streamdf immutability, made
                                           VISIBLE by the inheritance split)
               follow-up  -1    ->  23     StaeckelGrid, measured stale (XPASS)
               follow-up  +8    ->  31     traced test_potential defects
               #1280     -10    ->  21     streamdf immutability FIXED

slow-skip      develop  jax 75 / torch 15 / torch-jit 2
               #1279    + jax-jit 11 (measured, not inherited)
               follow-up+ jax-jit  8 (measured, 8 of 10 files)
```

**The 23 → 31 step is measurement, not regression.** `test_potential.py` has
only per-test eager slow-skip entries, never a whole-file one, so the rest of it
already ran traced *before* #1279 — nobody was looking, because no workflow runs
`--jit`. All 8 pass eager. This is traced-only exposure being written down.

## Awaiting Jo

**One decision:**

* **Should CI gain a `--jit` dimension?** No workflow runs `--jit` today
  (checked across all 10 files in `.github/workflows/`), so the traced ledger,
  the traced slow-skip list and `assert_jit_matches_eager`'s premise are
  validated on the dev box and **nowhere else**. Options: a nightly shard on the
  two measured files / fold into `backend-tests.yml` / accept and document.
  Needs `workflow` scope, which my token lacks.

**One action:** merge **gh#1274** (green, targets `main`). It is now
`ready` — it had been left as a draft, and therefore un-mergeable, which is why
this line sat here unactioned for several days.

**One thing worth knowing:** `build.yml` triggers on both `push:` (no branch
filter) and `pull_request:`, so **every in-repo PR runs ~52 build jobs where 26
would do**. Observed directly: develop's `f963eb42` carried two complete,
identical job sets created 3 s apart. A branch filter on the push trigger halves
the CI bill. Also a workflow edit.

## In flight locally

A serial traced sweep of the **10 files** whose eager-jax slow-skip entries stop
being inherited under #1279 but were never measured traced (54 entries). Serial
because concurrent jax oversubscribes the XLA CPU pool and corrupts the timings.

**8 of 10 done:**

| file | tests | eager entries | → jax-jit | un-deferred |
|---|---|---|---|---|
| `test_dynamfric` | 7 | 1 | 1 | — |
| `test_FDMdynamfric` | 9 | 3 | 2 | 1 |
| `test_galpypaper` | 15 | 1 | 0 | 1 |
| `test_streamspraydf` | 38 | 1 | 0 | 1 |
| `test_interp_potential` | 40 | 6 | 1 | 5 |
| `test_qdf` | 56 | 2 | 1 | 1 |
| `test_orbits` | 116 | 2 | 2 (1 **new**) | 1 |
| `test_actionAngle` | 209 | 1 | 1 (1 **new**) | 1 |
| `test_potential` | 1385 | 0 | 0 | — (**8 defects**, below) |
| | | **17** | **8** | **11** |

**Two-thirds of the eager deferrals do not survive a traced run**, and the
measurement found things inheritance cannot produce at all:

* `test_FDMdynamfric` had it wrong in **both** directions at once — one test
  deferred that traces fine, two slow ones with no traced entry.
* **2 of the 8 keepers were never eager entries** (`test_bruteSOS_3D` 277 s,
  `test_actionAngleStaeckel_python_c_freqsAngles` 300 s). Inheritance can only
  propagate eager entries forward, so it could never have found these.
* One threshold call: `test_bruteSOS_3D` at 277 s sits *inside* the 240–300 s
  band. Every other keeper hit the cap. (An earlier claim that the two
  populations were cleanly separated was **wrong** and is corrected — it held
  for 6 files and broke on the 7th.)

**Zero defect candidates** across 490 traced tests in 8 files.

Also settled a prediction: `jax-jit StaeckelGrid_basicAndConserved_actions`
**XPASSES traced** (65.1 s). #1279's backend-aware bar fixes the traced arm too,
so that ledger entry is stale — it goes in the follow-up, not by amending #1279
(a new commit there restarts 153 checks for hours). Ledger net becomes
**17 → 13**.

`test_potential.py` (9th file) added no slow entries but **8 defects** — the
first time the slow-vs-defect split had anything to separate, and it kept all 8
out of the slow-skip list. Three groups, each now naming a *mechanism*:

| n | what | mechanism |
|---|---|---|
| 5 | `poisson_surfdens[mockRotatedAndTilted…]` | `RotateAndTiltWrapperPotential` never sets `_backend_accepts_arrays`; the guard's `is_backend_array` half *is* satisfied under tracing. A one-flag opt-in — but the flag must not be flipped before verifying array-correctness, since the gate exists to keep arrays out of bare-numpy internals |
| 2 | `test_McMillan17`, `test_Cautun20` | one root: `mass()` integrates with `scipy.integrate.quad`, which calls `float()` and rejects a tracer. Cautun20 fails in `__init__`, so it cannot even be *constructed* traced |
| 1 | `forceAsDeriv[AnyAxisym…]` | traced `Phi` at **exactly z==0** is off by ~7.3e-5 from its own neighbour; the test's one-sided difference divides that by dz=1e-8 → 7.35e+03. `zforce(0.5,0)=0` is correct — the derivative is the wrong side |

The AnyAxisym one was **A/B'd against pre-#1278** by swapping only that file:
byte-identical (`7.345878e+03` both sides). **Pre-existing, not the grading
change that landed today.** The first probe returned `diff = 0` and I explicitly
refused to read that as exoneration — it used a centred difference, which
cancels across z=0 by symmetry, and the wrong surfdens.

Remaining: `test_orbit` (245 functions), in flight, 0 failures so far.

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
