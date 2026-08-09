# Vacation status snapshot — 2026-08-09 (09:25 EDT)

**This file is OVERWRITTEN, not appended.** It is the "where are things now"
answer. `log.md` is the trace; read this first and go there only for the why.

---

## PR pipeline

| state | PR | effect |
|---|---|---|
| **merged since the last snapshot (08-04)** | ~19 PRs, `#1256` → `#1278` | develop is now `f963eb42` |
| **MERGED** | **gh#1279** | `--jit` slow-skip stops inheriting eager; 11 measured traced entries; 3 tolerance burned. develop `f963eb42` → **`d2ad93c9`** |
| **MERGED** | **gh#1280** | streamdf out-of-place LB jacobian; **10 jax-jit ledger entries cleared**. develop `d2ad93c9` → **`a45a4f1d`** |
| **open, FOR JO** | **gh#1274** → `main` | **151 green.** Targets `main`, so it is yours to merge. It was still marked **draft** until 2026-08-09 — i.e. un-mergeable while I was asking you to merge it; now `ready`, blocked only on required review |
| **open** | **gh#1281** `backend-jit/sweep-measured-entries` @ `1be681c3` | the completed 10-file sweep: 15 measured jax-jit slow-skip entries, 8 diagnosed traced defects, 1 un-ledger. Rebased onto `a45a4f1d` through two real conflicts |
| **open** | **gh#1282** `backend-jit/kd-mass-quadrature` | **fixes 2** of gh#1281's 8 traced defects (KD `_mass` → backend quadrature, clearing `McMillan17`+`Cautun20`). I earlier described this as *7 of 8*: the RotateAndTilt half **failed its pytest gate and is reverted** — opting the wrapper into array input removes a `TypeError` but exposes a silent **6.8%** Poisson-surfdens error at \|Z\|=10, because the traced path uses fixed-order GL where scipy's adaptive quad coped |
| **filed as issues** | gh#1276, gh#1277 | two pre-existing numpy accuracy defects, mpmath-validated |

**CI right now:** gh#1281 29 green / 22 running / 101 queued, and the single
red is **`codecov/project`, which is an upload phantom** — `coverage=15.03%`
with `sessions=0` at 25/153 checks, i.e. almost nothing has uploaded yet. It
self-heals; it is only worth acting on if it is still red once CI completes.

gh#1281 also spent **two hours at 2/151 for a reason worth knowing**: its own
**pre-rebase SHA's runs were still executing** and holding every runner, while
the real head sat queued behind them. `gh pr checks` cannot show this — it only
ever reports the head. Cancelling the orphans moved it to in-progress within
five minutes. See the two workflow gaps under *Awaiting Jo*.

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

## Torch sweep — running now

gh#1281's body used to say a torch sweep was blocked on a `parallel_map`
`--jit` deadlock. **That was stale**: #1266 makes `parallel_map` fall back to a
serial `map` whenever traced, so it never forks. Verified by running
(`test_backend_multi.py`, 3 passed, 10.66 s under `--backend torch --jit`), not
by reading. I had carried that blocker for days and written it into a PR.

Four files (the three carrying torch's 15 eager entries, plus `test_potential`
because **defects are per-backend** and jax's 8 all lived there). Three done:

| file | result |
|---|---|
| `test_dynamfric` | 1 fail: `test_dynamfric_c` at the 300 s cap → genuine slow, **keeper** |
| `test_FDMdynamfric` | wrong in **both** directions: its eager entry `central_limit` is not slow at all, it **fails in 35.9 s** (`numpy.amax` over a Tensor — the file already applies the `as_numpy` fix at 4 other lines, 49/59 were missed); and `const_FDMfactor` hits the cap but was never an entry |
| `test_potential` | 1385 tests, **1 slow + 1 defect** (jax: 0 slow, 8 defects) |
| `test_orbit` | running — 13 of torch's 15 entries live here |

**Of jax's 8 traced defects exactly one reproduces on torch** (`AnyAxisym
forceAsDeriv` — which is the cross-backend confirmation that it is pre-existing,
not caused by gh#1278). The 5 RotateAndTilt ones become a *performance* problem
on torch, not a correctness one. `McMillan17`/`Cautun20` **cannot be compared**:
that tree carried my KD fix, so them passing is evidence about the fix, not
about torch.

Backend-specific traced behaviour is the rule, not the exception — which is the
case for having run this rather than inheriting jax's answers.

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

**Two workflow gaps, both needing scope my token lacks.** Together they cost
gh#1281 two hours today:

1. **No `concurrency:` group.** After a rebase or force-push the superseded
   SHA's runs keep executing and hold the runners. `concurrency: {group:
   <workflow>-<ref>, cancel-in-progress: true}` would auto-cancel them. I did it
   by hand today; without the fix it needs a human every time.
2. **`build.yml` triggers on both `push:` (no branch filter) and
   `pull_request:`**, so every in-repo PR runs ~52 build jobs where 26 would do.
   On a runner-constrained account this is not just a 2x bill, it is 2x
   wall-clock. With (1) unfixed it compounded to **four** complete job sets live
   for one PR.

## The traced sweep — DONE, and it is gh#1281

All **10 files**, ~2456 traced cases, ~5 h serial (concurrent jax oversubscribes
the XLA CPU pool and corrupts the very timings being collected). **42 eager
entries → 15 measured jax-jit entries, + 8 diagnosed defects, + 1 un-ledgered.**

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
| `test_orbit` | 581 | 25 | **7** | 18 |
| | | **42** | **15** | **26** |

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

`test_orbit` finished **opposite to the average**: across the other nine files
two-thirds of eager entries were dropped, but here most survive (25 → 7 is still
a big cut, yet every survivor is a genuine timeout). Per-file variation that
large is what a single inheritance rule — set either way — cannot express.

Also corrected: **3** of the 15 sit inside the 240–300 s judgement band
(296/277/265 s), not "only one threshold case" as an earlier note claimed.

## Queued next, in order

1. ~~Merge #1279, #1280~~ — both **MERGED**, develop at `a45a4f1d`.
2. ~~File the sweep's measured entries~~ — **gh#1281**, open, CI running.
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
