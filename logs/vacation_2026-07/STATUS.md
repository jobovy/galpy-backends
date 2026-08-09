# Vacation status snapshot — 2026-08-09 (19:20 EDT)

**This file is OVERWRITTEN, not appended.** It is the "where are things now"
answer. `log.md` is the trace; read this first and go there only for the why.

---

## PR pipeline

develop is at **`277bf897`**.

| state | PR | effect |
|---|---|---|
| **merged today** | gh#1279, gh#1280, gh#1281, gh#1283, **gh#1284** | develop `f963eb42` → `d2ad93c9` → `a45a4f1d` → `de0745c5` → `9dc31569` → **`277bf897`** |
| **open, FOR JO** | **gh#1274** → `main` | **151 green, MERGEABLE.** Targets `main`, so it is yours. Also the sole blocker on #211(B) — see below |
| **open** | **gh#1285** `backend-jit/rt-vertical-split` @ `a2d24422` | the RotateAndTilt vertical-split fix, **clears 10 ledger entries**. Just gained a coverage commit (below); CI restarted from it |
| **open** | **gh#1286** `backend-jit/eager-undefer-measured` | **−24 measured-stale eager-jax entries**, `slow_skip` 108 → 84. Every dropped entry verified fast AND green. CI just started |
| **filed as issues** | gh#1276, gh#1277 | two pre-existing numpy accuracy defects, mpmath-validated |

Two PRs at a time: **gh#1285 + gh#1286**, both open, neither blocked on me.

**gh#1285 needs no rebase.** The standing plan said to rebase it onto the new
develop and drop the 5 torch-jit lines — but that means a **force-push** on an
open PR, which is forbidden and would discard its CI. `git merge-tree` shows the
un-rebased branch merges cleanly (0 conflicts), so those 5 lines become their own
follow-up, which also lets them get the **torch --jit** evidence the drop
actually requires (gh#1285 only gated eager torch).

## Where the ledger actually stands

Projected across all three in-flight PRs (**not** what `develop` shows — with
PRs in flight, pick targets from the projection):

| file | develop | projected |
|---|---|---|
| `backend_slow_skip.txt` | 108 (post-gh#1284) | **84** via gh#1286 |
| `backend_xfail.txt` | 28 | **23** after gh#1285, → 18 with the torch-jit follow-up |

    slow-skip projected:  jax 51 | jax-jit 26 | torch-jit 6 | torch 1

**Torch eager is essentially finished** — 1 slow-skip + 3 xfail entries left in
the whole suite. The residual burndown is ~92 % jax.

**And the jax remainder is mostly real, not stale.** Measured eager-jax
staleness so far is 24 of 44 (55 %), but what is left is documented slow:
`test_DehnenBinney98` ~2641 s, the AndTilted MWP14 Poisson checks ~1750–1840 s,
`test_Cautun20` ~1440 s — 5–9× the cap. Those do not fall to a stopwatch; they
need SCF/Multipole/wrapper vectorisation. **The burndown is near its floor; the
next real lever is vectorisation.**

**The eager-jax burndown is now CLOSED, and the last measurement refuted my own
hypothesis.** I re-ran the 9 jax `test_interp_potential` + `test_FDMdynamfric`
entries expecting the adapter work (#1271/#1272/#1273) to have gutted the
per-point XLA dispatch the ledger names as their cause. It had not:

    drop = 0    keep = 9    defects = 0        (49 tests, 1935.1 s)

Four hit the 900 s cap (`Failed: Timeout` — *slow*, not defects; reading the
letter would have moved four correct entries into the xfail ledger), the other
five ran 356-701 s against a 240 s bar. `test_interpolation_potential_c` was
ledgered at 263 s and now measures **438.6 s**. So the entries stand.

Cumulative: **24 of 44 measured stale** (noninertial 11/19, test_orbit 13/25),
**0 of 9** for interp/FDM, and 12 jax `test_potential` entries documented at
1440-2641 s that no stopwatch will rescue. Every eager-jax cluster worth
measuring has now been measured.

## What the coverage check found in gh#1285

Asked codecov *which* lines it flagged instead of waiting for the verdict. Two
real gaps, both in code the PR adds, both of which would have failed the check
(`codecov.yml` sets no patch target → default `auto` = base **99.95 %**, and
50/51 = 98.04 %):

* `RotateAndTiltWrapperPotential.py:189` — the early `return 0.0` for a
  no-rotation, no-offset wrapper. Constructible, verified.
* `CompositePotential.py:318` / `:352` — the backend branch of **both**
  delegating surfdens routes, run by **none** of that file's 527 tests. The
  third of the four links in the fix had no test at all.

Both now covered; whole file 529 passed. Writing the test caught a 54 % error in
my own reference (the density route applies each component's `_amp`).

**New pre-existing defect, filed not folded in:** a `CompositePotential` holding
a dissipative force is not backend-compatible, so `@backend_input` skips the
coercion while `get_namespace` still resolves to the forced backend — `z`
reaches `xp.abs` as a Python float and torch raises. A/B'd: **identical on clean
develop**, so it is not from this PR.

## Awaiting Jo

**One action:** merge **gh#1274** (151 green, `MERGEABLE`, targets `main`). It
now blocks more than itself: its hunks land at lines 246/271/296/324/346/368/390
of `interpRZPotential.py`, which are exactly the seven grid-build loops
**#211(B)** would vectorise. Verified at hunk level, not guessed.

**One decision:** should CI gain a `--jit` dimension? `--jit` is set only by a
`workflow_dispatch` with `jit=true`, so the traced ledger is validated on the dev
box and **nowhere else**. Needs `workflow` scope, which my token lacks.

**Two workflow gaps, same missing scope:** no `concurrency:` group (superseded
SHAs keep burning runners — I force-cancel them by hand each push; `gh run
cancel` silently no-ops on queued runs), and `build.yml` triggering on both
`push:` and `pull_request:` (~52 build jobs where 26 would do).

## Standing caveats

* **numpy stays byte-identical.** Every deviation is flagged, never quietly shipped.
* **Traced green is asserted by me, not enforced by CI** — see the decision above.
* `actionAngleVerticalInverse` and `actionAngleTorus` are permanently out of scope.
* The 98 in-place augmented array writes remain a **watchlist, not a defect list**.

## Queued next, in order

1. **gh#1285** and **gh#1286** → green, then merge (audit each at merge time).
2. Follow-up PR: drop the 5 torch-jit `poisson_surfdens` lines — needs a
   **torch --jit** run to back the claim, which gh#1285 never gated (it gated
   eager torch). Measurement dispatched.
3. **Vectorisation, not measurement** — this is now the only lever left on the
   jax ledger. SCF/Multipole/wrapper eval is where the 1440-2641 s entries live.
4. **#211(B)** `interpRZPotential` grid-build vectorisation — blocked on gh#1274.
5. Follow-up: the dissipative-composite forced-backend coercion defect
   (characterised, deliberately not patched at the symptom site).
