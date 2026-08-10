# Vacation status snapshot — 2026-08-10

**This file is OVERWRITTEN, not appended.** It is the "where are things now"
answer. `log.md` is the trace; read this first and go there only for the why.

---

## PR pipeline

develop is at **`5b08b507`**.

| state | PR | effect |
|---|---|---|
| **merged** | gh#1279, gh#1280, gh#1281, gh#1283, gh#1284, gh#1288, **gh#1285**, **gh#1286** | develop → `de0bee8e` → `fb159ff7` → **`5b08b507`** |
| **open, FOR JO** | **gh#1274** → `main` | 151 green, MERGEABLE. Targets `main`, so it is yours. Still the sole blocker on #211(B) |
| **open, CI running** | **gh#1291** `backend-jit/torchjit-poisson-undefer` @ `2d9be8da` | xfail **21 → 16**: the 5 **torch-jit** `poisson_surfdens` wrapper rows. Ledger only, 1 file, `0+ / 5−` |
| **open, CI running** | **gh#1292** `backend-jit/gl-order-100` @ `dfe8c4f3` | `symmetric_quad` default 50 → `_QUAD_N`, deleting 3 duplicated `n=50` overrides + a test that pins the DEFAULT |
| **filed as issues** | gh#1276, gh#1277, gh#1289 | pre-existing numpy accuracy defects, all mpmath/Simpson-validated |

Two-PR cap is full. Nothing else may be pushed until one of these merges.

### Both merges this wake, and what they cost

gh#1285 (RotateAndTilt mid-plane split) and gh#1286 (−24 eager-jax slow-skips)
both reached `all-clear = SUCCESS`, 152 checks, zero reds, and were squash-merged
after re-verifying each diff against the earlier audit.

Between them they carried **five reds, of which exactly one was real**:

    gh#1285  windows streamgapdf flake  -> re-run (flake proven by same-SHA split)
             codecov/patch              -> ONE REAL GAP, fixed with a test
             codecov/project            -> upload phantom, self-healed
    gh#1286  windows streamgapdf flake  -> re-run
             codecov/project            -> upload phantom, self-healed

Four of five needed **no source change**. Each was diagnosed with the right
instrument — codecov's `sessions` count on the compare endpoint to tell an
incomplete upload from a real drop, and the same-SHA push/pull_request duplicate
to prove the Windows failure a flake by construction — rather than by hunting in
the diff, which would have found nothing.

`all-clear` trails **31 eager-backend shards** (18 torch, 13 jax). Nothing is
hung; that is simply what the gate waits on, and it is why the eager-GL routing
item below is really a CI-throughput item.

### Rebasing the two queued branches after a SQUASH merge

Both were prepared against pre-merge bases and both needed surgery:

* `gh#1291` was written when the ledger read 26; gh#1285 removed the five
  **jax-jit** rows first, so its claim became **21 → 16**. Title and body
  corrected before opening — a stale PR description is worse than none.
* `gh#1292` sat on gh#1285's **pre-squash merge commit** `82aa6309`, so a plain
  rebase would have replayed commits that are already in develop under a
  different SHA. Used `git rebase --onto develop 82aa6309` to keep only my two.

Post-rebase both were re-verified at the hunk level, not by "it applied cleanly":
gh#1292's three `n=50` deletions still sit at the three real `symmetric_quad`
call sites, now carrying gh#1285's `interior_point=` argument alongside.

## Where the ledger actually stands

| file | develop (`5b08b507`) | projected |
|---|---|---|
| `backend_slow_skip.txt` | **85** | 85 (no PR in flight touches it) |
| `backend_xfail.txt` | **21** | **16** via gh#1291 |

    slow-skip:  jax 51 | jax-jit 26 | torch-jit 6 | torch 1

**Torch eager is essentially finished** — 1 slow-skip + 3 xfail entries left in
the whole suite. The residual burndown is ~92 % jax.

**The eager-jax burndown is CLOSED at the stopwatch.** Cumulative: **24 of 44
measured stale** (noninertial 11/19, test_orbit 13/25), **0 of 9** for
interp/FDM, **0 of 5** for the AndTilted Poisson entries, and 12 jax
`test_potential` entries documented at 1440–2641 s that no re-measurement will
rescue.

The AndTilted re-measurement after gh#1285 is the clearest negative result:
gh#1285 cut poisson-route boundary crossings 5307 → 2299, and the five entries
still run **411–1633 s against a 240 s bar** (1.7× to 6.8× over). They pass; they
are just slow. Wall time follows the *direct* route, which is not what gh#1285
halved. **The next lever is vectorisation, not measurement.**

## The eager-GL routing question — the real throughput lever, mechanism still open

Both arms of `test_potential.py -k surfdens` under eager torch completed:

    scipy (today)   19 passed, 0 failed   1110.5 s
    GL forced       18 passed, 1 failed     39.3 s

(Timings **contaminated** — the control shared the box with an 8-worker gate.
The pass/fail counts are what decides the design, and they are unaffected.)

That ~28× is aimed at exactly the cluster the ledger cannot otherwise clear: the
AndTilted `poisson_surfdens` entries above are eager-jax surfdens tests. So this
is not a tidiness item — **it is the vectorisation lever for the largest
remaining ledger cluster.**

**But the design I proposed on the back of it is refuted.** I claimed
array-safety "is decidable" via `_backend_accepts_arrays`. Measured 2026-08-10,
that flag is wrong in *both* directions:

    MiyamotoNagai        declares False  -- but works fine with arrays
    RotateAndTilt(Any)   declares TRUE   -- while WRAPPING a scalar-only potential

which is exactly `_quad_needs_backend`'s documented point. The failure is also
not a clean pre-flight rejection — the `TypeError` arrives *after* an integrand
call, and one path raises `AttributeError` instead. So try-GL/fall-back needs a
broad catch around a numerical routine, which can mask real defects.

**Do not build the eager-GL routing PR on the declared-flag design.** It needs an
empirical mechanism: probe the *integrand itself* once with a 2-element array,
check shape, cache the verdict — decided where the failure actually surfaces
rather than at a declaration that nothing validates. Prototyping now; it is a
separate PR and both slots are full.

Nothing here touches the accuracy case for gh#1292, which stands on the mpmath
table and the traced gate.

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

* **numpy stays byte-identical.** Every deviation is flagged, never quietly
  shipped. gh#1292 verified at the source: all three `symmetric_quad` call sites
  return through `scipy.integrate.quad` before they can reach the rule.
* **Traced green is asserted by me, not enforced by CI** — see the decision above.
* `actionAngleVerticalInverse` and `actionAngleTorus` are permanently out of scope.
* The 98 in-place augmented array writes remain a **watchlist, not a defect list**.

## Queued next, in order

1. **gh#1291** and **gh#1292** → green, then merge (audit each at merge time).
2. **eager-GL routing**, on an integrand probe rather than the refuted flag —
   the one change that can move the AndTilted cluster.
3. **Vectorisation** of SCF/Multipole/wrapper eval, where the 1440–2641 s
   entries live.
4. **#211(B)** `interpRZPotential` grid-build vectorisation — blocked on gh#1274.
5. Follow-up: the dissipative-composite forced-backend coercion defect
   (characterised, deliberately not patched at the symptom site).
