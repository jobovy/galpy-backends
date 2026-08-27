# Vacation status snapshot — 2026-08-27

**This file is OVERWRITTEN, not appended.** "Where are things now"; `log.md` is
the trace. Refresh on every merge / branch change.

---

## Now

    develop  feat/backends-develop @ 8c320b39
    open PR  NONE
    ledger   8 xfail + 57 slow-skip = 65 active
    cron     5451d57e, hourly at :13, created 2026-08-24, expires ~2026-08-31

`/var/tmp/jitsafe` = worktree for the MERGED gh#1401 branch (keep until #1402
lands; it is the byte-identity "branch" arm's sibling).
`/var/tmp/h2single` = worktree for gh#1402.
`/var/tmp/h2base` = detached at develop, the byte-identity BASE arm.

## The gh#1204 review is fully addressed — three rounds

    round 1+2  ->  gh#1381 gh#1382 gh#1383 gh#1388 gh#1391   (status comment 5422012558)
    round 3    ->  gh#1401                                    (reply comment 5434247936)

Round 3 raised two PR-local findings (traced beta under external jit;
symmetric_quad mixed finite/infinite limits) plus an audit note on small-B
hyp2f1. All three are IN develop via gh#1401.

## Merged into develop this cycle (ledger delta 0 on every one)

    00a7153c  gh#1401  review round 3 + small-B hyp2f1 floor cleared
    5c3c8671  gh#1399  actionAngleAdiabaticGrid exact off-grid actions
    bc69771e  gh#1394  surfdens both routes on-backend under eager torch grad
    366ffc04  gh#1393  Einasto d_n gradient un-gated to torch + f64 pin
    8980402a  gh#1392  constantbetaHernquistdf d f_E/d beta
    cfe23abe  gh#1391  traceable constructors + hyp2f1/hyp1f1 parameters
    107a8060  gh#1388  map_coordinates dispatches on the GRID
    4e2cc322  gh#1383  coords inverse galcen + f64/f32 promotion
    06267d1d  gh#1382  quadrature batching + AnyAxisym torch construction
    206ccd78  gh#1381  gammainc endpoint gradient

Also merged to MAIN by Jo: gh#1398 (`3df11b96`), actionAngleAdiabatic `_justjr`
scalar return.

## gh#1402 — MERGED `8c320b39`, 154/154 green. Evidence

One tanh-sinh rule for every B; the `c - B >= 1` constraint deleted with it.

    768 cases, B ABOVE the old 0.25 threshold (the regime GL was kept for):
        GL          worst 1.5e-04
        tanh-sinh   worst 2.9e-15      cases where tanh-sinh is worse: 0

    900 cases through the PUBLIC entry, incl. z=0:
        develop (pre-1401)  8.807e-01
        gh#1401             4.577e-03
        gh#1402             2.665e-15

    numpy BYTE-IDENTICAL  sha256 245bc0e3... vs develop 00a7153c,
                          arms grep-asserted to differ
    gate                  385 passed / 1 skipped, re-run AFTER the rebase
    size                  91 insertions, 156 deletions (NET DELETION)

## FOR JO — open decisions, none taken unilaterally

1. **torch `surfdens(forcepoisson=True)` gradient is silently 23% wrong in R,
   72% in z** (jax AD matches FD to 1.6e-10, so torch is the wrong one).
   Reported, NOT fixed: the fix is a behaviour change (use the backend GL rule
   when `_force_accepts_arrays(pot)`, else RAISE). Own PR when you decide.
2. **Array beta on constantbetaHernquistdf** now raises a broadcasting error
   rather than "truth value ambiguous". Both loud, neither clear; array beta was
   never supported. A validation guard would be a small follow-up.
3. **gh#1277 / gh#1276** filed as issues on your instruction (AnyAxisym R2deriv
   at small R; RazorThinExpDisk force digits in a |z| band).

## Standing traps that keep re-biting (see log.md for the full list)

* A worktree branched BEFORE a fix still runs the code you deleted. Bit me
  twice this week; both times the symptom was a gate hanging, not failing.
* After a SQUASH merge, a follow-up branched from an intermediate commit needs
  `git rebase --onto <develop> <old-base> <branch>`, NOT a plain rebase.
* `all-clear` CONCLUDES; it does not re-evaluate when the shard it gated on is
  re-run. It needs its own targeted re-run.
* codecov/patch and codecov/project are DIFFERENT signals. project self-heals
  when uploads complete; patch can be a real uncovered-lines finding.
* Local coverage tooling is broken in this env (numpy re-import). A branch whose
  omission RAISES can be verified by the test passing at all — that trick does
  NOT generalise to branches that silently return a different number.
