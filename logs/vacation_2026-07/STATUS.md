# Vacation status snapshot — 2026-08-28

**This file is OVERWRITTEN, not appended.** "Where are things now"; `log.md` is
the trace. Refresh on every merge / branch change.

---

## Now

    develop  feat/backends-develop @ b331faa0
    open PR  gh#1406  backend/hyp2f1-stale-comments -> develop  (CI running)
    in prog  backend/vertical-quad-order (/var/tmp/quadn) -- surfdens quadrature,
             code written + numpy gate green, jax measurement running
    regen    run 33087562112 (workflow_dispatch, regen=true, eager) QUEUED
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

## gh#1403 — MERGED `26388056`, 154/154 green

Fixed the review blocker (a FOURTH copy of the admissibility predicate, in
`_nonpositive_traced`, still on the old `c - B >= 1`: eager exact, jit inf/nan)
and removed the Euler-TRANSFORM branch, which gh#1402's relaxation made
provably dead (2031 parameter sets reach it under the old bound, ZERO under the
new one). That deletion is what repaired the coverage regression Jo flagged:
misses 16 -> 15, back to the value held across all eleven earlier merges.

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

## PRE-MERGE CHECKLIST (added after gh#1402 landed a coverage regression)

Before squash-merging, in addition to the diff audit:

    tools/codecov_miss_sweep.py   # misses must NOT increase vs the base commit

A GREEN codecov check is NOT the standard. gh#1402 passed both checks with
"All modified and coverable lines are covered by tests" while Misses went
15 -> 16: that phrasing covers MODIFIED lines only, and says nothing about a
line that was covered before and is not now. Read the Hits/Misses table.

Swept the whole cycle after the fact -- misses held at exactly 15 across all
eleven earlier merges, so gh#1402 was an isolated lapse, not a pattern.

## Ledger status — verified, not assumed

    8 xfail + 57 slow-skip = 65 active

xfail half VERIFIED from CI artifacts (7432 results, run on 00a7153c):
  * all 4 EAGER entries still legitimately xfail, ZERO xpass;
  * ZERO unledgered failures;
  * the 2 AdiabaticGrid entries are NOT stale despite gh#1399 -- re-measured
    with the C extension actually built (jax 1 xfailed / torch 1 xfailed,
    XPASS count 0 in both).

slow-skip half: CI SKIPS them in every mode, so junit can only ever say
"skipped" -- the `regen` toggle does NOT change this (it gates the xfail ledger
only). The only way to verify is to EMPTY the list locally and run.

  Done for the 6 jax `test_potential.py` entries, 2026-08-27, in /var/tmp/h2base
  (list emptied, C extension built, quiet box, whole file, 1h59m):

      900.0s FAIL(timeout)  poisson_surfdens[mockRotatedAndTiltedMWP14Wrapper]
      900.0s FAIL(timeout)  poisson_surfdens[...MWP14WrapperwInclination]
      900.0s FAIL(timeout)  poisson_surfdens[mockRotatedTiltedOffsetMWP14Wrapper]
      900.0s FAIL(timeout)  poisson_surfdens[mockOffsetMWP14Wrapper]
      349.8s pass          poisson_surfdens[...TriaxialLogHalowInclination]
      249.8s pass          test_DehnenBinney98

  All 6 REPRODUCE their 2026-08-21 annotations (which recorded >700s timeouts
  and 371.9s / 289.3s for the two passers; today's box was ~6-14% quieter).
  So: none are stale, and the two passers stay deferred -- at CI~250s and
  CI~180s they sit only 12-32% under the 300s cap.

  The 4 timeouts are SLOW, not hung: the same test passes on the
  single-component LogHalo wrapper, and cost scales with the wrapped
  component count.

Harvester: `tools/ledger_harvest.py`, arm-aware. Validating it caught a
-jit-vs-eager false-stale bug IN THE HARVESTER (it read 4 torch-jit entries as
stale by comparing them against plain-torch results).

## gh#1406 — review follow-up, comments only

The codex review's last item: comments still describing the deleted
Euler-transform route and the replaced GL/substitution rule as current. Two
named `_TS_B_MAX`, a constant with ZERO occurrences in hyp2f1.py. Swept both
files for the stale vocabulary and classified each hit present-tense-wrong vs
correctly-past-tense; 8 fixed, the history passages left alone. 253 passed.

## gh#1407 — surfdens vertical quadrature, MEASURED END TO END

Eager jax/torch sent both vertical surface-density integrals to
scipy.integrate.quad (1113 scalar integrand calls @ ~48 ms for a wrapped
MWP2014). Routed to the batched GL rule the TRACED path already uses:
22-83x faster at worst 1.6e-14 vs scipy converged to 1.4e-16.

Whole test_potential.py under jax, the 6 entries un-deferred:

    before   4 failed, 1284 passed, 102 skipped   7144 s (1:59:04)
    after    0 failed, 1288 passed, 102 skipped   2733 s (0:45:33)

    poisson_surfdens[mockRotatedAndTiltedMWP14]   >900 FAIL -> 20.5 s
    poisson_surfdens[mockRotatedTiltedOffset..]   >900 FAIL -> 19.1 s
    poisson_surfdens[mockOffsetMWP14Wrapper]      >900 FAIL -> 17.1 s
    poisson_surfdens[..MWP14wInclination]         >900 FAIL -> 16.5 s
    poisson_surfdens[..TriaxialLogHalowIncl]      349.8 s   ->  8.0 s
    test_DehnenBinney98                           249.8 s   -> 77.9 s

All six >80% under the 300 s cap -> all six un-deferred. The file is now FASTER
with them running than it was with them skipped.

THE GATE IS NOT `_force_accepts_arrays` -- using it did nothing, and the smoke
test caught it (`symmetric_quad calls=0`). `_surfdens_poisson` runs on the inner
RotateAndTiltWrapperPotential, whose forces carry the scalar-only marker (right
for the NUMPY contract) while the potential declares `_backend_accepts_arrays`,
the escape hatch `check_potential_inputs_not_arrays` implements for backend
arrays. The gate now asks the question the guard asks.

numpy byte-identical (gate returns False on its first line; a test pins it).

## READY follow-ups (measured, not opened — 2 PRs already at the cap)

**(a) A WRONG de-ledgering on develop.** `test_pickling_potential_unitful_dens`
was de-ledgered 2026-08-18 as "MEASURED PASSING traced"; it FAILS on jax-jit AND
torch-jit (passes both eager) — exactly what the ledger's own comment predicts
for a units density hitting `as_numpy()`. The 2026-08-18 regen almost certainly
never reached it and absence-of-data was read as a pass. Fix = re-add 2 lines +
correct the comment. Zero CI impact (CI never runs jit mode); only a local jit
run could find it.

**(b) READY, MEASURED: 4 more ledger entries.** `test_integrate_backwards` and
`test_integrate_negative_time` bypass `_backend_integrators()`, the existing
helper that drops pure-Python integrators under a backend. Applying it:
608.3 s -> 5.0 s and 401.4 s -> 0.7 s (same-session A/B). Burns 2 jax + 2
jax-jit entries. numpy unaffected by construction. Verify full-file context
before opening.

**(c) `Potential.mass` asks the wrong array-safety predicate** -- SHIPPED in gh#1408

`Potential.mass` asks `_force_accepts_arrays` for the same array-safety
decision gh#1407 fixes for surfdens, so it inherits the same wrong answer: the
predicate returns False for a potential that declares `_backend_accepts_arrays`,
and mass() then drives its backend quadrature node-by-node.

Only ONE class is actually affected -- the two opt-in classes are
RotateAndTiltWrapper (mass() raises NotImplementedError for non-axisymmetric
potentials, so it can never reach this) and DoubleExponentialDisk. Measured on
the latter, eager jax:

    vectorized=False (today)   0.62 s
    vectorized=True  (fixed)   0.08 s      mass identical to 12 digits

~8x, but 0.6 s absolute and NO ledger entry rides on it. So this is a
consistency fix, not a burndown: after gh#1407 two call sites ask array-safety
differently, which is exactly what a reviewer should flag. Open it when a PR
slot frees up; do not amend gh#1407 for it (its CI is queued behind a ~3 h
backlog and a new commit would throw that away).

## WHERE THE LEDGER ACTUALLY STANDS — classified 2026-08-28

After gh#1410 lands: **10 xfail + 47 slow-skip = 57**. By file:

        16  test_orbit.py          9  test_noninertial.py     7  test_FDMdynamfric.py
         4  test_orbits.py         4  test_streamTrack.py     3  test_dynamfric.py
         2  test_qdf.py            2  test_actionAngle.py

**39 of the 47 slow-skips are orbit-integration cost** (orbit + noninertial +
FDMdynamfric + orbits + dynamfric). That is one cause, not thirty-nine.

### The test-side integrator-reduction lever is SPENT

`_backend_integrators()` (drop pure-Python integrators under a forced backend)
is what burned entries all cycle, and gh#1410 took the last clean pair. Audited
every remaining `test_orbit.py` entry -- 10 distinct tests behind the 16 rows:

        5  ALREADY use it   ecc_rperi_rap, zmax, analytic_zmax,
                            energy_jacobi_conservation, liouville_planar
        2  BY DESIGN        test_dxdv_3d_c_vs_python,
                            test_integrate_indiv_t_python_paths
                            -- the Python path IS the subject; filtering guts them
        1  already _c-only  lyapunov_spectrum_henonheiles_chaotic
        1  single Python    test_interprz_dxdv_3d (dop853 only -> filter deletes it)
        1  partial          lyapunov_spectrum_consistency (dop853 + dop853_c)

`test_noninertial.py` is the same story, checked call-by-call. Of its 5 tests,
`_omegadot` and `_omegafunc` have exactly ONE `check_orbit` call each, at pure-
Python `dop853` -- filtering deletes the test. `test_arbitraryaxisrotation`
varies `zvec`/`omega` ACROSS methods, so dropping the Python calls drops
configurations, not just duplicate integrators. Only the two `linacc_changingacc`
tests have gh#1410's shape (method+tol only), and even there the Python arm
asserts at **tol=1e-9** while the `_c` arm asserts at **1e-5** -- so the filter
would trade 3 ledger entries for the strictest assertion in the file. Timing
probe running to see whether the trade is even worth proposing.

### So the remaining 39 need a different lever

Eager backends pay ~5 ms per potential call (task #79) against ~27 us on numpy.
A Python-stepped ODE makes ~1e4 such calls per orbit, so these tests are ~200x
slower *structurally*. Vectorizing cannot help -- ODE steps are sequential by
construction -- and `no-internal-jit` rules out compiling inside galpy. The only
two real exits are (a) the C integrators, which is exactly what the now-spent
test-side lever routed to, and (b) the in-backend diffrax/torchdiffeq ODE path.

**This is a roadmap item for Jo, not an opportunistic wake-time fix.** I am not
proposing to weaken assertions or delete Python-path coverage to move the count.

## FOR JO — open decisions, none taken unilaterally

1. ~~torch surfdens(forcepoisson=True) gradient~~ **CLOSED** — re-measured on
   develop: torch AND jax both match FD to 3.2e-11 (R) / 7.7e-11 (z). gh#1394
   was the fix; my earlier "open decision" entry was stale.
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
