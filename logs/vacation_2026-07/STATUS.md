# Vacation status snapshot — 2026-08-10

**This file is OVERWRITTEN, not appended.** It is the "where are things now"
answer. `log.md` is the trace; read this first and go there only for the why.

---

## PR pipeline

develop is at **`de0bee8e`**.

| state | PR | effect |
|---|---|---|
| **merged** | gh#1279, gh#1280, gh#1281, gh#1283, gh#1284, **gh#1288** | develop → `9dc31569` → `277bf897` → **`de0bee8e`** |
| **open, FOR JO** | **gh#1274** → `main` | 151 green, MERGEABLE. Targets `main`, so it is yours. Still the sole blocker on #211(B) |
| **open** | **gh#1285** `backend-jit/rt-vertical-split` @ `82aa6309` | RotateAndTilt vertical-split fix, clears 10 ledger entries. Audited (see below). CI **runner-starved: 1 of 149 started** |
| **open** | **gh#1286** `backend-jit/eager-undefer-measured` @ `6e56330f` | −24 measured-stale eager-jax entries, `slow_skip` **109 → 85**. 13 green + the codecov upload phantom |
| **filed as issues** | gh#1276, gh#1277, **gh#1289** | pre-existing numpy accuracy defects, all mpmath/Simpson-validated |

### gh#1288 — a merge of mine that had to be undone

gh#1284 un-deferred `torch …test_FDMDynamicalFrictionForce_central_limit` on the
strength of a **6.47 s** measurement. That number was an artifact: the tree
lacked gh#1284's own `as_numpy` fix, so the test crashed early on the very
collision it was fixing. *A test that fails fast is not a fast test.* Re-measured
with the fix applied: **>658 s on an idle 72-core box against a 300 s cap.**
gh#1288 re-defers it. Verified functionally — the entry now reports `s` in 8.5 s.

### gh#1285's two reds, both resolved

* **Windows `test_streamgapdf_sample`** — a **flake, proven by construction**.
  `build.yml` triggers on both `push` and `pull_request`, so every commit runs
  the matrix twice. On `bcb2172e` the same shard came back `failure` (push) and
  `success` (pull_request). Same code, same platform, opposite outcomes. The PR
  check list surfaces only one of the two, which is why it looked unattributable.
* **`codecov/patch`** — was 50/51. The missed line was `Potential.py:871`, the
  `xp.where` in `_vertical_quad_interior`, reachable only when the mid-plane
  crossing is nonzero. Fixed with a test that pins the *contract* (both arms;
  the reject arm returns **0.0, not the nearer boundary**), verified executed
  with `trace.Trace`.

## Prepared locally, NOT pushed (do not lose these)

| branch | @ | what | status |
|---|---|---|---|
| `backend-jit/torchjit-poisson-undefer` | `c4acbf96` | xfail 26 → 21: drops the 5 **torch-jit** `poisson_surfdens` wrapper entries, keeps both AnyAxisym | ready. Re-verified 2026-08-10: disjoint from gh#1285 (which drops the 5 **jax-jit** rows — same tests, different mode prefix), merge-tree clean, 26 → 16 rows combined |
| `backend-jit/gl-order-100` | `2701dd8d` | `symmetric_quad` default 50 → `_QUAD_N` (100), **deleting** 3 duplicated `n=50` overrides rather than adding a constant, + a test that pins the DEFAULT | **push-ready.** Message amended with the corrected site attribution; test A/B'd (default passes, 50 fails); committed state re-verified green (quadrature 62 passed, wrappers 530 passed) |

Both must land **after** gh#1285.

## The GL-order change, and why it became a deletion

The plan said "add its own named constant". Reading first showed
`galpy/backend/quadrature.py` already owns `_QUAD_N = 100`, documented for
exactly this criterion ("smooth galpy integrands (mass/surface-density
profiles)"), and that `quad` already defaults to it — only `symmetric_quad`
disagreed at 50, with all three callers *also* passing `n=50`. So the magic
number appeared four times and the two defaults contradicted each other. Net
**+5/−5 lines, no new constant.**

Measured against mpmath at 30 dps, both branches of the rule:

    finite clustered, worst corner (R=0.01,|z|=50)      infinite (`mass`)
                      n=50      n=100                    n=50      n=100
    MWP14[0]      3.41e-06   7.29e-12                 2.20e-14   1.81e-15
    MiyamotoNagai 2.94e-08   1.10e-15                 6.78e-13   9.86e-16
    NFW           8.05e-07   1.41e-12                 7.85e-16   5.23e-16

Traced jax gate on the branch: 1385 shared nodeids, **0 outcome flips**, +0.2 %
time, restore verified. A **new test** now discriminates — it asserts on the
DEFAULT (no `n=`), so a lowered default fails it; A/B'd (default=`_QUAD_N`
passes, default=50 fails). The first draft of that test passed `n=_QUAD_N`
explicitly and would have sailed through the regression it exists to catch.

**Outstanding before push:** the commit message must be amended. It says numpy
is untouched *because* `_quad_needs_backend` routes numpy and eager to scipy —
true for numpy, wrong for eager.

The site is **`surfdens(R, z=None)`**, not `_mass` (all three `symmetric_quad`
call sites are in `surfdens`/`_surfdens_poisson`/`_surfdens`; none in `mass` —
earlier notes had this wrong). It is guarded by a plain `if xp is numpy:` rather
than `_quad_needs_backend`, so eager backends *do* reach the rule there.

The eager suite arms I ran are **vacuous**: no test calls `surfdens` with
`z=None`, so AFTER==BEFORE means "untested", not "unchanged". The real evidence
is direct parity on that branch at the current n=50 (MiyamotoNagai 6.7e-13,
Hernquist 5.7e-16, NFW 1.3e-15, MWP14[0] 2.9e-11) plus the mpmath table showing
n=100 equal-or-better there everywhere.

## The eager-GL routing question — settled, and it is a separate PR

Both arms of `test_potential.py -k surfdens` under eager torch completed:

    scipy (today)   19 passed, 0 failed   1110.5 s
    GL forced       18 passed, 1 failed     39.3 s

The one failure is `AnyAxisym`, the deliberate array opt-out, and the guard does
fire **loudly** through composition and wrapping (never silently wrong).

**But the design I proposed on the back of that is refuted.** I claimed
array-safety "is decidable" via `_backend_accepts_arrays`. Measured 2026-08-10,
that flag is wrong in *both* directions:

    MiyamotoNagai        declares False  -- but works fine with arrays
    RotateAndTilt(Any)   declares TRUE   -- while WRAPPING a scalar-only potential

which is exactly `_quad_needs_backend`'s documented point. The failure is also
not a clean pre-flight rejection — the `TypeError` arrives *after* an integrand
call, and one path raises `AttributeError` instead. So try-GL/fall-back needs a
broad catch around a numerical routine, which can mask real defects.

**Do not build the eager-GL routing PR on that design.** It needs a different
mechanism (an explicit, verified per-potential opt-in, or a cheap array probe).
Nothing here touches the accuracy case for the order raise, which stands on the
mpmath table and the traced gate. **Timings above are contaminated** anyway (the
control shared the box with an 8-worker gate).

## Where the ledger actually stands

Projected across all three in-flight PRs (**not** what `develop` shows — with
PRs in flight, pick targets from the projection):

| file | develop (`de0bee8e`) | projected |
|---|---|---|
| `backend_slow_skip.txt` | **109** | **85** via gh#1286 |
| `backend_xfail.txt` | **26** | **21** after gh#1285, → **16** with the torch-jit follow-up (merge-tree verified) |

(The slow-skip absolute numbers moved twice under gh#1286 — gh#1284, then
gh#1288 re-adding the torch FDM line. Its **−24 delta** never changed; its title
and body were corrected from `108 → 84` to `109 → 85`.)

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

## Found: a real accuracy defect in the TRACED path — now fully gated

`surfdens(forcepoisson=True)` on a **traced** backend uses a fixed-order GL rule
at **n=50** and carries **2.6e-05 relative error at (R=0.01, |z|=50)** today.
Eager and numpy use `scipy.integrate.quad` and are correct, which is why nothing
catches it; the traced suite only runs on a `workflow_dispatch`.

Measured against adaptive on the real MWPotential2014 integrand:

    (R,|z|)          n=50      n=100      n=200
    (0.01, 50)   2.57e-05   1.24e-10   2.04e-14
    (0.1,  50)   1.81e-09   1.04e-14   3.87e-15

**n=100 fixes it, and the change is free.** Full traced `test_potential.py`
A/B, identical command and tree:

    BEFORE n=50    1278 passed, 102 skipped, 5 xfailed   593.6 s
    AFTER  n=100   1278 passed, 102 skipped, 5 xfailed   566.6 s

Compared per-test, not just by count: 1385 tests each, **0 outcome flips**,
summed time -0.6 %. numpy is untouched by construction (it keeps scipy).

**Design decision, from searching first:** a `_BACKEND_GL_ORDER = 50` already
exists in `galpy/actionAngle/actionAngle.py` (5 users). **Do not reuse it** —
`Potential.py` imports nothing from `actionAngle` and should not start
(wrong direction), and it orders a *different* quadrature (`fixed_quad` over
[0,1] on the action radicand, vs the split+clustered `symmetric_quad` over
[-|z|,|z|]). Sharing the symbol would silently re-order the AA quadratures,
unmeasured. The surfdens PR gets its own named order beside the rule it governs.

Ready to open as one PR (raise the order, then route eager through it) when a
slot frees; evidence complete, only the two-PR cap holds it.

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
