# Vacation status snapshot — 2026-08-18 (gh#1316+gh#1333 merged; queue-bound)

**This file is OVERWRITTEN, not appended.** It is the "where are things now"
answer. `log.md` is the trace; read this first and go there only for the why.

---

## Nothing is blocked on Jo except merging main PRs

## Branch state

    main                   c7891fca   (gh#1323, gh#1324, gh#1333 all merged)
    feat/backends-develop  b4a21907   (gh#1316 merged 2026-08-17)

## Open PRs

**gh#1333 → `main`** `fix/anyaxis-numpy-midplane-cusp`, head `07549c0a`.
**NEW today. Jo's to merge.** Two changes that cannot be separated:

* `AnyAxisym._evaluate` routed through the existing `_quad_apeak`, so Phi stops
  missing the `2*pi*Sigma(R)*|z|` midplane cusp for `|z| << R`. Before/after at
  R=0.5, against `2*pi*Sigma = 2.102952` and galpy's own
  `zforce(0.5,1e-8) = -2.102952`:

        Phi(z)-Phi(0)   before        after        expected
          z=1e-8      +4.436e-12   +1.876e-08   +2.103e-08
          z=1e-6      -9.637e-11   +2.103e-06   +2.103e-06

* `test_forceAsDeriv_potential` midplane probe one-sided -> CENTRED, at z==0
  only. The old assertion only passed because Phi was flat: two cancelling
  errors. With Phi fixed the one-sided FD reads -1.876 and it fails.

**NOT byte-identical on the numpy path** — deliberate, flagged to Jo first.
Gate: full `test_potential.py` 8 failed / 1225 passed / 102 skipped on BOTH this
branch and pristine main, identical failing sets (all environmental here: no C
extension, no pynbody).

**gh#1316 → `feat/backends-develop`** interpRZPotential grid guard, head
`033684e1`. Reds are 6 x `Set up job` (infra) + the known Windows
`test_streamgapdf_sample` flake + codecov. Re-runs FIRED 2026-08-17 once the
GitHub outage cleared.

**gh#1332 → `feat/backends-develop`** ledger prune. Reds were 4 x `Set up job`
+ codecov. Re-runs FIRED.

## Committed, not yet pushed

**`851510e3`** on `backend-jit/anyaxis-traced-phi` (off develop `091053a9`):
traced Phi's a=R log singularity at z=0, `_evaluate_gl` -> `transformed_quad`.

    R      before      after
    0.2    2.97e-05    1.86e-12
    0.5    4.18e-05    1.05e-12
    1.0    3.18e-05    2.54e-14

z>0 unchanged; numpy untouched (13 passed / 0 failed on `-k anyaxisym`).
**LEDGER DELTA: none** — 11 passed, 2 xfailed, both still `x` not `X`.

Held back only to keep PR count at two-into-develop. It is DISJOINT from
gh#1316 (AnyAxisym*.py vs interpRZPotential.py), so it can open as soon as a
slot frees.

## The cherry-pick backlog (found 2026-08-17, Jo asked)

**gh#1323 was never cherry-picked into the backend branches and it matters.**
Structural checks all said "present" (`galpy/util/quadpack.py` on all three
branches, matching symbol counts); the behavioural check disagreed:

    AnySpherical Rforce, whole-mesh vs cell-by-cell
      main                    0 of 77 cells differ
      feat/backends-develop  76 of 77 cells differ, max rel 9.4e-01

gh#1324 is milder — develop already accepts arrays via the backend migration
(verified by calling `__call__`/`Rforce` with arrays), `numpy.infty` clean on
both; only its tests are missing.

**Ordering:** gh#1323 also edits `interpRZPotential.py`, which gh#1316 touches,
and once it lands AnySpherical broadcasts correctly on develop — removing the
numpy fallback gh#1316's real-potential test relies on. So cherry-pick AFTER
gh#1316 merges, then re-check that test's premise.

## Remaining AnyAxisym `-jit` ledger entries: 4, none burnable yet

    jax-jit / torch-jit  test_poisson_surfdens_potential[AnyAxisym...]
    jax-jit / torch-jit  test_forceAsDeriv_potential[AnyAxisym...]

`forceAsDeriv` needs gh#1333's centred-FD change to reach the backend branch;
fixing traced Phi alone moves the FD from 8.3e3 to the PHYSICALLY CORRECT
-2.105 (= -2*pi*Sigma), which the old assertion still rejects.
`poisson_surfdens` is a separate, uninvestigated cause.

## Standing corrections to how I work

* **Never decrease coverage, not even one line** (Jo).
* **File presence is not behaviour.** For "did this fix land", run the
  measurement the fix changes, not a symbol grep.
* **A test whose predicate duplicates the guard it tests measures nothing.**
* **Different singularity, different rule.** xi^3 clustering fixes Phi's LOG
  (3e-5 -> 1e-12) and destroys zforce's POLE (1.7e-02 at z=1e-8 vs the
  ladder's 3.8e-10). Measure per-quantity before generalising.
* **Probe inside `jax.jit`.** Eager hands concrete scalars to scipy via
  `_bk_dispatch`, so an eager probe never enters the GL path — it returns a
  column of exact 0.000e+00 that looks like agreement.
* **Don't pipe a gate through `tail`.** It silently truncated a FAILED list to
  7 of 8 and nearly produced a false A/B mismatch.
* **`ps` on a shell pipeline measures the WRAPPER.** 0.0% CPU there meant
  nothing; the real pytest was at 103%.
* **Pre-commit rejects the first commit** when ruff reformats. Verify the SHA
  moved.

## Cron

`351c0c5a`, hourly at :13, session-only, created 2026-08-13 → **expires
~2026-08-20**. Verified alive this wake.

---

## 2026-08-18 addendum — everything is queue-bound, nothing is broken

    gh#1334 -> develop   150 success, 0 fail. BLOCKED: its `all-clear` step
                         "Wait for all other checks to pass" has been polling
                         since 20:22:08Z, waiting on a `backend status report`
                         job queued behind gh#1332's backlog. GitHub's 6h job
                         cap expires it ~02:22Z. If it times out that is queue
                         STARVATION, not a defect -- re-run all-clear once
                         gh#1332 drains. Diff already audited clean.

    gh#1332 -> develop   running at last. Only red is codecov/project on
                         partial uploads (41% with 12 of 150 shards in), the
                         same preliminary artifact as before. Base now reads
                         b4a2190, so the develop-merge refresh took.

## Ready to push, waiting on gh#1334

`backend/anyspherical-array-mass`, two clean commits off b4a21907:

    8784f65d  AnySpherical: array input returned the value at r[0]
              (AnySphericalPotential.py + quadpack.py)   76/77 cells -> 0/77
    d28283d0  AnyAxisym: port main's numpy Phi cusp fix + centred midplane FD
              (AnyAxisym*.py + tests/test_potential.py)

**A third commit is owed and deliberately NOT written yet:** the -2 xfail
ledger delta for `test_forceAsDeriv_potential[AnyAxisym]` (jax-jit + torch-jit).
Both were MEASURED passing traced with gh#1334's fix applied locally, but the
removal cannot land until gh#1334 is on the base -- without it the test would
fail UN-ledgered and redden the PR. Order: gh#1334 merges -> rebase -> add the
ledger commit -> push.
