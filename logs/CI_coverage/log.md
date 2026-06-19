# CI + coverage drive for #891 (variational) and #892 (backends)

Branches pushed to github.com/jobovy/galpy; both opened as **ready-for-review draft→ready** PRs (user wants CI + review now, NOT merge; revert to draft after initial review).

## Branch / PR model (finalized with user)
- **#891** `feat/variational3d` → main: variational feature branch, **merges to main first**; sub-PRs (Pvar-pot.*, Pvar-rhs, Pvar-lyapunov, Pvar-dissip-*) target it.
- **#892** `feat/backends` → main: long-lived integration branch (foundation PR0+P1); parallel P2.x series + Tracks D/E/F target it; rebased on main after #891 merges; merged last.

## Coverage analysis (galpy/ is what `--cov` measures; tests/ is not)
- **#892** initial gap: the new `galpy/backend/*` tests weren't in any `build.yml` job → `galpy/backend/*` uncovered. **Fix (commit e4230172):** added a CI job running `test_backend_pilot.py`+`test_backend_conventions.py` with jax+torch+array-api-compat (new `REQUIRES_TORCH` flag + PyTorch-CPU install step; array-api-compat added to the JAX step), + setup.py `backend/jax/torch` extras. Simplified `_namespaces` (dropped the never-reached compat-numpy normalization + vendored fallback; require array-api-compat for non-numpy), pragma'd the defensive dep-absent guards, added resolver tests. **galpy/backend now 100% covered locally with jax+torch present** (`__init__` 100%, `_namespaces` 100%, `_resolver` 100%).
- `_optional_deps` torch/compat `except` branches: covered in CI aggregate (most jobs lack torch/compat → except runs; jax/torch job has them → try runs). No pragma needed.
- **#891**: new C lines = NULL-init (3, executed always) + SpiralArms 3D-Hessian wiring (5, executed when SpiralArms integrated in C — covered by test_SpiralArmsPotential / test_liouville_planar); the 3 new aggregators are `LCOV_EXCL` (dead until Pvar-rhs). Expect coverage to pass; confirming in CI.

## Gotchas found
- `pytest --cov galpy.backend` (narrow target) + jax/torch import triggered a numpy `_pocketfft` "cannot load module more than once" error; **`--cov galpy` (the real CI invocation) is fine** — not a CI blocker.
- CI uses `-W error::DeprecationWarning -W error::FutureWarning`; verified the 76 backend tests pass under those flags (no torch/jax warning-as-error issues).

## Status (update as CI resolves)
- e4230172 pushed to feat/backends; awaiting #892 CI (esp. the new backend job + codecov patch/project).
- #891 awaiting CI (inert ABI change; expect green).

---

# All-backend existing-suite CI + the xfail-ledger as the Pillar-1 metric (2026-06-18)

The original section above is about *coverage* of the new `galpy/backend/*` code under #891/#892.
This section covers the second, larger CI mechanism that landed afterward: re-running the **entire
existing** galpy suite under a *forced* jax/torch backend, kept green by a checked-in ledger, so that
the shrinking ledger *is* the migration's progress metric.

## Approach — the all-backend harness (#956, merged 2026-06-15 → feat/backends)

Pillar 1 = "the existing test suite passes under every backend at numpy tolerances." #956 builds the
machinery to drive that down without paying 3× CI on ordinary PRs:

- **`backend-suite` job** re-runs the existing suite (test_potential / test_orbit / test_actionAngle /
  df shard) under `--backend=jax` and `--backend=torch`, **sharded** `BACKEND ∈ {jax,torch} × TEST_FILES`
  so wall-clock ≈ one normal cycle. CPU-only. **Gated**
  `if: ref==feat/backends || schedule || workflow_dispatch` — runs per-merge on the integration branch,
  on the existing weekly cron, and on demand; **never on ordinary PRs** (no 3×-per-commit cost).
- **xfail-ledger** = `tests/backend_xfail.txt` with `strict=True`. Currently-failing tests are XFAIL so
  the job is green; a *fixed* test XPASS-fails the run, forcing the ledger to shrink. A regression in a
  passing test fails red. A shard goes **RED only when a NOT-in-ledger test fails** (a new failure or a
  newly-fixed XPASS); **feat/backends itself is GREEN**. `--backend=numpy` ignores the ledger entirely
  (verified byte-identical, zero xfail/xpass). There is also a companion `tests/backend_slow_skip.txt`
  (deferred-slow tests skipped in both modes, never regenerated).
- A `workflow_dispatch` with `regen=true` runs ledger-free and uploads a refreshed ledger artifact
  (this is how the seed gets completed / re-pruned).
- **Per-PR `backend-smoke`** (~3 min, both backends, ungated): 8 curated consistency tests, green today,
  catches gross regressions on every PR without bloating PR CI.
- **Step-summary burndown:** every run appends `Backend <jax|torch>: N passed, M xfail (ledger),
  K XPASS(fix-me)` to `$GITHUB_STEP_SUMMARY`.

Seed baseline at merge (representative slice, C ext built): jax 40 / torch 154 failing nodeids; the
header marks it SEED and the first scheduled/dispatch regen completes it. (The systemic torch≫jax gap
is scalar-input rejection — `xp.sin/sqrt(<python float>)` across scalar paths — later fixed by #960's
central coordinate coercion.)

## Results — milestone-1 ledger prune (#969, merged 2026-06-17 → feat/backends)

The ledger policy is to regen+review only at a **major milestone** (where it has permanently moved
significantly), not per-#892-integration. Milestone 1 was that regen prune. Built from the full
13 jax + 13 torch shard fragments of a `regen=true` run on `regen/milestone-1-ledger`
(GH Actions run 27621760108). Burndown (from the #969 PR table / commit subject):

| backend | xfail | slow_skip | **total ledger** |
|---|---|---|---|
| jax   | 296 → **253**  | 39 (held) | 335 → **292** (−43)  |
| torch | 1683 → **1484** | 1 (held)  | 1684 → **1485** (−199) |

(Verified against the #969 merge commit `18e1349`: the parent `tests/backend_xfail.txt` has
`grep -c '^torch '` = 1683, `'^jax '` = 296; post-#969 = 1484 / 253. The title's totals add the held
slow_skip entries: 1484+1=1485 torch, 253+39=292 jax.) The permanent move that justifies the prune:
#960 central coordinate coercion (the torch scalar-input fix is the bulk of the −199), plus the landed
P2.x potential migrations + Pspecial work. `backend_slow_skip.txt` is byte-identical (held).

**Carry-forward gotcha:** one jax shard (test_SpiralArmsPotential / test_potential / test_scf /
test_MultipoleExpansionPotential / test_snapshotpotential) hit the 75-min session timeout (ran 259/410),
so 3 prior ledger entries among the un-run 151 are carried forward; torch ran all 410 of that shard, no
torch carry-forward.

## Dead-ends / gotchas

- **Unpaginated artifact listing understated the ledger.** An earlier revision of #969 built the regen
  ledger from only the *first page* of the run's 53 artifacts (a 30-item default `gh api` page), which
  undercounted; rebuilt from the complete fragment set. (Lesson now in auto-memory.)
- **`grep -c '^torch '` counts xfail only, not the total ledger.** The grep over `backend_xfail.txt`
  excludes the held `backend_slow_skip.txt` entries — so the grep number is always
  `total − slow_skip` (torch −1, jax −39). Both the #969 title and the current baseline below are
  reported on the **xfail-only** (grep) basis to keep them comparable.

## The ledger as the migration-progress metric — figure g13

The shrinking ledger is the Pillar-1 metric, plotted as **g13 (`g13_ledger_burndown`)**: # tests still
in the ledger (still failing under jax / torch) vs time, reconstructed **exactly from the git history**
of `tests/backend_xfail.txt` + `tests/backend_slow_skip.txt` on feat/backends
(`measure/burndown_from_git.py`), with dashed milestone markers. It falls to 0 when the full suite
passes under both backends at numpy tolerances.

**State now (current feat/backends `tests/backend_xfail.txt`, xfail-only grep basis):
torch 1494, jax 254** — verified via
`git show origin/feat/backends:tests/backend_xfail.txt | grep -c '^torch '` (1494) / `'^jax '` (254);
slow_skip held at torch 1, jax 39 (so totals 1495 / 293). The small post-#969 *rise* from 1484/253 to
1494/254 is the **triage adds** from landing the Track-D orbit-integration PRs (#972 C-STM PR-B merged
2026-06-17; #973 in-backend diffrax/torchdiffeq PR-A merged 2026-06-18) — newly-exercised orbit paths
ledgered as they're driven under a backend; they burn down at the next milestone.

**TODO (Track G code task, not this log):** g13 needs a fresh data point for the post-#969/#972/#973
state (torch 1494 / jax 254). Because g13 is reconstructed from git history of the ledger files, the
data point should already be picked up by re-running `measure/burndown_from_git.py` + `run` — but the
figure has **not** been regenerated here (do not regenerate; that is a Track G task at the next
milestone review).

## 2026-06-19 — per-potential ledger regen (post-#977 parametrization)

#974 (orbit accessors) + #975 (potential convenience/functional) merged to feat/backends;
feat/backends then rebased onto post-#977 main (39 commits, conflict-free — feat/backends
doesn't touch test_potential.py, #977 only touches it). #977 parametrized the 12
test_potential loop-tests, so the coarse ledger entries (prefix-matching every potential)
were regenerated at full per-(potential × test) granularity via GALPY_BACKEND_XFAIL_REGEN
→ **PR #978** into feat/backends.

- test_potential: jax **19 → 5**, torch **135 → 618**. Ledger total: jax 298 → **284**,
  torch 1589 → **2072** (1887 → **2356**).
- torch +483 = granularity refinement (coarse prefix entries hid 618 real failures + 598
  passes), NOT a regression; jax −14 = was over-ledgered. Validated torch:
  598 pass / 618 xfail / 0 failed / 0 xpass. jax via #978 CI.
- Burndown discontinuity handling: see appendix E (pre-#977 rows = coarse undercount, kept
  labeled; per-potential is the true metric from here; only test_potential changes granularity).
- g13 TODO: plot pre-#977 coarse series dashed up to the 2026-06-19 marker, per-potential solid after.
