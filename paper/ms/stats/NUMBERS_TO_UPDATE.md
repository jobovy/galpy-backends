# Numbers that change with each round of galpy PRs

Run through this list after each round of PRs merges. Locations are given as
file plus a phrase to grep for, because line numbers drift. Values are those
in the paper as of 2026-10-29.

## 1. Cut-off date
| Value | Where |
|---|---|
| "October 28" | `process.tex` "Between 2026 June 1 and October 28"; `acknowledgments.tex` "between 2026 June 1 and October 28" |
| "through the 28th" | `process.tex` "work, added 129 through the 28th" |
| "between June and October 2026", "over four months" | `introduction.tex`; `abstract.tex` (these change only if the work runs into November) |

## 2. PR counts
Run `python paper/ms/stats/pr_stats.py`. First classify every new PR into main by hand, adding it to `MAIN_FIXES`, `MAIN_INFRA` or `MAIN_FEATURES`; the script prints any `unclassified` PRs. New unrelated PRs go in `EXCLUDED`.

| Value | Where |
|---|---|
| 717 opened, 685 merged; 591 backend, 29 variational, 65 main (40 fixes, 16 features, 9 tests/CI) | `process.tex` §8.1, second paragraph |
| Median PR size: 183 changed lines (quartiles 74 and 371), additions + deletions of the merged project PRs | `process.tex` §8.1 "median size of 183 changed lines" (the only place) |
| 101 stack members | `process.tex` §8.1 "101 of the PRs are members of such stacks" |
| October: 129 | `process.tex` "work, added 129 through the 28th" (June 88/29, July 176, August 114 and September 112 are final) |
| "40 PRs" (main fixes) | `process.tex` §8.6 "led to the 40 PRs against the main branch"; `conclusion.tex` "which 40 PRs to the main branch fixed" |
| "in eight PRs" (island removal) | `process.tex` §8.5 "these to the backend, in eight PRs"; this changes only for island PRs |

## 3. Costs (from JB)
| Value | Where |
|---|---|
| \$13,100 / 16,500 / 15,800 / 10,200 / 5,800 "through the 18th"; about \$56,000 attributable | `process.tex` "the GitHub Actions bill" |
| subscription tiers | `process.tex` "fixed-price subscription" |

## 4. Test counts at the tip
- Existing suite: `pytest --collect-only` on `tests/test_*.py` except `test_backend*`.
- Backend suite: `pytest --collect-only` on `tests/test_backend*.py`.

Run both on the tip with every merged PR (`CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu`, `--continue-on-collection-errors`).

| Value | Where |
|---|---|
| 4,458 existing tests (excluding 27 torus tests) | `abstract.tex`; `conclusion.tex` "now 4,458 tests"; `process.tex` §8.4 "from 2,462 to 4,458" and twice in the Fig. 24 caption |
| 27 torus tests = 23 in `test_actionAngleTorus.py` + 4 streamdf useTM | `conclusion.tex`; `process.tex` §8.4 (×3) and the caption. Also `TORUS_FILE`/`TORUS_USETM` in `figures/g13_ledger_burndown/make.py` |
| 18,604 backend tests in 98 files | `abstract.tex`; `conclusion.tex`; `process.tex` "now 18,604 tests in 98 files"; `design.tex` "18,604 tests in 98 files" |
| 7,045 parametrized cases added by the two sweeps (the jump in the bottom panel) | Fig. 24 caption. The sweeps are 2,791 + 4,249 = 7,040 cases, so account for the other 5 when recounting |
| Fig. 24 itself | `GALPY_REPO=... python measure/suite_from_git.py --incremental --tip`, then `python -m paper.figures._harness run g13_ledger_burndown` |

## 5. Derivative and value sweeps
Collect `tests/test_backend_param_sweep.py` and the value sweep at the tip.

| Value | Where |
|---|---|
| 761 derivatives in 199 configurations | `process.tex` §8.6 "761 derivatives in 199 configurations" |
| 2,791 cases (derivative sweep), 4,249 cases (value sweep) | `process.tex` §8.6 |

## 6. Coverage (Codecov API, branch heads)
`curl https://api.codecov.io/api/v2/github/jobovy/repos/galpy/branches/{main,feat%2Fbackends}`, reading `totals.lines` and `totals.misses`.

| Value | Where |
|---|---|
| 14 of 56,701 (backend branch), 24 of 41,704 (main) | `process.tex` "leaves 14 of its 56,701" |
| 24 of 41,669, 99.94 %; 2,360 tests; 95,000 / 42,000 / 26,000 lines | `introduction.tex` "cover $99.94\,\%$". **Does not agree with §8.2's 24 of 41,704 for main**, and its 2,360 tests are fewer than the 2,462 that the backend branch started from (perhaps counted differently). Make the intro use today's main, counted the same way as §8.4, or say when it was measured. |

## 7. Island audit (Table 1 and §8.5)
A forced-torch run of the whole existing suite with the instrument. Use `stats/island_audit/run_audit.sh`, then `merge.py` and `classify.py`; this takes several hours on cores 36–47. Rerun only when a PR changes a hand-off. The two exception PRs do (Part F).

| Value | Where |
|---|---|
| The table rows (1,902 / 45.5 % … 4,181 / 100 %), "4,181 tests that ran" | `process.tex` `tab:islands` |
| 45 %, 77 %, the 23 % split (12.8 / 3.5 / 5.6 / 1.6 %), "three documented exceptions" (×3), 65 tests | `process.tex` §8.5 text |
| The description of each exception | `process.tex` "disk of arbitrary surface density … DF of an evolving disk …" |

## 8. Outside the paper
- The README log in `galpy-backends`: rows to merged dates.
- `review/response.md`: Part A recount notes for the round.
