# Numbers that change with each round of galpy PRs

Run through this list after each round of PRs merges. Locations are given as
file plus a phrase to grep for, because line numbers drift. Values are those
in the paper as of 2026-10-30 (`feat/backends` 14a186d7d, after #1780).

## 1. Cut-off date
| Value | Where |
|---|---|
| "October 30" | `process.tex` "Between 2026 June 1 and October 30"; `acknowledgments.tex` "between 2026 June 1 and October 30" |
| "through the 30th" | `process.tex` "work, added 135 through the 30th" |
| "between June and October 2026", "over four months" | `introduction.tex`; `abstract.tex` (these change only if the work runs into November) |

## 2. PR counts
Run `python paper/ms/stats/pr_stats.py`. First classify every new PR into main by hand, adding it to `MAIN_FIXES`, `MAIN_INFRA` or `MAIN_FEATURES`; the script prints any `unclassified` PRs. New unrelated PRs go in `EXCLUDED`.

| Value | Where |
|---|---|
| 723 opened, 696 merged; 602 backend, 29 variational, 65 main (40 fixes, 16 features, 9 tests/CI) | `process.tex` §8.1, second paragraph |
| Median PR size: 182 changed lines (quartiles 74 and 372), additions + deletions of the merged project PRs | `process.tex` §8.1 "median size of 182 changed lines" (the only place) |
| 101 stack members | `process.tex` §8.1 "101 of the PRs are members of such stacks" |
| October: 135 | `process.tex` "work, added 135 through the 30th" (June 88/29, July 176, August 114 and September 112 are final) |
| "40 PRs" (main fixes) | `process.tex` §8.6 "led to the 40 PRs against the main branch"; `conclusion.tex` "which 40 PRs to the main branch fixed" |
| "in eight PRs" (island removal) | `process.tex` §8.5 "these to the backend, in eight PRs"; this changes only for island PRs |

## 3. Costs (from JB)
| Value | Where |
|---|---|
| \$13,100 / 16,500 / 15,800 / 10,200 / 8,600 (October in full, \$8,615.61, JB 2026-10-30); about \$59,000 attributable (each October change added to the total, as June, July and October count in full) | `process.tex` "the GitHub Actions bill" |
| subscription tiers (JB 2026-10-30: still on Max 20×, so "for the final two weeks the 20× tier again" stands) | `process.tex` "fixed-price subscription" |

## 4. Test counts at the tip
- Existing suite: `pytest --collect-only` on `tests/test_*.py` except `test_backend*`.
- Backend suite: `pytest --collect-only` on `tests/test_backend*.py`.

Run both on the tip with every merged PR (`CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu`, `--continue-on-collection-errors`).

| Value | Where |
|---|---|
| 4,462 existing tests (excluding 27 torus tests; 4,489 collected) | `abstract.tex`; `conclusion.tex` "now 4,462 tests"; `process.tex` §8.4 "from 2,462 to 4,462" and twice in the Fig. 24 caption |
| 27 torus tests = 23 in `test_actionAngleTorus.py` + 4 streamdf useTM | `conclusion.tex`; `process.tex` §8.4 (×3) and the caption. Also `TORUS_FILE`/`TORUS_USETM` in `figures/g13_ledger_burndown/make.py` |
| 18,807 backend tests in 100 files | `abstract.tex`; `conclusion.tex`; `process.tex` "now 18,807 tests in 100 files"; `design.tex` "18,807 tests in 100 files" |
| 7,045 parametrized cases added by the two sweeps (the jump in the bottom panel) | Fig. 24 caption. The sweeps are 2,791 + 4,249 = 7,040 cases, so account for the other 5 when recounting |
| Fig. 24 itself | `GALPY_REPO=... python measure/suite_from_git.py --incremental --tip`, then `python -m paper.figures._harness run g13_ledger_burndown` |

## 5. Derivative and value sweeps
Collect `tests/test_backend_param_sweep.py` and the value sweep at the tip.

| Value | Where |
|---|---|
| 761 derivatives in 199 configurations | `process.tex` §8.6 "761 derivatives in 199 configurations" |
| 2,791 cases (derivative sweep), 4,249 cases (value sweep) | `process.tex` §8.6 |

## 6. Coverage (Codecov API, branch heads)
`curl https://api.codecov.io/api/v2/github/jobovy/repos/galpy/branches/{main,feat%2Fbackends}`, reading `totals.lines` and `totals.misses`. The `feat%2Fbackends` branch URL returned 404 on 2026-10-30; `commits/<full sha>` works, once the tip's CI uploads are all in (`state` alone does not say so: compare with the PR head's Codecov comment).

| Value | Where |
|---|---|
| 14 of 56,950 (backend branch, 14a186d7d, 2026-10-30), 24 of 41,704 (main, 188d2cb81) | `process.tex` "leaves 14 of its 56,950" |
| 24 of 41,669, 99.94 %; 2,360 tests; 95,000 / 42,000 / 26,000 lines | `introduction.tex` "cover $99.94\,\%$". **Does not agree with §8.2's 24 of 41,704 for main**, and its 2,360 tests are fewer than the 2,462 that the backend branch started from (perhaps counted differently). Make the intro use today's main, counted the same way as §8.4, or say when it was measured. |

## 7. Island audit (Table 1 and §8.5)
A forced-torch run of the whole existing suite with the instrument. Use `stats/island_audit/run_audit.sh`, then `merge.py` and `classify.py`; this takes several hours on cores 36–47 (about 25 minutes with `run_audit_tag.sh`-style 16-way parallelism on 28 cores; orbit group 5 needs a 32 GB memory cap). Rerun only when a PR changes a hand-off. The two exception PRs do (Part F).

| Value | Where |
|---|---|
| The table rows (2,011 / 46.2 % … 4,353 / 100 %), "4,353 tests that ran" | `process.tex` `tab:islands` |
| 46 %, 78 %, the 22 % split (12.9 / 3.4 / 5.8 / 0.1 %), "one documented exception" (×3), 4 tests | `process.tex` §8.5 text |
| The description of the exception | `process.tex` "a stream that the tests build with \numpy\ before the backend is forced" |

## 7b. JB's input and token use (§8 Review, "Claude Code records every message")
Run `python paper/ms/stats/session_stats.py ~/.claude/projects/-home-bovy-Repos-galpy/e83d6165-0836-4adf-9f62-52251f942f9d.jsonl --until <cut-off>` (~1 min for the 2.1 GB transcript). The split between JB's own text and relayed material is heuristic, so the paper rounds and calls it an estimate. The Codex count (18 at the cut-off of 2026-10-30) is by hand: the messages mentioning Codex, minus those that only ask about a review.

| Value | Where |
|---|---|
| "2026 June 4 and October 30" (first message; cut-off), "141 of the 149 days" | `process.tex` "Between 2026 June 4 and October 30, JB sent" |
| about 2,500 messages (2,508), almost a quarter while the agent worked (574) | same paragraph |
| median 14 words; about 57,000 own words (56,555) = roughly 110 pages at 500 words/page; about 33,000 relayed (33,356) | same paragraph |
| about 114 million output tokens (75.7M main + 38.5M subagents, a third); about 44 billion input tokens (36.6B + 7.5B, 97% cache reads) | same paragraph |
| "18 reviews in all" (Codex) | `process.tex` "usually by pasting them into the conversation" |
| about \$31,000 API-equivalent (\$30,682: main \$22,830, subagents \$7,852); 61 % cache reads, 29 % cache writes, 9 % output. Prices in `PRICES` (pricing page, 2026-11-01; Internet Archive snapshots of it every ~5 days from 2026-06-01 to 11-01 show no price change for any model while in use: re-check with the CDX API `web.archive.org/cdx/search/cdx?url=platform.claude.com/docs/en/about-claude/pricing` if a new model or month is added) | `process.tex` "Had the agent's usage" |
| Fable on "six days in June and July" (June 9-13, July 1-2), "about 4 %" of the output tokens (4.86M of 114M; 5,682 of ~126,000 responses) | `process.tex` "Claude Fable did it on six days" (changes only if Fable codes again) |

## 8. Outside the paper
- The README log in `galpy-backends`: rows to merged dates.
- `review/response.md`: Part A recount notes for the round.
