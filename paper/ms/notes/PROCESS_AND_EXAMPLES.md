# Notes for the paper: development process, AI use, and the showcase examples

Compiled 2026-10-06 from the GitHub API (`gh`, repo `jobovy/galpy`, PRs created
2026-05-01..2026-10-06, all paginated), the galpy and galpy-jax git histories, and the
galpy-jax working documents (`README.md`, `plan/*.md`, `logs/vacation_2026-07/`,
`paper/figures/*/caption.md`, `make.py` docstrings, `_harness/manifest.json`,
`logs/figure_rerun_2026-10-06/log.md`). Every number below is reproducible from the
cited source; dates are absolute (UTC). "Claude" = Anthropic's Claude Code agent;
"Jo" = Jo Bovy.

---

# Part A -- the process

## A.1 Timeline and volume

**Start.** The galpy-jax working repository's initial commits are dated 2026-06-03
("Initial commit"; "Add initial claude convo about autodiff backend implementations")
and 2026-06-04 ("Initialize multi-backend project log: overview, per-track plans, PR0
log") (`git log --reverse` in `/home/bovy/Repos/galpy-jax`). The design discussion that
seeded the project is `galpy_backend_autodiff_discussion.md`; the master plan is
`master_plan.md` (its scope section is headed "Scope = ALL of galpy, maximal on every
axis (user-confirmed)", `master_plan.md:11`). On the galpy side the first backend
commits are all dated 2026-06-04: `5e66bc033` "PR0: galpy.backend resolver pilot
(Plummer + Isochrone)", `26cebf4c3` "P1: productionize galpy.backend (config, force
mode, --backend, convention guard)", squashed into `ff11a9674` "Multi-backend (JAX /
PyTorch / array-API) support for galpy", the first commit of `feat/backends`
(`git log --reverse origin/main..feat/backends`). The pilot's own log is
`logs/PR0_pilot/log.md` (70 tests: value parity numpy/jax/torch at rtol 1e-12,
grad-vs-finite-difference, the identities dPhi/dR = -F_R and d(-F_R)/dR = R2deriv under
AD, jax jit+vmap, torch dtype preservation).

**First PRs.** The integration PR **#892** "Multi-backend (JAX / PyTorch / array-API)
support [integration branch]" was opened 2026-06-04 (base `main`; still OPEN on
2026-10-06; it now carries +105,207/-13,851 lines, i.e. the whole backend delta vs
main). The first feature PRs into `feat/backends` were opened the same day: #893
"P2.1 Spherical potentials: backend-agnostic namespace-swap", #894 (analytic disks),
#895 (ellipsoidal/triaxial), #896 (halo/bar/non-axisymmetric), #897 (dissipative
scaffold). The first merges into `feat/backends` were on 2026-06-06 (#903 differentiable
potential parameters, #893, #894; `git log`). The parallel C-side branch
`feat/variational3d` (3D variational equations + Lyapunov, 28 PRs) was opened as #891 on
2026-06-04 and merged to main on 2026-06-13 (galpy-jax `README.md:15`).

**PR counts (created 2026-06-01..2026-10-06, `gh pr list --state all --limit 2000`):**

| | PRs | merged | closed unmerged | open |
|---|---|---|---|---|
| all PRs in the repo | 710 | 656 | 39 | 15+ |
| not into `main` (backend/variational work) | 589 | 546 | 28 | 15 |
| of which base `feat/backends` | 325 | | | |
| of which base `feat/backends-develop` (autonomous stretch) | 145 | 143 | | |
| of which base = another working branch (stacked sub-PRs) | 119 | | | |
| of which base `feat/variational3d` | 28 | | | |
| into `main` | 121 | 110 | | |
| into `main`, excluding bots (dependabot/pre-commit/doc-notebook) | ~100 | | | |

Non-main PR authors: `jobovy-agent` 323, `jobovy` 266. NOTE: the `jobovy-agent`
GitHub account first appears on 2026-07-26 (#1205); before that Claude's PRs were opened
under Jo's own account (every non-main PR after 2026-07-27 but one, #1471, is by
`jobovy-agent`), so **GitHub author counts do not separate human from agent authorship
before 2026-07-26**. Commit authorship across all branches since 2026-06-01 is "Jo Bovy"
3614, "Jo Bovy's agent" 2106, "jobovy-agent" 972, "Claude" 10 (`git log --all`; includes
rebased duplicates). The honest statement is the one in the task description: Jo
directed, Claude implemented; essentially all code was written by Claude under Jo's
direction and review.

**Merging.** `mergedBy` over the 635 merged human-authored PRs: Jo 492, `jobovy-agent`
143. By base: `feat/backends` Jo 309 / agent 1; `feat/backends-develop` agent 141 / Jo
2; `main` Jo 89; stacked sub-PRs Jo 92 / agent 1. I.e. Claude merged only into the
develop branch during the autonomous stretch; Jo merged everything into
`feat/backends` and `main`.

**Monthly cadence (merged non-main PRs):** 2026-06: 111, 2026-07: 167, 2026-08: 127,
2026-09: 110, 2026-10 (6 days): 31. Merged date range 2026-06-05..2026-10-07.

**PR size (merged non-main PRs, additions+deletions):** median 185 lines (quartiles
74 / 371), mean 341, max 19,353; median additions 150.5, median deletions 18.5. Total
+154,163 / -31,796 over the 546 merged PRs (double-counts the develop->feat/backends
umbrella #1204, +15,928/-3,425).

**Splitting/stacking.** 119 PRs have a working branch (not an integration branch) as
base, i.e. are members of real GitHub stacks (`gh stack`; memory rule
`two-prs-and-larger-prs`). 12 PR titles carry an explicit "(k/n)" or "part k" (e.g.
#1604-#1608 "GPU numpy islands (1/5)..(5/5)", #1627/#1628/#1630
"actionAngleVerticalInverse (1/3)..(3/3)"). Splits by family are recorded in the README
table, e.g. Track F Pdf.2 "split by family 2026-07-02: base #1052 + #1053 isotropic +
#1054 constantbeta + #1055 osipkovmerritt + #1056 eddington+king" (`README.md:191`).
Branch state: `feat/backends` is 463 commits ahead of `origin/main` (merge-base
`d243f989b`, 2026-09-11; rebase-only, no merge commits).

## A.2 PR discipline

**Written rules** (standing constraints, `logs/vacation_2026-07/STATUS.md:71-84`):

> * Jo merges the main PRs. I can merge develop, not main.
> * I **cannot** edit `.github/workflows/**` -- provide patches; Jo applies.
> * Never name a working branch `feat/...` -- use `backend/...`.
> * numpy must stay byte-identical (flag any exception: the King tolerance is one,
>   approved); coverage never decreases, not one line.
> * Rebase only, no merge commits. Real `gh stack`s for dependent PRs. Cancel the
>   PRIOR sha's CI on push.

The master plan's tolerance policy (`master_plan.md`, "Tolerance policy (hard
requirement)"): "**numpy path is byte-identical** -- guaranteed ... no numpy-path test
or tolerance changes. **jax/torch must pass the *same* tests at the *same*
tolerances.** ... A tolerance may be loosened only if absolutely necessary and then
only slightly, per-test, with a one-line justification logged". Coverage: "held at
**all-but-27-lines**" (goal 5). The autonomous-stretch ground rules
(`logs/vacation_2026-07/log.md:14-25`) add: one PR at a time (relaxed by Jo on
2026-08-02 to two, `log.md:19884`); "Meaningful assertions only -- no
`isfinite`/nonzero smoke checks; tolerances near the numerical limit"; every PR
self-audited for duplication/placement ("general helpers belong in utilities").

**Review.** Review happened mostly outside GitHub's formal review UI. Formal reviews
exist on 24 PRs (Jo: 23 APPROVED, 10 CHANGES_REQUESTED, 3 COMMENTED; 51 inline review
threads on 8 PRs, 52 inline comments by Jo; 2 Copilot auto-reviews) and Jo left 47
issue-comments on PRs. Jo's review style is visible in the inline threads, e.g. on
#1545 (actionAngleSphericalInverse): "Is this a hand-rolled root finder? If so, might
need to go in a util file or be replaced by scipy", "Nothing to be gained by
vectorizing? For speed-ups in general, do A/B timing tests", "one should avoid testing
internal functions directly ... use public methods" (#1415); on #1404: a C hygiene
finding ("`midchi` ... omitted from the `private(...)` clauses ... a shared write/data
race under OpenMP"; "The math looks right to me"). The bulk of the review was
conversational (Jo's directions are logged as dated entries in `log.md`: "Jo: backend
precision must MATCH numpy/scipy precision" 2026-08-11 `log.md:41168`; "Jo: never
decrease coverage, even by one line" 2026-08-16 `log.md:44937`; "Jo: 'no numpy islands;
tests must meaningfully run on the backend under a force'" 2026-08-22 `log.md:51042`;
"Jo caught a coverage regression I merged in gh#1402" 2026-08-27 `log.md:56568`).

**Codex reviews.** Jo ran OpenAI Codex code reviews over Claude's work at several
points and posted the findings as PR comments (the Codex text appears under Jo's
account; 8 PRs reference "codex" in their bodies: #966, #1381, #1382, #1406, #1417,
#1510, #1596, #1597; `log.md` has 12 mentions). Documented rounds:

* 2026-06-16, #966 "backend.special fallback edge hardening ... prompted by a Codex
  review of the module. Each finding was adversarially re-verified numerically vs
  scipy ... before fixing; refuted/unreachable items were dropped" (PR body). Finding
  fixed: "`assoc_legendre` -- the m=0 x-derivatives divided by (x^2-1), so at the
  symmetry-axis poles x=cos(theta)=+-1 they returned NaN. SCF/Multipole forces and
  2nd-derivatives on the z-axis (R=0) were NaN under jax/torch".
* 2026-08-24..27, the review of the autonomous-stretch umbrella **#1204** (5 findings
  in the first pass, then 10, then follow-ups; "all five reproduced", `log.md:52781`).
  Representative findings (verbatim from the #1204 comments):
  1. "`galpy/backend/special/_fallback/gammainc.py:196` The new torch autograd path
     returns `nan` for `d/dx gammainc(a, x)` ... at `x=0`. Values are correct, but
     gradients are not ... the derivative should be finite (`+/-1` at `a=1`, zero for
     `a>1`). The endpoint value tests exist, but endpoint gradient tests are missing."
     -> fixed in #1381.
  2. "`galpy/backend/quadrature.py:622` `finite_part_quad` is documented as accepting
     array `b`, `c`, and `peak_width`, but the residual-log slope estimate is
     scalar-only ... the reduction lacks `axis=-1`" -> fixed in #1382 (which also found
     "a bug the byte-identity A/B turned up on its own": torch tensors have a `.to()`
     method, so AnyAxisym's units duck-typing raised TypeError under torch).
  3. "`galpy/potential/Potential.py:128, :817, :914` Generic surface-density
     quadrature gates backend quadrature only on `under_trace(absz)`. That misses
     differentiation ... with respect to `R` when `z` is fixed. I reproduced this with
     JAX: `grad(lambda R: p.surfdens(R, fixed_z, forcepoisson=True))` falls into SciPy
     and raises a concretization error".
  4. Follow-up pass: "`galpy/backend/special/_fallback/hyp2f1.py:212` - the traced
     Euler-label selector still uses the old `c - B >= 1` rule ... Concrete/eager is
     correct; traced/JIT returns `nan` or `inf`. Minimal repro: `jax.jit(lambda a:
     gsp.hyp2f1(a, -0.2, 2.5, jnp.asarray(-0.7)))(jnp.asarray(2.0))` Expected/scipy:
     `1.09213476189247`, Current JAX traced result: `inf`".
  The reviewer's overall verdict on #1204: "the work is coherent, the ledger is honest,
  and I did not find broad duplication against `galpy.backend`".
* 2026-08-31, #1417: "from a codex review of `d8098c7e`. `test_mass_batches_...`
  asserted `rtol=0, atol=0` between the batched and node-by-node quadrature drives ...
  That is not a galpy contract" (over-tight bit-identity assertion across XLA
  fusion; valid).
* 2026-09-30..10-01, #1596 (main) / #1597 (feat/backends), TwoPower forces via the
  incomplete beta: seven Codex rounds ("Codex 1-7"); e.g. round 6 found "beta << 0
  lower series 2F1(1,b;c;z) alternates and cancels (C 12.9% off, wrong sign at -100)
  -> Euler's transformation" (`log.md:60717`).

**CI.** Five test modes gate every backend PR: numpy (`build.yml`, ~20 sharded jobs,
plus `tests/test_backend*.py` run under jax and torch in 5 split groups,
`build.yml:49-85`), and the "All-backend tests" workflow (`.github/workflows/
backend-tests.yml`) which runs galpy's *existing* suite with `--backend={jax,torch}` x
`MODE: [eager, jit]` (`backend-tests.yml:135,147`; `--jit` = "jax.jit / torch.compile
at every galpy entry point", `backend-tests.yml:67`), sharded, 180-min shard timeout,
with a per-file x backend status table posted on the PR. History: the all-backend
existing-suite job + xfail ledger landed as #956 on 2026-06-14; the suite-wide `--jit`
trace mode as #1202 (2026-07-25); the jit dimension became a per-PR matrix axis with
#1536 on 2026-09-21 (before that it ran only by `workflow_dispatch(jit=true)` and was
not enforced -- `README.md:259-266`; the agent could not add it because its token lacks
workflow scope and prepared the patch `logs/vacation_2026-07/0001-CI-run-the-
backend-suite-TRACED-...patch` for Jo). Codecov (`codecov.yml` ignores only the vendored
Torus and xsf code) gates patch and project coverage; the rule "coverage never
decreases, not one line" is Jo's (2026-08-16) and was enforced by reading the
hits/misses table, not the check colour (`log.md:56568`: "Both codecov checks on
gh#1402 PASSED ... Misses 15 -> 16 (+1). I read the check status and not the table").
Also `all-clear.yml` (aggregate status for merge) and, since #1471 (2026-09-10), the
backend suite runs once per commit instead of twice.

**Forced-backend gate before push.** A local rule (memory `forced-backend-gate-before-
push`): the touched test suites are run locally under `--backend torch` and `--backend
jax` (and the numpy byte-identity A/B) before any push; e.g. `log.md:11371` "#192
forced-backend gate (the check numpy runs cannot make)", `log.md:41930` "forced-backend
gate on the rebased develop: 1 inherited red". Numpy byte-identity is checked by
sha256 of outputs on an A/B pair of worktrees that differ only in the PR's files (e.g.
#1382: "32 AnyAxisym R2deriv/z2deriv values, bit-for-bit equal on numpy, jax and torch
(float64), arms differing only in `quadrature.py`"; #1192 "numpy byte-identity:
measured, not argued", `log.md:11427`).

## A.3 The autonomous stretch

Source: `logs/vacation_2026-07/log.md` (60,759 lines, 2,037 dated section headings),
`STATUS.md` (overwritten snapshot), `STATUS.md.prev`.

* **Dates.** Header: "Autonomous stretch, 2026-07-26 -> ~2026-08-09. Jo away ~2 weeks.
  Goal: fully jittable/compilable galpy -- the whole existing suite passing under jax
  and torch, the xfail ledger emptied, and no numpy islands" (`log.md:1-5`). Jo's
  remote directions resumed ~2026-08-02 ("two PRs at a time") and 2026-08-10 (direct
  questions in the log); the deferred review model continued until Jo reviewed the
  umbrella PR #1204 on 2026-08-24 (Codex + own review, follow-up PRs #1381-#1410), and
  "AUTONOMOUS MODE ENDED (Jo's instruction)" on 2026-09-01 (`log.md:60068`): "merging
  feat/backends-develop -> feat/backends, then switching to supervised stacked PRs that
  Jo merges". #1204 "Autonomous jit/compile development" (base feat/backends, head
  feat/backends-develop) was created 2026-07-26 and merged 2026-09-02.
* **Mechanics.** Jo created `feat/backends-develop`; Claude opened PRs there and
  squash-merged them itself after CI was green ("Review model: Deferred: Jo reviews
  the accumulated branch on return", `log.md:18`). The agent account had write but not
  owner rights; the `feat/backends*` ruleset made those refs owner-only; workflow files
  were off-limits (`log.md:110-118`). A cron-driven "loop" woke the agent repeatedly
  (entries are headed "Wake N"); the loop died twice and was re-created
  (`log.md:41059,41872`). Jo merged all PRs targeting `main` (e.g. #1274, #1293-#1296).
* **Volume.** 145 PRs were opened against `feat/backends-develop` (2026-07-25..
  2026-09-02), 143 merged; 141 of them merged by `jobovy-agent`. In the two unsupervised
  weeks 2026-07-26..2026-08-09: 74 merged PRs (+8,237/-2,510 lines). 2026-07-26..
  2026-08-24: 122 merged (119 agent-authored). Scorecard written into #1204's
  description: "ledger 392 -> 140 against a target of 0, so NOT met" (`log.md:52776`).
  The agent also opened 19 PRs against `main` in that window (numpy/C fixes, see A.5).
* **What was done.** The jit (`--jit`) ledger was burned down from 100 (2026-07-26) to
  ~15 within three days (#1206 100->49, #1208 49->28, #1212 28->15); the slow-skip
  ledger 210 -> 92 in one day (2026-08-07/08, #1267 -> #1273); the `@backend_input`
  coercion-crossing cost was found and fixed ("A `@backend_input` leaf costs ~178 us
  per call under a forced backend ... diskdf.oortA 133,332 crossings -> 0",
  `log.md:27064-27100`); the `--jit` dimension got its own measured slow-skip list
  (#1279/#1281); units/Quantity under `--jit` resolved from 33 ledger entries to 2
  (`log.md:42220,42336`); AnyAxisymmetricRazorThinDisk quadrature and finite-part
  second derivatives (#1245 main, #1278, #1307); two pre-existing numpy bugs filed as
  issues #1276/#1277 after validation against a 30-digit mpmath Hankel reference
  (`README.md:338-354`).
* **What went wrong, and how it was handled** (all self-reported in the log):
  - #1201 (jax trace fixes) broke `test_energy_jacobi_conservation[RazorThin...]` on
    plain numpy; the agent first called it a regression, reversed on grid-identical
    values, and a controlled one-file-swap experiment showed the first call was right
    ("grid agreement is not proof of equivalence", `log.md:60-105`).
  - #1212 introduced an all-NaN traced `jeans.sigmalos` (found by the agent's own
    traced sweep two days later, `log.md:2909`); #1261 introduced a traced
    `test_potential` regression and a diskdf Oort-A slowdown "found by a tool I built
    for exactly that purpose, then closed within twelve hours" (#1268, `log.md:27064`;
    `README.md:288`); #1300 caused a 180x perf regression (1.7 s -> 300 s timeouts),
    found by comparing junit durations, fixed before merge (`log.md:42285`).
  - #1245 was returned to draft when "my fix was wrong, and the bug has layers"
    (`log.md:10581`); the final fix merged to main 2026-08-03.
  - Process slips: stale rule read ("One PR at a time" after Jo had relaxed it,
    `log.md:19884`); conflict markers left in the main working tree (`log.md:10943`);
    a `git stash` in a forbidden worktree (`log.md:60707`); the README's PR table went
    stale for 3 days/10 PRs and later for 89 PRs ("rebuilt from the GitHub API",
    `README.md` 2026-09-16 status note); a merged coverage drop of one line in #1402
    caught by Jo (`log.md:56568`; the uncovered line was dead code and was deleted).
  - Standing "recurring traps" are recorded in `STATUS.md:48-67` (anomaly-mode false
    NaNs, double-counted gradients through detached tables, frozen interpolation
    knots, FD on noisy adaptive-quadrature paths, 300 s timeouts under concurrent
    local suites).

## A.4 The xfail-ledger burndown (figure g13)

**Mechanism.** The all-backend CI keeps the *existing* galpy suite green under
`--backend jax/torch` via checked-in ledgers: `tests/backend_xfail.txt` (expected
failures, `xfail(strict=False)`), `tests/backend_slow_skip.txt` (tests that pass but
exceed the 300 s cap under a backend), `tests/backend_skip.txt` (permanent skips:
Torus, by-design floors), each line "<mode> <nodeid>" with mode in {jax, torch,
jax-jit, torch-jit} (`backend-tests.yml:4-8`; `g13/measure/suite_from_git.py`). A
`workflow_dispatch(regen=true)` run regenerates the ledger from reality; the ledger
*size* is the progress metric ("Pillar 1"), and milestone "prune" PRs (e.g. #969
2026-06-16 "torch 1684->1485, jax 335->292"; #996 2026-06-22 "-1017"; #1077 2026-07-06
"jax -146, torch -158"; #1188 2026-07-23 "torch 177->99, jax 77->54") moved tests
xfail->pass. "Numpy islands" -- code paths that silently drop to numpy under a forced
backend so a test passes without exercising the backend -- were ruled out by Jo
("no numpy islands; tests must meaningfully run on the backend under a force",
2026-08-22) and hunted with a CUDA run (memory `cuda-run-is-numpy-island-detector`;
PRs #1604-#1608 "GPU numpy islands (1/5)..(5/5)"). Whole-file ledger entries were
split so the count reflects tests, not lines (#1363 "split the streamTrack whole-file
defer -- 43 tests back into jax CI"; #978 per-potential granularity).

**Measurement.** `g13/measure/suite_from_git.py` collects the suite with pytest at
every ledger-touching commit's original PR head (`refs/pull/N/head`) and expands each
ledger entry to the tests it matches (whole-file defers counted at their real size:
"the raw entry count understated jax's early ledger by ~700", `g13/make.py`
docstring). Series: **151 ledger-touching commits** (2026-06-14..2026-10-05) in
`data/suite.json`; 146 in `data/burndown.json` (raw entries); 7 milestones in
`data/milestones.json`: 2026-06-14 ledger baseline; 2026-06-23 #1000 Orbit via the
C-integrator helper; 2026-07-19 #1161 float64 conftest fix + regen; 2026-07-25 #1196
`@backend_input` coercion boundary; 2026-07-26 `--jit` whole-suite mode; 2026-09-12
#1492 eager ledgers ~0; 2026-10-05 #1627-#1630 actionAngleVerticalInverse ported.

**Numbers** (`data/suite.json`, tests still ledgered = xfail + slow-skip; "skipped" =
permanent skips):

| mode | peak (date) | 2026-06-30 | 2026-07-26 | 2026-08-09 | 2026-08-31 | 2026-09-29 | now (2026-10-05) |
|---|---|---|---|---|---|---|---|
| jax (eager) | 1042 (2026-06-14) | 523 | 390 | 202 | 27 | 1 | **0** (+8 permanent skips) |
| torch (eager) | 2080 (2026-06-21) | 930 | 113 | 4 | 5 | 1 | **1** (+4 skips) |
| jax-jit | 100 (2026-07-26, mode introduced) | - | 100 | 39 | 22 | 4 | **3** (+4 skips) |
| torch-jit | 146 (2026-09-11) | - | 0 | 14 | 6 | 2 | **2** (+0 skips) |

Anchor: 2466 tests in the full pre-migration suite on 2026-06-04 (`data/anchor.json`;
an upper bound, since backend-inert tests never failed -- `caption.md`). The existing
suite grew from 2489 (2026-06-14) to **4275** tests (2026-10-05); the dedicated backend
test modules grew from 2545 tests to **8585 tests in 88 files**. Raw ledger entries
peaked at 2419 on 2026-06-22 (torch 2080, jax 339) and are now 1 eager + 5 jit. The
early torch >> jax gap "is the systemic torch scalar-rejection fixed by the central
coordinate-coercion work" (#960, 2026-06-16; `caption.md`). Current ledger contents
(`tests/backend_xfail.txt`, `backend_slow_skip.txt`, 2026-10-06): 3 xfails (all
`--jit` pickling of unit wrappers), 3 slow-skips (FDM `central_limit` under torch /
torch-jit; ExpTruncNFW `analytic_ecc_rperi_rap` under jax-jit, plus qdf
`sampleV_interpolate` entries), 16 permanent skips (`README.md:21-27`).

## A.5 Fixes to galpy `main` found through the backend work

Merged into `main`, created 2026-06-01..2026-10-06 (bots excluded; `gh pr list --base
main`). Discovery route in brackets where documented.

Variational/C second-derivative fixes (Track B, June): #899 "Fix incorrect SpiralArms C
R2deriv/PlanarR2deriv used by integrate_dxdv" (2026-06-05); #902 "Fix incorrect power of
R in TwoPowerSpherical C PlanarR2deriv"; #912 "Fix incorrect coefficient in
FlattenedPowerPotential C PlanarR2deriv"; #921 "SCFPotential: analytic second
derivatives (replace numerical mixin)"; #923 "Fix sign of SCFPotential C planar 2nd
derivatives" [FD-of-flow test; "det(M)=1 cannot catch sign errors -- standing rule",
`README.md:132`]; #925 DiskSCF planar 2nd derivatives in C; #943 "KuijkenDubinski layer
drops t for the embedded expansion (time-dependent DiskMultipole frozen at t=0)"; #947
list-of-Potential input to wrappers; #948 wrapper repr eats first character; #953 "Fix
silent-skip in _parse_pot: planarized potentials without planar C support segfaulted
the C integrator"; #954 RotateAndTilt C: include t in the force-cache key; #924
interpolation-range warning for multiple orbits; #980 deepcopy of
MultipoleExpansionPotential under scipy>=1.18; #994 "Fix numpy 2.5.0 compatibility
(eig/eigvals complex + chararray deprecation)" (2026-06-21); #1010 actionAngleStaeckel
pure-Python frequencies and angles (c=False, previously C-only); #1032 sphericaldf
sampling crash for slightly-negative isotropic DFs; #1086 symplectic C integrators
variational (dxdv) integration; #1091 1D variational integration.

Numpy/C defects found by backend validation (July-October; author `jobovy-agent`):
#1245 "AnyAxisymmetricRazorThinDisk: fix the small-|z| force/2nd-derivative band"
(2026-08-03) [pre-existing nan band stepped on by the traced quadrature]; #1274
"interpRZPotential: keep grids and off-grid fallbacks in internal units (pre-existing
numpy bug)"; #1293 "Two wrong tolerances: surfdens quadrature (epsabs) and the
streamgapdf sample distance" [issues #1289/#1290: scipy `quad` default
`epsabs=1.49e-8` truncating small integrals]; #1295 RazorThinExponentialDisk k=R
endpoint singularity [issue #1276]; #1296 AnyAxisym second derivatives as finite-part
integrals [issue #1277]; #1323/#1324 array input for AnySpherical / SphericalShell /
HomogeneousSphere; #1333 "AnyAxisym: Phi misses the midplane cusp for |z| << R (and the
test that hid it)"; #1349 sphericaldf per-call ro=/vo= (issue #1198); #1355
actionAngle masking an inner AttributeError; **#1356 "actionAngleAdiabatic: regularize
the action integrands' sqrt endpoints (fixes #1354)"** [issue #1354: "plain
Gauss-Legendre on sqrt-endpoint integrands loses ~4 digits in jr/jz", found when the
backend GL path was compared to C]; #1357 actionAngleStaeckel sqrt turning points (C +
Python); #1398 Adiabatic multi-point `_justjr` TypeError; #1400 "integrate_dxdv:
transform the base state in C, restoring bit-identity with plain integrate"; #1404
Staeckel chi-anomaly quadrature at turning points in C; #1424/#1431 Staeckel C
frequencies/angles near turning points, NaN and near-axis accuracy; #1405
IsochroneInverse Kepler solve vectorised; #1418 time-dependent SCF tests 76 s -> 11 s;
#1422 ChandrasekharDynamicalFrictionForce construction ~100x faster; #1439 "jeans: don't
let scipy.quad's default epsabs truncate sigma_r's small outer tail"; #1458 "Do not fork
a worker per task when numcores=1 (Orbit.lyapunov 94x faster)"; #1468 OblateStaeckel
tabulated C evaluation (10-16x); #1475 actionAngle_c one potentialArg copy per OpenMP
thread; #1484 "ChandrasekharDynamicalFrictionForce: the force cache has never hit (2.1x
on the friction path)"; #1519 linearPotential x2deriv; #1527/#1553 actionAngleSpherical
circular orbits and boundedness at infinity; #1560 Staeckel focal length from the
potential; **#1568 "Fix spherical DF sampling velocity bias"** [issue #1343, found by
figure gD2: sampled sigma_r biased 0.6-0.8% low by the n_v_vesc=100 inverse-CDF grid];
#1585 "potentials: machine-precision small r (NFW, Burkert, TwoPower), finite edge
limits, Staeckel midplane vmin"; **#1596 "TwoPowerSphericalPotential: forces, second
derivatives and mass from the incomplete beta (accurate near beta=3, finite at large
r)"** [up to 1e-3 off near beta=3, NaN at large r; validated vs 50-digit mpmath to
<=3e-14]; #1611 "Orbit: setting up from sky coordinates with uvw=True no longer
modifies the input"; **#1633 "Dynamical friction: sigma_r on a log-r grid (was 20-60%
off at small r); Python evaluates C's natural spline"** (2026-10-06) [found by
re-measuring figure g11: C and Python interpolated sigma_r with different splines,
2.6e-3 apart at r=0.05]; #1638 "Pyodide build: fetch GSL from independent mirrors,
SHA-256 verified (GNU hosts down)" (+ Jo's #1409 GSL mirror). Infrastructure by Jo:
#979 all-clear aggregate check, #1044 sdist build, #1470 no duplicate builds, #1583
Python 3.15 wheels, #1590 Pyodide wheels for 3.14, #1409.

Also: actionAngleVerticalInverse / SphericalInverse work on main by the agent
(#1309, #1320, #1342, #1350, #1351, #1415, #1510, #1516, #1545 open), the new
`actionAngleStaeckel` C-native differentiable actions (#1051, feat/backends), and the
contributed #976 ExpTruncNFWPotential (sgpfaff) and #1501 Einasto n=6 (shaneraphel).

Issues opened 2026-06-01..2026-10-06 (7, six by the agent): #1198 (sphericaldf ro=/vo=,
Jo), #1276, #1277, #1289, #1290, #1343, #1354 -- all closed. Policy (Jo, 2026-08-08):
pre-existing numpy defects found by the backend work are filed as issues and fixed in
PRs to `main`, never quietly absorbed into `feat/backends` (`README.md:338-343`; memory
`numpy-fixes-go-to-main`; "Removing an exception hides the count").

## A.6 How correctness was established

1. **numpy byte-identity.** The cardinal rule (A.2). Verified per PR by sha256 /
   IEEE-754 byte comparison of outputs on A/B worktrees (e.g. #946 "18,822-value
   byte-parity", #949 "92+324-record byte proofs", `README.md:134-136`). Deviations
   are flagged, never shipped quietly; the one approved exception is noted
   (King tolerance, `STATUS.md:79`). The 2026-06-05 overnight adversarial review found
   that CI's rtol=1e-12 parity tests were "~4 orders too loose to catch the ~1e-16
   drifts" from re-associated arithmetic in #893/#895/#896 and set the reference
   pattern: "keep the original expression byte-for-byte, wrap only the branch in
   `xp.where`, never re-derive an algebraically-equal form"
   (`logs/overnight_review_2026-06-05/report.md`).
2. **Gradient-vs-finite-difference**, with h-convergence rather than a single
   tolerance ("grad-vs-FD or reference-match, NEVER finite-and-nonzero"; FD must
   REBUILD the object from the parameter, `README.md:59`). Measured results in the
   README status (2026-09-16): 49 of 50 instantiable potentials differentiable w.r.t.
   `amp` under grad and jit; orbits 1.3e-6..2.2e-8; Staeckel 1.7e-10, Adiabatic
   1.9e-10, Spherical 2.7e-11; streamspraydf 1.2e-8; dehnendf 6.7e-10. Audits:
   `plan/differentiability_audit_2026-09-16.md`, `..._followups_2026-09-19.md`.
   For the variational C code: the "FD-of-flow doctrine" (det M=1 and symplecticity
   are only necessary conditions; the STM columns are pinned against finite
   differences of independently integrated orbits; `README.md:130-132`, figure g9).
3. **Reference implementations.** mpmath at 30-50 digits as the arbiter for special
   functions and quadratures (#1596 "match 50-digit references to <=3e-14"; issues
   #1276/#1277 "checked against an mpmath Hankel reference at 30 digits, and the
   reference itself validated per quantity before use", `README.md:350-354`);
   `scipy` for the special-function fallbacks (#964 "vs scipy 1e-13..1e-15 + gradcheck").
4. **Adversarial subagent reviews.** Independent Claude agents instructed to break
   the PR against the real code before merge (e.g. the seven "overnight deep-review"
   reports of 2026-06-05; #1181 "Adversarial review of the first commit surfaced two
   real defects ... `sample()` handed the *same* key to both the position- and
   velocity-angle samplers"; #1027; #1604-#1608 "Adversarial review: numpy
   byte-identical (241 probes); fixed ... radec/lb backend IC (pre-existing silent
   wrong answer)", `log.md:60695`). Memory rule: "subagent reviews, not happy-path
   pytest" (`adversarial-review-complex-migrations`).
5. **Codex reviews** by Jo (A.2) at the module (#966), umbrella (#1204) and
   numerics (#1596) levels.
6. **Five-mode CI** (A.2), with the ledger making every remaining gap explicit and
   measured; the `--jit` axis found "five distinct problems ... visible only under
   `--jit`" in its first sweep (`README.md:284`).
7. **GPU runs.** The CI is CPU-only, so GPU correctness was established on the dev
   box's TITAN Xp and on an H100 (Trillium): "GPU device-correctness for every
   backend-migrated action-angle method ... 17/17 torch CUDA + 6/6 jax GPU at machine
   precision ... surfaced + fixed the GL-node and delta device-anchoring bugs (CI has no
   GPU, so they were latent)" (`g17/caption.md`); "jax main-suite GPU sweep done: no
   correctness failures" (2026-10-01, `log.md:60735`); the CUDA run doubles as the
   numpy-island detector (#1604-#1608); #1636 "VI tables follow the potential's CUDA
   device" was found by figure gV2's GPU run.
8. **Figures as tests.** The reproducible-figure harness (`plan/trackG_figures.md:55-
   75`: every figure has `make.py`, a sha256 in `manifest.json`, `verify` as "the
   reproducibility gate", measure/plot separation) repeatedly caught galpy defects:
   gD2 -> #1343/#1568; g11 -> #1633; gV2 -> #1635/#1636; gS4 -> #1637; g17 -> a
   pending numpy Staeckel c=False performance regression (26.4 s vs 2.07 s at N=1e4,
   `logs/figure_rerun_2026-10-06/log.md`).

---

# Part B -- the showcase examples

Source for each: `paper/figures/<name>/caption.md` where present, else the `make.py`
docstring (noted "[docstring]") or `README.md` (g19/g20). "Full example" = a fit or
inference run end to end; "illustration" = a derivative/validation demonstration.

## B.1 Autodiff / differentiability showcases

**g14_hvs_potential_fit -- "The Great Escape"** [docstring + `data/fit.json`; no
caption.md]. *Shows:* a hypervelocity star (S5-HVS1-like mock, ejected from Sgr A*)
integrated backward must return to the Galactic Centre; the closest approach to the GC
is the differentiable objective. (a) C-STM initial-condition fit at fixed potential,
within the observational uncertainties (gradients dx(t)/dx0 from the C variational
integrator); (b) in-backend AD fit of the halo flattening q by reverse-mode autodiff
through the diffrax solve (a parameter gradient the C-STM cannot give), Adam descending
the closest-approach landscape. *Point:* the two differentiable-orbit mechanisms
composed in one science problem. *Numbers:* potential "cored flattened
LogarithmicHalo", q_true = 0.85, recovered q = 0.8512; IC fit reduces the closest
approach from r = 0.0245 to 0.0143 (internal units) in 7 iterations. **Full example**
(two fits). Caveat in the docstring: "real S5-HVS1 data is a follow-on".

**g19_neural_aa_inverse -- neural AA inverse (1D vertical)** [README.md; not in the
harness; PNG dated 2026-07-13]. *Shows:* a 3->2 MLP (3 tanh layers, width 128) learns
the inverse action-angle map (J_z, theta_z) -> (z, v_z) of the anharmonic vertical
potential of MWPotential2014 at R0, by backpropagating through galpy's differentiable,
jittable forward `actionAngleVertical` (cycle-consistency term + data term, action-
weighted loss, harmonic-amplitude input features); trained identically in flax/optax
(jax, GPU) and torch. *Point:* galpy's AA machinery sits inside an ML training loop
("autodiff, jit, and ML in both backends"). *Numbers:* learned inverse vs the exact
numpy `actionAngleVerticalInverse` (ground truth only, never in the graph): RMS
phase-space error ~5e-3 at small J_z growing smoothly to ~2e-2 for the largest orbits,
no systematic bias; panels: tori vs exact, jax vs torch training curves, accuracy vs
J_z. **Full example** (a trained model), validated against an exact inverse.

**g20_neural_staeckel_inverse -- neural Staeckel inverse** [README.md; PNG dated
2026-07-14]. *Shows:* the harder sibling: a 4-layer MLP learns (J_R, J_z, theta_R,
theta_z) -> (R, v_R, z, v_z) at fixed L_z in MWPotential2014 by backpropagating
through `actionAngleStaeckel(c=False)` (the jittable pure-Python path; ~0.5 s per
512-orbit step on GPU), with supervised warm-up and NaN guards. *Point:* a map with no
closed form and no reference inverse is learned purely from the differentiable forward.
*Numbers:* round-trip F_J(G(J,theta)) = J RMS 7.3e-3 (from 1.3e-2 before output-box
resizing), no high-J bias; reconstructed meridional (R,z) tracks of integrated orbits.
**Full example** (torch works too but eager torch is slower; headline run is jax).

**g3_param_gradients** [caption]. *Shows:* (a) d(Phi, F_R)/d(amp, b) for Plummer and
Isochrone: jax vs torch agree to ~1e-16 (below float64 eps), AD vs central FD to
~1e-10 (the FD floor); (b) Plummer scale length recovered by gradient descent from
b=1.5 to b_true=0.85: "recovering b=0.850000 and a loss below 1e-17", both backends
overlie. *Point:* potentials differentiable w.r.t. constructor parameters, exactly and
identically on both backends. **Illustration + toy fit.**

**g7_orbit_sensitivity** [caption]. *Shows:* in-backend orbit integration (diffrax):
(a) STM elements dx(t)/dx0 by reverse-mode AD through the ODE solve coincide with
central FD of the flow; (b) Gauss-Newton fit of the launch velocity (v_R0, v_z0) to a
target end-point using the `jax.jacobian` of `integrate_orbit`: "the end-point miss
falls quadratically to machine precision in ~5 iterations". Backend note: torch uses
dopri5 because torchdiffeq's dopri8 backward carries ~1e-5 gradient error. **Toy fit /
illustration.**

**g9_variational_validity** [caption]. *Shows:* the C variational equations in
MWPotential2014 (`dop853_c`, rtol=atol=1e-12): (a) |det M - 1| <= 2.6e-8 (3D) /
1.3e-6 (planar) over t=0-100 (~3.6 Gyr); (b) symplecticity defect <= 1.0e-6 / 1.3e-6;
(c) the ground truth: STM columns vs FD of independently integrated orbits trace the
classic V in eps, agreeing at the floor 1.3e-8 at eps=1e-6 -- "the analytically
propagated derivative is at least as accurate as the best finite difference"; (d) 2D
reduction bridge 1.0e-8 with exactly zero out-of-plane leakage. **Validation
illustration.** (Measured against an early build, `7ca49a1b` + a local fix.)

**g10_lyapunov** [caption]. *Shows:* `Orbit.lyapunov` (Benettin renormalised deviation
vectors with the C variational equations): (a) Henon-Heiles E=1/8 chaotic orbit F
converges into the literature band 0.04-0.08: lambda(1.5e4)=0.0425, lambda(3e4)=0.0456;
the regular orbit E at the same energy decays as ln(t)/t to 2.7e-4 (factor ~170); (b)
a regular Sun-like MWPotential2014 orbit decays as ln(t)/t to 2.8e-3 (~0.08/Gyr).
**Validation illustration.**

**g11_dissipative_phase_volume** [caption, re-measured 2026-10-06]. *Shows:* for
velocity-dependent forces det M(t) = exp(int tr(dF/dv) dt') replaces Liouville:
Chandrasekhar friction sinks an orbit from r~1.0 to 0.05, contracting phase volume by
>5 decades, det M(10) = 6.2e-6; FDM friction at the same GMs contracts only to
det M(10)=0.52 (quantum-pressure suppression C_FDM/C_CDM in [0.48,0.59]). Both sides
agree to 5.7e-6 (Chandrasekhar; a trapezoid quadrature floor) and 3.5e-7 (FDM). *Point:*
the dissipative variational equations are right, and the cross-check "caught a C/Python
inconsistency in the friction force at small radius" -> galpy #1633. **Validation
illustration** that found a bug.

**g16_icjacobian_precision** [docstring + `data/precision.json`]. *Shows:* the IC
Jacobian dx(t)/dx0 of a MWPotential2014 orbit over 3 Gyr computed four ways -- C-STM,
FD of the flow, jax AD (jacrev through diffrax), torch AD (torchdiffeq) -- (a) one
element the four ways, indistinguishable; (b) relative Frobenius residual vs C-STM.
*Numbers (data):* FD median 4.8e-10 (max 6.2e-10); jax AD median 1.0e-10 (max
2.8e-10); torch AD median 8.5e-11 (max 5.6e-10) (the docstring's older text says "~1e-8"
for AD; the current data is tighter). *Point:* "The autodiff gradients ARE the
variational state-transition matrix, to solver precision." **Illustration.**

**g18_actionangle_sensitivity** [docstring + README]. *Shows:* actionAngleIsochrone
(closed form) under jax: (a) J_r(v_R), J_z(v_R) with dJ/dv_R overlaid vs FD; (b) J_r(b)
with dJ_r/db -- "the gradient an inference loop backpropagates when it tunes the
potential to match measured actions". *Numbers:* max |AD-FD| 4.9e-10 for dJ_r/db
(README). The docstring notes the Staeckel analogue waits on the derivative-integrand
work (since done: #1051 C-native, README 2026-09-16 "Staeckel(c=False) 1.7e-10").
**Illustration.**

**gD1_df_sensitivity** [docstring]. *Shows:* (a) `jeans.sigmar` d(sigma_r)/da through
the Jeans quadrature by `jax.grad`, vs FD; (b) `isotropicHernquistdf` f(E) and
df/d(amp) vs FD. *Point:* DFs are differentiable; practical lesson: dispatch follows the
coordinate array, not the parameter. **Illustration.**

**gD2_df_sample_autodiff** [docstring]. *Shows:* d(sigma_r)/dr by `jax.grad` of a
local-linear kernel estimator over a FIXED 4e6-particle Hernquist sample vs `jax.grad`
of `jeans.sigmar` -- two code-disjoint legs. *Numbers:* median agreement 0.6 sigma of
the MC noise (8 sub-samples); innermost point ~5 sigma from kernel bias, kept. *Found:*
"At galpy's shipped n_v_vesc=100 the sampled sigma_r is biased LOW by ~0.6-0.8% at every
radius ... a real pre-existing galpy bug, reported as jobovy/galpy#1343" (fixed #1568).
**Illustration** that found a bug.

**gD4_spherical_df_param** [caption]. *Shows:* derivatives through DF *construction and
sampling*: King models, d ln rho/dW0 at fixed (M, r_t) through the King ODE solve,
eagerly (scipy DOP853 + forward sensitivities) and under `jax.jit` (diffrax in-graph),
vs FD rebuilding the DF on numpy: AD-FD 5e-8..2e-7, jit vs eager <= 2.3e-7, torch vs jax
1.3e-14; an Osipkov-Merritt Hernquist DF constructed and sampled (4e6 particles) in one
jitted function of the scale radius a with a fixed PRNG key: d sigma_r/da by `jax.jvp`
through the sampler agrees with FD of the same sample to 6.5e-8 and with the Jeans
expectation within MC noise (max pull 2.1) -- "the pathwise derivative is an unbiased
estimator". **Illustration** (derivative validation; ~12 min CPU).

**gS1_stream_track_sensitivity** [docstring]. *Shows:* d(streamdf track)/d(halo q) by
`jax.grad` through the whole stream model (auxiliary orbit integration, AD `calcaAJac`,
`lax.map` over track chunks): (a) the track at q0 with first-order AD predictions for
+-dq; (b) per-component AD vs central FD of the same backend track. Explains why
`vmap(jacrev)` under an adaptive-step solver is wrong by 2-13% (DirectAdjoint through
adaptively chosen steps; constant steps restore ~3e-7) and why `lax.map` is used.
**Illustration** (~100 s per track evaluation).

**gS2_streamspray_sensitivity** [docstring]. *Shows:* d(particle-spray stream)/d(q)
through the stripping-time sampler and every particle's orbit integration, with common
random numbers; (b) FD converging onto the AD value as h shrinks for several N.
Requires the theta-probe fix galpy#1346. **Illustration.**

**gS3_track_extrapolation** [caption]. *Shows:* one forward-mode `jax.jvp` pass over a
1500-particle `fardal15spraydf` stream (4.5 Gyr, log halo q0=0.9) predicts the smooth
track at q0+dq for dq up to +-0.2: "A single pass removes 85% of the shift at |dq|=0.025
and 62% at |dq|=0.2"; per-particle tangents agree with FD to 1.6e-5 (second-order
convergence 5.9e-2, 6.0e-4, 1.6e-5 at h=1e-2,1e-3,1e-4). **Illustration with a
quantified use case** (track extrapolation for a fitter).

**gV1_vertical_inverse** [caption]. *Shows:* `actionAngleVerticalInverse`
(momentum-matched torus map) for a self-gravitating isothermal disk, construction and
evaluation natively in jax under one `jax.jit`: (a) tori at sigma0 vs rebuilt at
sigma0+0.1 vs the first-order AD prediction; (b) forward-mode AD vs 5-point FD along the
J=0.06 torus: <= 1.4e-10 for d(z,v_z)/dsigma, 2e-11 for dOmega/dsigma, <= 2.3e-12 for
d/dJ and d/dtheta; Poisson bracket {z, v_z} = 1 to 9e-16 from the AD Jacobian alone;
(c) FD converges onto AD as h^4; (d) **a fit**: 120 noisy stars on three tori,
Levenberg-Marquardt on (sigma, amp) + 120 angles with an all-AD Jacobian, converging
in 10 steps from (0.7, 0.6) to sigma = 0.460 +- 0.021, amp = 1.037 +- 0.030 (chi^2 =
98.4 / 118 dof), truth inside the 95% contour. **Full example** (panel d) +
illustration.

**gF1_autodiff_forces** [docstring]. *Shows:* F_R = -dPhi/dR by autodiff vs galpy's
hand-coded `Rforce` for SCF, DiskSCF, spiral arms, a rotating bar and composites:
residuals "within a few ULP ... autodiff is not an approximation to the analytic force
here, it IS the analytic force". **Illustration.**

**g1_backend_parity** [caption] (context for the above): numpy vs jax bit-identical
(exactly zero difference) for Plummer/Isochrone values, forces, second derivatives on a
60x60 grid; torch within 1 ULP (non-correctly-rounded `pow`); AD identities to ~1e-15.

## B.2 Speed figures (one-line headlines from the captions)

* **g2_throughput** -- Plummer F_R throughput vs N: compute-only the GPU overtakes the
  CPU at N~1e4; at N=1e7 the TITAN Xp is ~100x its CPU (~1e9 vs ~1e7 pts/s), the H100
  reaches ~7e10 pts/s under jax (~800x Trillium's CPU); end-to-end with host<->GPU
  copies the win is a modest ~4-6x on either GPU (PCIe-bound, saturating at ~1e8 /
  4-8e8 pts/s).
* **g5_autodiff_overhead** -- gradient/value wall-time ratio: jax ~1.1x (TITAN Xp; 1.9x
  at N>=1e6) and ~1.0x (H100); eager torch 2-5x (2.3-2.9x on H100); N-independent.
* **g6_jit_amortization** -- jit compile ~110-120 ms (TITAN Xp) / ~60 ms (H100), then
  0.7 ms (N=1e6) / 0.07 ms per call vs eager 2.2-3 ms / 0.7 ms; break-even ~71 calls
  (TITAN Xp), ~102 (H100).
* **g8_galax_orbit_speed** -- MWPotential2014 orbits, 2 Gyr, rtol=atol=1e-8: galpy
  `dop853_c` ~0.1 ms/orbit single-thread; galpy in-backend jax GPU amortises to ~0.21
  ms/orbit (TITAN Xp, N=3e5), galax per-orbit 0.209-0.232 ms (equal to galpy within
  1-5%), galax batched-controller 0.07-0.08 ms; on the H100 galpy reaches ~0.025
  ms/orbit (~4x single-thread C, crossing it near N~2e4).
* **g12_inbackend_orbit_gpu** -- galpy's own in-backend integrator: H100 9-12x faster
  per orbit than the TITAN Xp (jax Dopri8 0.025 ms/orbit at N=3e5 vs 0.21; torch eager
  dopri5 0.009 ms at N=1e6 vs 0.076); GPU lines overtake single-thread C at N~3e4 on the
  H100 and are 4-11x cheaper at the largest N; same-work check: jax Dopri5 545/3547
  steps/nfev ~ torch dopri5 578/3470.
* **g15_icderiv_speed** [docstring; no caption] -- cost of d(orbit)/d(x0): C-STM (six
  column solves in C) vs reverse-mode AD through diffrax/torchdiffeq, single orbit and
  vs batch size N, TITAN Xp + H100 bars.
* **g17_staeckel_actionangle_speed** (`_jax`, `_torch` PDFs) -- Staeckel
  `actionsFreqsAngles`: vectorised numpy ~ single-thread C at every N (26 vs 25 s at
  N=1e5) and 13-17x faster than the old per-object loop; torch GPU amortises to ~18x
  faster than C/numpy by N=1e5 (1.4 s); eager jax dispatch-bound (~7 s flat). (CPU
  series flagged load-corrupted on 2026-10-06, see B.3.)
* **gD3_spherical_df_speed** -- `sample(n=1e7)` jitted jax 3-17x numpy on CPU; on GPU
  39-68 ms (TITAN Xp) and 3.9-7.2 ms (H100, ~1000x numpy); constantbetadf construction
  2-2.6 min (numpy) -> 0.13 s warm under `jax.jit` (10-1200x for Eddington-type
  inversions); kingdf's scipy ODE already as fast as any backend.
* **gS4_streamdf_speed** -- streamdf: numpy/C build 4-27 s for 4-32 chunks vs ~90-110 s
  warm differentiable jax build (flat in chunks; 15-24 min to jit-compile; eager torch
  ~22 min and >100 GB, not plotted); sampling and DF evaluation ~100x faster under
  `jax.jit` on CPU (N=1e5: 0.51 s vs 77 s; 0.13 s vs 17 s); jitted DF evaluation at
  N=1e6 67 ms (TITAN Xp) / 1.7 ms (H100) vs 149 s numpy.
* **gS5_streamspray_speed** -- `fardal15spraydf.sample(N)`: jitted jax ~1-1.5
  ms/particle on CPU with no fixed overhead (vs non-differentiable C ~2.1 ms/particle);
  eager jax/torch floors ~6 s / ~12 s; GPU jitted jax 0.15 ms/particle at 1e6 (TITAN
  Xp), 7 us/particle on the H100 (7.2 s for 1e6, ~20x the TITAN); eager torch H100
  22 us/particle.
* **gV2_vertical_inverse_speed** -- `xvFreqs` for N angles: on CPU numpy fastest at every
  N (2.1 s at N=1e5 vs torch 5.1 s, eager jax 16.5 s, jitted jax 30 s -- a fixed-schedule
  bisection of 43/100 passes vs numpy's Newton, "a known, fixable inefficiency"); H100
  jitted jax 30 ms at N=1e6 (0.95 s TITAN Xp); construction at n_E=256: torch 0.53 s,
  eager jax 0.65 s, jitted jax 1.4 s vs numpy 4.7 s; H100 warm jitted construction
  3.5-5.7 ms.

Hardware/software per caption: dev box NVIDIA TITAN Xp (sm_61; float64 at 1/32 of f32
rate; jax 0.10.1, torch 2.12.0+cu126; Xeon E5-2697 v4, shared and heavily loaded, CPU
series pinned to 4 cores/1 thread); NVIDIA H100 80GB on the Trillium cluster (jax
0.11.2, torch 2.14.1+cu130; data in `<fig>/data/h100/`, added 2026-10-06 evening).
H100 vs TITAN Xp summary (`logs/figure_rerun_2026-10-06/log.md`): 8-12x for ODE
stepping and eager torch, 30x for the bandwidth-bound Plummer force (1.7 TB/s), 30-40x
for arithmetic-dense kernels (gV2 root-finding, gS4 DF evaluation).

## B.3 Current vs stale, per the manifest and the 2026-10-06 re-run log

`paper/figures/_harness/manifest.json` stamps `env.galpy_git` (added 2026-10-06). All
30 harness figures were re-run on 2026-10-06 against galpy `feat/backends` **9f6c9d6f7**
(+ cherry-picked #1634/#1635/#1636 for the timing chain), except:

* **g11_dissipative_phase_volume**: stamped **4abce12a5** = `feat/backends` + the
  friction fix #1634 (deliberate; the caption says so). Current.
* **g2b_transfer**: `galpy_git` absent; marked "DRAFT" in its description (superseded
  by g2's end-to-end series). Not a paper figure.
* **g19_neural_aa_inverse, g20_neural_staeckel_inverse**: NOT in the harness (no
  `make.py`/manifest entry; `train_and_dump.py` + `make_figure.py`,
  `make_staeckel_figure.py`); PNGs last generated 2026-07-13/14 (galpy-jax commits
  "Track G: neural action-angle inverse figures" and "g20 ... fix high-action bias",
  2026-07-14). **Stale relative to current galpy** (not re-run since July; the README
  numbers date from then). Their READMEs are also mis-numbered internally ("G18"/"G19").
* Re-run outcome (`logs/figure_rerun_2026-10-06/log.md`, CPU phase 21:37-22:04):
  byte-identical g0, g3, g18; round-off/MC-level data changes only for g1 (jax now
  bit-identical to numpy, as the caption claimed), g7, gF1, gD1, gD2 (MC 2e-3), gS1
  (1e-5), gS2, gS3, g9, g10, g14, g16; g12 work counts reproduce exactly after a
  script fix; g13 rebuilt from 4 to 146 data points with the jit series.
* **Flagged as load-corrupted and to be re-measured on an idle box** (log, evening
  entry): **g17** (all CPU series; its torch GPU series was re-measured at load ~9 and
  "kept only as a placeholder") and **gV2**'s CPU series (box load ~190 during the CPU
  phase). Treat g17 and gV2 CPU timings as provisional.
* Open from the re-run: a galpy regression found via g17 (numpy Staeckel `c=False`
  actionsFreqsAngles N=1e4: 26.4 s now vs 2.07 s in August, C unchanged; bisect
  pending); gS4 "streamdf with a torch progenitor cannot be constructed" (fixed in
  #1637, which also measured the full torch build at 1346 s); #1637's jax.jit of the
  construction works (723 s; 1.1e-5 rel from eager -- follow-up).
* Figures lacking a `caption.md` (text lives in the `make.py` docstring only): g14,
  g15, g16, g18, gD1, gD2, gF1, gS1, gS2 (and g19/g20 have README.md instead).
