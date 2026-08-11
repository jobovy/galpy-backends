# Vacation status snapshot — 2026-08-10

**This file is OVERWRITTEN, not appended.** It is the "where are things now"
answer. `log.md` is the trace; read this first and go there only for the why.

---

## Jo's current plan (supersedes the earlier "rebase now" thread)

1. Drive the open PRs green and merge the develop-targeting ones.
2. Fix the four issues I filed on `main`, plus the interpRZ vectorisation.
3. **Everything finishes, then Jo rebases** `main` -> `feat/backends` ->
   `feat/backends-develop`.
4. Work resumes on `feat/backends-develop` after that.

So: **do not push anything that is not part of step 1 or 2**, and expect to
rebase every open branch after step 3.

## PR pipeline

`main` is at `b68c3ec9` (gh#1274 merged). develop is at `5b08b507`.

| PR | target | what | state |
|---|---|---|---|
| **gh#1291** `backend-jit/torchjit-poisson-undefer` | develop | xfail **21 → 16**, the 5 torch-jit `poisson_surfdens` rows. Ledger only | queued, **no reds** |
| **gh#1292** `backend-jit/gl-order-100` | develop | `symmetric_quad` default 50 → `_QUAD_N`, deleting 3 duplicated overrides + a test pinning the DEFAULT | queued, **no reds** |
| **gh#1293** `fix/quad-tolerances` | **main** | closes **gh#1289** (surfdens `epsabs=0`) **and gh#1290** (streamgapdf distance bar) | queued, **no reds** |
| **gh#1294** `perf/interprz-vectorized-grids` | **main** | interpRZ grid build vectorised, **17–25x** | queued, **no reds**; ULP question **ACCEPTED** |
| **gh#1295** `fix/razorthin-endpoint-substitution` | **main** | closes **gh#1276** (k=R endpoint singularity) | queued, **no reds** |
| *(no PR yet)* `fix/anyaxisym-r2deriv-finite-part` | **main** | closes **gh#1277** (`R2deriv`/`z2deriv` finite part) | **5 commits @ `cb4ae823`, LOCAL-ONLY, green 1329/0/0.** Held back deliberately — see below |

## !! CI IS SERIALISED — THREE PRs ARE PARKED AND MUST BE RESTARTED !!

The runner pool is only 3–5 concurrent and five PRs x ~150 checks thrashed it
(70 queued, zero completing). Per Jo, serialised:

* **RUNNING: gh#1291, gh#1292** (-> develop) **+ gh#1293** (restarted, -> main).
* **PARKED (runs cancelled on purpose): gh#1294, gh#1295.**

Jo's rule: **keep restarting as the queue drains — do not leave runners idle.**
So bring a parked PR back whenever the backlog gets thin, rather than waiting
for a merge.

**When gh#1291 and gh#1292 are green and merged, RESTART the parked three:**

    gh run list --limit 200 --json databaseId,status,headBranch \
      | # pick headBranch in {fix/quad-tolerances,
        #   perf/interprz-vectorized-grids, fix/razorthin-endpoint-substitution}
    gh run rerun <id>        # for each cancelled run

or push an empty commit to each branch. **They cannot merge until this is done** —
their checks are currently `CANCELLED`, which fails all-clear. This is the single
easiest thing to forget in this whole plan.

**Do NOT cancel notebook jobs.** I cancelled two develop `Test documentation
notebooks` runs thinking 99–115 min meant hung; Jo: they sometimes just take
that long, and they are **required for all-clear**. Re-run (both are back as
attempt 2). My "starving the pool" reasoning was also wrong — measured at RUN
level, when the other runs were at 2-of-26 and 3-of-44 jobs active, i.e.
progressing normally. Use `gh run view <id> --json jobs` for throughput.

## gh#1294 ULP question — ACCEPTED by Jo (2026-08-10)

The vectorised interpRZ grid build is **not strictly byte-identical**: NFW and
`AnySphericalPotential` differ from the cell-by-cell loop by **1–2 ULP in 1–4
cells out of 2501**, ~10 orders of magnitude below the cubic spline's own
interpolation error. **Jo accepted it**; landing as-is, no code change.

Standing rule, reconfirmed: **always FLAG a non-byte-identical numpy change**
(size + extent, in the PR body and the status report) — but flagging is not
blocking. Only QUALITATIVE changes wait for an answer. I over-escalated this one
into a gate across several wakes when the existing rule already covered it.

Also disclosed there: potentials that fall back build ~15 % slower
(`DoubleExponentialDiskPotential`), since they pay a failed vectorised attempt
plus the row checks.

## The five main-side items — four fixed, one diagnosed

| item | outcome |
|---|---|
| gh#1289 surfdens `epsabs` | **fixed**, gh#1293. 9.5e-06 → 1.3e-15 vs mpmath |
| gh#1290 streamgapdf tolerance | **fixed**, gh#1293. Verified on the WHOLE `test_quantity.py`: 242 passed |
| gh#1276 RazorThin singularity | **fixed**, gh#1295. 5.21e-03 → 5.38e-05, at the same node count |
| interpRZ vectorisation | **done**, gh#1294. 17–25x |
| gh#1277 AnyAxisym `R2deriv` | **FIXED** — branch is green but **not pushed yet**, see below |

### gh#1277 — SOLVED end-to-end (branch `fix/anyaxisym-r2deriv-finite-part`)

**This row said "DIAGNOSED, NOT FIXED" for a while after it was actually fixed.
It is fixed.** 4 commits, based on `main` `b68c3ec9`, `test_potential.py` green at
**1328 passed / 0 failed**, two files touched
(`AnyAxisymmetricRazorThinDiskPotential.py` +63/-39, `tests/test_potential.py` +63).

The fix is the diagnosis below, carried through: **reparameterise the integrand in
`d = a - R`**, **symmetrise about `a = R`** (which cancels the odd `B/d` term, so
no `Sigma'` is ever needed — only `C = Sigma(R)/2`), substitute **`d = z*sinh(t)`**
so the nodes follow the peak instead of crowding a fixed point, and **snap
`|z| < 1e-9` to the `z = 0` branch** (module constant `_R2DERIV_ZFLOOR = 1e-9`),
which is what removes the sub-1e-11 regression the prototype had. One shared
helper `_finite_part_quad(integrand_d, R, az, C)` serves both derivatives —
`_R2deriv` passes `C = +sdens(R)/2`, `_z2deriv` passes `C = -sdens(R)/2`.

Measured relative error, shipped vs fixed:

    quantity   shipped    fixed
    R2deriv    9.90e-06   4.29e-10
    z2deriv    5.06e-07   2.97e-11

**and it is better at every `|z|` in the band**, not just at `z = 0` — the earlier
"~10x milder" note for `z2deriv` was a `z = 0`-only figure, and shipped code is in
fact *worse* off-plane (9.0e-07 at `z = 1e-12`).

**Why it is not pushed, and the CONCRETE trigger to push it.** The runner pool is
the binding constraint; a sixth PR adds ~77-150 checks to a backlog rather than
filling idle runners. "Push when capacity frees" was too vague and cost several
wakes of re-deliberating, so:

> **TRIGGER: push gh#1277 as soon as gh#1293 reaches all-clear (or is merged).**
> That replaces a finishing PR in the rotation instead of adding a sixth. Do not
> wait for the whole queue to drain — it will not, until PRs land.

Everything else is ready. Body is written at `/var/tmp/aafix/PR_BODY_1277.md`;
two commands:

    /var/tmp/push_ci.sh fix/anyaxisym-r2deriv-finite-part
    gh pr create --base main --head fix/anyaxisym-r2deriv-finite-part \
      --title "AnyAxisymmetricRazorThinDisk: evaluate the second derivatives as finite-part integrals" \
      --body-file /var/tmp/aafix/PR_BODY_1277.md

It is finished work on a **local-only** branch, so it is also the most losable
thing in this plan — if in doubt, push it.

Two scars worth keeping, and they are the *same* mistake twice:

1. Commit `6c069b00` restores `_Rzderiv`, which my
   `s[:index('def _z2deriv')] + new + s[index('def _surfdens'):]` splice silently
   deleted. 1327 passing tests did not notice — `test_potential.py` guards on
   `hasattr(tp, "_Rzderiv")` and `test_orbit.py` excludes this potential outright.
2. **That restore dropped the method's `@check_potential_inputs_not_arrays`** —
   base `main` has 6 of that decorator, the branch had 5. Fixed in `cb4ae823`.

The pinning test added in `2ff3e9c6` caught neither, because it calls `Rzderiv`
with **scalars**: it pins existence and numbers, not the contract. Measured both
ways — with the decorator removed the gold-value test still PASSES, while an array
call dies as `ValueError: The truth value of an array with more than one element
is ambiguous` from inside the integrand, which is exactly the opaque failure the
decorator converts into a clean `TypeError`.

`cb4ae823` therefore pins the **contract across the whole method set**: each of
`__call__/Rforce/zforce/R2deriv/z2deriv/Rzderiv` must accept a scalar (so a
deleted method cannot masquerade as the TypeError under test) and must raise
`TypeError` on an array in either R or z. Full `test_potential.py` after the fix:
**1329 tests, 0 failures, 0 errors, 102 skipped** (junitxml, 709.6 s).

**Swept the whole fleet for the same class**: comparing the `^\s*@decorator`
multiset per changed file, head vs base, gh#1291/1292/1293/1294/1295 are all
identical and gh#1277 is now identical too. Isolated, and closed.

### The diagnosis that got there — a Hadamard finite-part integral

Four hypotheses tested, **three refuted** (details in the issue comment and
`log.md`): not already fixed on develop, not a truncating tolerance, not
integrand cancellation, not a bad reference.

The integrand diverges as `~1/(a-R)^2` with the **same sign on both sides**
(2.5656e+07 vs 2.5686e+07 at `a-R = 1e-4`), so the two sides *add*. The integral
exists only as a Hadamard finite part; `points=[R]` selects QUADPACK's QAGP,
built for *integrable* singularities, and its finite answer is an extrapolation
artifact — accurate to ~1e-5, better at large R where the singularity is
relatively weaker. That is exactly the observed R-dependence.

**Both singular coefficients are now derived and verified**: `C = Sigma(R)/2`
(to 1e-8) and `B = [2R Sigma'(R) - Sigma(R)]/(4R)` (to ~1e-5). Symmetrising about
`a=R` cancels the odd `B/d` term, so **no `Sigma'` is needed** — only C.

**Root cause of the small-|z| band is representability, not quadrature.** The
integrand recomputes `d = a-R`; at R=0.2 the float64 spacing is 2.78e-17, so an
offset of 1e-10 carries 8.3e-8 relative error, amplified by `1/(d^2+z^2)^2`. Any
scheme placing nodes at `R +- eps` with `eps << R` is capped by this — which is
why graded `points` (3.4e+08 at z=1e-10) and a tan-substitution (1.7e+08) both
EXPLODE below |z| ~ 1e-6 while helping above it.

**Prototype** (d-parameterised + symmetrised + `d = z*sinh(t)`), measured:

    |z|        0      1e-12    1e-10    1e-08    1e-06    1e-04
    shipped  9.9e-06  1.0e-05  3.3e-06  1.9e-05  1.2e-05  3.9e-09
    proto    1.9e-10  5.9e-04  2.3e-06  5.5e-08  1.2e-09  6.1e-12

A large win at z=0 and |z| >= 1e-8; a **regression below ~1e-11** (the 5.9e-04 at
z=1e-12 above). The analytic `2C/(u^2+z^2)` model subtraction, added specifically
to handle the ~C*pi/z peak mass, did NOT help.

**RESOLVED in the shipped fix:** that whole regime is below the representability
floor described above, so there is no quadrature scheme that recovers it — the
answer is to stop trying to resolve it and snap `|z| < _R2DERIV_ZFLOOR = 1e-9` to
the exact `z = 0` branch, which is accurate to 4.29e-10 there. All four prototype
results are on the issue.

## Where the ledger stands

| file | develop (`5b08b507`) | projected |
|---|---|---|
| `backend_slow_skip.txt` | **85** | 85 |
| `backend_xfail.txt` | **21** | **16** via gh#1291 |

    slow-skip (85):  jax 51 | jax-jit 26 | torch-jit 6 | torch 2
    xfail     (21):  torch-jit 8 | jax-jit 6 | jax 4 | torch 3

(Counted from the files on develop `5b08b507`. An earlier version of this line
said `torch 1`; it is 2. The 85 total was right.)

**Projected once gh#1291 lands** — it deletes 5 `torch-jit poisson_surfdens`
rows, so xfail 21 -> 16 and the total deferral count **106 -> 101**:

    family        slow-skip   xfail   total
    jax eager        51          4      55
    jax-jit          26          6      32
    torch-jit         6          3       9
    torch eager       2          3       5
                                       ---
                                       101

**That is the shape of what remains: 87 of 101 (86 %) is the jax family, and
torch is down to 14 across both modes.** And per the stopwatch work below, the
eager-jax block is mostly *genuinely slow*, not defective — 24 of 44 measured
stale, 0 of 9 for interp/FDM, 0 of 5 for AndTilted — so it is not a burndown
target so much as a performance one, which is exactly what the shelved
gegenbauer item addresses.

Torch eager is essentially finished (1 slow-skip + 3 xfail). The residual
burndown is ~92 % jax, and **the eager-jax burndown is closed at the stopwatch** —
24 of 44 measured stale, 0 of 9 for interp/FDM, 0 of 5 for AndTilted.

## Closed off this session

**Eager-GL routing is refuted on all three mechanisms** and should not be
retried without a genuinely different idea. The declared `_backend_accepts_arrays`
flag is wrong in both directions; an empirical integrand probe is *necessary but
not sufficient* (`Composite(Any, MN)` passes it and the GL answer is 62 % wrong);
and an n-vs-2n convergence check cannot catch that either, because GL50/100/200
agree to 1e-15 with each other while all three are 62 % off. Prototype committed
on the local-only branch `backend-jit/eager-gl-probe` with the measurements in
its commit message.

Consequence: **vectorisation is the only live lever on the jax ledger**, and the
AndTilted cluster is not reachable by routing.

## Standing caveats

* **numpy byte-identity**: every deviation is flagged, never quietly shipped.
  gh#1293 and gh#1295 move numpy values *deliberately* (both are accuracy fixes
  on `main`); gh#1294's 1–2 ULP was flagged and **accepted**. Flag always; block
  only on qualitative changes.
* **A `codecov/project` red on this queue is the DEFAULT, not a signal.** Both
  gh#1293 and gh#1291 flashed red within half an hour and both were pure upload
  artifacts. The check fires on the first upload of a matrix that eventually lands
  ~27, so every PR here goes red early and clears later. **Discriminate with the
  session count, do not investigate coverage:**

      curl -s "https://api.codecov.io/api/v2/github/jobovy/repos/galpy/compare/?pullid=<N>" \
        | python3 -c "import json,sys; t=json.load(sys.stdin)['totals']; \
          print({s:(t[s]['sessions'],t[s]['coverage'],t[s]['lines']) for s in ('base','head')})"

  head sessions << base sessions with **identical line counts** = phantom, wait.
  Measured: gh#1293 5-vs-6 then healed to 10-vs-8 (+5.63 pp); gh#1291 **1-vs-27**
  on a PR whose entire diff is 5 deleted lines of `tests/backend_xfail.txt`.
  Only re-run `all-clear` if it already FAILED on the phantom — if it is still
  `IN_PROGRESS` it will evaluate correctly on its own.
* Traced green is asserted by me, not enforced by CI — the `--jit` CI dimension
  still needs `workflow` scope, which my token lacks.
* `actionAngleVerticalInverse` and `actionAngleTorus` are permanently out of scope.

## Queued next, after Jo's rebase

0. **Push `fix/anyaxisym-r2deriv-finite-part` and open the gh#1277 PR** as soon as
   runner capacity frees — done, green, local-only. Do this before anything below.
1. Rebase every open branch onto the new develop/main.
2. The 1440–2641 s jax ledger entries — **root-caused, and it is NOT vectorisation.**
   **SHELVED by Jo to post-rebase**; direction already decided, see the section below.
3. Follow-up: the dissipative-composite forced-backend coercion defect.

## The 1440–2641 s jax ledger cluster — root-caused, needs Jo's call

I had this filed as "SCF/Multipole eval needs vectorising". **That was wrong — SCF
eval is already batched.** Profiling 10 scalar `vcirc` calls on
`DiskSCFPotential(normalize=1., N=10, L=10)` under eager jax:

    12290 calls   0.345 s self / 0.441 s cum   jax ufunc_api.__call__
      300 calls   0.043 s self / 0.844 s cum   backend/special/_fallback/gegenbauer.py:25

That is **~1,229 jax ufunc dispatches per scalar `vcirc`**, funnelled through
Gegenbauer. And `gegenbauer.py` **already has** a rolled `lax.scan` fast path — it
is just gated on `under_jax_trace(x)`, so **eager jax takes the Python loop**,
running the three-term recurrence N-1 times stacked over L orders. The eager suite
never traces, so the scan never engages.

So these entries are not missing point-vectorisation and not a missing scan. They
are slow because the fast path is trace-only.

### SHELVED to post-rebase — Jo, 2026-08-10

**Do not start this before the rebase.** It is a future PR and we are driving to a
stopping point. Jo's steer, recorded so it is not re-litigated later:

* **Option 1 (accept as permanently ledgered) is OFF the table** — "it should be
  either 2 or 3".
* **Option 2 is not blocked.** My "it crosses the no-internal-jit directive"
  framing was wrong: *"In principle there is nothing really wrong with 2, it's not
  internal jit."* Rolling a recurrence is not compiling galpy.
* **But: "ideally we would remain more backend agnostic."** That is the actual
  design constraint on this work.

**What to build, then.** 2 and 3 are not really alternatives — 3 was the vaguest of
the three and its *mechanism* is what makes 2 fast. Option 3 concretely = keep the
Python loop but cut dispatches per iteration (hoist the scalar recurrence
coefficients into numpy/Python floats — each is currently its own jax scalar op —
and fuse the per-iteration elementwise ops). That shaves the constant factor but
leaves the count O(N*L), so it cannot reach option 2.

The synthesis Jo's constraint points at: restructure the recurrence body to operate
on the **whole L-order array at once with a handful of `xp` ops**, so the dispatch
count drops for *numpy and torch as well*, with no jax-only branch. That is 3's
mechanism at 2's scale, and it satisfies "more backend agnostic". Prefer it over
un-gating the `lax.scan`. Keep the existing traced `lax.scan` path either way;
the question is only what **eager** does.

Guardrail when it is picked up: the numpy path is byte-identity-sensitive, and
reassociating a three-term recurrence **will** move the last bits. Measure and
flag the size + extent per the standing rule.
