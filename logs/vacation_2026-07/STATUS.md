# Vacation status snapshot — 2026-08-16, gh#1327 merged

**This file is OVERWRITTEN, not appended.** It is the "where are things now"
answer. `log.md` is the trace; read this first and go there only for the why.

---

## Nothing is blocked on Jo

Two of the three PRs are his to merge (they target `main`), which is the normal
end state, not a blocker.

## Branch state

    main                   54c0324a   (gh#1320 landed since my branch point)
    feat/backends-develop  24500321   (gh#1327 merged 2026-08-16)

## The ledger, verified on develop

    xfail        15   real defects, individually
    slow-skip    78   performance-gated
    skip         68   PERMANENT by design (backend_skip.txt says so)
    ---------------
    TOTAL       161      IN SCOPE = 93  (skip is not pending work)

The (ledgered ∧ passed) sweep over CI artifacts finds **0 stale entries** among
everything a normal PR run can evaluate. The `-jit` half was measured once, via
a dispatched traced run — see below.

## Open PRs

**gh#1323 → `main`** `fix/anyspherical-array-mass`, head `95cb7127`.
**149 SUCCESS, 0 failures.** The AnySpherical array bug (forces silently wrong by
87.7 % away from `r[0]`; `_revaluate` raised outright on arrays), plus the
`_grid_eval` fallback test that keeps repo coverage at 24 misses rather than 25.
Two flakes (SIMBAD outage, 1-ULP symplectic) cleared on re-run. Ready.

**gh#1324 → stacked on gh#1323** `fix/potential-array-input-where`, head
`69db2249`. **CLEAN, 105 SUCCESS.** SphericalShell + HomogeneousSphere accept
array input (if→where, scalar path bit-identical behind `numpy.ndim`);
`numpy.infty` → `numpy.inf` at two live-`AttributeError` sites. Retargets to
`main` automatically when gh#1323 lands.

**gh#1328 → `feat/backends-develop`** `backend-jit/traced-surfdens-fixture`,
head `8771458a`. **Mine to merge.** Both AnyAxisym `surfdens` fixtures dispatch
on `get_namespace(R)` instead of `numpy.exp`. Just opened.

## Merged this wake

**gh#1327** → develop `24500321`. The develop coverage regression, all five
lines, 20 → 15 misses. Ended 152 SUCCESS / 0 failures, `codecov/patch` and
`codecov/project` both green, `mergeStateStatus=CLEAN`. Audited before merging:
no if→`where` array-input issues, no duplication, no misplaced helpers; the one
source change removes a dead `try/except` whose tracer cases (grad, jit, vmap,
vmap-of-grad, torch.compile) were each verified to be diverted upstream.

## #218 is now measured, and it is not what its title says

The title says the backend GL path "lacks BOTH numpy singularity treatments".
It does not — `_R2deriv_gl` already calls `_bquad.finite_part_quad` (#219
landed). The traced error at z=0, against the mpmath gold:

        R      R2deriv      z2deriv
      0.1     2.00e-04     1.08e-05
      0.2     6.65e-04     8.40e-06
      2.0     5.57e-07     2.03e-07

Two facts settle the diagnosis. The semi-infinite tail contributes **nothing**
(n=400 vs n=800 agree to 3 digits), and the finite-part error is exactly
`1/n^2`: 6.65e-4 → 1.67e-4 → 4.18e-5 → 1.05e-5 for n = 100/200/400/800.
Gauss-Legendre is exponential on a smooth integrand, so algebraic `1/n^2` means
a **residual non-smoothness left after the `2c/u^2` subtraction** — most likely
the `log` in `ellipkm1` as `m -> 1`. Reaching 1e-8 by order alone would need
~25 000 nodes; the fix has to remove the residual singularity, not out-resolve
it. scipy's adaptive quad reaches 1e-10 on the same integrand, so the float64
cancellation floor the helper's docstring describes is real but is not what is
binding here.

**The fix is prototyped and validated** (numpy, against the mpmath gold, at the
current default n=100 — no extra nodes):

                   current          with log subtraction
    R2deriv        2.0e-4..6.7e-4   5.6e-12..6.8e-11
    z2deriv        2.0e-7..1.4e-5   8.1e-12..8.2e-11

Subtract `h(u) = -Λ ln(u/b)`, integrate `g - h` with the existing rule, add back
the exact `∫h = Λb`. Λ comes from a least-squares slope of `g` against `-ln u`
over five points `u = b·10^-k`, `k = 3…5` — plain arithmetic on integrand
evaluations, so traceable and differentiable. Λ must not be hand-derived: it
runs from 44.78 (R=0.1) to 0.0002 (R=2.0) for one potential, both `ellipe` and
`ellipkm1` feed it, and my derivation of the `ellipkm1` term alone gave 6.42
against a measured 8.02. The `k=3…5` window is measured, not guessed:
`k=2…4` and `k=4…6` both degrade to ~1e-8 (above, the `O(u)` term biases the
slope; below, the `c/u²` cancellation noise does).

**IMPLEMENTED** on `backend-jit/finite-part-log`, head `67702d93`, pushed but
deliberately not a 4th PR — it wants to sit on top of gh#1328. Measured end to
end under `jax.jit`: `R2deriv` 2.0e-4 → **4.9e-10** against the numpy path's
2.0e-10, `z2deriv` 1.1e-5 → **1.2e-10** — a ~1e6 gap closed to ~4x, at the same
n=100. The tail is not the limiter (n=100/200/400 identical). New test builds a
known-log integrand with a closed-form finite part; A/B flips 1.45e-5 → rtol
1e-9 on all three backends. Gates: quadrature 74 passed, numpy `-k anyaxisym`
13/1 unchanged, same forced-jax and forced-torch. No ledger delta (these traced
failures were never ledgered).

**#218 was two problems under one title.** This fixes the R2deriv/z2deriv half
only. `test_anyaxisymrazorthin_smallz_limit` is `_zforce_gl` → `_bk_split_quad`,
which never calls `finite_part_quad`, so it is untouched and still fails.

**Measurement trap worth recording:** the obvious probe (build the potential
under `use("jax", force=True)`, pass `jnp.asarray(R)`) reports ~1e-10 and looks
fine. That is the numpy fallback — `_bk_dispatch` hands a *concrete* backend
scalar to scipy. Only a real `jax.jit` reaches the GL path.

**Prototype fidelity:** transcribe integrands from the source, never re-derive
them. My first z2deriv prototype was order-unity wrong on *both* arms because I
wrote it from memory; the corrected one reproduces the independently measured
traced errors (1.07e-5 / 8.40e-6 / 2.03e-7) to three digits, and that agreement
is the only real evidence the prototype is faithful.

## Traced state (from the one dispatched jit run, 31914531334)

7 failures, none of them new library defects:

* **2 test-side tracer errors** — the numpy-only `surfdens` fixture above; both
  now gone, 1 as a pass and 1 revealing the accuracy failure below. I first
  reported these as genuine tracer leaks; that was wrong, corrected in the log.
* **2 → 3 known AnyAxisym singularity accuracy** — task #218 (the unmasked one
  joins them).
* **3 torch traced timeouts** — slow-skip material.

## Standing corrections to how I work

* **Never decrease coverage, not even one line** (Jo, 08-16). The loss is always
  a *defensive branch going unreachable*, never new untested code — so a
  fallback needs its forcing test in the PR that introduces it.
* **codecov red ≠ phantom.** Discriminate: could this PR move coverage at all?
  gh#1322 changed 0 source lines (phantom); gh#1327 changes source (judge it).
* **Read the traceback to the innermost frame** before naming a culprit —
  exception *type* is not a diagnosis.
* **The same-SHA twin is not a clean A/B once `main` moves**: the PR arm tests
  head *merged with* new main, the push arm tests head alone.
* **Flake rates sampled from run conclusions are undercounts** — a re-run
  overwrites the failure. Walk `attempts` instead.
* **Check `git branch --show-current` before committing** in a multi-branch
  worktree; `git push` answering "Everything up-to-date" is not success.

## Cron

`351c0c5a`, hourly at :13, session-only, created 2026-08-13 → **expires
~2026-08-20**. Verified alive this wake.
