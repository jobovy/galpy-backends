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

CI is saturated — 67 queued / 5 running, because these five PRs each fire ~150
checks. Nothing is wrong; it just has to drain.

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
| gh#1277 AnyAxisym `R2deriv` | **DIAGNOSED, NOT FIXED** — see below |

### gh#1277 is a Hadamard finite-part integral

Four hypotheses tested, **three refuted** (details in the issue comment and
`log.md`): not already fixed on develop, not a truncating tolerance, not
integrand cancellation, not a bad reference.

The integrand diverges as `~1/(a-R)^2` with the **same sign on both sides**
(2.5656e+07 vs 2.5686e+07 at `a-R = 1e-4`), so the two sides *add*. The integral
exists only as a Hadamard finite part; `points=[R]` selects QUADPACK's QAGP,
built for *integrable* singularities, and its finite answer is an extrapolation
artifact — accurate to ~1e-5, better at large R where the singularity is
relatively weaker. That is exactly the observed R-dependence.

**A correct fix subtracts the `1/(a-R)^2` and `1/(a-R)` parts analytically.** A
numerical dodge does not exist: because both sides are positive, the divergence
must be *removed*, not paired away. Left undone deliberately — real analysis, and
the issue itself marks it low priority.

## Where the ledger stands

| file | develop (`5b08b507`) | projected |
|---|---|---|
| `backend_slow_skip.txt` | **85** | 85 |
| `backend_xfail.txt` | **21** | **16** via gh#1291 |

    slow-skip:  jax 51 | jax-jit 26 | torch-jit 6 | torch 1

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
  on `main`); gh#1294's 1–2 ULP is the open question above.
* Traced green is asserted by me, not enforced by CI — the `--jit` CI dimension
  still needs `workflow` scope, which my token lacks.
* `actionAngleVerticalInverse` and `actionAngleTorus` are permanently out of scope.

## Queued next, after Jo's rebase

1. Rebase every open branch onto the new develop/main.
2. **Vectorisation** of SCF/Multipole/wrapper eval — the 1440–2641 s ledger entries.
3. gh#1277's analytic finite-part subtraction, if wanted.
4. Follow-up: the dissipative-composite forced-backend coercion defect.
