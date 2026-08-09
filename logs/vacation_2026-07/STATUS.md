# Vacation status snapshot — 2026-08-09 (15:55 EDT)

**This file is OVERWRITTEN, not appended.** It is the "where are things now"
answer. `log.md` is the trace; read this first and go there only for the why.

*(Discipline note: I had been appending to this file all day, against its own
first line, so it accumulated three superseded accounts of the same problem.
Rewritten from scratch.)*

---

## PR pipeline

develop is at **`de0745c5`**.

| state | PR | effect |
|---|---|---|
| **MERGED today** | gh#1279, gh#1280, **gh#1281** | develop `f963eb42` → `d2ad93c9` → `a45a4f1d` → **`de0745c5`** |
| **open, FOR JO** | **gh#1274** → `main` | **151 green, 0 red, MERGEABLE.** Targets `main`, so it is yours. Blocked only on required review |
| **open** | **gh#1283** `backend-jit/kd-mass-quadrature-v2` @ `07deb9c1` | KD `_mass` → backend quadrature; clears `McMillan17` + `Cautun20`. **138/153 green, 0 red.** Merge audit **done** (below) — merges on green, no changes wanted |
| **open** | **gh#1284** `backend-jit/torch-sweep-fixes` @ `19832bb1` | the torch half of the traced sweep. **Corrected today** — see *Mis-ledgering*. CI restarted from the correction |
| **ready, unpushed** | `backend-jit/rt-vertical-split` @ `cf8828e7` | **clears 10 ledger entries** (5 jax-jit + 5 torch-jit). Fully validated. Opens when the two-PR cap frees |
| **filed as issues** | gh#1276, gh#1277 | two pre-existing numpy accuracy defects, mpmath-validated |

**Every codecov red seen today was an upload phantom and every one cleared on
its own.** The discriminator is the `sessions` count from the *compare*
endpoint (the `commits/<sha>` endpoint does not populate it):

    curl -s "https://codecov.io/api/v2/github/jobovy/repos/galpy/compare/?pullid=<N>"

gh#1283 mid-matrix read base **28 sessions @ 99.95%** vs head **14 @ 72.28%**,
patch **100%** — exactly half the uploads in. It climbed 14 → 26 → clear with no
intervention. Why this repo phantoms so much: **the baseline is 99.95%**, so any
missing shard looks catastrophic.

## The RotateAndTilt surfdens problem — SOLVED

This was the day's main thread. gh#1283 diagnosed it but deliberately left it
open; `rt-vertical-split` closes it.

**Defect.** The traced surfdens quadratures split the vertical interval at `z=0`
and cluster nodes there — right only when the density sits at `z=0`. An offset
or tilted wrapper puts its disk plane elsewhere, so the rule samples a sharply
peaked integrand where it is flattest. Reproduces the failing `tdens =
0.07209774` **to all 8 digits**. Node count is not the lever: n=200 wrong-split
is 6.2e-03; n=50 right-split is 1.2e-10.

**Fix, in four links** — each one was a separate discovery:

1. the wrapper broadcasts over a trailing node axis (re-applies what gh#1283
   reverted; without it the array guard rejects the input outright);
2. `Potential._vertical_quad_split` + the wrapper's plane-crossing solve;
3. **`CompositePotential` delegates both surfdens routes per component** — it
   defines `_dens` but *inherited* `_surfdens`, so the quadrature ran once over
   the summed density at the composite's plane, which is always 0;
4. `testMWPotential` forwards `_surfdens_poisson`, completing a delegation it
   already does for eight other methods.

| mock | split | true density peak | err(split 0) | err(split c) |
|---|---|---|---|---|
| RotatedAndTilted | −0.5000 | −0.4950 | 1.383e-02 | **1.070e-10** |
| …wInclination | −1.0925 | −1.0850 | 4.736e-03 | **1.450e-06** |
| Offset | −1.0000 | −1.0000 | 6.366e-02 | **1.176e-10** |

**Gates:** numpy whole-file **1283 passed / 0 failures**; traced whole-file
**1276 / 0** (7 expected xfail); torch **5/5**; 6 new tests **A/B'd — 6 pass
with the fix, 6 fail without**. Ledger 347 → 342, and 5 more torch-jit lines
once gh#1284 lands (the exact lines are in the PR body).

The 5×-cost hybrid panel rule I had escalated to you as a decision is **no
longer needed** — the declared-plane approach works once the delegation chain is
complete. Keep it in mind only for a user class that sums displaced components
internally without delegating.

## Mis-ledgering found in my own open PR (gh#1284)

Chasing a suspected speedup instead found this. **The torch sweep ran in a
worktree at `e1fa8bf9`, which carried the then-unmerged RotateAndTilt broadcast
fix.** With that fix the wrapper tests run slowly (300–400 s, which is what got
measured); on clean develop the array guard rejects the input and they **fail
fast** — 4 of them in **35.6 s total**.

So gh#1284 as filed classified **4 defects as `slow-skip`**, and slow-skip
*skips* the test: it would have silently masked four real failures in exactly
the traced runs meant to surface them, plus left a fifth unrecorded. Corrected
in `19832bb1` (all five now `backend_xfail.txt` entries) with the body's
reconciliation table and provenance section rewritten.

**Per-PR CI was never at risk:** `JIT_FLAG` is set only by a
`workflow_dispatch` with `jit=true`, so the backend shards run **eager**. The
damage was to the ledger's honesty.

Completeness pass on the rest of that sweep: exactly one other row named the
wrapper (`test_dxdv_3d_c_vs_python[...tiltedTriaxialNFW]`, a removal). Re-measured
on clean develop source: **1 passed in 68.9 s** against a 240 s threshold —
justified, stands. Everything else in the delta is unrelated to that wrapper.

## gh#1283 merge audit — done, clean

- **array-input class:** nothing data-dependent added; `numpy.cos/sin` →
  `xp.cos/sin`, and `out += …` → `out = out + …` (rebinding is the safe form
  once `out` can be traced; identical on numpy floats, consistent with the
  sha256 byte-identity check).
- **duplication:** reuses `backend.quadrature.quad`, adds no helper. The one
  sibling still on `scipy.integrate.quad` (`EllipsoidalPotential._mass`) is
  deliberately untouched with its rationale recorded.
- **helper placement:** the function-local import matches an existing idiom —
  `Potential.py:938` and `AnySphericalPotential.py:185` already use literally
  `from ..backend.quadrature import quad as _bk_quad` inside a function.
- **ledger:** −2, verified by set difference.

## Awaiting Jo

**One action:** merge **gh#1274** (151 green, `MERGEABLE`, targets `main`).

**One decision:** should CI gain a `--jit` dimension? Confirmed today from the
workflow file: `--jit` is set **only** by a `workflow_dispatch` with
`jit=true`, so the traced ledger and the traced slow-skip list are validated on
the dev box and **nowhere else**. Options: a nightly shard on the measured
files / fold into `backend-tests.yml` / accept and document. Needs `workflow`
scope, which my token lacks.

**Two workflow gaps, both needing that same scope:**

1. **No `concurrency:` group** — superseded SHAs' runs keep executing and hold
   runners. Cost real hours twice today; I cancel them by hand each time.
   Note `gh run cancel` **silently no-ops** on a fully-queued run; the
   `force-cancel` API endpoint is what actually works.
2. **`build.yml` triggers on both `push:` and `pull_request:`** — ~52 build jobs
   where 26 would do, on a runner-constrained account.

## Standing caveats

* **numpy stays byte-identical.** Every deviation is flagged, never quietly
  shipped.
* **Traced green is asserted by me, not enforced by CI** — see the decision above.
* `actionAngleVerticalInverse` and `actionAngleTorus` are permanently out of scope.
* The 98 in-place augmented array writes remain a **watchlist, not a defect
  list**; a site only matters if a backend array reaches it.

## Queued next, in order

1. Merge **gh#1283** on green (audit done), then open **rt-vertical-split**.
2. **gh#1284** → green, then rebase rt-split onto it and drop the 5 torch-jit
   lines named in its PR body.
3. **#211(B)** `interpRZPotential` grid-build vectorisation — you chose (B).
   7 structurally identical sites → one module-private helper. Gates:
   per-potential numpy byte-identity by sha256, and an explicit decision for the
   2 array-unsafe potentials. Blocked on gh#1274 merging (shared call sites).
