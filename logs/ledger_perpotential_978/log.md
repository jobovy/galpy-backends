# #978 — per-potential `test_potential` ledger regen + post-#980 rebase

PR: #978 (`feat/backend-ledger-perpotential` → `feat/backends`). Milestone: the parametrization
of the 12 `test_potential.py` loop-tests (#977, on main) let the all-backend xfail-ledger be
regenerated at full per-`(potential × test)` granularity.

## Approach
1. **#977** (merged to main) parametrized the `test_potential.py` loop-tests per-potential.
2. Rebased `feat/backends` onto post-#977 main; regenerated the `test_potential` portion of
   `tests/backend_xfail.txt` at per-potential granularity (`GALPY_BACKEND_XFAIL_REGEN=1`,
   jax `-n 4` memory-safe, torch serial).
3. CI: split `test_potential` into **5 `pytest-split` sub-shards** (`--splits 5 --group K`) in
   `backend-tests.yml` so each jax sub-shard finishes under the 75-min session ceiling
   (parametrization → ~1300 jax cases × eager overhead blew the single-shard timeout). Dropped
   the earlier xdist attempt (galpy potentials are not all picklable → xdist worker pickling
   fails; `pytest-split` partitions the collected list in one process, no pickling).
4. **Post-#980 rebase (2026-06-19):** after #979 (all-clear check) and #980 (deepcopy-of-module
   fix) merged to main, rebased `feat/backends` onto the new main (clean — delta was exactly
   #979 + #980), force-pushed, then rebased #978 onto the new `feat/backends`.

## Results
- `test_potential` granularity: jax **19→14**, torch **135→618** (the torch +483 is a
  measurement-resolution change, not a regression — the old coarse entries prefix-matched every
  potential, hiding 618 real failures behind a handful of lines; jax was *over*-ledgered).
- Post-#980 rebase delta: **+1 torch xfail, +1 jax slow-skip**, both for #980's new
  `test_MultipoleExpansion_deepcopy` (torch fails at Multipole *construction*; jax constructs &
  evaluates correctly but exceeds the 300 s per-test timeout → slow-skip). No drops.
- **Final ledger:** jax **293 xfail + 40 slow-skip = 333**; torch **2073 xfail + 1 = 2074**.
  `test_potential` portion: jax 14, torch 619.

## Dead-ends / pitfalls
- **jax `-n auto` (72 workers) OOM-drops results** → under-counted the jax regen (5 instead of
  14); a stable `-n 4` re-run (1308 collected, consistent with torch) gave the true count. Use a
  memory-safe worker count for jax regen/validation.
- **xdist worker pickling** fails on un-picklable galpy potentials → abandoned for `pytest-split`.
- **py3.13-vs-py3.14 regen blind spot:** the 59 torch `amp_mult_divide[*Multipole*/*SCF*]`
  entries were *assumed* to be #980's deepcopy-of-module bug and expected to drop once #980 was
  in the base. They are NOT — the dev box is py3.13 (tolerates module deepcopy), so the local
  regen never saw the deepcopy bug; those 59 are genuine torch construction failures. Verified
  by reading the actual traceback (`TypeError` at `tclass()` construction) — they still `xfailed`
  (not `xpassed`) on the #980 base. #980's real value: greens the numpy `build.yml` py3.14 red
  and prevents un-ledgered py3.14 backend deepcopy reds the local regen can't capture.

## Validation
- torch full `test_potential.py` (current ledger applied, non-regen, 26m41s): **1 failed
  (only `test_MultipoleExpansion_deepcopy`, now ledgered), 598 passed, 92 skipped, 618 xfailed,
  0 XPASS**. The single FAILED is the new test; 0 XPASS → no drop-candidates → the torch ledger
  matches behavior exactly.
- jax full `test_potential.py` (CPU, `-n 8`, 1 h 58 m): **0 not-in-ledger FAILED**, 1188 passed,
  5 xfailed, **9 XPASS**. The 9 XPASS are slow-but-correct tests (580–2641 s each, junit-measured)
  that time out on CI → **moved xfail → slow-skip** (see below). jax `test_potential` now 5 xfail
  + 15 slow-skip.
- `test_MultipoleExpansion_deepcopy` under jax: **passes in 185 s on CPU** (so slow-skip, not
  xfail; the ~185 s risks the 300 s timeout on slower CI runners).
- **9-test recategorization (xfail → slow-skip).** `test_DehnenBinney98`, `test_Cautun20`,
  `test_McMillan17`, `test_diskscf_overflow`, + 5 rotated/tilted-wrapper `test_poisson_surfdens`
  cases: PASS under jax but 580–2641 s each (≫ 300 s CI timeout) → as xfail they burned ~190 min/
  run of pure timeout. Moved to slow-skip (correct category; reclaims the time; 5-way split groups
  ~46 min each, comfortable under the 75 min ceiling). They fail FAST at construction under torch
  (0.0 s) → stay in the torch xfail-ledger. Final: jax 284 xfail + 49 slow-skip = 333; torch 2073
  + 1 = 2074.

**Methodology bug caught mid-validation (GPU OOM).** The first jax `-n 4` validation reported
~90 % spurious `F` — the py3.13 jax here is CUDA-enabled and **GPU-default**, so the 4 xdist
workers OOM'd the TITAN Xp (`CUDA_ERROR_OUT_OF_MEMORY`). The same OOM is why the test first
appeared to "time out" at 300 s. CI runs jax on **CPU**; the fix is to force CPU
(`CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu`) for all ledger validation/regen — baked into
`/var/tmp/validate_potential.sh`. On CPU the jax run is clean (no OOM). Memory:
`jax-defaults-to-gpu-force-cpu`.

See `appendices/E_coverage_tolerance_ledger.md` (⟂ granularity discontinuity + the post-#980
rebase sub-section) for the burndown table and full reasoning.
