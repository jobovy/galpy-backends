# Vacation status snapshot — 2026-08-21 (gh#1352 + gh#1356 both near-green, awaiting last shards)

**This file is OVERWRITTEN, not appended.** It is the "where are things now"
answer. `log.md` is the trace; read this first and go there only for the why.

---

## Nothing is blocked on Jo except gh#1356 (a main PR — main PRs are Jo's to merge)

## Branch state

    main                   45e2dcde   (gh#1355 merged)
    feat/backends          982b1781
    feat/backends-develop  d25dac4a   (gh#1348, gh#1336 merged; 106 ahead of feat/backends)

## Open PRs — TWO ARE MINE. Four others share my git identity and are NOT mine.

**`--author @me` returns SIX open PRs. Only two are mine.** The other agent
session (`session_01N2Jm4Y1A9eQRXMEaHaqwwJ`) commits under the same identity.
Attribute by the `Claude-Session` trailer on the head commit before acting:

    gh#1352  MINE          backend/streamdf-grad-tintj  -> feat/backends-develop
    gh#1356  MINE          fix/adiabatic-action-quadrature -> main
    gh#1357  OTHER AGENT   staeckel-chi-quadrature
    gh#1353  OTHER AGENT   aainv-spherical-quadrature
    gh#1351  OTHER AGENT   aainv-vertical-quadrature
    gh#1347  OTHER AGENT   aainv-staeckel-single

**gh#1352 → develop**, head `cae49223`. 150 pass / 0 fail / 3 pending
(`all-clear`, a jax qdf/pv2qdf/streamgapdf shard, a torch quantity/coords shard).
MERGEABLE/BLOCKED only on those. **Pre-merge audit is DONE** (see log): diff is
`backend_slow_skip.txt` +21, `backend_xfail.txt` −28, `test_backend_streamdf.py`
±50, **no source files** — so the if→xp.where / duplication / helper-placement
criteria are N/A by construction. The test change is a CONVERGENCE fix (tintJ
8→30, FD settled to 8.7e-06 at h=1e-4), **not** a tolerance loosening; the 3e-3
bar is unchanged. **Ledger delta: xfail 11→8, slow_skip 75→79, skip 68 flat.**
Next action: verify green + head still `cae49223`, then squash-merge. Mine to merge.

**gh#1356 → main**, head `40716dfe`. 149 pass / 0 fail / 2 pending / 3 skipping.
Fixes gh#1354: galpy's C adiabatic action quadrature lost ~4 digits at the sqrt
turning points. jr 7.88e-04 → 6.83e-10, jz 1.52e-04 → 9.31e-10 (order 10→20,
cost +33%). **Jo's to merge.** Conflicts with gh#1357 in `HISTORY.txt` ONLY;
I offered to take the one-hunk resolution — gh#1357 is still OPEN, so there is
nothing to resolve yet.

## Flagged to Jo, not acted on

gh#1357 removed the C/Python bit-parity claim for Staeckel, so any backend-vs-C
test asserting ULP-level agreement may now assert something untrue. Not mine to
change (other agent's PR).

## Ledger (at develop d25dac4a)

    backend_xfail       11      -> 8  once gh#1352 lands
    backend_slow_skip   75      -> 79
    backend_skip        68      unchanged

Remaining 8 xfail after gh#1352 = 6 by-design/upstream + 2 gated on gh#1356
reaching develop. **Zero open fixable defects in the xfail ledger.**

## Deferred / next up

* torch namespace-threading: 89 of 192 graph breaks are `_resolver.py:45`
  `_BACKEND_CTX.get()`. ContextVar reads ARE the break (positive control: 1 for
  a ContextVar, 0 for a module global / default arg / closure). Fix is a ~100-site
  change blocked on a thread-safety design call. **The jax/torch jit gap is ONE
  cause, not diffuse rot.**
* Off-grid AA fallback — gated on gh#1356 reaching develop.
* Appendix F (`galpy-jax/appendices/F_adiabatic_action_quadrature.md`, 202 lines)
  documents the GL endpoint theory: for (1−x)^α × analytic, error is O(n^−(2α+2));
  √ endpoint ⇒ O(n⁻³), factor 8 per doubling (measured exponent 2.90–2.99).
  §11 was CORRECTED: AnyAxisym is a Cauchy PV + Hadamard finite part (by design),
  NOT the same √-endpoint bug.

## Standing corrections to how I work

* **Never decrease coverage, not even one line** (Jo).
* **`--author @me` is not "mine"** — check the `Claude-Session` trailer.
* **An accepted `gh run cancel` is not a completed one.** A run with 44 done + 1
  queued job has nothing to interrupt; escalate to
  `gh api -X POST repos/jobovy/galpy/actions/runs/<id>/force-cancel`.
* **Check the same-SHA twin FIRST** on any shard red — but a twin split only
  proves nondeterminism if `behind=0`.
* **A failed shard uploads no coverage**, so a flake presents AS a codecov drop.
  Judge a codecov red by WHICH FILES dropped, not by session counts.
* **File presence is not behaviour.** Run the measurement the fix changes.
* **`TORCH_LOGS` writes to stderr**, and pytest discards stderr for passing
  tests — needs both `2>` redirection and `-s`, or you get a false zero.
* **A built `.so` is not where `build_ext --inplace` says it is.** Assert
  `load_libgalpy()` and `aA._c` or an A/B silently compares Python to Python.

## Cron

`e935a60c`, hourly at :13, session-only, **created 2026-08-19 → expires
~2026-08-26 (5 days)**. Verified alive this wake via CronList.
