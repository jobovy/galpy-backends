# Vacation status snapshot — 2026-08-13 (no PR open; #188 gating locally)

**This file is OVERWRITTEN, not appended.** It is the "where are things now"
answer. `log.md` is the trace; read this first and go there only for the why.

---

## Nothing is blocked on Jo

## Open PRs: NONE of mine

`gh pr list` shows only #1309 (Jo's actionAngleVerticalInverse, -> main), the
standing integration PR #1204, and two unrelated old WIPs (#872, #748). Both of
my two PR slots are free. Six PRs merged in the previous stretch: gh#1300,
gh#1303, gh#1305, gh#1306, gh#1307, gh#1308.

`feat/backends-develop` is at `50dbf316`.

## In flight locally: task #188 (assert_jit_matches_eager adoption)

Worktree `/var/tmp/fpwire`, branch `fix/impulse-quad-tolerances` (name is
vestigial — reset hard onto develop `50dbf316`; rename before pushing).

Six test files swapped from the hand-rolled

    eager = ...; jitted = ...; assert_allclose(jitted, eager)

onto `assert_jit_matches_eager`, which additionally inspects the jaxpr to
confirm the trace actually CONSUMES its arguments. That is the blind spot the
value comparison cannot see, and it is the same class of blind spot that let
the torch-tracing bug (`startswith` on `array_api_compat.torch`) survive
undetected until codecov flagged 11 dead lines.

Sites adopted: `test_backend_interpolate.py`, `_evolveddiskdf.py`,
`_anyaxisymdisk.py` (via `ref=`), `_constantbetadf.py`, `_dynamfric.py` (via
`ref=`), `_actionAngle.py`.

**Classification first, per the task note** — of 67 raw `jax.jit` sites:
17 direct `jit(f)(x)` CANDIDATES, 17 bound-to-a-name, 9 AD-compile (grad/jac,
NOT adoptable), 21 other. Only the candidates are adoptable; the raw 67 is not
the backlog. `test_backend_trace.py:58` inspected and explicitly REJECTED (it
asserts `under_trace` fired, not a value match).

Gate running: 6 files x {numpy, jax, torch}, log
`scratchpad/jit188b.log`. Not yet committed or pushed.

### Two self-inflicted errors already caught in this task (both by guards)

1. Patch strings built from `sed` output I had piped through `sed 's/^/    /'`
   carried 4 extra spaces -> 0 matches. The script REPORTED `SKIP`, so it was
   caught immediately — unlike the earlier silent-write failure, where an
   assert failed and the script wrote nothing at all.
2. The "insert after the last import" heuristic picked a line inside a
   multi-line parenthesized `from ... import (`, splitting it -> SyntaxError at
   collection in 3 files. Fixed by AST-locating the last top-level
   Import/ImportFrom `end_lineno`. Also: the `"backend_jit_helpers" not in s`
   guard silently skipped `test_backend_actionAngle.py`, which already had a
   FUNCTION-LOCAL import of a different helper from that module.

## Cron

`351c0c5a`, hourly at :13, recurring, session-only. Created 2026-08-13 ->
**expires ~2026-08-20**. Verified alive this wake.

## Next up (queue, in rough priority order)

* **#66** impulse quadrature: `epsabs=0` REFUTED with a dense scan (spikes MOVE,
  they do not vanish). Acceptance gate for any future fix: dense scan, spacing
  <= 2.5e-5, over at least [-3.5,3.5], ZERO points above 1e-10. Untested leads:
  `points=` to name the integrand structure; compare against a high-order fixed
  GL over the same grid (also reveals whether the backend twin
  `_GENERAL_QUAD_N=100` shares the defect).
* **#215** COMPUTE_METHODS widening — 84 candidates, 14 violations, needs
  per-site judgment.
* **#217** residual 8 jax `--jit` units failures: `Quantity` built INLINE in the
  method body.
* **#218** PARKED with a measured negative result (needs traced Rforce ~7e-11
  absolute; unreachable).
