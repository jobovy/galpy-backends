# Vacation status snapshot — 2026-08-21 (gh#1352 MERGED; gh#1358 opened; gh#1356 awaits Jo)

**This file is OVERWRITTEN, not appended.** It is the "where are things now"
answer. `log.md` is the trace; read this first and go there only for the why.

---

## Nothing is blocked on Jo except gh#1356 (a main PR — main PRs are Jo's to merge)

## Branch state

    main                   45e2dcde   (gh#1355 merged)
    feat/backends          982b1781
    feat/backends-develop  a7cf6ed3   (gh#1352 merged; 107 ahead of feat/backends)

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

**gh#1358 → develop** `backend/torch-graphbreak-contextvar`, head `a2cca9a5`, base
`a7cf6ed3`. Head a2cca9a5 (follow-up: `.set` made traceable too -- the first commit was incomplete). CI re-running. Adds `galpy/backend/_tracectx.py::TracedContextVar`
(ContextVar + a `threading.local` mirror read only while `torch.compiler.is_compiling()`)
and moves all three galpy context vars onto it. **Measured on this base: 9/9
`fullgraph=True` FAIL before -> 9/9 OK after, bit-identical, zero galpy break sites**
A/B on galpy's OWN jit("torch") path: baseline 18 break lines (`_resolver.py:45` x5,
`_jit.py:174` x1) -> 0. `_jit.py:174` is `_TRACING.set(True)`, so the `.set` half was
load-bearing.. Tests A/B-verified: the 2 integration
tests fail with the wiring reverted. Also hoists `_no_torch_compile_deprecations` out of
3 byte-identical copies. Gate: torch/jax/numpy 155 passed each. **No ledger change.**
Mine to merge once green.

**gh#1352 MERGED** as `a7cf6ed3` (2026-08-21 06:41Z). Ledger on develop is now
**xfail 8 / slow_skip 79 / skip 68**. Its last shard sat QUEUED 5h51m on pure runner
contention and picked up a runner on its own -- not a wedge, and cancelling would have
cost hours.

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

* **torch graph breaks — NO LONGER BLOCKED. This is the next PR.** Prototyped and
  measured 2026-08-21; patch saved at `prototypes/torch-graphbreak-contextvar.patch`
  (73 lines, 2 files). Two defects: (a) `_backend_dtype` is EAFP and dynamo turns
  the `numpy.issubdtype` raise into a hard `InternalTorchDynamoError` instead of
  routing it to the `except`; (b) the `_BACKEND_CTX.get()` read needs only a
  trace-time mirror, since the backend context is fixed during a trace.
  **A/B with `fullgraph=True`: 9/9 pass with the fixes, 9/9 fail without, output
  bit-identical, galpy breaks 8 -> 0.** The thread-safety objection that had this
  filed as a ~100-site refactor is answered: `threading.local()` traces just as
  cleanly. numpy path measured unaffected (202 passed). Still to do in the PR:
  size the reduction on the FULL torch `--jit` suite (9/9 is 3 analytic potentials;
  the 192-break figure came from a real test file).
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
