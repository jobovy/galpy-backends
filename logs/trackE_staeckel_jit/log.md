# Track E — Staeckel jit-compatibility, eager-vs-jit speed, and the rolled-loop lever

Branch: `feat/backends`. Surfaced while building the g17 Staeckel action-angle
speed figure (Track G) and asking "should the jax curve be eager or jitted?".

## Approach / what happened
The g17 figure originally timed **eager** jax `actionsFreqsAngles`. Every *other*
jax speed figure (g2/g8/g12/g15) jits the call (compile once, time the warm call) —
so g17 was the lone eager one. Trying to fix that revealed the code **could not be
jitted at all**: `jax.jit(actionsFreqsAngles)` raised `TracerBoolConversionError`
at `actionAngleStaeckel.py:_staeckel_prep`, `if bool(xp.any(unbound)): raise
UnboundError` — a data-dependent Python branch on a traced value. So a user could
not jit the Staeckel action-angle map, contradicting the project goal.

**Fix (PR #1016).** Guard the eager raise behind `not is_backend_array(R)`: numpy
still raises `UnboundError` byte-identically; the backend path skips the branch and
unbound orbits fall through to a (garbage, finite ~1e46) value the caller checks.
Grep confirmed it was the *only* data-dependent branch on the c=False path (the
bisection uses fixed-count `for` loops; the angle tree uses `xp.where`). After the
fix, `jax.jit(actionsFreqsAngles)` traces and matches eager to **9.3e-10**.

## Results — eager vs jit (CPU, MWPotential2014, jax 0.10.1 x64)

| path | small-N per-call | mechanism |
|---|---|---|
| eager jax CPU | ~3.0 s **flat** to N~3e4 | per-op Python dispatch of ~hundreds of ops |
| eager jax GPU | ~7.0 s **flat** to N~3e3 | per-op kernel launches, op-count-bound |
| **jit warm** jax CPU | **11 ms** at N=1, scales with N | one compiled kernel |
| jit **compile** | **~8.6 min / array shape** (one-time) | the unrolled graph (below) |

So jit is a ~280× win at small N **warm**, but behind a brutal one-time compile.
Eager is dispatch-bound (op count, not data size); jit moves the floor from ~3–7 s
to ~11 ms but pays minutes to compile. → jit pays off only for **repeated calls at
a fixed shape** (an optimisation/training loop), not one-off evaluation. g17 now
shows **both** curves (split into g17_jax / g17_torch; torch stays eager —
`torch.compile` errors with `InternalTorchDynamoError` here).

## The root cause of the slow compile: unrolled Python loops

The turning-point solve is fixed-count Python `for` loops (bisection for
umin/umax/vmin ~45 halvings each + 60/80-step expanding brackets), each calling the
3-component MWPotential2014 integrand. XLA **unrolls** them → a giant static graph:
- `make_jaxpr(actions)` → **47,528 equations**, ~3 min just to *trace* (no compile).
- `jit(_staeckel_uminumax)` alone → **226 s compile** / 2184 ms warm.
- full `actionsFreqsAngles` → ~8.6 min compile.

## Prototype — rolled (lax.fori_loop) vs unrolled bisection
`logs/trackE_staeckel_jit/proto_bisect.py` (faithful to `backend.optimize.bisect_root`
over the real J_R integrand, n=45, N=1000, CPU):

| bisection | compile | warm | roots |
|---|---|---|---|
| **unrolled** (Python `for`) | **11.3 s** | 263 ms | — |
| **rolled** (`lax.fori_loop`) | **1.0 s** | 437 ms | **bit-identical (0.0e0)** |

→ **~12× faster compile, ~1.7× slower warm, identical answer.** Rolling kills XLA's
cross-iteration fusion (hence the slower warm) but collapses the graph (hence the
fast compile). It is a genuine **compile-vs-runtime trade**, not a pure win:
- one-off / varied-shape calls (compile-bound) → **roll** (or partial-unroll).
- fixed-shape training loops (warm-bound, compile amortised) → **unroll** (status quo).
- `lax.fori_loop(..., unroll=k)` is the tunable middle ground.

Caveat: rolled-loop primitives (`lax.fori_loop/scan/while_loop`) are **jax-only**;
numpy/torch have no drop-in equivalent, so a rolled fast path must be a
**jax-specific kernel** in `galpy/backend/_jax/` (consistent with the separate
`_jax`/`_torch` file convention + the shared root-find helper, #86/#88). numpy/torch
keep the Python-loop path; a test pins the two numerically identical.

## Further optimisation candidates (for the paper's "cost of differentiability" section)
1. **Rolled `lax.fori_loop` jax kernel** for the 3 bisections + bracket loops (~12×+
   compile; tune `unroll=k`). Biggest lever. Jax-specific.
2. **Fuse the 3 bisections** (umin/umax/vmin) into one batched rolled solve.
3. **`custom_root` / IFT** on the root-find: forward via rolled loop, backward via
   implicit-function-theorem (exact grads, smaller backward graph) — extends the
   existing implicit-diff helper.
4. **Avoid per-shape recompile**: shape-bucketing / padding so a loop over varied N
   reuses one compiled kernel; consider `jax.experimental.compilation_cache`.
5. **Prune the angle quadrant tree**: 12 `xp.where` leaves all evaluate; fold shared
   sub-integrals (the partial integrals repeat across leaves).
6. **Closed-form fast paths** (Isochrone/Harmonic AA — no loops, already jit-trivial)
   used wherever applicable; Staeckel is the one iterative case.
7. **Native special fns** already routed via Pspecial; keep adopting native as it ships.
8. **torch.compile** currently errors (dtype interpretation) — investigate for torch
   users (torch eager is already ~10× lighter dispatch than jax eager, so lower pri).

A systematic codebase sweep for these (AA / orbit / df hot paths) is worth a
dedicated investigation — flagged for follow-up.

## Status
- **PR #1016** (jit-compat guard + GPU device-anchoring + eager no-raise test) → feat/backends, open 2026-06-26.
- g17 split into g17_jax / g17_torch (eager vs jit); GPU re-measure in progress.
- Rolled-loop jax kernel: **not yet implemented** — prototype + numbers above; a future jax-specific PR.
