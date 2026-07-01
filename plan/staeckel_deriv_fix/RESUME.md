# RESUME — session park (2026-07-01)

Master handoff for a fresh session. Three things were in flight; all background
processes were stopped for the restart, but their **on-disk artifacts persist**.
Read this top-to-bottom, then pick up at "Next actions".

Design reference (this dir):
- `design_scoping_wpo9oiys0.md` — the full Staeckel derivative-fix decision doc
  (triage table, PR A–E breakdown, risks). Authoritative design.
- `prA_wip_uncommitted.diff` — snapshot of PR A's WIP code (3 files) at park time.

---

## 1. #1046 (Track F Pdf.1 — df foundation: surfaceSigmaProfile + jeans)  — OPEN, CI running

- Worktree `/var/tmp/pdf1`, branch `feat/pdf1-df-foundation`.
- Was BLOCKED only on `codecov/patch` (96.66%): exactly ONE uncovered line,
  `galpy/df/jeans.py:84` — the callable-`beta` backend branch
  `intFactor = lambda x: xp.exp(2.0 * quad(beta(y)/y ...))` (tests only used
  `beta=0.0`).
- FIX committed + pushed: `7db5c086` "test_backend_df: cover jeans.sigmar
  callable-beta backend branch" — adds `test_jeans_sigmar_callable_beta_parity`
  (r-dependent anisotropy, numpy↔backend parity). Verified locally: 2 passed
  (jax+torch).
- At park time the fresh CI run for `7db5c086` was **queued/in-progress** (45
  passed, ~105 pending). The 3 FAILUREs then showing (codecov/patch,
  codecov/project, orbit `from_name`) were **stale carryover from the prior
  commit** `9f6c978e`, NOT this commit. Expect codecov/patch to flip green once
  the coverage job uploads; `from_name` is a known Simbad/network flake.
- **Resume:** `gh pr checks 1046`. If codecov green + others green → set
  auto-merge (user prefers reviewing the opened PR; they already said "keep going
  with Track F after this lands"). If `from_name` (or run-notebooks/VizieR)
  flaked, `gh run rerun <id> --failed` — do NOT cancel (blocks automerge).
  Task #116.

## 2. PR A — Staeckel ACTION derivatives (the keystone the user prioritized) — WIP, HAS A BUG

User request (verbatim): "we will need to fix the staeckel (and likely other)
derivatives, so not just in C but also Python it seems." Sequencing decision
(user-approved via AskUserQuestion): **"Start PR A now, in parallel"** with
Track F (disjoint files).

- Worktree `/var/tmp/staeckel-jvp`, branch `feat/staeckel-actions-jvp` (off
  `origin/feat/backends` @ `e2458887`). **Uncommitted**, 3 files modified
  (see `prA_wip_uncommitted.diff`): `galpy/backend/_namespaces.py` (+
  `stop_gradient`, `under_torch_grad`), `galpy/backend/quadrature.py` (+
  `graft_jacobian`), `galpy/actionAngle/actionAngleStaeckel.py` (graft wired
  into `_staeckel_actions`). NO test file written yet. NOT built, NOT committed.

### The approach (settled)
Staeckel actions `Jr,Jz` use plain GL whose integrand `√S` vanishes at turning
points (value fine) but whose derivative `∂S/(2√S)` is SINGULAR there → naive
backend AD gives a silently-wrong gradient (~5% off FD). Fix = keep the exact
forward value, REPLACE the gradient with the analytically-correct one via a
**stop_gradient graft** (same 1st-order construction as
`galpy/backend/optimize.py::bisect_root`; NOT a registered custom_jvp — one
uniform jax+torch path). The analytic Jacobian `∂(Jr,Jz)/∂(E,Lz,I3)` already
exists and is validated: `_staeckel_jacobian` (actionAngleStaeckel.py ~L298)
returns `(djrdE,djrdLz,djrdI3,djzdE,djzdLz,djzdI3)` — feeds the freqs that match
C to ~4e-16. Graft: `jr = sg(jr) + Σ sg(∂jr/∂p_i)·(p_i − sg(p_i))` over
`p=(E,Lz,I3U)`; jz over `(E,Lz,I3V)`. `sg`=stop_gradient. numpy path unchanged
(byte-identical); graft gated on `is_backend_array & (under_jax_trace |
under_torch_grad)`. Scope = IC gradients `d(actions)/d(x,v)`; parameter grads
`d/dθ` through the node-potential are a documented follow-up (out of PR A).

### !!! THE BUG (must resolve before committing) !!!
Validation found: **`∂Jr/∂R`: grafted ≈ 1.46 vs finite-difference ≈ 0.0019**
(off by ~1000×), for ALL test orbits — while `∂/∂vR, ∂/∂vT, ∂/∂vz` match FD
well. (Didn't get to `∂/∂z` but suspect it's also wrong.)

Interpretation: `vR,vT,vz` enter Jr ONLY through `E=½v²+Φ` and `Lz=R·vT`, so the
`(E,Lz,I3)` boundary is complete for them. `R` (and `z`) ALSO enter Jr through
the **reference geometry** in the integrand (`u0, sinh2u0, v0u, sin2v0u,
potu0v0`) and possibly through how `I3U` is defined at the reference point. The
design (`design_scoping_wpo9oiys0.md` §2 reason 2) claimed the u0/potu0v0 gauge
terms "cancel against I3U's definition" so `(E,Lz,I3)` is the complete
independent set — the R-mismatch says that cancellation is **NOT holding
numerically** (the true `∂Jr/∂R` ≈ 0.0019 is a near-cancellation the graft's
+1.46 misses by ~-1.458).

TWO hypotheses to disambiguate first (the agent was about to decompose the chain
rule when stopped):
  (a) **Graft incompleteness** — the reference-geometry quantities carry real
      R,z-dependence; need either (i) their own Leibniz derivatives added to the
      graft primal set, or (ii) let AD flow through the smooth reference terms
      and only graft the (E,Lz,I3)-via-√S singular part, or (iii) move the graft
      boundary. Check: numerically evaluate `∂Jr/∂u0` (should be ~0 if pure
      gauge; if not, it's the missing term).
  (b) **FD test bug** — `∂Jr/∂R≈0.0019` being tiny AND constant across orbits is
      itself suspicious; verify the FD actually perturbs R and recomputes Jr
      cleanly (delta held fixed, no caching, aA object rebuilt or R varied in the
      call), and cross-check against order-300 naive-AD (design risk #1: grafted
      must converge to high-order naive-AD; if high-order naive-AD ≈ 1.46 too,
      then the graft is right and the FD is wrong; if ≈ 0.0019, the graft is
      wrong).

**Resume:** in `/var/tmp/staeckel-jvp` (build first: `GALPY_COMPILE_SINGLE_EXT=1
GALPY_COMPILE_NO_OPENMP=1 python setup.py build_ext --inplace`; always
`PYTHONPATH=/var/tmp/staeckel-jvp` + assert `galpy.__file__` under it;
`CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu`). Decompose `∂Jr/∂R` into the three
grafted terms vs FD vs order-300 naive-AD to decide (a) vs (b). Then finish per
the agent brief (below): add `tests/test_backend_staeckel_grad.py`, verify numpy
byte-identity, run `tests/test_actionAngle.py -k staeckel`. Do NOT commit until
`∂/∂R` and `∂/∂z` match FD. Task #119.

## 3. g17 — AA-speed figure refresh (full C/CPU/GPU) — was RE-MEASURING, now stopped

- User request: "action angle speed tests, full c, cpu, gpu." Figure lives at
  `~/Repos/galpy-jax/paper/figures/g17_staeckel_actionangle_speed/`.
- The measurement (`python -m paper.figures._harness run
  g17_staeckel_actionangle_speed_jax`, log `/tmp/g17_remeasure.log`) was RUNNING
  and holding the TITAN Xp GPU; **stopped** for the restart → **must re-launch**
  from scratch (it deletes the stale JSON first, then measures C / numpy-old /
  jax-CPU / jax-CPU-jit / jax-GPU(+transfer) / torch). Design note per
  `design_scoping_wpo9oiys0.md` §5 wants it EXTENDED with actionsFreqs(Angles)
  rows + a GRADIENT panel + custom_jvp-vs-naive-AD — but that gradient panel
  should come AFTER PR A lands (needs the real graft). For now just the forward
  re-measure. GPU rows need the `galpy-gpu` conda env (sm_61: jax 0.10.1 + torch
  2.12.0+cu126); `JAX_PLATFORMS`/`CUDA_VISIBLE_DEVICES` set per row. Task #118.

---

## Stopped background tasks (all dead now; artifacts on disk)
- Agent `a0678eb68a69732dc` (PR A impl) — killed; report: djr/dR bug above.
- Bash `burudqvtx` (g17 re-measure) — killed; re-launch needed.
- Monitor `bl4v2qzci` (#1046 CI watch) — killed; re-arm with `gh pr checks 1046`.

## Next actions (priority order on resume)
1. `gh pr checks 1046` → green? auto-merge : rerun flakes. (Track F unblocks.)
2. PR A: reproduce + root-cause the `∂Jr/∂R` mismatch (hypothesis a vs b) BEFORE
   any further build-out. This is a genuine design question, not mechanical.
3. Re-launch g17 forward re-measure (GPU) once the box is free.
4. After #1046 lands: continue Track F Pdf.2 (spherical DFs). Remove the Pdf.1
   leaf data-guards after Pdf.2/Pdf.3 (task #117 — kill the transitional
   numpy-island; user directive).

## Hard constraints (do not violate)
- numpy path BYTE-IDENTICAL (gold-reference `numpy.array_equal` before/after).
- Every commit ends with `Co-authored-by: Claude Fable 5 <noreply@anthropic.com>` (current-model trailer; user confirmed the switch 2026-07-01).
- No numpy-island / hollow-green; the Pdf.1 leaf data-guards are TRANSITIONAL and
  MUST be removed once consumers migrate (#117).
- Concise backend comments (reviewer flagged verbosity).
- Grep for conflict markers before every push (repo lacks the hook).

## The PR A agent brief (verbatim, to re-run or complete)
See conversation history / the design doc. Key spec: (1) `_namespaces.stop_gradient`
+ `under_torch_grad` [DONE in WIP]; (2) `quadrature.graft_jacobian(xp, value,
primals, jac_rows)` = `sg(value)+Σ sg(jac_i)*(p_i-sg(p_i))` [DONE]; (3) wire into
`_staeckel_actions` gated on backend+differentiating [DONE but BUGGY for ∂/∂R];
(4) `tests/test_backend_staeckel_grad.py` grad-vs-FD (generic + edge orbits
vR=0/vz=0/z=0), graft-vs-order-300-naive-AD witness, value byte-identity [NOT
DONE]; (5) verify numpy byte-identity + `test_actionAngle.py -k staeckel` [NOT
DONE]. Files must stay disjoint: `_namespaces.py`, `quadrature.py`,
`actionAngleStaeckel.py`, `tests/test_backend_staeckel_grad.py`.
