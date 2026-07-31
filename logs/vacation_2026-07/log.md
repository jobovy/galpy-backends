# Autonomous stretch, 2026-07-26 → ~2026-08-09

Jo away ~2 weeks. Goal: **fully jittable/compilable galpy** — the whole existing
suite passing under jax and torch, the xfail **ledger emptied**, and **no numpy
islands**. Aim for all tests; get as many as possible.

Running log, newest section appended at the bottom. Written so it can be
summarised later without re-reading a two-week trace: every entry states what
was done, what was *measured*, and what is still open.

---

## Ground rules for this stretch

| | |
|---|---|
| **Target branch** | `feat/backends-develop` — I open PRs there and squash-merge them myself |
| **Review model** | Deferred: Jo reviews the accumulated branch on return and rebases onto `feat/backends` |
| **Pace** | **One PR at a time.** Drive to green, merge, then start the next |
| **`feat/backends`** | Jo's alone — the `Agent` ruleset's `update` rule makes any ref update there owner-only |
| **Workflows** | I **cannot** touch `.github/workflows/**` (token lacks `workflow` scope). Never put a `.github/` change in a PR — it makes the PR unmergeable by me, and only at merge time |
| **Branch names** | Never `feat/backends*` (ruleset). Use `backend-jit/…`, `fix/…` |
| **numpy** | Byte-identical. Any deviation gets flagged, never quietly shipped |
| **Hard cases** | Narrowly-scoped skip + written reason + note in the final report; keep going. Not a revived ledger |
| **Tests** | Meaningful assertions only — no `isfinite`/nonzero smoke checks; tolerances near the numerical limit, and any loose tolerance carries a stated reason |
| **Every PR** | Self-audit for duplication and placement: general helpers belong in utilities (**search existing backend utils first**), never buried in a specific potential/df module |

CI reality worth remembering: eager jax/torch green is **enforced** by the
all-backend suite on develop PRs. Traced (`--jit`) green is **not** — it runs
only via `workflow_dispatch(jit=true)`, is not attached to a PR, and so gates
nothing. Traced status in any summary is *asserted by me from dispatch runs*,
not enforced by CI. Making it enforced needs a matrix axis = a workflow edit.

---

## 2026-07-26 — setup

**Landed (Jo merged):**
- **#1199** jit/compile test dimension + untraceable `_backend_dtype` fix.
  Blocked for hours on `codecov/patch` that turned out to be a *real* 2-line
  gap, not the usual phantom: the refactor moved "already a backend dtype" out
  of an `except TypeError` into a new `isinstance` guard, leaving the `except`
  reachable only via a bogus dtype *string*. Fixed with direct unit tests in
  `test_backend_dtype_policy.py`. Lesson recorded: **a refactor that relocates a
  case orphans its old branch.**
- **#1197** df coercion boundary for sphericaldf + Quantity passthrough. Its red
  was `test_backend_sphericaldf.py` importing astropy — that shard is
  deliberately astropy-free, so the import is a hard error, not a skip. The
  end-to-end Quantity test moved to `tests/test_quantity.py`, and `build.yml`'s
  quantity shard gained jax. **The move exposed that jax was running float32
  there** (`test_quantity.py` never enabled `jax_enable_x64`, unlike every
  `test_backend_*.py`) — fixed by enabling x64 rather than loosening `rtol`.
  `test_quantity.py` is now the second known place where a backend can run
  without x64 configured by conftest.
- **#1203** CI: all-backend suite enabled for `feat/backends-develop` + the
  `jit=true` dispatch input + shard timeout 90 → 180 min. `feat/backends` was
  pinned in **three** places, not two — push filter, PR filter, **and a
  job-level `if` on `github.base_ref`**. Fixing only the triggers would have
  produced develop PRs that go green while running **zero** jax/torch jobs.

**Still open:** #1200 (torch trace fixes, rebased + `_NOT_TRACEABLE` pruned by
12 verified entries), #1201 (jax trace fixes — **red, see below**), #1202
(suite-wide `--jit` trace mode).

### RazorThin energy-conservation regression in #1201 — unresolved

`test_energy_jacobi_conservation[RazorThinExponentialDiskPotential]` fails on
**plain numpy** shards, every OS and Python version.

Established by controlled experiment (one build, only the potential file
swapped):

| tree | result |
|---|---|
| #1201 head `32a404c0` | **FAILS** |
| same, RazorThin file reverted to `feat/backends` | **passes** |

So the cause is in #1201's `RazorThinExponentialDiskPotential.py`.

But the values are identical. Transcribed both versions of `_evaluate`,
`_Rforce`, `_zforce` and compared: worst relative difference **exactly 0.0**
over R ∈ [0.5, 5] × z ∈ {0, 1e-12, 1e-8, 1e-7, 1e-6, 2e-6, 1e-4, 0.01, 0.1,
−0.05}, spanning the in-plane branch boundary, through the real potential object
with `normalize(1.0)` applied. `_amp` matches to all digits. Public `Rforce`
returns `float64`, same type, in both.

**So: the rewrite changes no value I have been able to sample, yet flips the
test.** Not yet explained. Next step is to instrument the actual failing orbit
(`setup_orbit_energy`, `odeint`, 5001 steps over t=210) and compare forces
*along the trajectory* rather than on a grid I chose — the difference is
evidently at inputs my sampling misses.

Two incidental findings, both worth keeping:
- **RazorThin has no `hasC`**, so `dopr54_c` silently falls back to `odeint`
  ("Cannot use C integration…"). The **Python** forces drive this orbit — which
  is why a Python-only change can break energy conservation at all.
- The `xp.where` rewrite is **not vectorised**: old raises on array input (the
  `if`), new raises a `(100,) vs (N,)` broadcast error. A real gap, but not this
  failure.

**Process note:** I called this a confirmed #1201 regression, then reversed on
the identical-values evidence, then the controlled experiment showed the first
call was right. The reversal was premature — grid agreement is not proof of
equivalence. Reach for the controlled experiment before re-attributing.

### Infrastructure notes

- `jobovy-agent` has `write`, **not** owner. Rulesets are cumulative and had
  `bypass_actors: NONE`; Jo's pushes work by repo ownership. Jo rescoped `feat`
  to `refs/heads/feat/backends*`, so ordinary `feat/*` branches are writable by
  me again.
- Branch **creation** under `feat/backends*` is blocked for me too —
  `required_status_checks` is enforced on create, and the tip's `all-clear` sits
  queued behind the full backend matrix. Jo creates `feat/backends-develop`.
- `gh run rerun` and PR create/merge all work on `repo` scope; only workflow
  *file* writes do not.

---

## 2026-07-26 — develop branch live, #1200 in flight

`feat/backends-develop` created by Jo at `3e1137b1` (post-#1203). I cannot create
branches under `feat/backends*` myself: `required_status_checks` is enforced on
*create*, and the tip's `all-clear` sits queued behind the full backend matrix.

**★ Verified the #1203 workflow change end-to-end.** #1203 was only ever
exercised with base `feat/backends`, which satisfies the first half of each `||`
— so the `feat/backends-develop` disjuncts had never been evaluated, and a
silent-skip would have shown up as a develop PR going green with **zero**
jax/torch jobs. Measured on #1200 (the first develop PR):

    BACKEND JOBS ON A DEVELOP PR: 44

Same count as a `feat/backends` PR. The develop path is real, not just
syntactically valid. This is the check to repeat if the workflow is ever touched
again — "the PR went green" is not sufficient evidence on its own.

**#1200** (torch trace fixes) retargeted to develop and rebased cleanly: 2
commits, 7 files, disjoint from #1197 and #1203. Force-push works again now that
Jo rescoped the `feat` ruleset to `refs/heads/feat/backends*`. Re-running its
tests locally too, since its earlier 468-pass run predated #1197's
coercion-boundary landing underneath it.

Order from here: #1200 (also proves the squash-merge path under my token — the
last untested link), then #1202 with its now-redundant `ci:` commit dropped
(#1203 carries it), then #1201, which needs real debugging rather than a rebase.

### RazorThin regression — SOLVED (root cause)

**The `if` was load-bearing as a VECTORISATION GUARD, not just a branch.**

`Orbit.E(ts)` first attempts a *vectorised* Φ call with shape `(N,1)`:

| version | `_evaluate` calls during `E(ts)` | what happens |
|---|---|---|
| old | **52** = 1 × `(51,1)` + **51 scalar** | the `(51,1)` call **raises** (`if xp.abs(z) < 1e-6` → "truth value ambiguous"); galpy catches it and falls back to per-point scalar calls → correct |
| new (#1201) | **1** × `(51,1)` | `xp.where` **succeeds** on arrays — and is wrong |

Wrong because `(ks + R)` broadcasts `(51,1)` against the 100 GL nodes → `(51,100)`,
and `xp.sum(weights * evalInt)` then reduces over **both** axes, collapsing the
data axis as well as the quadrature axis. Every point gets the same scalar Φ; E
still varies only through kinetic energy. Measured: `(std/mean)²` = 3.75e-13
(old, passes) vs 1.97e-07 (new, fails); trajectories `R(t)`, `z(t)` **bit-identical**,
so the forces were never at fault.

**Why every value comparison agreed.** Scalar calls are identical in both — I
verified `_evaluate`/`_Rforce`/`_zforce` to 0.0 relative difference over
R ∈ [0.5, 149] and z ∈ [0, 10], including exactly z=0 and both sides of the 1e-6
boundary. The change was never to a *value*; it was to **which call path galpy
takes**. A grid of scalar probes structurally cannot find this.

**Generalise:** in this codebase a data-dependent `if` may be silently acting as
"I am scalar-only", with a vectorised fallback upstream. Replacing it with
`xp.where` for traceability removes that signal and turns a loud failure into a
silent wrong answer. **Whenever converting `if` → `xp.where`, check what happens
on ARRAY input, not just that scalars still match.**

**Fix direction** (for #1201, not yet written): broadcast the quadrature nodes on
a *new trailing axis* and reduce only that axis — `R[..., None]`, `z[..., None]`,
`xp.sum(..., axis=-1)` — which makes the function genuinely vectorised (closing
the separate "not vectorised" gap noted earlier) and keeps the output shape equal
to the input's. **Must then check numpy byte-identity**: the old path summed 100
contiguous float64 in a 1-D `sum`; the new one reduces the last axis of a
`(N,1,100)` array. Expected identical (same pairwise algorithm over the same
contiguous 100 elements) but this MUST be measured against the scalar path, not
assumed — the whole point of this bug is that plausible-looking equivalences were
not equivalent.

### #1200 MERGED — pipeline proven end to end

`feat/backends-develop` → **59c9ae2c**. Merged by **jobovy-agent**, which settles the
last unverified link: I *can* squash-merge into develop under my own token (develop
lacks the `update` rule that reserves `feat/backends` to the owner). Full working
loop confirmed: push a non-`feat/` branch → PR into develop → 44 backend jobs →
green → self-merge. Only workflow edits still need Jo.

Final CI: **154/154**. Content: `under_trace` + `untraceable_setup` in
`_namespaces.py`; AnyAxisym's scipy-quad dispatch and MultipoleExpansion's two
lazy setup builders made torch.compile-safe; **12** verified `_NOT_TRACEABLE`
torch entries pruned (AnyAxisym ×4, Multipole ×4, DiskMultipole ×4).

**Second real codecov/patch red in a row** (92%, exactly `_namespaces.py:94`
and `:128`). Both were the `"torch" not in sys.modules` fallbacks — the
**numpy-only user's path**, unreachable on the coverage shard because torch is
always imported there. Same structural shape as #1199's gap. Fixed with a test in
`test_backend_trace.py` (the file that already owns those helpers) that hides
torch via `patch.dict` and asserts both fallbacks fire **and that neither helper
imported torch to answer** — the actual contract, not just line-touching.
Updated prior: codecov/**project** reds are usually noise; codecov/**patch** reds
have now been genuine 2 for 2. Always pull the missed lines before dismissing.

**Audit for the RazorThin `if → xp.where` class: clean.** #1200 contains no such
conversion — AnyAxisym only *widens* an existing `if not (...)` with
`or under_trace(R, z)` (the `if` stays an `if`, so scalar-only semantics and any
upstream vectorised fallback are untouched, and it carries
`check_potential_inputs_not_arrays` besides); MultipoleExpansion just decorates
two setup builders. This audit is now a standing pre-merge step for #1202.

Process note: ruff-format silently aborted the first commit attempt and the
following `git push` reported "Everything up-to-date". Check `git log` after a
pre-commit-hook failure — the push output alone looks like success.

### #1201 complete (local) — RazorThin fixed, ledger moved

Rebased onto develop (it was still on the pre-#1199 base, where the jit harness
was named `test_backend_jit.py` — which is why its `_NOT_TRACEABLE` entries had
never been checked). 3 commits, 7 files.

**RazorThin vectorisation fix** (`_evaluate`, `_Rforce`, `_zforce`): broadcast the
GL nodes on a new trailing axis, reduce only that axis. Verified:

| check | result |
|---|---|
| failing energy test | passes |
| byte-identity vs **pre-#1201** scalar path | **9/9 bitwise**, max diff 0.0, R ∈ [0.8,149], both branch boundaries |
| vectorised (N,1) and 1-D vs per-point | exactly equal |
| new regression test without the fix | fails all 3 methods |

Not just a restoration of the old accidental guard — RazorThin is now *genuinely*
vectorised, which neither version was.

**Ledger: −4 jax, +1 torch.** The 4 jax entries (RazorThin ×3, TwoPowerTriaxial
`__call__`) now trace AND match eager, so the harness demanded their removal.
The +1: `TwoPowerTriaxial.__call__` under torch. Routing `_psi` through the
backend `hyp2f1` is REQUIRED for jax (scipy's converts a traced value to numpy),
and under inductor something in that path yields `inf` where eager is finite —
controlled experiment: passes with the pre-#1201 file, fails with it. **Same
defect already listed for `TwoPowerSpherical.__call__`, which also goes through
`hyp2f1`** — exposed here, not introduced.

**Next PR is therefore well-defined:** find and guard the `inf`-producing path in
the `hyp2f1` route; it clears BOTH torch entries. Note the `_fallback/hyp2f1.py`
tiny-|z| site is *already* double-`where` guarded (`w_for_int = 1.0`), so the
culprit is elsewhere — the router dispatch or an `m → 0` path in `_psi`. Needs
its own investigation; deliberately kept out of #1201 to keep it scoped.

**★ NEAR-MISS worth remembering.** The controlled experiment used
`git checkout <ref> -- <file>`, which **stages** as well as writes. Copying the
file back restored only the worktree, so the next `git add <test> && git commit`
**silently reverted #1201's headline `_hyp2f1` change**. Tests still passed —
scipy's hyp2f1 traces fine under torch, it just isn't jax-traceable — so nothing
would have caught it before review. Caught by `git show --stat` after committing;
amended. Now a memory: **check `git show --stat --format= HEAD` after every
commit.** Second time a commit has gone wrong quietly (ruff-format aborting a
commit, then `git push` reporting "Everything up-to-date", is the other).

**Flake taxonomy so far** (all handled without escalation): RTD duplicate builds
(self-resolving — do NOT retrigger), SIMBAD/`from_name` `DALServiceError`
(re-run), notebook `known_objects.ipynb` (re-run). Against those,
**codecov/patch reds have been genuine 2 for 2** (#1199, #1200) while
codecov/project reds have been noise — pull the missed lines before dismissing a
patch red.

### hyp2f1 under torch.compile is broken EVERYWHERE — existing diagnosis was wrong

Measured (TwoPowerTriaxial defaults, `z = -m/a`), eager vs `torch.compile`:

| m | hyp2f1 eager | hyp2f1 compiled | _psi eager | _psi compiled |
|---|---|---|---|---|
| 1e-8 | 1 | **inf** | -39.2654 | **inf** |
| 1e-4 | 0.999987 | **inf** | -38.8227 | **inf** |
| 0.1 | 0.986902 | **inf** | -25.313 | **inf** |
| 1 | 0.886838 | **inf** | 0.390708 | **inf** |
| 5 | 0.642699 | **inf** | 25 | **inf** |

**Not a degenerate-input problem.** The standing comment on the
`TwoPowerSpherical.__call__` torch entry says "the compiled graph evaluates the
DEAD side of an xp.where (the alpha/beta special-case branch), which is singular
at these parameters" — and I repeated that framing in #1201's new entry. The
measurement refutes it: compiled `hyp2f1` is wrong at EVERY input, including
benign ones far from any branch boundary. Both ledger entries share this one
cause, and it is a plain correctness bug in the fallback under inductor, not a
guard that needs tightening.

Also note `_fallback/hyp2f1.py`'s tiny-|z| site is already correctly
double-`where` guarded (`w_for_int = 1.0`), so that is not it either.

**Next PR (well-defined):** find why `hyp2f1_fallback` miscompiles under
inductor and fix it. Suspects, in order — the `X = xg**k` regularising power
(k up to 12) feeding `T ** (B - 1.0)` with negative exponent, where an
underflowed T gives `0 ** -0.5 = inf`; and `xp.expm1`/`xp.log1p` lowering.
Clears BOTH torch entries and makes two potentials traceable under torch.
Fix the misleading comment on the TwoPowerSpherical entry at the same time.

#### Exact cause: inductor flushes a denormal to zero, then a negative power blows up

Bisected `hyp2f1_fallback` piece by piece (TwoPowerTriaxial defaults ->
B=0.5, A=2, q=1, **k=12**, so the exponent B-1 = **-0.5**):

| piece | eager range | compiled range | |
|---|---|---|---|
| `X = xg**k` | [2.03e-49, 0.9989] | [2.03e-49, 0.9989] | ok |
| `T = expm1(X*L)/w` | [**1.555e-49**, 0.9987] | [**0**, 0.9987] | **flushed** |
| `T**(B-1)` | [1.001, **2.536e+24**] | [1.001, **inf**] | **inf** |
| `(1-T)**(q-1)` | [1,1] | [1,1] | ok |
| `(1+wT)**(-A)` | [0.3604, 1] | [0.3604, 1] | ok |

k=12 pushes the first GL node's `T` into the denormal range; inductor flushes it
to exactly 0, and `0 ** -0.5 = inf`. Confirms the corrected diagnosis: a plain
numerical-robustness bug, nothing to do with dead branches or parameter
singularities.

**The fix is what the code's own docstring already describes.** For small X,
`expm1(x) ~ x`, so

    T^(B-1) * dX = (X L / w)^(B-1) * k xi^(k-1) = (L/w)^(B-1) * k * xi^(kB-1)

and `kB-1 = 5` — benign. The intended formulation never underflows; the
implementation just evaluates `T^(B-1)` (-> 1e-49^-0.5) and `dX` (-> 1e-45)
separately and lets each run to an extreme before multiplying.

**★ Tension to resolve carefully:** folding those factors reorders the
floating-point ops, so the NUMPY path may move in the last ulp. Byte-identity is
a hard rule. Plan: implement, MEASURE the numpy delta against the current
fallback, and if non-zero, surface it for review rather than shipping quietly.
If it is zero (plausible — the dominant nodes are unaffected), it is a pure win:
both torch ledger entries clear and the quadrature stops relying on 1e+24
intermediates cancelling 1e-45 ones.

#### Cause pinned exactly: inductor's FUSED expm1 is lossy for tiny args

Two wrong guesses before the right one — recorded because both were plausible:

1. "denormal flush-to-zero" — wrong. `T = 1.55e-49` is a perfectly NORMAL double.
2. "inductor lowers expm1 as exp(x)-1" — wrong *as tested*: a standalone
   `torch.compile(lambda v: torch.expm1(v))(1e-49)` returns 1e-49, exactly right.

The isolating measurement:

| expression | eager (min over nodes) | compiled |
|---|---|---|
| `xg**k` | 2.02970e-49 | 2.02970e-49 ok |
| `X*L` | 1.03682e-49 | 1.03682e-49 ok |
| **`expm1(X*L)/w`** | **1.55524e-49** | **0** |

So it IS expm1 — but only when **fused**. A lone compiled `expm1` dispatches to
the ATen op and is accurate; inside a fused elementwise kernel inductor codegens
it (effectively `exp(x)-1`), which is 0 for any x below ~1e-16. The standalone
test was simply not representative of the kernel the fallback actually produces.

**Two candidate fixes, differing on byte-identity — decide next tick:**

- **(a) Stable ratio everywhere.** `T = u * E / wb` with
  `E = expm1(u)/u` guarded by its series (`1 + u/2 + u^2/6`) below ~1e-8.
  Robust for every backend, one code path. **But** `u*(expm1(u)/u)` is not
  bit-identical to `expm1(u)` — the numpy path moves in the last ulp. Would need
  to be measured and surfaced for review.
- **(b) Dual path.** Keep `expm1(u)/wb` verbatim on numpy (byte-identical by
  construction) and use the guarded ratio only for non-numpy namespaces. Same
  shape as the existing accepted `numpy=scipy / backend=native` split, so it is
  NOT a numpy island — the backend branch is a real implementation, not a
  quarantine.

Leaning (b): byte-identity is a hard rule and this is a shared special function
many potentials route through, so silently shifting every numpy hyp2f1 consumer
to buy torch traceability is the wrong trade to make unsupervised. Measure (a)'s
delta anyway — if it is exactly zero, (a) is simpler and wins.

#### Measured: the one-path fix is NOT byte-identical -> going with the dual path

Option (a) (`T = u*(expm1(u)/u)/wb`, series-guarded below 1e-8) vs the current
`T = expm1(u)/wb`, on numpy, a,b,c = (0.5, 2.0, 1.5):

| z | current | stable | equal |
|---|---|---|---|
| -1e-12 | 0.99999999999933331 | same | yes |
| **-1e-06** | 0.99999933333394564 | 0.999999333333945**75** | **NO** |
| -0.01 | 0.99339276740631732 | same | yes |
| **-0.5** | 0.76854320901688999 | 0.7685432090168901**1** | **NO** |
| -1, -3.3, -10, -100 | — | same | yes |

**6/8 bitwise identical, 2/8 off by exactly 1 ulp** (max rel diff 1.445e-16).

So (a) would change numpy output for every hyp2f1 consumer. Small, but it is a
real numpy-path change to a SHARED special function, and byte-identity is a hard
rule — not something to relax unsupervised for a torch capability. **Decision:
option (b)**, dual path — numpy keeps `expm1(u)/wb` verbatim (byte-identical by
construction), non-numpy namespaces get the guarded ratio. Precedent: the
accepted `numpy=scipy / backend=native` split, which is explicitly NOT a numpy
island (the backend branch is a real implementation, not a quarantine).

Worth surfacing on Jo's return regardless: if a 1-ulp shift in hyp2f1 were
acceptable, (a) is simpler (one code path, robust everywhere) and would let the
dual path be deleted.

### Correction: the dual path was unnecessary — numpy never reaches this code

The reasoning above rests on a premise I had not checked: that numpy values flow
through `hyp2f1_fallback`. **They do not.** The router
(`galpy/backend/special/_router.py`) dispatches numpy to `scipy.special.hyp2f1`
natively; the fallback exists only because torch lacks `hyp2f1` and jax's is
wrong for z < -1. Verified by monkeypatching the fallback and calling
`gsp.hyp2f1(...)` on a numpy array: **never called**.

Consequences:

- The 1-ulp table above is irrelevant to galpy's numpy output. Byte-identity was
  never at stake, so the constraint that forced option (b) does not exist.
- Worse, `if xp is numpy:` would have been **unreachable dead code** — an
  uncovered line straight into a `codecov/patch` red.

**Final: option (a), single path for all namespaces.** Simpler, no dead branch,
and it drops the `import numpy` the dual path had needed.

Method note: this is the fourth wrong hypothesis on this bug (denormal flush →
refuted; "inductor lowers expm1 as exp(x)-1" → refuted, only the FUSED one is
lossy; "dead side of an xp.where", which was the pre-existing ledger comment I
repeated → refuted; now "numpy byte-identity is at stake" → refuted). Each was
killed by a measurement, none by re-reading the code. The ledger comment being
wrong is the notable one: a plausible-sounding comment survived in-tree and I
propagated it instead of testing it.

### What landed (commit `e8e9f760`, branch `backend-jit/hyp2f1-fused-expm1`)

- `hyp2f1.py`: factor out `expm1(u)/u -> 1`, series below a 1e-8 crossover.
- **Accuracy measured, not assumed**: fallback vs scipy is 1.35e-14 worst-case
  over the realistic grid, for *both* jax and torch. The existing parity test
  ran at `rtol=1e-9` — ~74000x looser than the code's real accuracy — so it was
  tightened to `1e-12` (~70x headroom for cross-platform libm differences).
- New `test_hyp2f1_survives_inductor_fusion`: compares **compiled vs scipy**
  (not compiled vs eager, which would pass if a future change broke both).
  Verified to FAIL without the fix, on the `(0.5, 1.0, 1.5)` case.
- Only the B < 1 cases actually blow up: `T**(B-1)` needs a negative exponent to
  turn `T=0` into inf. `_euler_labeling` gives B=0.5 for exactly one of the six
  parametrised cases; the others have B >= 1 and were never affected.
- Dropped a test I had written that duplicated `test_hyp2f1_value_parity`
  outright (same grid, same cases, same comparison) — tightened the existing one
  instead.
- Hoisted two shared pieces out of `test_backend_potential_jit.py` into
  `conftest.py` rather than copying them into a second file: the private
  `TORCHINDUCTOR_CACHE_DIR` and the `torch.compile` capability probe. The cache
  hoist also closes a latent gap — `--backend torch --jit` compiles too and had
  no private cache, so xdist workers could collide there.

**Milestone in reach**: `_NOT_TRACEABLE` currently holds RazorThin x3 (jax),
TwoPowerTriaxial (jax), TwoPowerSpherical (torch). #1201 clears the four jax
entries and adds TwoPowerTriaxial (torch). This fix clears **both** remaining
torch entries (verified: both compile and match eager). So once #1201 merges and
this rebases on top, **`_NOT_TRACEABLE` is empty** — the entire default-
constructible potential zoo traceable under both `jax.jit` and `torch.compile`.

Sequencing: hold this PR until #1201 merges, then rebase and remove both entries
in one go (they conflict in the same file otherwise).

### Measuring the jax-jit ledger after #1201 (local, no CI dispatch spent)

Ran `pytest tests/test_potential.py --backend jax --jit` in regen mode on the
#1201 worktree (confirmed rebased onto post-#1202 develop, so the comparison is
against the same code base that produced the 100-entry baseline).

**Two methodology traps, both hit:**

1. **`-n 16` inflates the failure count.** First pass reported ~130 failures vs
   a 100 baseline, clustering in the three heaviest tests
   (`poisson_surfdens`/`2ndDeriv`/`evaluateAndDerivs`). That is XLA compile time
   under 16-way CPU contention, not breakage. Re-running one alleged failure
   alone made it pass.
2. **Comparing regen output to the ledger with plain `comm` is wrong.**
   `conftest` matches a ledger entry against EITHER the full parametrised nodeid
   OR the base nodeid with `[...]` stripped, and the committed ledger stores
   base-form entries while regen writes fully-parametrised ones. A literal diff
   manufactures phantom "new" failures — it claimed 65. Applying conftest's own
   matching semantics gives the truth.

**Result after correct comparison: 0 genuinely new failures, 35 ledger entries
no longer failing.** They are exactly the families #1201 fixes (TwoPowerTriaxial
x N, RazorThin, InterpSnapshot, DehnenBar). So #1201 takes the jax-jit ledger
**100 -> 65** on top of emptying `_NOT_TRACEABLE`.

Serial re-verification of those 35 flips is running (parallel runs over-count;
a flip is only real if it fails without the fix). Prune lands as its own PR
after #1201 merges, per the one-PR-at-a-time rule.

### The full causal chain (verified, not inferred)

Chased a `__array_wrap__ ... (Deprecated NumPy 2.0)` warning surfacing from
`TwoPowerTriaxialPotential.py:161` during the torch-compile zoo run. Checked it
against unmodified develop first: **identical there, so not mine.** The cause is
that on develop that line is `from scipy import special` -> `special.hyp2f1`,
i.e. scipy called directly on a torch tensor. That is a real numpy island: it
detaches gradients, forces CPU, warns under numpy 2, and is exactly why
`TwoPowerTriaxial.__call__` carries a jax `TracerArrayConversionError` entry.

So the three states line up:

| | `_psi` calls | jax | torch |
|---|---|---|---|
| develop | `scipy.special.hyp2f1` | ledgered (tracer->numpy) | works, via numpy island (+warning) |
| #1201 | `backend.special.hyp2f1` | **traces** | **inf** (inductor fused expm1) |
| +this fix | same | traces | **correct** |

#1201 removes the island; routing through the router is precisely what exposed
the inductor bug. Both torch entries then clear, and the warning goes with the
scipy call. Nothing further to do here — noted so the warning is not
re-investigated later.

Whole-shard check on the fix branch: `test_backend_potential_jit.py` +
`test_backend_jit_mode.py` + `test_backend_trace.py` = **385 passed, 2 skipped,
4 xfailed** (the 4 = the jax entries #1201 clears). conftest hoist holds.

### Remaining jax-jit work after #1201 (65 entries), scoped into families

| family | entries | |
|---|---|---|
| LogSpiral (Steady/Transient + 5 mocks) | ~28 | **root cause found** |
| Multipole / SCF (nonaxiDisk*) | ~24 | |
| noGL NFW / Triaxial | ~16 | |
| RazorThin residue (Any*, Exp) | ~12 | not covered by #1201's entry-point fix |

**LogSpiral diagnosed** (`SteadyLogSpiralPotential.py:173`), two plain bugs, both
numpy islands rather than jit exotica:

1. `chi = self._alpha * numpy.log(R) - ...` — bare `numpy.log` on a traced array
   -> `TracerArrayConversionError`. Fix: `xp.log`.
2. `if t < self._tform:` ... `elif t < self._tsteady:` — a data-dependent Python
   branch on the traced coordinate `t`, selecting the `smooth` turn-on window.
   Fix: `xp.where`. Both dead sides are safe here (the only division is by the
   nonzero constant `tsteady - tform`), so no dead-branch guard is needed --
   worth stating explicitly since eager `xp.where` evaluates both sides.

That is the next fix PR after the ledger prune: one shared migration clearing
~28 entries across 7 parametrisations. Ordering stays one-PR-at-a-time:
#1201 -> hyp2f1/_NOT_TRACEABLE -> ledger prune (100->65) -> LogSpiral.

### LogSpiral scoping refined: the helpers already exist, only 2nd derivs missed

Looked before writing anything. Both classes ALREADY carry a migrated
time-window helper -- `SteadyLogSpiralPotential._smooth(t)` and
`TransientLogSpiralPotential._envelope(t)` -- and `_evaluate`/`_Rforce`/
`_phitorque` already go through them on the backend namespace.

The three SECOND-derivative methods (`_R2deriv`, `_phi2deriv`, `_Rphideriv`) in
each class were simply never migrated: they re-inline the
`if t < self._tform: ... elif t < self._tsteady:` branch and call bare
`numpy.log/sin/cos`. That is exactly why the failures are
`test_2ndDeriv_potential` / `test_evaluateAndDerivs_potential` and not the
force tests.

So the PR is "finish the migration these two classes already started": route the
2nd derivs through the existing `_smooth`/`_envelope` and the resolved
namespace. **No new helper** -- the right one is already in the class, which is
the whole point of checking first. 6 methods across 2 files, ~28 ledger entries.

### Flip verification closed properly

35 claimed flips: **34 passed + 1 skipped, 0 failures** serially on #1201 (not
just under the -n 16 run that first surfaced them). Confirming the other half of
the claim -- that they FAIL without #1201 -- is running against unmodified
develop now. A flip only counts if both halves hold; passing-with-fix alone has
over-counted before.

### Flip attribution corrected: 29 real, not 35

Ran the 35 candidates against **unmodified develop**. 29 fail there, 6 pass.
So:

- **29** entries are genuine #1201 flips (fail without it, pass with it).
- **6** were **already stale** before #1201 — they pass on develop too, i.e. an
  earlier migration fixed them and the ledger was never refreshed. Still
  prunable, just not #1201's credit:
  `test_DehnenBar_special`, `test_mass_spher_z`, `test_rE_powervc`,
  `test_amp_mult_divide[mockTimeDependentMultipoleExpansion]`,
  `test_potential_at_infinity[mockMultipoleExpansionAxi]`,
  `test_toVertical_toPlanar[DiskMultipoleExpansion]` (note the Multipole
  clustering — consistent with the Multipole work having landed earlier).

Ledger prune is still **100 -> 65**; only the attribution changes.

One methodology note worth keeping: my first classifier over-counted the real
flips as 31 because it credited a per-parametrisation entry whenever ANY sibling
parametrisation failed. Base-nodeid entries and parametrised entries need
different matching rules (base: any parametrisation failing counts; parametrised:
exact match only). The tell was that 31+4 did not reconcile with pytest's own
29 F / 6 dots — when two independent counts of the same thing disagree, the
classifier is wrong, not the run.

So across this stretch the naive numbers were 35 flips and 65 new failures; the
true ones are 29 and 0. Both inflations came from comparison logic, not code.

### Regen-under-xdist DOUBLES every ledger entry (correcting my family table)

The regen output file written by a `-n 16` run contains each nodeid **twice**:
130 lines, 65 unique. pytest_sessionfinish runs in every xdist worker as well as
the controller, so each writes its own share. My earlier "130 failures" and the
family table built from it were uniformly 2x inflated.

The set-based comparisons were unaffected (sets dedupe), so **"100 -> 65" still
holds** -- and is now confirmed twice over: 100 - 35 pruned = 65, and the regen
has exactly 65 unique entries.

Corrected family breakdown of the 65 remaining:

| family | entries |
|---|---|
| LogSpiral (7 parametrisations x 2 tests) | **14** |
| Multipole / SCF (nonaxiDisk*) | 12 |
| noGL NFW / Triaxial | 8 |
| RazorThin residue | 6 |
| long tail | 25 |

Rule for next time: **`sort -u` the regen output before counting**, and prefer a
serial regen when the number itself matters.

### LogSpiral migration done (commit `97f7eb35`, branch `backend-jit/logspiral-2ndderivs`)

Both classes already routed their time dependence through `_smooth(t)` /
`_envelope(t)` in the 1st derivatives. The 2nd derivatives never got converted:
bare `numpy.log/sin/cos/exp`, and in the Steady case the entire `_smooth` body
re-inlined as a Python `if t < self._tform` chain -- 39 duplicated lines that
also happen to be the exact thing a tracer cannot execute.

Fix = call the helper already in the class. **No new helper written**, which was
the point of reading before typing. Net **-30 lines**.

Verified three ways, not one:
- numpy **byte-identical, 27/27** (2 classes x 3 methods x t below/inside/above
  the turn-on window, plus `tform=None`).
- jax `--jit`: **14 fail on develop, 0 with the fix** (ran the exact 14 nodeids
  on both trees, not just the fixed one).
- numpy path: 57 passed, 15 skipped.

Takes the jax-jit ledger 65 -> 51 once it lands. Held unpushed behind #1201 and
the hyp2f1 PR, per one-PR-at-a-time.

### noGL family (8 entries) scoped -- NOT a mechanical numpy->xp swap

`EllipsoidalPotential._forceInt` / `_2ndDerivInt` (the `glorder=None` path, which
is exactly what the `noGL*` test potentials exercise) integrate with
`scipy.integrate.quad`. That is adaptive quadrature driven by a Python callable
on scalar floats -- untraceable by construction, not because of a stray
`numpy.`. The bare `numpy.sqrt` at EllipsoidalPotential.py:520 is a symptom, not
the cause; the docstring already says "numpy-only, deferred to Pspecial".

Checked for an existing utility before designing anything, and there is one:
**`galpy/backend/quadrature.py::quad`** already is the drop-in for
`scipy.integrate.quad(...)[0]`, and already implements the accepted dual path --
numpy/Python-scalar inputs delegate to scipy and stay byte-identical, jax/torch
inputs get fixed-order Gauss-Legendre that is differentiable w.r.t. both the
closed-over parameters and the limits. So this needs **no new helper** either.

**The trap to avoid when implementing:** `quad` dispatches on the *limits and
`args`*. Here the limits are the literals `0.0` and `1.0` and the traced
coordinates are captured in the integrand's *closure*, so a naive swap would
resolve to numpy, silently keep the scipy path, and look like "the fix didn't
work". The coordinates must be threaded through `args=(x, y, z)` (and the
integrand written elementwise in the namespace) for dispatch to see them.

Family queue by size, all verified against the deduped 65:
LogSpiral **14** (done, `97f7eb35`) -> Multipole/SCF **12** -> noGL **8** ->
RazorThin residue **6** -> tail 25.

### Pre-merge audit of #1201 found a live numpy-path bug (pre-existing)

Audited the diff for the `if`-on-array-input class, duplication, and helper
placement, per the standing instruction.

Clean: no bare `numpy.` math left in either production file (the remaining
`numpy.` hits are test-fixture reference values), and the `if self._new:`
branches are a static config flag -- correctly NOT converted to `xp.where`,
since turning a config branch into a select would be wrong.

**But `RazorThinExponentialDiskPotential._R2deriv` still has
`if xp.abs(z) < 10.0**-6.0:`** -- the exact data-dependent-branch-on-a-
coordinate that #1201 removes from `_evaluate`/`_Rforce`/`_zforce`. Measured:

| | `_Rforce(array, array)` | `_R2deriv(array, array)` |
|---|---|---|
| develop | ValueError | ValueError |
| #1201 | **OK** | **ValueError** |

So it is **pre-existing, not a #1201 regression** -- #1201 is a strict
improvement and merges as-is. But this is not merely a jit gap: `R2deriv` with
array R, z raises "truth value of an array ... is ambiguous" for ordinary numpy
users today. Worth Jo knowing on return; it is a real bug on the supported path,
not backend-only.

It also explains the "RazorThin residue 6" ledger entries, which are
`test_2ndDeriv_potential` / `test_evaluateAndDerivs_potential` -- the same
2nd-derivatives-left-behind pattern as LogSpiral.

**Consequence for sequencing:** fold `_R2deriv` into the LogSpiral PR and rename
it "finish the 2nd-derivative migrations": LogSpiral 14 + RazorThin 6 = **20**
entries, one defect class, one rationale. Matches the less-atomic-PR guidance.

---

## #1201 MERGED (squash `e6fa36a3`) -- and `_NOT_TRACEABLE` is now EMPTY

Merged after the audit above. Note on the gate: the rollup showed `all-clear`
CANCELLED, but there were **two** all-clear runs for the head SHA (concurrency
duplicate) and the sibling was SUCCESS -- `mergeStateStatus` was CLEAN, so the
cancelled one was cosmetic. Same shape as the RTD duplicate-build red: check for
a successful sibling run before treating a cancelled required check as a blocker.

Rebased the hyp2f1 branch on top. The conflict in
`test_backend_potential_jit.py` was the expected one and resolved to the
milestone:

```python
_NOT_TRACEABLE = set()
```

Every default-constructible potential in the zoo now traces under **both**
`jax.jit` and `torch.compile`, on all four entry points (`__call__`, `Rforce`,
`zforce`, `dens`). The list started this stack at 5 entries and is the first of
the burndown lists to reach zero.

Wrote it as `set()`, not `{}` -- the latter is a dict, and since every use is a
membership test it would have "worked" while being wrong.

Marker check after the rebase used a strict `^(<<<<<<< |>>>>>>> |={7}$)`; a loose
grep matched RST title underlines in AUTHORS.txt / dfking.rst and looked alarming.

### Correcting my own plan: "RazorThin 6" was two different classes, and `_R2deriv` is not the LogSpiral pattern

Last entry I said to fold `RazorThinExponentialDisk._R2deriv` into the LogSpiral
PR for "20 entries". Reading the code instead of pattern-matching the symptom
shows that was wrong on two counts.

**1. The 6 entries are two unrelated classes.** Only 2 are
`RazorThinExponentialDisk`; the other 4 are
`AnyAxisymmetricRazorThinDiskPotential`, which has its own `_bk_dispatch`
mechanism and needs separate diagnosis. Grouping by the substring "RazorThin"
conflated them.

**2. `_R2deriv` is a DOMAIN check, not a vectorisation guard.** Off-plane has no
closed form -- the else-branch *raises AttributeError*. So unlike
`_evaluate`/`_Rforce`/`_zforce`, there is no second value to select, and
converting it to `xp.where` would be actively wrong: it would turn a loud "not
implemented" into a silent NaN. The LogSpiral fix pattern does not apply.

**3. Fixing the array bug does NOT clear any jit entry**, because
`if xp.all(...)` is still a data-dependent branch on a tracer. Traceability here
needs a raise-vs-NaN semantics decision -- a user-facing API call, left for Jo.

So it is its own small branch (`fix/razorthin-r2deriv-array`, `39a48f41`), scoped
honestly as a **numpy bugfix that clears 0 ledger entries**, not bundled into a
jit PR where the claim would be misleading. Verified: scalar byte-identical,
array in-plane works, off-plane still raises (incl. a mixed `[0.0, 0.5]` array).

Revised: LogSpiral PR = **14** entries (unchanged, pure jit).

### Status board

| item | state |
|---|---|
| #1201 | **MERGED** `e6fa36a3` |
| #1205 hyp2f1 + `_NOT_TRACEABLE = set()` | **OPEN**, CI running |
| ledger prune 100 -> 65 | staged (35 nodeids verified) |
| LogSpiral 2nd derivs (`12cadb8f`) | staged, rebased, 14 entries |
| RazorThin `_R2deriv` array (`39a48f41`) | staged, numpy bugfix, 0 entries |

`_NOT_TRACEABLE` verified empty with no xfail net: **367 passed, 2 skipped, 0
xfailed**.

### Jo's call: NaN off-plane under a trace (and it clears 2 entries, not 0)

Asked whether `RazorThinExponentialDisk.R2deriv` should keep raising or return
NaN off-plane under a trace. Answer: **NaN**.

Implemented as "decide when the domain is decidable, NaN only when it is not",
which is what makes it safe: `test_potential.py` pins
`R2deriv(1.0, 0.1) -> PotentialError`, so NaN could not be unconditional without
breaking the documented API.

`galpy.backend.has_concrete_truth_value` added for this -- in the backend
utilities, not in the potential, because "can I read a bool from this right now"
is the general question any domain check under a trace has to ask. (Checked
first: `_TRACING` exists but only covers galpy's own trace mode, not a user's
`jax.jit(...)` wrapper, which is exactly what the ledger entries hit.)

Off-plane behaviour, measured in every regime:

| input | off-plane |
|---|---|
| numpy scalar | raises PotentialError (unchanged) |
| numpy array | raises |
| jax / torch EAGER | raises (concrete -> decidable) |
| `jax.jit` traced | **NaN**, elementwise on mixed arrays |
| `torch.compile` | raises -- dynamo graph-breaks and decides concretely |

The last two rows differ by framework. That follows from the rule rather than
contradicting it: torch.compile *can* decide, so it does. Documented rather than
smoothed over.

In-plane unchanged: numpy scalar byte-identical, arrays elementwise, traced to
~1e-14 (XLA reassociation).

**Correction to my own estimate:** I predicted this would clear 0 ledger entries
because `if xp.all(...)` is still data-dependent. With the NaN branch there is no
`if` left on the traced path, so both
`test_2ndDeriv_potential[RazorThinExponentialDisk]` and
`test_evaluateAndDerivs_potential[RazorThinExponentialDisk]` now **pass** (0/2
failing). So it is a **2-entry** fix, and the 2nd-derivative theme totals
LogSpiral 14 + RazorThin 2 = **16**.

numpy path re-run: 25 passed, 2 skipped. Commit `4fc4667a`.

### Next PR assembled: 2nd-derivative migrations + ledger 100 -> 49

CI is the scarce resource (~249 checks/cycle, and merging #1201 also kicked off a
push-event run, so #1205 is sitting behind a saturated queue). Rather than three
serialized PRs I combined the two 2nd-derivative fixes with the ledger prune into
one branch (`backend-jit/logspiral-2ndderivs`):

1. Log spirals -- 2nd derivs routed through the existing `_smooth`/`_envelope`
   helpers (14 entries, -30 lines).
2. `RazorThin._R2deriv` -- array input + NaN off-plane under a trace (2 entries).
3. Ledger prune: **51 removed = 35 (#1201's verified flips) + 16 (this PR)**,
   leaving **49**. Arithmetic checks out against the deduped count from two
   directions.

Putting the prune in the same PR is not just convenience: removing an entry makes
the suite ASSERT that test passes, so the prune IS the verification of the fixes.
A removed-but-still-broken entry reds the shard.

Also rewrote the ledger's jax-jit header comment, which had gone stale -- it still
claimed "100 ... they cluster on RazorThinExponentialDisk and TwoPowerTriaxial",
both of which this stack fixed, and pointed at a `_NOT_TRACEABLE` that is now
empty. New header carries the current 49, the per-family residue, and the noGL
implementation trap (`quad` dispatches on limits/args, so coordinates must go
through `args=`, not the closure) so the next person does not rediscover it.

Verification of all 51 un-ledgered nodeids under `--jit` is running.

### Pre-push verification of the combined branch

Beyond the 51 un-ledgered nodeids (50 passed, 1 skipped, 0 failed), ran the five
backend test files that touch the changed potentials -- `test_backend_disk`,
`halobar`, `potential_diff`, `conventions`, `dtype_policy`: **886 passed, 131
skipped, 0 failed**. That check exists because the ledger I verified is
**jax-jit only**, while the LogSpiral/RazorThin edits sit on paths torch uses
too; verifying only the dimension that motivated the change is the exact
shard-context miss to avoid. Full numpy `test_potential.py` still running.

### #1205 codecov: not treating it as real yet

Both `codecov/patch` and `codecov/project` are red at **9 of ~149** checks
complete, i.e. almost no coverage has uploaded. Same shape as #1201, where
project was red on partial data and went green on its own once the shards
finished. Re-evaluate when uploads are in; `patch` red at the end WOULD be real
(that is how #1199's genuine 2-line gap showed up), so this is deferred, not
dismissed.

Runner starvation explained: a `feat/backends-develop` push-event run
(All-backend tests + Windows + Linux/Mac) is queued alongside #1205's 149 checks,
so the two are competing. Nothing to fix -- just slow.

### Combined branch fully verified locally (ready to push behind #1205)

| check | result |
|---|---|
| 51 un-ledgered nodeids, `--jit` | 50 passed, 1 skipped, **0 failed** |
| 5 backend files (disk/halobar/potential_diff/conventions/dtype_policy) | 886 passed, 131 skipped, **0 failed** |
| full numpy `test_potential.py` | 1224 passed, 102 skipped, **0 failed** |

Commits: `12cadb8f` (log spirals) + `29fabaee` (RazorThin) + `ae35c47b` (ledger
100 -> 49).

### noGL: recipe validated by experiment before touching the code

Read the three `scipy.integrate.quad` sites (`_potInt`, `_forceInt`,
`_2ndDerivInt`, all on the `glorder=None` path) and found the design is simpler
than expected: **`galpy.backend.quadrature.quad` already does the
numpy->scipy / backend->GL dispatch internally**, so it SUBSUMES the branch
rather than needing one added.

Validated standalone, without editing anything:

| path | value |
|---|---|
| `scipy.integrate.quad` | 0.2147102810534048 |
| `bquad` on numpy | 0.2147102810534048 -- **bit-identical**, 3/3 components |
| `bquad` on jax | 0.21471028105340476 (~2e-16 rel) |

numpy byte-identity is structural here (quad delegates to scipy), not luck, and
the fixed-order GL matches adaptive quad to machine precision on this integrand.
Dispatch does follow `args`, as documented.

So the remaining work is mechanical: rewrite each integrand as `f(s, x, y, z)`
in the namespace, thread `xp` into `_forceInt`/`_2ndDerivInt` (only
`_potInt`/`*_all` take it today), and swap
`integrate.quad(f, 0, 1)[0]` -> `bquad(f, 0, 1, args=(x, y, z))`. Deliberately
stopped at the validated recipe rather than half-applying a 3-site edit deep in a
context window; the next wake does the edit with the numbers above as the
acceptance test.

### noGL implemented (commit `f3a5bba9`, branch `backend-jit/nogl-quadrature`)

Applied the validated recipe to all three `glorder=None` sites (`_potInt`,
`_forceInt`, `_2ndDerivInt`): `integrate.quad(f, 0, 1)[0]` ->
`bquad(f, 0, 1, args=(x, y, z))`, integrands rewritten in the namespace, and `xp`
threaded into `_forceInt`/`_2ndDerivInt` (which previously took no namespace).

**No new helper** -- `galpy.backend.quadrature.quad` already was the drop-in and
already implemented the dual path. The trap I had flagged in advance was real and
worth flagging: `quad` dispatches on the LIMITS and `args`, and here the limits
are literals while the coordinates sit in the integrand's closure, so passing
them only via the closure would have silently kept the scipy path and looked like
the change did nothing.

Verified:

| check | result |
|---|---|
| numpy byte-identity (TriaxialNFW oblate+triaxial x 3 positions x Phi/Rf/zf/R2, glorder=None) | **24/24 identical** |
| the 8 ledgered noGL entries under `--jit` | **0/8 failing** |
| ellipsoidal-family tests (backend_ellipsoidal + test_potential -k) | 907 passed, 7 skipped |

Ledger effect once it lands: **49 -> 41**. Not pruned in this branch yet, because
the prune has to sit on top of the ls2 prune or the two collide in
`backend_xfail.txt`; it happens on rebase after ls2 merges.

Left the file's other `integrate.quad` (the `m**2 * mdens` mass integral) alone --
separate concern, already flagged in its own comment.

### #1205 codecov reads as partial-upload, quantified

`codecov/patch: 0.00% of diff hit`, `codecov/project: 81.66% (-18.30%)` at ~55 of
149 checks. **0.00%** of the diff hit is the tell: the changed lines are covered
by `test_backend_special.py`, which simply has not uploaded yet. A genuine gap
looks like #1199's "2 specific lines missed", not "nothing in the diff counted".
Still deferred, not dismissed.

### Multipole/SCF (~12 entries): the LIBRARY is fine, the test fixture is not

Diagnosed `test_2ndDeriv_potential[nonaxiDiskMultipoleExpansionPotential]` and the
traceback points at **tests/test_potential.py:12070** -- the mock's own
constructor args, not galpy:

```python
dens=lambda R, z, phi: 13.5 * numpy.exp(-3.0*R) * numpy.exp(-27.0*numpy.fabs(z)) + ...
Sigma=[lambda R: numpy.exp(-3.0*R), ...]
hz=lambda z: 13.5 * numpy.exp(-27.0 * numpy.fabs(z))
```

These are **user-supplied callables**. galpy cannot make an arbitrary numpy-only
lambda traceable -- if a user hands `MultipoleExpansion` a function written in
raw numpy, tracing it is impossible by construction, and that is correct
behaviour, not a gap.

Proved the library side is sound by building the same potential twice:

| user callables | traced |
|---|---|
| numpy-only lambdas (what the fixture uses) | TracerArrayConversionError |
| namespace-agnostic lambdas | **traced == eager**, match to 1e-9 |

So `DiskMultipoleExpansionPotential` already traces end to end; the ledger entry
measures the fixture, not the code.

**Consequence for the roadmap:** galpy's real traced coverage is better than the
49-entry ledger suggests -- ~12 of those 49 are fixture-limited. The fix is a
test-side one (rewrite the mock lambdas namespace-agnostically, which keeps numpy
byte-identical since `xp.exp` IS `numpy.exp` for numpy input), and it is
legitimate rather than ledger-gaming: it makes the test actually exercise the
library instead of tripping over its own scaffolding.

Worth stating plainly for Jo: a ledger entry can encode "our test is written in
numpy", not only "galpy cannot do this". I did not assume that -- I built the
agnostic-callable version and measured it.

### Multipole/SCF fixture fix written (on the noGL branch for now)

Replaced the numpy-only lambdas in BOTH `nonaxiDiskSCFPotential` and
`nonaxiDiskMultipoleExpansionPotential` with one shared helper,
`_exp_disk_profile_callables(thp)`, written namespace-agnostically
(`get_namespace(R).exp(...)` etc.).

Two birds: it makes the fixtures traceable AND removes a duplication that was
already there -- the two mocks carried byte-identical 20-line lambda blocks. The
helper's docstring states why this is a fixture fix and not a galpy gap, so the
next reader does not re-diagnose it as a library problem.

Note `numpy.fabs` -> `xp.abs`: identical values for real input, and `xp.exp` IS
`numpy.exp` on the numpy path, so eager results should be unchanged -- being
verified rather than asserted (numpy run in flight alongside the jit run).

Method note: the first attempt to run the 13 nodeids under `--jit` died at the
2-minute foreground cap. SCF/Multipole XLA compiles are slow; these have to go to
the background runner. Not a failure signal -- worth not misreading as one.

### Multipole/SCF fixtures: 12/13 cleared by the shared helper, 13th was a different site

Running the 13 ledgered Multipole/SCF nodeids with the agnostic fixtures:
**12 pass, 1 still fails**. The holdout is
`test_DiskSCFPotential_againstDoubleExp`, which is not one of the mock classes at
all -- it is a standalone test building its own `DiskSCFPotential` with its own
inline numpy lambdas. Same defect, different site; the class-level fix could not
have covered it, and grouping it under "Multipole/SCF mocks" hid that.

Fixed it inline (3 lambdas) rather than reaching for
`_exp_disk_profile_callables`: that helper returns the LIST-valued
`Sigma`/`dSigmadR` shape for the two-component mocks, while this test passes
scalar `Sigma_amp=1.0` with single callables. Reusing it here would have meant
bending the helper to fit a caller it was not shaped for -- worse than three
explicit lines.

`get_namespace` was only imported locally inside one test, so it is now a
module-level import next to `as_numpy` (the file already imports galpy at module
scope, so no import-timing change to the conftest units toggle).

**13/13 cleared** after fixing the standalone site too (commit `067fee26`). The
noGL branch now carries two clearly separated commits -- `f3a5bba9` (library:
EllipsoidalPotential quadrature) and `067fee26` (tests: agnostic fixtures) --
so the library change and the test change stay reviewable apart even though they
ship as one PR to save a CI cycle.

Projected ledger after the three staged PRs land: 100 -> 49 -> 41 -> **28**.

---

## #1205 MERGED (`2bdc6c7b`) -- `_NOT_TRACEABLE` empty on develop

Audited before merging: one production file changed (hyp2f1), no coordinate `if`
left in it (the branch became `xp.where`), and `test_backend_potential_jit.py`
SHRANK because the conftest hoist removed duplication rather than adding any.

## #1206 OPEN: 2nd-derivative migrations + ledger 100 -> 49

Three separately-reviewable commits (log spirals / RazorThin R2deriv / prune).
Pushed only after the full local battery: 51 un-ledgered nodeids under `--jit`
(50 passed, 1 skipped), 886 backend-file tests incl. torch paths, and the full
numpy `test_potential.py` (1224 passed). Early CI: 6/151, codecov red on partial
upload -- the now-familiar pattern, ignore until uploads land.

Also confirmed the fixture change leaves numpy alone: the DiskSCF/Multipole
numpy tests give **96 passed, 3 skipped**.

Staged behind it, both fully verified:
* `f3a5bba9` noGL quadrature -- 24/24 byte-identical, 8/8 entries
* `067fee26` DiskSCF/Multipole fixtures -- 13/13 entries

Projected: 49 -> 41 -> **28**.

### AnyAxisymmetricRazorThinDisk (4 entries): a DIFFERENT failure class -- wrong VALUE, not untraceable

Diagnosed. It traces fine; it produces a **numerically wrong answer**. The test
fails on `R2deriv` vs the finite difference of `Rforce` with rel. diff ~1.0
(traced `-4.8e4` where the FD reference is O(1)), at (R, z) = (0.5, **0.0**).

Cause: the eager path is

```python
integrate.quad(r2derivint, 0, 2 * R, points=[R])   # + "probably divergent" warning
```

i.e. adaptive quadrature with an **interior singularity declared** at `s = R`.
Fixed-order Gauss-Legendre cannot represent that, so the backend path integrates
straight through the singularity and returns garbage.

**This is the important part: the noGL recipe must NOT be generalised here.**
`bquad` was the right answer for EllipsoidalPotential because that integrand is
smooth on (0,1); here the integrand is singular at an interior point and needs
the interval split at R with an appropriate transform (or a different scheme
entirely). Applying the same swap would silently produce wrong numbers rather
than an error -- the worst failure mode.

Also worth noting the general lesson: everything cleared so far failed LOUDLY
(TracerArrayConversionError / TracerBoolConversionError). This one is the first
that traces and lies. Any future "make it traceable" change on an integral needs
a value check against the eager result, not just "does it trace".

Deferred rather than attempted: it is a genuine numerical-methods task, not a
mechanical migration, and 4 entries do not justify rushing it deep in a context
window.

### The "long tail" is not a tail: 18 of 24 are ONE library site

Grouped the 65 post-#1201 entries properly instead of eyeballing:

| group | n |
|---|---|
| log spirals | 14 |
| Multipole / SCF | 13 |
| noGL | 8 |
| AnyAxisymmetric (parked, wrong-value class) | 4 |
| RazorThin R2deriv | 2 |
| **"tail"** | **24** |

and then grouped the tail by TEST function rather than by potential -- which is
where the structure was hiding:

| test | n |
|---|---|
| **`test_poisson_surfdens_potential`** | **18** |
| six one-offs (Cautun20, DehnenBinney98, ExpDisk_special, McMillan17, mass_spheroidal, plotting) | 6 |

Grouping by *potential* made it look like 12 unrelated potentials; grouping by
*test* showed one shared cause. It is `Potential.surfdens`'s Poisson fallback
(`galpy/potential/Potential.py:731`): `numpy.fabs(z)` on a traced z, and
`integrate.quad(..., -numpy.fabs(z), numpy.fabs(z))`.

Fixed with `bquad` + `xp.abs`. This is the textbook case for it -- the LIMITS are
the traced coordinate, and bquad dispatches on the limits, so nothing has to be
threaded through `args=`. Beyond the ledger it makes `surfdens` differentiable
and GPU-capable for every potential that has no analytic `_surfdens`.

numpy byte-identity: **24/24** across 8 potentials x 3 positions with
`forcepoisson=True`. Traced run in flight.

### CORRECTION: the surfdens fix cleared 3 of 18, not 18 -- and I reverted it

Last entry I projected "ledger 100 -> ~10" on the assumption that one site
explained all 18 `test_poisson_surfdens_potential` entries. **Measured: 3 of 18.**
The projection was wrong and is withdrawn.

The `bquad` swap does make `surfdens` traceable -- the 15 residual failures are no
longer TracerArrayConversionError, they are **accuracy** failures. Hernquist at
(R, Z) = (0.5, **10.0**): rel. diff 3.6e-4 against a 1e-8 tolerance.

Measured the cause rather than guessing at it:

| Z | n=50 | n=100 | n=200 | n=400 |
|---|---|---|---|---|
| 2 | 2.6e-10 | 5e-16 | — | — |
| 10 | 3.0e-02 | **3.7e-04** | 2.7e-08 | 2.6e-14 |

The default order is 100, which reproduces the observed 3.66e-4 exactly. The
required order **scales with the integration range**: fine at |z|~2, badly short
at |z|~10.

**So a fixed order cannot be correct here.** Any constant I choose is accurate
inside whatever envelope I tested and *silently wrong* outside it -- which is
exactly the failure mode I wrote a warning about one entry earlier, for
AnyAxisymmetric. Shipping it would have traded a loud xfail for quiet numerical
error across every potential without an analytic `_surfdens`.

**Reverted.** The branch keeps only what is verified: noGL (`f3a5bba9`, 24/24
byte-identical, 8/8) and the fixtures (`067fee26`, 13/13).

Doing this properly needs a range-aware scheme (order scaled to the range, or a
transform that handles the decaying tail) -- a numerical-methods decision with a
real accuracy/cost tradeoff, not a mechanical swap. Left for Jo rather than
decided unilaterally.

The general rule now has two confirmations: **an integral that starts tracing is
not thereby correct.** Both times the value check was what caught it.

### #1206 red was infrastructure, not code

`build (ubuntu-latest, 3.14, test_actionAngleTorus/test_conversion/...)` went red.
Checked the failing STEP rather than assuming: it is **"Install lcov"** -- an apt
step -- not a test. Transient runner/mirror flake, unrelated to the diff.

`gh run rerun --failed` refuses while the run is still in progress, so a watcher
is armed to fire it as soon as the run completes. Worth remembering: the rerun
has to wait for the whole run, not just the failed job.

### AnyAxisymmetric un-parked: diagnosed, and the purpose-built helper already exists

Earlier I parked this as "a genuine numerical-methods task". Looking properly, it
is much better defined than that.

`_R2deriv_gl` -> `_bk_split_quad` already has the right STRUCTURE:

```python
fixed_quad(xp, integrand, 0, R)  +  fixed_quad(xp, integrand, R, 2R)
                                 +  fixed_quad_semiinfinite(xp, integrand, 2R)
```

so the singularity at `a = R` already sits on a panel boundary and the infinite
tail is already handled. What it does NOT do is cluster nodes toward that
boundary -- it uses plain GL panels.

`galpy/backend/quadrature.py::transformed_quad(..., interior_point=c)` is built
for exactly this, and its docstring names the case: *"the disk `a = R` pattern"*.
It splits at `c` AND applies the `X = xi**k` boundary-layer clustering toward
`c`.

Corroborating evidence that this is the right diagnosis: `Phi` and `Rforce` use
the SAME `_bk_split_quad` and their tests pass; only `R2deriv` fails -- the most
singular of the three integrands, i.e. exactly the one that needs the clustering.

So the fix is to route the two finite panels through `transformed_quad` with
`interior_point=R` rather than plain `fixed_quad`, and validate against scipy.
Not attempted this wake (deep in context, and it changes shared numerics used by
Phi/Rforce too, so it needs a value check on all three, not just R2deriv).

Note this also refines the surfdens verdict: the blocker there is a genuinely
range-dependent order, with no interior singularity to exploit -- so
`transformed_quad` does NOT rescue it. Those two parked items are different
problems and should not be conflated.

### CORRECTION: AnyAxisymmetric is NOT a bug -- it is a documented, deliberate tradeoff

Last entry I wrote "un-parked: diagnosed, and the purpose-built helper already
exists", planning to route `_bk_split_quad` through
`transformed_quad(interior_point=R)`. **That was wrong.** I had read the helper
but not the rationale comment six lines above it:

> A backend-array input routes here. When a gradient is actually being taken --
> a jax/functorch tracer, or a grad-tracking torch tensor -- the compute is
> in-backend fixed-order Gauss-Legendre so it differentiates. A PLAIN concrete
> backend scalar instead reuses scipy's accurate adaptive quad (byte-consistent
> with the numpy path; **GL cannot resolve the a=R principal value / small-z
> sheet structure** that the derivative tests FD-probe at z~1e-8, and **cannot
> form the z=0 Hadamard-type second-derivative**) ...

So the split is deliberate: concrete -> scipy for accuracy, tracer -> GL for
differentiability, with the accuracy loss at the singular configurations
knowingly accepted. Under `--jit` the input is a tracer, so it takes the GL path.
And the failing point is (R, Z) = (0.500, **0.000**) -- precisely the z=0
Hadamard-type second derivative the comment names.

Node clustering does not fix that. A Hadamard finite-part integral is not merely
a "needs more nodes near the singularity" problem; clustering does not make an
ordinarily-divergent integral converge. My proposed fix attacked a problem the
author had already judged infeasible for GL, and I would have discovered that
only after writing and validating it.

**Conclusion: those 4 entries are correctly ledgered.** They encode an accepted
tradeoff, not a defect. The right action is to say so in the ledger rather than
to "fix" them -- and NOT to spend a PR on it.

Lesson, since this is twice now in two wakes: read the rationale next to the code
before proposing to change it. The first miss (surfdens) cost a revert; this one
would have cost a wasted implementation. Both were caught before shipping, but
the cheaper catch was reading.

### #1206 codecov: this one WAS real -- and it caught a genuine hole in my testing

Once all uploads landed, codecov read `patch: 91.17% of diff hit (target
99.96%)`, `project: 99.95% (-0.01%)`. That is the genuine-gap signature. The
distinction that has held all stack long:

* `0.00% of diff hit` + double-digit project drop = **nothing uploaded yet**
* a specific shortfall like 91.17% + -0.01% = **real**

The uncovered lines were the branches that only run under a TRACE: `_R2deriv`'s
`xp.where(...)` NaN return and `has_concrete_truth_value`'s `except`. The
coverage-producing job runs eager, so nothing reached them. Root cause on my
side: I verified the NaN semantics **by hand** and never converted that into a
test. Codecov found a real hole, not a formality.

Fixed with two tests in `test_backend_disk.py` (where the sibling RazorThin jit
test already lives -- no new file):
* the full domain contract in one test: concrete raises (numpy AND concrete jax
  array), traced in-plane matches eager, traced off-plane NaN, mixed array NaN
  elementwise;
* both directions of `has_concrete_truth_value`, so the check cannot degrade to
  always-NaN (hides bad input) or always-raise (stops tracing).

Verified the tests reach the previously-missed lines rather than assuming: with
only those two selected, 254-263 are covered and the file's remaining misses are
`_surfdens`/`_mass`, untouched here.

The lcov infra flake cleared on rerun (run 30232104865 -> completed/success);
`all-clear` had recorded the transient failure and re-runs with the new push.

### Confirmed the coverage fix can actually land (not just that it looks right)

Before trusting the new tests to move `codecov/patch`, checked that the file
they live in is run by a COVERAGE-producing job -- otherwise the fix would be
invisible no matter how good the tests are (backend-tests.yml uploads no
coverage; only build.yml does).

`build.yml` has a `TEST_FILES: tests/test_backend*.py` shard, its pytest
invocation carries `--cov galpy`, and the matrix entry sets **`REQUIRES_JAX:
true`** and `REQUIRES_TORCH: true`. So `test_backend_disk.py` runs there with jax
present, and my `skipif("jax" not in BACKENDS)` test is not silently skipped.

Current reading (`patch 91.17%`, `project 86.18% / -13.79%`) is still on
incomplete uploads -- the large project drop is the tell -- so the unchanged
patch number is not yet evidence either way. Waiting for settle rather than
reacting to a mid-upload number, which is the mistake this stack has now taught
three times.

### #1206 coverage fix WORKED: `codecov/patch 100.00% of diff hit`

After the full upload set landed: **`patch: 100.00% of diff hit` (target 99.96%)**
and **`project: 99.96% (+0.00%)`** -- both green. The two traced-only branches
are now genuinely covered, and the earlier 91.17% was a real gap that the new
tests closed (not a number that drifted back on its own).

That completes the partial-vs-real discrimination for this stack:
`0.00% + double-digit drop` = not uploaded; `91.17% + -0.01%` = real; and after
the fix, 100%.

### New red on #1206: NOT infra this time

`build (ubuntu-latest, 3.10, tests/test_orbit.py -k 'test_energy_jacobi_conservation
or from_name')` failed at step **"Test with pytest"** -- so a genuine test
failure, unlike the lcov flake. GitHub will not serve job logs while the run is
in progress, so rather than wait idle I am reproducing the selection locally.

`test_energy_jacobi_conservation` is the exact test class the RazorThin
vectorisation work touches, so this is checked as a possible real regression from
this branch first, and only treated as the known SIMBAD `from_name` flake if the
energy half passes locally. Assuming "orbit + from_name in the name" => flake
would be exactly the wrong instinct here.

### The orbit red WAS the SIMBAD flake -- but only checkable after ruling out the real half

The shard's `-k` filter matches two unrelated things:
`test_energy_jacobi_conservation or from_name`. Reading the log:

```
FAILED tests/test_orbit.py::test_from_name_values  - pyvo.dal.exceptions.DALServiceError
FAILED tests/test_orbit.py::test_from_name_errors  - pyvo.dal.exceptions.DALServiceError
  https://simbad.cds.unistra.fr/simbad/sim-tap/capabilities: Connection failed
=== 2 failed, 115 passed ===
```

so both failures are the known SIMBAD network flake, and the 115 passes include
the energy half.

Worth recording the sequencing, not just the answer: `test_energy_jacobi_
conservation` is the exact class the RazorThin vectorisation work touches, so
"orbit shard + from_name in the name => flake, rerun" would have been a
coin-flip dressed as a diagnosis. I ran the energy selection locally first (112
passed, 0 failed) and only then treated it as the flake. GitHub also refuses to
serve job logs mid-run, so the local reproduction was the only way to make
progress at that moment rather than idling.

Reran the failed jobs; watcher armed for settle.

---

## #1206 MERGED (`557e0c73`) -- ledger 100 -> 49. #1208 OPEN -- 49 -> 28.

#1206 audit before merge: no residual coordinate `if` in any changed production
file, and the one new helper (`has_concrete_truth_value`) sits in
`galpy/backend/_namespaces.py`, not in a potential module.

Red-classification scorecard for #1206 -- every red was diagnosed, none waved
through:
| red | verdict |
|---|---|
| `Install lcov` step | infra flake -> rerun |
| `codecov` at 0.00% of diff | partial upload -> wait |
| `codecov` at 91.17% of diff | **real gap** -> wrote the missing tests |
| orbit shard (`...or from_name`) | SIMBAD network flake -- but only after running the energy half locally (112 passed) to rule out a real regression |
| `all-clear` | recorded the SIMBAD failure -> rerun after its inputs went green |

Final: 152 checks green, `codecov/patch 100.00% of diff hit`.

**#1208** carries noGL (8) + fixtures (13) + the 49->28 prune, and rewrites the
ledger header to describe the residue honestly: ~15 surfdens (needs a range-aware
quadrature, attempted and reverted), ~4 AnyAxisymmetric (deliberate accuracy
tradeoff, documented in the class), ~9 one-offs. "28 left" should not read as "28
migrations remain".

### Scoping the NEXT frontier: the jit ledger only ever covered test_potential.py

Worth stating plainly, because "ledger 100 -> 28" could be misread as "galpy is
nearly fully jittable". It is not. The `jax-jit` ledger has only ever contained
`tests/test_potential.py` entries -- the trace mode (#1202) shipped with that one
file measured. Orbits, action-angle, DFs and streams have **never been run under
`--jit` at all**, so their traced status is unknown rather than good.

So the potential zoo being ~98% traced is a real milestone but a bounded one. To
size what is actually left I am probing `test_orbit.py --backend jax --jit` on the
`test_energy_jacobi_conservation` selection -- the same 112-test selection I ran
eagerly earlier, so eager-vs-traced is a like-for-like comparison rather than a
new unknown. That answers "is orbit integration close, or is this a fresh
frontier?" for a bounded cost.

Whatever it shows, the honest framing for Jo on return is: **potentials are
nearly done; the rest of galpy under trace is measured-at-zero, not
verified-good.**

### Orbit probe: 112/112 pass under `--jit` -- orbits are ALREADY traced-clean

Ran `tests/test_orbit.py --backend jax --jit -k test_energy_jacobi_conservation`:
**112 passed, 0 failed**, zero regen entries.

That is the same selection I ran eagerly two wakes ago (also 112 passed), so it
is a like-for-like eager-vs-traced comparison, and it covers orbit integration
across many potential/integrator combinations rather than one path.

This meaningfully updates the picture in the OPTIMISTIC direction, and I want to
be as careful stating that as I was stating the pessimistic version: the
`jax-jit` ledger contains no orbit entries not because orbits are broken and
unmeasured, but because -- at least for energy/Jacobi conservation -- they
already trace. The "measured-at-zero" caveat from the previous entry was correct
as a caveat and is now partly discharged by measurement.

Scope of the claim, precisely: one selection of one file. It does not establish
that all of `test_orbit.py` traces, only that orbit integration + conservation
does.

Next probe running: `tests/test_streamspraydf.py --jit` in full -- streams are
the end-goal showcase, so their traced status is the most valuable unknown left.

### Streams probe: `test_streamspraydf.py --jit` fully clean too

38 tests, **0 failures**, no regen entries. Combined with the orbit probe, two
independent areas outside `test_potential.py` trace cleanly on the first try.

Guard against the obvious objection -- "maybe --jit is a no-op on these files":
it is not. The same mode produced 100 failures on `test_potential.py`, so trace
mode demonstrably bites; passing under it is a real signal, not an untriggered
harness.

**Revised picture of the whole mission**, and the most useful thing for Jo to
read on return:

| area | traced status |
|---|---|
| potentials (`_NOT_TRACEABLE`) | **empty** -- whole zoo, jax.jit AND torch.compile |
| potentials (`--jit` suite) | 28 left of 100, and ~19 of those are accepted tradeoffs / need a numerical-methods decision |
| orbit integration + conservation | **112/112 clean** |
| streamspraydf | **38/38 clean** |
| action-angle, DFs, rest of orbit | still unmeasured |

So the remaining risk is concentrated in what has NOT been probed, not in what
has. The next cheap wins are AA and the DF files -- each is one bounded `--jit`
run, and each either confirms clean (no work) or produces a precise entry list
(work, but scoped). That is a much better position than "28 ledger entries" alone
suggests in either direction.

### DF probe: qdf 54/55 -- and the 1 failure names the STRUCTURAL ceiling

`tests/test_qdf.py --backend jax --jit`: 55 tests, **1 failure**
(`test_call_diffinoutputs`).

It is not a migration gap. The test deliberately builds an unbound orbit and
asserts `actionAngleStaeckel` raises `UnboundError`:

```python
try:
    acs = taAS(0.9, 10.0, -20.0, 0.1, 10.0)
except UnboundError:
    pass
else:
    raise AssertionError("Test orbit in qdf that is supposed to be unbound is not")
```

Under a trace the raise cannot fire, because **an exception conditioned on a
traced value is not expressible during tracing** -- the branch that would decide
"this orbit is unbound" has no truth value yet.

This is the same category as `RazorThin._R2deriv`'s off-plane raise, which Jo
resolved with NaN. Naming it as a category is the useful part:

> **Error paths conditioned on traced values are the irreducible residue of
> jit-ability.** They are not "unmigrated code". Each one has exactly three
> honest options: (a) keep the raise and ledger the test (correct behaviour, not
> traceable), (b) adopt a NaN/sentinel convention under trace (what Jo chose for
> R2deriv), or (c) hoist the check to a concrete pre-pass outside the traced
> region.

That reframes the endgame. "All tests passing under --jit" is not reachable by
migration alone for tests that ASSERT an exception -- those need an explicit API
decision, exactly like the R2deriv one. Worth Jo's attention on return, since it
is a policy question rather than an engineering one.

Running tally of areas measured under `--jit`:

| area | result |
|---|---|
| potentials | 28 of ~1300 (from 100); ~19 accepted tradeoffs |
| orbit integration/conservation | 112/112 |
| streamspraydf | 38/38 |
| quasiisothermaldf | 54/55 -- the 1 is an asserted exception |

### Probing the last major unmeasured area: action-angle

`tests/test_actionAngle.py --backend jax --jit`, full file, running. AA is the
biggest remaining unknown and the one most likely to be genuinely hard: it is
full of root-finds, bisections and domain checks (`UnboundError` among them),
which is exactly the category the qdf probe just showed is structurally
untraceable when an exception is asserted.

Prediction, recorded before the result so it can be checked honestly: I expect AA
to be the WORST of the four probes -- not because the migration is incomplete
(Track E moved it to the backend) but because AA is where "raise if
unphysical/unbound" lives. If the failures cluster on asserted exceptions rather
than on tracer errors, that confirms the category finding and means the residue
is a policy decision, not engineering. If instead they are
TracerArrayConversionError/TracerBoolConversionError, that is ordinary migration
work and I can scope it the same way as the potential families.

### AA probe: 30/207 fail -- but 24 are OUT OF SCOPE BY PRIOR DECISION. Prediction was wrong.

`tests/test_actionAngle.py --backend jax --jit`: ~207 tests, **30 failures**.

Breakdown:

| group | n | status |
|---|---|---|
| `actionAngleVerticalInverse*` | **24** | **out of scope by prior decision** -- it stays `_reject_backend`-guarded; the Inverse migration was scoped to Harmonic+Isochrone only |
| `actionAngleVertical_linear_angles` | 1 | |
| non-Vertical | 5 | `AdiabaticGrid_outsidegrid_c`, `Adiabatic_linear_angles_cylsep`, `StaeckelGrid_basicAndConserved_actions`, `Staeckel_unboundr_angles_c`, `physical_vertical` |

**Checking my recorded prediction honestly: it was wrong.** I predicted AA would be
the worst probe and that the failures would be dominated by asserted exceptions.
Instead the dominant cause -- 24 of 30 -- is a scope decision that was already
made and that I had written down in memory
(`verticalinverse-out-of-scope`). I simply did not apply it when predicting. The
exception class does appear (`Staeckel_unboundr_angles_c`,
`AdiabaticGrid_outsidegrid_c` both read as domain/unbound checks), so that part
of the prediction holds, but it is a minority effect here, not the driver.

Lesson worth keeping: when predicting, check the recorded scope decisions first.
The information to get this right was already in my own notes.

**So AA is in much better shape than predicted**: excluding the deliberately
out-of-scope family, ~6 of ~183 in-scope tests fail, and at least two of those
are the asserted-exception category rather than migration gaps.

### Full frontier picture (all four areas now measured)

| area | traced result | real residue |
|---|---|---|
| potentials | 28 of ~1300 (from 100) | ~9 one-offs; ~19 accepted tradeoffs |
| orbit integration | **112/112** | none found |
| streamspraydf | **38/38** | none found |
| quasiisothermaldf | 54/55 | 1 asserted exception |
| action-angle | 177/207 | ~6, incl. asserted exceptions; 24 out of scope |

galpy under `--jit` is far closer to complete than the potential-only ledger
suggested. The residue is dominated by three categories that are NOT "unmigrated
code": deliberate scope exclusions, accepted numerical tradeoffs, and
exception-asserting tests that cannot pass under a trace by construction.

---

## #1208 MERGED -- jax-jit ledger at 28 (from 100)

Audit before merge: the two `if numpy.fabs(...)` hits in EllipsoidalPotential are
on `self._b` and `pa` -- construction-time shape parameters, not coordinates, and
pre-existing rather than diff additions. Correctly left as static config. The two
"new" defs are existing functions that gained an `xp=numpy` parameter, not
misplaced helpers.

## Next: make the `jit=true` dispatch actually usable

Read the workflow (read-only -- I cannot edit it): `jit=true` applies `--jit` to
**every shard**, i.e. it traces the WHOLE suite, not just test_potential.py. So
a dispatch today would be red on the ~31 non-potential failures I measured (30
AA + 1 qdf), because the ledger has no `jax-jit` entries outside
test_potential.py.

That makes the jit dimension currently unusable as a gate: it cannot go green, so
nobody can run it, so traced regressions in orbits/streams/DFs would not be
caught even though those areas MEASURE clean today. Fixing that is worth more
than shaving a few more potential entries.

Precondition I have to respect: I measured `test_orbit.py` only under
`-k test_energy_jacobi_conservation` (112/112). A full dispatch runs the whole
file, which I have NOT measured -- so ledgering only AA+qdf could still leave the
dispatch red on unmeasured orbit tests. Full `test_orbit.py --jit` is running now
rather than guessing.

Also cleaned up after myself: the `git checkout <ref> -- <workflow>` I used to
read the file is exactly the staged-revert trap from earlier in this vacation;
checked `git status` and confirmed nothing was staged.

### Preparing the non-potential ledger PR -- diagnosing before writing comments

Branch `backend-jit/ledger-nonpotential` off develop. The entries to add:

| group | n |
|---|---|
| `actionAngleVerticalInverse*` | 24 |
| other AA | 6 |
| qdf `test_call_diffinoutputs` | 1 |

I am NOT writing the reason comments from the test names. Guessing the cause is
precisely the failure mode I have criticised twice in this vacation -- the wrong
in-tree comment on the TwoPowerSpherical ledger entry, which claimed a dead
`xp.where` branch and sent me down a false trail. A ledger whose comments are
guesses is worse than no comment, because the next reader (or the next me)
trusts it.

So all 7 non-VerticalInverse failures are being re-run with junitxml to extract
the actual exception type per test. The 24 VerticalInverse ones need no
diagnosis: that family is `_reject_backend`-guarded by an explicit prior scope
decision, which is a documented fact rather than an inference.

### Diagnosing before ledgering found 2 REAL BUGS among the 6

Extracted the actual failure text per test rather than inferring from names:

| test | actual cause | class |
|---|---|---|
| `actionAngleAdiabaticGrid_outsidegrid_c` | 4.7e-3 vs 1e-8 tol | numerical (grid extrapolation) |
| `actionAngleAdiabatic_linear_angles_cylsep` | 2.35e-7 vs 1e-7 tol | numerical, marginal (2.3x over) |
| `actionAngleStaeckelGrid_basicAndConserved_actions` | JR = 4.8e-14 vs tol | numerical, near-zero |
| `actionAngleStaeckel_unboundr_angles_c` | ap = 3.44, expected > 1000 | unbound/domain class |
| `actionAngleVertical_linear_angles` | **IndexError: array is 0-dimensional, but 1 were indexed** | **REAL BUG** |
| `test_physical_vertical` | **TypeError: unsupported operand for -: 'tuple' and ArrayImpl** | **REAL BUG** |

The last two are genuine traced-shape defects -- indexing a 0-d array, and a
tuple reaching arithmetic where an array is expected. Those should be FIXED, not
ledgered.

**This is the whole argument for diagnosing before ledgering.** Ledgering by test
name would have buried two real bugs under a comment like "actionAngle residue,
traced accuracy" and they would have looked like accepted tradeoffs forever. The
cost of finding them was one junitxml run.

Revised plan for the ledger PR: it should NOT be a flat list. Correct split:
* 24 VerticalInverse -> ledger (documented out-of-scope)
* 3 numerical -> ledger with the measured deviation quoted, so a future tightening
  is checkable
* 1 unbound -> ledger, flagged as the asserted-exception/domain category that
  needs Jo's API call
* **2 real bugs -> fix, or at minimum ledger with "REAL BUG, fix me" rather than
  a tradeoff framing**

### Real bug #1 fixed: 0-d indexing in actionAngleVertical's backend integrands

`_calc_J_backend`'s integrand did `lim[:, None]`, `xmax[:, None]`, `E[:, None]`,
which assumes 1-D input. Under a trace a SCALAR input stays 0-d, and `[:, None]`
on a 0-d array raises `IndexError: array is 0-dimensional, but 1 were indexed`.

Fix: `[..., None]` -- the nodes broadcast on a NEW TRAILING axis, which
`fixed_quad` reduces (`axis=-1`), so the input shape is preserved for 0-d and
1-D alike. Exactly the pattern used for the RazorThin vectorisation fix in #1201;
reusing a known-good shape idiom rather than inventing one.

**Fixed the class, not the instance**: grep found the same `[:, None]` pattern at
THREE sites (the J, omega and angle integrands). Patching only the one in the
traceback would have moved the crash to the next site and looked like a fix.

Verified across every input shape, not just the failing one:

| input | result |
|---|---|
| numpy scalar | 0.023128601561541364 |
| numpy array | unchanged, 2 values |
| jax 0-d | 0.023128601561541395 |
| jax 1-D | matches |
| **jax TRACED** | **0.023128601561541392** -- previously `IndexError` |

Traced vs numpy agree to ~1e-15.

Incidental observation worth chasing for bug #2: the return TYPE differs between
paths -- numpy returns something iterable, jax 0-d returns a bare 0-d array. That
smells like the `TypeError: unsupported operand for -: 'tuple' and ArrayImpl` in
`test_physical_vertical`, i.e. the second bug may be a return-shape/packing
inconsistency rather than an arithmetic one.

### Bug #1 fixed and committed (`98560c7e`); bug #2 characterised, deferred

Verified: `test_actionAngleVertical_linear_angles` now PASSES under `--jit`, and
the numpy path is untouched (32 passed, 0 failed).

**Bug #2 (`test_physical_vertical`) is NOT what it first looked like.** The error
is `TypeError: unsupported operand for -: 'tuple' and ArrayImpl`, which reads like
a trace artifact. It is not:

```
EAGER  physical (ro/vo set): tuple
EAGER  plain               : ndarray
TRACED physical            : tuple (len 1)
TRACED plain               : ndarray
```

The tuple appears in EAGER too. So the real oddity is that
`actionAngleVertical.__call__` returns a **different container depending on
whether ro/vo are set** -- tuple with, bare array without -- and trace mode only
makes it visible.

My probe did not reproduce the test's exact conditions (the test runs with
astropy units active, where the physical path yields a Quantity rather than a raw
tuple), so I am NOT claiming a diagnosis. What is established: the container
asymmetry exists eagerly, and it is a units-wrapper concern rather than a tracing
one. Deferred with that note rather than guessed at -- the same discipline that
turned up these two bugs in the first place.

---

## Session interruption: background tasks were killed; loop re-armed

Several background runs were stopped mid-flight (the full `test_orbit.py --jit`
measurement, plus stale settle-watchers for the already-merged #1201/#1205).
Nothing was lost that mattered: the killed watchers were for merged PRs, and the
orbit measurement is a pure read -- restarted, no state to recover.

Hourly loop re-armed as job `3333140d`. Rounded 55m -> hourly: 55 does not divide
the hour, so `*/55` fires at :00 and :55 and then leaves a 5-minute gap. Offset to
**:07** rather than :00 so it does not land on the top-of-hour spike.

## #1209 OPEN: the actionAngleVertical 0-d fix

Shipped on its own rather than held behind the ledger work -- it is verified,
self-contained, and does not depend on the orbit measurement. Holding a finished
fix hostage to an unfinished measurement would be the wrong sequencing.

Branch renamed `ledger-nonpotential` -> `actionanglevertical-0d` to match what it
actually contains; the ledger work will get its own branch once the orbit run
lands.

### Bug #2 root-caused: `len(out)` cannot tell "N outputs" from "array of N values"

Reproduced exactly by matching the test's real conditions (forced jax backend --
my earlier probe used plain floats, so `aAVnu` returned numpy and the bug hid):

```
FORCED aAV   : tuple len=1        # ro/vo set
FORCED aAVnu : ArrayImpl          # no ro/vo
FORCED expr  : TypeError: unsupported operand for -: 'tuple' and ArrayImpl
```

Root cause is in `galpy/util/conversion.py`'s physical-output wrapper:

```python
newOut = ()
try:
    for ii in range(len(out)):
        newOut = newOut + (out[ii] * fac[ii],)
except TypeError:  # happens if out = scalar
    newOut = out * fac[0]
```

It infers "multiple outputs" from `len(out)` succeeding. That is ambiguous: a
TUPLE of N actions and an ARRAY of N values both have a length.
`actionAngleVertical` returns a 1-element array, so it is rebuilt as a 1-tuple,
while the non-physical path returns the array unchanged. **numpy masks this**
(`tuple - ndarray` coerces); jax does not, so the trace/backend path surfaces it.

So this is a **pre-existing, backend-independent bug** in the units layer that
happens to be invisible on numpy -- not a tracing defect.

**Not fixing it unilaterally.** The obvious fix is
`if isinstance(out, tuple): ... else: out * fac[0]`, which distinguishes the two
cases properly. But `conversion.py` is shared by every actionAngle class, and if
any path returns an ARRAY of N actions expecting per-element factors
(`fac[ii]`), the isinstance form would apply `fac[0]` to all of them and change
results silently. That blast radius across the whole actionAngle API is Jo's call,
exactly like the R2deriv raise-vs-NaN decision -- and it is the third time this
vacation that the right move was to stop at a diagnosed root cause rather than
ship a plausible one-liner.

For the record the fix is one line; the risk is not in writing it, it is in not
knowing which callers rely on the ambiguous branch.

### The AA jit failures are almost entirely ALREADY-KNOWN eager-jax gaps

Checked every AA `--jit` failure against the existing eager-`jax` ledger instead
of assuming they were new traced problems:

| group | n | already in eager-jax ledger? |
|---|---|---|
| `actionAngleVerticalInverse*` | 24 | **24/24 yes** |
| the 4 "numerical" ones | 4 | **4/4 yes** |
| `actionAngleVertical_linear_angles` | 1 | no -- **REAL BUG**, fixed in #1209 |
| `test_physical_vertical` | 1 | no -- **REAL BUG**, root-caused (units `len(out)`) |

So **28 of 30 were pre-existing eager failures**, already tracked, that simply
re-appear under a different ledger key. The jit dimension revealed exactly **two**
new problems in action-angle, and both were genuine bugs rather than tradeoffs.

That is a much stronger result than "30 AA failures" reads, and it also kills the
ledger PR as originally conceived.

### Design consequence: the ledger key should inherit, not duplicate

`conftest` derives the key as `jax` eager / `jax-jit` traced, and the two lists
are disjoint. So a test ledgered under `jax` is NOT xfailed under `jax-jit`,
which is why all 28 known-eager failures resurfaced in my regen.

Duplicating 28 entries into a `jax-jit` list would be the wrong fix: it is pure
redundancy that then has to be pruned twice, and a future eager fix would leave a
stale traced entry behind.

The principled rule: **a test that fails EAGER cannot be expected to pass TRACED**
-- tracing does not repair a numerical gap or an out-of-scope guard. So
`jax-jit` should INHERIT the `jax` entries, and the `-jit` list should contain
only genuinely trace-specific gaps. That is a ~2-line change in
`_load_ledger`/`_matches` and it makes the traced list mean something precise:
"fails ONLY when traced".

Worth noting this is a design change to shared CI plumbing, so it wants care and a
clear commit rationale rather than a quiet edit -- but unlike the units-wrapper
fix it has no numerical blast radius: strictly more tests are xfailed, never
fewer, so it cannot turn a red into a false green for anything that was passing.

### CORRECTION: I over-claimed twice on the "two real bugs"

Both corrections matter for the record, so stating them plainly.

**1. "The jit dimension revealed exactly 2 NEW problems in action-angle" -- wrong.**
Both `actionAngleVertical_linear_angles` and `test_physical_vertical` were
ALREADY in the eager-jax ledger (lines 119-120), and in the torch one. I checked
the 24 VerticalInverse entries and the 4 numerical ones against the ledger, found
them present, and then asserted the other two were absent **without running that
check on them**. They were not new discoveries; they were known eager failures.

**2. "I root-caused bug #2" -- mostly rediscovery.** The ledger header already
says:

> The lone add, physical_vertical, is a pre-existing decorator quirk (1D
> `__call__` returns a 1-tuple); numpy broadcasts `tuple - ndarray` but jax/torch
> raise on `tuple - array` -- not introduced by this dispatch.

That is the same phenomenon I reported as a finding. My actual contribution is
narrower: pinpointing the *mechanism* -- `len(out)` in the physical-output
wrapper cannot distinguish a TUPLE of N outputs from an ARRAY of N values -- and
the concrete fix shape. A refinement of a documented quirk, not a discovery.

**What genuinely stands:**
* the 0-d `IndexError` FIX (#1209) -- a real fix to a real, ledgered failure;
* the exact mechanism for the tuple quirk, which the old note did not have;
* the ledger-inheritance design (below), which is new.

Lesson, and it is the same one twice now: I checked a set, then generalised the
result to items I had not checked. The earlier version of this was claiming 35
flips before verifying 6 of them were stale. Cheap to check, expensive to assert.

### Ledger inheritance verified

With `-jit` inheriting eager entries, `test_actionAngle.py --backend jax --jit`
goes from **30 failures to 0** -- all `x` (xfail), one `X` (xpass). No `F`.

The XPASS is the entry my #1209 fix repaired: `actionAngleVertical_linear_angles`
now PASSES, so its eager `jax`/`torch` entries are stale and should be pruned.
That is real burndown produced by the fix, and it is only visible because the
inherited run surfaced it as an XPASS rather than hiding it.

### Verifying the stale entries BEFORE pruning (both backends)

The inherited `--jit` run surfaced `actionAngleVertical_linear_angles` as an
XPASS, which suggests #1209 repaired it and its eager `jax` + `torch` entries are
now stale. Running both eagerly to confirm rather than pruning on the XPASS
alone: an entry removed on an assumption is exactly how a red gets reintroduced,
and the XPASS only proves it passes TRACED.

Also re-checking `test_physical_vertical` on both, since the `[..., None]` fix
changes the output shape for scalar input (0-d instead of 1-element), and the
units wrapper's `len(out)` probe behaves differently on a 0-d array -- so #1209
may have incidentally fixed the tuple quirk too. If it did, that is worth
knowing precisely rather than inferring, because the quirk is documented in the
ledger header as pre-existing and would need that note updated.

Two possibilities, both actionable:
* passes eagerly on both -> prune from jax AND torch eager ledgers (real burndown);
* passes traced but not eager -> the entry stays, and the XPASS is trace-specific,
  which would itself be a notable asymmetry worth recording.

### Checking BOTH backends before pruning caught a second, independent bug

Eager verification of the two entries #1209 might have made stale:

```
eager jax   : .F   -> Vertical_linear_angles PASSES, physical_vertical still fails
eager torch : FF   -> BOTH still fail
```

So the jax entry is stale and prunable, but the **torch entry is not** -- pruning
it on the strength of the jax result (or of the traced XPASS) would have put a
red straight back into CI. This is the concrete payoff of the both-backends rule
I have been applying.

**And the torch failure is a different bug entirely**, at
`actionAngleIsochroneApprox.py:1301`:

```python
addto = numpy.cumsum(w.astype(int), axis=1)   # w is a torch Tensor -> no .astype
```

`dePeriod` is a numpy island: `numpy.roll`, `.astype`, `numpy.cumsum` on what is
a backend array whenever the caller is. jax tolerated it (jax arrays expose
`.astype`); torch does not. So the same test failed on two backends for two
unrelated reasons, and fixing the jax one told me nothing about torch.

Fixed it namespace-agnostically. Chose `xp.where(w, 1, 0)` over an int cast
because the array-api spelling of the cast is not uniform across backends, and on
numpy it produces exactly the array `w.astype(int)` did.

Verified with wraps actually exercised -- my first check used data where
`diff < -6` never fired, so it compared nothing:

| check | result |
|---|---|
| wraps triggered | 3 |
| numpy vs old formula | **byte-identical**, max diff 0.0 |
| jax vs numpy | bit-for-bit equal |
| torch vs numpy | bit-for-bit equal |

`dePeriod` is also imported by `streamdf`, so this unblocks the stream path on
torch as well, not just action-angle.

### Reverted: the `dePeriod` fix regressed 10 passing torch tests

The verification above was real but incomplete -- it proved the *formula* was
byte-identical and that the function works when handed a backend array. It did
not check what happens when the function is handed a **numpy** array *under a
forced backend*, which is the common case.

`get_namespace(numpy_array)` under a forced backend returns **torch**, not numpy
(that is the whole point of a forced backend). So `xp.roll(numpy_ndarray, ...)`
became `torch.roll(ndarray)`:

```
TypeError: roll(): argument 'input' (position 1) must be Tensor, not numpy.ndarray
```

Ten previously-passing tests broke -- every `*_linear_angles` test that
`as_numpy`s its actions before calling `dePeriod` (Harmonic, Isochrone x3,
Adiabatic, Staeckel x4, IsochroneApprox):

| `-k linear_angles --backend torch` | result |
|---|---|
| ledgered xfails (`x`) | 6 -- exactly the 6 torch ledger entries |
| **unledgered failures (`F`)** | **10 -- all `torch.roll` on ndarray** |

Caught only because I ran the whole `-k` group rather than the single test I was
chasing. The single test I *was* chasing would have looked fixed.

Reverted (patch parked in scratchpad). Two lessons, both ones I already had
written down and did not apply:

1. **"Coerce under forced backend, don't data-guard"** is not optional -- if a
   function resolves the namespace from a forced context it *must also coerce*
   its input, or it hands numpy arrays to backend ops. I did the first half.
2. Verifying a migration by running the one test it targets is not verification.

**Reframed as a de-duplication, not a migration.** `_backend_dePeriod(xp, arr)`
already exists at `actionAngleIsochroneApprox.py:1191` and the migrated AA path
already uses it -- so the backend path was never an island. What actually exists
is **two implementations of the same function**, one numpy and one agnostic. The
right PR collapses them into one (coercing) `dePeriod`, deletes the private twin,
and casts at the ~11 test call sites. Deferred to its own PR.

### Active PR: ledger inheritance (`<backend>-jit` inherits `<backend>`)

Current jit/eager overlap is **0** (jax 52, jax-jit 28, torch 99), which is the
point: without inheritance a traced run reds on all 52 eager-known jax failures,
so `jit=true` dispatch cannot be green. Inheritance is the gate on traced CI.
`xfail(strict=False)`, so it is strictly widening -- it cannot manufacture a
false green. Verification run (whole `test_actionAngle.py` under `--backend jax
--jit`) in flight.

**PR #1210 opened** (`backend-jit/ledger-inherit-eager` -> `feat/backends-develop`).

While writing it, two things changed the shape of the change:

1. **The rule belongs in the parser, not in `_load_ledger`.** All three lists
   (`backend_xfail.txt`, `backend_slow_skip.txt`, `backend_skip.txt`) go through
   one `_load_backend_nodeids`, and "-jit inherits eager" is a property of the
   backend *name*, not of which list is being read. Putting it in the shared
   parser instead of one caller avoids three copies of the same rule.
2. **The slow-skip half matters more than the ledger half.** Measured:

   | list | jax | jax-jit before | jax-jit after |
   |---|---|---|---|
   | xfail-ledger | 52 | 28 | **80** |
   | slow-skip | 215 | **0** | **215** |
   | permanent-skip | 6 | **0** | **6** |

   A traced dispatch was going to attempt all 215 pathologically-slow tests
   *and* the 6 SIMBAD-network ones. That, not the ledger, is what would have
   made `jit=true` unrunnable rather than merely red.

Also closed a regen footgun: regen xfails nothing, so a traced regen reports the
eager failures too and would have written the entire eager list back out as
duplicate `-jit` entries. `_regen_entries` subtracts what the eager list already
covers (base entries covering their parametrizations included).

Added `tests/test_backend_ledger.py` (5 tests). Without it the rule is reachable
only from a traced dispatch, which does not run on every push -- it would have
landed unexercised. `galpy/` is untouched, so numpy is byte-identical by
construction.

### #1210 verification: measured, then derived exactly

Traced `test_actionAngle.py` under `--backend jax --jit`, with inheritance on:

| | |
|---|---|
| tests | 207 |
| **failures / errors** | **0 / 0** |
| xfail (`pytest.xfail`) | 29 |
| skip (`pytest.skip`) | 3 |
| wall | 920 s |

I started a baseline run at the parent commit to substantiate the before-number,
then realised it was unnecessary -- and that the baseline was going to take an
hour partly *because* it lacked the change. The ledger answers it exactly:

```
RAW backend_xfail.txt, test_actionAngle.py:
  jax     entries: 30
  jax-jit entries:  0
```

So every one of the 29 xfails above is inherited; without inheritance all 30
eager entries are unledgered under `jax-jit` and become failures -- which is
exactly the 30 I measured before. Killed the baseline run rather than spend an
hour re-deriving a number the ledger already fixes.

(The baseline was also informative while it ran: 56 min of CPU versus 15 min for
the inheritance-on run, because without the inherited slow-skips the traced run
actually attempts the pathologically-slow tests.)

**codecov/project** went 36.96% (-63.01%) -> 70.05% (-29.92%) as uploads landed,
with 91 jobs still pending: partial upload, not a real shortfall.
`codecov/patch` is green ("Coverage not affected") -- correct for a tests-only
diff.

### Next PR prepared: `dePeriod` de-duplication (`/var/tmp/dp`)

One `dePeriod` now, coercing (not merely resolving) its input; `_backend_dePeriod`
deleted and its 4 call sites pointed at the public function. Byte-identity of the
unified formula on numpy -- `concatenate`-roll vs `numpy.roll`, and
`xp.astype(w, dtype)` float cumsum vs `w.astype(int)` int cumsum:

| check | result |
|---|---|
| randomised trials | 200 |
| wraps actually triggered | 408 |
| max abs diff | **0.0**, bitwise identical |
| in-tree numpy `dePeriod` vs old formula | bitwise identical (9 wraps) |

Fallout measurement (whole `-k linear_angles` group, both backends) in flight --
this is the check that caught the previous attempt, so it runs before anything
is claimed.

### `dePeriod` de-duplication done and committed (not yet pushed -- #1210 still in flight)

Branch `backend-jit/deperiod-dedup` @ `e41fe536`, off post-#1209 develop.

One `dePeriod` now, resolving **and coercing**; `_backend_dePeriod` deleted, its
4 call sites pointed at the public function. Consumers cast at their boundary:
5 sites in `test_actionAngle.py` (2 tests plus the shared
`check_actionAngle_linear_angles` helper the other 8 use) and 4 in
`streamdf.plotTrack(..., ang=)`, where the result feeds `.T[::-1]` and torch has
no negative-step slicing.

Measured on the **whole** `-k linear_angles` group, which is the check that
caught the first attempt:

| | torch | jax | numpy |
|---|---|---|---|
| before | **10 failures** / 6 xfail | 0 failures | -- |
| after | **0 failures** / 6 xfail | 0 failures / 1 xfail | 16 passed |

Byte-identity, measured not assumed: 200 randomised trials, **408 wraps actually
triggered**, `max|old - new| = 0.0`, bitwise equal.

Two things worth recording:

**The test's own guard caught a weakness in my test data.** I asserted the wrap
branch had to fire more than 10 times; it fired 9. Reason: `dePeriod` detects a
wrap as `diff < -6.0`, but a wrap only drops the angle by `2pi - step`, so with
steps up to 0.9 rad most wraps are *below* the threshold and never register.
Tightened the steps so every wrap fires. A weaker assertion ("some wraps") would
have passed while testing almost nothing.

**Attribution checked before claiming a flip.** The `jax` ledger entry for
`test_actionAngleVertical_linear_angles` XPASSes, so it is pruned -- but I ran it
on the *untouched* library first and it already XPASSed there. That is #1209's
doing, not this PR's, and the commit message says so. The torch entry for the
same test still fails and stays.

**#1210** meanwhile: 87 pass / 65 pending / codecov-project only. codecov climbing
36.96% -> 70.05% -> 74.79% as uploads land; `codecov/patch` green throughout.

### The traced-green endgame is one test function, and it is solved

With the traced ledger now readable as a whole, the remaining `jax-jit`
entries are **entirely `test_potential.py`**, and 18 of the 28 are a single test:

| group | count |
|---|---|
| `test_poisson_surfdens_potential[*]` | **18** |
| `AnyAxisymmetricRazorThinDisk` (2ndDeriv / at_zero / at_infinity) | 3 |
| one-offs (Cautun20, DehnenBinney98, McMillan17, ExpDisk_special, mass_spheroidal, plotting) | 6 |

`Potential.surfdens`'s Poisson branch uses `scipy.integrate.quad`, which cannot
trace. The ledger recorded that a naive swap to fixed-order GL was attempted and
**deliberately reverted**, because the required order scales with the range
(n=100 gives 3.7e-4 at |z|=10; n=400 reaches 2.6e-14) -- a constant order is
accurate only inside whatever range was tested and silently wrong outside it.

That conclusion was right about *plain* GL and wrong about the problem being
hard. The fix is an **existing** utility:
`galpy.backend.quadrature.transformed_quad(..., interior_point=0.0)` -- split the
range at z=0 and cluster nodes toward the split with the `X = xi**k`
boundary-layer remap. A decaying integrand has its mass near z'=0, and the
clustering is defined *relative to the range*, so it is range-aware by
construction rather than by tuning. Same total node count: 50 per panel x 2
versus 100 in one.

Measured over the test's own grid, R in {0.5, 1, 2} and z in {0.125, 0.25, 1, 10},
worst relative error per potential:

| potential | tol | GL n=100 | split n=50 | |
|---|---|---|---|---|
| BurkertPotential | 1e-8 | 6.03e-07 | **3.80e-14** | fixed |
| FlattenedPowerPotential | 1e-8 | 8.43e-03 | **8.16e-14** | fixed |
| HernquistPotential | 1e-8 | 3.62e-04 | **1.20e-11** | fixed |
| JaffePotential | 1e-8 | 2.35e-03 | **6.94e-13** | fixed |
| KuzminDiskPotential | 1e-8 | 9.82e-03 | **3.10e-15** | fixed |
| NFWPotential | 1e-8 | 2.03e-04 | **4.04e-13** | fixed |
| PlummerPotential | 1e-8 | 1.72e-07 | **2.79e-15** | fixed |
| PowerSphericalPotential | 1e-8 | 3.92e-05 | **3.78e-15** | fixed |
| Isochrone / Kepler / DoubleExpDisk / SphericalShell | | already pass | unchanged | |

Nothing is left failing, and the margin is five orders. The failures are exactly
small-R/large-z, which is the signature of nodes spread evenly over a range whose
integrand lives near one point.

Worth being precise about why the earlier attempt failed: it swapped in plain
`fixed_quad` -- no split, no clustering. The revert was correct for *that* change.
Implementation is dual-path (numpy keeps `scipy.integrate.quad` and stays
byte-identical; the backend namespace takes `transformed_quad`), no new helper,
queued behind the dePeriod PR as task #161.

### #1210 MERGED (`70d5b4ff`), #1211 opened

All 154 checks green; `codecov/project` finished at **99.96% (+0.00%)**, having
climbed 36.96% -> 70.05% -> 74.79% -> 99.96% as the shards uploaded. That is the
partial-upload signature behaving exactly as diagnosed, and worth stating plainly
because it looked like a 63-point regression for most of the run.

Squash-merged into `feat/backends-develop`. Traced runs now inherit the eager
ledger, slow-skips and permanent skips, which is the precondition for a
`jit=true` dispatch being green at all.

**PR #1211** (`dePeriod` de-duplication) opened against `feat/backends-develop`,
rebased clean onto the merged base, diff confined to its 5 files.

### surfdens implemented — and my "fixes all 18" prediction was wrong

Committed on `backend-jit/surfdens-quad` (`d6456975`), held unpushed while #1211
is the active PR.

**8 entries flip, not 18.** The numpy-level experiment in the previous entry
tested the quadrature *math* and I read it as testing the *tests*. Measured under
an actual trace:

| | |
|---|---|
| XPASS | **8** — Burkert, Hernquist, Isochrone, Jaffe, KuzminDisk, NFW, Plummer, SphericalShell |
| still xfail | 6 |

The residual 6 have nothing to do with quadrature:

- **AnyAxisymmetricRazorThinDisk, DoubleExponentialDisk** — `TypeError: Methods
  in X do not accept array inputs. Please input scalars`.
- **FlattenedPower, Kepler, PowerSpherical, testMWPotential** —
  `TracerArrayConversionError: __array__() was called on traced array`, i.e. a
  tracer -> numpy conversion somewhere else in the potential.

FlattenedPower and PowerSpherical are the instructive pair: my experiment
predicted both fixed (8.16e-14, 3.78e-15) and both still fail, because the
failure was never accuracy.

**And the first version was a real regression.** Feeding the node array to
scalar-only potentials turned two *passing* eager-jax tests into TypeErrors —
trading 8 traced flips for 2 eager breakages. Caught by running eager as well as
traced, on the reasoning that a change routing new data into potentials could
break the path that already worked. Gated on the existing `_force_accepts_arrays`
detector; eager back to 0 failures.

One more CI-only trap caught before pushing: `numpy.fabs(z)` on a torch tensor
emits a NumPy 2 `__array_wrap__` DeprecationWarning, and the coverage shard runs
`-W error::Deprecation`. Computing `absz` per-namespace fixes it; re-verified
with warnings escalated.

| check | result |
|---|---|
| numpy byte-identity | **240 hex-exact values, all match** |
| eager jax | 0 failures (was 2 with the first version) |
| traced jax | 0 failures, 8 XPASS |
| `-W error::Deprecation` | 8 passed |

**#1211** meanwhile: 14 pass / 137 pending, `codecov/patch` 14.28% and
`codecov/project` -53.97% — the partial-upload signature again (both recover as
shards report; #1210 went 36.96% -> 99.96% the same way).

### Traced `test_orbit.py` does not fit on one box — use the sharded dispatch

The local `test_orbit.py --backend jax --jit` measurement hit its 90-minute cap
before writing any results, even *with* the inherited slow-skips (127 of them
apply to that file). So the answer is not a longer local timeout: a single-box
traced orbit run is the wrong instrument.

The right one is now available. `backend-tests.yml` has a `workflow_dispatch`
input:

```
jit:  Run the whole suite TRACED (jax.jit / torch.compile at every galpy entry
      point); burndown lists are keyed "<backend>-jit"
```

That shards the traced suite across runners, which is precisely what #1210's
inheritance made *possible* — before it, every one of the eager-known failures
and all 215 slow tests would have run and red the dispatch.

Holding the dispatch until #1211's CI drains rather than firing it now: 122 jobs
are still queued on the active PR and the project is runner-constrained, so
launching a full traced matrix would starve the PR I am trying to land. One PR at
a time applies to CI capacity too.

**#1211 codecov** is recovering exactly on the expected curve: project
32.21% -> 63.75%, patch 14.28% -> 71.42%.

### Chasing the residual paid off: surfdens now 12 flips, not 8

The 6 that survived the first fix turned out to be **three different bugs**, and
finding that out took one probe rather than more speculation. Reproducing the
call directly under `jax.jit` (no pytest) showed
`surfdens(forcepoisson=True)` on FlattenedPower **already worked** — so the
residual failure was never in the branch I had changed:

| what actually failed | where |
|---|---|
| generic `Potential._surfdens` (the ANALYTIC route) still on `scipy.quad` | `Potential.py:801` — FlattenedPower |
| `PowerSphericalPotential._surfdens` calling `scipy.special.hyp2f1` on a tracer | `PowerSphericalPotential.py:388` — Kepler, PowerSpherical |
| potentials that reject array input at all | AnyAxisymmetricRazorThinDisk, DoubleExponentialDisk |

Fixed the first two — the analytic `_surfdens` gets the same dual path, and the
closed-form one routes through `galpy.backend.special.hyp2f1` (whose numpy path
goes straight back to scipy, so nothing moves). The third stays: those two
potentials fail on the test's own `check_potential_inputs_not_arrays` guard under
a tracer, which is a property of the potential, not of surfdens.

Measured with `--runxfail`, so the ledger cannot mask anything:

| | tests | failures |
|---|---|---|
| eager jax | 64 | **0** |
| traced jax | 64 | **2** (the scalar-only pair) |

**jax-jit ledger 28 -> 16.** numpy byte-identical on *both* routes, hex-exact
against the base library: 240 values through Poisson, 312 through analytic.
Whole convenience suite green with `-W error::Deprecation` (116 passed), numpy
surfdens/dens tests 19 passed.

The lesson I keep re-learning, stated once more because it cost real time twice
today: an experiment that reproduces the *mechanism* is not an experiment that
reproduces the *failure*. The numpy quadrature harness said 18 would flip; the
trace said 8; the direct probe said the other 6 were three unrelated problems.

### 13th flip found; and a classifier bug that briefly inverted my reading

Swept the 6 "one-off" traced entries. Corrected picture:

| | |
|---|---|
| **PASS** (prunable) | `test_plotting`, `test_plotting_xy_effective_error` |
| plain `pytest.skip` (entries inert) | McMillan17, Cautun20, DehnenBinney98 |
| genuinely failing | `test_mass_spheroidal` (TracerArrayConversionError), `test_ExpDisk_special` |

`test_ExpDisk_special` is worth flagging as **permanent**, not pending work: it
asserts `RazorThinExponentialDisk.R2deriv` raises AttributeError off-plane, and
#1206 made that path return NaN under a trace (your call: "Let's do NaN"). An
exception conditioned on a traced value cannot fire, so this one stays ledgered
by construction.

`test_plotting` was verified against the untouched library — it **fails** there
and passes with the fix, so it is a genuine 13th flip, not a stale entry.
**jax-jit ledger 28 -> 15.**

A process note, because it nearly cost me a wrong claim: my first classifier used
`tc.find('failure') or tc.find('error')`, and an ElementTree element with no
children is FALSY, so failures were silently reported as passes. Python printed
the DeprecationWarning that says exactly this and I read past it. The suite-level
`failures=2` disagreed with my per-test table, and that contradiction is what
caught it. Two numbers that must agree are worth more than one number I trust.

**#1211**: 92 pass / 59 pending. I also walked back an earlier worry here — I
read `codecov/patch` stalling at 71.42% as a real shortfall, but
`test_streamdf.py` already contains an explicit "Dangerous hack to test case
where l decreases along the stream" covering the descending branches I touched.
The 2 misses are unuploaded shards (sessions 14 of 28), not untested code. I
should have checked for the test before theorising about the number.

### #1211 MERGED (`be4f01eb`); #1212 (surfdens) opened

All 154 checks green. `codecov/patch` finished at **100.00%** and project at
99.96% (+0.00%), which settles the earlier worry: the 71.42% plateau was
unuploaded shards, exactly as the existing descending-branch test implied.

Audited the diff before merging — no `if`-on-array introduced, the `as_numpy`
casts sit at consumer boundaries rather than inside the library, and the
duplication removed is the whole point of the change.

**PR #1212** opened: surfdens traceable end to end, `jax-jit` ledger **28 -> 15**.

Running total for the traced ledger this vacation:

| | jax-jit entries |
|---|---|
| 2026-07-25 baseline | 100 |
| after #1206 | 49 |
| after #1208 | 28 |
| **after #1212 (open)** | **15** |

Of the 15 left, only a handful are real work: `test_mass_spheroidal`
(TracerArrayConversionError), 3 that plain-skip (McMillan17 / Cautun20 /
DehnenBinney98, so their entries are inert), 5 mock-wrapper surfdens entries that
also skip, and the rest are the two by-design families —
AnyAxisymmetricRazorThinDisk (GL vs the z=0 Hadamard second derivative) and
`test_ExpDisk_special` (asserts an exception that cannot fire under a trace,
following the NaN decision).

### Next PR ready: the "Pspecial-blocked" masses were blocked by a stale comment

`backend-jit/pspecial-masses` @ `efe815ce`, held while #1212 is active.

Two `_mass` implementations each carried a comment saying their closed form
"requires scipy.special.hyp2f1 / scipy.special.erf, which has no
backend-agnostic (jax/torch array-API) replacement -> numpy only". Both
functions have been in `galpy.backend.special`'s router since the Pspecial work.
The comments were simply out of date, and they were the entire reason
`test_mass_spheroidal` sat in the `jax-jit` ledger.

The interesting part is what the stale comment **hid**: routing the special
function through the router was not sufficient. `TriaxialGaussian._mass` also
calls `numpy.exp` on `R`, which converts a tracer, so after the erf fix it still
failed. I only found that because I re-probed each potential individually rather
than assuming the named blocker was the only one — the same shape of mistake as
the surfdens residual, caught the same way.

| | |
|---|---|
| eager jax / traced jax / eager torch | 0 failures each |
| numpy byte-identity | **48 hex-exact mass values**, 8 potentials x 6 radii (0.05 to 1000) |

Also deleted the now-dead `from scipy import special` in
`TriaxialGaussianPotential` — pycln left it, and my first grep for `special.`
would have missed the fact that `_bspecial.erf` contains that substring, so I
checked with a word-boundary match before removing.

No new test: both call sites are unconditional (no backend branch), so the numpy
coverage shard exercises them, and `test_mass_spheroidal` itself now runs under
the eager jax/torch shards and traced. A separate backend test would duplicate
it — which is the kind of thing the audit is supposed to catch.

**#1212**: 9 pass / 143 pending, codecov partial as usual.

### #1212: CI found five eager-backend regressions, and the fix was a design change

The surfdens change went green in every local check I ran, then CI reported **5**
failures — all EAGER backend, none traced:

| shard | test | cause |
|---|---|---|
| jax + torch jeans | `test_sigmalos_wlog_zerobeta` | `z = inf` — jeans integrates the whole line of sight. scipy.quad transforms an infinite limit; a fixed-node GL rule puts every node on `inf`. |
| torch potential grp 2 | 5 `poisson_surfdens[mock*Wrapper*]` | the wrapper is scalar-only, but the mock subclass overrides `_Rforce` **undecorated**, so `_force_accepts_arrays` cannot see the marker |
| torch quantity | `test_potential_function_*` | `phi=None` reaching `phi[..., None]` |

I patched each in turn -- None-safe indexing, a finite-range guard, then a
delegation-aware detector. The third attempt is where I should have stopped
patching: the mock composes via `_potlist`, not `_pot`, and special-casing a test
attribute in library code is obviously wrong. **"Does this potential accept an
array" is not decidable** -- the codebase's own `_force_accepts_arrays` is
explicitly best-effort, and `mass()` has the same latent hole.

So I replaced the whole accumulated guard stack with **one decidable predicate**:

```python
def _quad_needs_backend(xp, absz):
    # the backend rule exists for exactly one reason -- under a trace scipy is
    # unusable. Whenever the limit is CONCRETE, scipy is still the better tool.
    if xp is numpy:
        return False
    return not has_concrete_truth_value(xp.all(xp.isfinite(absz)))
```

Concreteness *is* decidable. Eager jax/torch have concrete limits, so they keep
scipy and are untouched by construction -- which kills all three failure classes
at once rather than one at a time. Three helpers and ~40 lines of guards
collapsed to this.

The honest cost: eager backend surfdens stays on scipy, so it is not
differentiable there. It never was, so nothing regresses, and the PR's goal was
traced-green, which is unaffected.

| check | result |
|---|---|
| numpy Poisson route | **240 hex values, byte-identical** |
| numpy analytic route | **312 hex values, byte-identical** |
| eager torch + traced jax | verification running |

Worth recording as a general lesson: when a gate needs its third special case,
the gate is asking an undecidable question. Look for the decidable one.

### Concreteness gate verified and pushed to #1212

| check | result |
|---|---|
| numpy Poisson route | 240 hex values, **byte-identical** |
| numpy analytic route | 312 hex values, **byte-identical** |
| **eager torch** (the shard CI failed on) | 64 tests, **0 failures** -- the mock wrappers pass again |
| eager jax + torch, jeans/dynamfric | 21 tests, **0 failures** each |
| traced jax | 64 tests, **2 failures** -- exactly the by-design scalar-only pair, still ledgered |

Pushed as `6ef5ee09`; cancelled the previous commit's two still-running workflows
since the project is runner-constrained.

One thing I checked rather than assumed: `test_jeans` under a **trace** fails on
an accuracy assertion. It is not a regression -- surfdens was scipy-only before,
which cannot trace at all, so that test was already traced-broken. It is not
ledgered only because `test_jeans` has never been run traced. Recorded here so a
future `jit=true` dispatch does not read it as new.

**Traced measurements of two never-before-measured files** (post-inheritance, so
the slow-skips apply):

| file | traced result |
|---|---|
| `test_diskdf.py` | 129 tests, **0 failures** |
| `test_sphericaldf.py` | 223 tests, **8 failures** -- all `TracerArrayConversionError` in the constantbeta `*_directint` family |

So beyond `test_potential.py`, the traced frontier is a single bounded cluster of
8 tests. That is the next target after the current PR queue.

### The 8 sphericaldf traced failures are a frozen-state boundary, not a missing migration

Characterised them rather than assuming, because "TracerArrayConversionError"
looks like an un-migrated numpy call and here it is not.

The tests do `dfh.fE = lambda x: dfh._fE_interp(x)`, and `_fE_interp` is **frozen
precompute state** built by `eddingtondf._ensure_fE_interp`: scipy
`InterpolatedUnivariateSpline` on the numpy path, galpy `Spline1D` under a forced
backend. The table is built eagerly at fixture time and then queried under a
trace. That is the same shape as the known qdf `_precomputergLzmax` problem --
state frozen in one mode and used in another -- not a missing `xp` swap.

Filed as task #162 with that framing, so whoever picks it up does not start by
grepping for `numpy.` calls that are not the cause.

**#1212** after the force-push: 17 pass / 134 pending, codecov partial as usual at
this stage. Cancelled the superseded commit's two workflows when I pushed.

### Correction: the 8 sphericaldf failures are duplication, not a frozen-state boundary

I framed these last entry as a frozen-interpolant-vs-trace problem, by analogy
with the qdf `_precomputergLzmax` case. That was wrong, and two direct checks
showed it:

- `Spline1D` traces fine, eager **and** under `jax.jit` -- so the interpolator is
  not the problem.
- Inside `_ensure_fE_interp` under a forced backend, `get_namespace()` **is**
  `jax.numpy`, and yet `_fE_interp` comes out a scipy
  `InterpolatedUnivariateSpline`.

The reason: `constantbetadf` has its **own** `_ensure_fE_interp`
(`constantbetadf.py:516`), a near-copy of `eddingtondf._ensure_fE_interp`
(`eddingtondf.py:112`) but **without the dual path** -- it always builds the scipy
spline, where eddingtondf branches to galpy `Spline1D` under a backend. The two
classes share no ancestor that owns this (`constantbetadf -> _constantbetadf ->
anisotropicsphericaldf -> sphericaldf -> df`), so they are genuinely independent
copies and only one was ever migrated.

**This is the dePeriod shape again** (#1211): one routine, two copies, one
migrated, and the un-migrated copy silently wins wherever it is reachable. Third
time this pattern has produced a traced failure this vacation -- dePeriod, the two
`_surfdens` routes in #1212, and now this. Worth treating "is there a second copy
of this function?" as a standing question rather than an occasional discovery.

Fix per the placement directive: hoist the shared f(E)-spline builder to
`sphericaldf` and delete BOTH copies, not patch the constantbeta one. Task #162
rewritten with this framing; the earlier one would have sent someone hunting for
`numpy.` calls that are not the cause.

**#1212**: 74 pass / 77 pending, codecov partial. No real reds -- the eager
regressions stay fixed.

### #1212: a REAL patch-coverage gap this time, caused by my own design change

codecov reached `project 99.95% (-0.01%)` -- flat, fully uploaded -- with
`patch 85.00%`. That combination is the discriminator for a genuine shortfall,
as opposed to the partial-upload pattern (big project drop + low patch) I have
been correctly dismissing all week. So I treated it as real, and it was.

The cause was the design change itself. Once the backend rule is gated on the
limit being a **tracer**, my two tests -- which called `surfdens` eagerly under
`gb.use(backend, force=True)` -- took the *scipy* branch. They asserted nothing
about the new code, and the new lines were never executed on any shard.

That is worth stating plainly: after switching the gate, the tests still passed
while testing the opposite path. Passing tests are not evidence that the intended
code ran.

Rewrote both to `jax.jit` the call, which is what the feature actually is, and
extended the second to the analytic `_surfdens` route (separate code from the
Poisson branch). Confirmed by line-level coverage rather than by re-reading the
tests:

| region | result |
|---|---|
| backend branch, both routes | **covered** |
| only uncovered line in the region | the scipy branch at `Potential.py:774`, which the numpy suite covers on the coverage shard |

An aside on method: my first check grepped for lines containing
`transformed_quad` / `_node_axis` and reported "2 misses". Those were argument
continuation lines, which are not statements and are never marked executed.
Reading coverage's own `missing_lines` gave the true answer -- one line, and an
expected one.

Full convenience suite: **112 passed** with `-W error::Deprecation`. Pushed as
`43ad13d8`; cancelled the superseded commit's two workflows.

### #1212 status; pspecial pre-flighted much harder than #1212 was

**#1212** after the coverage fix: 147 pass / 3 pending / 3 fail. Of the three:

- **`codecov/patch` 80%, `codecov/project` -13.78%** at 25 of 28 sessions -- still
  the partial-upload shape, so not yet actionable. Codecov's own file report lists
  772/773/774, 783, 831-837 and 846 as **covered**, i.e. both surfdens routes and
  both branches are now exercised. Whatever the residual misses are, they are not
  the lines I was worried about; I will pin them down from the file report if the
  number survives full upload.
- **py3.14 `from_name` build**: `pyvo.dal.exceptions.DALServiceError: Unable to
  access the capabilities endpoint` -- the known SIMBAD network flake. Confirmed
  from the log rather than assumed (the first grep only surfaced C++ compiler
  noise from the Torus build, which looks alarming and is not the failure).
  Re-run is queued behind the workflow still being in progress.

**Pre-flighting the NEXT PR against its consumers, which is the #1212 lesson.**
`backend-jit/pspecial-masses` changes two `_mass` implementations, so I ran the
shards that actually consume them rather than only the test I was chasing:

| shard | result |
|---|---|
| `test_backend_ellipsoidal` + `amp_grad` + `dtype_policy`, **torch** | 1048 tests, **0 failures** |
| same three, **jax** | 1048 tests, **0 failures** |
| `test_quantity.py -k "mass or dens or surfdens"` | 7 tests, **0 failures** |

That is the check #1212 did not get before its first push, and it is why #1212
cost three extra CI rounds. Doing it up front is cheaper than discovering it from
five red shards.

### #1212 MERGED (`dd0f124e`); #1213 (Pspecial masses) opened

All 154 green. Two things had to be untangled to get there, and both were
process rather than code:

- **`codecov/patch` finished at 100.00%** (20 hits, 0 misses). The 80% I flagged
  last entry was partial upload after all -- the traced tests had already fixed
  the real gap. Codecov's file report confirmed both surfdens routes covered.
- **The last red was `all-clear`, not a test.** The py3.14 `from_name` build had
  flaked on SIMBAD (`DALServiceError`), GitHub had already auto-retried it to
  success on attempt 2, but the aggregate gate had recorded its verdict during
  attempt 1 and stayed red. `gh run rerun` refused ("cannot be retried") because
  the build run itself had already succeeded -- the thing needing a re-run was
  the gate. Re-ran `all-clear` alone and it went green.

  Worth remembering: when a flaked shard self-heals, the aggregate gate does not
  re-evaluate. Check *which* check is red before reaching for the shard.

**Rebase conflict, resolved by intent rather than by picking a side.** Both PRs
edit `backend_xfail.txt`, and the conflict offered `test_mass_spheroidal` (HEAD)
vs `test_plotting` (mine). The correct answer was **neither**: #1212 had already
removed `test_plotting` upstream, and this commit removes `test_mass_spheroidal`.
Taking either side would have silently resurrected a pruned entry. Verified after:
exactly one line removed, no markers anywhere in the tree, byte-identity intact.

**jax-jit ledger 15 -> 14.**

Vacation running total for the traced ledger: **100 -> 49 -> 28 -> 15 -> 14**.

### #1213 in CI; #162 (the f(E) duplicate) implemented locally

`backend-jit/fE-interp-dedup`, off post-#1212 develop.

Confirmed the hierarchy before touching anything: `constantbetadf` and
`eddingtondf` share exactly one non-trivial ancestor, `sphericaldf`. Diffed the
two `_ensure_fE_interp` copies -- identical energy grid, and eddington's is a
strict superset (it additionally `as_numpy`s the bounds, a no-op on numpy, and
branches to `Spline1D` under a backend).

So the fix is the one the placement directive asks for: **hoist the superset to
`sphericaldf`, delete both copies**, rather than patching the constantbeta one
into agreement. `sphericaldf` already imports everything it needs
(`as_numpy`, `get_namespace`, `Spline1D`, `scipy.interpolate`), which is a good
sign the method belongs there.

Byte-identity probe covers **both** classes (constantbeta at three betas plus
eddington, five energies each), because a hoist that silently changes the other
class would be the obvious way for this to go wrong. Running.

This is the third de-duplication this vacation -- dePeriod (#1211), the two
`_surfdens` routes (#1212), and now this. In every case the un-migrated copy was
invisible until something traced, because on numpy both copies behave
identically. That is the whole hazard: **duplicate routines diverge silently in
exactly the dimension the migration cares about.**

### #162 result: the hoist fixes 3 of 8, and the other 5 are a different bug

Traced `test_sphericaldf.py` after hoisting `_ensure_fE_interp` to `sphericaldf`:
**223 tests, 8 failures -> 5**.

The split is exact, and checking *which* tests flipped explained it:

| | monkeypatches `dfh.fE = lambda x: dfh._fE_interp(x)` | traced result |
|---|---|---|
| `dehnencore_in_nfw_{dens,meanvr,sigmar}` | **yes** (3 of 3) | **now pass** |
| `selfconsist_dehnencore_{dens,meanvr,sigmar}`, `differentpotentials_dens`, `exptruncnfw_dens` | no | still fail |

`grep -c "dfh.fE = lambda"` over the whole file returns exactly **3**, and those
are precisely the three that flipped. The monkeypatching tests route through
`_fE_interp`, which the hoist made a traceable `Spline1D`; the other five call the
real `constantbetadf.fE`, which carries its own numpy island. So the duplication
fix is complete for what it targets, and the residual is a separate,
larger piece of work -- migrating `fE` itself.

That is a better outcome than it sounds: it means my earlier root-cause note
(task #162) was right about the mechanism but described only **part** of the
population. Recording the correction rather than quietly shipping "8 -> 5" as if
it were the plan.

Scope decision: ship the de-duplication on its own. It removes a real duplicate,
flips 3, and prevents the next silent divergence. `constantbetadf.fE` gets its own
task -- bundling it here would make an unreviewable PR out of two unrelated fixes.

**#1213**: 5 pass / 146 pending, codecov partial. Nothing failing.

### #162 committed (`0e2295b7`), held behind #1213

Full three-mode verification of the hoist:

| mode | result |
|---|---|
| numpy | 223 tests, **0 failures** |
| eager torch | 223 tests, **0 failures** |
| traced jax | **8 -> 5 failures** |
| numpy f(E), both classes | **20 hex-exact values, byte-identical** |

No eager regression -- which is the check #1212 taught me to run before pushing,
not after.

Ledgered the residual 5 under `jax-jit` with the measured reason attached (real
`constantbetadf.fE` still converts a tracer; the three siblings that monkeypatch
the interpolator now pass, and `grep -c` for that monkeypatch returns exactly 3).
Filed as task #163 with a caution to probe `fE` directly first -- a
mechanism-level guess has over-predicted twice this vacation, so the note tells
the next person to measure before editing.

`jax-jit` ledger 14 -> 20. **That is deliberate growth**: these five were always
broken traced, just never measured, because `test_sphericaldf.py` had never been
run under a trace until this vacation. Ledger size is only a progress metric when
the denominator is fixed; measuring new files moves it up before it moves down.

**#1213**: 28 pass / 123 pending, codecov at 8 of 28 sessions.

### #163 root cause measured: frozen `_rphi`/`_logstartt`, and my guess was wrong

Followed my own note ("probe before editing") and it paid: the guess recorded in
task #163 -- `_evalpot_asnumpy` at constantbetadf.py:735 -- was **not** the site.

Direct `jax.jit` probe of `dfh.fE`:

```
fE eager -> OK
fE jit   -> TracerArrayConversionError
frames:   constantbetadf.py:585 fE
       -> constantbetadf.py:657 _fE_backend
       -> galpy/backend/_namespaces.py:204 as_numpy
```

Line 657 is `Enp = as_numpy(xp.where(indx, Eb, xp.ones_like(Eb) * emin))`, and its
own comment says why: *"clamp out-of-bounds E for the frozen numpy interpolators"*.
It exists so `self._rphi(Enp)` and `10 ** self._logstartt(Enp)` can be queried --
both scipy `InterpolatedUnivariateSpline`s.

So the residual five ARE a frozen-interpolant boundary after all, just for
`_rphi`/`_logstartt` rather than `_fE_interp`. Two entries ago I replaced the
frozen-state framing with "duplication"; the honest resolution is that the
population was **mixed** -- 3 duplication, 5 frozen-state -- and each of my
single-cause framings was right about part of it and wrong about the rest.

The fix is now the same pattern #162 just applied, which is a good sign the
abstraction is right: give `_rphi`/`_logstartt` the dual path (scipy on numpy,
`Spline1D` under a backend), then drop the `as_numpy` clamp entirely. `Spline1D`
is already verified traceable. Task #163 rewritten with the exact site, the failed
guess recorded so nobody repeats it, and a reminder to check for a THIRD copy of
the spline-building pattern first -- the duplication existed because nobody looked.

**#1213**: 86 pass / 66 pending; `codecov/patch` already **100%**; the
`run-notebooks (3.14)` SIMBAD flake is being superseded by a concurrency-duplicate
run that already shows pass, so no intervention needed.

### #1213 MERGED (`48af8808`); #1214 (f(E) de-dup) opened

154/154 green. Two failures along the way, neither a real red, and both worth
distinguishing because they look alike in the checks list:

- **`run-notebooks (3.14)`**: `DALServiceError` from a SIMBAD query inside a
  notebook. Resolved itself -- the notebook workflow had a concurrency duplicate
  whose 3.14 job already passed.
- **`backend torch test_orbit --splits 5 --group 4`**: the tests all PASSED
  ("107 passed, 2 xfail, 0 FAILED"); the failing STEP was *"Upload shard JUnit
  XML"* with `Failed to CreateArtifact: ECONNRESET`. A GitHub artifact-service
  network error, not a test failure at all.

  That one is a good argument for reading the failing *step* before the test log.
  A red "backend torch test_orbit" check on a PR that changes `_mass` looks
  exactly like a real regression, and it was infrastructure.

Audit before merge found nothing to fix, and confirmed one thing worth writing
down: `TwoPowerTriaxial` still imports `scipy.special` and still uses
`special.gamma` at lines 121-123 -- but those run in `__init__` on scalar
*parameters*, never on traced coordinates. So removing that import (as I did for
`TriaxialGaussian`, where nothing was left) would have been wrong. The rule is
not "no scipy in migrated files", it is "no scipy on values that can be traced".

**#1214** opened: the f(E) spline builder hoisted to `sphericaldf`, both copies
deleted. `jax-jit` ledger 14 -> 20, which is measurement rather than regression
(the five were always broken traced; `test_sphericaldf.py` had simply never been
run under a trace before this vacation).

Vacation totals: **#1210, #1211, #1212, #1213 merged**; #1214 open; #163 next,
with its exact site already measured.

### #1214: CI caught a real bug of mine -- the hoist stole `dMdE`'s decorator

Windows shard went red on four `test_quantity.py::test_sphericaldf_method_*`
tests: `AttributeError: 'numpy.ndarray' object has no attribute 'to'`, i.e.
`dMdE` had stopped returning a Quantity.

Cause was entirely mechanical and entirely mine. I inserted the hoisted method at
the anchor `    def dMdE(self, E):` -- but decorators sit ABOVE that line, so the
new method landed **between `@physical_conversion("massenergydensity", pop=True)`
and `dMdE`**, taking the decorator with it. `dMdE` was left bare.

```
393:    @physical_conversion("massenergydensity", pop=True)
394:    def _ensure_fE_interp(self):      <- stole it
432:    def dMdE(self, E):                <- undecorated
```

Reproduced locally in 20s, fixed by moving the insertion above the decorator, and
confirmed: `test_quantity -k sphericaldf` **9 passed**.

Two things worth keeping:

1. **Anchor insertions on a `def` line are unsafe in a decorated codebase.** The
   safe anchor is the first decorator, or a blank line between methods. I have hit
   this class of edit twice now (the earlier one was ruff reformatting my
   exact-match strings out from under a second edit); text-level surgery on Python
   needs the decorator block treated as part of the function.
2. **My local verification missed it because I ran `test_sphericaldf.py`, not
   `test_quantity.py`.** Same shape as #1212: I test the file I am thinking about,
   not the file that consumes what I changed. The units tests are a consumer of
   every df method, so they belong in the pre-flight for any df signature or
   placement change.

Pushed `4530a968`; cancelled seven stale workflows from the superseded commit.

### #1214 pre-flight: the consumer sweep I said I would start doing

Ran the six backend df consumer suites against the fixed branch rather than only
the file I had been thinking about:

`test_backend_{sphericaldf,constantbetadf,eddingtondf,isotropicsphericaldf,osipkovmerrittdf,kingdf}.py`
-> **218 tests, 0 failures.**

Complete local picture for the branch:

| check | result |
|---|---|
| `test_quantity -k sphericaldf` (what CI caught) | 9 passed |
| `test_sphericaldf.py`, numpy | 223 passed |
| 6 backend df suites | **218 passed** |
| numpy f(E), both classes | 20 hex-exact values, byte-identical |
| eager torch / traced jax | 0 failures / 8 -> 5 |

I pushed the decorator fix before this sweep finished, which is the same
push-then-verify shape that cost #1212 three CI rounds. The difference is
deliberate: the fix was a pure code MOVE with no semantic delta, already verified
by the exact four tests that failed, and a known-broken PR sitting in a
runner-constrained queue costs more than it saves. The sweep ran in parallel
rather than being skipped, and it confirmed the pushed commit rather than
contradicting it -- but if it had contradicted, the cost would have been another
force-push, so this is a judgement call and not a rule.

## Wake 15 (2026-07-28 ~15:20Z) -- #1214 still running; #163 built locally

### #1214 CI

The pass count had gone *backwards* since the last wake (151 -> 52). Not a
regression: the decorator-theft force-push at 14:26Z restarted every check, so
the 151-pass reading belonged to the superseded SHA. `run_attempt=1` on the
current runs confirms it. Progressed 52 -> 72 -> 84 pass over the wake, with only
`codecov/patch` + `codecov/project` red.

Those two are the partial-upload phantom, not a real gap: codecov reports the
project at **63.42%** against galpy's true ~99.6% while 67 shards are still
pending. The discriminator I wrote down after #1212 (flat project + low patch =
real gap; large project drop = partial upload) says wait, so I waited.

### #163: constantbetadf.fE traceable

Two frozen interpolators set the fE integration limits. The measured break was
`as_numpy` at `constantbetadf.py:657` clamping E to numpy so both could be
queried -- under `jax.jit` that clamped E is a tracer and the conversion raises.

The pre-work grep (the "check for a third copy" rule that #1214 taught me) paid
off twice, and both times by *shrinking* the job:

- `_rphi` needed **nothing**. `_setup_rphi_interpolator` already lives on
  `sphericaldf` and already returns `Spline1D`. My own task note had said to give
  it the dual path; it already had one.
- `_logstartt` was the only scipy spline left, and it is constantbetadf-only
  (one construction site), so it warranted a swap, not a hoist. `Spline1D` is
  already the idiom in kingdf / eddingtondf / sphericaldf / streamgapdf.

So the fix is one interpolator swap plus dropping the clamp -- not the two-site
dual-path build I had written down.

**Which half does the work.** I mis-set this once mid-wake and the measurement
corrected me. `beta=-0.5` and `twobeta=-1` are *not* the same DF:

| construction | `_halfint` | `_logstartt` |
|---|---|---|
| `beta=-0.5` | False | built |
| `twobeta=-1` | True | absent |
| `beta=0.25` | False | built |

The five ledgered tests build with `twobeta=-1`, so they return before
`_logstartt` and are fixed purely by the clamp drop feeding `_rphi`. The
`_logstartt` swap fixes the non-half-integer branch, which was equally
untraceable but had no ledgered test. Both changes are needed; neither is
redundant.

### Verification

| check | result |
|---|---|
| `Spline1D` vs scipy, numpy queries | array / scalar / extrapolated all **maxdiff exactly 0.0** |
| traced vs eager, `twobeta=-1` | 7.8e-16 |
| traced vs eager, `beta=0.25` | 1.5e-13 |
| out-of-bounds E under trace | exactly 0.0 |
| the 5 ledgered tests, `--jit --runxfail` | **5 passed** |

New test `test_generic_fE_jit_traceable` covers both branches plus the
out-of-bounds clamp, at `rtol=1e-12` -- ~7x over the measured 1.5e-13 floor.
Confirmed load-bearing by stashing only the source change: it fails with
`TracerArrayConversionError` at `_namespaces.py:204`, the exact site.

jax-jit ledger **19 -> 14**. Everything left is test_potential.py surfdens /
RazorThin, which is the by-design tail.

Held from PR: #1214 is still the active PR and the rule is one at a time.

### #163 verification complete

| suite | result |
|---|---|
| `test_backend_constantbetadf.py` (incl. new test) | 32 passed |
| `test_sphericaldf.py`, numpy | 223 passed |
| `test_sphericaldf.py --backend jax --jit` | **223 tests, 0 failures** |
| `test_quantity.py -k sphericaldf` | 9 passed |

The traced sphericaldf file is now **fully green**: 223 -> 8 -> 5 -> **0** failures
across #1214 and #163. Worth stressing that the 0 is real and not masked -- the
five entries were already removed from the ledger, so they ran as ordinary tests
and passed on their own.

Committed `41037190`, stacked on #1214, held from PR until #1214 lands.

### What is actually left in the jax-jit ledger

Ran all 14 remaining entries traced with `--runxfail`. They are not 14 problems:
**6 fail, 8 never run at all.**

The 8 are dead weight -- the test is slow-skipped under jax-jit, so the xfail
entry can never fire (Cautun20, DehnenBinney98, McMillan17, and five
`poisson_surfdens_potential[mock*Wrapper*]`). Pruning them is zero-risk and
takes the ledger 14 -> 6. If a slow-skip is ever lifted they come back, so that
belongs in the PR text.

The 6 real ones are three clusters:

| cluster | count | failure |
|---|---|---|
| AnyAxisymmetricRazorThinDisk | 4 | at_zero / at_infinity / 2ndDeriv assertions + surfdens `TypeError: do not accept array inputs` |
| DoubleExponentialDisk | 1 | same array-input TypeError via surfdens |
| test_ExpDisk_special | 1 | RazorThin R2deriv asserted-exception -- **deferred to Jo** |

Clusters 1 and 2 share one root cause and it is the tail #1212 deliberately did
not chase: both potentials carry `check_potential_inputs_not_arrays`, and the
traced surfdens path evaluates on an array of quadrature nodes. `_quad_needs_backend`
gates on tracer-ness precisely *because* "does this potential accept arrays" is
undecidable, so closing this means making those two potentials array-safe -- a
real migration, not a ledger tweak.

Queue after #1214 + #163: the 8-entry prune (small), then that migration
(larger). Not started -- one PR at a time.

### #1214 codecov, resolving as predicted

63.42% -> **86.18%** project coverage over the wake as shards uploaded, against
149 of 151 checks now green. That is the partial-upload phantom behaving exactly
as the #1212 discriminator says it should, so the two codecov reds still need no
action beyond letting the remaining 22 checks finish.

## Wake 16 (2026-07-28 ~16:3xZ) -- codecov cleared; a real defect found in my own merged code

### #1214 codecov: phantom confirmed, resolved

63.42% -> 86.18% -> **99.96%**. Both codecov checks green. The discriminator held
end to end: large project drop + shards still uploading = partial upload, wait.
No action was ever needed. #1214 now 152 pass / 2 pending (last test_backend
shard + all-clear).

### Traced measurement of four never-measured df suites

| suite | traced |
|---|---|
| test_qdf | 56 tests, 0 failures |
| test_evolveddiskdf | 35 tests, 0 failures |
| test_dynamfric | 7 tests, 0 failures |
| test_jeans | 5 tests, **1 failure** |

### The jeans failure is a defect I introduced in #1212

`test_sigmalos_wlog_zerobeta` returns all-NaN under trace. Chased it to a
one-line reproduction on LogarithmicHaloPotential:

| call | result |
|---|---|
| `lp.surfdens(1.0, inf)` numpy | 0.25 |
| `lp.surfdens(jnp(1.0), inf)` eager | 0.25 |
| `jax.jit(...)` **at #1210, before #1212** | `TracerArrayConversionError` |
| `jax.jit(...)` **after #1212** | **nan** |

So the test failed before *and* after -- the ledger looks unchanged -- but
**#1212 converted a loud crash into a silent wrong value**, which is strictly the
worse failure mode. numpy is 0.25 on both sides, so byte-identity is intact and
this is backend-only.

Root cause is my design error in `_quad_needs_backend(xp, absz)`: it decides
whether scipy.quad is usable by testing the concreteness of the **integration
limit** alone. With `z=numpy.inf` the limit is concrete, so the gate says "scipy
is fine" -- but `R` is a tracer, and scipy silently yields NaN instead of
raising. The gate has to consider tracer-ness of the **coordinate**, not just the
limit.

I got the hard part of #1212 right (concreteness is decidable where
"does this potential accept arrays" is not) and then keyed it off the wrong
argument. Recorded as task #164 with the fix sketch. Deliberately NOT folded into
#163: different file, different concern, and #163 is already verified.

Worth noting what caught this: nothing in CI would have. test_jeans.py had never
been run under --jit, so the ledger had no opinion about it. The gap was found
only because I went looking at previously unmeasured suites while blocked on CI.

### #164 started locally (WIP, not PR-ready)

Branch `backend-jit/quad-gate-coords` @ `9cecf59b`, stacked on #163.

Two of three parts done and verified:

1. **Gate fixed.** `_quad_needs_backend(xp, absz, *coords)` now asks
   `under_trace(absz, *coords)`. I did NOT write a predicate --
   `galpy.backend._namespaces.under_trace` already existed for exactly this
   ("the 'do I have a concrete value?' probes that pick an out-of-backend
   computation"), and `AnyAxisymmetricRazorThinDiskPotential.py:173` already
   calls `under_trace(R, z)`. Searching the backend utils first, per the
   placement rule, turned a new-helper job into a one-line import. It also
   catches `torch.compile`, which the old probe missed.
   Traced surfdens at a **finite** limit now matches numpy to 1.6e-15 / 2.8e-15
   -- that path was simply unreachable before.

2. **`symmetric_quad` added** to `backend/quadrature.py` (a quadrature concern
   with two Potential.py callers, so it belongs there, not in Potential.py).
   Infinite `b` splits into two *separate* semi-infinite halves rather than
   doubling one, because the integrand need not be even in z -- offset and
   tilted wrappers are not. Standalone check: 1/(1+x^2) over (-inf,inf) = pi to
   1.5e-15.

   The subtlety that cost me a wrong first attempt, worth remembering: **inside
   `jax.jit`, `xp.isfinite(concrete_constant)` still returns a tracer.** Every op
   in jit scope is traced, so asking the backend whether the limit is infinite
   always answers "unknown" and silently picks the finite rule. The limit's
   finiteness has to be read with numpy.

3. **Not done.** Infinite-limit surfdens is still NaN under trace. Localised: the
   rule is right in isolation and the finite path is right, so the NaN is
   produced inside the traced integrand at the huge nodes the reciprocal map
   generates -- `symmetric_quad` on `_dens` gives 0.25 eager and nan traced with
   identical inputs. Next step is to instrument which term goes non-finite.

Time-boxed there rather than pushing on: #1214 is one shard from mergeable and
the rule is one PR at a time. Committed as explicit WIP so the diagnosis is not
lost, with "do not open a PR from this commit as-is" in the message.

### #1214 MERGED, #1215 opened

`3aeca4fd` -- squash-merged into feat/backends-develop at 17:06Z on a full green
board (**154 pass, 0 fail, 0 pending**). Pre-merge audit against the three
criteria passed: no `if`-on-array-data (`hasattr` and `xp is numpy` are Python /
namespace checks), duplication *removal* is the PR's entire purpose, and
`_ensure_fE_interp` lands on `sphericaldf`, the common ancestor of both
subclasses.

The codecov arc end to end: 63.42% -> 86.18% -> 99.96%, both checks green. The
discriminator was right at every step and no intervention was ever needed. Worth
recording that the *first* reading (63.42%) looked alarming and the right move
was to wait ~90 min.

#163 rebased onto the post-squash base (the squash rewrote its parent), clean,
diff exactly the 3 intended files, conflict-marker swept, pushed, and opened as
**#1215**. jax-jit ledger 19 -> 14.

Also folded into #1215 during the audit: the ledger comment block #1214 added
above those five entries. My `sed` had removed the entries but orphaned the
comment, leaving text that claimed "the five below call the real fE" sitting over
unrelated RazorThin entries -- and that comment also guessed
`constantbetadf.py:735` as the culprit, which #163 disproved (the real site was
`as_numpy` at :657). Removing a wrong comment I wrote two PRs ago is exactly the
sort of thing the diff audit is for.

Queue: #1215 to green -> merge; then #164 (WIP, part 3 outstanding); then the
8-entry dead-ledger prune; then the RazorThin/DoubleExpDisk array-safety
migration.

## Wake 17 -- #164 diagnosed properly, and two of my own hypotheses killed

#1215 is early in CI (2/151). Spent the wait on #164, and the honest outcome is
that I was wrong twice before getting it right. Both corrections came from
measurement, not reasoning.

**Wrong #1 -- "the gate keys on the wrong argument."** I had claimed
`_quad_needs_backend` tested only the limit's concreteness and so mis-routed a
traced R. It does test only the limit -- but under jit `absz = xp.abs(z)` is
ALWAYS a tracer, because jax traces the op rather than folding it. So the
ORIGINAL gate already routed traced calls to the backend rule. Measured on
unmodified code: traced surfdens at a **finite** limit already matched numpy to
1.58e-15. I had told the user that path "was previously unreachable" -- it was
not. The `under_trace(absz, *coords)` change fixed nothing, so I reverted it
rather than ship churn.

**Wrong #2 -- "read the finiteness off the raw z."** There is no raw z.
`_surfdens` receives z as a `DynamicJaxprTracer`: `@backend_input` coerces it at
the boundary. Measured with a spy on the method.

**Actual defect, narrow:** `transformed_quad` cannot integrate an infinite range
-- its affine map sends the nodes to +-inf and the sum is nan. #1212 routed every
traced call there, inf included. That is the whole of it.

**Built and verified:** `symmetric_quad` in `backend/quadrature.py` -- two
separate semi-infinite halves (not doubled; the integrand need not be even in z,
offset/tilted wrappers are not). 1/(1+x^2) over (-inf,inf) = pi to 1.5e-15, and
both halves of LogHalo's density trace correctly at 0.125 each.

**Blocked, and deliberately so.** The finite-vs-infinite choice cannot be made
inside `_surfdens` -- z is already a tracer, its magnitude unknowable. Fixing it
needs either (a) detecting non-finite z at the decorated boundary before
coercion, or (b) a distinct whole-line entry point (z=None) so the choice is
structural rather than value-dependent. (b) is cleaner and jit-friendly but is an
API addition. That is Jo's call. I am flagging it rather than guessing, and left
an interim option in the task: make the infinite traced case RAISE, restoring
#1210's loud failure while keeping #1212's finite-limit traceability.

Also noted en route: MiyamotoNagai has its own analytic `_surfdens` which ALSO
returns NaN at z=inf under trace (arctan of inf/inf) -- a separate site with the
same symptom, worth covering by the same tests.

Committed as WIP `80684184`, "do not open a PR from this commit". Two wrong
hypotheses cost most of a wake, but the branch now carries a correct diagnosis
instead of a plausible-looking fix that did nothing.

## Wake 18 -- inert-ledger prune prepared

#1215 early in CI (25/151); codecov at 38.67% with 126 shards pending, the same
partial-upload phantom, no action.

Used the wait to finish the first half of the ledger burndown. Confirmed the
eight suspect jax-jit entries are **inert**: they appear verbatim in
`backend_slow_skip.txt` under `jax`, and `jax-jit` inherits the eager lists
(#1210), so the tests never run traced and the xfail entries have no effect.
Running all eight with `--backend jax --jit` gives **8 skipped, 0 failures**.

Prune committed on `backend-jit/prune-inert-ledger` @ `7c44d07b`, stacked on
#1215, held until that merges. jax-jit ledger **14 -> 6**.

Likely origin worth recording: a heavy test that hits the session timeout records
FAILED, so a regen at that moment ledgers it; the later slow-skip then makes the
entry inert without removing it. That is the same timeout-vs-xfail interaction
already in the notes, seen from the other side.

What remains is the genuine tail: four AnyAxisymmetricRazorThinDisk failures plus
one DoubleExponentialDisk sharing a root cause (both carry
`check_potential_inputs_not_arrays` while the traced surfdens path evaluates on
an array of quadrature nodes), and test_ExpDisk_special, the
asserted-exception-under-trace policy question for Jo.

The prune also required checking a detail I had gotten wrong once: the slow-skips
are recorded under `jax`, not `jax-jit`. Grepping for `^jax-jit ` in
backend_slow_skip.txt returns nothing, which briefly looked like the entries were
not slow-skipped at all. The inheritance rule I wrote in #1210 is what makes them
apply.

### Traced sweep: three more clean, one CRASH

| suite | traced |
|---|---|
| test_coords | 97 tests, 0 failures |
| test_conversion | 18 tests, 0 failures |
| test_streamspraydf | 38 tests, 0 failures |
| **test_scf** | **process aborted, exit 134 (SIGABRT), core dumped** |

`pytest tests/test_scf.py --backend jax --jit` kills the interpreter at 76%.

I did NOT take the first observation at face value: it happened while four
jax-jit suites ran concurrently, so memory pressure was a live alternative.
Re-ran test_scf **alone** -- identical abort, exit 134. Real fault, not resource
exhaustion.

Crash site, from the faulthandler traceback:

    tests/test_scf.py:1546  test_tdep_c_dynamical_friction_dens
    -> Orbits.py:2369 integrate -> Orbits.py:1957 _integrate_impl
    -> integrateFullOrbit.py:715 integrateFullOrbit_c      <-- the ctypes call

The test integrates a **time-dependent** SCFPotential (`Acos=lambda t: ...` with
a tgrid) plus Chandrasekhar friction through `method="dop853_c"`. The bridge
marshals potential arguments with `numpy.require(..., float64)` + ndpointer;
under a trace the t-dependent coefficients are backend arrays, so it is handed
something it cannot legally point at and aborts instead of raising. That is the
ctypes-bridge / forced-backend leak already in the notes, in a new place.

**Why this is worse than a failure:** SIGABRT writes no junitxml at all, so every
other test in the file becomes unreportable. Same blind-spot class as a timed-out
shard -- a jit dispatch run would silently lose test_scf whole.

Almost certainly pre-existing (test_scf had never been run traced; the branch
under test changes only constantbetadf), but that must be confirmed on a clean
base before any fix. Recorded as task #165 with two separable pieces: a cheap
skip entry to keep the rest of the file reportable, and the real fix (marshal
under `use("numpy", force=True)` so the bridge never sees a backend array).

Not pinged to the phone: it does not block my progress, and it only affects the
dispatch-only jit mode, which is not in the merge gate.

### RETRACTION: the test_scf crash was my environment, not galpy

I reported `test_scf --jit` aborting with SIGABRT and filed it as a
ctypes-bridge/forced-backend leak. **That was wrong and I have retracted it.**

On a clean worktree at develop head 3aeca4fd the same full file runs **exit 0,
all pass**. The single test passes alone too.

Cause: when I created /var/tmp/logstartt I **symlinked** the C extension from the
main repo rather than building it. The main repo sits on feat/backends @ dbb09ffc
-- a different branch, several PRs back -- so the worktree ran its own Python
against a foreign libgalpy. md5 differs from the clean worktree's real .so.
Mismatched Python/C ABI aborts, which is precisely what a SIGABRT with no Python
traceback looks like.

What misled me: I *did* assert `galpy.__file__` pointed at the worktree, which is
the check I have written down. That check does not cover the C extension, which
has separate provenance. Saved as memory `worktree-so-provenance`.

Also worth noting what saved me from shipping the bad report: running the
pre-existence check on a clean base. I nearly skipped it as a formality since the
crash "obviously" predated my branch. It was the step that falsified the whole
finding.

**Corrected measurement:** test_scf traced is GREEN, joining test_coords,
test_conversion, test_streamspraydf, test_qdf, test_evolveddiskdf and
test_dynamfric as traced-clean.

**What survives unaffected:**
- #164 (surfdens infinite-limit NaN) re-verified on the clean worktree with a
  matching .so: numpy 0.25, traced nan. Stands.
- The test_jeans traced failure was independently reproduced in /var/tmp/fe and
  /var/tmp/nogl, both with real .so files. Stands.
- #1215's correctness is being validated by CI against a proper build, which is
  the authoritative check regardless of my local worktree.

### #164 blast radius measured -- narrower than feared, and it is TWO sites

Swept 10 potentials on the clean worktree (matching .so), numpy vs traced
`surfdens(R, inf)`:

| potential | numpy | traced | verdict |
|---|---|---|---|
| LogarithmicHalo | 0.25 | **nan** | broken -- generic quadrature path |
| MiyamotoNagai | 0.1849 | **nan** | broken -- its OWN analytic `_surfdens` |
| DoubleExpDisk | 0.0716 | 0.0716 | OK |
| NFW, Hernquist, Plummer, Jaffe, PowerSpherical, Isochrone, Kepler | nan | nan | numpy also undefined -- not a regression |

Only **two** potentials actually lose a good value, and they break for different
reasons: LogHalo hits the transformed_quad infinite-range problem; MiyamotoNagai
NaNs at `arctan(inf/inf)` inside its closed form, nothing to do with quadrature.

The useful third data point is **DoubleExpDisk**, which works traced at z=inf.
Its analytic `_surfdens` takes the z->inf limit correctly with no branching at
all. That is an existence proof for a third design option I had not considered:
make the implementations **inf-safe** rather than detecting inf. It fixes
MiyamotoNagai cleanly, but cannot fix the quadrature fallback -- no finite-range
rule spans an infinite range -- so the likely answer is (c) for the closed form
plus (a) or (b) for the quadrature path.

Recorded in task #164 so Jo has the measured scope, not just the symptom, when he
makes the call. Also recorded there: the two hypotheses of mine that measurement
killed, so they are not retried.

### Consolidated traced (jax --jit) status, 2026-07-28

Re-verified `test_dynamfric` (7/0) and `test_qdf` (56/0) against the corrected
.so -- identical to the earlier numbers, so the bad-.so window did not corrupt
them. Full picture from this vacation's measurements:

| suite | traced result |
|---|---|
| test_actionAngle | 207 / 0 failures |
| test_sphericaldf | 223 / 0 (was 8 -> 5 -> 0 across #1214 + #1215) |
| test_diskdf | 129 / 0 |
| test_coords | 97 / 0 |
| test_qdf | 56 / 0 |
| test_streamspraydf | 38 / 0 |
| test_evolveddiskdf | 35 / 0 |
| test_conversion | 18 / 0 |
| test_dynamfric | 7 / 0 |
| test_scf | 0 failures (the SIGABRT was my .so mismatch, retracted) |
| **test_jeans** | **1 failure** -- the #164 surfdens(inf) consumer |
| test_potential | 6 real failures after the pending prune (2 clusters + 1 policy Q) |
| test_orbit / test_orbits | exceeds 90 min locally; belongs on a sharded dispatch |

So the traced tail across everything measured is now **one df failure (#164) and
six potential failures**, of which five are two root causes and one is a policy
question for Jo. That is the honest state of "is galpy jittable" for the parts
that have been measured -- test_orbit remains the big unmeasured block.

## Wake 19 -- #1215 is runner-starved, not stalled

#1215 sat at 103 pass / 48 pending across a whole wake, which looked like a
stall. It is not: the all-backend run has **22 jobs queued**, 6 in progress, 16
completed. GitHub-hosted runner capacity, nothing to fix. Worth writing down
because "pass count unchanged for an hour" is not evidence of a hung job -- check
queued-vs-in_progress before reacting.

Started sampling the one big unmeasured block: **test_orbit under --jit**. I had
written it off as ">90 min locally", but CI uses pytest-split, so the same
mechanism gives a bounded sample: `--splits 10 --group N` for groups 1-5, i.e.
50% of the file, each group ~1/10 of the runtime. Partial coverage of the largest
remaining unknown beats no coverage.

### test_orbit traced, sampled: one real failure, one gap, and a caveat

pytest-split made the ">90 min" block measurable: `--splits 10`, groups 1-10 run
in parallel batches.

| groups | result |
|---|---|
| 1,2,3,4,5,6,9,10 | 0 failures |
| **8** | **1 failure -- test_eccentricity** |
| **7** | **no XML -- hit the 3000s timeout, UNMEASURED** |

**test_eccentricity**: passes `--backend jax` eager, FAILS `--backend jax --jit`.
"Eccentricity of a circular orbit is not equal to zero". Reproduced in isolation,
and it is in **none** of the three lists -- completely unledgered, so a jit
dispatch run would show it as a genuine red. Jit-specific, so a tracing-path
defect rather than a tolerance issue. Suspect rperi/rap, which are on the
deferred SETUP-path list where `as_numpy` was the scoped fix -- to be verified,
not assumed. Task #166.

**The caveat that keeps this honest.** I nearly reported "test_orbit is
traced-clean" off the first five groups. Checking the skip lists first: test_orbit
has only 2 jax xfail entries and 0 jax-jit ones, so almost nothing is
xfail-masked -- but **112 slow-skip entries under `jax`** (inherited by jax-jit)
plus 8 hard skips. Of ~590 collected, about a fifth never runs. So "0 failures"
means no unexpected failures *among tests that ran*, which is a much weaker claim
than "test_orbit is jittable". Every summary of this needs to say so.

Also notable: group 8 took 2154s on its own. The jax eager-XLA slowness is why
this file is a dispatch-only proposition, and why group 7 timed out.

### test_eccentricity root-caused: by design, and it consolidates the RazorThin tail

Pulled the exact failure out of the group-8 junit rather than guessing:

    Eccentricity of a circular orbit is not equal to zero by 2.94267e-11
    for potential AnyAxisymmetricRazorThinDiskPotential and integrator dopr54_c
    assert (5.42463803e-06)**2 < 1e-16

`AnyAxisymmetricRazorThinDiskPotential._bk_dispatch` switches implementation on
tracedness on purpose:
- concrete, non-grad input -> scipy's **adaptive** quad (accurate)
- `requires_grad` or `under_trace(R, z)` -> in-backend fixed-order
  Gauss-Legendre, a differentiable **approximation**

So under a trace the forces carry ~1e-6 error and a nominally circular orbit
picks up ecc ~5e-6. A 1e-16 tolerance is unmeetable on that path **by
construction**. Not a defect -- it is the differentiability-vs-adaptive-accuracy
trade-off the dispatch exists to make.

My minimal repro (MWPotential2014 + dop853_c) gave e=0 exactly under the jit
harness, which is why the parameterised failure text mattered: the bug lives in
one potential, not in tracing generally.

**Consolidation.** AnyAxisymmetricRazorThinDisk now accounts for FIVE traced
failures in two groups:
 (a) GL-approximation vs tight tolerances -- test_eccentricity, and very likely
     2ndDeriv / at_zero / at_infinity (to verify individually);
 (b) scalar-only decorator vs array quadrature nodes -- poisson_surfdens.
The "six real jax-jit failures" are really *one potential with two
characteristics* plus test_ExpDisk_special's policy question.

**Action:** ledger it (autonomous-safe, matches its siblings) and fold the
alternative -- a backend-aware tolerance, loose only on the traced path -- into
the same proposal to Jo that covers #164. Both are "the differentiable path is
less accurate than scipy" questions, and the tolerance-loosening rule requires
his approval per batch. Explicitly NOT raising the GL order to force accuracy:
that would tax every traced call to satisfy a test, and eager users currently get
scipy's accuracy for free.

## Wake 20 -- #1215 MERGED, #1216 opened

`ebf0215f` -- squash-merged on a full green board (**154 pass, 0 fail, CLEAN**).
Pre-merge audit passed on all three criteria; a mechanical check of the diff
confirmed no new `if` or `def` lines anywhere in `galpy/`, i.e. the change really
is a spline swap plus dropping the clamp.

Traced `test_sphericaldf.py` is now 223 / 0 on develop, unmasked.

### #1216: reconcile the jax-jit ledger with reality

Promoted the queued prune and strengthened it with this wake's finding, so the PR
is one coherent idea rather than two:

- **Drop 8** entries that can never fire (slow-skipped under `jax`, inherited by
  `jax-jit`) -- verified 8 skipped / 0 failures.
- **Add 1** that does: `test_orbit::test_eccentricity`, failing under --jit and in
  none of the three lists, so a jit dispatch run would have shown a genuine red.

jax-jit ledger **14 -> 7**, and those 7 are now all genuinely-failing tests
rather than a mix of real and inert.

Verified the new entry is load-bearing: `x` with the entry, `F` under
`--runxfail`.

Deliberately ledgered rather than loosening the tolerance. A backend-aware
tolerance (loose only on the traced path, numpy untouched) is the better
long-term answer, but that is a maintainer call per the tolerance rule, so it is
proposed in the PR body alongside #164's design question instead of done
unilaterally.

Jo checked in mid-wake asking for the three #164 options; answered briefly
(boundary detection / structural whole-line entry point / inf-safe formulas, with
DoubleExpDisk as the existence proof for the third) and continued.

### Correction: the RazorThin tail is THREE characteristics, not two

I claimed last wake that AnyAxisymmetricRazorThinDisk's five traced failures
split into two groups. Ran all of them with `--runxfail` and read the actual
messages instead of inferring -- it is three:

| tests | message | class |
|---|---|---|
| test_eccentricity, test_2ndDeriv | tolerance assertions | GL less accurate than scipy |
| test_poisson_surfdens | `TypeError: ... do not accept array inputs` | scalar-only decorator |
| test_potential_at_zero, test_potential_at_infinity | **"gave NaN"** | GL returns NaN at degenerate args |

The third is not a tolerance issue at all, which is what I had assumed. And
`at_infinity` is the **same family as #164** -- "infinite argument -> NaN under
trace". So the design decision on `surfdens(R, inf)` probably wants to cover
`_evaluate(R, z=inf)` as well; better to put that to Jo as one question than two.

Net picture of the jax-jit tail after #1216: one potential with three traced
characteristics, one scalar-only issue shared with DoubleExpDisk, and one policy
question (test_ExpDisk_special). Nothing unexplained left in it.

## Wake 21 -- #164 implemented and tested (Jo approved 3-where-possible + 2-as-fallback)

Jo checked in, asked for the three options, then approved **3 where possible, 2
as fallback**. Implemented on `backend-jit/surfdens-wholeline`, stacked behind
#1216: `b98ecb6b` (fix) + `ae750fd6` (tests).

**Option 2 alone covered all three broken potentials**, including MiyamotoNagai,
which I had expected to need option 3. The whole-line path integrates the density
rather than MN's arctan closed form, so the `inf/inf` NaN never arises. Option 3
is therefore still open only for `_evaluate(R, z=inf)` -- the at_zero/at_infinity
NaNs -- not for surfdens.

Traced results, all previously NaN:
| potential | traced | rel vs numpy |
|---|---|---|
| LogarithmicHalo | 0.25 | 2.8e-15 |
| MiyamotoNagai | 0.18494223 | 7.0e-13 |
| DoubleExpDisk | 0.07164229 | 3.8e-14 |

**test_jeans: 5/5 traced (was 4/1), 5/5 numpy, 5/5 torch.** #164's consumer is
closed.

### The byte-identity trap I nearly walked into

My first version had `z=None` use GL on numpy too. That would have shifted
`jeans.sigmalos` numpy results (0.25 -> 0.2499999999999998) -- a silent
numpy-path change of exactly the kind the notes say to always surface. The
whole-line path is now dual: numpy makes the *same* `scipy.quad(-inf, inf)` call
the `z=inf` spelling already made, verified byte-equal, and only the backend uses
GL. The test asserts hex equality rather than a tolerance so a future drift fails
loudly.

### What the tests taught me

The byte-equality test failed for DoubleExpDisk, and it was right to. That
potential's analytic `_surfdens` already handled z=inf, so z=inf and z=None take
different (both correct) routes and agree to 4e-14, not exactly. My test premise
had been too broad. Split it: byte-equality for the two that route through the
integral, and a separate assertion that the integral REPRODUCES DoubleExpDisk's
known closed form -- which is a stronger statement than self-consistency.

That leaves one rough edge, stated in the commit rather than hidden: for a
potential with an inf-capable closed form, z=None and z=inf can differ at
quadrature level. Letting such potentials serve z=None from their closed form is
a natural opt-in; left as a follow-up rather than guessed at.

Also of note: `z=None` returns a correct finite value where `z=inf` gave NaN even
on numpy (NFW 0.2747), because the closed form cannot express the limit.

### CI

#1216 is fully runner-starved -- 71 jobs queued, ZERO running, ~27 min after
creation, because the develop merge and #1215's tail are still holding capacity.
Nothing to fix.

### test_orbit group 7 closed out: not a timeout, a force-exit

The group-7 gap was NOT a timeout. With a 7000s limit it produced a full progress
line -- `...............sss.....s.....s...ss.ssss...sssssss`, **50 marks, no F or
E** -- then stopped with no summary and no junitxml. That is the conftest
force-exit (the torch+fork-deadlock guard from #1083), which bypasses pytest's
XML write.

Group 7 collects **59**; 50 reported clean, so **9 are unobserved, not failing**.

Worth flagging beyond this one group: the force-exit destroys junitxml, so a jit
dispatch run would lose an entire group's results the same way -- the same
blind-spot class as a timed-out shard, and the same class as the SIGABRT I
mis-attributed earlier. When a shard reports nothing, the answer is never "it
passed".

test_orbit under --jit is therefore: **0 failures across every group except the
by-design test_eccentricity (ledgered in #1216), with 9 of ~590 unobserved** --
and the standing caveat that ~a fifth of the file is slow-skipped and never runs
at all.

### #1216

Showed 1 fail mid-wake; it is the codecov partial-upload phantom again (66.4%
with 113 shards pending), not a real red. Runner queue is draining: 39 pass.

## Wake 22 -- option 3 delivered: RazorThin degenerate radii

#1216 still runner-starved (53 pass / 99 pending, codecov phantom at 66.4%).

Did the remaining half of what Jo approved: **option 3, inf-safe formulas**, for
the at_zero / at_infinity NaNs. Commit `c829ece3`.

Instrumented the three panels of the a=R split rather than guessing, and the two
degenerate radii turn out to fail for *different* reasons:

| R | panel [0,R] | panel [R,2R] | panel [2R,inf) |
|---|---|---|---|
| 0 | **nan** | **nan** | 0.785 (finite) |
| 1 | 0.237 | 0.0517 | 0.00205 |
| inf | **nan** | **nan** | **nan** |

At R=0 the first two panels have zero width but the integrand is 0/0 there, so
each is `0*nan`; the tail alone is already the whole [0,inf) integral, which is
exactly right. At R=inf every panel spans an infinite range, and all these
quantities vanish as R->inf, so the answer is 0.

Both guarded with `xp.where` in the **shared** `_bk_split_quad`, so all six _gl
kernels benefit rather than just `_evaluate`, with each dead branch evaluated at
a harmless R so a nan cannot leak through a gradient.

| | numpy | traced |
|---|---|---|
| Phi(0) | -2.7855674 | **-2.7855674** (was nan) |
| Phi(inf) | 0 | **0** (was nan) |

Finite radii unchanged (6.7e-15 / 4.7e-13 / 2.5e-11 at R=0.3/1/3 -- the
pre-existing GL-vs-scipy floor, not a shift). Backend-only change: the concrete
path goes to scipy via `_bk_dispatch`, so numpy is untouched by construction.

**jax-jit ledger 7 -> 5.** The two remaining RazorThin entries are genuinely
different problems, not more of the same: test_2ndDeriv is the by-design GL
accuracy limit, test_poisson_surfdens is the scalar-only-inputs decorator.

Tests assert VALUES against numpy, not "not NaN" -- Phi(0) is a real number worth
pinning and a finiteness check would pass on garbage. Confirmed load-bearing by
stashing the guard: the degenerate-radii test fails without it.

A small process note: the first commit attempt was rejected by ruff-format (it
reformatted a file). Worth remembering that a pre-commit hook that *modifies*
files fails the commit, so `git log` after committing is the check that it
actually landed.

## Wake 23 -- first REAL red on #1216, and it is a known flake

#1216 reached 117 pass / 34 pending and grew a second failure. Unlike the codecov
phantom this one is a build job, so it needed reading rather than dismissing:

    build (windows-latest, 3.12, test_orbit.py test_orbits.py ...)
    FAILED tests/test_orbits.py::test_from_name_values
      pyvo.dal.exceptions.DALServiceError: Unable to access the capabilities endpoint

That is the known SIMBAD/VizieR network flake, not a real red -- and worth
noting that #1216 only edits the xfail ledger, which a plain (non---backend) run
does not even consult, so a genuine regression from this PR in a Windows orbit
shard was implausible on its face. The log confirmed it rather than assuming it.

**New operational detail:** `gh run rerun <id> --failed` refuses with "This
workflow is already running" while any job in the run is still in flight. So a
mid-run flake cannot be re-run immediately; it has to wait for the run to settle.
Worth remembering, since the instinct is to re-run the moment a flake appears.

### torch parity for the degenerate-radii guards

`_bk_dispatch` selects the GL path on `requires_grad` as well as on tracedness,
so torch reaches the guarded split by a different door than jax. Added the
matching grad-vs-FD check rather than trusting jax to speak for torch -- torch
lands on exactly the jax figures (1.8e-10 / 2.4e-11 / 3.5e-11 at R=0.3/1/3).
Commit `bed56efc`; the file is 17 passed.

Also: the ruff-format rejection bit me a SECOND time on this commit, one wake
after writing the lesson down. A hook that *modifies* files fails the commit, and
the only reliable check is `git log` afterwards -- which is exactly how I caught
it both times.

#164 branch now: whole-line surfdens, its cross-backend tests, the degenerate
radii guards (ledger 7 -> 5), the gradient verification, and torch parity.

## Wake 24 -- SIMBAD outage, five shards, one cause

#1216 went to 5 reds. All five were the same thing:

    FAILED tests/test_orbit(s).py::test_from_name_values / _errors
      pyvo.dal.exceptions.DALServiceError: Unable to access the capabilities endpoint

SIMBAD is unreachable, so every shard that runs the from_name tests failed
together. A cluster of reds appearing at once looks alarming and is exactly when
it is worth reading one log before reacting -- five shards with one cause is a
service outage, not five problems.

Re-ran the failed jobs; four cleared immediately (150 pass). The fifth belonged
to a third Linux/Mac run still in flight, which `gh run rerun --failed` refuses
while any job in the run is live, so it had to wait for that run to complete
before it could be re-queued. Now re-running too.

The signal that mattered underneath the noise: **All-backend tests
completed/success** -- the shard that actually exercises this PR's ledger change
is green.

Also started a full traced `test_potential.py` (5 parallel splits) on the #164
branch, to verify the ledger 7 -> 5 claim end to end before it becomes a PR:
only the 5 remaining entries should xfail, with nothing unexpected failing.

### #164 branch verified end to end, traced

Ran the full `test_potential.py` under `--backend jax --jit`, 5 parallel splits:

| group | tests | failures |
|---|---|---|
| 1 | 266 | 0 |
| 2 | 266 | 0 |
| 3 | 266 | 0 |
| 4 | 266 | 0 |
| 5 | 262 | 0 |

**1326 tests, 0 failures**, 122 skipped (which includes the 5 remaining ledgered
xfails). That is the end-to-end confirmation of the ledger 7 -> 5 claim: the
degenerate-radii guards flip at_zero and at_infinity, and nothing else in the
file regressed under a trace. Checking the whole file rather than just the two
flipped tests is the point -- a guard in a SHARED helper (`_bk_split_quad` feeds
all six _gl kernels) could easily have perturbed something else.

#1216: SIMBAD flakes all cleared on re-run, **152 pass / 0 fail**, waiting on the
last re-run shard plus all-clear.

## Wake 25 -- #1216 MERGED, #1217 opened (the #164 fix)

`e2dd0759` -- merged on a full green board (154 pass, 0 fail). Every SIMBAD flake
cleared on re-run; nothing about them was real.

**#1217** now open with the whole #164 arc, five commits:

1. `surfdens(R, z=None)` -- the whole line asked for STRUCTURALLY, because
   `@backend_input` coerces z at the boundary and a traced z=inf is not
   inspectable. Traced LogHalo/MiyamotoNagai/DoubleExpDisk all correct where two
   were NaN. numpy asserted byte-equal (hex, not tolerance).
2. Cross-backend tests for that path.
3. RazorThin degenerate radii R=0 and R=inf, guarded in the shared
   `_bk_split_quad` so all six _gl kernels benefit. Phi(0) and Phi(inf) now match
   numpy exactly where both were NaN.
4. Gradient verification -- grad-vs-FD with h-convergence, because commit 3's
   claim that the dead branches cannot poison AD was reasoning, not evidence.
5. Torch parity for the same, since `requires_grad` reaches the guarded split by
   a different door than tracing does.

jax-jit ledger **7 -> 5**, and `test_jeans` under --jit goes 4/1 -> **5/5**.

Verification worth naming: the **full** traced test_potential.py, 1326 tests,
0 failures. The guard lives in a shared helper feeding six kernels, so
"the two tests I aimed at now pass" would have been the wrong evidence.

The PR body states the one rough edge rather than burying it: for a potential
with an inf-capable closed form, z=None and z=inf can differ at quadrature level
(~4e-14), because z=None integrates the density. The same property is why z=None
returns a correct finite value where z=inf gives NaN even on numpy (NFW 0.2747).

## Wake 26 -- a blind spot: torch --jit has never been measured

#1217 is early in CI (2/151).

Went looking for the next ledger cluster and found something more interesting.
The remaining jax-jit entries are 5, and three of those are by-design or a policy
question for Jo, so that ledger is essentially at its floor. Counting what the
ledger actually contains:

    51  jax
     5  jax-jit
    99  torch
     0  torch-jit     <-- never measured

The harness supports `--backend torch --jit` -- conftest says so explicitly, and
the "<backend>-jit" inheritance rule is generic -- but **every traced measurement
this vacation has been jax-only**. torch traced is a complete blind spot, and the
mission is jax *and* torch.

That reframes the next step. Squeezing the last two fixable jax-jit entries (the
scalar-only `check_potential_inputs_not_arrays` pair) is a real migration for a
2-entry gain; measuring torch-jit is unknown territory that could be anywhere
from "works" to "nothing traces". Measurement first, as task #167.

Started with test_coords and test_conversion -- the two fastest suites under
jax-jit (12s, 14s) -- to learn whether torch tracing works at all before
committing to a sweep. Deliberately not fixing anything until the scale is known.

Carried forward from the #165 retraction: a shard that reports NOTHING is not a
pass, and with torch.compile a hang is a different animal from a failure.

### torch --jit WORKS -- first measurement ever

| suite | torch --jit | jax --jit (for scale) |
|---|---|---|
| test_coords | 97 tests, **0 failures**, 9s | 12s |
| test_conversion | 18 tests, **0 failures**, 110s | 14s |

So torch tracing is not broken -- it just has never been looked at. That is the
best possible answer to the blind spot: no emergency, but a whole backend's
traced status is unmeasured and evidently in decent shape.

The interesting number is the **8x slowdown on test_conversion** (110s vs 14s)
while test_coords is actually *faster* (9s vs 12s). torch.compile warmup
dominates when there are many distinct graphs to compile and is free when there
are few. That is a real scaling caveat for any future torch-jit CI shard, and
worth knowing before proposing one.

Sweeping the suites already measured under jax-jit next, with a per-run
TORCHINDUCTOR_CACHE_DIR so parallel runs cannot collide on the inductor cache.

### torch --jit sweep: 10 suites, 702 tests, ZERO failures

| suite | tests | torch --jit | jax --jit (where known) |
|---|---|---|---|
| test_sphericaldf | 223 | 0 fail, 1042s | 0 fail |
| test_diskdf | 129 | 0 fail, 611s | 0 fail |
| test_coords | 97 | 0 fail, 9s | 0 fail, 12s |
| test_scf | 94 | 0 fail, 278s | 0 fail |
| test_qdf | 56 | 0 fail, 1179s | 0 fail |
| test_streamspraydf | 38 | 0 fail, 314s | 0 fail, 186s |
| test_evolveddiskdf | 35 | 0 fail, 374s | 0 fail |
| test_conversion | 18 | 0 fail, 110s | 0 fail, 14s |
| test_dynamfric | 7 | 0 fail, 605s | 0 fail |
| test_jeans | 5 | 0 fail, 139s | 0 fail (with #164) |

**Zero torch-jit ledger entries are needed for any of it.** The blind spot turned
out to be a blind spot in the *measurement*, not in the code: torch tracing has
been working the whole time and nobody had looked.

Two things this changes:

1. `test_jeans` passes under torch-jit only *because* this branch carries the
   #164 fix. So #1217 repairs torch tracing as well as jax -- the whole-line
   surfdens change is backend-agnostic and I had only claimed eager-torch in the
   PR body. Worth adding there.
2. The cost profile is **uneven, not uniformly worse**. test_coords is FASTER
   under torch (9s vs 12s) while test_conversion is 8x slower and test_dynamfric
   ~2x. torch.compile warmup scales with the number of distinct graphs, not the
   number of tests. Any future torch-jit CI shard should be composed by graph
   count, not test count -- a naive "torch-jit costs about what jax-jit costs"
   would be wrong by an order of magnitude on the wrong suite.

test_potential under torch --jit now running (5 splits); test_orbit remains the
one large unmeasured block on both traced backends.

### torch --jit on test_potential: 4/5 groups clean, and a caveat I checked

| group | tests | failures |
|---|---|---|
| 1 | 266 | 0 (253s) |
| 3 | 266 | 0 (1215s) |
| 4 | 266 | 0 (267s) |
| 5 | 262 | 0 (585s) |
| **2** | — | **still running** |

Before calling that clean I checked two things that could have made it vacuous.

**Is torch.compile actually engaged, or silently falling back to eager on graph
breaks?** If it fell back, every "pass" would be a numpy/scipy pass wearing a
traced label. Probed inside a real galpy call under `gb.jit("torch")`:

    is_compiling = True ,  under_trace = True ,  value = -0.9422394447085976

So torch.compile IS on, `under_trace` correctly detects it, the GL path is taken,
and the value matches numpy. torch tracing is real, not a fallback.

**Are the jax-jit failures merely masked by inherited `torch` ledger entries?**
No -- **zero** of the four (2ndDeriv[AnyAxisym], poisson_surfdens[AnyAxisym],
poisson_surfdens[DoubleExpDisk], ExpDisk_special) appear in the torch ledger.

Which leaves a genuine and interesting asymmetry: tests that FAIL under jax
tracing appear to PASS under torch tracing. But only partly measured -- **2 of
those 4 live in group 2, which has not finished**. So the honest statement today
is: 1060 test_potential tests measured under torch --jit with 0 failures, with
the two most interesting cases still outstanding. Not "torch-jit is clean on
test_potential".

Worth understanding once group 2 lands: the two tracers disagree about the same
code, which usually means one of them is taking a different branch (torch's
`under_trace` goes through `torch.compiler.is_compiling()`, jax's through tracer
type). A difference there is worth knowing regardless of which way it falls.

### The torch-jit green was partly HOLLOW -- and it corrects my own earlier call

test_potential under torch --jit finished: all 5 groups, **1326 tests, 0
failures**. But group 2 had 64 skips against 3-14 elsewhere, which was worth
pulling on rather than banking.

The two tests that FAIL under jax --jit --
`test_poisson_surfdens_potential[AnyAxisymmetricRazorThinDisk]` and
`[DoubleExponentialDisk]` -- came back **PASSED** under torch --jit, in no ledger
and not skipped. Tests disagreeing between tracers needs a reason, so I spied on
the gate inside a `gb.jit("torch")` region:

    _quad_needs_backend -> [False]     # False = take scipy.integrate.quad

So under torch tracing surfdens takes the **scipy branch**; the traced GL path --
the one that fails under jax -- is never entered. The torch pass is **vacuous**:
scipy runs node-by-node with scalars, the scalar-only decorator is satisfied, and
the test goes green without exercising the traced path.

torch.compile itself is genuinely on (is_compiling=True inside a galpy call), so
this is not a global fallback -- it is this one gate being fooled. And
`under_trace`'s own docstring predicted it: dynamo turns `float(x)` into a
symbolic scalar, so a concreteness probe "wrongly takes the concrete branch and
drags the scipy/numpy code into the graph".

**This corrects a call I made earlier.** In #164 I wrote a change switching this
gate to `under_trace(absz, *coords)`, measured that it did not fix the jax NaN,
called it speculative churn, and reverted it. That was right about jax and wrong
about torch: `under_trace` consults `torch.compiler.is_compiling()` instead of a
float() probe, which is exactly what this needs. The change was correct for a
reason I had not measured yet -- I had tested the hypothesis I happened to have,
not the space of reasons the change might matter.

Filed as task #168, to do after #1217 lands (same function). Expect the fix to
make the torch poisson_surfdens tests start FAILING like jax's -- which is the
right outcome and gets ledgered. A loud shared failure beats a silent divergence.

So the honest torch-jit headline is: 2028 tests measured across 11 suites with 0
failures, but at least the surfdens-touching subset of that green is not evidence
the traced path works.

## Wake 27 -- a codecov failure that was NOT the phantom

#1217 showed codecov/patch + codecov/project red. My reflex by now is "partial
upload, wait" -- and that reflex was wrong here. Checking the number rather than
assuming:

    codecov project = 99.95%   (true is ~99.6; a partial upload reads 38-86%)

Essentially complete. So by my own discriminator -- flat project + failing =
REAL gap -- these were genuine. Good reminder that a heuristic that has been
right five times running is exactly the one you stop checking.

**The gap:** `symmetric_quad` had exactly ONE caller, the `z=None` path, which
always passes `numpy.inf`. Its finite branch was never executed by anything.
Uncovered, untested, and speculative generality besides.

**The fix, which is better than a coverage patch:** the finite surfdens paths
were already spelling out

    transformed_quad(xp, f, -absz, absz, n=50, interior_point=0.0)

which is exactly `symmetric_quad(xp, f, absz, ...)`. Routing them through it
removes the duplicated spelling and covers the branch through REAL USE, rather
than through a test written to satisfy a number. Three call sites now, one
helper.

Safe by construction: symmetric_quad only takes its infinite branch for a
*concretely* infinite limit, and those two sites are reached only when the limit
is traced -- so they land on the same transformed_quad call as before. Verified:
Hernquist/NFW traced finite surfdens unchanged at 8e-16..3.3e-15, 141 tests pass.
numpy never reaches the code at all.

Pushed as `cc585e24`.

## Wake 28 -- torch.compile can return a silently WRONG surfdens

The most serious correctness finding of the vacation, and I got it wrong twice
before getting it right. Recording the path, not just the answer.

**The fact, reproducible with no instrumentation:**

    AnyAxisymmetricRazorThinDisk.surfdens(1.0, 0.5, forcepoisson=True)
      numpy / torch eager :  0.06621732146004472
      under gb.jit("torch"):  2.2352231511396381e-15    <-- ~13 orders out
      under jax.jit        :  raises TypeError (GL path, as ledgered)

**Order-dependent.** In a loop that first compiled surfdens for Hernquist / NFW /
MiyamotoNagai, the same AnyAxisym call returned the CORRECT value. So it depends
on torch.compile's compile history within the process. Order-dependent silent
wrongness is worse than deterministic wrongness -- a suite can go green while a
user's script is wrong.

**Two wrong calls of mine on the way, both worth keeping:**

1. I first saw 2.235e-15 with a monkeypatched spy installed, then saw the loop
   give the right answer, and concluded the wrong value was an instrumentation
   artifact. **That was wrong** -- it reproduces with no patching. I had one real
   instrumentation burn already this vacation (the symlinked .so), and pattern-
   matched to it too eagerly.
2. I then hypothesised a cold/warm cache interaction. **Also wrong** -- cold and
   warm both give the wrong value in isolation.

The lesson is narrow and useful: "it didn't reproduce" is only evidence if the
second run differs from the first in the ONE way you think it does. My second run
differed in two ways (no spy, AND three other potentials compiled first), and the
second one was the operative difference.

**Mechanism.** Non-invasive discriminator: the GL path RAISES for this scalar-only
potential, so raises => GL, returns-a-number => scipy. jax raises, torch returns.
So under torch.compile `_quad_needs_backend` answers False and scipy runs inside
the compiled region. `under_trace`'s docstring predicted exactly this: dynamo
makes `float(x)` symbolic, so a concreteness probe "wrongly takes the concrete
branch and drags the scipy/numpy code into the graph".

**Fix direction unchanged, motivation much stronger:** switch the gate to
`under_trace`. That is the change I wrote during #164 and reverted as speculative
churn after measuring it did not fix the jax NaN. It did not -- and it is exactly
right for torch. Recorded in #168 with the pre-work: check every other
has_concrete_truth_value gate for the same exposure, and re-run the torch --jit
sweep afterwards, because some of that 702-test green may be resting on the same
scipy fallback.

Not phone-pinged: unreleased branch, nothing blocked, and I can document it fully.
But it belongs near the top of Jo's queue.

### Blast radius of the torch.compile concreteness-probe problem: exactly TWO sites

Audited every `has_concrete_truth_value` gate and every raw float()-probe in
galpy:

| site | jax traced | torch traced | severity |
|---|---|---|---|
| `Potential._quad_needs_backend` (:134) | GL path (raises for scalar-only pots) | **scipy path -> 2.235e-15 instead of 0.0662** | silently wrong number |
| `RazorThinExponentialDisk._R2deriv` (:255) | **NaN** (Jo's chosen contract) | **raises PotentialError** | contract violation |
| `AnyAxisymmetricRazorThinDisk` float()-probe (:192) | -- | -- | already guarded by `under_trace`, NOT exposed |

Two exposed sites, one root cause, one fix: use `under_trace` (which asks
`torch.compiler.is_compiling()`) instead of a concreteness probe that dynamo
turns symbolic. The third site is the counter-example that proves the pattern --
someone already hit this and guarded it correctly, which is also where
`under_trace`'s warning docstring came from.

The second site is milder than the first: raising is honest, whereas returning
2.235e-15 is not. But it is still a divergence from a contract Jo chose
deliberately (NaN off-plane under a trace), and torch quietly does not honour it.

Recorded in #168. Both are pre-existing, neither is from my PRs, and the fix is
the single `under_trace` swap I reverted during #164.

## Wake 29 -- #1217 MERGED, #1218 opened (the torch.compile silent-wrong fix)

`1b6f74af` -- #1217 merged on a full green board (154 pass, 0 fail, CLEAN),
closing the whole #164 arc. Audit passed: every new `if` structural (`z is None`
identity, `xp is numpy` namespace, one short-circuit-guarded `bool()`),
duplication net REMOVED, helpers in shared homes.

**#1218** now open with the #168 fix, and it is the most consequential change of
the vacation because it removes a way to get a wrong number:

| site | before (torch.compile) | after |
|---|---|---|
| surfdens | **2.235e-15**, silently wrong, order-dependent | raises TypeError, same as jax |
| R2deriv off-plane | raises, ignoring the traced contract | NaN, same as jax |

The audit found exactly two exposed sites and no more. The third candidate --
AnyAxisymmetricRazorThinDisk's own float() probe -- was already guarded with
`under_trace`, and that guard is where the warning docstring came from. So the
codebase already knew; two sites predated the knowledge.

**Verification, including the part that had to come out red:**

| check | result |
|---|---|
| torch --jit test_potential (5 splits) | 1326 tests, 0 failures |
| jax --jit jeans + anyaxisymdisk + convenience | 141 tests, 0 failures |
| numpy -k "RazorThin or AnyAxisym" | 15 passed |

The two poisson_surfdens tests now FAIL under torch --jit, exactly as they
already do under jax --jit, and are ledgered as the first two `torch-jit`
entries (51 jax / 5 jax-jit / 99 torch / **2 torch-jit**). That is the point, not
a regression: a loud shared failure beats an order-dependent silent wrong number.

numpy cannot reach the changed line -- the gate returns False for numpy first.

Process note: a pre-commit hook rejected the commit for the THIRD time this
vacation (isort this time, ruff-format twice before). The pattern is now
unmistakable -- any hook that *modifies* files fails the commit, and `git log`
afterwards is the only reliable confirmation. I have caught it every time by
checking, which is the habit worth keeping rather than the mistake worth fixing.

### Wake 29 (cont.) -- the poisson_surfdens ledger entries: I built the wrong fix, measured it, and threw it away

The four remaining `poisson_surfdens` ledger entries (2 jax-jit + 2 torch-jit)
were labelled "scalar-only decorator; needs an array-safety migration". Under a
trace the Poisson route hands `Rforce` **50 GL nodes at once** and
`check_potential_inputs_not_arrays` rejects them.

**Attempt 1 (built, measured, discarded).** galpy already has the machinery for
this: `fixed_quad(vectorized=False)` calls a scalar-only integrand node-by-node,
with `_galpy_scalar_only` as the marker and `_force_accepts_arrays` as the
dispatch that `Potential.mass` already uses. The Poisson route simply never
adopted it. I threaded `vectorized=` through the three rules surfdens uses
(`symmetric_quad` / `transformed_quad` / `fixed_quad_semiinfinite`), extracting
the node-evaluation into one shared `_eval_on_nodes` (which also removed an
inline duplicate already sitting in `fixed_quad`).

It works, and it is **unusably slow**: the two tests did not finish in 25
minutes. Each of the 50 unrolled scalar force calls itself runs a 3-panel
order-100 GL quadrature, so the jaxpr contains 50 copies of a large subgraph.
Shipping that would trade a clean xfail for a CI timeout. Diff parked at
`scratchpad/vectorized-plumbing-prototype.diff`; worktree reset.

**What the measurement bought.** Two experiments, both cheap, that reframed the
problem:

1. Is the decorator merely over-conservative? **No.** With numpy arrays both
   potentials genuinely fail -- `AnyAxisym` does a `bool()` on an array
   (`if R == 0 and z == 0`), `DoubleExpDisk` broadcasts against a fixed
   10000-node grid.
2. But those are the *numpy* paths. With **backend** arrays the failures are
   different and much smaller:
   - `AnyAxisym`: `Incompatible shapes: [(3,100), (3,)]`
   - `DoubleExpDisk`: `(10000,) vs (3,)`

   Both are **pure missing-node-axis broadcasting**, not value-decided branches.
   The GL nodes already carry the right shape; a closed-over `R`/`z2` just lacks
   its trailing axis.

**The real fix, for the next PR.** Give the closed-over coordinates a node axis
in the six `_gl` integrands. galpy already has the exact helper -- `_node_axis`,
a no-op on 0-d input -- so the scalar path stays **byte-identical by
construction** while arrays start broadcasting. That yields a *vectorised*
traced surfdens rather than a 50x unroll, i.e. the fast fix and the correct one
are the same fix.

It needs one genuine design decision, which is why I am not rushing it into a
tail-of-wake PR: the decorator must let arrays through, and that is public
behaviour. Intended shape -- allow arrays only when the potential opts in AND
the namespace is not numpy, so the documented numpy contract
("do not accept array inputs") is preserved exactly and only the
already-vectorised backend path opens up. `_node_axis` should also move to
`galpy.backend.quadrature` once it has two consumers in different subpackages.

Worth naming the misread: I took "needs an array-safety migration" from my own
earlier ledger note at face value and started building around it, when ten
minutes of probing showed the migration is far smaller than the note implied --
and that the cheap-looking alternative was the expensive one.

**De-risked before committing to it.** I patched `_Rforce_gl` alone (node-axis on
the closed-over `R`/`z2`) and checked the two properties the whole PR rests on:

```
scalar GL : -1.5341582621184369  -1.34158831864304  -1.167249999317013
array  GL : -1.5341582621184369  -1.34158831864304  -1.167249999317013   shape (3,)
BIT-IDENTICAL elementwise: True
```

So the array path broadcasts *and* reproduces the scalar path bit-for-bit -- not
"to 1e-15", bitwise. `_node_axis` being a no-op on 0-d input means the scalar
graph is unchanged, which is exactly why byte-identity is structural here rather
than something to re-argue per method. Patch reverted; the PR gets built properly
(all six `_gl` methods + decorator opt-in + tests) rather than half-landed now.

**And the second potential is the same fix.** I expected `DoubleExponentialDisk`
to be the hard one -- a fixed 10000-node Ogata table sounded like it needed
restructuring. It does not. Its `_Rforce` is already fully backend-native; `fun`
just closes over `R`/`z` without a node axis, and `_de_quadsum` already accepts
`axis=` (call sites currently omit it, i.e. `axis=None`, which for the 1-D
scalar-input table is identical to `axis=-1`). So its fix is the same two moves:
node-axis the closed-over coords, pass `axis=-1`. Byte-identical for scalars for
the same structural reason.

So #169 is one pattern applied twice, not two migrations -- materially smaller
and safer than the ledger note implied. Both halves can go in one PR after all.

## Wake 30 -- #169 built, and half of it deliberately thrown away

Committed `460f1a0b` on `backend-jit/array-safe-quad-potentials` (held until
#1218 merges; one PR at a time). The plan from wake 29 was "one pattern applied
twice". It turned out to be one pattern applied **once**, and the second half
had to be dropped for a reason worth recording.

**What landed.** `DoubleExponentialDisk`'s five force/deriv methods now
broadcast: `node_axis` on the closed-over `R`/`z`, `axis=-1` on the reduction.
Both are no-ops for scalars, so the scalar arithmetic is untouched -- verified
**bitwise on numpy AND jax**, all five methods, array result identical to
looping the same coordinates as scalars. `check_potential_inputs_not_arrays`
grows an opt-in (`_backend_accepts_arrays`) gated on every array argument being
a BACKEND array, so the documented numpy contract is unchanged. `node_axis`
moved to `galpy.backend.quadrature`, where the convention lives and where it now
has two consumers. **The traced Poisson surfdens for DoubleExpDisk passes: 13s,
vectorised.** 2 ledger entries dropped.

**What I pulled back out, and why.** `AnyAxisymmetricRazorThinDisk` has the
identical symptom, and the identical fix made its six `_gl` methods broadcast
bit-identically too (verified, z=0 edge included). I reverted all of it.

Running the actual test after the fix, the traced Poisson surfdens came back
`-8.5e-06` against numpy's `0.6065`. Not a quadrature shortfall -- raising the
order drove it to `1.8e-14`, i.e. it converges cleanly to **zero**. Splitting
the two terms found the boundary term exact to all digits and the integral at
`-6.644` instead of `+0.978`. Probing the integrand toward the plane:

```
   |z|      numpy(scipy)      traced GL        relerr
  1e-02   4.1259264991e+00  4.1259264991e+00   1.6e-12
  1e-03   4.1605916883e+00  4.4090723647e+00   6.0e-02
  1e-04   4.1640252925e+00 -6.8505866336e+01   1.8e+01
  1e-05   4.1643683070e+00 -9.3066065216e+04   2.2e+04
```

That is the `a=R` principal value the class comment already documents, and
`symmetric_quad(interior_point=0.0)` clusters its nodes at `z=0` precisely
there. So for this potential the array gate would have converted a loud
`TypeError` into a silently wrong `~0` -- exactly the failure mode I argued
against in #1218 two hours earlier. The gate stays shut for it; its ledger
entries stay, with the reason corrected from "scalar-only decorator" to the
sheet limit, and a test now pins the refusal so nobody opens it casually.

Net: 4 ledger entries were going to drop; 2 dropped, 2 kept **on purpose**. The
useful lesson is that "same symptom, same fix" was true about the code and false
about whether the fix should ship.

Also worth noting: this is the fourth pre-commit rejection of the vacation
(ruff-format again). Checking `git log` after every commit has caught all four.

**#169 verification (both tracers, not just the one I fixed on).** I dropped a
`torch-jit` ledger entry on the strength of a *jax* result, which is exactly the
kind of inference that has bitten this project before, so I ran torch directly:

```
torch --jit: 8 tests, 0 fail, 107s
  pass   test_poisson_surfdens_potential[DoubleExponentialDiskPotential]
  xfail  test_poisson_surfdens_potential[AnyAxisymmetricRazorThinDiskPotential]
  pass   (all 6 new tests in test_backend_array_inputs.py)
```

So the drop is real on torch and the AnyAxisym refusal registers as xfail rather
than a red. Plus **numpy `test_potential` in full: 1326 tests, 0 failures** --
the decorator rewrite is global, so that was the check that mattered most -- and
jax --jit group 1/4: 332 tests, 0 failures (2-4 still running).

Deliberately NOT in this PR, noted for the reviewer: `Potential.mass`'s numerical
path still asks `_force_accepts_arrays`, which sees only the scalar-only marker,
so DoubleExpDisk's mass integral keeps unrolling node-by-node even though it can
now vectorise. That is a speed follow-up, not a correctness one (the two spellings
reduce bitwise identically), and folding it in would widen a PR that is currently
clean and verified.

## Wake 31 -- #167 CLOSED: torch --jit is fully measured, and it is clean

The last unmeasured traced surface is done. `test_orbit.py` under
`--backend torch --jit`, all 8 pytest-split groups:

```
581 tests, 0 fail, 0 err, 23 skip
```

Combined with the 11 suites measured earlier, torch tracing now has **zero known
failures anywhere**, from a starting position two days ago of "0 torch-jit ledger
entries, because nobody had ever run it". The two entries #1218 added are the
only ones that have ever existed, and #169 removes one of them.

**#169 verification is complete and green on every axis:**

| run | result |
|---|---|
| numpy `test_potential` (full) | **1326 tests, 0 fail** |
| jax --jit `test_potential` (4 groups) | **1326 tests, 0 fail** |
| torch --jit (poisson_surfdens + new tests) | **8 tests, 0 fail** (DoubleExpDisk pass, AnyAxisym xfail) |
| `test_backend_array_inputs.py` | 6 pass |

The numpy run is the one that mattered -- `check_potential_inputs_not_arrays`
wraps a large fraction of the potential library, so a gate rewrite there is the
kind of change that breaks things far from where it was edited. It did not.

**#1218 status:** 148/154 green, 3 pending (two py3.14 `test_backend` shards at
90+ minutes, plus all-clear). codecov reads 86.17% against a ~99.6% baseline --
a 13-point drop that a 3-file change cannot produce, so this is the
partial-upload signature with the heaviest shards still running, not a real gap.
Per my own discriminator: wait, do not chase. Not touching it.

### #1218 MERGED (`0036e42e`), #1219 opened

154/154 green. The two codecov reds **flipped to pass on their own** as the last
uploads landed -- exactly the partial-upload signature I called earlier (86.17%
against a ~99.6% baseline is not a drop a 3-file change can produce). Waiting
instead of chasing was the right call; no action was ever needed.

Audit before merge, on the three axes:
* **if -> xp.where array class**: the new `if under_trace(z)` and `if xp is numpy`
  are STRUCTURAL -- tracing context and namespace identity -- so they yield one
  Python bool regardless of input shape and cannot mis-handle arrays. The genuinely
  value-dependent decision uses `xp.where(inplane, val, nan)`, the array-safe form.
  `if not allin` is reachable only on the concrete path, and `allin` is still used
  there (checked -- the traced early return did not orphan it).
* **duplication**: reuses the shared `under_trace`; removes the now-dead
  `has_concrete_truth_value` import from both files. No new helper.
* **placement**: nothing general buried in a potential module.

`gh pr merge` reported "failed to run git" -- the API merge had already succeeded
and only gh's LOCAL branch cleanup failed, because feat/backends-develop is checked
out in another worktree. Checked the PR state before retrying rather than
re-issuing the merge; it was already MERGED. Worth remembering: a gh error after a
mutating call is not evidence the mutation failed.

**#1219** is now open with the DoubleExpDisk work, rebased clean onto the squashed
develop (1 unique commit, 5 files, no conflict markers).

## Wake 32 -- attacked the AnyAxisym principal value, measured it, discarded it

Three of the five remaining jit-ledger entries share one root cause (the `a=R`
principal value), so it looked like the highest-leverage thing left. galpy already
has the tool: `transformed_quad(interior_point=R)` clusters nodes at the split,
where `_bk_split_quad` instead uses two PLAIN GL panels. A six-line swap.

**It works spectacularly off-plane.** The Poisson integrand at R=0.5:

```
   |z|      before        after
  1e-02   1.6e-12       4.4e-14
  1e-03   6.0e-02       1.9e-14      <- 12 orders
  1e-04   1.8e+01       3.0e-06      <-  7 orders
  1e-05   2.2e+04       5.5e-03
```

and `test_eccentricity`'s circular orbit improves **5.4e-06 -> 7.7e-08**, a 70x
gain.

**And it regresses a passing test.** `test_forceAsDeriv_potential[AnyAxisym]`
under --jit goes red at **(R,Z) = (0.500, 0.000)**, 8.9e-2 relative. The reason is
worth keeping: at z != 0 the `a=R` pole is SOFTENED by z, so the feature has width
~z and clustering finds it. At z = 0 exactly the pole is unsoftened, and plain GL's
node placement -- which never lands near a=R -- was accidentally acting as a crude
principal-value regularisation. Clustering deliberately puts nodes ON the pole. The
two regimes pull in opposite directions.

**Ledger arithmetic, so the trade is explicit:** the change clears **zero** entries.
`test_2ndDeriv` still fails at z=0 (Hadamard finite part, a third distinct cause),
`test_eccentricity` still misses its 1e-8 bar at 7.7e-8, and `poisson_surfdens`
stays blocked for a reason I only found by building the whole thing: the OUTER
surfdens quadrature uses `interior_point=0.0`, so it samples the integrand down to
`z*xi^3 ~ 3e-12`, far below where even the improved scheme holds. The surfdens
integrand is smooth and bounded near z=0 (it converges to 4.1644), so that outer
clustering is not even buying anything there -- but changing it would affect every
other potential.

So: large accuracy gain off-plane, a real regression on-plane, no ledger movement,
and the honest fix (choose the rule from z, guarding the dead branch so the z=0
side cannot NaN-poison gradients) is a design, not a patch. **Reverted.** The
entries keep the accurate explanation they got in #1219.

That is twice this wake that measurement killed an attractive fix -- the
`vectorized=` unroll and now this. Both looked right on paper and neither
survived contact with the actual numbers. Cheap to find out, expensive to have
shipped.

## Wake 33 -- swept the never-measured suites; found a whole file invisible to jax

With #1219 queued behind the runner backlog, I ran the 13 non-backend test files
that have never been executed traced. Clean so far: `test_streamgapdf` (30),
`test_streamgapdf_impulse` (14), `test_kuzminkutuzov`, all 0 failures.

**The interesting one is `test_streamTrack.py`, which reported 45 tests and 45
SKIPS.** It is not ledgered -- it is a FILE-LEVEL entry in
`backend_slow_skip.txt` (`jax tests/test_streamTrack.py`, added because it pushed
the shard over its timeout). Since `-jit` inherits plain `jax` entries, an entire
suite has been invisible under both jax and jax --jit for its whole life. That is
a bigger blind spot than any single ledger entry, and exactly the class of thing
the ledger's own size metric cannot show.

So I bypassed the slow-skip locally (temp edit, reverted) and measured it, split
4 ways so it could not blow a timeout:

```
test_streamTrack.py under jax --jit: 45 tests, 5 failures
  5x NotImplementedError
     test_streamTrack_plot_smoke
     test_streamTrack_out_of_range_returns_nan
     test_streamTrack_cov_per_call_unit_overrides
     test_streamTrack_cov_basis
     test_streamTrack_plot_spread_non_cartesian
```

All five are ONE deliberate, documented gap: `streamTrack.cov()` on a backend
track supports only `basis='galcenrect'`, and sky-frame bases are the deferred
PR #861 follow-up (see memory `streamtrack_todo`). **40 of 45 pass traced.**

No action taken, deliberately. Ledgering the five would add inert entries (they
are skipped, so they would never fire), and narrowing the file-level slow-skip so
the 40 passing tests actually run is a real coverage gain but re-risks the shard
timeout that motivated the skip -- a CI-time judgement I would rather not make
unprompted with a PR in flight and no ability to edit workflows. What this buys
now is knowing the answer: if the skip is ever narrowed, expect exactly those
five, for exactly that reason.

### #1219 red was MY over-claim, not the code (`4e4e1fb7`)

The py3.14 backend shard failed on my own new test:

```
R2deriv: vectorised evaluation is not bit-identical to the scalar one
  array : [-0.12049472, -0.07845821, -0.04761014, -0.02634406]
  scalar: [-0.12049472, -0.07845821, -0.04761014, -0.02634406]
```

Identical at print precision, differing in the last bits -- on py3.14, where all
five methods are EXACTLY equal locally on py3.13.

The cause is real and I had missed it: vectorising changes the **shape of the
reduction**. A scalar call sums a `(10000,)` Ogata table; an array call sums
`(len, 10000)` along the trailing axis, and a backend may associate a batched
reduction differently. `R2deriv` is the one method summing TWO Ogata terms, so it
has the most reassociation available -- which is exactly why it, and only it, is
the one that broke.

So "bitwise, not to a tolerance" was **too strong a claim**, and I had put it in
the commit message, the PR body and this log. The operations are identical; only
their grouping can differ. Corrected the assertion to a ~sqrt(10000)*eps
reassociation bound, which is still tight enough that a dropped node, a wrong
axis or a lost factor fails it by orders of magnitude. Locally the observed
difference is exactly 0.0 for all five, so the bound is margin for the platform,
not slack in the claim.

Worth being precise about what survived: the FIX is unaffected -- the
vectorisation is still correct, and numpy is still untouched. What was wrong was
the strength of the property I asserted about it. My py3.13 box cannot see
py3.14 reassociation, which is the same blind spot as the `regen-py-version`
memory, now with a second instance.

Cancelled the five stale `30afbbe` runs on push (prior-SHA, so safe -- same-SHA
push runs are branch-protection-required and must never be cancelled).

## Wake 34 -- a real ledger flip, from reading what the test actually asserts

**`test_ExpDisk_special` is un-ledgered (`08c3acf1`, next PR after #1219).**

Its last block asserts that `RazorThinExponentialDisk.R2deriv` RAISES for
z=/=0 -- correct, since off-plane is outside its domain. But #1218 (Jo's explicit
"let's do NaN") made the traced path return NaN there instead, because under a
trace z is a tracer, the domain cannot be decided, and raising would mean
refusing to trace at all. The test could not express that, so it was ledgered.

Now it asserts the actual two-regime contract: a non-raise is acceptable ONLY on
a backend, and only as NaN. The numpy refusal is still required, so this is not a
weakening -- it states the case the old assertion had no way to say.

Verified in all four modes, and -- per my own stash-and-rerun rule -- confirmed
the flip is real rather than a test that was passing anyway:

```
  numpy     -> PASS      WITHOUT the fix, jax --jit -> FAIL
  jax       -> PASS      (tests/test_potential.py:4307 AssertionError)
  jaxjit    -> PASS
  torchjit  -> PASS
```

jax-jit ledger: 5 -> 4 with #1219, -> 3 with this. What remains is
`test_2ndDeriv[AnyAxisym]` and `poisson_surfdens[AnyAxisym]` (the sheet limit
characterised in wake 32) and `test_eccentricity` (same root cause, 7.7e-8
against a 1e-8 bar). All three are one physics problem, now documented in detail.

**Incidental find, worth a note:** the same test file has FIVE commented-out
array-input blocks for DoubleExpDisk's forces and derivatives -- Rforce, zforce,
R2deriv, z2deriv, Rzderiv -- disabled long ago precisely because the potential
rejected arrays. #1219 makes exactly those work, but only for BACKEND arrays;
they use `numpy.linspace`, so un-commenting them as-is would fail on numpy, where
the scalars-only contract deliberately still stands. Left alone; noted because it
shows the gap #1219 closes was felt in the test suite years before the backend
work existed.

**And a measured cost:** `test_noninertial.py` under jax --jit hit a **45-minute
timeout** (exit 124) without finishing. That matches the known
`jax-noninertial-timeout` flakes; now quantified rather than suspected.

### The sweep paid off: two NEW jit-only gaps in `test_interp_potential.py`

`test_interp_potential.py` under jax --jit: 40 tests, **2 failures** --
`test_interpolation_potential_use_c` and `..._force_use_c`. The ledger already
carries two entries for this file, but for DIFFERENT tests
(`..._force_c`, `..._secondderivs`), so these are genuinely new.

Both PASS under eager jax and fail only traced, which is the discriminating check
I ran before drawing any conclusion. Both compare an interpRZPotential grid built
with `use_c=True` against the same grid built in Python, to 1e-14:

```
            pot grid    rforce grid   (bar 1e-14)
  eager     2.665e-15     1.776e-15
  jax --jit 5.596e-14     6.621e-13
```

So the traced Python-side grid construction loses one to two digits. That makes
this **numerical over-tolerance, not a wrong result** -- the traced grid is still
correct to ~1e-13 -- and the distinction matters for how it gets fixed.

Ledgered with the measured magnitudes (`ba4fa851`), not fixed: the honest fix is
a backend-aware tolerance, and per standing policy loosening a tolerance needs
Jo's sign-off. Verified they register as `xx`, not as reds. Two entries added
against one removed this wake, which is the right trade -- the ledger got MORE
accurate, and a silent traced gap became a documented one.

Running never-measured suites has now produced findings four times
(#164 #165 #166 #167, and this). It is the highest-yield thing I do while CI is
queued.

### Never-measured-suite sweep COMPLETE

13 files dispatched, 12 with results (`test_noninertial` timed out at 45 min and
produced none). Under `--backend jax --jit`:

```
test_MultipoleExpansionPotential   107   0 fail
test_streamTrack                    45   0 fail  (45 SKIP -- file-level slow-skip)
test_interp_potential               40   2 fail  <-- the only red
test_streamgapdf                    30   0 fail
test_SpiralArmsPotential            19   0 fail
test_streamgapdf_impulse            14   0 fail
test_symplectic_dxdv                12   0 fail
test_kuzminkutuzov / test_pv2qdf    10   0 fail  (each)
test_linear_dxdv                     9   0 fail
test_FDMdynamfric                    9   0 fail
test_util                            4   0 fail
------------------------------------------------
TOTAL                              309   2 fail
```

So the traced surface is in far better shape than the ledger alone suggested:
307 of 309 previously-unmeasured tests pass, and the 2 that do not are the
characterised interpRZPotential over-tolerance now recorded with numbers.

Three residual blind spots, all now named rather than unknown:
1. `test_streamTrack` -- 45 tests hidden by a FILE-level slow-skip; measured
   separately at 40/45 passing, 5 on the deferred sky-frame `cov()` basis.
2. `test_noninertial` -- does not complete in 45 min traced, so it is
   unmeasurable in any reasonable CI budget without work on its speed first.
3. The AnyAxisym sheet limit (wake 32), 3 entries, one physics problem.

### Root-caused the interpRZPotential jit gap: it is XLA, not galpy

Rather than leave "traced construction loses one to two digits" as a black box,
I localised it. Evaluating each MWPotential component on the same grid, eager vs
traced:

```
  MiyamotoNagaiPotential   3.465e-16
  HernquistPotential       5.790e-16
  NFWPotential             1.066e-14   <-- 30x the others
```

Then the decisive pair at the worst grid point (R=0.0896, z=0.02):

```
  eager numpy     -4.08407232333320369e+00
  jax EAGER       -4.08407232333320369e+00   rel = 0.000e+00   (bit-identical)
  jax jit         -4.08407232333324721e+00   rel = 1.066e-14   (same every run)
```

jax **eager** reproduces numpy bit for bit; only the **jitted** value moves, and
it moves deterministically. So XLA compiles NFW's `log1p(r/a)/(r/a)` differently
from how it evaluates it eagerly -- and that is precisely the cancellation-prone
shape for a reassociation to show up in. It is a jit-mode numerical
characteristic, **not a galpy defect**: there is nothing in the potential to fix.

Two things this changes. The ledger entry now names the mechanism instead of the
symptom, which is the difference between an entry someone can act on and one
that just says "traced is worse". And it is the SECOND time this vacation that
the answer was floating-point reassociation under compilation -- the first being
my own over-strict bitwise test on py3.14 a few hours ago. Worth carrying
forward: under `--jit`, bit-identity to the eager path is not a property to
assume anywhere.

An earlier probe of mine at nearby points showed only ~1e-16 and nearly sent me
away; it entered and left the jit context per point instead of once for the
grid. The reproduction only became stable when I matched the original
measurement exactly. Same lesson as the torch.compile hunt: a non-reproduction
is only evidence if the retry differs in the one way you think it does.

## Wake 35 -- #1219 MERGED (`c1abf586`), #1220 opened

154/154 green, and codecov/project flipped to pass on its own for the SECOND
time as uploads completed. That pattern is now well-established enough to treat
as the default reading: a large project drop while shards are still running is
the partial-upload signature, and waiting costs nothing.

Audit before merge. The three new conditionals, and what each decides on:

| line | decides on | verdict |
|---|---|---|
| `v[..., None] if getattr(v, "ndim", 0)` | array RANK, static under trace | structural |
| `hasattr(x,"shape") and x.shape != () and len(x) > 1` | SHAPE, static under trace | structural |
| `arrays and not (flag and all(is_backend_array(x)))` | list truthiness, class attr, TYPE | structural |

None branches on element values, so none needs `xp.where`. Duplication: net
NEGATIVE -- `node_axis` moved rather than copied, and `git grep "^def node_axis"`
returns exactly one hit, in `galpy.backend.quadrature` where the convention is
defined and where it now has two consumers in different subpackages.

**Rebase trap worth recording.** After the squash-merge, rebasing the stacked
branch tried to REPLAY #1219's pre-squash commits (`30afbbed`, `4e4e1fb7`) --
git cannot tell they are already in develop, because squashing changed their
content -- and conflicted against the file they had themselves added. The fix is
not to resolve those conflicts: it is to abort, reset to the new develop, and
cherry-pick only MY commits. Verified after: exactly 2 commits, exactly 2 files,
no conflict markers, no #1219 content leaked in.

**#1220** carries the ledger-accuracy work: `test_ExpDisk_special` un-ledgered
(real flip, stash-verified) and the two NFW/XLA interpRZPotential gaps recorded
with their root cause. Re-verified on the new base after the cherry-pick --
ExpDisk passes numpy/jax-jit/torch-jit, and the interp pair registers `xx`.

## Wake 36 -- caught a gap in my OWN process: the README table was 24 PRs stale

Dispatched the torch `--jit` sweep over the same 11 suites I ran under jax last
wake (torch tracing only became real with #1218, so those files have never been
traced on torch). Clean so far: streamgapdf 30, streamgapdf_impulse 14,
kuzminkutuzov 10, all 0 fail; interp_potential still compiling.

While that ran I checked something I had been quietly skipping. `README.md`'s
PR table -- the surface Jo actually reads, per "I won't be reading through a
two week trace, but will ask you to summarize" -- **stopped at #1196**. Every
PR of this vacation (#1197 through #1219, 24 of them) was missing from it. I
have been meticulous about `logs/vacation_2026-07/log.md` and let the summary
surface rot, which is precisely backwards: the log is the trace he said he would
NOT read.

Added five rows covering the wave, at the granularity the rest of the table
uses: the `--jit` harness (#1200/#1201/#1202/#1205), the ledger burndown
(#1206→#1217, jax-jit 100 → 5), the torch.compile silently-wrong surfdens
(#1218, with the 2.235e-15-instead-of-0.0662 number in it), DoubleExpDisk
array-safety (#1219, including WHY AnyAxisym is deliberately excluded), and the
measurement work with its two named residual blind spots.

The memory `keep-galpy-jax-log-current` already said "update plan/track*.md +
README table + logs/ as part of LANDING each PR" -- I was doing one of the
three. Worth stating plainly rather than quietly fixing.

## Wake 37 -- the torch sweep found a REAL pre-existing bug (not a jit gap)

torch `--jit` over the 11 suites: 9 clean, **2 red**.

```
test_MultipoleExpansionPotential   107   0     test_streamgapdf          30   0
test_interp_potential               40   0     test_streamgapdf_impulse  14   0
test_SpiralArmsPotential            19   0     test_kuzminkutuzov        10   0
test_pv2qdf                         10   0     test_util                  4   0
test_linear_dxdv                     9   5  <-- RED
test_symplectic_dxdv                12   2  <-- RED
```

**First, a confirmation worth having.** `test_interp_potential` PASSES on torch
while failing on jax --jit. That is independent evidence for yesterday's root
cause: the gap is XLA compiling NFW's `log1p(r/a)/(r/a)` differently, so it
should NOT appear on torch's compiler -- and it does not.

**Then the real find.** All 5 `test_linear_dxdv` failures are one TypeError:

```
  x = np.float64(0.3), t = np.float64(0.0)
  return -x * (self._K / xp.sqrt(x**2 + self._D2) + 2.0 * self._F)
  TypeError: sqrt(): argument 'input' must be Tensor, not numpy.float64
      galpy/potential/KGPotential.py:79
```

I checked whether it was jit-specific before concluding anything. **It is not:
torch EAGER fails identically, 5/9.** So this is a plain pre-existing torch bug,
unledgered and unseen only because `test_linear_dxdv.py` had never been run under
`--backend torch` at all -- not by the sweeps, not by CI.

Root cause is the known `scalar-under-forced-backend` cluster.
`_evaluatelinearForces` is deliberately "raw, undecorated for internal use", so
the 1D variational RHS hands `_force` a raw numpy scalar; under a forced backend
`get_namespace(x)` resolves to torch, and `xp.sqrt(numpy.float64)` dies. The 3D
path does not have this problem because `check_potential_inputs_not_arrays`
already coerces for `_backend_compatible` potentials -- **the 1D analogue of that
coercion is simply missing**. Fix is to coerce (per
[[coerce-under-forced-not-dataguard]]), not to special-case sqrt.

Queued as the PR after #1220. Not started: #1220 is the active PR and its
Windows orbit job went red.

**#1220 Windows red**: job completed=failure with NO failing step recorded and
logs returning `BlobNotFound` -- the infrastructure signature, not a test
failure. My PR touches only `test_potential.py` and the ledger, so a Windows
ORBIT shard is not plausibly mine. Cannot `rerun --failed` yet: the run still has
5 jobs live and gh refuses while any are running. Waiting for it to settle first.

### The 1D coercion fix is built and green in all five modes (`897199ed`)

Held for #1220. `galpy/potential/linearPotential.py` +27, two test files +22.

Added the coercion at the three shared raw 1D entries
(`_call_nodecorator` / `_force_nodecorator` / `_force2deriv_nodecorator`), gated
on `_backend_compatible` exactly as the 3D side is, leaving `t` alone for the
same reason 3D does (it can be a hashable cache key). One helper, one place, all
1D potentials -- not a per-potential patch.

The KGPotential `sqrt` errors went away immediately; **3 of the 7 failures
survived**, and they were a different thing: three test assertions feed a backend
Orbit accessor straight into `numpy.amax`, which dispatches to
`Tensor.max(axis=, out=)` and torch rejects. Cast with `as_numpy` at those sites
(a no-op on numpy). Worth noting the fix was two distinct causes wearing one
symptom -- had I stopped at "torch 7 -> 3, good enough", the PR would have been
half a fix.

```
  numpy     21/21      torch      21/21      torch --jit  21/21
  jax       21/21      jax --jit  21/21
```

Self-inflicted detour worth recording: my scripted insert of
`from galpy.backend import as_numpy` anchored on `from galpy.orbit import Orbit`,
which in these files is a FUNCTION-LOCAL import -- so it landed indented inside
`_build_M` and produced a syntax error. Caught by the immediate `ast.parse`, not
by the test run. Anchor on module-level text (`import pytest`) when injecting a
module-level import.

**#1220** meanwhile: the Windows red was NOT a test failure. `Test with pytest`
has `conclusion: null` and so does every step after it -- a killed/lost runner.
Re-ran the failed job once the run finished (gh refuses `rerun --failed` while
any job is live, hence the wait). codecov/project cleared on its own again.

## Wake 38 -- #1220 MERGED (`07cb8426`), #1221 opened

154/154. Audit was short because the diff has **zero galpy/ source changes** --
tests + ledger only, so there is no if->xp.where surface at all. The two new
asserts are a namespace identity check (`get_namespace(1.0) is not numpy`) and a
value assertion inside a test, both correct. No helpers, no duplication.

The Windows red resolved as diagnosed: not a test failure. `Test with pytest` had
`conclusion: null`, as did every step after it -- a killed runner. Re-ran the
failed job once the run finished (gh refuses `rerun --failed` while any job in
the run is live, which is why it had to wait), and it came back green.
codecov/project cleared on its own for the third time.

**#1221** is the 1D coercion fix. Worth restating the headline honestly: this is
NOT a jit finding. `test_linear_dxdv.py` and `test_symplectic_dxdv.py` had never
been run under `--backend torch` by ANYTHING, so 7 plain torch failures had been
sitting there unledgered. The `--jit` sweep is just what finally executed them.
That is the third distinct thing this vacation that was invisible purely because
nothing had ever run the relevant configuration -- after the torch-jit blind spot
(#167) and the file-level slow-skip on test_streamTrack.

Ledger arithmetic: **none**. These were never ledgered, because nothing had ever
run them on torch. That is a worse state than a ledger entry, and the reason the
"run the configuration nobody has run" habit keeps paying.

## Wake 39 -- the streamTrack slow-skip is JUSTIFIED, and now with a number

Two measurements settled the open question from wake 33 ("should the file-level
slow-skip be narrowed so 40 passing tests actually run?").

**1. Independent corroboration of the earlier measurement.** The 5 tests I found
failing on jax by bypassing the skip are ALREADY LEDGERED FOR TORCH -- the skip is
`jax`-only, so torch runs the file:

```
torch tests/test_streamTrack.py::test_streamTrack_cov_basis
torch tests/test_streamTrack.py::test_streamTrack_cov_per_call_unit_overrides
torch tests/test_streamTrack.py::test_streamTrack_out_of_range_returns_nan
torch tests/test_streamTrack.py::test_streamTrack_plot_smoke
torch tests/test_streamTrack.py::test_streamTrack_plot_spread_non_cartesian
```

Exactly the five, same names, same cause (the deferred sky-frame `cov()` basis).
So the gap was never undocumented -- it was documented on the backend that runs
the file, and invisible on the one that skips it. My measurement and the existing
ledger agree, which is the best kind of confirmation.

**2. The reason for the skip, quantified.** Eager jax, whole file:

```
45 tests, 5 fail, TOTAL = 779s  (13 minutes)
```

Thirteen minutes for ONE file. The skip's own note said it "pushes
test_streamTrack over the shard timeout", and that is plainly still true.
Narrowing it would buy 40 tests of jax coverage at the price of +13 min on a jax
shard that is already the slowest thing in CI -- and I cannot edit workflows to
rebalance the shards.

**Decision: leave the file-level skip.** Not because it is comfortable, but
because the trade is now measured rather than assumed: 40 tests vs 13 minutes,
with the 5 real gaps already recorded via torch. If Jo wants that coverage, the
cheap version is to move `test_streamTrack.py` into its own shard rather than to
un-skip it in place -- a workflow change, which is his to make.

This closes the second of the three residual blind spots as *understood*, not
fixed. The third (`test_noninertial`, does not finish in 45 min traced) is the
same shape of problem: a speed issue masquerading as a coverage gap.

### #1221 red: SIMBAD outage, and the step-conclusion tell is worth keeping

151/154 with one red, and this time I could NOT dismiss it by shape: my PR touches
`linearPotential.py`, so a Windows ORBIT shard failing is at least conceivable.

The step conclusions distinguished the two cases immediately:

* #1220's Windows red -- `Test with pytest` had `conclusion: null`, and so did
  every step after it. Never ran to completion. Lost runner.
* #1221's Windows red -- `Test with pytest` has `conclusion: failure`. It RAN and
  a test failed. Different animal, so I pulled the log instead of assuming.

```
tests/test_orbits.py::test_from_name_values FAILED
pyvo.dal.exceptions.DALServiceError: Unable to access the capabilities endpoint at:
  https://simbad.cds.unistra.fr/simbad/sim-tap/capabilities: Connection failed
```

The known SIMBAD/VizieR network flake ([[no-rerun-vizier-notebook-flakes]]),
nothing to do with 1D coercion. Re-ran the failed job (the run had settled, so
`rerun --failed` was accepted this time without waiting).

Keeping the tell explicitly: **`conclusion: null` on the pytest step means the
job died; `conclusion: failure` means a test failed.** That one field is the
difference between "re-run it" and "read the log", and I have now needed it twice
in two PRs with opposite answers.

## Wake 40 -- #1221 MERGED (`5869d62f`); and I tripped my own import-shadow memory

154/154 after the SIMBAD re-run. Audit: one new conditional in galpy/
(`if not getattr(pot, "_backend_compatible", False)`) which decides on a class
attribute -- structural, no value branch, no `xp.where` needed. `_coerce_x`: one
definition, three uses, and it calls the SHARED `coerce_coords` rather than
reinventing it. Placement is module-local beside the 1D raw entries it guards,
symmetric with the 3D equivalent living inline in `Potential.py`'s decorator.

**Then I wasted a probe on a mistake my own memory documents.** I launched
`test_noninertial` under torch --jit as

```
python -m pytest /var/tmp/nogl/tests/test_noninertial.py ...
```

with cwd left at the MAIN repo. It "finished" in 12s with 48 errors, all of them:

```
AttributeError: module 'galpy.backend' has no attribute 'jit'
    tests/conftest.py:456
```

That is not a finding about noninertial. `galpy` resolved to the main repo
(`feat/backends @ dbb09ffc`), which predates `galpy.backend.jit` (#1202) -- the
editable-install shadow that [[galpy-editable-install-import-shadow]] exists to
prevent, and whose stated remedy is "always PYTHONPATH=<worktree> + assert
galpy.__file__". I set PYTHONPATH and skipped the assert, and cwd beat it.

Relaunched from inside the worktree, with the provenance asserted FIRST:

```
galpy: /var/tmp/nogl/galpy/__init__.py
has jit: True
```

The tell was there in plain sight -- 12 seconds for a file that takes >45 min
under jax, and errors in the FIXTURE rather than failures in the tests. A
suspiciously fast green (or red) is itself evidence that the thing you meant to
run did not run. Worth adding to the memory: assert `galpy.__file__` **and** run
from inside the worktree; PYTHONPATH alone does not win against cwd.

## Wake 41 -- third blind spot ANSWERED, and the ledger metric is 44% inflated

**`test_noninertial` under torch --jit: 53 tests, 0 fail, 0 err, 670s.** It
completes cleanly in 11 minutes on torch, where jax --jit did not finish in 45.
So the last residual blind spot is **jax-specific** -- a jax eager-XLA compile
cost, not a galpy limitation -- and torch already has full traced coverage of that
file. All three blind spots from wake 33 are now answered:

1. `test_streamTrack` -- file-level jax slow-skip, 779s eager, 5 gaps already
   ledgered on torch. Justified; leave it (wake 39).
2. `test_noninertial` -- jax-only compile-time problem; torch traces it in 11 min.
3. AnyAxisym sheet limit -- 3 entries, one physics problem, characterised (wake 32).

**Then I clustered the eager ledger to pick the next burndown slice, and found the
metric itself is wrong.** `test_actionAngle.py` is 38 torch + 29 jax entries, 45%
of the total -- but 24 of the 38 torch and 24 of the 29 jax are ONE class:

```
  total entries                156
  actionAngleVerticalInverse    48   (31% of the ledger)
  reducible by fixes           108
```

`actionAngleVerticalInverse._reject_backend` raises NotImplementedError as soon as
a backend is active, so those 48 cannot be moved by incremental fixes at all.
Since [[ledger-milestone-burndown]] treats ledger size AS the progress metric,
quoting 156 overstates the remaining work by ~44%.

**#1222** states the split in the file itself. Comment-only; verified the ledger
still parses (156 counted), its self-test passes 5/5, and a ledgered test still
registers `x` rather than a red.

One deliberate restraint: my memory says Jo decided VerticalInverse stays
`_reject_backend`-guarded ("Inverse migration = Harmonic+Isochrone only"), but
the class docstring says "NOT **yet** backend-migrated (under active
development)" -- and task #94 is marked completed, which contradicts both. Three
sources, three stories. So the note asserts only the mechanical fact (incremental
fixes cannot reduce those entries) and explicitly leaves the scope question to
Jo. Verified the guard still exists in current code before writing any of it,
rather than trusting the memory.

## Wake 42 -- first eager-ledger burndown: 6 entries, and two over-claims I caught myself

#1222 (ledger reading note) still in CI. Meanwhile, started the eager burndown by
clustering the 108 reducible entries: after VerticalInverse, `test_quantity.py` is
the largest cluster at 20, and inside it the tightest is `*_turnphysicaloff` --
3 tests x 2 backends = 6 entries, one shared mechanism.

Cause: those tests assert `isinstance(result, float)`, a numpy-specific PROXY for
"not a Quantity". Under a forced backend galpy returns a 0-d BACKEND array by
design -- exactly what makes the call traceable -- so a CORRECT result fails.
Test-side, same class as #1220's ExpDisk work. Committed `1489fb1d`, held for
#1222. Ledger **156 -> 150**.

**Two mistakes of mine, both caught before they shipped, both worth recording.**

1. *A regex that produced valid syntax and broken semantics.* My first pass used
   `re.subn(r"isinstance\((.*?), float\)", ..., flags=re.S)` and it ate across a
   boundary, leaving `assert isinstance(potential.evaluatelinearPotentials(pot, 1.1)), (`
   -- a ONE-ARGUMENT isinstance. `ast.parse` passed happily; it would have raised
   TypeError at runtime. **`ast.parse` is not validation for this class of edit.**
   Redid it with 11 exact string replacements plus an explicit residual count.

2. *An over-claim in my own docstring.* I wrote that the numpy contract "is not
   relaxed", then probed it and saw a 0-d numpy array apparently accepted -- so I
   added a guard against `numpy.ndarray`. Probing further, BOTH readings were
   wrong: `is_backend_array(numpy.array(1.5))` is **False**, so numpy 0-d arrays
   were never accepted, and the True I saw came from `numpy.float64` being a
   `float` SUBCLASS -- which the original `isinstance(v, float)` accepted too. My
   guard was dead code resting on a false premise. Removed it, and replaced the
   prose claim with a case-by-case table proving the only changed case is
   `0-d jax -> True`.

That second one is the same failure mode as the py3.14 "bitwise" over-claim
earlier this vacation: I asserted a property about my own change instead of
measuring it. The fix both times was a table of cases, not a better sentence.

### Next cluster probed and deliberately NOT started

Looked at the sphericaldf cluster (8 entries) expecting another type-proxy fix.
It is not. From `test_sphericaldf_method_value` under jax:

```
  dfh.vmomentdensity(1.1, 0, 2)      = 1.64026634e+10   (physical output)
  dfh_nou.vmomentdensity(1.1, 0, 2)  = 0.00192737       (natural units)
```

and the physical number is **correct** -- `0.00192737 * (9.0027e10 / 8**3) * 220**2`
= 1.64e10 exactly, so there is no missing conversion factor. The test also flips
`galpy.util._optional_deps._APY_UNITS = False` mid-test as a documented hack, and
the failing comparison has backend Arrays on both sides. So the suspect is
physical_conversion x the _APY_UNITS hack x backend arrays -- a real
investigation, not a mechanical cast.

Recorded as task #171 with the numbers and an explicit warning to myself: **do not
assume it is another as_numpy fix; verify first.** Two hours ago I assumed
"sqrt(numpy.float64)" was the whole 1D story and 3 of 7 failures turned out to be
a second cause. Starting a units investigation at the tail of a long wake, with a
finished 6-entry commit already in hand and #1222 still mid-CI, is how that
mistake gets repeated rather than avoided.

**#1222**: 150/154 with the two py3.14 backend shards (90+ min each) plus
all-clear outstanding, and codecov/project red. Comment-only PR, so coverage
cannot actually have moved -- consistent with the partial-upload pattern that has
self-cleared every time this vacation. Watching, not touching.

## Wake 43 -- #1222 MERGED (`08e19572`), #1223 opened (first eager burndown lands)

154/154, codecov self-cleared for the FOURTH time. Audit was mechanical and I made
it provable rather than eyeballed: 16 added lines, `grep -vc "^+#"` gives **0
non-comment additions**, and `grep -E "^[+-](jax|torch)"` gives **0 entry-line
changes**. A comment-only ledger change should be demonstrably inert, not asserted
to be.

**#1223** is the turnphysicaloff burndown, ledger **156 -> 150**.

Caught a self-consistency problem before pushing: the reading note I merged in
#1222 says "total entries 156", and this very PR makes it 150. Left alone I would
have shipped a note that its own file contradicts one commit later. Refreshed it to
150 / 48 / 102 and verified each number against a live count rather than arithmetic
(`grep -c` for both the total and the VerticalInverse subset).

That is a small thing but the general shape matters: **a note that states counts
becomes wrong the moment the counts move, so whoever moves them owns the note.**
Worth remembering for every future burndown PR -- the ledger now has a header that
must be updated in lockstep.

## Wake 44 -- task #171 root-caused: it is a LATENT TEST-DESIGN issue, measured at 1 ulp

Took on the sphericaldf/qdf cluster (13 entries) with the warning I had written
myself: do not assume another as_numpy fix, verify first. Good thing.

**It is not a units bug and not a backend bug.** Reproducing the test's exact
comparison:

```
  numpy  lhs= 1.64026633916166115e+10
         rhs= 1.64026633916166115e+10   absdiff = 0          (bit-identical)
  jax    lhs= 1.64026634257233601e+10
         rhs= 1.64026634257233620e+10   absdiff = 1.9073e-06
                                        reldiff = 1.16e-16   (~1 ulp)
```

The two sides agree to **one ulp**. The test's bound is
`numpy.fabs(physical - scaled_natural) < 1e-8` -- **ABSOLUTE**, on a quantity of
magnitude 1.6e10. Float64 eps at that magnitude is 3.6e-6, so the bound is
strictly tighter than representable: it can only be satisfied by BIT-IDENTITY.
numpy satisfies it by accident of identical operation order; jax misses by a
single ulp.

`test_quantity.py` uses that absolute form **882 times**. It is fine for
quantities of order unity and silently demands bit-identity for large ones
(masses ~1e10, densities). So the cluster is one story: *tests whose compared
quantity is large enough that an absolute 1e-8 bound means bit-identity.* This is
**latent test design that predates the backend work** -- the backends only exposed
it.

**Deliberately did NOT change it.** The only real fix is absolute -> relative
tolerance, and the standing rule is that tolerance changes get PROPOSED and
approved per batch ([[torch-tolerance-loosening-ok]]), not taken unilaterally. It
is also not blocking: the entries are ledgered and CI is green. So this waits for
Jo, with the numbers ready.

Options for him, cheapest first:
1. leave ledgered (13 entries, now accurately explained);
2. targeted: make the ~13 affected assertions relative -- small diff, looks
   arbitrary next to 869 untouched absolute ones;
3. a shared `assert_physical_matches_natural(...)` helper with a relative bound,
   used at the affected sites and available for the rest -- cleanest, extensible,
   biggest diff.

My recommendation is (3) scoped to the affected sites only, because it names the
comparison being made instead of hiding it in an arithmetic expression, and gives
the remaining 869 an obvious migration path if they ever bite.

Worth noting what the warning bought: had I "fixed" this by casting with as_numpy
as I first assumed, I would have changed nothing and then loosened a tolerance
without understanding why -- and would not have found that the file has 882
instances of the same latent pattern.

---

## Wake 45 (2026-07-30) — the ledger has been lying, and the tell is invisible

**PR #1224 open: drop 16 stale torch `test_orbit` entries. Ledger 150 -> 134.**

The overnight full torch orbit shard finished (`exit=0`, 2h15m). Its stdout was
truncated -- the conftest force-exit that works around the torch+fork deadlock
([[torch-fork-deadlock-streamdf]]) cuts the `-rX` summary off -- so `grep ^XPASS`
found nothing even though the progress dots clearly showed a run of `X`s. Read the
`--junitxml` instead, exactly as that note says to, and computed the set
`(ledgered AND passed)`:

```
ledgered torch entries in test_orbit.py + test_orbits.py : 18
of those, PASSED (= XPASS, stale)                        : 16
of those, still xfailed                                  :  2
```

The 2 survivors are `test_orbit_obs_Orbits_issue322` and
`test_orbit_obsvel_Orbits_issue322`; they stay. None of the 16 is
torch-slow-skipped, so all 16 genuinely ran.

**Every one of the 16 was added by `be996203` -- the commit that CREATED the
ledger (#956).** They were fixed by the intervening dxdv/C-STM routing PRs, the
energy-conservation work and the forced-backend coercion PRs, and never retired.

### Why this went unnoticed for so long

`conftest` applies `xfail(strict=False)`. A fixed entry therefore reports XPASS,
the shard still exits 0, and the count keeps carrying it. **Nothing anywhere goes
red, or yellow, or prints a warning.** The only way an entry retires is if someone
re-measures and looks. That is a genuinely bad property for the number the project
uses as its progress metric, so #1224 also adds a STALENESS paragraph to the
reading-note saying so and giving the two-line recipe (`-rX`, or junitxml +
`ledgered AND passed`).

Also fixed a stale `108` I left in that note last wake, which disagreed with its
own table. Now 134 / 48 / 86, each from a live `grep -c`.

### The bigger implication, and the free way to test it

16 of 18 stale in one file is not a local accident. The header's own prune log
shows the sanctioned method is a `workflow_dispatch regen=true` full-matrix run,
and the **last one was 2026-07-23 on `feat/backends` post-#1185**. Everything from
#1196 to #1223 has landed since, none of it pruning. So the same backlog is
presumably sitting in the other 14 files.

I was about to spend runner time dispatching a regen when I noticed I do not have
to: **a PR into `feat/backends-develop` already fires the whole matrix, and every
shard uploads its JUnit XML as an artifact.** So #1224's own run yields exactly the
input the computation above needs -- for all 15 files, in the authoritative CI
environment, on both Python versions, at zero extra runner cost. Next wake: pull
the artifacts, compute `(ledgered AND passed)` per shard, and prune what it finds.

(Aside on reading workflow files: the `on:`/`if:` filters in the main checkout are
gated to `feat/backends` only, which says my PRs should get no backend coverage at
all. They do. The main checkout is on `feat/backends`, whose copy of the file is
older -- `feat/backends-develop`'s version lists both branches, with a comment
explaining that without it "the jax/torch dimension would go unverified on exactly
the branch whose whole purpose is that dimension." Read the workflow from the
branch that runs it.)

### Self-inflicted, caught

While breaking the ledger down per-file I ran `git checkout <ref> -- <file>` to
get the pre-prune population, which silently reverted and STAGED away the prune I
had just committed -- precisely
[[checkout-ref-file-stages-a-revert]]. The commit was already made and pushed so
nothing was lost, and `git status` was clean again before anything else touched
that file. Reading the population from `git show <ref>:<file>` into a pipe, which
is what I should have done, cannot do this.

### Wake 45, part 2 — did the sweep immediately instead of waiting, and it corrected me

I did not need to wait for #1224's own run. **A completed earlier run's artifacts
were already sitting there**: run `30510631354`, the post-#1223 push on
`feat/backends-develop`, all 44 shards green, 45 artifacts totalling **103 KB**.
Downloaded them and computed `(ledgered AND passed)` over the whole matrix.

One wrinkle: pytest's junitxml has no `file` attribute, only
`classname="tests.test_quantity"`, so the nodeid has to be rebuilt as
`tests/<mod>.py::[<Class>::]<name>`. First pass silently matched 0 of 7270
testcases, which is the kind of zero that means "my parser is wrong", not "nothing
is stale".

```
ledger entries                 150
observed running                144   (the 6 unobserved are exactly the -jit
                                       entries -- this workflow has no --jit
                                       dimension, so unmeasured, not stale)
  of those, xfailed (still real) 124
  of those, PASSED  -> stale      20
```

**The sweep overruled my local measurement, in both directions.**

*It removed one.* My local run said 16 stale in `test_orbit`; CI says 15.
`test_analytic_ecc_rperi_rap` passes on my box and **xfails in CI**, so it stays
ledgered. Two reasons, and the second is the one I had not internalised: the
backend matrix is **python 3.14** and this box is 3.13
([[regen-py-version-blind-spot]]), *and* CI shards `test_orbit` into **5
pytest-split groups** -- this test lands in `g3`. So a local run of "the whole
shard" is not the CI shard even when it is the whole file, because CI's shard is a
*split* of the file. That sharpens [[verify-exact-ci-shard-context]]: match the
split, not just the file list.

*It added five*, in files I had not looked at: 2 `test_actionAngle` (
`IsochroneApprox_plotting`, `Staeckel_indivdelta_EccZmaxRperiRap`), 1
`test_noninertial`, 1 `test_quantity` (`test_potential_paramunits`), 1
`test_streamdf` (`test_bovy14_callMargXZ`).

#1224 rewritten to the CI-authoritative set of 20. **Ledger 150 -> 130, reducible
102 -> 82.** Force-pushed and cancelled the prior SHA's 8 queued runs -- one of
which was the 44-shard matrix, so that is real runner time back
([[cancel-stale-pr-ci-on-push]]).

The reading-note now carries the recipe, so the next person does not have to
rediscover it: PR into an integration branch -> matrix runs -> download
`backend-junit-*` artifacts -> `(ledgered AND passed)`. And a warning not to do it
on a dev box, with this exact miss as the evidence.

**The healthy number to take away: 124 of 150 entries are genuinely still
failing.** The ledger was carrying ~13% dead weight, not the majority I half
expected after seeing 16/18 in one file. `test_orbit` was the outlier because it
is where the most fixing has happened.

### Wake 45, part 3 — mined the 124 real failures, found a plain-numpy bug

With the ledger now telling the truth, I broke the **124 genuinely-failing**
entries down. 48 are `actionAngleVerticalInverse` (refuses backends by
construction), leaving **76 addressable**. Of those, 14 clusters fail on **both**
jax and torch — and a both-backend failure is not a backend idiosyncrasy, it is a
galpy-side gap. That is the list worth working.

Picked `test_physical_vertical` (2 entries) because my notes already called it "a
pre-existing decorator quirk". It is worse than a quirk.

`physical_conversion_actionAngle` decides whether a method returned a *sequence*
of quantities or *one* quantity like this:

```python
try:
    for ii in range(len(out)):
        newOut = newOut + (out[ii] * fac[ii],)
except TypeError:   # happens if out = scalar
    newOut = out * fac[0]
```

`len()` does not answer that question — a length-N array has a `len()` too. A 1D
`__call__` returns the action J alone, one array-valued quantity, so it took the
loop; and `fac` for `"__call__"` is always three long. Therefore, **on plain numpy,
no backend anywhere:**

```
>>> aAV = actionAngleVertical(pot=IsothermalDiskPotential(...), ro=7., vo=230.)
>>> aAV(numpy.linspace(-0.1, 0.2, 4), numpy.full(4, 0.1))
IndexError: list index out of range
```

and at n ≤ 3 it returns a *tuple of n scalars* where the natural call returns a
length-n array. Both `actionAngleVertical` and `actionAngleHarmonic`, in the plain
and the astropy-Quantity branch alike. `numpy` broadcasting `tuple - ndarray` is
the only reason nobody noticed; jax/torch raise on `tuple - Tensor`, which is how
the ledger entry came to exist. **The backend work found a latent numpy bug** —
the same pattern as the `test_quantity` 1e-8 discovery, and the second time this
week that a "backend failure" turned out to be galpy telling the truth.

Fix: dispatch on type, which is what the code meant. `_actionAngleInverse` has the
same shape but not the bug (its 1D `__call__` genuinely returns `(x, v)`) — left
alone rather than "made consistent".

**Byte-identity, checked rather than asserted:** ran 16 probes against both
revisions and compared float *bytes* — 3D `__call__`/`actionsFreqs`/`aFA`/
`EccZmaxRperiRap` for Staeckel, Isochrone, Spherical, array-valued 3D input, and
the 1D `actionsFreqs`/`aFA`. **All 16 identical.** What changes is the container of
the 1D `__call__` under physical output, plus n ≥ 4 no longer raising. Flagging it
as a user-visible API change even though no value moves
([[flag-numpy-byte-identical-changes]]).

Tests: containers must match at n = 1,2,3,4,9 (straddling the old ceiling) with
**exact** equality — the decorator's only arithmetic is one multiply by `ro*vo`, so
a tolerance would be admitting I did not know what it computes. Plus a counterpart
pinning 3D methods as tuples whose entries keep their **own** factors (actions by
`ro*vo`, frequencies by `freq_in_Gyr`), which is exactly what collapsing to one
shared factor would break. Pass on numpy/jax/torch; both fail without the fix
(stash-reran, [[verify-burndown-flips-stash-rerun]]).

Committed locally on `fix/aa-physical-conversion-1d`, **not pushed** — #1224 is
still the one PR in flight. Full-shard verification running meanwhile (numpy
aA+quantity, torch aA, jax aA). The ledger drop for the 2 entries goes in after
#1224 merges and this rebases, so the two PRs never touch the ledger at once
([[all-clear-queued-rerun-and-ledger-rebase]]).

This one is arguably `main` material, not backend material — `conversion.py` is
shared and the IndexError has nothing to do with jax. I am putting it on
`feat/backends-develop` because that is where the ledger entries live and where I
can merge; worth splitting a `main` cherry-pick when Jo is back.

---

## Wake 46 (2026-07-30) — a self-inflicted red, and the burndown map

**#1224 still queued** (14 of 153 checks in; `codecov/project` shows
`25.70% (-60.48%)` with 138 shards pending and `codecov/patch` green, which is the
partial-upload artifact for a ledger-text-only PR — it has self-cleared five times
on exactly these circumstances).

### The numpy shard "failure" was mine

Last wake's verification of the conversion fix came back `torch exit=0`,
`jax exit=0`, **`numpy exit=1` with 104 failures**. All 104 were "does not return
Quantity when it should" — across `Orbit.E`, `evaluatePotentials`, `jeans.sigmar`,
`streamdf`, things my actionAngle-decorator change cannot reach. Cause: I ran

```
pytest tests/test_actionAngle.py tests/test_quantity.py
```

in one process. `tests/test_quantity.py` sets `astropy-units = True` **at import
time, before any galpy import**, because the config is read once and latched; with
`test_actionAngle.py` collected first, galpy is already imported and the setting
never takes. CI never combines those two — `test_quantity.py` has its own shard.
Re-run as CI shards it: **`quantity+coords exit=0`**. Clean.

This is [[conftest-no-module-level-galpy-import]] from the other side: not conftest
importing galpy, but *me* pairing a config-latching test file with one that imports
galpy first. Worth remembering that "run more files together for coverage" is not
free — some files are only correct alone.

So the conversion fix now has torch aA clean (209 tests), jax aA clean (209), numpy
quantity+coords clean; numpy aA still running. Waiting on #1224 to merge before
pushing, per one-PR-at-a-time.

### Mapped the 76 addressable failures into three classes

First tried to classify all 124 from the CI artifacts, which does not work and is
worth writing down: an xfail is recorded as
`<skipped type="pytest.xfail" message="backend-xfail-ledger"/>` — conftest's reason
string, no traceback. **The artifact sweep tells you WHICH entries are stale, never
WHY the rest fail.** Noted in [[ledger-staleness-sweep-from-artifacts]] so I do not
retry it. Classification still needs a local `--runxfail` per candidate.

Did that for the both-backend clusters, and they fall into three groups:

**A. `hyp2f1` fallback domain — 6 entries, one root cause, clearly fixable.**
`_euler_labeling` needs a parameter that is positive *and* has `c - B >= 1`:

| case | example | why it raises | scipy at z=-2 |
|---|---|---|---|
| (i) | `a=-3.2, b=4.4, c=5.2` | `b>0` but `c-b=0.8 < 1` | 24.7061738568 |
| (ii) | `a=-0.51, b=-0.98, c=2.51` | **both** `a,b <= 0`, no valid `B>0` | 0.602686827309 |

and the error message is **wrong for (ii)**: it reports "requires
`c - max(a,b) >= 1`", but that quantity is 3.02 there. The real blocker is the sign.
Euler's transformation maps (ii) to `(3.02, 3.49, 2.51)`, which lands straight in
(i), so this needs the `t=1` endpoint regularized the way `X=xi^k` already handles
`t=0` — real special-function work, not a patch. Task #174.

**B. unbound-orbit detection — ~4 entries, probably one root cause.**
`test_actionAngleStaeckel_unboundr_angles_c` ("does not have large ap") and
`test_qdf.py::test_call_diffinoutputs` ("orbit that is supposed to be unbound is
not") are the same story from two directions: the unbound sentinel does not survive
the backend path.

**C. absolute bounds that demand bit-identity — the [[torch-tolerance-loosening-ok]]
pile.** `test_actionAngleStaeckelGrid_basicAndConserved_actions` asserts
`|Jr| < 1e-16` for a circular orbit; torch gives `4.8354e-14`. That is a fine
numerical zero being measured against a bound 480x tighter.
`test_actionAngleAdiabatic_linear_angles_cylsep` misses by `4.36e-06 %` = 4.4e-8
relative. Same family as the 13-entry `test_quantity` cluster in #171 — **all of it
waits on one decision from Jo**, and I am accumulating the magnitudes so the
proposal is quantified rather than a request to "loosen some tolerances".

Incidental confirmation of #1224: `test_actionAngleIsochroneApprox_plotting`, one of
the 20 entries it drops, **passes** in an independent isolated run too.

### Wake 47 (2026-07-30) — #174 designed off measurements, not guesses

**#1224: 98 pass / 54 pending / codecov-only red.** Still the partial-upload artifact.
**All of #173's local verification is now green** — numpy `test_actionAngle` exit 0,
numpy `test_quantity`+`test_coords` exit 0, torch aA 209/0, jax aA 209/0. It is
ready to push the moment #1224 merges.

**Second confirmation that this box over-reports XPASS.** The jax orbit shard from
wake 45 finished: 584 tests, 0 failures, and **both** jax `test_orbit` ledger
entries (`test_analytic_ecc_rperi_rap`, `test_analytic_zmax`) XPASS locally — while
CI says both still xfail. Combined with the torch disagreement on
`test_analytic_ecc_rperi_rap`, that is three data points where local py3.13 XPASSes
and py3.14 CI xfails, all in the same ecc/zmax family. Systematic, not noise, and
exactly why #1224 was rebuilt off CI artifacts. (There is a `galpy-py314` conda env,
but it is the **free-threaded** 3.14t build — it looks for
`libgalpy.cpython-314t-*.so` — so it is not a faithful CI proxy. Parked rather than
chased.)

**Spent the CI wait designing #174 properly.** Three findings, each measured:

*1. hyp2f1 is the ONLY blocker for those 6 entries.* Patched `hyp2f1_fallback` to
return scipy's value for the out-of-regime parameters and ran the three tests under
jax `--runxfail`: **tests=3 failures=0**. So this is not "a fix that might help" —
it is the whole gap.

*2. The gap is only ever hit at small |z|.* Instrumented spy over a full run:

| (a, b, c) | in regime | \|z\| |
|---|---|---|
| (-3.2, 4.4, 5.2) | no | 0.617 |
| (-0.98, -0.04, 2.98) | no | 0.617 |
| (-0.51, -0.98, 2.51) | no | 0.060 |
| (-2.2,2.4,4.2), (-1.9,1.8,3.9), (-1.2,0.4,3.2) | yes | 0.617 |

I had been sizing this problem against `|z| ~ 500`. The actual requests are all
below 0.62, which changes the cost picture completely.

*3. There are two sub-cases, and only one is representable by the existing
integral.* Enumerating identity/Euler/Pfaff-1/Pfaff-2 for a parameter in `(0, c)`:

- `(-3.2, 4.4, 5.2)`: Pfaff-1 gives `(A,B) = (-3.2, 0.8)` with `c-B = 4.4 >= 1` —
  **in regime**. So the current guard is stricter than mathematically necessary; it
  is a quadrature-accuracy condition, and a Pfaff transformation first makes the
  existing 128-node machinery apply unchanged.
- both `a,b <= 0`: **none** of the four transformations lands a parameter in
  `(0, 2.51)` (identity `(-0.51,-0.98)`, Euler `(3.02,3.49)`, Pfaff-1
  `(-0.51,3.49)`, Pfaff-2 `(3.02,-0.98)`). The Euler integral genuinely cannot
  represent it — so this branch needs a series, not a smarter labeling.

For that branch there is a tidy general result: apply the Pfaff transformation that
puts `max(a,b)` in the `c-·` slot and the Gauss series has `C-A-B = |a-b| > 0`,
hence converges even in the `x -> 1` (`z -> -inf`) limit. Prototype vs scipy:
**machine precision (1e-16..1e-14) at |z| = 0.5 .. 500** for both gap cases. Term
counts to 1e-10: case (i) 10 terms at |z|=1 and only 27 at |z|=500; case (ii) 22 at
|z|=1, 166 at 10, 732 at 50. Given measured usage below 0.62, a small fixed-length
sum is ample and trace-friendly — with an accuracy-vs-|z| table in the docstring,
matching how the existing quadrature documents its own degradation.

Prototypes kept in `scratchpad/hyp_proto{,2}.py`. This is the shape of PR #174: two
routes, the in-regime path untouched so everything working today stays
byte-identical, plus the misleading error message corrected to name the real
condition (the sign, not `c - max(a,b)`).

### Wake 48 (2026-07-30) — #1224 green bar one shard; #174 built and passing

**#1224: 153 pass, 1 pending, 0 failures.** `codecov/project` cleared on its own
exactly as expected once the shards finished uploading — the sixth time that
partial-upload pattern has self-resolved. The last check is the
`test_backend*.py` coverage shard. Pre-merge audit done and clean: one file, zero
`galpy/` source changes (so the `if`->`xp.where`, duplication and helper-placement
checks are all vacuous here), header counts match live greps exactly (130 - 48 =
82), no conflict markers, exactly 20 removals and 0 additions.

**#174 (hyp2f1 domain) is implemented, tested and committed locally.** Three
things worth recording, because two of them changed the design.

*The Pfaff transformation cannot be used for route 2.* My first plan was "Pfaff to
get in regime, reuse the quadrature". Wrong: Pfaff maps `z <= 0` to `z/(z-1)` in
`[0,1)`, and the integral's boundary-layer substitutions assume `z <= 0`
(`log1p(-z)` etc.). **Euler's** transformation is the one that leaves `z` alone,
turning `(-3.2,4.4,5.2)` into `(8.4,0.8,5.2)` — in regime, and measured **7e-15**
across `|z| = 0.06 .. 500`. Caught by testing rather than by reading, which is the
only reason it did not ship wrong.

*Route choice is not a free improvement, so I did not touch it.* Measuring all
three routes on in-regime triples:

| (a,b,c) | route 1 (today) | route 2 Euler | route 3 series |
|---|---|---|---|
| (1.5, 2.5, 3.5) | **6.8e-15** | 1.1e-06 | 0 |
| (-2.2, 2.4, 4.2) | 2.1e-06 | **1.0e-14** | 2.2e-11 |
| (-1.2, 0.4, 3.2) | 2.7e-08 | n/a | 1.9e-05 |

They are complementary — no route dominates. So re-routing in-regime parameters
would change existing numpy values in *both* directions, which is not this PR's
business. `_in_regime` is therefore checked **first**, and I verified byte-identity
over 10 in-regime triples x 9 values of z: **all identical.** The change is a pure
capability addition.

**A separate finding worth Jo's attention:** that table shows route 1 is only
**2.1e-06** at `|z|=50` for `(-2.2,2.4,4.2)` — a triple galpy itself requests —
while the docstring claims "~1e-10 or better vs scipy at realistic radii
(|z| <~ 50)". It is already 1.9e-07 at `|z|=1`. The claim does not hold for
negative `a`. Not fixed here (it would change numpy values), but it means the
in-regime path is weaker than advertised, and Euler's transformation would
*improve* exactly those cases if we ever decide to prefer it.

*The honest limit on route 3.* `a == b` (so `|a-b| = 0`) reaches the series and,
at 128 terms, returned a **1.25e-02** error at `|z|=50` — a silently wrong number,
precisely what the original author's comment refused. I checked whether a static
gate on `|a-b|` would help: it would not, because the dominant variable is `|z|`,
not the parameters (even `|a-b|=4` degrades at `|z|=50`, while at `|z| <= 1`
everything including `a == b` is exact). Since `|z|` is data, it cannot be raised
on under a trace. So I sized the term count instead: **512 terms** makes route 3
exact to <=9e-13 through `|z|=20` and 2e-6 at `|z|=50` — which is what the
quadrature route also delivers there, so up to `|z|=50` the new route is no less
accurate than the one galpy already ships. Beyond that it falls off faster;
documented with measured numbers and pinned by a test **from both sides**, so a
silent loss and an accidental improvement both fail.

Payoff verified before and after: making the fallback return scipy's value for just
these parameters took the three tests to 0 failures (so hyp2f1 was the sole
blocker), and the real fix reproduces that — **torch 3/0**. jax check running.
63 hyp2f1 tests pass on both backends. Committed on
`fix/hyp2f1-fallback-domain`, unpushed; #173 goes first.

### Wake 48b — #1224 MERGED, #1225 opened

**#1224 merged as `7fd43d7d`. Ledger 150 -> 130.** All 154 checks green;
`gh pr merge` printed `failed to run git: 'feat/backends-develop' is already used
by worktree at /var/tmp/nogl`, which is the known cosmetic worktree complaint — the
merge itself had already succeeded (`state=MERGED`). Checking PR state rather than
trusting the exit code is the habit that makes this a non-event.

**#1225 opened**: the `physical_conversion_actionAngle` fix, rebased cleanly onto
post-#1224 develop (disjoint files, no conflict) plus the 2-entry ledger drop it
earns. **Ledger 130 -> 128, reducible 82 -> 80**, both refreshed against live greps.

The PR leads with the plain-numpy reproducer rather than the backend symptom,
because that is the real defect:

```
>>> aAV = actionAngleVertical(pot=IsothermalDiskPotential(...), ro=7., vo=230.)
>>> aAV(numpy.linspace(-0.1, 0.2, 4), numpy.full(4, 0.1))
IndexError: list index out of range
```

and flags the container change (tuple -> array for the 1D physical `__call__`) as
user-visible even though all 16 byte-probes are identical, with a note that
`conversion.py` is shared so this is arguably `main` material to cherry-pick later.

Two PRs' worth of ledger burndown now sits behind #1225 and #174: 128 -> 126 -> 120
if both land. Of the remainder, 48 are `actionAngleVerticalInverse` (refuses
backends by construction), so the reducible count would be 72.

### Wake 49 (2026-07-30) — the "unbound" cluster is two different things

**#1225: 2 pass, 149 pending, no reds.** #174 is committed and its four real CI
shards are running locally (`tests/test_backend*.py` with the exact
`-W error::DeprecationWarning -W error::FutureWarning` CI invocation, torch and jax
`test_sphericaldf`, numpy `test_potential`). Note I first launched the backend shard
as just two files and corrected it to the full ~78-file glob — "the shard" means
CI's whole `TEST_FILES`, and a two-file subset is exactly the mistake
[[verify-exact-ci-shard-context]] is about.

**Diagnosed cluster B, and it is not one bug.** I had it filed as "unbound-orbit
detection, ~4 entries, probably one root cause". Wrong on both counts.

*2 entries are a real, narrow bug.* For an unbound orbit numpy returns the 9999.99
sentinel in every angle; torch returns **3.442** for `js[7]` and the sentinel
everywhere else. `9999.99 mod 2*pi = 3.4421762772780866` — the sentinel is being
*folded*, not preserved. Located at `actionAngleStaeckel.py:628`:

```python
anglephi = xp.remainder(anglephi + phi, 2.0 * numpy.pi)  # fold in the azimuth
```

which folds unconditionally, where the C wrapper explicitly guards it
(`actionAngleStaeckel_c.py:1017`, `badAngle = Anglephi != 9999.99`). The test uses
`c=True`, but I instrumented `actionAngleFreqAngleStaeckel_c` and **it is never
called under a forced backend** — the C path is skipped entirely and the backend
implementation, which lacks the guard, runs instead. Fix is an `xp.where` on a data
mask (traceable, no Python branch), and byte-identical on numpy because the numpy
path raises `UnboundError` at line 299 before any unbound value reaches the angle
code. Task #175.

*The other 2 are by design and will never pass.* `test_qdf::test_call_diffinoutputs`
requires `UnboundError` to be **raised**; under a backend it gets `Jr=inf`. The code
says why, at `actionAngleStaeckel.py:293`: "Unbound orbits raise eagerly on the
numpy path ...; under a backend they must stay jit-traceable, so we cannot branch on
the traced `unbound`". Raising means branching on a traced value. So the test asserts
numpy-only behaviour, and either it becomes backend-aware (assert non-finite instead
of a raise) or those 2 stay ledgered forever.

**That matters for the burndown arithmetic.** They belong with the
`actionAngleVerticalInverse` 48 in a third category: not stale, not a bug, just
structurally unreachable. Running total of what the ledger's 130 actually contains:

| | entries |
|---|---|
| VerticalInverse (refuses backends by construction) | 48 |
| unbound-raise under trace (by design) | 2 |
| absolute bounds demanding bit-identity (awaits one decision) | ~15 |
| genuinely code-fixable | ~65 |

The honest "reducible by fixes" number is nearer **65 than 82**, and I would rather
say that now than keep quoting a figure that flatters the work. Two of those 65 are
#1225's (landing), 6 are #174's (ready), 2 are #175's (localised, one-liner).

### Wake 50 (2026-07-30) — correcting yesterday's localisation, twice

**#1225: 35 pass, 116 pending, 2 codecov reds** (`patch` and `project`).
`project` shows `52.82% (-33.36%)`, which cannot come from a 34-line diff, and
`patch` cannot report on lines whose shards have not uploaded — same premature
pattern as #1224, which cleared on its own. I tried to settle it locally rather than
assume, but `coverage run` in this env dies with
`ImportError: cannot load module more than once per process` out of numpy's
`_multiarray_umath`, so local coverage is simply unavailable. Falling back to direct
evidence: all four branches of the new conditional are exercised — 3D-with-units and
1D-with-units by the old/new probe I ran earlier, the two no-units paths by the new
tests. No genuine gap; waiting on uploads.

**#174: three of four shards clean** (torch `test_sphericaldf`, jax
`test_sphericaldf`, numpy `test_potential` all exit 0); the full
`tests/test_backend*.py` shard is still going.

### I got #175's location wrong twice, and the correction is the useful part

Yesterday I recorded: "`actionAngleStaeckel.py:628` folds the sentinel; the C path is
skipped under a backend". Both halves were wrong.

Getting exact values instead of `%.4g` was what broke it open. numpy and torch agree
**bit-for-bit at exactly 9999.99 on all nine outputs** except `anglephi`, where torch
gives `3.4421762772781` = `9999.99 mod 2*pi`. Bit-identical sentinels mean the C code
ran in **both** cases — so "the C path is skipped" could not be true, and the fold
had to be happening after C returned.

Then I instrumented `_staeckel_angles` (the line-628 function) and it is **never
called** for this test either. So neither of my candidate sites was involved. The
c=True class under a backend routes to a *third* path: the C-native differentiable
wrappers from #1051, which call a different C entry and do their own fold:

```
galpy/backend/_torch/staeckel_c.py:99   anglephi = torch.remainder(raw[6] + phi, 2.0 * torch.pi)
galpy/backend/_jax/staeckel_c.py:155    anglephi = jnp.remainder(raw[6] + phi, 2.0 * jnp.pi)
```

unguarded, where the numpy C wrapper guards it (`actionAngleStaeckel_c.py:1017`,
`badAngle = Anglephi != 9999.99`).

This is a *better* place to fix it than what I had: those files are backend-only, so
byte-identity on numpy is not something to argue for — no numpy code is touched at
all.

Two process notes worth keeping. My first spy said "the C entry is never called under
torch", which was true of `actionAngleFreqAngleStaeckel_c` specifically; I
generalised it to "the C path is skipped", which was not true — a different C entry
fires. Wrapping *one* candidate and concluding from its silence is how that happens;
wrap all the candidates and see which one speaks. And
`galpy.actionAngle.actionAngleStaeckel` resolves to the **class**, not the module, so
monkeypatching needs `sys.modules['galpy.actionAngle.actionAngleStaeckel']` —
`setattr` on the name silently patches a class attribute that nothing calls.

### Wake 51 (2026-07-30) — #175 implemented; codecov converging as diagnosed

**#1225: 138 pass, 14 pending, `codecov/project` the only red.** `codecov/patch`
cleared on its own, and `project` is walking up as shards upload:
`52.82% (-33.36%)` -> `86.18% (-13.78%)`. That is the partial-upload signature
converging, not a real drop.

**#175 implemented and committed** on `fix/staeckel-unbound-anglephi`, guarding the
sentinel in both C-grad wrappers:

```python
anglephi = xp.where(raw[6] == 9999.99, raw[6],
                    xp.remainder(raw[6] + phi, 2.0 * pi))
```

Under the CI configuration (jax x64 / torch float64) both backends are now
**bit-identical to numpy on all nine outputs**, and the ledgered test passes on both
with `--runxfail`.

Worth naming why this bug was worse than a wrong number: `9999.99 mod 2pi = 3.442`
lands *inside* `[0, 2pi)`, so it reads as a perfectly ordinary azimuthal angle. Any
caller testing "is this unbound?" against the other eight outputs — which all carry
9999.99 — would pass, while `anglephi` silently said "bound". A NaN would have been
loud; this was quiet.

Three judgement calls in it:

- **Where the constant lives.** I first wrote `_UNBOUND` and then had to decide where
  to define it. `galpy/backend/` has no constants module, and importing it from
  `galpy.actionAngle` is impossible — that package imports *these* files. Creating a
  module for one number to avoid two literals is worse than the literals, especially
  since ~8 existing consumers already spell it out. So: literal, with a comment
  naming the numpy guard it mirrors and why the import is not available.
- **Exact equality on a float.** It mirrors the numpy wrapper (`!= 9999.99`) and the
  value is a constant C writes, so there is no arithmetic to perturb it. I did check
  the float32 case out of caution: my first probe (no x64) showed `9999.990234`, and
  the comparison still worked because both sides downcast identically — but that was
  my probe missing `jax_enable_x64`, which conftest sets. Re-ran in the CI
  configuration to confirm exactness rather than rely on a coincidence.
- **Test placement.** `tests/test_backend_actionAngle.py`, not
  `tests/test_actionAngle.py` — partly because that is where backend-behaviour tests
  belong, and partly to keep files disjoint from #1225, which is editing the other
  file right now.

Ledger drop deferred until this rebases onto post-#1225 develop, so the two PRs never
touch `backend_xfail.txt` simultaneously.

Also: `ruff-format` rejected the first commit attempt (reformatted a file). Re-added
and re-committed — the pre-commit hooks modify files rather than just complaining, so
a rejected commit is normal and needs `git add -A` again, not investigation.

**Queue: #1225 landing, then #174 (6 entries), then #175 (2 entries).** 128 -> 122 ->
120.

### Wake 52 (2026-07-30) — #1225 MERGED; the full shard earns its keep

**#1225 merged as `48146a7e`. Ledger 130 -> 128.** `codecov/project` cleared itself
once the last shards uploaded (52.82% -> 86.18% -> green), exactly the partial-upload
pattern. Audit before merge: the two new `if`s are `isinstance(out, (tuple, list))` —
a *type* check — and `_APY_UNITS`, a module-level flag; neither branches on array
values, so neither is the `if`-should-be-`xp.where` class. The restructure also
*removes* duplication (two near-identical try/except blocks became one type dispatch),
and adds no helpers.

**The full `tests/test_backend*.py` shard caught something my subset run missed** —
2h13m, 6460 passed, **2 failed**:
`tests/test_backend_constantbetadf.py::test_hyp2f1_domain_limit[jax,torch]`.

That test exists to lock the *old* boundary:

```python
# ... cannot reach the general-beta Hernquist DF for beta>=0.5
# (b=1-2beta<=0 after the Pfaff transform) ... This locks that documented boundary.
with pytest.raises(NotImplementedError):
    dfh.fE(_arr(backend, ...))
```

My fix moves that boundary, so the test fails by construction. Had I pushed on the
strength of the three quick shards, CI would have caught it — but the point of
running the real shard locally is not to be caught. Worth noting the shard took
**2h13m**, which is why the temptation to run a two-file subset exists and why it
keeps being wrong.

**And the failure is good news.** `constantbetaHernquistdf(beta=0.7)` does not just
stop raising — it now agrees with scipy to **1.55e-15** on both backends. So this PR
is bigger than "6 ledger entries": **the entire general-beta Hernquist family
(beta >= 0.5) was numpy-only and is now available on jax and torch.** That is a
capability the ledger never measured, because there was no test asserting it could
work — only one asserting it could not.

Rewrote the test accordingly: parametrised over beta = 0.5, 0.7, 0.9, asserting value
parity against the scipy path at `rtol=1e-13`. It locks the *new* boundary as tightly
as the old one was locked, rather than just deleting an assertion.

Also measured, rather than assumed, whether two further torch entries
(`test_constantbeta_interpolatedpotentials_{beta,dens_directint}`) fall out of the
same fix — runs in flight on both backends before I finalise the ledger drop.

### Wake 52b — #1226 opened (hyp2f1 domain)

**#1226 open. Ledger 128 -> 122.**

Measured the two extra torch `constantbeta` entries rather than assuming they came
along with the fix, and they did not:

```
torch  PASS constantbetadf_against_hernquist
torch  PASS osipkovmerritt_hernquist_dMdE_limits
torch  PASS osipkovmerritt_hernquist_dMdE_limits_generalimplementation
torch  FAIL constantbeta_interpolatedpotentials_beta
torch  FAIL constantbeta_interpolatedpotentials_dens_directint
         RuntimeError: vmap: ... calling .item() on a Tensor or ...
         data-dependent control flow
jax    3 PASS, 2 skipped (slow-skip)
```

Same file, same DF family, unrelated cause — the interpRZPotential track (#112).
Dropping them because they looked related would have been a silent false prune, and
the ledger has burned me in both directions this fortnight, so: 6 entries, each
verified with `--runxfail` on both backends.

`test_backend_constantbetadf.py` full file: 36 tests, 0 failures with the rewritten
domain test.

The PR leads with what the ledger could not count — `beta >= 0.5` gives
`b = 1-2beta <= 0`, the both-non-positive case, so the whole general-beta Hernquist
family was numpy-only. **No ledger entry existed for it, because no test asserted it
should work; one asserted it could not.** That is a blind spot worth naming: the
ledger measures tests that exist, so a capability nobody tried is invisible to it,
and "entries retired" undercounts what a fix actually buys.

Queue: **#1226 in CI, then #175** (2 entries, committed and verified). 122 -> 120.

### Wake 53 (2026-07-30) — the shape of what is left

**#1226 in CI** (1 pass, 150 pending). **#175 rebased** onto post-#1225 develop and its
four real shards launched (the ~2h `tests/test_backend*.py` among them).

Re-derived the ledger's composition rather than trusting my running notes, and the
picture has changed qualitatively. Of the 128 on develop (122 once #1226 lands):

| | entries |
|---|---|
| `actionAngleVerticalInverse` (refuses backends by construction) | 48 |
| both-backend clusters | 28 |
| singletons | 52 |

**Nearly every remaining both-backend cluster is already spoken for**: AdiabaticGrid
off-grid (a design decision, #103), the qdf unbound pair (by design), and the rest
tolerance-class. The genuinely-fixable both-backend work is close to exhausted — which
means the remaining reducible work has moved into the **singletons**, and those are
overwhelmingly **torch** (7 `test_actionAngle`, 5 `test_quantity`, 5
`test_streamTrack`, 4 `test_noninertial`, ...).

Diagnosed the six torch `test_actionAngle` singletons that form two thematic groups,
and they split 5:1:

```
Spherical_linear_angles                 5.00909e-06 deviation from linear trend
Spherical_linear_angles_fixed_quad      5.00909e-06
Spherical_noninclinedorbit_linear_angles 1.17986e-06
Spherical_almostnoninclinedorbit_...     1.17986e-06
Vertical_Harmonic_actionsFreqs          2.76469e-07 %  (= 2.8e-9 relative)
Vertical_linear_angles                  TypeError: all() received an invalid
                                        combination of arguments ...
```

The last one is `numpy.all(<torch tensor>)` at an assertion site — test-side, not a
source bug, byte-identical for numpy. One more free entry (#176).

The other five join the tolerance batch, and I have now consolidated that batch with
measured magnitudes (#171): **~24 of the ~72 reducible entries hinge on that single
decision** — the 13 `test_quantity` absolute-1e-8-on-1e10 cluster, these 5, the
Adiabatic cylsep pair at 4.4e-8, the StaeckelGrid pair asserting `|Jr| < 1e-16`
against a 4.8e-14 numerical zero, and the AnyAxisym quadrature pair. That is a third
of the reducible ledger sitting behind one call, so it is worth Jo seeing it as a
single costed proposal rather than as recurring noise.

Refined running total of what the ledger actually contains:

| | entries |
|---|---|
| VerticalInverse — structurally blocked | 48 |
| unbound-raise under trace — by design | 2 |
| tolerance decision — awaiting Jo | ~24 |
| genuinely code-fixable | ~48 |

Two of those 48 are #175's (verified, queued), one is #176's.

### Wake 54 (2026-07-30) — measuring the traced state while #1226 uploads

**#1226: 21 pass, 130 pending, both codecov checks red and both premature.**
`patch: 20.00% of diff hit (target 86.18%)` is not a coverage gap — it is 20% of the
*uploads* being in. The shard that exercises the new code is
`tests/test_backend*.py`, which is also the coverage shard, and it has not reported
yet. Same for `project: 61.37% (-24.81%)`; a 173-line diff cannot move the project
figure by 25 points. On #1225 both cleared at ~40 shards pending; here we are at 130.

**#175's actionAngle shards are clean on all three backends** (numpy, torch, jax);
the ~2h `tests/test_backend*.py` is still going.

**Dispatched `--jit` runs of `test_sphericaldf` on both backends** against the #1226
tree. This is the measurement the mission actually cares about — the ledger's
`-jit` section is down to **6 entries** (3 AnyAxisym, 2 interpRZ, 1 orbit
eccentricity), and the constant-beta / OsipkovMerritt DFs previously could not be
traced at all because `hyp2f1` raised before tracing began. Whether they now trace,
rather than merely evaluate eagerly, is a separate question from the 6 entries #1226
retires, and worth knowing before claiming the DF family is "backend-ready".

Also probing all 35 remaining torch singletons with `--runxfail` to classify them the
way the six `test_actionAngle` ones were classified — the useful output is not the
pass/fail but the *shape* of each failure: a `TypeError`/`AttributeError` naming a
numpy function applied to a Tensor is a free test-side fix, a quantified `%`
deviation joins the tolerance batch, anything else is real work. That distinction is
what turns 52 opaque singletons into a plan.

### Wake 55 (2026-07-30) — the spherical-DF suite is fully jittable

**Measured, on the #1226 tree:**

| run | tests | failures | skipped | time |
|---|---|---|---|---|
| `test_sphericaldf --backend jax --jit` | 223 | **0** | 28 | 10.5 min |
| `test_sphericaldf --backend torch --jit` | 223 | **0** | 2 | 18.5 min |

and — the part I checked rather than assumed — the three hyp2f1-blocked tests
**PASS** under trace on both backends rather than inheriting their eager xfail. That
distinction matters: `<backend>-jit` inherits `<backend>`, so a ledgered-eager test
would have shown as skipped and the file would have looked "green" either way. They
genuinely trace.

So #1226 is not only "6 eager entries". **The whole spherical-DF suite now traces
green on both backends** — constant-beta and OsipkovMerritt included, which
previously could not be traced at all because `hyp2f1` raised before tracing began.
That is the mission's actual currency (jittable, not merely backend-agnostic), and it
is worth stating in those terms rather than as a ledger delta.

**#1226: 108 pass, 43 pending, both codecov checks still red and still converging.**
`patch` was 20% of diff hit when a fifth of the uploads were in; the shard that
exercises the new code is `tests/test_backend*.py`, which is the coverage shard and
reports late. Watcher armed.

**#175: clean on numpy, torch and jax `test_actionAngle`**; its backend shard is
still running. The 35-singleton classification probe is still running too — that one
is the input to planning the tail, so it is worth its runtime.

Remaining `-jit` ledger is **6 entries**: 3 AnyAxisym (the quadrature-accuracy
problem awaiting the tolerance decision), 2 interpRZ (#112), 1 orbit eccentricity.
Nothing in the DF families.

### Wake 56 (2026-07-30) — #1226 MERGED, #1227 opened

**#1226 merged as `bc6f079e`. Ledger 128 -> 122.** 154 checks green; both codecov
checks cleared themselves once the coverage shard uploaded, which is now the third
consecutive PR where the premature-codecov diagnosis held.

Audit before merge: the three new conditionals in `hyp2f1.py` are all on **scalar
DF/potential parameters** (`if b > a`, `if not _in_regime(a,b,c)`,
`if _in_regime(c-a,c-b,c)`) — static under a trace, so this is not the
`if`-should-be-`xp.where` class; the only data-dependent logic in the new code is
already a `where`. The three new helpers live beside their sole consumer in
`hyp2f1.py`, which is correct — they are hyp2f1-specific, not general utilities that
belong in a shared module.

**#1227 opened**: the Staeckel unbound-sentinel guard, rebased onto post-#1226
develop, plus its 2-entry ledger drop. **Ledger 122 -> 120, reducible 74 -> 72.**

Both PRs deliberately leave a neighbouring pair alone and say so in the description:
#1226 kept the two `interpolatedpotentials` entries (measured: unrelated `vmap`
cause), and #1227 keeps the `test_qdf::test_call_diffinoutputs` pair (requires
`UnboundError` to be *raised*, which cannot happen under a trace). Both would have
looked like natural companions to drop. Naming why they stay is the difference
between a ledger that means something and one that just gets smaller.

**Bookkeeping caught up.** The galpy-jax README PR table had stopped at #1219 —
seven PRs of drift, against my own rule that it is part of landing each PR. Added
rows for #1220-#1223 (eager-ledger wave), #1224 (artifact sweep, 150->130), #1225
(the plain-numpy `IndexError`), #1226 (the beta>=0.5 unlock + `test_sphericaldf --jit`
223/0). That table is what Jo will actually skim, so letting it lag is worse than it
looks.

Ledger arc for the fortnight so far: **156 -> 120**, and the composition is now
understood rather than guessed — 48 structurally blocked, 2 by design, ~24 awaiting
one tolerance decision, ~46 genuinely fixable.

### Wake 57 (2026-07-30) — 52 opaque singletons resolve into three causes

**#1227: 28 pass, 124 pending, `codecov/patch` the only red** (premature, as on the
last three PRs). Watcher armed.

**The singleton classification finally worked, after fixing how I ran it.** The
first attempt — all 35 torch singletons in one pytest process — timed out at 3h
(exit 124), wrote no junitxml, and left me with a row of dots: nothing enumerable,
exactly [[triage-gather-misses-timed-out-shards]]. Relaunched as 10 per-file jobs
with `--timeout=300` each; all completed in minutes, each writing its own XML. A
hang in one test can no longer consume the whole run.

**31 failures, and they are not 31 problems.** 21 are "a numpy function or operator
meeting a torch Tensor". I was about to file all 21 as free test-side casts. Then I
pulled the **failing frame** out of each junitxml rather than reading only the
exception message — and the message cannot tell you where the collision is, because
`ndarray OP Tensor` reads identically whether it happens in a test assertion or deep
inside galpy:

| where the failing frame lives | count |
|---|---|
| `tests/` — free `as_numpy` cast at the assertion site | **11** |
| `galpy/` — **real source defect** | **10** |

Casting at the test site would have hidden ten real bugs. That check cost one script
and changed the conclusion completely.

And the ten source ones collapse into **two** causes:

- **8 at `quasiisothermaldf.py:2213`** (+1 at `:2420`):
  `indx = VDatprop > numpy.log(numpy.random.random(size=nmore))` — a torch Tensor
  compared against a freshly drawn **numpy** random array. This is the known qdf
  sampler boundary, and [[qdf-sampler-numpy-boundary]] already records that a
  boundary wrap alone is *insufficient* (tried before): it needs the backend RNG
  shim, i.e. existing task #130. Folding them there rather than attempting a local
  patch.
- **2 at `Orbits.py:9738/9822`**: `thiso[3] - numpy.arctan2(obs[1], obs[0])` — the
  orbit state is a backend array while `obs`, a user-supplied observer *parameter*,
  stays numpy. Tractable with `as_backend_constant` at those sites, with care not to
  coerce `obs` as if it were a coordinate.

So the tail is not 52 separate problems. It is **~12 assertion-site casts, one
sampler needing its RNG migrated, one observer-parameter coercion, ~24 awaiting the
tolerance decision, and the structurally-blocked remainder.** That is a plan, and it
is the first time this fortnight the singletons have been anything other than a
number.

### Wake 58 (2026-07-30) — building the test-side cast PR, and a subtlety in it

**#1227: 104 pass, 48 pending, `codecov/patch` the only red** (premature, fourth PR
running). Watcher armed.

**Built the 11-entry test-side cast branch** (`fix/torch-assertion-casts`). Six sites
were the simple shape — `numpy.amax(numpy.fabs(<Tensor expr>))` in dynamfric ×4,
`numpy.all(numpy.fabs(o.L() - ndarray))` in galpypaper, and two list comprehensions
in `test_potential::test_rE_MWPotential2014` that build object arrays of Tensors.
Two files needed the `as_numpy` import added.

**The four `test_noninertial` sites were not that shape, and the difference matters.**
Those tests already `as_numpy`-wrap every orbit accessor — carefully and
consistently. The Tensors re-enter *afterwards*, through the `coords.*` helpers:

```python
op_xs = as_numpy(op.x(ts)) + x0[0](ts)          # already numpy
Rp, phip, _ = coords.rect_to_cyl(op_xs, op_ys, op_zs)   # <- comes back a Tensor
...
vTp += omega * Rp                                # ndarray * Tensor -> TypeError
```

`coords.rect_to_cyl` is backend-agnostic, and under a **forced** backend it coerces
even numpy inputs — which is correct, deliberate behaviour
([[coerce-under-forced-not-dataguard]]). So the test's numpy reference arithmetic,
which mixes in the plain-Python `x0`/`v0` callables, collides with a backend array
that arrived from a *helper*, not from the object under test.

The fix therefore is not "cast at the assertion" — the TypeError fires well before
any assertion. It is to land each `coords.*` result back on numpy at the point it
returns, which is exactly what the surrounding code already does for the accessors.
12 casts across the 4 tests, each with the reason stated once at the top.

Worth naming as a pattern: **a test can be fully as_numpy-converted at its
boundaries and still fail, if a migrated helper re-introduces backend arrays in the
middle.** Looking only at assertion sites would not have found these.

Verification of the six simple sites is running (they are slow orbit-integration
tests); the four noninertial ones are running separately. Neither is finished, so
nothing is claimed yet.

### Wake 58b — the cast PR is not as free as it looked

Verification of the 11-site cast branch came back mixed, which is the useful part.

**Worked (8/8):** all four ledgered `test_noninertial` tests pass, plus the four
unledgered siblings in the same `-k` selection — so the `coords.*` casts fixed the
target without perturbing neighbours. 5.8 min for the batch.

**Two problems, both worth having found locally:**

*1. `test_rE_MWPotential2014` still failed after my first patch.* I had cast the two
list comprehensions, but missed that the grid itself is seeded from backend values:

```python
Emin = Ec(rmin)          # <- Tensor under a forced backend
Es   = numpy.linspace(Emin, Emax, 101)   # carries Tensors into Es
```

so `Ecs - Es` still met a Tensor. Casting the comprehensions treated the symptom;
the fix is to land `Emin`/`Emax` on numpy at the source. Re-running. The lesson is
the same one as the noninertial sites from the other direction: I patched where the
*error* surfaced instead of where the backend value *entered*.

*2. `test_dynamfric_c` TIMED OUT at 600s.* Not a cast failure — the test now runs to
completion instead of aborting early at the TypeError, and integrating orbits with
dynamical friction under torch eager is genuinely slow. This is
[[migration-flips-numpy-fallthrough-to-eager-slow]] exactly: fixing an error can
convert a fast failure into a slow pass, and a slow pass **fails CI as a timeout**
([[timeout-overrides-xfail-move-to-slowskip]]). So this entry may belong in
`backend_slow_skip.txt` rather than being un-ledgered. Measuring how slow it actually
is with a 3600s allowance before deciding — un-ledgering it on the strength of "it
passes locally in 10 minutes" would hand CI a red.

That reframes the PR honestly: of the 11 entries, some are clean un-ledgers, at least
one is a slow-skip re-home, and the count is not final until measured. Better to know
that now than after pushing.

**#1227: 111 pass, 42 pending, `codecov/patch` cleared — zero failures.**

### Wake 59 (2026-07-30) — #1227 MERGED; the cast branch committed with an honest caveat

**#1227 merged as `d2a0e8c8`. Ledger 122 -> 120.** 154 checks green;
`codecov/patch` cleared itself again (fourth PR running). The audit was unusually
easy: this PR *is* the correct form of the class I audit for — a `where()` on a data
mask rather than a Python branch. The one thing I'd normally flag is that the
explanatory comment is duplicated verbatim across `_jax` and `_torch`, but those two
files are strict mirrors of each other throughout, so keeping them symmetric is the
local convention rather than accidental duplication.

**Cast branch rebased and committed** (`fix/torch-assertion-casts`), with the ledger
drop deliberately withheld.

`test_rE_MWPotential2014` needed a second pass. My first patch cast its two list
comprehensions and it still failed, because the energy grid is *seeded* from backend
values:

```python
Emin = Ec(rmin)                          # Tensor under a forced backend
Es   = numpy.linspace(Emin, Emax, 101)   # carries Tensors into Es
```

Casting `Emin`/`Emax` at the source fixes it (11s, green). Same lesson as the
noninertial sites, arriving from the other direction: **patch where the backend value
enters, not where the error surfaces.** It is now the commit's title.

**And one entry is not a free un-ledger.** `test_dynamfric.py::test_dynamfric_c` hit
a 600s timeout once the TypeError was gone — it used to abort early, now it runs to
completion, and orbit integration with dynamical friction under torch eager is slow.
A timeout **overrides** xfail and reds the shard, so "it passes locally" is not
grounds to drop it; it may belong in `backend_slow_skip.txt` instead. Measuring with
a 3600s allowance before deciding. The commit says so explicitly rather than quietly
un-ledgering 11 and hoping.

That is the whole point of running these locally: the PR is 11 sites but not 11
un-ledgers, and I would rather find that here than in a red shard.

### Wake 60 (2026-07-30) — #1228 opened: 120 -> 109, and one entry that is not a win

The dynamfric measurement came back and settled the question:

| test | torch | vs the 300s CI per-test timeout |
|---|---:|---|
| `test_dynamfric::test_dynamfric_c` | **3598s** | **12x over** |
| `test_FDMdynamfric::test_dynamfric_c` | 208s | 69% |
| both `_minr` | 36s | fine |
| `test_rE_MWPotential2014` | 11s | fine |
| `test_quasiisothermaldf_method_value` | 67s | fine |
| the four noninertial | ~43s each | fine |

So **10 un-ledgered, 1 re-homed to `backend_slow_skip.txt`** — not 11 retired.

**The reasoning that decides it is worth stating, because I nearly got it backwards.**
My first instinct was to slow-skip *both* `test_dynamfric_c` variants, since 208s
felt uncomfortably close to 300s on a loaded runner. That is wrong, and the reason is
that **an xfail test still RUNS**. The FDM sibling already spends its 208s in CI
today, as a ledgered xfail, without timing out — un-ledgering changes its status, not
its cost. Only `test_dynamfric_c`'s *runtime* actually changes, because only it was
exiting early at the TypeError. Fixing the error converted a fast failure into an
hour-long pass.

That is [[migration-flips-numpy-fallthrough-to-eager-slow]] in its purest form, and
the ledger note records the measured 3598s so the next person does not have to
rediscover it.

Also finished the 11th site, which I had listed but not actually patched:
`test_quantity.py:16774`, `qdf(o).to(...).value - qdfnou(o)/ro**3/vo**3` — the
Quantity side is numpy, the DF side a Tensor. Evaluation, not sampling, so it is
genuinely test-side rather than the qdf-sampler source bug that accounts for the
other eight `test_quantity`/`test_qdf` entries. 67s, green.

**#1228 open. Ledger 120 -> 109, reducible 72 -> 61.** Fortnight arc: 156 -> 109.

### Wake 61 (2026-07-30) — the `obs=` failure is a real API gap, not a test artefact

**#1228 converging** (70+ pass, `codecov/project` the only red — premature, fifth PR
running). Watcher armed.

Used the wait to settle the `Orbits.py obs=` pair properly, and the conclusion
changed once I probed instead of read.

The test does this:

```python
o.helioX(times[ii], obs=[obs.x(times[ii]), obs.y(times[ii]), 0.0], ro=1.0)
```

and under a forced backend `obs.x(t)` returns a **Tensor**, so the list galpy
receives is `[Tensor, Tensor, float]`, which then meets a numpy `thiso` at
`Orbits.py:9738`. My first instinct was to file this with the test-side casts — the
test is, after all, constructing the awkward list. But that is exactly what **any
user** would produce under a forced backend: they call an accessor and pass the
result to `obs=`. So this is a genuine backend-agnosticism gap in a **user-facing
parameter**, and casting in the test would hide it.

Design settled rather than implemented, because #1228 is the PR in flight:
`Orbits.py` has **four** `isinstance(obs, (numpy.ndarray, list))` branches (9735,
9766, 9818, 9904) and **no** obs-normalising helper. The fix is one module-level
private helper landing an obs sequence in the orbit state's namespace, called at all
four — not inline at the two that happen to be covered by a ledgered test. It stays
in `Orbits.py` rather than `galpy/backend` because it is Orbits-specific, not general
([[audit-placement-and-duplication]]). Byte-identity on numpy is a strict
pass-through and must be probe-verified, and the change fixes the same latent bug at
the two untested sites for free.

That is the third time this fortnight that "where is the failing frame?" changed the
classification of a failure. It is becoming the single highest-yield question to ask.

### Wake 62 (2026-07-30) — a real red on #1228, and a self-inflicted git mess

**#1228 has a genuine failure**, not a codecov artefact:
`tests/test_noninertial.py::test_linacc_changingacc_xyz_accellsrframe_scalaromegaz`
**FAILED under jax**. That test was ledgered for **torch only**, i.e. it was passing
under jax before — so my cast broke it. A regression I introduced.

The cause of the miss is plain: I verified the noninertial casts under **torch** and
never ran them under **jax**. The four entries were torch-ledgered, so I reasoned
about torch and stopped. But the file is shared, and a test-side change lands on
every backend. [[verify-exact-ci-shard-context]] says run the shard; I ran the
backend I was fixing.

**Then I made it worse trying to A/B it.** I reached for
`git stash` → run baseline → `git stash pop`. The tree was clean (I had just
committed), so the stash saved **nothing**, and the pop restored
`stash@{0}: On feat/native-grid-spline: WIP ...` — an unrelated branch's work — into
the worktree, conflicting in `actionAngleStaeckelGrid.py` and
`backend/interpolate.py`, files this branch never touches. **The stash stack is
per-repository, not per-worktree.**

Damage assessment, done before anything else: worktree clean, both commits intact,
`git diff develop..HEAD --name-only` shows only my 8 files, and local HEAD ==
`origin/fix/torch-assertion-casts`, so CI ran exactly what I intended and the mess
never left my machine. The stash itself was kept (the pop failed), so nobody's WIP
was lost. Recorded as [[git-stash-is-repo-wide-not-worktree]] — for A/B use two
worktrees, never stash.

Re-running the comparison properly now: AFTER in `/var/tmp/casts`, BEFORE in
`/var/tmp/nogl` (which sits at develop). No stash involved.

The honest read on the PR: the torch half is verified and the jax half was never
checked, which is exactly the kind of gap that makes a "free test-side cast" PR not
free. It stays open until jax is green.

### Wake 63 (2026-07-30) — regression found, root-caused, fixed, verified on both backends

The A/B settled it without argument: **develop `exit=0`, my branch `exit=1`** on the
same test under jax. The error:

```
ValueError: output array is read-only
```

**jax arrays are immutable, so `as_numpy` returns a READ-ONLY view**, and the
reference arithmetic in those tests mutates in place (`phip += ...`,
`vTp += ...`). **torch's cast is writable.** That asymmetry is the whole story: my
torch verification was green and told me *nothing* about jax.

Fixed with a module-level `_np_rw` (`as_numpy` + copy) at the 16 cast sites.
**Now green on both: jax 8/8 (14.7 min), torch 8/8 (5.7 min)** over the full
`-k linacc_changingacc_xyz_accellsrframe` selection, which includes 4 unledgered
siblings. Pushed; prior-SHA runs cancelled.

Two memories written, both about traps that produce *silent asymmetry* — the
dangerous kind, because the happy backend tells you everything is fine:

- **[[as-numpy-on-jax-is-readonly]]** — `as_numpy(jax)` is read-only, torch's is
  writable, so cast-then-mutate is a jax-only bug. And the generalisation that
  actually matters: **ledger entries are per-backend, test files are not.** A
  test-side change lands on every backend and must be verified on every backend, not
  merely the one whose entries are being retired. That was the entire cause of this
  miss.
- **[[git-stash-is-repo-wide-not-worktree]]** — from the cleanup: `git stash` is
  per-repository, so `stash pop` in a worktree can restore an unrelated branch's WIP.
  Use two worktrees for A/B, never stash.

Worth being blunt in this log: I shipped a regression into a PR because I reasoned
about the backend I was fixing instead of running the one I wasn't. CI caught it,
which is the system working, but the cost was a wasted full-matrix run on a
runner-constrained project. The fix to my process is mechanical rather than
attitudinal — for any change under `tests/`, run jax and torch before pushing.

### Wake 64 (2026-07-30) — #1228 re-running green so far; obs= implemented with the new discipline

**#1228 after the fix: 133 pass, 19 pending, only `codecov/project` red.** The jax
shard — the one that caught my regression — is still running; that is the check that
matters, and I am not calling this green until it lands.

**Implemented the `Orbits.py obs=` fix** (task #177B) on
`fix/orbits-obs-backend-arrays`. One module-level `_obs_asnumpy` helper called at all
**four** `isinstance(obs, (numpy.ndarray, list))` branches, not just the two a
ledgered test covers:

```python
return [as_numpy(o) if is_backend_array(o) else o for o in obs]
```

Scope stated honestly in the docstring: this makes `obs=` **accept** backend arrays;
it does **not** migrate those accessors, which are numpy by construction
(`numpy.arctan2`/`numpy.sqrt`/`numpy.zeros_like` throughout). Calling that out
matters — it would be easy to present a 4-line coercion as "helioX is now
backend-agnostic", and it is not.

Two details the recent mistakes fed directly into:
- the read-only trap does **not** apply here, because `obs` is only ever *read*
  (arctan2/sqrt/divide), never mutated — checked rather than assumed;
- **verified on numpy first**: 15 float-byte probes across `helioX/Y/Z`, the 3D
  `U/V/W`, both `obs=Orbit` and `obs=[list]` forms — **all identical**. Then jax AND
  torch launched together, not sequentially-and-optimistically.

Also a near-miss worth noting: my first patch attempt asserted on an import line that
did not match, because I had read the worktree *while `git worktree add` was still
settling* and acted on a stale view. The assert caught it and nothing was written —
which is the argument for asserting on the exact text you expect to replace rather
than doing a blind `str.replace`.

### Wake 65 (2026-07-30) — #1228 MERGED green; #1229 opened

**#1228 merged as `69c2fd29`. Ledger 120 -> 109.** All 154 checks pass, *including
the jax shard that caught my regression* — which is the only check I actually cared
about. Audit: zero `galpy/` changes (so the `if`->`xp.where` class is vacuous),
`_np_rw` defined once and used 16 times (the opposite of duplication), living in the
test file that needs it, no data-dependent branches, ledger counts matching the
header.

**#1229 opened: `obs=` accepts backend arrays. Ledger 109 -> 107.**

Nearly made a mess of the sequencing, and the failure mode is worth recording. I ran
the ledger-drop script *before* the rebase, the rebase then refused ("Please commit
or stash them" — my uncommitted `Orbits.py`), and the script had already edited the
**pre-#1228** ledger: 2 entries dropped from 120, while the header string it looked
for ("109"/"61") did not exist yet, so the count and the header disagreed. A silently
inconsistent ledger.

Two things saved it. The script `assert`s on the exact header text before replacing,
so the mismatch surfaced rather than being papered over. And — having learned this
the hard way an hour earlier — I did **not** reach for `git stash`; I restored the
file from HEAD, committed the source change, rebased, and only then applied the drop
against the correct 109 baseline. Order matters: **commit, rebase, then touch the
ledger.**

Verification followed the new rule rather than the old habit: numpy byte-identity
first (15 probes, all identical), then jax **and** torch together (2/2 each), plus
numpy `-k obs` 7/7 as a regression guard.

Fortnight arc: **156 -> 107**, with 48 of the remainder structurally blocked and ~24
awaiting one tolerance decision.

### Wake 66 (2026-07-30) — the tail is now fully classified

**#1229 early in CI** (1 pass, 150 pending, no reds). Used the wait to finish the
last unexamined group: the 7 singleton failures my classifier had binned as "other".
They are not miscellaneous — they are four more known causes:

| what | n | cause |
|---|---|---|
| `Timeout` at `galpy/util/multi.py:143` | **3** | the torch+fork `parallel_map` deadlock ([[torch-fork-deadlock-streamdf]]) — `test_fardalwmwpot_trackaa`, `test_plotting`, `test_streamTrack_multi_parallel`, all the same frame |
| `Timeout` at `exp1.py:66` | 1 | `test_analytic_ecc_rperi_rap`, the py-version-sensitive ecc/zmax family |
| `vmomentdensity does not return correct Quantity` | 2 | the `test_quantity` units cluster → tolerance batch |
| `supposed to be unbound is not` | 1 | by design (cannot raise under a trace) |

So **every one of the 52 torch singletons now has a named cause**, and the tail is
not 52 problems but a handful:

| group | entries | status |
|---|---|---|
| assertion-site casts | 11 | **DONE** (#1228) |
| `obs=` user-facing coercion | 2 | **#1229 in CI** |
| qdf sampler (`Tensor > numpy.random`) | 8 | needs the backend RNG shim (#130) |
| `parallel_map` torch+fork deadlock | 3 | known; slow-skip or fork-mode fix |
| tolerance / absolute-bound cluster | ~24 | **awaiting Jo's one decision** |
| by design (unbound raise under trace) | 2 | permanent |
| `actionAngleVerticalInverse` | 48 | structurally refuses backends |

That is the whole remaining ledger, accounted for. Two weeks ago it was a number;
it is now a list of six causes with owners.

The method that produced it, and the one I would repeat: **read the failing frame,
not the exception message.** The message is identical for a test-side cast and a
source defect; the frame separates them. It changed the classification four separate
times this fortnight, and twice stopped me from "fixing" a test that was correctly
reporting a real bug.

### Wake 66b — the fork deadlock has a cheaper exit than I thought, and a stale premise fell over

Investigated the `parallel_map` deadlock (3 entries). Last wake I ranked "switch to
spawn" as the preferred fix. **It is blocked, and the block is written in the source
I had not read**, `galpy/util/multi.py:202`:

```python
# Use fork-based parallelism (because spawn fails with pickling issues, #457)
ctx = multiprocessing.get_context("fork")
```

So I measured *how* blocked, by sweeping every default-constructible potential:

```
picklable = 57,  UNPICKLABLE = 3,  skipped 8 (need args)
  AnySphericalPotential            <lambda> held as an attribute
  DiskMultipoleExpansionPotential  local object KuijkenDubinski._parse_S...
  DiskSCFPotential                 same
```

The "spawn fails with pickling issues" blocker is therefore **three closures**, not a
general problem — hoist them to module-level callables and spawn becomes viable,
which would retire this whole deadlock class. Scoped as its own piece of work
(measure spawn's runtime first: it re-imports galpy per worker), definitely not
bundled into a ledger PR.

**And a premise of mine fell over, which is the more useful outcome.** Task #68 has
sat for weeks as "un-picklable *migrated* potentials (stored namespace modules)". A
`HernquistPotential` pickles fine **after a forced-torch evaluation** — the backend
migration does not break picklability at all. The three failures have nothing to do
with backends. Corrected the task rather than leaving a plausible-sounding but wrong
diagnosis in the queue for whoever picks it up.

Worth noting the shape of the error: I had a *reasonable-sounding* cause recorded,
never tested it, and it survived weeks because nothing forced it to be checked. The
sweep took one command.

**#1229: 15 pass, 137 pending, `codecov/project` the only red.**

### Wake 67 — the whole potential suite is jittable, and the pickle class is closed

Two results this wake, one measured and one built.

**Measured: `test_potential.py` under `--backend jax --jit` is fully green.**

```
tests 1326   failures 0   errors 0   skipped 120   time 338s
```

That is every potential in galpy, every method, traced end-to-end through
`jax.jit` at the `@backend_input` boundary. Task #158 was written as "fix the ~5
potentials that block traced-green"; there are now zero, and the ledger has no
`-jit` entries at all. The torch counterpart is running as I write this. I had
not measured this file traced since the jit stack landed, so the number was
stale by several PRs' worth of fixes — worth re-measuring the other big files on
the same basis rather than assuming the old gaps are still there.

**Built: every potential galpy ships now pickles.** 61/3 → **64/0**.

The three failures were `AnySphericalPotential`, `DiskSCFPotential`,
`DiskMultipoleExpansionPotential`, and the cause in both cases was a closure
pickle cannot look up by qualname: the `dens=` default was a lambda (in three
separate copies), and the dict-specified Sigma/hz profiles, the axisymmetric
`dens` wrapper and the phiME density are all built as closures at construction.

The design question — hoist the closures to module level, or drop-and-rebuild
them on unpickle — I did not have to answer myself: `ChandrasekharDynamicalFrictionForce`
already does the second, so the house idiom decided it. `__getstate__` deletes
the derived callables, `__setstate__` re-derives them from `_Sigma_dict` /
`_hz_dict`, which were already being kept for precisely this purpose. The
construction moved into `_set_dens_funcs` / `_set_rawdens` so `__init__` and
`__setstate__` share one copy rather than two that can drift.

The part worth stating: **the fix must not make a user's lambda pickle.** That
is a documented limitation with a test asserting it raises, and a `__getstate__`
that quietly dropped the density would have "fixed" the failure by discarding
the physics. The wrappers are dropped, the wrapped callable is kept, so a
`dens=lambda ...` still raises — asserted.

Verification, in the order that mattered:
- **858 method values bit-for-bit unchanged** on numpy across the four touched
  classes (`__call__`/forces/`dens`/2nd derivs/`mass` on a 5×4 (R,z) grid).
- **The new tests fail on the parent commit** — 5 failed / 54 passed there, 59
  passed here. A green test that was already green proves nothing.
- Coverage of the configurations the default constructor cannot reach: both
  Sigma profiles, both hz profiles, user-supplied profile callables, a
  non-axisymmetric density, and all three astropy-units combinations.

Two process notes. The first byte-identity probe reported "BYTE-IDENTICAL" over
**zero lines** — it had died on its own import assertion with stderr suppressed,
and an empty file diffs equal to an empty file. A clean-looking all-clear from a
comparison that never ran is the most expensive kind. Second, the test helpers
had to be hoisted to module level too: a `def` inside a test body is just as
unpicklable as a lambda, so the first run failed on its own fixtures rather than
on the code under test.

**#1229: 127 pass, 21 pending, 5 red — all five are `test_from_name_values`
failing on `DALServiceError: 503` from SIMBAD.** An upstream outage, not the
diff; the run has to finish before the jobs can be re-run. `codecov/project`
cleared itself again, as it has on all six PRs now.

**Traced measurements, same wake (jax, `--jit`, whole file):**

| file | tests | failures | errors | skipped | time |
|---|---|---|---|---|---|
| `test_potential.py` | 1326 | **0** | 0 | 120 | 338s |
| `test_actionAngle.py` | 209 | **0** | 0 | 30 | 166s |

Two of the three biggest files in the suite run end-to-end under `jax.jit` with
nothing failing. `test_qdf.py`+`test_diskdf.py` and the torch counterparts are
running. The point of re-measuring rather than trusting the old gap lists: task
#158 had said "~5 potentials block traced-green" and the real answer today is zero.

**qdf sampler (8 ledger entries) — scoped, and a recorded conclusion was wrong.**

I had this filed as needing a "focused investigation of ALL frozen attrs the
sampler path touches — whack-a-mole". Measured on a qdf **built under forced
torch** (the construction context that matters):

- exactly **2** frozen torch attrs exist, `_precomputergLzgrid` and `_precomputergLzmax`;
- `fmin_powell` on the DF **already succeeds**;
- the only failure is `Tensor > ndarray` at the accept step.

So the fix is `as_numpy(VDatprop)` at **two lines** — `quasiisothermaldf.py:2212`
(before the `numpy.reshape`, which also removes an `__array__`-on-Tensor
deprecation that would fail CI-only under `-W error::Deprecation`) and `:2420`.
`as_numpy` is a no-op on numpy, so the numpy path and the RNG draw order are
untouched.

Why the old note was wrong is the useful part. The July-21 attempt wrapped the
*whole sampler body* in `use("numpy", force=True)`, which makes new values numpy
while frozen attrs stay torch, so `_rg`'s `lz > self._precomputergLzmax` becomes
`numpy > torch`. That failure was caused by the wrap. Casting only the DF
*result* at the comparison leaves the DF evaluation on torch and the collision
never arises. **A fix that failed does not prove the class of fix is
insufficient** — the note should have named which approach failed and why, and
because it did not, it sat for ten days looking like a hard problem.

**Torch-traced result, and the cost that comes with it.**

`test_potential.py --backend torch --jit`: **1326 tests, 0 failures, 0 errors,
102 skipped — in 1792s.** So both backends are fully traced-green on the whole
potential suite. But the wall-clock is worth writing down next to jax's:

| backend | tests | failures | workers | wall |
|---|---|---|---|---|
| jax   | 1326 | 0 | 24 | 338s |
| torch | 1326 | 0 | 16 | 1792s |

Normalised for workers that is ~8.1k core-seconds for jax against ~28.7k for
torch — **~3.5x** the compute for the same suite, essentially all of it
`torch.compile` front-end cost (the run sat at 97% for roughly 25 minutes on the
last handful of tests). Green is green, but this says a per-push torch-traced CI
shard is not affordable at present; it belongs in a nightly. Filing that
observation rather than acting on it now — I cannot edit workflows anyway.

**Correction to the df numbers above.** The 6 failures I first got from
`test_qdf.py + test_diskdf.py` traced were **my harness's fault, not traced
gaps**: all six are in `test_diskdf.py` and all read `'NoneType' object has no
attribute ...`, i.e. the `DFcorrection` savefile one test writes and others
consume, scattered across xdist workers. `test_diskdf` is on the known
run-serially list and I ran it `-n 12`. `test_qdf.py` contributed zero failures.
The serial re-run is still going; until it finishes I am not claiming the file is
clean, only that the six do not mean what they appear to.

**Confirmed: `test_diskdf.py` traced serially is 129 tests, 0 failures** (678s).
So the six were entirely the xdist artifact, and the traced picture for this wake
is clean across the board:

| file | jax `--jit` | torch `--jit` |
|---|---|---|
| `test_potential.py`   | 1326 / 0 fail | 1326 / 0 fail |
| `test_actionAngle.py` | 209 / 0 fail  | — |
| `test_qdf.py`         | 0 fail        | — |
| `test_diskdf.py`      | 129 / 0 fail  | — |
| `test_sphericaldf.py` | 223 / 0 fail (measured wake 66) | 223 / 0 fail |

The lesson I nearly published as a finding: **a parallel-runner artifact and a
real defect look identical in a summary line.** Six failures in a traced run is
exactly the shape of "the trace exposed something", and I said as much before
reading them. What separated them was the *signature* — `'NoneType' object has no
attribute ...` repeated six times is shared fixture state, not six independent
numerical failures — plus the fact that `test_diskdf` is already on the
run-serially list for precisely this reason. Read the failures, then classify;
the count alone carries no information.

### Wake 68 — the qdf sampler fix, and my "two lines" was wrong

Built the qdf sampler boundary fix (branch `fix/qdf-sampler-boundary`, commit
`6dbce4ee`). It flips **6 torch ledger entries**, every one verified to fail
without it.

**I have to correct the previous entry.** I wrote that the fix was "exactly two
lines", on the strength of a probe where I hand-built `VDatprop` and checked the
comparison. Running it through the real path, `logmaxVD` — `fmin_powell`'s
optimum, fed straight into the same arithmetic — is *also* a backend scalar. So
it is four source sites, not two: the DF value and the optimum, in each of
`sampleV` and `_sampleV_preoptimized`. Then three test-side
`samples - qdf.meanvT(...)` assertions needed the same landing.

That is precisely the error the old note warned about and that I criticised the
old note for: **verifying a fix on a reconstruction of the code path instead of
the path itself.** I caught the July-21 version of this mistake and then made my
own variant of it within the hour. The difference is only that this one surfaced
in minutes rather than sitting in a memory file for ten days.

The direction still held — a boundary cast, not the whole-body `use("numpy")`
wrap — so the correction was additive rather than a rethink. Worth separating
those two things when judging a plan: *where* to put the boundary was right,
*how many* values cross it was guesswork I presented as measurement.

Two details worth keeping:

- `as_numpy` on a numpy array returns **the same object**, so the numpy path is
  structurally unchanged rather than merely numerically equal. That is a stronger
  byte-identity argument than a value diff.
- The samples are **not** bit-identical numpy-vs-torch, and I deliberately did
  not assert that they are. `fmin_powell` converges in a different number of
  evaluations on a backend (46 vs 37), so `maxVT` differs in its last bits and
  shifts the proposals themselves. Measured max difference 4.5e-10 over a
  12-sample draw, so the test asserts 1e-8 — a ~20x margin, tight enough that a
  real regression in draw order or values cannot slip through, and honest about
  where the difference comes from.

Also worth noting the two new backend tests are parametrised over jax and torch,
but only the **torch** cases fail without the fix — `jax_array > ndarray` happens
to work. The jax parametrisations are guards, not flips, and I am not counting
them as such.

**#1229: 152 pass, 2 pending, 0 red.** Ready to merge as soon as `all-clear` and
the last backend shard land.

**#1229 MERGED** (`f9ecaaa5`) — 154/154 green. Ledger **109 → 107**.

Audit before merging, against the three things I check every time:
- *`if` → `xp.where` array-input class*: nothing added. `is_backend_array(o)` is a
  **type** check over a user-supplied sequence, not value-dependent branching, and
  these accessors are numpy-by-construction so they are never traced.
- *Duplication*: the helper is defined once and called at four sites; inline it and
  the cast appears four times.
- *Placement*: `as_numpy` is scalar/array-only (a list passes straight through), so
  it cannot absorb this. `_obs_asnumpy` sits next to its only two consumers and is
  named and documented for galpy's `obs=` contract. Hoisting a one-line comprehension
  into `galpy.backend` with a single consumer would be designing against one example;
  revisit if a second appears.

One thing the audit caught that green CI would not have: the helper converts an
**ndarray `obs` into a list**. That is the sort of type change that goes wrong
silently — `obs * 2` would repeat rather than scale. Checked every use in both
branches: `obs` is only ever indexed (`obs[0]`, `obs[1]`, `obs[2]`), never used in
whole-object arithmetic. Safe. But "the tests pass" would not have told me that.

**PR #1230 opened**: potential picklability (64/0, byte-identical, 5 tests that fail
on the parent). Next in the queue behind it is the qdf sampler boundary, already
built and committed on `fix/qdf-sampler-boundary`.

### Wake 69 — a headline of mine was wrong, and the ledger said so

I reported last wake that "the ledger has no `-jit` entries at all" and closed the
jit-burndown task on it. **That was wrong.** I ran the check in
`/home/bovy/Repos/galpy`, which is the MAIN worktree sitting on `feat/backends` —
a different branch, ~30 commits behind. Against `origin/feat/backends-develop`
there are **six**:

```
jax-jit   test_orbit.py::test_eccentricity
jax-jit   test_interp_potential.py::test_interpolation_potential_use_c
jax-jit   test_interp_potential.py::test_interpolation_potential_force_use_c
jax-jit   test_potential.py::test_2ndDeriv_potential[AnyAxisymmetricRazorThinDiskPotential]
jax-jit   test_potential.py::test_poisson_surfdens_potential[AnyAxisymmetricRazorThinDiskPotential]
torch-jit test_potential.py::test_poisson_surfdens_potential[AnyAxisymmetricRazorThinDiskPotential]
```

This is exactly the trap I have a memory note about — *diff the ledger against
origin, not the main repo* — written after making the same mistake before.

**What it does to the traced result.** "test_potential.py under jax `--jit`: 1326
tests, 0 failures" is literally true and materially misleading. I checked the
junit: both `AnyAxisymmetricRazorThinDiskPotential` entries were **xfail-SKIPPED**
under the ledger, so they never ran. A suite reports zero failures precisely
because the failures are ledgered — that is what the ledger is *for*, and I quoted
the number as if it meant the opposite.

The torch run is the stronger of the two after all: both tests **ran**, and the
`torch-jit` poisson_surfdens entry now **XPASSes**, i.e. it is stale and can be
dropped.

Corrected statement: the potential suite is traced-green under torch, and under
jax **except** `AnyAxisymmetricRazorThinDiskPotential`'s 2nd derivatives and
Poisson-surfdens check. Task reopened with the real remaining list.

The general form of this mistake is worth naming, because it is the second time
this fortnight: **"0 failures" is a claim about the run, not about the code.**
With an xfail ledger in play, the two only coincide if you also check what the
ledger suppressed. `xfail(strict=False)` means a green shard can hide both a
known failure and a silently-fixed one.

**Merged + queued.** `fix/qdf-sampler-boundary` is rebased onto post-#1229
develop with the ledger drop applied: **107 -> 101** (reducible-by-fixes 59 -> 53),
6 entries retired. Waiting behind #1230 (picklability), whose CI is running.

**And a correction to the correction, within the hour.** I wrote above that the
`torch-jit` RazorThin poisson_surfdens entry "now XPASSes, i.e. it is stale and
can be dropped". Then I ran it directly:

```
torch-jit RazorThin poisson_surfdens (isolated, --runxfail): 1 test, 1 failure
  TypeError: Methods in AnyAxisymmetricRazorThinDiskPotential do not accept array inputs
```

**Full-file and isolated disagree.** In the whole-file `--jit` run it recorded as
a plain pass; alone it fails with the same TypeError as its jax twin. So it is
**context-dependent** — trace-cache warmth or test ordering — not stale. Had I
acted on the first reading I would have dropped a live entry, and the shard would
have gone red on a future run with nothing pointing at why.

Both readings were of real data; neither was sufficient alone. The rule I already
have written down covers it — *reproduce the exact CI shard context, not an
isolated subset* — and I applied it in the direction I happened to start from. The
useful generalisation is that **it cuts both ways**: an isolated run can make a
passing test look broken just as easily as a full-file run can make a broken test
look fixed. A ledger edit needs the shard.

Meanwhile the other two jax-jit entries are confirmed **real**:
`test_interp_potential::test_interpolation_potential_use_c` and
`::test_interpolation_potential_force_use_c` both fail in isolation (interpolation
grid disagrees with the C result under trace). Those tie into the pending
interpRZPotential migration.

**#1230**: 5 pass, 146 pending, 2 red — both codecov, at 5 of 151 shards reported.
The premature-codecov pattern; leaving it to fill in.

### Wake 70 — #1230's red was upstream, and my sweep had a blind spot

`build (windows-latest, 3.14, ...)` went red on my own new tests:

```
TypeError: cannot pickle 'module' object
  when serializing dict item '_xp'
  when serializing scipy.interpolate._interpolate.BPoly state
  ... dict item '_I_inner_cos' ... MultipoleExpansionPotential
```

**scipy 1.18.0** on that runner (read out of the install log); my local env is
**1.17.1**. scipy 1.18 caches the array namespace — a *module* — on `PPoly`/`BPoly`,
so any object holding one stops pickling. `MultipoleExpansionPotential` is
affected and **nothing in this branch touches it**: the new test surfaced a
pre-existing upstream problem rather than a regression I introduced.

Two things worth recording.

**My 64/0 sweep had a blind spot I did not state.** It ran on one scipy version.
"Every potential pickles" was really "every potential pickles on scipy 1.17.1",
and I quoted the first. The number was right about the closures I fixed and wrong
as a general claim.

**I did not guess a fix.** The tempting move is a `__getstate__` that strips the
cached `_xp` — five lines, plausible, and completely unverifiable here because I
cannot reproduce the failure on 1.17.1. Pushing that would have cost a full CI
cycle to learn whether a guess was right. Excluded the two multipole entries with
the cause named in the code, and filed the reproduction (install 1.18 in a
throwaway env) as its own piece of work. The PR's claim is now stated with its
dependency: all default-constructible potentials round-trip on scipy 1.17, all
but the two multipole classes on 1.18.

Also cancelled the superseded run on the prior SHA — the project is runner-bound.

**Reproduced the scipy 1.18 break properly** (throwaway venv, scipy 1.18.0 + numpy
2.5.1; the main env stays on 1.17.1):

```
fresh BPoly dict keys: ['_delegate_to', '_xp', '_xp_internal']
fresh pickles: FAIL TypeError cannot pickle 'module' object
```

Two things this changes. First, the attributes are there **from construction**,
not cached lazily on first call — so the trigger is not "some earlier test warmed
it", it is simply "scipy 1.18 is installed". Second, and more important, the blast
radius is much larger than the two multipole potentials: **any** galpy object
holding a `PPoly`/`BPoly` is unpicklable on scipy 1.18 — `interpRZPotential`, the
`galpy.backend.interpolate` wrappers, DF interpolants. Same family as the
numpy-2.5.0 break.

Fix viability, measured rather than assumed:
- stripping `_xp`/`_xp_internal` before pickling **does not work** — scipy does not
  recreate them and the unpickled spline raises `AttributeError` when evaluated;
- reconstructing through the **public constructor** does, and is bit-identical:
  `(type(p), (p.c, p.x, p.extrapolate, p.axis))` round-trips `BPoly` and `PPoly`
  with `array_equal` on the evaluated values.

So a fix exists. Which shape it should take is a real decision and I am not taking
it unilaterally: a global `copyreg` reducer inside galpy is one place and
future-proof but changes pickle behaviour for a third-party class for anyone who
imports galpy; per-class `__getstate__`/`__setstate__` is conventional and
contained but has to be found everywhere and will silently miss future holders. My
recommendation is to report it upstream (a class that pickled for years no longer
does is almost certainly unintended) *and* ship the contained version so users on
1.18 are not broken meanwhile. Recorded with the measurements; not blocking, since
#1230 already excludes the two classes with the cause named in code.

**RazorThin (A) is much smaller than I scoped it.** I had written that the
R-dependent panel split `(0, R, 2R, inf)` meant the split quadrature would have to
be rewritten to carry a trailing node axis per point — a PR of its own. I tested
it instead of trusting the reading:

```
array _evaluate_gl -> ValueError: Incompatible shapes for broadcasting: [(3, 100), (3,)]
```

`_bk_split_quad` and `fixed_quad` **already** handle array limits — they produced
the `(3, 100)` node array. The only gap is that the integrands close over
scalar-shaped `R`/`z2`, so the node axis has nothing to broadcast against. Adding
it in-memory:

```
broadcast  -> [-1.52980074 -1.11440673 -0.7851416 ]
per-scalar -> [-1.52980074 -1.11440673 -0.7851416 ]
max rel diff = 0.0
```

Bit-identical. So the fix is `R[..., None]` / `(z**2)[..., None]` in three
integrands plus `_backend_accepts_arrays = True` — not a quadrature rewrite.

That is the third time this fortnight that reading the code produced a plausible
scope and running it produced a much smaller one (hyp2f1, the qdf sampler, now
this). The pattern is consistent enough to be worth acting on: **for "how big is
this?", the cheapest experiment beats the careful read**, because the read
compounds every uncertainty pessimistically while the experiment answers the
actual question. I have been treating measurement as verification-after-deciding;
it is cheaper as the deciding step.

### Wake 71 — RazorThin: four scopings, and the fix I shipped in my head was wrong

I said last wake this was "a small PR: 3 integrands + a flag". It is not, and the
way I found out is worth writing down because the same mistake produced all four
wrong estimates.

Sequence:
1. Read the code → "needs a per-point panel-split rewrite". **Wrong**: `fixed_quad`
   already takes array limits.
2. Measured one integrand → "3 integrands + a flag, bit-identical". **Wrong**: there
   are **six** `_*_gl` methods; `grep 'def _.*_gl'` would have said so in one second.
3. Patched all six → the original TypeError is gone, but now:
   - `test_2ndDeriv` fails with `unsupported format string passed to
     numpy.ndarray.__format__` — R2deriv returns shape `(1,)` for **scalar** input.
     `[..., None]` is not shape-preserving for the second derivatives: unlike
     `_evaluate_gl` they combine terms *outside* `_bk_split_quad`, which keep the
     added axis.
   - `test_poisson_surfdens` now **runs and asserts** — silently wrong numbers,
     which is a worse state than the TypeError it replaced.

So the current branch is **not shippable** and I am not committing it.

The common thread in all four: **I generalised from a representative measurement
to a universal claim.** One integrand measured bit-identical, so I assumed six
would be; one scipy version pickled, so I said "every potential pickles"; one
worktree's ledger had no `-jit` entries, so I said the ledger had none. Each
measurement was real. Each conclusion was broader than what was measured. The
cheap discipline is to state the scope of the evidence *in the claim* — "bit-identical
for `_evaluate_gl` on jax" is a sentence I could have written and then noticed was
not what I needed.

Untested next step recorded in the task: make the axis conditional on
`R.ndim != 0` (shapes are static under trace, so this is traceable), which leaves
the scalar path byte-identical by construction rather than by luck, then re-derive
the Poisson assertion by hand — the second-derivative integrands were edited by
regex and need reading, not trusting.

**#1230**: 39 pass, 112 pending, reds are codecov-only at ~40/153 shards reported.
**test_orbit.py jax --jit, groups 1-3 of 5: 300 tests, 0 failures.**

**RazorThin, continued — the guard works, the goal does not, and I am not shipping it.**

The per-value `ndim` guard was right: `v if v.ndim == 0 else v[..., None]`, per
value and only for arrays. It has to be per value because the traced Poisson
quadrature calls with a **scalar R and an array z**, so which arguments need the
node axis differs *within one call* — a single check on `R` would have been wrong
again.

With it:
- **scalar GL path bit-identical** — 18 values across all six integrands, byte-for-byte;
- **no regression** — RazorThin under jax eager 10/10, under numpy 10/10;
- `test_2ndDeriv` traced is back to its *original* pre-existing assertion, so my
  change is shape- and value-neutral there.

But `test_poisson_surfdens` traced now **runs and asserts**. And that is the point:
turning the flag on while the vectorised path returns wrong numbers replaces a
loud `TypeError` with a silently wrong value. **That is strictly worse than the
bug I set out to fix**, so the branch stays uncommitted with the flag off until
the vectorised Poisson value is explained.

Worth separating two things I had been treating as one: the *diagnosis* was right
(the guard rejects arrays; the integrands lacked a node axis) and the *fix for
that diagnosis* works. The failure is that fixing it revealed a second, unrelated
problem in the vectorised path — which the original TypeError had been hiding.
"The error I predicted is gone" is not the same as "the test passes", and only
the second one is the deliverable.

Since the scalar path is bit-identical, the next step is not to re-read my edits —
they are provably neutral — but to compare vectorised vs per-scalar evaluation on
the same node array directly, to localise which term diverges only under array
input. Suspects: `_bk_split_quad`'s `finite`/`positive` guards, computed from a
scalar `R` while `inner` carries the z batch, and the semi-infinite tail panel.

**RazorThin resolved — as a diagnosis, not a fix, and the distinction is the point.**

Localised it properly instead of re-reading my edits. Vectorised vs per-scalar,
array `z`, all six integrands:

```
_evaluate_gl   max rel 0.000e+00  OK
_Rforce_gl     max rel 0.000e+00  OK
_zforce_gl     max rel 0.000e+00  OK
_R2deriv_gl    max rel 0.000e+00  OK
_z2deriv_gl    max rel 0.000e+00  OK
_Rzderiv_gl    max rel 0.000e+00  OK
```

Exactly zero. **The vectorisation is right.** So the Poisson assertion is not mine:

```
surfdens traced = -4.16e-06     truth = 0.2967679     rel diff = 1.000014
```

It returns *zero*, not a slightly-wrong number. That is the physics. A razor-thin
disk's volume density is a **delta function at z=0**; the traced path integrates it
over z with fixed-order Gauss-Legendre, whose nodes never land on z=0, so the delta
contributes nothing. Eager passes only because concrete scalars route to scipy's
**adaptive** quad, which finds the spike. No amount of node-axis work touches this —
it is a quadrature-versus-distribution mismatch.

So the guard has been **hiding a real limitation behind a TypeError**. Turning the
flag on trades a loud error for a meaningless ~0. That is a policy question, not a
coding one, and it is the same question already decided once for
`RazorThin.R2deriv` off-plane under trace (answer then: return NaN). I am not
deciding it again unilaterally — options and a recommendation (mirror the NaN
decision) are in the task; the branch stays uncommitted.

What this wake actually produced is worth naming honestly: **no ledger entry moved,
and an unexplained xfail became an explained one.** I spent five scoping attempts
getting there, four of them wrong, but the failure now has a cause anyone can read
rather than "RazorThin fails under trace". That is a smaller deliverable than I
projected and a real one.

**test_orbit.py traced, complete: 581 tests, 3 failures, 121 skipped** (3856s over
5 pytest-split groups). That is the last big file measured under `--jit`.

| file | jax `--jit` |
|---|---|
| `test_potential.py` | 1326 / 0 fail (torch: 1326 / 0) |
| `test_actionAngle.py` | 209 / 0 fail |
| `test_orbit.py` | 581 / **3** fail |
| `test_qdf.py` | 0 fail |
| `test_diskdf.py` | 129 / 0 fail |
| `test_sphericaldf.py` | 223 / 0 fail (torch: 223 / 0) |

The three orbit failures: two are 1800s **timeouts** (`test_integrate_indiv_t_1D/2D`
— performance, not correctness) and one is `test_orbit_c_sigint_fulldxdv`.

I checked the sigint one in isolation rather than assuming xdist had mangled the
signal — **it fails serially too**, so it is real. Then, checking further: it fails
under jax **eager** as well, and it appears in **no** list — while its three
siblings (`sigint_full`, `sigint_planar`, `sigint_planardxdv`) are all jax
slow-skipped at `backend_slow_skip.txt:317-319`. The matcher is exact-nodeid /
param-stripped / file-level, not prefix, so `...sigint_full` does not cover
`...sigint_fulldxdv`. It looks like the fourth sibling was simply missed.

**But develop CI is green**, so I am not calling it a hole. Either CI skips it for
an environment reason (signal delivery on runners) or my box differs. Filed with an
explicit instruction to confirm against a backend-jax orbit shard artifact before
touching any list — the same discipline that stopped the torch-jit RazorThin entry
from being dropped wrongly. Local-only evidence has now misled me twice this
fortnight in both directions; a list edit needs the shard.

**Checked the sigint finding against real CI artifacts — and I was wrong, usefully.**

Pulled the five `backend-junit-jax-test_orbit_*` artifacts from #1230's own
All-backend run:

```
test_orbit_c_sigint_fulldxdv  -> PASS
test_integrate_indiv_t_1D     -> PASS
test_integrate_indiv_t_2D     -> PASS
```

So `sigint_fulldxdv` is **not** a missed skip-list entry. It passes on CI and fails
only on this box — a local difference in signal delivery. Had I acted on the local
evidence I would have added a `backend_slow_skip` entry that permanently hid a
**passing** test, and nothing would ever have flagged it. That is the more expensive
direction of this error: a wrong xfail gets caught by an XPASS sweep eventually, a
wrong *skip* is invisible forever.

Second thing the artifacts settled, which I would otherwise have got wrong in the
other direction: **CI's backend shards run EAGER, not `--jit`.** So these artifacts
can confirm eager claims and can say *nothing* about traced ones. The
`indiv_t_1D/2D` timeouts are traced-only and simply have no CI counterpart today —
nothing to ledger, just a note for whenever traced orbit joins CI (and then as
`slow_skip`, since a timeout records FAILED and xfail is the wrong key).

Worth stating plainly because it is the third time the rule has paid: I predicted a
gap, the check refuted it, and the check cost one command. The pattern from earlier
in the fortnight — measurement scoped narrower than the claim — has an inverse that
is just as costly, which is acting on a local reproduction as if it were the shard.

**#1230: 151 pass, 2 pending, `codecov/project` cleared** (seventh consecutive PR).
Only `codecov/patch` outstanding, with the coverage shard still to report.

### Wake 72 — the codecov patch red was real, and I nearly waved it through

`codecov/patch` stayed red on #1230 while `codecov/project` cleared as usual. My
first read was the familiar premature-codecov pattern, and the numbers seemed to
back it:

```
head: 48437 lines, 21 misses, 27 sessions
base: 48255 lines, 20 misses, 28 sessions
```

One extra miss, **and** one fewer upload session — so I reasoned the miss belonged
to the shard that had not reported yet. Wrong. When the last shard landed:

```
head: 21 misses, 28 sessions      base: 20 misses, 28 sessions
```

Equal sessions, still one extra miss. **A real gap**, and `all-clear` had gone red
behind it. Waving it through on the pattern-match would have merged uncovered code
and normalised ignoring the one check that was telling the truth.

The line, from codecov's `report` endpoint after the `file_report` and `compare`
endpoints gave nothing usable: `AnySphericalPotential.py:150`, the
`del pdict["_rawdens"]` in `__getstate__`. It runs only when `_dens_needs_numpy` is
True, which is only ever set inside `if _APY_LOADED` — so the unitful test that
covers it is **skipped wherever coverage is collected**. I have a note for exactly
this shape (astropy-gated branches need an astropy-free cover); it took finding the
line to recognise it.

Fixed with a forced-flag synthetic rather than an astropy requirement, asserting
what the branch is *for*: the wrapper leaves the state, **the callable it wraps
stays** (or the density is silently lost), and `__setstate__` rebuilds an
equivalent. Verified the branch actually executes before committing:

```
before force: _dens_needs_numpy = False | '_rawdens' in getstate: True
after  force: '_rawdens' in getstate: False -> del branch executed: True
```

The lesson is narrower than "don't trust patterns": the premature-codecov pattern
is real and has held seven times. What I got wrong was letting a **second**
observation (sessions 27 vs 28) that *fit* the pattern substitute for the one check
that would have distinguished them — waiting for equal sessions. Two pieces of
weak evidence pointing the same way is not the same as one piece of decisive
evidence, and it took me an extra cycle to go get the decisive one.

**scipy 1.18: I measured the blast radius and it is 2 classes, not 22 modules.**

I had written that "22 galpy modules reference scipy splines, so `interpRZPotential`,
`interpSphericalPotential`, the Staeckel/Adiabatic action-angle grids, `sphericaldf`,
`kingdf`, `qdf`, `diskdf` and the whole stream family all become unpicklable" and
flagged it as the most user-visible thing found this fortnight. That came from a
**grep for spline references**, not from what objects actually hold. Installed
matplotlib+astropy into the scipy-1.18 venv, pointed it at the worktree, and ran it:

```
default-constructible potentials:  62 picklable, 2 UNPICKLABLE
interpRZPotential ......... OK      actionAngleStaeckelGrid ... OK
interpSphericalPotential .. OK      actionAngleAdiabaticGrid .. OK
quasiisothermaldf ......... OK      dehnendf .................. OK
isotropicHernquistdf ...... OK      Orbit(integrated) ......... OK
```

**Real radius: `MultipoleExpansionPotential` and `DiskMultipoleExpansionPotential`.**
That is it. And the reason is now principled rather than empirical:

```
BPoly / PPoly / CubicSpline    _xp=True   pickle FAIL
make_interp_spline             _xp=True   pickle OK
InterpolatedUnivariateSpline   _xp=False  pickle OK
UnivariateSpline               _xp=False  pickle OK
RectBivariateSpline            _xp=False  pickle OK
```

Only `_interpolate.py`'s polynomial classes gained `_xp`; the FITPACK-backed splines
galpy leans on did not. The multipoles are the ones storing `BPoly`.

So this is **not** a repo-wide compat break and needs **no global `copyreg` reducer** —
the decision I escalated to Jo has largely dissolved. #1230's workaround (exclude
those two, cause named in code) turns out to be exactly proportionate. What is left
is small and optional: report upstream, and optionally a contained two-class
`__getstate__`/`__setstate__` rebuilding the `BPoly` through its public constructor.

Same error as the RazorThin scopings and the "every potential pickles" claim, and
now with a clean statement of it: **a grep tells you where a name appears, not what
the program does.** I keep reaching for static reads when a five-line runtime probe
is available, and the static read is systematically pessimistic — it cannot tell
"references" from "stores", or "stores" from "stores in a way that matters".

**And the fix works — with a caveat about what "works" covers.** Under scipy 1.18,
reconstructing `BPoly` through its public constructor makes both affected classes
round-trip bit-identically:

```
MultipoleExpansionPotential:     round-trips, bit-identical=True
DiskMultipoleExpansionPotential: round-trips, bit-identical=True
```

The caveat, stated up front this time rather than discovered later: I verified it by
registering a **global `copyreg` reducer** — the mechanism I explicitly said galpy
should not ship. So what is proven is that *the reconstruction preserves values
exactly*. The per-class `__getstate__`/`__setstate__` packaging is different code
walking the stored `BPoly` lists, and it is **not** verified. Writing that down as
part of the result is the whole correction I have been making all fortnight: the
measurement is real, the claim must not be wider than it.

Net effect on the escalation: the global-mechanism question I raised for Jo has
dissolved. Radius is two classes, fix is contained, #1230's exclusion is
proportionate. What is left is optional and small.

**Traced matrix completed — and the torch-is-3.5x claim needs qualifying.**

`test_actionAngle.py --backend torch --jit`: **209 tests, 0 failures, 168s.**

| file | jax `--jit` | torch `--jit` |
|---|---|---|
| `test_potential.py`   | 1326 / 0 (338s) | 1326 / 0 (**1792s**) |
| `test_actionAngle.py` | 209 / 0 (166s)  | 209 / 0 (**168s**) |
| `test_orbit.py`       | 581 / 3 | 581 / 0 |
| `test_sphericaldf.py` | 223 / 0 | 223 / 0 |
| `test_qdf.py`         | 0 fail | — |
| `test_diskdf.py`      | 129 / 0 | — |

I wrote earlier that torch tracing costs "~3.5x the compute" and framed it as a
property of `torch.compile` — enough to say a per-push torch-traced shard "is not
affordable". On action-angle it is **168s vs 166s, i.e. parity**. So the 3.5x is
`test_potential`-specific, not a characteristic of torch tracing.

That is the same over-generalisation again, in a place I did not expect it: one
file's wall-clock became a claim about the backend. The mechanism is presumably
that `test_potential` has an enormous number of distinct small traced entry points
(1326 tests over ~60 potential classes) so front-end compile cost dominates, while
`test_actionAngle` re-enters a few big kernels. Whether a traced CI shard is
affordable therefore depends on which file, and the honest version of my earlier
recommendation is: `test_potential` traced is expensive on torch, and I have no
evidence either way about the rest.

Also worth noting for the record: `test_orbit` traced is **581/0 on torch** and
581/3 on jax, where all three jax failures are the two performance timeouts plus
the local-only sigint artifact. So torch is the *cleaner* of the two traced
backends on orbit, which is the reverse of the eager picture.
