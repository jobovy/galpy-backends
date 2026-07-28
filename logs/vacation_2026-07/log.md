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
