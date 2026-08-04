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

### Traced matrix COMPLETE — both backends, every major suite

`test_qdf.py` torch `--jit`: **56 / 0** (232s). `test_diskdf.py` torch `--jit`,
serial: **129 / 0** (685s).

| file | jax `--jit` | torch `--jit` |
|---|---|---|
| `test_potential.py`   | 1326 / 0  (338s) | 1326 / 0  (**1792s**) |
| `test_orbit.py`       | 581 / 3          | 581 / 0 |
| `test_actionAngle.py` | 209 / 0  (166s)  | 209 / 0  (168s) |
| `test_sphericaldf.py` | 223 / 0          | 223 / 0 |
| `test_diskdf.py`      | 129 / 0  (678s)  | 129 / 0  (685s) |
| `test_qdf.py`         | 0 fail           | 56 / 0   (232s) |

**Every major suite runs traced on both backends with nothing failing**, except
`test_orbit` on jax, whose 3 are two 1800s performance timeouts and one local-only
signal artifact (CI-confirmed passing). Modulo the six `-jit` ledger entries, which
suppress the RazorThin pair and the interpRZ `use_c` pair, that is the mission's
headline in one table.

It also settles the cost question properly. Wall-clock parity holds on
action-angle (168 vs 166s) and diskdf (685 vs 678s); the only place torch is
dramatically slower is `test_potential` (1792 vs 338s). So the "torch traced costs
~3.5x" line I wrote — and used to argue a torch-traced CI shard is unaffordable —
was one file generalised to a backend. The defensible version: **`test_potential`
traced is expensive on torch and everything else measured is at parity**, which
points at per-entry-point compile cost dominating where there are ~1300 small
distinct traces, not at `torch.compile` being slow.

That is now the third measurement this fortnight that came in *cheaper* than my
projection once actually run (hyp2f1, qdf sampler, RazorThin scope) and the second
where a number I had generalised turned out to be file-specific. The habit worth
keeping: when a figure is going to justify a decision, re-measure it on a second
instance before quoting it as a property.

**Ledger trajectory this fortnight** (for the summary Jo will want, rather than a
two-week trace):

```
156 -> 150 -> 130 -> 128 -> 122 -> 120 -> 109 -> 107   (merged)
                                                 -> 101  (qdf, built & waiting)
```

Current 107 on develop, split `jax 43 / jax-jit 5 / torch 58 / torch-jit 1`.
Of those, **48 are `actionAngleVerticalInverse`** (24 jax + 24 torch), which
refuses backends by construction and moves only if that class is migrated — a
scoping decision, not a bug hunt. So the reducible remainder is **59**, dropping to
**53** once the qdf sampler PR lands.

The other structural block is the **~24-entry tolerance cluster** that hinges on
one decision from Jo (torch-vs-numpy absolute-vs-relative bounds, magnitudes all
measured). Between VerticalInverse (48) and the tolerance batch (~24), roughly two
thirds of the remaining ledger is waiting on decisions rather than on work.

### Wake 73 — #1230 MERGED, #1231 opened

**#1230 merged** (`b68b3f02`): every shipped potential picklable, 64/0 on scipy
1.17 (was 61/3), numpy byte-identical over 858 probed values, 6 new tests.

The last red before merge was `all-clear`, and it was worth diagnosing rather than
blind-rerunning: its log listed **every** check as `success` and still reported "1
failing check run". It had evaluated at 14:05 while `codecov/project` was red, and
that cleared afterwards — the documented codecov-blocks-via-all-clear pattern. One
re-run, green.

**#1231 opened**: qdf sampler boundary, ledger **107 → 101**.

Worth recording what the merge audit caught on this PR, because CI could not have:
two stray `.coverage.geir.pid*` binaries swept in by `git add -A`. `.gitignore` had
`.coverage` but not the per-process `.coverage.<host>.<pid>.*` files coverage writes
under xdist. All 154 checks passed *with them committed* — binaries in the tree break
no test. Removed, and the pattern widened to `.coverage.*`.

That is two PRs running where the audit found the only real problem: #1229's
ndarray→list type change (`obs * 2` would have repeated, not scaled) and this. Both
invisible to the test suite by construction. The audit step is not ceremony.

**scipy 1.18 fix: attribute set mapped, and checking the second case mattered.**

Walked the object graph for where the `BPoly`s actually live:

```
MultipoleExpansionPotential      -> _I_inner_cos, _I_outer_cos        (axisymmetric)
                                 -> + _I_inner_sin, _I_outer_sin      (NON-axisymmetric ONLY)
DiskMultipoleExpansionPotential  -> only via `_me`, which IS a MultipoleExpansionPotential
```

So **one** `__getstate__`/`__setstate__` on `MultipoleExpansionPotential` fixes both
classes — DiskMultipole inherits it through `_me`.

The part worth flagging: the `_sin` attributes exist **only** for a non-axisymmetric
density. My first probe used the default (axisymmetric) potential and found two
attributes; the non-axi one has four. Had I written the fix from the first probe it
would have looked correct, passed the default-constructed test, and **silently
failed to restore non-axisymmetric potentials** — the values would come back wrong
rather than the pickle raising.

That is the same class of error as everything else this fortnight (evidence
narrower than the claim), but caught before it cost anything, and only because I
had started asking "what did this measurement *not* cover?" as a habit rather than
after being burned. Two probes, thirty seconds.

Implementation spec is recorded in the task. Not started: #1231 is in CI and the
rule is one PR at a time.

**What is actually left in the ledger** (101 post-qdf), by file:

```
61  test_actionAngle.py      <- 48 are actionAngleVerticalInverse
12  test_quantity.py         <- mostly the tolerance cluster + the 2 qdf sample entries qdf drops
 5  test_streamTrack.py       5  test_qdf.py         4  test_orbit.py
 4  test_interp_potential.py  3  test_streamdf.py    3  test_sphericaldf.py
 3  test_potential.py         3  test_diskdf.py      2  test_noninertial.py
 1  test_SpiralArmsPotential.py   1  test_galpypaper.py
```

Two clusters are **two thirds of the total and both are decisions, not work**:
`actionAngleVerticalInverse` (48, refuses backends by construction — moves only if
that class is migrated) and the tolerance batch (~24, one call on torch-vs-numpy
bounds). The genuinely mechanical remainder is a long tail of 2–5 per file.

Also closed **task #55** as stale: the 4 KuzminKutuzov torch entries it existed to
remove are already gone from the ledger, cleared by some earlier coords wave
without the task being updated. Worth noting the pattern — the task list has
accumulated entries whose premise expired, and I have now hit three of them
(#68's "namespace modules", #158's "zero -jit entries", #55). Checking a task's
premise against the tree before working it costs one grep and has twice saved a
wasted branch.

### Wake 75 — #1231 went red on my own fix, and the cause was my instrument

`backend torch tests/test_conversion.py tests/test_galpypaper.py ...` failed on
`test_galpypaper::test_qdf`, one of the six entries this PR un-ledgers. The
traceback was not in stdout (conftest force-exit truncates it); the junit artifact
had it:

```
AttributeError: 'torch.dtype' object has no attribute 'type'
  scipy/optimize/_optimize.py:568:  fx = _dt.type(np.asarray(fx).item())
```

`fmin_powell`'s objective returns `-self(...)` — a **torch scalar** under a forced
backend — and scipy coerces a non-scalar return via `fx.dtype.type(...)`, which
torch does not implement.

**The reason I missed it is the reason I keep missing things.** I measured, and
wrote down, "`fmin_powell` on the DF **already succeeds**", and used it to bound
this fix at two sites (later four). That measurement was *true* — on the local
**scipy 1.17.1**, which coerces silently. CI's py3.14 runner ships a newer scipy
that raises. It is the same single-version blind spot as the pickle sweep, in the
same PR family, hours after I had written that lesson up.

So the honest framing is not "I was careless three times". It is: **my local
environment is one point in a version space and I have been treating it as ground
truth.** The 2 -> 4 -> 6 site progression was not compounding sloppiness; it was
one instrument that could not see the last boundary. The remedy is a better
instrument, not more care — the scipy-1.18 venv now exists and boundary-sensitive
verification should run against it, not only the main env.

Fixed both `fmin_powell` objectives (`6894ac28`). torch: `test_galpypaper::test_qdf`
1/0, `test_qdf` sampleV 3/0. numpy `test_qdf` + `test_backend_qdf`: **106/0** once
I forced CPU — the first run's 5 "failures" were jax CUDA-init noise from my own
box, not the code.

Pushed; prior-SHA runs cancelled (5 of them — the project is runner-bound and
#1231 had already queued ~180 jobs once).

**Added `logs/vacation_2026-07/STATUS.md`** — a snapshot that is OVERWRITTEN each
time, not appended. Jo said he will not read a two-week trace but will ask for
summaries, and this log is now long enough that "read the tail" no longer answers
"where are things". STATUS.md holds the ledger trajectory, the traced matrix, what
is in flight, and the three decisions waiting on him; `log.md` stays the trace for
the *why*. Reaching for a growing file when the question is "what is true now" was
the wrong shape.

**A better discriminator for codecov reds.** #1231 shows `head 28 misses / 27
sessions` vs `base 20 / 28` — 8 extra at one fewer upload. On #1230 I read that
same shape (1 extra, 1 fewer) as "the missing shard explains it" and was **wrong**;
it was a real uncovered branch. So the session count alone is not a discriminator.

The check that actually separates the two cases is **"are the missed lines inside
this PR's diff?"** — via codecov's `report/?path=...&sha=...` endpoint, which is the
one that returns per-line data (`file_report` and `compare` do not):

```
galpy/df/quasiisothermaldf.py             misses 0  cov 100.0
galpy/potential/MultipoleExpansionPotential.py  misses 0  cov 100.0
```

Both touched files clean, so the 8 misses are in files this PR never edits —
positive evidence for the missing-session explanation rather than an inference
from a count. On #1230 the same query pinpointed `AnySphericalPotential.py:150`
immediately. Still confirming at 28/28 before merging, but the reasoning is now
grounded in *where* rather than *how many*.

### Wake 79 — #1231 MERGED, #1232 opened

**#1231 merged** (`87c5e7de`): qdf sampler boundary, ledger **107 -> 101**. Six
torch entries retired, each verified to fail without it.

Final codecov read at **28/28 sessions: head misses 20 = base 20**. The per-file
check called it early and correctly — when head showed 8 extra at 27 sessions,
both touched files reported 0 misses, so the extras were elsewhere. That is the
discriminator #1230 taught me, applied before rather than after being wrong.

**#1232 opened**: scipy 1.18 spline pickling. One `__getstate__`/`__setstate__` on
`MultipoleExpansionPotential`, bit-identical on both scipy versions including the
non-axisymmetric case.

What #1231 cost, honestly: **three rounds**, because each boundary I landed exposed
the next (DF value -> `logmaxVD` -> `fmin_powell` objective), and the last was
invisible to my local scipy. Not three careless passes — one instrument that could
not see the far edge. The remedy now exists as a tool rather than a resolution: the
scipy-1.18 venv, plus a memory note saying to use it for anything crossing into
`scipy.optimize`/`integrate`/`interpolate`.

Ledger **101**. Of which 48 are `actionAngleVerticalInverse` and ~24 the tolerance
cluster — so the reducible-by-work remainder is now roughly **29**.

## Wake 80 — 2026-07-31

**#1232** (scipy 1.18 spline pickling) in CI: 17/153 pass, 134 pending, both
codecov checks red — premature at that shard count, same self-clearing pattern
as #1230/#1231. Watcher armed for non-codecov reds.

**Verified the ledger's composition instead of inferring it.** Last wake I
published "~29 reducible by work" in STATUS.md; that number came from subtracting
two remembered cluster sizes (48 VerticalInverse + ~24 tolerance) from 101, and
the tolerance cluster's membership was stale — it had shrunk as other PRs landed,
and several of its entries live in `test_actionAngle.py`, so the two buckets
overlapped in a way subtraction cannot see. Enumerated from the tree:

```
101 = 48 VerticalInverse + 21 tolerance + 2 RazorThin-jit + 30 workable
```

So 71 of 101 are gated on one of Jo's three decisions, and **30 are work**. The
answer landed near my estimate, but the method was unsound and I would not have
known if it hadn't. STATUS.md now carries the enumerated list, not the estimate.

Also repaired STATUS.md: the previous edit had left the old 107-era split
(`torch 58`) under the new 101 headline, and had split the in-flight table with a
stray blank line. Both were mine, both from editing by string-replacement without
re-reading the result.

**Largest coherent workable cluster: the 8 torch stream entries** (streamTrack 5
+ streamdf 3) — the natural PR after #1232. Built `/var/tmp/streamtorch` off
`87c5e7de` and started an un-ledgered run of the 5 streamTrack entries to get
real failure modes rather than assuming they are the `as_numpy`-at-assertion
class (#176). Two of the 3 streamdf entries are separately explained: one is the
`parallel_map` fork deadlock (#178).

**Correction, same wake.** I reported the first stream run's zero-byte output as
the `torch`+fork stdout truncation (a real, documented failure mode I have a
memory note for). It was not. Both runs exited `RC=4` — pytest *usage* error —
because I passed `--backend-no-xfail`, a flag that does not exist. The conftest
exposes only `--backend` and `--jit`; the ledger is applied automatically and the
way to see through it is pytest's own `--runxfail`.

Worth naming the failure mode, because it is subtler than a typo: an invented
flag produced an empty result, and I reached for a remembered phenomenon that
*explained empty results* and fit the surface facts (torch, streams, no output).
The memory was correct; it just wasn't what happened. A remembered failure mode
matching the symptom is not evidence it is the cause — `RC=4` was sitting in the
return code the whole time and settles it in one look. Check the exit code before
reaching for the explanation.

**Stream cluster characterised — and my grouping of it was wrong.** I described
the 8 torch stream entries as "the largest coherent cluster" and the natural next
PR. Measured, they are not one thing:

| n | entries | cause |
|---|---|---|
| 4 | streamTrack `cov_basis`, `cov_per_call_unit_overrides`, `plot_smoke`, `plot_spread_non_cartesian` | `NotImplementedError` at `streamTrack.py:2368` — backend `cov()` supports only `basis='galcenrect'` |
| 1 | streamTrack `out_of_range_returns_nan` | numpy `fromnumeric` calling `.max(out=, axis=)` on a torch Tensor |
| 3 | streamdf trio | one is the `parallel_map` fork deadlock (#178); other two unmeasured |

The 4 are a **deliberate, self-documented deferral** — sky/cyl-frame covariance
bases via Jacobians, the follow-up parked in PR #861 (see memory
`streamtrack_todo`). They are not a boundary-coercion fix and cannot be swept up
in one; un-ledgering them means *implementing the Jacobian transform* backend-
agnostically, which is real feature work that happens to serve the differentiable
-track goal.

Two lessons, both about my own inference:
1. I grouped by **file name and backend**, which is a property of the ledger's
   formatting, not of the failures. Same file + same backend implied same cause
   to me; it was four causes across eight entries.
2. I had already written down the true cause of the 4 — `streamtrack_todo` names
   the deferred #861 sky-basis work — and did not connect it until the traceback
   said so. Memory is only useful if consulted against the specific claim.

Next PR after #1232 is therefore **not** "the stream cluster". Candidates, honest
sizes: the 1 boundary bug (small, real), or the sky-basis Jacobian feature (4
entries, substantial, and worth doing on its merits).

**Sky-basis cov: scoped properly, and one more correction.** Last wake I offered
the `out_of_range_returns_nan` boundary bug as a small standalone next PR. It is
not standalone: the failing assertion is `numpy.all(numpy.isnan(Cs[0]))` on a
torch tensor (genuinely the #176 as_numpy-at-assertion class), but four lines
later the same test calls `track.cov(tps, basis="sky")`. Fixing the casts just
advances it into the same `NotImplementedError` as the other four. **All 5
streamTrack entries are gated on the sky-basis feature.** There is no cheap
subset here.

Scoped the real work by reading it rather than estimating:

| blocker | n | why it breaks under a backend |
|---|---|---|
| `float(...)` scalarisation in `_analytical_jacobian` | 13 | severs trace and gradient |
| in-place `J[0] *= 1/_DEGTORAD` | 6 | jax arrays are immutable |
| `numpy.eye(6)[perm]` | 1 | wrong namespace/device |
| `coords.XYZ_to_lbd_jac` numpy refs | 5 | the only chain link needing migration |

The other four `coords.*_jac` helpers contain zero numpy references — pure
arithmetic, so backend-agnostic by construction. That is the good news: the
Jacobian *chain* is already portable; only the scalarising glue around it is not.

Also note the numpy branch carries the exact pattern Jo asked me to audit for:

```python
for k, tp_k in enumerate(tp_arr):
    if not in_range[k]:
        continue        # data-dependent branch + per-element Python loop
```

so the migration is a batched `xp.where` formulation, with the dead branch
guarded (`xp-where-dead-branch-guards`) so out-of-range entries don't NaN-poison
the gradient.

This is real feature work that serves the mission directly — it makes sky-frame
stream covariances differentiable, which is what Track G's showpiece needs — and
it retires 5 ledger entries. Sized at one focused PR. Not started: #1232 is still
in CI and the rule is one PR at a time.

**Correction to the sizing I published one turn earlier.** I wrote that four of
the five `coords.*_jac` helpers are "backend-agnostic by construction — zero
numpy references". That is false. Ran them under torch: all four return numpy
`ndarray` when fed tensors. Counted properly, every link in the chain is
pure-numpy:

```
galcenrect_to_galcencyl_jac    50 lines   3 numpy refs   0 xp
galcenrect_to_XYZ_jac          48        15              0
XYZ_to_lbd_jac                126         5              0
galsky_to_sky_jac             109        12              0
sky_to_customsky_jac          100        14              0
lbd_to_XYZ_jac                 76        12              0
```

The "zero references" came from `awk '/^def f/,/^def [a-z]/'` — the start line
*also* matches the end pattern, so the window was **one line long**. I counted
numpy references in a function signature and reported it as a property of the
implementation.

So the PR is ~5x the size I stated, and splits the way this project has split
this shape before (Streams PR-1 did the coords leaf as a keystone, then
consumers):

* **PR-A (keystone):** `coords.*_jac` family backend-agnostic, numpy byte-identical.
* **PR-B:** streamTrack sky-basis `cov()` consuming it — batched `xp.where` over
  `tp`, drop the 13 `float()` casts and 6 in-place row-scalings. Un-ledgers 5.

**The pattern across four corrections in this cluster is one thing:** subtraction
over remembered cluster sizes, a grep in a malformed awk window, a remembered
failure mode that merely fit the symptom, a `-k` filter I never confirmed
selected anything. Every one was a *proxy* for the measurement, and every proxy
failed silently in the direction of "looks fine". The two that held up — the
junit traceback and running the helpers under torch — were the two where I
executed the actual thing. Cheap proxies are fine for search; they are not
evidence for a claim I put in front of Jo.

**#1232 verified functionally, not just by CI colour.** 148/153 green, 4 pending,
`codecov/project` red at 26 of 28 sessions. The per-line API said 27 of my 54
added lines were missed — which would mean the fix is untested. Chased it instead
of waiting, and the chase produced two more self-inflicted measurement errors plus
one real verification:

* `import galpy.potential.MultipoleExpansionPotential as M` binds the **class**,
  not the module (galpy re-exports the class under the module's name). My
  monkeypatch `M._pack_splines = counting` therefore set an attribute on the class
  and intercepted nothing — reported as "0 invocations", i.e. "the fix never
  runs". Reached the module via `sys.modules[...]` instead.
* A nesting error: the spline attrs are lists-of-lists, so checking `v[0]` for the
  surrogate tuple looked one level too shallow and found none.

The real verification, via the flip test (`verify-burndown-flips-stash-rerun`),
run on scipy **1.18** in `venv118` against two worktrees:

```
WITHOUT fix (develop 87c5e7de):  MultipoleExpansion  -> TypeError: cannot pickle 'module'
                                 DiskMultipole nonaxi -> TypeError: cannot pickle 'module'
WITH fix    (#1232 head):        axisym  live_splines=2 packed=2 unpacked_left=0 roundtrip=True
                                 DiskMultipole nonaxi  pickles=True  roundtrip=True
```

So the fix is real, and `_pack_splines` demonstrably executes. Two incidental
facts worth keeping: an *axisymmetric* multipole populates only the `_cos` spline
attrs (`_sin` are empty), which is exactly why the PR adds a non-axisymmetric
case; and `DiskMultipoleExpansionPotential` does **not** inherit from
`MultipoleExpansionPotential` — the one-class fix reaches it by composition, which
I had asserted in the PR body and have now actually tested.

Conclusion on the codecov red: the lines are exercised, so the 27 misses are the
26-of-28 incomplete-upload artifact, same shape as #1231. Not merging on that
inference alone — waiting for sessions to complete and re-reading.

**#1232 codecov resolved itself, as the functional check predicted.** Coverage
went 86.17 -> 99.95 as sessions reached 27 and `codecov/project` cleared. 150
pass / 3 pending, zero failures. The 27 "missed" added lines were the incomplete
upload; the flip test had already established the code runs. Merging once
`all-clear` and the two slow shards (jax `test_qdf` ~45min, the 3.14 backend
build ~70min) finish.

**Started PR-A: `coords.*_jac` backend-agnostic.** Branch `feat/coords-jac-backend`
in `/var/tmp/streamtorch`, first of six helpers done
(`galcenrect_to_galcencyl_jac`, commit 3837b53e). Two shape changes:

* `numpy.zeros((6,6))` + 14 item assignments -> rows built with `xp.stack`.
  jax arrays are immutable, and stacking is also what makes a batched
  `(..., 6, 6)` fall out for free.
* `if R == 0.0: R = 1e-30` -> `xp.where(R == 0.0, 1e-30, R)` — the exact
  if->array-input class Jo asked me to audit for.

Verified on every axis rather than asserted:

```
numpy byte-identical      400/400 random states, incl. the R==0 edge
torch vs numpy            max|diff| 0.0
jax (x64) vs numpy        max|diff| 0.0
jax.jit                   traces
batched (5,6,6)           matches per-element construction
grad-vs-FD (central)      4.6e-07 -> 4.6e-09 -> 4.6e-11   (clean h^2)
torch grad vs jax grad    max|diff| 0.0
```

One trap worth recording: the **first** gradient run showed rel err *growing* as
h shrank (6.8e-04 -> 5.6e-02) and jax differing from numpy by 7.6e-06. That is
the float32 signature — jax defaults to 32-bit and I had not set
`JAX_ENABLE_X64`. As run, that test was not evidence of anything. With x64 it
converges properly and the jax/numpy difference is exactly 0. **A gradient test
that gets worse as h shrinks is a dtype bug in the harness, not a correctness
finding in the code.**

**#1232 MERGED** — `0eeae2fc`, 154/154 green. Fourth merge of the vacation
(#1229, #1230, #1231, #1232). This one carries no ledger entries: it is a
correctness fix for a dependency regression, so the count stays at 101.

Merge audit:

| check | result |
|---|---|
| if -> `xp.where` array-input class | N/A — pure Python serialization, no array code |
| duplication | no existing spline-pickling helper; `backend/interpolate.py` converts splines to coefficient *tables*, so it never stores the scipy object (which is also why interpRZ / AA grids were unaffected — verified, not assumed) |
| shadowed parent `__getstate__` | none; no ancestor in the MRO defines it |
| helper placement | **judgment call, flagged** — see below |

**Placement call I want Jo to see.** `_pack_splines`/`_unpack_splines` are
general-ish ("make a nested list of scipy piecewise-polynomials picklable") and
Jo's standing directive is to hoist general functions into utilities rather than
leave them in specific potential files. I kept them module-local in
`MultipoleExpansionPotential.py` because they have exactly **one** consumer, and
no existing utility is a natural home (`galpy/backend/interpolate.py` is about
backend-agnostic *evaluation*, not serialization; `galpy/util/` has no
serialization module). Creating a new util module for two 8-line private
functions with a single caller seemed worse than the alternative. **Trigger for
revisiting: the moment a second class needs to pickle PPoly/BPoly, hoist them.**
Happy to move them now if Jo prefers the rule applied strictly.

## Wake 84 — PR-A progress (3 of 6 jac helpers)

No PR open (#1232 merged), so the whole wake went to PR-A on
`feat/coords-jac-backend`. Three helpers migrated, each verified before commit:

| helper | numpy byte-identical | torch | jax x64 | jit | grad-vs-FD floor |
|---|---|---|---|---|---|
| `galcenrect_to_galcencyl_jac` | 400/400 (+ R=0 edge) | 0.0 | 0.0 | yes | 4.6e-11 |
| `galcenrect_to_XYZ_jac` | 600/600 (3- and 6-arg, ±Xsun) | 0.0 | 0.0 | yes | 1.4e-11 |
| `lbd_to_XYZ_jac` | 800/800 (4 call forms) | 0.0 | 0.0 | yes | 6.5e-11 |

`tests/test_coords.py` 97 passed.

The shared pattern is now settled and mechanical: rows built with `xp.stack`
instead of assigned into `numpy.zeros((6,6))` (jax immutability, and it gives
batched `(..., 6, 6)` for free), `numpy.*` -> `xp.*`, namespace via
`get_namespace(...)` + `promote_scalars`. Per-helper extras found by reading
rather than assuming:

* #1 `if R == 0.0: R = 1e-30` -> `xp.where` (the if->array-input class).
* #3 `l *= _DEGTORAD` mutated the **caller's** array in place; and
  `out[:, 0] *= _DEGTORAD` scaled a column, which becomes a broadcast
  column-vector multiply sized from the 3x3-vs-6x6 result.

Static Python branches (`len(args)`, `degree=`) stay Python-level — they are
arity/flag decisions, not data-dependent, so they neither block tracing nor need
`where`.

**Two harness traps caught this wake, both of which would have produced
meaningless green:**

1. jax defaults to **float32**. My first gradient check showed rel err *growing*
   as h shrank (6.8e-04 -> 5.6e-02) and jax-vs-numpy at 7.6e-06. With
   `JAX_ENABLE_X64=1` the difference is exactly 0.0 and FD converges cleanly.
   A gradient test that degrades as h shrinks is a dtype bug in the test.
2. For helper #2 my first objective was `sum(J**2)`, whose gradient is
   identically **zero** — cos^2+sin^2 = 1 makes it invariant. It "passed" while
   testing nothing. Re-done with a random weight matrix.

Remaining: `XYZ_to_lbd_jac` (126 lines), `galsky_to_sky_jac` (109),
`sky_to_customsky_jac` (100). Then PR-A opens; PR-B (streamTrack sky-basis cov)
consumes it and retires 5 ledger entries.

## Wake 85 — PR-A opened as #1233 (4 of 6), scope honestly cut

`XYZ_to_lbd_jac` done (the singular one) — three data-dependent guards to
`xp.where`, each over a SAFE denominator, and the `out[3:6,0:3] = -M @ Q @ P`
block assignment rewritten as a concat of `[[P, 0], [C, M]]`. 800/800
byte-identical incl. the r=0 pole, torch 6.9e-17, jax x64 0.0, jit traces,
grad-vs-FD to 6.1e-11.

**Finding that would have shipped silently under a weaker test.** The gradient
AT the exact pole is NaN, and my guards do NOT prevent it: the source is
`r = sqrt(X^2+Y^2)`, whose derivative is infinite at 0, not the `where`. I could
force it to zero by guarding the sqrt as well, but `dl/dX` genuinely does not
exist at the pole, so NaN is the honest signal — consistent with Jo's RazorThin
`R2deriv` off-plane call. The property that actually matters for batched work is
that it is **element-local**: verified a pole element in a batch of 4 leaves its
neighbours finite. Only the pole-edge gradient check surfaced this; the
well-conditioned check was clean.

**Scope cut, stated not smuggled.** The last two helpers cannot be finished
inside this PR's premise. `galsky_to_sky_jac` calls `lb_to_radec` and
`sky_to_customsky_jac` calls `custom_to_radec`/`radec_to_custom` — all numpy-only
(74 lines / 32 numpy refs / 0 xp for `lb_to_radec`; measured by running them,
which return numpy float64 whatever backend goes in). Two dishonest options were
available: duplicate their math inline inside the jac helpers (violates the
no-duplication directive Jo set), or quietly grow PR-A by ~200 lines of unrelated
leaf migration. Took neither. **#1233 ships the 4 verified helpers** — a complete
galactocentric<->heliocentric<->lbd sub-chain — and the sky-frame leaves become
PR-A2.

Consequence for PR-B worth being clear about: `cov(basis='sky')` still needs
`galsky_to_sky_jac`, so the 5 streamTrack ledger entries stay blocked until PR-A2
lands. `basis='galcencyl'` only needs a helper that is already done.

**PR-A2 scoped by reading `lb_to_radec`, and it is not what the reference count
implied.** I had sized it as "74 lines, 32 numpy refs". The real structure: the
**default path delegates to astropy** `SkyCoord`, with a pure-numpy
rotation-matrix fallback used only when astropy is absent (`_APY_COORDS`, which
is 1 here). astropy is neither traceable nor differentiable, so "migrate
lb_to_radec" cannot mean "replace numpy with xp" — the numpy path is not numpy.

Measured the two paths against each other over 200 random (l, b):

```
max |astropy - manual| = 1.554e-15 rad  (3e-10 arcsec)   byte-identical: False
```

Roundoff-level agreement. So the shape of PR-A2 is the **dual path** already
established in this project (and explicitly not a numpy island per
`no-numpy-islands-eliminate-fast`): numpy input keeps the astropy branch
untouched and byte-identical; a backend array routes to the rotation-matrix
branch migrated to `xp`. The 1.55e-15 rad discrepancy is the cost, and it is
below any tolerance in the suite.

Also noted for the migration: the fallback ends with
`ra[ra < 0.0] += 2.0 * numpy.pi` — an in-place boolean-mask assignment that jax
cannot do and that assumes an array (it would fail on a scalar). Becomes
`xp.where(ra < 0, ra + 2*pi, ra)`.

Not implemented this wake — #1233 is the open PR and the rule is one at a time.
Recorded so the next wake starts from the real design rather than re-deriving it.

## Wake 86 — #1233 real red, fixed

`backend torch tests/test_quantity.py tests/test_coords.py` went red with 3
failures NOT in the ledger: `test_lbd_to_XYZ_jac`,
`test_galcenrect_to_galcencyl_jac_on_z_axis`,
`test_XYZ_to_lbd_jac_at_celestial_pole`. All three the same
`numpy/_core/fromnumeric.py:97: TypeError` — numpy calling a reduction with
`out=/axis=` on a torch Tensor.

**This is the migration working, not a regression.** Before #1233 these helpers
returned a numpy `ndarray` even when fed torch tensors (I measured exactly that
two wakes ago, when checking whether they were "backend-agnostic by
construction"). So `numpy.fabs(jac[0,0])` in the tests worked *by accident*. Now
they correctly return a torch Tensor and the numpy-side assertions break — the
#176 as_numpy-at-assertion-site class, showing up as a direct consequence of the
source fix.

Fix: wrapped all **12** jac call sites in `as_numpy(...)`, including the ones
that happened to still pass, so the trap cannot resurface when helpers 5 and 6
land. `as_numpy` on a numpy array returns THE SAME OBJECT, so the numpy path is
structurally unchanged, not merely numerically equal.

Verified locally on all three backends before pushing: `tests/test_coords.py`
97/97 on numpy, torch and jax.

Process note: stdout was truncated again (conftest force-exit), and the run
looked like a bare `FFF` with no traceback. Read the tracebacks from
`--junitxml`, which is what my own note says to do for exactly this shard. That
worked first try this time.

**PR-A2 sized properly, and it is a wave rather than a follow-up.** Measured the
actual return types instead of reasoning from the source:

```
lb_to_radec, ARRAY input        numpy -> ndarray   torch -> ndarray   jax -> ndarray
lb_to_radec, manual path only   numpy -> ndarray   torch -> ndarray   jax -> ndarray
```

Backend type is stripped on **every** path, and there are three separate reasons,
each needing its own fix:

1. **astropy branch** (the default, `_APY_COORDS=1`) ends in
   `numpy.array([...]).T` — always numpy, by construction.
2. **manual branch** uses `numpy.array` / `numpy.dot` for the rotation matrices,
   which convert their inputs.
3. **`@scalarDecorator`** does `numpy.array([args[ii]])` on the scalar path,
   stripping the backend type before the function even runs. (Array input passes
   through it untouched — so this one only bites scalars.)

So "migrate `lb_to_radec`" is really: migrate the manual branch to `xp`, make
`scalarDecorator` backend-preserving, and add the dispatch that routes a backend
array to the manual branch even when astropy is importable. Then the same again
for `custom_to_radec` / `radec_to_custom`. That is a coords-leaf **wave**, not the
small follow-up I implied when I cut #1233's scope.

Not started — #1233 is the open PR. Recorded so the estimate is on the table
before the work rather than after it.

This is the third size estimate in this cluster that was wrong until I ran the
code (subtraction over cluster sizes; grep in a malformed awk window; "74 lines,
32 numpy refs"). The pattern is consistent enough to state as a rule: **for a
migration, the only sound size estimate comes from running the thing and looking
at what type comes out.** Reading the source undercounts, because the numpy-ness
hides in decorators, in library delegation, and in constructors.

## Wake 87 — the torch fix is confirmed green by CI

`backend torch tests/test_quantity.py tests/test_coords.py` **PASS (6m38s)** on
the fixed commit. The 3 `as_numpy` reds are gone. Confirmed by the shard itself,
not inferred from a local subset.

Process note worth keeping: I started a local reproduction of the *exact* shard
pairing (`test_quantity.py + test_coords.py` together under torch, serial —
`verify-exact-ci-shard-context` plus the `xdist-shared-state` rule for
test_quantity) because last wake I had only verified `test_coords.py` in
isolation. That was the right instinct, but CI answered first and CI is the
authoritative context. The local run is redundant here; the instinct is still
correct for cases where CI has not reached the shard yet.

Remaining on #1233: other backend shards + all-clear still running, codecov/project
red (premature, 100+ of 154 uploaded). No real reds outstanding.

**Local shard run disagreed with CI — both failures explained, neither real.**
My local reproduction of the exact shard found 2 failures that CI's identical
shard passed. Rather than "fix" them, ran them down:

1. `test_streamgapdf_sample` — already documented in the `backend_xfail.txt`
   header from an earlier session: *"a local clean-CPU regen had env-specific
   false failures, e.g. test_streamgapdf_sample, that CI passes"*. Known
   local-only artifact.
2. `test_sphericaldf_quantity_radius_is_not_coerced_on_a_backend[jax]` — fails
   with `TypeError: unsupported operand type(s) for /: 'jaxlib._jax.ArrayImpl'
   and 'Tensor'`. Not the jax-GPU default (still fails with
   `JAX_PLATFORMS=cpu`). The cause is structural:

```python
_BACKENDS = [b for b in ("jax","torch") if importlib.util.find_spec(b) is not None]
```

The test parametrizes over **installed** backends, independently of the forced
`--backend`. My dev box has both installed, so under `--backend torch` the
`[jax]` case builds the DF's `_psi0` as a torch Tensor and then feeds it a jax
array. `backend-tests.yml` installs jax only when `matrix.BACKEND == 'jax'` and
torch only when `matrix.BACKEND == 'torch'`, so on any CI shard `_BACKENDS` has
exactly one entry and this case **cannot occur**.

**That is a structural blind spot, not a flake:** no CI job ever exercises
jax-array-meets-torch-state, because no CI job has both frameworks installed. The
crash it exposes is real (mixing frameworks is unsupported and fails with a raw
TypeError rather than a clear error), but it is out of scope for #1233 and not a
regression from it. Worth a deliberate decision later: either make the mixed case
raise something informative, or assert single-framework use at the boundary.

Neither failure blocks #1233. The authoritative signal remains CI's own shard,
which passed.

## Wake 88 — codecov gap was real, and chasing it found a pre-existing bug

The codecov per-line discriminator on #1233 showed 71 of 212 added lines missed
at 26 sessions. Instead of assuming the incomplete-upload pattern again,
attributed the misses to functions:

```
33 XYZ_to_lbd_jac | 20 lbd_to_XYZ_jac | 10 galcenrect_to_XYZ_jac | 8 galcenrect_to_galcencyl_jac
```

`galcenrect_to_XYZ_jac` appears **0 times** in `tests/test_coords.py` — it had no
test at all. So its misses were a genuine gap, not an upload artifact, and per
`no-hiding-uncovered-code` the fix is a test, not a wait.

**Writing that test found a real pre-existing defect.** The FD check failed on
all three backends. First reaction was "my tolerance is too tight" — I loosened
the step to per-component relative, and the disagreement did **not** move
(1.469e-06 either way). A discrepancy invariant to h is not truncation error.
Located it:

```
_extra_rot=True  (the DEFAULT)   max|J - J_fd| = 1.469e-06
_extra_rot=False                 max|J - J_fd| = 1.902e-10
```

`galcenrect_to_XYZ_jac` differentiates the **`_extra_rot=False`** transform, while
`galcenrect_to_XYZ` **defaults to `_extra_rot=True`**. The FD had non-zero
couplings exactly where the analytic Jacobian has structural zeros — the
signature of an unmodelled rotation.

**This predates the migration.** My change restructured the formula
byte-identically (600/600 random states), so the mismatch is in galpy already.
It matters because `streamTrack._analytical_jacobian` calls
`coords.galcenrect_to_XYZ(x, y, z, Xsun=ro, Zsun=zo)` with the default and pairs
it with this Jacobian — so sky-frame covariance transforms inherit a ~1.5e-6
inconsistency. Small, but wrong, and exactly the kind of thing a "does it run"
test never catches.

Handled honestly rather than blessed: the test pins the contract the function
actually implements (`_extra_rot=False`, matching to 1.9e-10 — a genuinely tight
bound) and names the discrepancy in a comment and the commit message. **Not fixed
in #1233**: changing the formula would break the byte-identical numpy guarantee
this PR rests on, and the right fix (align the Jacobian with the default, or make
the caller pass `_extra_rot=False`) is Jo's call.

`tests/test_coords.py`: 98/98 on numpy, torch and jax. Pushed as cb0f6511.

**Closed the coverage question by tracing, not by waiting.** Rather than assume
the remaining 61 misses were the same incomplete-upload artifact, ran
`trace.Trace` over exactly the calls `test_coords.py` makes and intersected with
codecov's missed set:

```
codecov-missed lines the tests DO execute: 70/71
not executed:                              [2114]
```

and 2114 is the `def galcenrect_to_galcencyl_jac(...)` statement itself, executed
at import before the trace starts — covered in reality. So **all 71 are
exercised**: the 10 in `galcenrect_to_XYZ_jac` by the test added this wake, the
other 61 already. The codecov misses were purely the 26-of-28 upload.

Worth separating the two cases, because they looked identical in the codecov
output and were not:
* `galcenrect_to_XYZ_jac` — **genuinely uncovered**, because the function had no
  test whatsoever. Waiting for sessions would never have fixed it, and it hid a
  real defect (the `_extra_rot` mismatch).
* the other three — upload artifact, would have cleared on its own.

The discriminator that separates them is not the session count; it is "does any
test actually call this code". Tracing answers that in seconds and does not
depend on codecov settling.

**The `_extra_rot` mismatch now has an exact, verified correction.** `_extra_rot`
applies `galcen_extra_rot`, a *constant* orthogonal matrix (`max|R-I| = 1.47e-06`,
`det = 1`), and `galcenrect_to_XYZ` rotates its **output** by `R.T`. So:

```
J_true = blockdiag(R.T, R.T) @ J_analytic
```

Checked it the way that distinguishes a real residual from FD noise — sweep the
step and watch the direction:

```
h_scale 1e-4 -> 9.609e-12      h_scale 1e-6 -> 2.244e-09
h_scale 1e-5 -> 9.917e-11      h_scale 1e-7 -> 1.603e-08
```

The residual GROWS as h shrinks: roundoff-dominated, so the identity is exact
(best estimate ~1e-11). Had I looked at only the 1e-6 row I would have reported a
2e-9 "discrepancy" that does not exist.

Posted both the finding and the correction to #1233 as comments. **Not
implemented**: applying it changes `galcenrect_to_XYZ_jac`'s numpy output by
~1.5e-6, which breaks the byte-identity guarantee this PR rests on. The
alternative (make callers pass `_extra_rot=False`) keeps byte-identity but leaves
the pairing easy to get wrong — it already was wrong. Recommended option 1 to Jo
and left the decision.

**Quantified the `_extra_rot` impact so the priority is honest.** Propagated a
realistic stream covariance (30 pc / 2 km/s, random correlations) through both
Jacobians:

```
max relative diff in covariance : 2.1e-06
relative diff in 1-sigma widths : 1.1e-06
position sigma                  : 0.029967 kpc either way
on-sky at D = 20 kpc            : 1.4e-04 arcsec
```

**Real but negligible** — 0.1 mas against stream widths measured in arcminutes.
So it is a correctness fix, not an urgent one, and nothing published is
meaningfully wrong. Worth saying plainly because "the Jacobian is wrong" reads as
alarming without the magnitude; the number turns it into a routine cleanup with a
known exact form. Posted to #1233.

**New traced-matrix row: `test_coords.py` 98/98 on both backends, 0 fail,
0 skip** (jax 17s, torch 7s). Measured after #1233's four migrations plus the new
`galcenrect_to_XYZ_jac` test.

Worth noting what makes this row different from the others in the matrix: it has
**zero ledger entries and zero slow-skips**. Every other traced-green file in
STATUS.md is green *with suppressions* — `test_potential` and `test_orbit` carry
`-jit` xfails, `test_orbit` has two 1800s timeouts. `test_coords` is green with
nothing hidden, which is the actual target state for the whole suite and a useful
reference point for what "traced-green" should mean.

It is also 20-100x faster than the other files, so it is cheap to keep in a
per-push traced run rather than a nightly.

**Workable count corrected 30 -> 28: the qdf unbound pair is design-permanent.**
Ran `test_call_diffinoutputs` un-ledgered rather than trusting my own "by-design"
label:

```
AssertionError: Test orbit in qdf that is supposed to be unbound is not
```

and then found the mechanism instead of stopping at the label. Both
`UnboundError` raise sites in `actionAngleSpherical.py` (898, 1273) fire from
**scipy solver failure modes** — `except RuntimeError` from `brentq`, and an
`rtry > 100.0` escape hatch. The backend path replaced scipy's solver with a
fixed-iteration bisection: it cannot fail, and has no escape, so it returns a
finite (wrong) answer rather than raising.

Note the run was **eager** torch, so "you can't raise under a trace" is not the
explanation for what I observed — I nearly asserted that and it would have been
wrong. Eagerly a check *could* be added; it is under `--jit` that a
data-dependent raise becomes impossible. So the design constraint is real but
applies one level further out than the failing test shows.

Reclassified: 73 of 101 are decisions or design-permanent, 28 are work.

## Merge audit for #1233 (done ahead of green, so the merge is immediate)

| check | finding |
|---|---|
| **if -> `xp.where` array-input class** | This is what the PR *is*. Four data-dependent guards converted (`R==0`, `r==0`, `D*cb==0`, `K*D==0`), each dividing by a **safe** denominator first so the dead branch cannot NaN-poison the gradient. Static branches (`len(args)`, `degree=`) deliberately left as Python — arity/flag, not data. |
| **new module-level helpers** | **None.** The only added `def` is a local closure `_degree_scale` inside `XYZ_to_lbd_jac`; the other diff hit is `galcenrect_to_galcencyl_jac`'s signature gaining `*, xp=None`. Nothing general-ish was parked in a specific module. |
| **duplication** | One borderline case, **considered and rejected**: `lbd_to_XYZ_jac` scales *columns* 0,1 by `_DEGTORAD` (inputs in degrees) and `XYZ_to_lbd_jac` scales *rows* 0,1 by `1/_DEGTORAD` (outputs in degrees). Same shape (~7 lines each: build a scale vector, broadcast-multiply), different axis and inverse factor. A shared helper would need to parameterize matrix, reference array, factor, axis and length — more indirection than the lines it saves, and it would obscure that one is an input convention and the other an output convention. Left separate; recording the reasoning so it is a decision, not an oversight. |
| **shadowed/duplicated behaviour** | `as_numpy` wrapped at all 12 test call sites, including ones that still passed, so the trap cannot resurface as helpers 5-6 land. |
| **byte-identity** | Verified per helper against the pre-edit implementation: 400/600/800/800 random states, plus the `R=0` and `r=0` pole edges. |

Diff is 349 lines in `coords.py` (net +129) and 80 in `test_coords.py`.

## CORRECTION: the codecov per-line "discriminator" was bogus

I have been enumerating "missed lines" from the codecov v2 `report/?path=` endpoint as

```python
miss = set(l[0] for l in file['line_coverage'] if l[1] == 0)
```

and intersecting that with the PR's added lines. On #1233 that returned 71
"missed" lines. The same query's **file totals** say:

```
galpy/util/coords.py   coverage=100.0  lines=971  hits=971  misses=0
```

and the distribution of the second element is `Counter({0: 971})` — **every**
entry is 0, in a fully covered file. So `l[1] == 0` does not mean "missed"; the
parse was enumerating every executable line in the file.

**This invalidates the method, not just this reading.** I used the same parse on
#1230 ("1 uncovered line"), #1231, #1232 ("27 missed") and here, and described it
to Jo as *the* discriminator for whether a codecov red is real. It never
distinguished anything. That the conclusions came out right each time was luck
plus the independent checks I happened to also run.

**What actually worked, and should be the method going forward:**
1. **Per-file `totals`** from the same endpoint — `coverage / lines / hits /
   misses` are real. `misses=0` is the answer I wanted all along.
2. **Local `trace.Trace`** over the tests — definitive, offline, and it is what
   found that `galcenrect_to_XYZ_jac` had no test at all (that finding stands
   independently: 0 references in `test_coords.py`, verifiable by grep, and it
   led to the `_extra_rot` bug).
3. The `codecov/project` / `codecov/patch` check states — the actual merge gate.

The uncomfortable part is that #1230's "real uncovered line" conclusion was also
derived this way. It happened to be true — I confirmed it by adding a test that
changed the check from red to green — but the reasoning I gave for it was wrong.

## #1233 MERGED — 618fc6c9, 154/154 green

Fifth merge of the vacation (#1229, #1230, #1231, #1232, #1233). Like #1232 it
carries **zero ledger entries**: it is the coords keystone, and the ledger stays
at 101.

What landed: 4 of 6 `*_jac` helpers backend-agnostic (rows stacked, four
data-dependent guards -> `xp.where` over safe denominators), byte-identical numpy
over 400/600/800/800 random states plus both singular edges, exact on torch and
jax x64, jit-traceable, grad-vs-FD converging to ~1e-11. Plus a real test for
`galcenrect_to_XYZ_jac`, which had none.

Two things found along the way that are NOT fixed here and are recorded as
decisions:
* the `_extra_rot` Jacobian mismatch — exact correction known
  (`blockdiag(R.T,R.T) @ J`, ~1e-11), impact measured (1.4e-4 arcsec at 20 kpc);
  fixing it breaks this PR's byte-identity guarantee.
* my codecov per-line parse was bogus — see the correction above; the method is
  now per-file `totals` plus local `trace.Trace`.

**Next: PR-A2**, the sky-frame coords leaves (`lb_to_radec`, `custom_to_radec`,
`radec_to_custom`) plus the last 2 `*_jac` helpers. Sized by measurement, not
reading: backend type is stripped on every path for three independent reasons
(astropy delegation, `numpy.array`/`numpy.dot` in the fallback, `@scalarDecorator`
re-wrapping scalars). That unblocks PR-B, which retires the 5 streamTrack entries
and is the first ledger movement since #1231.

## PR-A2: the obvious sequencing is wrong — measured, not guessed

Started PR-A2 with what looked like the natural first step: make
`scalarDecorator` backend-preserving, since it is the layer that converts inputs
via `numpy.array([arg])` and strips the framework before the wrapped function
even runs. Verified first that a uniform `xp` formulation is byte-identical on
numpy for float / float64 / 0-d ndarray / int64, so no data-guard was needed, and
made the change.

**It regressed both backends.**

```
                numpy    torch    jax
before          98/98    98/98    98/98
after change    98/98    97/98    94/98
```

New failures: `test_vrpmllpmbb_to_vxvyvz` (torch); `test_pmrapmdec_to_pmllpmbb`,
`test_sky_to_customsky_jac`, `test_pmrapmdec_to_custom_againstlb`,
`test_custom_to_pmrapmdec_againstlb` (jax). Cause is structural, and counting it
made it obvious:

```
functions wrapped by @scalarDecorator : 18
of those, backend-aware               :  0
```

The decorator is a **shared choke point for 18 coords functions, none of which
are backend-aware**. Making it preserve the framework stops it from protecting
them: backend arrays start flowing into numpy-only bodies. The conversion I was
treating as a bug is currently load-bearing.

Reverted; back to 98/98 on all three.

**Revised plan for PR-A2 — opt-in, not big-bang.** Migrating all 18 at once is
too large for one reviewable PR, and leaving the decorator alone blocks the sky
path. The incremental route is an **opt-in marker**: the decorator preserves the
framework only for functions explicitly marked backend-aware, and keeps today's
numpy conversion for everything else. Then functions migrate one at a time,
each flipping its own marker, with no flag day.

Worth noting how cheap this was to discover: one edit, one test run, three
numbers. I could equally have written the whole sky-leaf migration on top of a
sequencing that was wrong from the first commit.

## PR-A2 in progress (NOT green) — opt-in marker + lb_to_radec

Branch `feat/coords-sky-leaves` off the merged develop, commit 7dcba9f1. No PR
open: this is honestly mid-migration, and I am recording the measured state
rather than a claim of completion.

**Landed on the branch:**
* `backendNative` marker. `scalarDecorator` keeps promoting scalars through
  numpy by default — that conversion is load-bearing for the 17 wrapped
  functions that are still numpy-only — and promotes through the resolved
  namespace only for functions that opt in. Verified byte-identical on numpy for
  float / float64 / 0-d ndarray / int64, and verified it is a **no-op** until
  something is marked (98/98 on all three backends with the marker in but
  unused). Placed next to `scalarDecorator`, its only consumer; checked
  `galpy.backend` first and found no existing per-function marker
  (`is_backend_compatible` is for galpy *objects*).
* `lb_to_radec` dual path: numpy keeps astropy (untouched, byte-identical); a
  backend array takes the rotation-matrix branch, now `xp`-native, with
  `ra[ra<0] += 2pi` -> `xp.where`.

**Measured:**

```
                 numpy    torch    jax
after            98/98    95/98    92/98
backend type     ndarray  Tensor   ArrayImpl     <- preserved, array AND scalar
```

**One bug of mine, found and fixed mid-wake:** I resolved the namespace but did
not coerce the inputs onto it, so under a *forced* backend `xp` was torch while
`l`/`b` were still numpy -> `cos(): argument 'input' must be Tensor`. That is
the documented pattern (`coerce-under-forced-not-dataguard`) and I skipped the
second half of it. Fixed with `promote_scalars`; torch went 93 -> 95.

**Remaining, characterised not guessed:**
* torch x3 — numpy reductions on a Tensor at *test assertion* sites, the #176
  `as_numpy` class, same as #1233.
* jax x6 — includes an in-place item assignment still present in a consumer or
  test (`x[idx] = y` is not available on jax). Not in the function I migrated.

Both are the boundary moving outward, which is what a leaf migration does. Next
wake: fix those, then the last two `*_jac` helpers, then open the PR.

## PR-A2 step 1 GREEN — 98/98 on all three backends

Commit 33486d20 on `feat/coords-sky-leaves`.

```
tests/test_coords.py     numpy 98/98    torch 98/98    jax 98/98
numpy byte-identity      1200/1200 scalar-form checks across 6 functions,
                         worst diff 0.0; plus the ARRAY form of lb_to_radec and
                         radec_to_lb, which is what exercises degreeDecorator's
                         column scaling
```

Four changes, sequenced so no step regressed anything:
1. `backendNative` marker — opt-in, verified a no-op until used.
2. `degreeDecorator`: `out[:, i] *= 180/pi` -> multiply by a scale row.
3. `pmllpmbb_to_pmrapmdec`: `dec[dec == dec_ngp] += 1e-16` -> `xp.where`.
4. `lb_to_radec`: numpy keeps astropy; backend arrays take the xp-native
   rotation-matrix branch.

**I made the same mistake twice in one wake, in two different places.** Both
times I resolved the namespace and then failed to reconcile it with what the
data actually was:

* In `lb_to_radec` — resolved `xp` but did not coerce `l`/`b`, so under a forced
  backend `xp` was torch while the inputs were numpy. `promote_scalars` fixed it.
* In `degreeDecorator` — used `get_namespace(out)` to build the scale, but this
  decorator also wraps 17 functions that still return **numpy** even under a
  forced backend. Multiplying their ndarray by a torch scale took torch from
  95/98 to **62/98**. Fixed by dispatching on `is_backend_array(out)`.

The rule that covers both, which I clearly do not have internalised: *resolving
a namespace tells you what the caller wants, not what the array is.* Under a
forced backend those differ, and every migration boundary has to reconcile them
explicitly — by coercing the input, or by dispatching on the actual type.

Next: the last two `*_jac` helpers (`galsky_to_sky_jac`, `sky_to_customsky_jac`)
now that `lb_to_radec` preserves the framework, then open the PR.

## PR-A2 complete locally: all six *_jac helpers backend-agnostic

Commits 33486d20 / d788b3ff / d742c6c7 on `feat/coords-sky-leaves`.

```
tests/test_coords.py        numpy 98/98   torch 98/98   jax 98/98
numpy byte-identity         1200/1200 (decorators + lb_to_radec family)
                            600/600   (galsky_to_sky_jac)
                            1000/1000 (custom-sky family)
torch / jax vs numpy        8.9e-16
jax.jit                     traces
grad-vs-FD                  7.4e-07 -> 1.1e-10
```

Shape of the finished migration: three shared pieces (`backendNative` opt-in
marker, `scalarDecorator`, `degreeDecorator`), three leaves (`lb_to_radec`,
`radec_to_custom`, `custom_to_radec`), two jac helpers, 30 test call sites cast.

**The honest headline of this stretch is that I hit one bug class three times.**
`xp` resolved from a *forced* backend while the data was still numpy —
in `lb_to_radec` (`xp.cos` on a numpy `b`), in `degreeDecorator` (a torch scale
against 17 functions that still return ndarray; torch fell 95 -> 62), and in
`pmrapmdec_to_custom` (`xp.where` on a numpy `dec`). Each time I found it from a
traceback rather than applying the rule up front.

The invariant, stated so it is actionable rather than a moral:
**`get_namespace()` reports the caller's intent, not the array's type.** Under a
forced backend they differ. Every boundary must therefore either
(a) coerce the data onto the resolved namespace (`promote_scalars`) when the
function IS backend-native, or (b) dispatch on `is_backend_array(x)` when it is
shared with functions that are not. Choosing (a) where (b) was needed is exactly
the degreeDecorator regression.

One new failure mode, caught cleanly: a scripted `s.replace()` matched the same
line in a *second*, unmigrated function and broke **numpy** too. All-three-
backends-failing is the signature of shared-behaviour damage rather than a
backend bug, which pointed straight at it. Now verifying match counts before
applying pattern edits — did that for the remaining three edits.

Next: push, open the PR. It unblocks PR-B (streamTrack sky-basis cov), which is
the first ledger movement since #1231.

**#1234 opened.** Pre-push check: ran the exact CI shard pairing
(`test_quantity` + `test_coords`, torch, serial) — 342 tests, 2 failures, both
the known local-only pair (`test_streamgapdf_sample`, documented in the ledger
header; and the `[jax]` param that **cannot exist** on a CI shard because
`backend-tests.yml` installs one framework per shard). `test_coords.py`
contributed zero. That check is worth its cost: the previous PR's only real red
appeared in exactly this pairing.

## #1234 REAL RED — the migration's blast radius exceeds the PR

`backend torch tests/test_orbit.py tests/test_orbits.py`: **9 FAILED**, all
`test_integrate_{2d,3d}_diff{ro,zo,vo,solarmotion,allsolarparams}`. Reproduced
locally in one run. Cause understood and **not a defect**:

```
tests/test_orbits.py:1218   numpy.fabs(...) / numpy.all(...) on a torch Tensor
```

`Orbit.ra()`, `.dec()`, `.ll()` etc. route through `lb_to_radec`/`radec_to_lb`.
Before this PR those silently returned **numpy even under a forced backend**;
now they correctly preserve the framework. That is the intended direction — an
Orbit's sky coordinates being differentiable is the point — but `test_orbits.py`
was written against the old behaviour.

**Scale, counted before touching anything** (I am not repeating the blind
`s.replace()` mistake):

```
test_integrate_3d_diffro            54 sky-accessor calls
test_integrate_3d_diffzo            48
test_integrate_3d_diffvo            42
test_integrate_3d_diffsolarmotion   36
test_integrate_3d_diffallsolarparams 30
test_integrate_2d_diffro            24
test_integrate_2d_diffvo            18
test_integrate_2d_diffsolarmotion   12
test_integrate_2d_diffallsolarparams 6
                                   ---
                                   270 call sites (~135 assertions)
```

So a coords-leaf PR now requires ~135 mechanical casts in `test_orbits.py`, a
file it otherwise does not touch. **This is the fourth time this migration has
been bigger than my estimate of it**, and the pattern is consistent: each leaf I
make framework-preserving pushes the numpy boundary one layer outward, and the
next layer's tests were written against the silent downgrade.

I stopped rather than mass-editing 135 assertions at the end of a long stretch —
that is exactly the condition under which I made the `s.replace()` error earlier.
Next wake, with the count known: wrap the assertion expressions, verify on all
three backends plus the exact shard, then re-push.

Worth flagging for Jo: an alternative to consider is *not* marking `lb_to_radec`
backendNative in this PR, which would keep Orbit's accessors numpy and shrink
this to a coords-only change — at the cost of leaving streamTrack's sky-basis
cov still blocked. I lean toward doing the test work, since the accessor
behaviour change is the mission, but it is a legitimate fork.

## #1234 red fixed — and the 270-site estimate was wrong (too big)

Before editing anything I probed which numpy op actually fails on a Tensor:

```
numpy.fabs(Tensor)        OK  -> Tensor   (but emits a DeprecationWarning)
Tensor < 1e-7             OK  -> Tensor
numpy.all(bool Tensor)    RAISES TypeError
```

So the 270 sky-accessor call sites were never the unit of work; the unit is the
**assertion**, and there are 100 of them across the 9 tests. My estimate was 2.7x
too big because I counted the wrong thing — the same error shape as the earlier
over-estimates, just in the other direction.

Placement matters and the probe decided it: `numpy.fabs(Tensor)` *works* but
warns, and the coverage shard runs `-W error::Deprecation`
(`test-backend-W-error-deprecation`), so casting only around `numpy.all` would
have passed this shard and failed the coverage one later. Cast goes **inside**
`numpy.fabs`.

Applied with a paren-matching transform scoped to the 9 target function bodies
(not a blind `s.replace` — that is the mistake from earlier today), 100 sites,
count reported and checked.

```
the 9 tests    numpy 9/9    torch 9/9
```

Full `tests/test_orbits.py` on torch running now before re-push.

**Fix pushed (553e9014).** `tests/test_orbits.py` **115/115 on torch** — full
file, not just the 9 targeted tests, so the 100 casts did not break anything
adjacent. The 9 also pass on numpy.

Estimate history for this one red, worth keeping because it is the pattern:
270 (counted accessor calls) -> 100 (counted assertions, the real unit). Three
estimates in this migration have now been wrong — twice too small, once too
large. The cause is never optimism or pessimism, it is counting a **proxy**
(reference counts, call sites, remembered cluster sizes) instead of running the
thing that actually fails. Each probe that settled it cost seconds.

**#1234 second red: `test_orbit.py::test_SkyCoord`** — same class (`ndarray -
Tensor` where `o.ra(times)` is now a backend array), one site, fixed.

**And I made the blind-replace mistake again, in the same wake I claimed to have
learned it.** I verified the match count for the assertion patch (count 1, good)
and then inserted the `as_numpy` import with

```python
s.replace("from galpy import potential", "...\nfrom galpy.backend import as_numpy", 1)
```

without checking that string's count. In `test_orbit.py` it is **not** a
module-level import — it appears inside function bodies as a local import, so
mine landed inside two functions and the module-scope name was still undefined
(`NameError`). Removed both, put one import at module level.

The lesson is not "check match counts" — I did that, for the edit I was thinking
about. It is that **the check has to cover every pattern edit in the change, not
the interesting one.** The import insertion felt like boilerplate and got no
scrutiny; that is precisely where it bit.

Full `test_orbit.py` on numpy and torch running before re-push.

**Local torch `test_orbit.py` check: TIMED OUT, not failed** — hit my own 2900s
budget with no junit written. Stating that precisely because "NO JUNIT" is easy
to misread as a failure; it is an absence of evidence, not evidence of a problem.

Operational reason, worth keeping: the CI shard runs
`--splits 5 --group N` (pytest-split), so each job takes a fifth of the file.
Reproducing it as one serial full-file run is ~5x the work and blows a sane
timeout. To reproduce that shard locally, pass the same `--splits/--group`, not
the bare file.

What IS established for that file: numpy 581/581 full-file, and `test_SkyCoord`
(the only failing test) green on torch individually. CI is the authority for the
shard and is running it now.

**#1234 torch orbit shards GREEN on b7d3b446** — all five `--splits 5` groups
plus the `energy_jacobi/from_name` shard pass. The group that was red is fixed.

Note the resolution came from CI, not from me: my local check on that file timed
out *because* I ran the whole file instead of a fifth of it, so the only evidence
I could produce locally was numpy-full-file plus the single failing test on
torch. CI's split shards answered it in the time my serial run was still going.
The rule to keep is not "run more locally" but **match the shard's own
`--splits/--group`** — otherwise the local reproduction is strictly slower than
the thing it is checking.

## PR-B re-measured on the migrated branch (do not reuse the old number)

`StreamTrack._analytical_jacobian`, 91 lines:

```
float( casts    17     (my earlier estimate said 13 -- low by 4 again)
numpy. refs      2     (was the bulk; the six *_jac migrations removed them)
in-place row *=  6
per-k loop       1     `for k, tp_k in enumerate(tp_arr): if not in_range[k]: continue`
```

The shape of the remaining work is now clear and small: the numpy dependency is
gone (2 refs left), so PR-B is **strip 17 `float()` casts, make 6 row-scalings
out-of-place, and batch the per-k loop into an `xp.where`** — plus lifting the
`NotImplementedError` guard at streamTrack.py:2314/2368.

That retires the 5 streamTrack ledger entries and is the first ledger movement
since #1231.

Recording the re-measurement rather than reusing the old estimate because the old
one was taken *before* the helpers were migrated, so both its numpy count and its
float() count were describing a different tree. Estimates have a shelf life; this
one expired the moment #1234's commits landed.

## #1234: codecov/patch was REAL — the backend-branch coverage gap

`codecov/project` cleared; `codecov/patch` stayed red with all 152 other checks
green, so it was not premature. Located the line without guessing, using the
**per-file totals** (the reliable field):

```
galpy/util/coords.py   coverage=99.89   lines=978   hits=977   misses=1
```

One line. My own note names the mechanism: backend-only branches cannot be
covered because the coverage-uploading shards run numpy. The line is
`degreeDecorator`'s

```python
if is_backend_array(out):
    scale = asarray_on_device(get_namespace(out), scale, device_of(out))
```

Two instruments failed honestly on the way there and are worth recording as
*not* usable for this:
* `trace.Trace` flags import-time lines (imports, docstrings) as unexecuted,
  because tracing starts after import — its "23 uncovered" list was noise.
* local `coverage.py` returns "No data to report" (the editable-install
  interaction already in my notes).

Fix: two tests in `tests/test_backend_coords.py`, which the build shard
`tests/test_backend*.py` runs **with jax and torch installed**
(`REQUIRES_JAX/TORCH: true`) and therefore uploads coverage for. 23/23.

**Checking that assumption caught a second error.** That same shard is
deliberately **astropy-free** (`REQUIRES_ASTROPY: false`), so its numpy reference
takes the rotation branch, not `SkyCoord`. My test comment asserted "numpy
reference goes through astropy" — false on precisely the shard the test targets.
Rewrote it to state that the reference path is environment-dependent and that
the assertion is backend-vs-numpy parity, which holds either way.

`all-clear`'s red is the documented stale evaluation (it gates on all checks and
ran while codecov was red); it clears on re-run.

## Merge audit for #1234 (done ahead of green)

Diff: `coords.py` +231, `test_backend_coords.py` +46, `test_coords.py` +72,
`test_orbit.py` +10, `test_orbits.py` +473 (100 casts plus ruff rewrapping).

| check | finding |
|---|---|
| **if -> array-input class** | 4 in-place mutations converted, all verified in the diff: `l[l<0] += 2pi` and `dec[dec==dec_ngp] += 1e-16` (x2, two functions) -> `xp.where`; `out[:, i] *= 180/pi` and `out[3:5,0:2] *= _DEGTORAD` (x2) -> out-of-place scale multiplies. No masked or sliced in-place assignment remains on a migrated path. |
| **new module-level helpers** | Exactly one: `backendNative`. Placed beside `scalarDecorator`, its only consumer; `galpy.backend` checked first and has no per-function marker (`is_backend_compatible` is for galpy *objects*). |
| **duplication** | The degree-scaling blocks in `lbd_to_XYZ_jac` (columns, `_DEGTORAD`) and `XYZ_to_lbd_jac` (rows, `1/_DEGTORAD`) remain separate — considered and rejected in #1233's audit for the same reason (different axis, inverse factor; a shared helper would need 5 parameters to save ~7 lines twice). |
| **shared-decorator safety** | Two `is_backend_array` dispatch sites, both deliberate: `degreeDecorator` and `pmrapmdec_to_custom` are shared with functions that are still numpy-only, so they must branch on what the array *is* rather than on the forced namespace. This is the bug I hit three times; the audit confirms both sites now have it. |
| **byte-identity** | numpy verified against pre-change: 1200/1200 (decorators + lb_to_radec family), 600/600 (galsky_to_sky_jac), 1000/1000 (custom-sky family). |

## #1234 MERGED — b440b5d2, 154/154

Sixth merge of the vacation. `coords.py` finished at **100.0% coverage
(978/978, 0 misses)** — the two backend-branch tests closed the one real gap.

All six `*_jac` helpers and the three sky-frame leaves are now backend-agnostic,
numpy byte-identical, exact on torch/jax, jit-traceable, with h-convergent
gradients. Ledger unchanged at 101: this was a keystone.

**What this PR cost, honestly.** The coords work itself was right the first time
and stayed verified throughout. Everything after was the blast radius of one
intended behaviour change — `Orbit.ra()/.dec()/.ll()` no longer silently
downgrading to numpy under a forced backend:

* 9 reds in `test_orbits.py` -> 100 assertion casts
* 1 red in `test_orbit.py` -> 1 cast (+ an import I placed inside two functions)
* 1 real `codecov/patch` gap -> 2 backend-branch tests

Four separate CI rounds. Each wave was legitimate, but I only found each one
*after* CI ran a shard I had not checked. The verification I did was always
scoped to what I had just edited, never to what depended on it. That is the
single correctable thing from this PR: **after a migration changes a return
type, enumerate the callers and run their tests before pushing.**

**Next: PR-B**, streamTrack sky-basis `cov()`. Scope re-measured on the merged
tree (17 `float()` casts, 2 numpy refs, 6 in-place row scalings, 1 per-k loop),
and its numpy baseline is already captured for all four bases plus the
out-of-range NaN case — recorded *before* any edit, so the migration is checked
against a reference rather than against itself. It retires the 5 streamTrack
entries: the first ledger movement since #1231.

---

## 2026-08-01 — PR-B (#1235): streamTrack sky-frame `cov()` on backends

**The guard was not the constraint.** `cov(basis=...)` raised
`NotImplementedError` for anything but `galcenrect` on a backend track, and the
error text called sky bases "a numpy-only follow-up". That framing was wrong:
the guard was a *symptom*. The real pin was **20 `float()` casts** across
`_cart_mean_at` (6) and `_analytical_jacobian` (14), which collapsed the traced
state to Python floats before any coords helper saw it. Delete those and the
sky bases work; the guard had nothing left to protect.

The single most load-bearing discovery was that `_cart_mean_at` returned
`numpy.array([float(self.x(tp)), ...])`. Every downstream link inherited numpy
from that one line, so migrating `_analytical_jacobian` first — which is what I
did — was inert until the mean was fixed. Worth remembering as a shape: when a
chain is numpy-pinned, find the *source* of the pin before migrating the links.

**Scope I declined.** `galsky_to_sky_jac` / `sky_to_customsky_jac` are
scalar-only by construction (`radec[0,0]`, `zeros_like(reshape(cosa, ()))`,
and an `_r` helper that reshapes every row entry to `()`). That is pre-existing,
not something this PR introduced. I tried full batching, hit it, and backed out
to a per-`tp` comprehension feeding **one** batched `(len,6,6)` matmul. The
matmul still had to change — the old `out[k] = J @ out[k] @ J.T` is an in-place
write, which is exactly what immutable backend arrays forbid. Batching those two
helpers is a clean follow-up PR.

**De-duplication caught mid-flight.** The rad→deg Jacobian row scaling already
existed as a closure in `coords.XYZ_to_lbd_jac`, and I was about to write it
three more times in `streamTrack`. Hoisted to `coords._scale_angle_rows` beside
`_DEGTORAD` — a coords convention belongs in coords, not in a DF file — so there
is one copy rather than four.

**Verification.** numpy byte-identical, worst `|diff| = 0.000e+00`, all four
bases plus the out-of-range NaN case, against a reference captured before any
edit. numpy 143 passed (streamTrack+coords) and 244 passed (quantity); torch
`test_streamTrack.py` **45/45**, up from 40/45.

**Ledger 101 → 96.** All five were genuine, not stale — the baseline junit at
`618fc6c9` shows 51 streamTrack-named torch tests passing and exactly 6 xfailing:
these 5 plus `test_streamTrack_multi_parallel`, which stays (fork deadlock).
First ledger movement since #1231.

### Two process errors worth keeping

1. **I read the wrong ledger for ~20 minutes.** A compound command
   `cd <worktree> && (pytest ...) & grep ... tests/backend_xfail.txt` — the `&`
   backgrounded the `cd`, so the `grep` ran in the *main* repo, whose ledger is
   older and much larger. I concluded "46 streamTrack entries, 31 of them stale"
   and started planning a large prune. The real file has 6. The tell was that
   line numbers ran to 715 in a 323-line file, and I did not check it.
   **`cd` in a compound command does not survive `&`.**

2. **A `-k` subset is not the shard context.** Running the 5 quantity entries
   with `-k` gave 7 failures ("did not return Quantity"). Running the *whole*
   file gave 0 — `test_quantity.py` carries cross-test unit state, so selecting
   a subset breaks tests that depend on earlier ones. I nearly filed this as a
   regression. This is the recurring miss recorded in
   `verify-exact-ci-shard-context`, and it cost a full re-run to disprove.

Both errors shared a cause: trusting a cheap measurement whose *context* differed
from the real one. The baseline CI junit artifact settled both questions
authoritatively and for free — it is the exact shard, already run.

### Correction to the entry above (same day, after running `test_backend_streamTrack.py`)

I claimed this PR makes sky-frame `cov()` work on backends and "keeps the
covariance differentiable through the basis rotation". Both claims were too
strong, and I only found out because the backend test file asserted the
`NotImplementedError` I had removed — so I had to open it. Had that assertion
not existed, this would have merged overclaimed.

What is actually true, measured three ways:

| mode | galcenrect / galcencyl | sky / galsky |
|---|---|---|
| forced backend, eager | backend array, exact | backend array, matches numpy (torch 1e-9, jax 6e-15) |
| unforced / mixed arrays | backend array, exact | **`TypeError: Multiple namespaces`** |
| gradient (torch, requires_grad) | **AD vs FD = 1.9e-9** | **`RuntimeError: Can't call numpy() on Tensor that requires grad`** |

One root cause for both sky failures: `coords.galcenrect_to_XYZ` and
`galcenrect_to_vxvyvz` are still plain `@scalarDecorator` without the
`@backendNative` marker from #1234, so they promote inputs through numpy.
`galcenrect_to_XYZ_jac` *is* backend-native — so the two halves of the chain
disagree, and a `requires_grad` tensor refuses the numpy hop outright.

The lesson is not "I missed a test file". It is that **every check I ran was
value-based** — byte-identity, `assert_allclose`, 45/45 — and value checks
cannot see a numpy fallback, because the fallback computes the *right values*.
The three things that would have caught it are exactly the three that did:
`is_backend_array` on the result, an unforced/mixed construction, and a
gradient. Byte-identity proves the numpy path is intact; it says nothing about
whether the backend path exists. **For a migration, at least one assertion must
be about the array's type or its gradient, not its value.**

Scope call: #1235 stays as-is with corrected claims (guard gone, 20 `float()`
casts gone, galcencyl differentiable, 5 ledger entries genuinely retired), and
the `@backendNative` migration of those two coords functions is the next PR
(task #184). Folding it in would mean a coords migration with a caller sweep —
the expensive half of #1234 — inside a PR that is already verified and green-ish.

**Caller sweep for the follow-up (done before writing any code, per the #1234
lesson).** Blast radius of marking `galcenrect_to_XYZ` / `galcenrect_to_vxvyvz`
`@backendNative`:

| function | Orbits.py | streamdf | streamTrack | coords-internal |
|---|---|---|---|---|
| `galcenrect_to_XYZ` | **0** | 3 | 2 | 2 |
| `galcenrect_to_vxvyvz` | **0** | 1 | 2 | 2 |
| `galcencyl_to_XYZ` | 13 | 2 | – | – |
| `galcencyl_to_vxvyvz` | 7 | 2 | – | – |

The Orbit ripple that made #1234 expensive lives entirely in the `galcencyl_*`
wrappers, and those are themselves plain `@scalarDecorator` — they promote their
inputs to numpy *before* delegating to `galcenrect_*`. So the inner call receives
numpy regardless, and migrating only the two `galcenrect_*` functions provably
cannot change `Orbits.py` behaviour. That makes the follow-up a contained 12-site
change rather than a repeat of #1234's four CI rounds.

### Jo's challenge: "tests involving C code are not jittable, so I don't believe the table"

Correct instinct, and it forced a real measurement. Two findings, the second of
which corrected the first.

**1. C paths bypass the trace silently.** Under `use("torch", force=True)` *and*
`jit("torch")`, `Orbit.integrate(..., method="dop853_c")` stores an `ndarray`
and returns values bit-identical to the pure-numpy C run. The C integrator ran,
on numpy, untraced, inside a "traced" session — and the test passes. So
`test_orbit.py` "581 / 0" does **not** mean 581 orbit tests traced; it means 581
tests ran with jit mode on and none failed.

**2. But the table is not uniformly hollow — I over-corrected.** Instrumenting
`traced_call` as a pytest plugin and counting `traced` vs `NOT_TRACED` boundary
calls on a 149-test slice of `test_potential.py` under `--backend jax --jit`:

```
BOUNDARY CALLS: traced=63742  not_traced=169   (99.7% traced)
```

So `test_potential` genuinely traces. I had generalised from a single
`dop853_c` probe to the whole matrix and told Jo the metric was hollow before
the aggregate came back. The accurate claim is narrower: **tracing is real on
pure-Python/backend paths and bypassed where C methods are requested**, which
concentrates in orbit integration rather than potential evaluation.

This is the same error twice in one day in opposite directions: first I trusted
a pass count that proved less than I claimed (the sky bases), then I distrusted
one that proved more. Both times the fix was the same — instrument the property
directly (`is_backend_array`, a gradient, a traced/not-traced counter) instead
of inferring it from green.

**Consequence for the metric.** Pass counts alone cannot express jit coverage.
The traced matrix needs a second column — *fraction of boundary calls actually
traced* — and C-method tests need a third category, `not-traceable`, distinct
from xfail and slow-skip, so they stop being counted as traced-green. That is
its own PR; `test_orbit.py` is being measured now to size it.

### The traced fraction is self-flattering — three numbers, not one

Measured after building the C detector. For `Orbit.integrate(method="dop853_c")`
under `use("torch", force=True)` + `jit("torch")`:

```
boundary calls : traced=0  not_traced=0   (0.0% traced)
C entries      : 1
orbit array    : ndarray
```

The C path makes **zero** `@backend_input` boundary calls. It is not counted as
`not_traced` — it is not counted *at all*. So C work contributes to neither the
numerator nor the denominator of a traced fraction, and a file can read "99.8%
traced" while most of its real work went through C. That is exactly what
happened: `test_orbit.py` slice 1/20 under `--backend jax --jit` reported
**traced=10363 / not_traced=20 (99.8%)** — a number I would have quoted as
evidence the orbit suite is traced, and which says nothing of the kind.

So the metric needs **three** numbers per file, not one ratio:

| | meaning |
|---|---|
| `traced` | boundary calls that were actually traced |
| `not_traced` | boundary calls that reached the boundary and were not traced |
| `c_entries` | work that **bypassed** the boundary into libgalpy |

Reporting `traced / (traced + not_traced)` alone hides the third column by
construction. Either report all three, or use
`traced / (traced + not_traced + c_entries)` so C dilutes the score instead of
vanishing from it.

This is the third variant of the same error today: a green-looking number that
measures something narrower than the claim it is used to support. First the sky
bases (values right, backend path fake), then the traced matrix (passes right,
tracing absent), now the traced fraction itself (ratio right, denominator
excludes the failure mode). The general form: **when a metric can only see the
cases that reached the instrument, absence of failure is not evidence of
coverage.**

**First real number, `test_orbit.py` slice 1/20 under `--backend jax --jit`:**

```
BOUNDARY CALLS:      traced=10363  not_traced=20   (99.8% traced)
C-LIBRARY ENTRIES:   162
NOT-TRACEABLE:       20 tests  (of 28 that ran; 3 skipped)
```

So **~71% of the executed tests in that slice entered C**, while the headline
traced fraction read **99.8%**. Both numbers are correct; only one of them was
ever going to be quoted. That is the whole argument for the third column in one
data point, and it is the number I would have reported as "the orbit suite is
traced" if Jo had not pushed back.

The detector is per-test (`pytest_runtest_logreport` deltas), so the
`not-traceable` list is generated, not curated — which was the precondition Jo
named for the metric being worth building.

**Jo: "will the C instrumentation only be for the tests or general?" — general.**

What I built today is test-only and hardcodes the six modules that hold a
`libgalpy` CDLL reference. That list is exactly the kind of thing that goes
stale: a seventh module loading the library would be silently invisible, and
the metric would go back to flattering itself.

The right home is `_load_extension_libs.load_libgalpy()`. Checked before
proposing it: all six consumers do `_lib, _ext_loaded = load_libgalpy()` and
then only `_lib.<symbol>`; nothing type-checks the CDLL, and the only `is None`
checks are inside the loader. So wrapping the returned object once covers every
consumer present and future, with one place to maintain instead of a list.

It is also worth more as a library feature than as test tooling. The silent
bypass is a *user-facing* trap: someone who jits their pipeline and gets
numpy+C back has no way to find out today — no error, no warning, correct
numbers, no tracing and no gradient. Exposing "did this touch C?" in
`galpy.backend`, and optionally warning when C is entered while a trace is
requested, turns the silent fallback into a visible one. That is the same class
of bug I hit three times today, promoted from something I catch by hand to
something the library reports.

Guardrails for that PR: default off, zero behaviour change with it off, numpy
byte-identity untouched, and the cost is one branch per C-symbol *resolution*
(per integration, not per force evaluation).

### #1235 MERGED (`3e002f61`) — ledger 101 → 96, seventh merge this vacation

All 154 checks green. The two `codecov` reds cleared on their own once the last
ubuntu coverage shards uploaded, exactly as predicted — chasing them earlier
would have been wasted work. Pre-merge audit was done during the wait: the four
added `if`s all branch on non-array data (`use_phys`, `self._backend` ×2,
`degree`), duplication is net-negative (rad→deg row scaling 4 prospective copies
→ 1), the helper sits beside `_DEGTORAD` in `coords`, and no in-place writes on
possibly-backend arrays survive.

### First not-traceable inventory — and `test_streamgapdf` is the punchline

Ran the C detector over three more suites under `--backend jax --jit`:

| suite | traced fraction | C entries | tests entering C |
|---|---|---|---|
| `test_actionAngle` | 99.3% (70409/529) | 707 | **128** |
| `test_kuzminkutuzov` | 92.0% (7608/666) | 4 | 3 |
| `test_streamgapdf` | **100.0%** (20/0) | **3186** | 4 |
| `test_orbit` slice 1/20 | 99.8% (10363/20) | 162 | 20 of 28 |

`test_streamgapdf` scores a **perfect 100% traced** on twenty boundary calls
while making **3186** calls into libgalpy. That is the metric failing as loudly
as it can: the fewer boundary calls a suite makes, the easier it is to score
100%, and C work makes none at all. A ratio whose denominator excludes the
failure mode rewards exactly the code that avoids the instrument.

`test_actionAngle` is the volume case — 128 tests enter C, and it is currently
listed in STATUS.md as "209 / 0 traced" on both backends. That row means far
less than it reads.

Note the inventory is cheap: three suites in the time one CI run took, all of it
while blocked. Measurement is the highest-yield thing to do against a queue.

### #184 in progress: the sky chain is FIVE links, not two

Branch `feat/coords-galcenrect-native` off the merged develop. The task assumed
two functions; measuring the chain showed five. The `*_jac` helpers turned out
NOT to be the problem — they carry no `@scalarDecorator` at all (migrated in
#1233), so they are naturally agnostic. The promoters are the
`@scalarDecorator`-wrapped links missing `@backendNative`:

| link | status | notes |
|---|---|---|
| `galcenrect_to_XYZ` | DONE `b64b2a55` | rotation is config-only -> stays a numpy constant |
| `galcenrect_to_vxvyvz` | DONE `b64b2a55` | shared `_galcen_rot` (was copy-pasted) |
| `XYZ_to_lbd` | DONE `3708fe6e` | masked in-place add + `numpy.empty` assign |
| `vxvyvz_to_vrpmllpmbb` | TODO | 25 code lines, 16 numpy refs |
| `pmllpmbb_to_pmrapmdec` | TODO | 22 code lines, 9 numpy refs |

Byte-identity held at every step: 16 cases for the first pair (scalar/array
inputs, both `_extra_rot`, scalar AND array `Xsun`/`Zsun` — the einsum's two
branches — plus the `galcencyl_*` siblings), 12 for `XYZ_to_lbd` (including the
`l<0` wrap branch and the scalar-input path). Both references captured BEFORE
any edit. The technique that makes this cheap: keep the numpy branch verbatim
under `if xp is numpy:` and add the backend branch beside it, so byte-identity
is structural rather than argued.

Caller sweep before widening scope (the #1234 lesson): `XYZ_to_lbd` has 1
`Orbits.py` caller, `vxvyvz_to_vrpmllpmbb` 0, `pmllpmbb_to_pmrapmdec` 2. Three
Orbit sites total, and #1234 already moved Orbit's sky accessors in this same
direction, so the ripple is bounded and expected.

Sky bases still raise unforced — correctly, since `vxvyvz_to_vrpmllpmbb` supplies
pmll/pmbb. The acceptance test for this PR is the pair that failed in #1235:
`cov(basis='sky')` unforced must return a backend array, and must survive a
gradient (it raised `Can't call numpy() on Tensor that requires grad`).

**Also re-made a known mistake**: sized the remaining functions with
`awk '/^def f/,/^def [a-z]/'` and got "~1 code line", the exact window bug in
[[cd-does-not-survive-ampersand]]'s sibling memory. Caught it because 1 line was
absurd, then sized properly in Python: 25 and 22 lines. A memory did not stop me
re-running the bad command; noticing the implausible number did.

### The four migrations were dead code — decorator order

Fourth link done (`vxvyvz_to_vrpmllpmbb`), and finishing it exposed that the
three before it did **nothing**.

`scalarDecorator` reads the opt-in marker off the function it wraps:
`getattr(func, "_galpy_backend_native", False)`. So `@backendNative` has to be
the **innermost** decorator, directly above `def`. I had written it outermost on
all four, which marks the `scalarDecorator` *wrapper* — the attribute lands on
an object nobody inspects, promotion through numpy continues, and the migration
is inert.

What makes this worth recording is why it survived three commits: **every
byte-identity check passed, 28 cases, `worst |diff| = 0.000e+00` — because the
backend path was never taken.** A byte-identity check is satisfied perfectly by
a migration that does nothing. I had the correct reference sitting in the same
file the whole time: `lb_to_radec` from #1234 has the marker innermost.

That is the fourth time this fortnight a green check has certified something
narrower than the claim it was being used for. The pattern is now explicit
enough to state as a rule: **byte-identity proves the numpy path is intact and
says nothing whatever about whether the backend path exists.** Only a type
assertion or a gradient can distinguish "migrated" from "unchanged".

With the order fixed, the sky chain is closed end to end:

| check | before | after |
|---|---|---|
| unforced `sky` | `TypeError: Multiple namespaces` | backend array, 5.6e-15 vs numpy |
| unforced `galsky` | `TypeError: Multiple namespaces` | backend array, 2.5e-15 vs numpy |
| gradient `sky` | `Can't call numpy() on Tensor that requires grad` | AD vs FD 6.8e-10 |
| gradient `galsky` | same | AD vs FD 2.1e-11 |

Those are exactly the two failures #1235 shipped with and named as the
follow-up's acceptance criteria. Byte-identity still holds on all 28 cases.

Remaining: `pmllpmbb_to_pmrapmdec` (customsky only), then the caller sweep of
the three `Orbits.py` sites before pushing.

### Sky chain 5/5, and the regression test took three attempts to become real

Fifth link (`pmllpmbb_to_pmrapmdec`) done — it was already half-migrated
(`xp.where` for the pole) but still built the final rotation with `numpy.array`
and divided in place. That contraction is a per-point rotation of (pmll, pmbb)
by phi, so the backend path writes the two components out. Byte-identical
across all 28 reference cases; all five links verified to return backend arrays.

**Then I tried to write the regression test, and it passed with the bug
reintroduced. Twice.** Each version was wrong for a different reason, and both
reasons are properties of the system I did not know until I measured:

1. **Forcing hides it.** Under `use(..., force=True)` the functions recover even
   with the marker misplaced — `get_namespace` reports the caller's *intent*, so
   the body takes its backend branch and `promote_scalars` re-coerces the numpy
   args that `scalarDecorator` just produced. The marker is irrelevant when
   forced.
2. **Array inputs hide it.** `scalarDecorator` only rewrites its arguments when
   they are 0-d: `if xp.asarray(args[0]).ndim == 0`. With arrays it passes them
   straight through, so the marker is never consulted. It is *only* consulted
   for scalars — which is exactly how `streamTrack` calls these, once per `tp`.

So the discriminating configuration is **unforced + scalar**, and nothing else:

| | scalar-in | array-in |
|---|---|---|
| marker correct | tuple of Tensor | Tensor |
| marker misplaced | tuple of **float64** | Tensor |

A third trap on the way: the first check asserted `is_backend_array(result)` on
a call that returns a **tuple** for scalar input, so it read False either way and
I briefly believed the fix had not worked at all. The check has to look at the
elements.

Final test: 20 cases (5 links x 2 backends x 2 shapes), and **verified by flip** —
reintroducing the bad decorator order fails 2 of 20 with "dropped back to numpy
(scalar, unforced)", restoring gives 20/20. A test that has not been watched to
fail is a guess about what it measures; this one took three tries to stop being
a guess.

Caller sweep: `test_coords.py` + `test_streamTrack.py` **143 passed** on numpy.
`test_orbit.py` still running (the 3 `Orbits.py` call sites are the #1234 ripple
risk, and it is not pushing until that is clean).

**Caller sweep clean.** `test_orbit.py` on numpy: **577 passed, 4 skipped** — so
the three `Orbits.py` call sites (`XYZ_to_lbd` x1, `pmllpmbb_to_pmrapmdec` x2)
produce no #1234-style ripple. Together with `test_coords.py` +
`test_streamTrack.py` at 143 passed, the numpy side is clean before pushing.

**Pre-push audit.** The in-place-op scan initially looked alarming — four
unguarded `l *= 180/pi` / `l[l < 0] += 2pi` sites in the file — until I checked
which function owned each:

| line | function | `@backendNative` | guard |
|---|---|---|---|
| 339 | `radec_to_lb` | no | none needed — only ever sees numpy |
| 590 | `vrpmllpmbb_to_vxvyvz` | no | none needed — only ever sees numpy |
| 656 | `vxvyvz_to_vrpmllpmbb` | **yes** | `if xp is numpy:` |
| 730 | `XYZ_to_lbd` | **yes** | `if xp is numpy:` |

So every in-place op inside a backend-native function is guarded, and the two
unguarded ones are pre-existing code in functions that are not backend-native.
That distinction is the whole audit: an in-place op is only a bug if a backend
array can reach it. Worth writing down because the naive grep flags all four and
the wrong conclusion is "fix them all", which would churn two functions nobody
asked me to migrate.

Also: no added `if` branches on array **values** — they branch on `batched`
(derived from `Xsun.ndim`, configuration), `xp is numpy` (namespace identity),
or `_extra_rot` (a bool kwarg). Duplication is net-negative: `_galcen_rot` /
`_apply_galcen_rot` were hoisted out of a copy-pasted pair, and the audit turned
up a *second* such pair (`vxvyvz_to_vrpmllpmbb` and its inverse share an
identical (3,3,N) build) — noted, not folded in, since the inverse is not in the
sky chain.

### #1236 OPENED — sky chain backend-native end to end

All local gates green before pushing:

| gate | result |
|---|---|
| numpy byte-identity | 28 cases, `0.000e+00` |
| `test_orbit.py` numpy | 577 passed, 4 skipped |
| `test_coords` + `test_streamTrack` numpy | 143 passed |
| `test_quantity.py` numpy | 244 passed |
| `test_streamTrack.py` torch | 45/45 |
| new type-assertion tests | 20/20, verified by flip |

The two acceptance criteria #1235 shipped with are met: unforced `sky`/`galsky`
return backend arrays (5.6e-15 / 2.5e-15 vs numpy), and gradients work
(AD vs FD 6.8e-10 / 2.1e-11).

Worth noting what the caller sweep cost and bought: ~33 minutes of wall-clock on
`test_orbit.py` alone, entirely while otherwise blocked, to confirm the three
`Orbits.py` call sites are clean. #1234 skipped that step and paid four CI rounds
for it. Running it locally is strictly cheaper than discovering it in CI, and it
is free when there is nothing else gating.

### #1236: two CI reds, both mine, neither reachable by my local gates

`test_coords.py` on torch (9 tests) and `test_orbit_setup` on jax.

**Red 1 split two ways.** Eight were `numpy.all(Tensor)` at assertion sites —
the migration *working*, same as #1233/#1234 — fixed with 21 `as_numpy` wraps on
the call results (matching the file's own idiom at line 295, one cast per call
instead of three per test). The ninth was a genuine bug: `vxvyvz_to_vrpmllpmbb`'s
backend branch called `xp.cos(l)` without `promote_scalars`, so a forced backend
with numpy inputs hit `torch.cos(ndarray)`.

**Red 2 was more interesting.** `Orbits._lbd` does `X[bad_indx] += 1e-15` behind
`if True in bad_indx` — a masked in-place nudge to move the exact origin off
itself. Harmless for years while `X` was numpy; fatal the moment
`galcenrect_to_XYZ` became backend-native and returned an immutable jax array.
I fixed it, re-ran, and a **sibling site** failed identically. At that point I
stopped iterating and grepped: **three** sites (`_helioXYZ`, `_lbd`,
`_lbdvrpmllpmbb`). All now `xp.where`, which is out-of-place *and* removes the
data-dependent `if True in ...` branch, so it helps tracing too.

**The real finding is about my gates, not the code.** I ran three things before
pushing: numpy byte-identity, unforced backend-array type assertions, and numpy
caller sweeps (including a 33-minute `test_orbit.py`). All three passed. **None
of them runs the library forced onto a backend** — which is exactly how the
all-backend shards run it, and exactly what both reds needed.
`test_coords.py --backend torch` takes about ten seconds and would have caught
the first outright. Added as a standing fourth gate.

There is a pattern across this whole PR worth stating once: a migration that
makes a function return backend arrays does not fail *in* that function — it
fails in whatever downstream code was quietly mutating its result. Byte-identity
cannot see that, because the numpy path is unchanged. The cheap systematic check
is to grep callers for in-place mutation the moment a return type changes,
rather than waiting for each one to surface.

Fixups verified: `test_coords` 98/98 torch and 98 numpy; `test_orbit` +
`test_coords` 675 passed / 4 skipped numpy; jax orbit sky accessors 8/8. Pushed
as `72b087ea`; stale prior-SHA runs cancelled.

### Tolerance batch re-classified — it is not one cluster, and I mis-described it

Jo approved a shared relative-bound helper for "~13 `test_quantity` entries" on
my description. Measuring the actual cluster in full-file torch context shows
**10 ledger entries with four different causes**, and only two are tolerances:

| entry | cause | ratio |
|---|---|---|
| `test_streamgapdf_sample` | tolerance | 1.31e-5 vs 1e-5 -> **1.3x** |
| `test_sphericaldf_method_value` | tolerance | 1.9e-6 vs 1e-8 -> **190x** |
| `test_sphericaldf_method_inputAsQuantity` | **type leak** (`numpy.float64` has no `.to`) | — |
| `test_sphericaldf_sample_outputunits` | **type leak** (`ndarray` has no `.to_value`) | — |
| `test_sphericaldf_densitymatch_issue677` | **unit error** | — |
| 2x `test_quasiisothermaldf_*` | **stale** (pass now) | — |

Three are numpy leaking into a Quantity path — a source-side coercion bug, not a
bound. Applying the relative-bound helper to them would fix nothing. They do not
belong in the tolerance PR.

**The 190x entry, investigated rather than absorbed** (Jo: "a really big
difference is a red flag"). The assertion is

```python
numpy.fabs(dfh.vmomentdensity(...).to(Msun/kpc**3).value
           - dfh_nou.vmomentdensity(...) * mass_in_msol(vo, ro) / ro**3) < 1e-8
```

Measured: `mass_in_msol(220, 8) / 8**3 = 1.758e+08`, so the compared quantity is
~1.8e8 Msun/kpc^3. An **absolute** 1e-8 bound on that demands **5.69e-17
relative** — *below* float64 eps (2.22e-16). The test as written demands better
than machine precision; it passed only because numpy reproduced the identical
operation order. The observed 1.9e-6 absolute is **1.08e-14 relative**, about
50 ULP, which is ordinary for a different summation order.

So the red flag resolves benignly, and the rule worked exactly as intended: it
forced me to find that the *bound* was unmeetable by construction rather than
quietly widening it. This is the strongest evidence yet for option (3) — the
shared relative helper — because it shows the absolute bound is not merely
awkward, it is unsatisfiable in principle for large-magnitude quantities.

Revised plan: the tolerance PR covers **2** entries (both genuine, both passing
comfortably under a relative bound), the 3 type leaks become a separate
source-side fix, and 2 stale entries get pruned. That is a smaller and more
honest PR than the "~21 entries" I quoted.

### #1236: stop iterating, enumerate — and the blast radius was smaller than it looked

Three CI rounds had each surfaced exactly one more shard, which is the wrong
loop: an hour of queued CI per fact I could measure locally. Switched to running
the affected shards under torch in one pass.

**First, a wrong hypothesis corrected by measurement.** The three streamdf
failures looked like streamdf's own `coords.*` calls returning Tensors, so I
wrapped all 12 of them in `as_numpy`. **Nothing changed.** The Tensor actually
enters through `self._progenitor.dist(**obskwargs)` — an *Orbit accessor*.
`Orbit.dist()` -> `_lbd` -> `XYZ_to_lbd`, which this PR made backend-native. So
the real surface is **six Orbit sky accessors** (`ll`, `bb`, `dist`, `pmll`,
`pmbb`, `vlos`) changing return type under a forced backend — the same thing
#1234 did for `ra`/`dec`/`ll` at the cost of 100 assertion casts. Reverted the
12 wraps rather than leave churn from a disproved hypothesis in the diff.

**Then the enumeration.** `test_quantity.py` on torch: **244 tests, 2 failed,
both pre-existing** — the cross-framework `[jax]` artifact that does not exist
in CI's single-framework shards, and the already-ledgered
`test_streamgapdf_sample` tolerance. **Zero new failures.** That is a much
better answer than the last wake's trajectory implied: the accessor type change
does not ripple through the units layer at all.

The lesson is not "the blast radius was large" — it is that **three sequential
CI rounds told me less than one local pass**, and that a plausible causal story
("streamdf calls coords, coords now returns Tensors") survived long enough to
produce a 12-site edit before a five-minute rerun disproved it. Wrapping call
sites *looked* like the systematic fix and was not; finding the actual producer
was.

### #1236: the 3 streamdf regressions, and what each one actually was

The census line in the CI job log had already named them; I re-derived it the
expensive way first (see below). All three are the same family — downstream code
that assumed numpy — but the *fix* differs per case, and two of the three are
not in streamdf's own coords calls at all:

1. **`test_bovy14_track_spread`** — `_ErrCovsLBScale` is built from
   `self._progenitor.dist/vlos/pmll/pmbb(...)`, i.e. **Orbit sky accessors**, and
   is then multiplied into numpy covariance arrays. Landed those four on numpy
   at construction. FIXED.
2. **`test_density_ll_wsampling`** — entirely **test-side**: a local `ll(apar)`
   helper calls `coords` directly and returns a Tensor, compared against a numpy
   `sample(lb=True)`. One `as_numpy` on the helper's return.
3. **`test_calcaAJacLB`** — `calcaAJac` dispatches on `is_backend_array(xv)`, and
   the backend path raises `NotImplementedError` for `lb=True`. Its own docstring
   explains why: *"the stream track never uses them on the backend"*. That was
   true only because `xv` used to arrive as numpy. Now it can arrive as a backend
   array, so the honest fix is a **numpy fallback**, not an error: `lb`/
   `coordFunc` land `xv` on numpy and take the finite-difference path, which is
   exactly what happened before.

Case 3 is the interesting one. A guard whose justification is "this can't
happen" becomes a bug the moment an upstream migration makes it happen, and the
right response is to restore the old behaviour rather than to widen the guard's
supported surface inside an unrelated PR.

**Two self-inflicted detours this wake, both already in my memories:**

* I tried to enumerate the shard locally. It **deadlocked** — 0.0% CPU, 0s CPU
  time, 60 min elapsed — on the `parallel_map` torch+fork hang, because several
  `test_streamdf` tests build streamdfs with `multi=` beyond the one I
  deselected. Meanwhile the CI job log had already printed
  `153 passed, 3 xfail, 3 FAILED` — a complete census. Written up as
  [[backend-shard-prints-full-census]].
* Inserting the `as_numpy` import into `test_streamdf.py`, I anchored on
  `from galpy.util import conversion, coords` and hit an occurrence **inside a
  function**, producing an IndentationError — the same mistake as #1234. Fixed
  by computing the module-level import block explicitly (last top-level import
  before the first `def`/`class`) instead of pattern-matching.

### A process failure worth more than the code fix

Jo asked "Nothing happening on CI?" — and the answer was that I had three
verified streamdf fixes sitting **uncommitted in the worktree** while I kept
iterating locally. `HEAD == origin`, so CI had nothing to run for an entire
wake, and the 150/4 he was looking at was the *previous* commit's result.

Two habits combined to produce that:

1. **Batching.** I was holding fixes until the whole set was verified, instead
   of pushing each verified piece and letting CI chew on it in parallel. The
   marginal cost of a push is zero; the cost of not pushing is a wasted CI cycle
   *and* an inaccurate picture for anyone watching.
2. **Not checking the SHA after committing.** galpy's `ruff format` / `isort`
   hooks rewrite files and then *fail* the commit — HEAD stays put and the
   subsequent push says "Everything up-to-date". That happened three times this
   day. Written up as [[precommit-hooks-reject-first-commit]].

Now pushed: `d4e669a0` (streamdf boundary + calcaAJac fallback) and `729b2e67`
(coverage for the batched rotation and `_extra_rot=False`, verified by flip —
perturbing the batched contraction by 1e-7 fails all 4 new cases). numpy
regression on the affected streamdf tests: 4 passed. 7 stale prior-SHA runs
cancelled to free runners.

### The ledger has been overstating remaining work by 2×

While waiting on #1236 CI I characterized the 34 ledgered torch `test_actionAngle`
entries by running them with `--runxfail` and clustering the real failures.
**24 of the 34 are `actionAngleVerticalInverse`** — the module Jo has twice
declared permanently out of scope (most recently: "the actionAngleVerticalInverse
and actionAngleTorus modules remain out of scope").

Extending the count across the whole ledger:

| backend   | total | out-of-scope | real burndown |
|-----------|------:|-------------:|--------------:|
| jax       |    43 |           24 |            19 |
| jax-jit   |     5 |            0 |             5 |
| torch     |    47 |           24 |            23 |
| torch-jit |     1 |            0 |             1 |
| **ALL**   |  **96** |       **48** |        **48** |

All 48 are `VerticalInverse` (Torus is already excluded from the suite outright).
So "ledger size = progress metric" has been reading 96 when the actual remaining
work is 48 — every burndown number I have reported is inflated ~2×, and it will
never converge to zero, because half of it is not work.

**The fix already exists in the harness.** `tests/backend_skip.txt` is a
conftest-driven skip list whose own docstring states the case exactly: *"a
PERMANENT exclusion, not pending work, so they are NOT part of any burndown."*
Verified the mechanism holds: the `backend_skip` block runs BEFORE the regen
early-return, so skipped entries are skipped in regen mode too and cannot creep
back into the ledger on the next regen. It is a `tests/` file, so no workflow
edit is needed.

One wrinkle: that list is currently documented (and its skip reason worded) for
external-service/network/flaky tests. Moving VerticalInverse there means
broadening it to "permanently out of scope" with two sub-categories. That is a
doc/reason-string change in `conftest.py`, not a mechanism change.

Queued as the next PR after #1236. The residual 10 real torch actionAngle
entries cluster as: 4 Spherical `linear_angles`, 2 Vertical Harmonic
`actionsFreqs`, 1 AdiabaticGrid outside-grid, 1 Adiabatic `linear_angles_cylsep`,
1 StaeckelGrid circular-orbit Jr=0, and 1 genuine torch bug —
`test_actionAngleVertical_linear_angles` raises `TypeError: all() received an
invalid combination of arguments`.

### 14 assertions that pass when the values are wrong

Chasing the one genuine torch bug in the actionAngle cluster
(`test_actionAngleVertical_linear_angles`, `TypeError: all() received an invalid
combination of arguments`) led somewhere more interesting than a type leak.

The failing line was:

```python
assert numpy.all(numpy.fabs(a - b)) < 1e-100
```

The `< 1e-100` is **outside** `numpy.all(...)`. So it computes `all(|a-b|)` —
a *bool*, True iff every element is non-zero — and compares that to 1e-100.
`True < 1e-100` is False and `False < 1e-100` is True, so the assertion passes
**iff at least one element of |a-b| is exactly zero**. That is close to the
opposite of the intent. Demonstrated:

```
d = [0.0, 5.0, 5.0]                     # one exact match, two wild errors
numpy.all(fabs(d)) < 1e-16   ->  True   # as written: PASSES
numpy.all(fabs(d)  < 1e-16)  ->  False  # as intended: fails
```

A repo-wide sweep found **14 instances**: 2 in `test_actionAngle.py`, 6 in
`test_orbit.py`, 6 in `test_orbits.py` (all six `getOrbit` component checks in
each). These are numpy-path defects — nothing to do with backends — and they
have been green the whole time.

Fixed all 14 by moving the comparison inside the `all()`. **All 4 affected tests
still pass on numpy**, so the underlying values really do agree to <1e-16 /
<1e-100; the assertions were just not checking it. The torch entry additionally
needed `as_numpy` (bare `numpy.all(Tensor)` delegates to
`Tensor.all(axis=, out=)`, which torch rejects) and now passes under torch too,
burning down a ledger entry — verified real, since it failed with the TypeError
before the fix.

Parked as a patch rather than folded into #1236: the files are disjoint
(`test_actionAngle`/`test_orbit`/`test_orbits` vs #1236's coords/streamdf), so it
is a clean separate PR, and #1236's worktree was reverted to keep it from
leaking (cf. [[stacked-pr-branch-surgery-leak]]).

This is the second time this wake that a green check turned out to certify
something narrower than its message claimed — the first being the ledger's
out-of-scope half. Worth a standing habit: when a test is *already* green, that
is not evidence the assertion is meaningful.

### The guard I removed was right about something

#1236's CI came back 149 pass / 2 real fails, both in `test_backend_streamdf.py`
— a file this PR does not touch, failing because of a source change it does:

```
[jax]   ValueError: assignment destination is read-only
[torch] Failed: DID NOT RAISE NotImplementedError
```

The torch one is the expected consequence: I replaced `calcaAJac`'s
`NotImplementedError` for `lb=`/`coordFunc=` with a numpy fallback, and a test
pinned the old contract with the comment *"they raise (not misbehave)"*. I
changed a documented behaviour without updating the test that documented it —
the audit checked for duplication and array-input branches but not for
"does this PR invalidate an existing contract".

The jax one is more interesting, and it vindicates the guard's instinct. My
fallback did `xv = as_numpy(xv)`, and **jax's cast is read-only** while the FD
path writes into `xv` in place. Worse, the in-place write is load-bearing: the
`lb` coordFunc is `lambda x: lbCoordFunc(xv, ...)`, which ignores its argument
and closes over `xv`, so mutating `xv` is the only way the FD perturbation
reaches it. So the fallback was silently broken on jax — exactly the "misbehave"
the raise existed to prevent. Fixed with `numpy.array(as_numpy(xv))`.

I still think replacing the raise is correct: the stream track legitimately
reaches `calcaAJac(lb=True)` with a backend `xv` now, and that call landed on
numpy and worked before the coords migration. But I re-pointed the test at
**matching numpy to rtol=1e-13** instead of at not-raising. A "does not raise"
check is precisely what the read-only bug would have sailed through on torch —
which is the same lesson as the 14 vacuous assertions, arriving twice in one day
from opposite directions.

Pushed `d68fd1b7`; 15 passed locally across both backends. Also: pre-running
`ruff format`/`isort` before committing made the commit land first try, after
three aborted commits earlier today.

### A wall of local failures that was entirely my own setup

Ran a local replica of the `test_backend*.py` CI shard to double-check the
calcaAJac fix, and got **1501 failed / 5033 passed** — against CI's 2 failed on
the *identical* 6825 tests. Spent most of a wake narrowing it: A/B'd CI's
`-W error::DeprecationWarning -W error::FutureWarning` (47 pass either way),
then A/B'd xdist `-n 4 --dist loadfile` (47 pass). Neither was the cause.

What gave it away was splitting the failures by parametrization: **718 `[jax]`
vs 1 `[torch]`**. I had omitted `CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu`, so
jax tried to use the TITAN Xp. I have a memory for precisely this, and had
applied it to a torch run earlier in the same session.

Re-dispatched correctly. Lesson folded back into
[[jax-defaults-to-gpu-force-cpu]]: a wall of local failures CI does not show is
far likelier to be a harness mistake than a real regression, and the
parametrization split is the cheapest first check — lopsided by backend means
environment, spread evenly means code. I had that backwards for an hour,
treating an env artifact as a possible regression.

Worth stating plainly since it cuts against the day's other finding: I have been
right to distrust green checks, but distrust has to cut both ways. A red result
from a harness I just hand-assembled deserves the same scepticism as a green one.

### Next PR verified before opening, on a risk I nearly shipped blind

#1236 sits at 149 pass / 3 pending (the two 1.5h `3.14 test_backend` coverage
shards plus all-clear). Those are the shards that caught the calcaAJac bug, so
no merge until they report.

Used the wait to de-risk the queued vacuous-assertion patch, and found a real
hazard in it. My rewrite turns

    numpy.all(numpy.fabs(a - b)) < tol   ->   numpy.all(numpy.fabs(a - b) < tol)

and 12 of the 14 sites are in `test_getOrbit`, which is **not ledgered** — it
runs live under torch and jax. If the new form broke on Tensors, it would red
the backend shards. Checked the primitive directly:

```
OLD  numpy.all(fabs(d))       < tol  ->  TypeError: all() received an invalid ...
NEW  numpy.all(fabs(d)        < tol) ->  TypeError: all() received an invalid ...
```

Identical. Both call `numpy.all` on the same object, so the rewrite cannot be
worse than what is already green — and since these sites ARE green under torch
today, their values must already be numpy.

That argument is airtight, and I still ran it end to end, because "essentially
nil risk" reasoning is exactly what let the dead `@backendNative` marker through
a week ago. Built the C extension in a fresh worktree off `feat/backends-develop`
(not symlinked — cf. [[worktree-so-provenance]]) and ran `test_getOrbit` under
`--backend torch` and `--backend jax`: both pass. Committed locally as
`7b8301de`, unpushed, since #1236 is still the active PR.

So the patch is now verified on numpy, torch, and jax, and the ledger split
(96 → 48) is staged behind it.

### Two PRs in parallel (Jo's call)

Jo checked in: *"things move quite slow. So maybe work on two PRs at the same
time if you can, two parallel ones."* Right diagnosis — the `3.14 test_backend*`
coverage shards take ~1.5h, and serializing on them was burning most of each
wake on polling. Folded into [[autonomous-vacation-mission]], superseding the
"one PR at a time" line in the loop prompt.

In flight now:

| PR | what | state |
|----|------|-------|
| #1236 | coords sky chain backend-native (5 links) + streamdf/Orbits fallout | 149 pass, 3 pending |
| #1237 | 14 assertions that compared a bool to the tolerance | just opened |

Staged and committed locally but unpushed (`17248a46`), to go out the moment one
of the above merges: the ledger scope split, 96 → 48 entries. Verified the moved
`VerticalInverse` entries skip in **both** normal and regen mode, and that the
nodeid appears in no other list, so the skip can only be coming from
`backend_skip.txt` and cannot creep back on the next regen.

All three touch disjoint files, so none needs rebasing on the others:
- #1236: `coords.py`, `Orbits.py`, `streamdf.py`, `test_coords`, `test_backend_coords`, `test_streamdf`, `test_backend_streamdf`
- #1237: `test_actionAngle.py`, `test_orbit.py`, `test_orbits.py`
- staged: `backend_xfail.txt`, `backend_skip.txt`, `conftest.py`

Holding at two open rather than three: the project is runner-constrained, and
three concurrent PRs x ~150 checks would starve the queue that #1236 is already
waiting on.

### #1236: the only blocker was 2 lines

All 152 build/backend checks passed; `all-clear` failed for one reason, visible
only inside its log: `{"conclusion":"failure","name":"codecov/patch"}`. So the
whole PR was gated on uncovered new lines, nothing else.

Pinning down *which* lines took two dead ends worth recording. Running
`coverage.py` locally is not an option — tracing **aborts the interpreter** on
`jaxlib.xla_client` import (`Fatal Python error: Aborted`). And codecov's v2
`compare` endpoint does carry per-line diff data, but with an inverted encoding:
`coverage.head == 0` means HIT, `== 1` means MISS. I filtered for `0` first and
got a confident list of 81 "uncovered" lines — which was the covered set. The
tell was the count: `totals.head.patch` said 81 hits / 2 misses. Both folded
into [[codecov-line-coverage-parse-is-bogus]].

The real gap was 2 lines, and it was genuine untested logic rather than a
coverage artifact. `vxvyvz_to_vrpmllpmbb` with **both** `XYZ=True` and
`degree=True`: `degreeDecorator` has already converted args 3/4 deg->rad, which
is wrong when they are X/Y rather than l/b, so the body undoes it — numpy in
place (`l *= ...`), backend arrays out-of-place because they are immutable.
Nothing in the suite passed both flags with backend arrays. Per-file patch
coverage: coords.py 81/83, streamdf.py 10/10, Orbits.py 4/4.

Added a test asserting `is_backend_array` (so a silent numpy fallback cannot
pass it) plus a match to numpy at rtol=1e-13. Flip-verified: perturbing the
backend branch by 1e-7 relative fails both jax and torch; restored byte-exact.
Pushed `961934d5`.

Parallel status: #1237 at 133 pass / 19 pending with codecov/project red — on a
test-only PR that is most likely the stale-base phantom
([[codecov-stale-base-892-suppression]]), to confirm once its coverage shards land.

### #1237 MERGED — first result of running two PRs in parallel

`03a81be7` on `feat/backends-develop`. 154 checks green, 0 pending. Audit before
merging: tests-only (no source changes), exactly 14 rewritten `numpy.all(` sites
plus two `numpy.fabs(` reflows from the multi-line restructures, no new helpers,
nothing to place. `codecov/project` had gone red mid-run and cleared itself —
the stale-base phantom again, not a real drop.

Net effect: 14 assertions that passed when the values were wrong now actually
check them, and one ledgered torch entry burns down.

The parallelism paid immediately. #1237 went from opened to merged inside the
window where #1236 was still grinding through its 1.5h coverage shards — under
the old serial rule that whole window was spent polling.

#1238 opened in the freed slot: the ledger scope split, 96 -> 48. Rebased onto
the new develop head first. Verified on the shard it affects —
`test_actionAngle.py --backend torch` shows the 24 torch `VerticalInverse`
entries as `s` and the remaining ledger entries still as `x`.

State: #1236 (coords, mid-matrix on `961934d5`) + #1238 (ledger split, fresh).

### Parallelism has a cost, and it showed up

#1236 reached **152 pass / 0 fail**, with only `all-clear` and `backend status
report` pending. Every shard in the `All-backend tests` run succeeded; the run
itself is **queued**, waiting for a runner to post a sticky comment. So the PR
is substantively green and blocked on a cosmetic job.

Cause: #1238's ~149 checks entered the queue ahead of it. Repo-wide queue depth
was 15 (8 x ledger-split, 5 x feat/backends-develop post-merge, 1 x coords,
1 x feat/backends). This is exactly the contention I flagged when declining to
open a third PR — two is the right ceiling on this runner budget, and even two
can stall a trailing job.

Did NOT cancel anything to clear it. The `feat/backends-develop` runs are
same-SHA push runs from the #1237 merge, and cancelling those blocks merges
([[cancel-push-run-leaves-cosmetic-red.md]]); cancelling #1238's would just
re-queue the same work.

Re-audited #1236's grown diff (it gained the streamdf fix and the coverage
test). Clean on all three dimensions. One self-inflicted scare: my
"in-place mutation added?" grep flagged `l[l < 0.0] += 2.0 * numpy.pi`, and my
guard-walker attributed it to line 339 under `if _APY_COORDS:`. That was the
walker matching the FIRST occurrence in the file by string equality, not the
diff's. The actual added occurrence is at 733, inside `if xp is numpy:` at 729,
where it belongs for byte-identity. Line 339 is pre-existing and untouched.

### Remaining torch ledger entries characterized (next PR planning)

Ran the 10 non-actionAngle, non-streamdf torch entries under `--runxfail`
(streamdf still deadlocks locally). Four distinct causes, not one:

| cause | n | note |
|---|--:|------|
| astropy `UnitConversionError` | 2 | type leak |
| `vmap: calling .item()` | 2 | **genuine backend bug** (constantbeta interpolated potentials) |
| `numpy.float64`/`ndarray` lacks `.to`/`.to_value` | 2 | type leak |
| unsatisfiable tolerance bound | 1 | `sphericaldf_method_value` — the 1e-8 ABSOLUTE bound needs 5.69e-17 relative, below float64 eps |
| by-design | 1 | qdf unbound orbit |

The 4 type leaks form the natural next PR (source-side, one coherent fix). The
2 vmap ones are a real defect worth its own investigation.

### #1236 blocked >1h on a single queued job

152 pass / 0 fail. 44 of 45 jobs in its `All-backend tests` run completed; the
45th (`backend status report`, which just posts a sticky comment) has been
`queued` since 16:20, assigned at 18:04, still unstarted at 19:16.

Ruled out the obvious causes rather than assuming: the run is on the **correct
head SHA** (`961934d5`), and there is **no stale same-branch run** holding a
concurrency slot (every older `All-backend tests` run on the branch is
completed). It is plain runner starvation — the repo's in-progress list is all
`feat/ledger-scope-split` and post-merge `feat/backends-develop` work.

Deliberately did not cancel anything this wake. Cancelling #1238's runs would
re-queue identical work; cancelling the `feat/backends-develop` push runs risks
[[cancel-push-run-leaves-cosmetic-red]]. If it is still stuck next wake I will
cancel #1238's CI to prioritise the merge, and re-trigger it after — that
reorders work rather than discarding it.

Worth stating plainly for the parallel-PR experiment: two concurrent PRs is
enough to starve a trailing job for over an hour on this runner budget. The
throughput win is real (#1237 merged inside #1236's window), but it is not free.

### Next-PR target located

`test_sphericaldf_method_inputAsQuantity` fails at
`dfh.vmomentdensity(1.1 * ro * units.kpc, 0, 0).to(...)` with
`AttributeError: 'numpy.float64' object has no attribute 'to'` — under torch the
method returns a bare float instead of a Quantity, i.e. the `physical_conversion`
wrapper is being lost somewhere in the backend path. Real source-side defect,
and the same shape as the `'numpy.ndarray' has no attribute 'to_value'` and the
two `UnitConversionError` entries. Those 4 are one coherent PR.

### #1236 MERGED

`6da07cff` on `feat/backends-develop`. 154 checks green. The starved
`backend status report` job cleared on its own about an hour after I diagnosed
it — no intervention needed, which vindicates not cancelling anything.

What landed: the five-link coords sky chain is backend-native
(`vxvyvz_to_vrpmllpmbb`, `XYZ_to_lbd`, `pmllpmbb_to_pmrapmdec`,
`galcenrect_to_XYZ`, `galcenrect_to_vxvyvz`), with the numpy path byte-identical
(0.000e+00 against a pre-edit reference on 4 bases + a NaN case). Plus the
fallout it exposed: three masked in-place nudges in `Orbits.py` converted to
`xp.where`, the `streamdf` accessor boundary, and the `calcaAJac` lb/coordFunc
numpy fallback with its read-only fix.

Two PRs merged today (#1237, #1236) and one open (#1238). The develop branch has
moved twice; #1238 needs no rebase — its files (ledger + conftest) are disjoint
from both, and per [[dont-reflexively-rebase-disjoint-prs]] a stale base is
harmless without a real conflict.

### Next-PR investigation: first hypothesis was wrong

Chasing `AttributeError: 'numpy.float64' object has no attribute 'to'` in
`test_sphericaldf_method_inputAsQuantity`, I noticed `vmomentdensity` carries
only `@backend_input("r")` while its sibling `dMdE` carries
`@physical_conversion("massenergydensity", pop=True)`, and hypothesised that
#1197 had replaced a `physical_conversion` decorator with the coercion one.

**Git history refuted that**: `@backend_input("r")` was *added* by #1197 on top
of a method that had no decorator at all. And a direct probe refuted it again —
`vmomentdensity(1.1*ro*units.kpc, 0, 0)` returns a bare `float64` with no `.to`
on **numpy as well as torch**, so the Quantity in the passing test comes from
fixture/config state my probe did not reproduce.

Confirmed the real asymmetry: the test **passes on numpy** (1 passed, 48s) and
fails on torch. So it is a genuine torch-only regression, but the mechanism is
not a missing decorator, and I do not yet know what it is. Next step is to
reproduce under the test's own fixture rather than a hand-built df.

Recording the dead end deliberately: two plausible-looking signals (an
asymmetric decorator, a suggestive git blame) both pointed the wrong way, and
the cheap probe is what caught it. Same discipline that caught the 1501-failure
false alarm — check the hypothesis before building on it.

### #1238 merged by Jo; both slots free. Root cause found for the vmap pair

Merge attribution (checked, not assumed): #1237 and #1236 merged by
`jobovy-agent` (me) at 17:29 and 20:13; **#1238 merged by `jobovy` at 20:41**.
It was 154 pass / 0 pending at that moment, so it landed green. Develop's own
CI shows 2 "failures" — both codecov, reported while 179 checks are still
queued, the usual premature pattern, not real. Ledger on develop is now **48**.

Then traced the 2 `vmap: calling .item()` torch entries to a precise root cause,
and it is **not** in the DF at all — it is in a shared backend helper:

```
galpy/backend/interpolate.py:139
    idx = xp.clip(xp.searchsorted(xb, r, side="right") - 1, 0, cb.shape[1] - 1)
    dr = r - xb[idx]        # <-- RuntimeError under torch vmap
```

There is **no `.item()` anywhere in galpy source** (grepped) — torch's message
lists three causes and the real one is (2): `xb[idx]` is advanced indexing with
a *data-dependent tensor index*, which functorch cannot batch. Same for
`cb[0, idx]` and `cb[j, idx]` further down the Horner loop.

That makes it a genuine vmap-compatibility defect in `eval_ppoly`, affecting any
vmapped use of an interpolated potential — which matters for GPU batching well
beyond the 2 ledger entries. Candidate fix is `xp.take(...)` in place of the
tensor-index reads, but I have NOT verified functorch batches `take`
(array-api-compat maps torch's to `index_select`) — that needs testing before
writing the PR, not assuming.

Also worth noting the shape of the find: the failing construct is a
data-dependent *index*, a close cousin of the data-dependent *branch* class the
standing audit looks for. The audit greps for `if` on arrays; it would not have
caught this.

### Correction: the vmap root cause I gave last wake was wrong

Last wake I said `xb[idx]` in `eval_ppoly` was "advanced indexing with a
data-dependent tensor index, which functorch cannot batch", and named
`xp.take` as the candidate fix. I flagged that I had not verified it. Good,
because **the diagnosis was wrong**: a toy showed bare `xb[idx]` batches fine
under plain `vmap`, as do `xp.clip` and `xp.searchsorted`.

The real cause is the **composition**. `galpy/backend/autodiff.py` builds
`torch.func.grad` + `torch.func.vmap` for the fE chain, and the construct fails
only under `vmap(grad(f))` — not under plain `vmap`, not under `grad` alone.
Isolated it op by op:

| construct (under `vmap(grad(...))`) | result |
|---|---|
| `searchsorted` alone | OK |
| index with a **constant** index | OK |
| index with a **computed** index | **FAIL** |
| `torch.take(xb, idx)` | OK (warns: no batching rule, falls back) |
| `torch.index_select(xb, 0, idx)` | OK, no warning |

So `take`/`index_select` *is* the right fix — but for a different reason than I
gave, and it matters only under the composed transform. `index_select` avoids
the fallback perf warning that `take` emits; choosing between portability
(`xp.take`) and that warning is the open question for the PR.

Two lessons, both already cost me time today: torch's vmap error message names
three causes and the real one was none of them, and a composed transform has to
be bisected layer by layer (`f`, `grad(f)`, `vmap(f)`, `vmap(grad(f))`) rather
than reasoned about. Written up as [[vmap-grad-computed-index]].

Also of note: the failing construct is a data-dependent **index**. The standing
pre-merge audit greps for data-dependent **branches** (`if` on arrays) and would
never have caught this.

### #1239 opened: eval_ppoly survives vmap(grad)

Fixed and landed in one pass after Jo pointed out I had deferred a verified fix
for no reason. `_take0()` over the 5 computed-index reads in `eval_ppoly`.
Burns down `test_constantbeta_interpolatedpotentials_dens_directint` (torch).

Two self-inflicted problems caught during the work, both worth recording:

1. **I broke the numpy path.** First version routed numpy through `take` too.
   `take` with a 0-d index returns a 0-d ARRAY where `a[idx]` returns a numpy
   SCALAR — same value, same shape, different type — and
   `interpSphericalPotential._revaluate` depends on the scalar, so it raised
   `Concatenation operation is not implemented for NumPy arrays`. numpy now
   takes `a[idx]` verbatim: byte-identical BY CONSTRUCTION, not by measurement.
2. **My first flip-verification was invalid.** `git stash` removed the new test
   along with the fix, so the run reported "150 deselected" — which I nearly
   read as a pass. Reverting only the source file gives the real answer: fails
   without, passes with.

The sibling `..._beta` now fails LATER, for a different reason (torch-built
`_Phimax` meeting numpy `r` in `_revaluate`). Distinct bug, left ledgered rather
than folded in.

### Type-leak investigation: negative result, recorded so I stop repeating it

Third attempt at `test_sphericaldf_method_inputAsQuantity`
(`'numpy.float64' has no attribute 'to'`). Established:

* `sphericaldf.vmomentdensity` resolves to itself, wrapped exactly once (by
  `@backend_input("r")`), with **no** `physical_conversion` anywhere in the
  chain or its closure. So the Quantity is inherited from an inner decorated
  call, not applied at this level.
* With `astropy-units=True` in config, a hand-built `isotropicHernquistdf`
  returns a **Quantity on BOTH numpy and torch**. The isolated reproduction does
  not exhibit the bug.

So the failure depends on state accumulated earlier *within* the test, not on
construction — which is why three successive hand-built probes have all come
back clean. Next attempt should bisect the test body itself (it makes many
calls before reaching vmomentdensity), not rebuild the fixture again.

### Type-leak: five negative reproductions. Stopping the reconstruct approach.

`test_sphericaldf_method_inputAsQuantity` fails under `--backend torch` with
`AttributeError: 'numpy.float64' object has no attribute 'to'` at
`dfh.vmomentdensity(1.1*ro*units.kpc, 0, 0).to(...)`. Five attempts to
reproduce it outside the suite, each controlling for one more variable:

| # | construction | result |
|---|--------------|--------|
| 1 | hand-built df, default config | float64 on **both** backends (config, not a bug) |
| 2 | + `astropy-units=True` | **Quantity on both** |
| 3 | + warm-up `__call__` then `dMdE` first (as the test does) | Quantity on both |
| 4 | + the test's exact potential (no ro/vo on the potential) | Quantity on both |
| 5 | + everything constructed INSIDE `use("torch", force=True)` | Quantity on both |

So the isolated call is correct on torch. The real run genuinely returns
`numpy.float64` at that same line, so the difference is something the suite does
that a script cannot: the conftest forces the backend at session start, **before
galpy is imported**, whereas every probe imports galpy first and forces after.
That is a real distinction here (cf. [[conftest-no-module-level-galpy-import]]).

**Stopping the reconstruct approach** — it has now cost five rounds and produced
no signal. The next attempt instruments the *actual* failing run (a print/hook
inside a real `--backend torch` session at that call site) rather than trying to
rebuild the conditions. Recording the whole table so nobody, including me,
re-runs any of these five.

Meanwhile #1239 is healthy: 65 pass, 86 pending, only the usual premature
codecov reds.

### #1239: codecov/patch was REAL this time

Unlike the premature codecov reds I have been dismissing all week, #1239's
`codecov/patch` was a genuine gap: interpolate.py at 72.7% (11 patch lines,
3 missed). Checked instead of assuming, which was the right call.

    102  return a[idx]                        <- _take0's numpy branch
    171  return _take0(xp, cb[0], idx) * 0.0  <- the nu > k zero shortcut
    181  term = _take0(xp, cb[j], idx) * fall <- the analytic 0 < nu <= k branch

Lines 171/181 were already untested before this PR; touching them made them
count as added, which is a fair prompt to finally cover them.

Covered with a nu-parametrised test {0,1,2,3,5} against **scipy
PPoly.derivative(nu)** as an independent reference (rtol 1e-10). With k==3,
nu==5 exercises the zero shortcut and nu in {1,2,3} the falling-factorial
branch; running on numpy is also what covers `_take0`'s numpy branch. Value
checks against an external reference, not "did not raise".

Confirmed the three lines are actually hit using `trace.Trace` rather than
assuming the parametrisation reaches them — coverage.py is unusable here
because it aborts the interpreter on jaxlib import. Pushed `71c61a43`.

Worth noting against the week's pattern: I have been right to treat codecov as
noisy, but "usually premature" is not "always wrong". Reading the actual missed
lines cost one API call.

### I spent three wakes chasing a stale error signature

Instrumented the real `--backend torch` session (pytest plugin wrapping
`sphericaldf.vmomentdensity`, writing to a file because the conftest force-exit
truncates stdout). The probe recorded **zero calls** — and re-reading the
CURRENT failure explains why:

```
AssertionError: sphericaldf method vmomentdensity does not return correct Quantity
  at tests/test_quantity.py:18044
```

It is an **AssertionError**, i.e. a NUMERICAL mismatch. Not the
`AttributeError: 'numpy.float64' object has no attribute 'to'` I have been
hunting since 2026-08-02. That signature came from an older commit; the entry
description I kept working from was stale.

So all five "negative reproductions" were negative because **there was nothing
to reproduce** — the Quantity machinery works, which is exactly why every
probe returned a Quantity. I read that as "cannot reproduce the bug" when it
actually meant "this is not the bug".

And the real failure is one I had already classified weeks ago: the
`vmomentdensity` numerical entry whose 1e-8 ABSOLUTE bound demands ~5.69e-17
relative — below float64 eps (2.22e-16). Unsatisfiable by construction. It was
in my own tolerance-batch table the whole time.

**Lesson, and it is an expensive one**: I anchored on a traceback captured days
earlier and never re-read the failure after the tree moved underneath it. Cost:
five probe rounds plus a plugin. Re-reading the current failure took one command.
Before investigating any ledger entry, re-run it and read TODAY's error — the
entry description and any saved traceback are both stale by default.

Net for the type-leak PR: the "4 type leaks" grouping is wrong. At least one of
the four is not a type leak at all. The remaining three need re-characterising
against current HEAD before any of them becomes a PR.

### #1239 red: my own fix broke 3 streamTrack torch tests

The 3.14 test_backend shard (1h34m) caught 3 real reds I introduced:

```
test_backend_streamTrack.py::test_fit_track_backend_prog_grad[torch]
                            ::test_fit_track_backend_prog_grad_jax_torch_agree
                            ::test_fit_track_endtoend_torch_prog_grad
TypeError: take() got an unexpected keyword argument 'axis'
```

`eval_ppoly`'s `xp` is sometimes the **RAW torch module**, not
array-api-compat's wrapper — and `torch.take()` has no `axis` kwarg (it
flattens). My local verification used `array_api_compat.torch`, which does
accept `axis`, so it never exercised the failing path. Measured properly this
time, across every namespace this is actually called with:

|                       | numpy | jax | compat-torch | raw torch |
|-----------------------|-------|-----|--------------|-----------|
| `take(a, i, axis=0)`  | OK    | OK  | OK           | **FAIL**  |
| `index_select(a,0,i)` | --    | --  | OK           | OK        |

Fixed by preferring `index_select` when the namespace has it, falling back to
`take(axis=0)`. Verified: streamTrack + interpolate = **218 tests, 0 failures**
(was 3). Pushed `75037755`.

**This is the second time in one PR that my local check used a narrower
namespace than production.** First it was numpy scalar-vs-0-d-array; now it is
compat-torch vs raw torch. The lesson generalises: when touching a helper that
takes `xp` as a parameter, enumerate which namespace objects actually reach it
rather than testing with the one that is convenient to import. galpy passes at
least three distinct things as `xp`.

Also vindicates the shard: a 1.5h coverage job I have grumbled about all week
is exactly what caught this, and neither my local runs nor the earlier CI
shards did.

### Re-characterised all 20 non-streamdf torch ledger entries at HEAD

Acting on [[reread-failure-before-investigating]] rather than trusting the
groupings. Ran them with `--runxfail` against current HEAD:

**3 are STALE and now pass** — un-ledger candidates:
* `test_actionAngleVertical_linear_angles` (fixed by #1237's assertion work)
* `test_analytic_ecc_rperi_rap`
* `test_constantbeta_interpolatedpotentials_dens_directint` (fixed by #1239)

**17 still fail**, and the type-leak cluster is NOT what I had recorded:

| n | cause |
|--:|-------|
| 4 | `AssertionError: Maximum deviation from linear trend` (Spherical linear_angles) |
| **3** | **`TypeError: to() received an invalid combination of arguments`** |
| 2 | `AssertionError: Frequencies of harmonic oscillator` |
| 1 | `AttributeError: 'numpy.ndarray' has no attribute 'to_value'` |
| 1 each | AdiabaticGrid outside-grid, Adiabatic cylsep, StaeckelGrid Jr=0, qdf unbound (by design), qdf `__call__` returntype, `Concatenation ... NumPy arrays`, Plummer densitymatch |

The 3-entry cluster is `torch.Tensor.to()` shadowing astropy `Quantity.to()` —
the method returns a **Tensor**, so `.to(units.Msun/...)` hits Tensor.to()'s
dtype/device signature. That is a coherent, single-mechanism group and a real
PR candidate. It is NOT `'numpy.float64' has no attribute 'to'`, which is what
I chased for three wakes and which no longer exists anywhere in the set.

### A parsing bug that told me everything passed

First read of the junit reported "NOW PASSING: 20" — i.e. the entire ledger
cleared. It was my parser: `tc.find('failure') or tc.find('error')`. An
ElementTree Element **with no children is falsy**, so real `<failure>` elements
were discarded and every test looked green. Python printed a DeprecationWarning
saying exactly this and I nearly scrolled past it. Correct form is
`f = tc.find('failure');  if f is None: f = tc.find('error')`.

Had I not re-read, I would have reported the torch ledger as fully burned down.
Same family as the `numpy.all(...) < tol` bug I fixed in #1237: a truthiness
mistake that fails silently in the optimistic direction.

### #1239 audit clean, waiting on the 1.5h shard

149 pass / 4 pending / **0 fail** — codecov cleared once the coverage shards
uploaded, confirming last wake's patch fix was the right read.

Pre-merge audit:
* **2 files**: `galpy/backend/interpolate.py` + its test. No source sprawl.
* **if→xp.where class**: none introduced. The one new branch is `if xp is
  numpy`, a namespace dispatch — static, not data-dependent.
* **Duplication**: 5 tensor-index reads collapse onto one `_take0`.
* **Placement**: checked `galpy/backend/*` for a pre-existing gather/take
  helper — there is none, so a new one is warranted. It lives in
  `interpolate.py`, which IS the backend-utility module, module-private next to
  its only consumer. Per the standing directive I checked before creating.

**Deliberately NOT expanding scope**: found a sibling tensor-index read at
`galpy/backend/random.py:354` (`a_t[idx].reshape(...)`) with the same shape as
the bug. It is in the RNG shim, which is probably not vmapped, but it is worth
checking. Logged as follow-up rather than folded into #1239.

**Deliberately NOT adding the un-ledger commits either.** #1239 fixes
`test_constantbeta_interpolatedpotentials_dens_directint`, so un-ledgering it
would normally belong in the same PR — but #1239 is 4 checks from green and a
new commit restarts the 1.5h `3.14 test_backend` shard. A stale ledger entry is
harmless (strict=False xfail is green either way), so the three stale entries go
into one follow-up burndown PR instead:

* `test_actionAngleVertical_linear_angles` (fixed by #1237, already on develop)
* `test_analytic_ecc_rperi_rap` (already on develop)
* `test_constantbeta_interpolatedpotentials_dens_directint` (needs #1239 merged first)

That ordering matters: un-ledgering the third before #1239 lands would red the
PR doing it.

### #1239 MERGED; #1241 opened (ledger 48 → 45)

`d66a3038` on `feat/backends-develop`, 154 checks green. `eval_ppoly`'s
coefficient reads now survive `vmap(grad)`, which unblocks vmapped use of
interpolated potentials generally — not just the two ledger entries it was
hiding behind.

Opened **#1241**: drop 3 stale torch ledger entries, verified PASS with
`--runxfail` at `d66a3038` before removal.

| entry | fixed by |
|---|---|
| `test_actionAngleVertical_linear_angles` | #1237 |
| `test_analytic_ecc_rperi_rap` | intervening work |
| `test_constantbeta_interpolatedpotentials_dens_directint` | #1239 |

**Why staleness accumulates here, structurally**: ledger entries are
`xfail(strict=False)` — green whether they fail OR pass. That is deliberate (it
stops flaky-slow jax tests redding the gate) but it means a fixed test stays
ledgered silently and the burndown metric overstates remaining work. Combined
with the finding two wakes ago that recorded *causes* also drift, neither the
entry list nor its descriptions can be trusted without re-running.

Ledger trajectory: 96 → 48 (#1238, out-of-scope family moved to
`backend_skip`) → 45 (#1241).

Also cleaned up: deleted the merged `feat/torch-typeleaks` remote branch.

Still open as follow-ups, deliberately not folded into #1241:
* the `Tensor.to()` cluster (3 entries) — `torch.Tensor.to()` shadowing astropy
  `Quantity.to()`; coherent single-mechanism group, real PR candidate
* `galpy/backend/random.py:354` — sibling `a_t[idx]` tensor-index read, same
  shape as the #1239 bug; needs checking for vmap exposure

### The "Tensor.to() cluster" was an artifact of my own test selection

Ran the FULL `test_quantity.py` under torch (244 tests) — the context CI
actually uses — and it contradicts the batch sweep I based two wakes of
conclusions on. Same commit, same backend:

| test | 20-nodeid batch | FULL file |
|------|-----------------|-----------|
| `..._returnunit` | FAIL | **PASS** |
| `..._returntype` | FAIL | **PASS** |
| `test_sphericaldf_method_inputAsQuantity` | `TypeError: to() received an invalid combination` | `AttributeError: 'numpy.float64' has no attribute 'to'` |

So:

* The **"Tensor.to() cluster, 3 entries, one mechanism"** I reported as the next
  PR candidate **does not exist**. It was an artifact of running the ledgered
  nodeids together as a hand-picked list.
* My conclusion the wake before — that the `'numpy.float64'` signature was
  *stale* — was also wrong, and wrong in the opposite direction. The original
  signature is the real one.

The same flawed method produced two contradictory wrong answers on consecutive
days, and I reported both with confidence. `test_quantity.py` carries cross-test
state, so a hand-picked nodeid list is a *novel* context, distinct from both
isolation and the full file. Written up as
[[handpicked-nodeid-batch-is-not-a-context]]: characterise by running the whole
file and filtering the junitxml.

Silver lining: the full-file run found 2 genuinely stale entries the batch had
marked failing. #1241 grew to **45 → 43** (`b2a26b4c`), still cheap to add since
it was only 13 checks in.

Ledger trajectory: 96 → 48 (#1238) → 45 → 43 (#1241).

### Sibling tensor-index site: checked, NOT the same bug

Followed up on `galpy/backend/random.py:354` (`return a_t[idx].reshape(tshape)`),
which I flagged during #1239's audit as structurally identical to the
`vmap(grad)` bug.

It is inside `random.choice(key, a, shape=None, p=None)`, whose index comes from
`torch.randint(..., generator=g)` or `torch.multinomial(..., generator=g)`.
Explicit-generator RNG is neither differentiable nor vmappable, so this code can
never sit inside the `vmap(grad)` composition that broke `eval_ppoly`. The bare
`a_t[idx]` is correct here and needs no change.

Closing it as a non-issue **with the reason recorded**, rather than leaving an
open "should probably check this" that would get re-investigated later. The
grep-alike was real; the exposure is not.

### Method correction now applied prospectively

`#1241` stands at ledger **43** (96 → 48 → 43). Its CI is early (9/142).

Note for the remaining burndown: every future characterisation of
`test_quantity` / `test_diskdf` / `test_sphericaldf` / `test_qdf` entries must
run the WHOLE file and filter the junitxml — see
[[handpicked-nodeid-batch-is-not-a-context]]. The three contexts (isolation,
hand-picked batch, full file) give three different answers, and only the last
matches CI. Two of my recent conclusions came from the wrong one.

### actionAngle cluster characterised in FULL-file context: all numerical

Ran the whole `test_actionAngle.py` under torch with `--runxfail` (209 tests,
9 failures) and filtered for the ledgered nodeids. Every remaining AA entry
still fails, and **all 9 are AssertionErrors** — no type leaks, no exceptions:

| n | cause |
|--:|-------|
| 4 | `Maximum deviation from linear trend` (Spherical linear_angles) |
| 2 | `Frequencies of harmonic oscillator` (Vertical Harmonic) |
| 1 | `Radial angle ... linear trend` (Adiabatic cylsep) |
| 1 | `actionAngleAdiabaticGrid evaluation outside grid` |
| 1 | `Circular orbit in MWPotential does not have Jr=0` (StaeckelGrid) |

**Negative control worth having**: for this file the hand-picked batch and the
full file AGREE. So the discrepancy that burned me on `test_quantity` is
specific to files carrying cross-test state, not a universal property of batch
runs. That bounds how expensive the corrected method has to be — full-file runs
are only mandatory for the four state-carrying files.

Consequence for planning: the AA cluster is **entirely a numerical/tolerance
question**, which is the batch Jo pre-approved with the "keep them stringent, a
really big difference is a red flag" caveat. It is not a code-fix cluster, so it
should not be scoped as one. Each of the 9 needs its actual deviation measured
against its bound before deciding fix-vs-loosen — and per the earlier
`vmomentdensity` finding, at least some galpy bounds are ABSOLUTE where they
should be relative, which makes them unsatisfiable rather than merely tight.

#1241 (ledger 43) at 63 pass / 89 pending; its `codecov/project` red is the
stale-base phantom — the PR contains no source changes at all.

### The ledger file warned me about this test by name, and I didn't read it

#1241's torch orbit shard went red. Not an assertion — a **Timeout**:

```
+++ Timeout +++
FAILED                                            [ 99%]
Backend torch: 123 passed, 0 xfail (ledger), 12 deferred, 1 FAILED, 0 ERROR
```

on `test_analytic_ecc_rperi_rap`, one of the entries I un-ledgered. And
`backend_xfail.txt` documents that exact test, by name:

> Do NOT re-measure on a dev box instead. The backend matrix is python 3.14 and
> shards test_orbit into 5 pytest-split groups; a local py3.13 unsplit run of
> the same files disagreed on exactly one entry (test_analytic_ecc_rperi_rap:
> passes locally, xfails in CI group g3).

I did exactly the invalid measurement it names — a py3.13 unsplit local run —
and never opened the file's header. Compounding it, my verification was
**pass/fail only, never wall-clock**: the test runs >120s on a 72-core box and
blows the per-test timeout inside CI's g3 group, so "it passed locally" was true
and useless.

Restored verbatim; ledger is **44**, not 43. The other four un-ledgers stand —
ordinary assertion-level entries with no split-group sensitivity.

Two habits added to [[verify-exact-ci-shard-context]]: read the ledger file's
comments for the nodeid before touching it, and check RUNTIME as well as
outcome, because a timeout is a failure mode that a passing local run cannot
detect.

Sobering pattern: this is the third consecutive wake where my verification
method, not the code, was the defect — batch-vs-full-file context, then Element
truthiness, now dev-box-vs-shard. The code changes have been fine.

### Spherical linear_angles: 118x–501x over bound — a RED FLAG, not a tolerance

Measured the actionAngle cluster properly (bound is `10**toldar`, `toldar=-8.0`,
so 1e-8):

| test | observed | bound | over |
|------|---------:|------:|-----:|
| `Spherical_linear_angles` | 5.00909e-06 | 1e-8 | **501x** |
| `Spherical_linear_angles_fixed_quad` | 5.00909e-06 | 1e-8 | **501x** |
| `Spherical_noninclinedorbit_linear_angles` | 1.17986e-06 | 1e-8 | **118x** |
| `Spherical_almostnoninclinedorbit_...` | 1.17986e-06 | 1e-8 | **118x** |

Per Jo's standing rule — *"minor traced-only loosened tolerances are fine as
long as they stay stringent; a really big difference is a red flag"* — 500x is
emphatically a red flag. **These 4 are not tolerance candidates; they look like
a real accuracy defect in the torch angle path** and must not be bundled into a
tolerance PR.

Two details that reinforce it: the deviations are IDENTICAL within each pair
(5.00909e-06 twice, 1.17986e-06 twice), which points at one shared root cause
rather than four independent numerical edges; and the magnitudes are ~1e-6,
far above anything float64 round-off explains.

Still unmeasured (messages truncate before the number): Vertical Harmonic
freqs x2, Adiabatic cylsep, AdiabaticGrid outside-grid, StaeckelGrid Jr=0.

Also: my first extraction of these numbers was **wrong** — the regex pulled the
exponent out of `10.0**-10.0` and reported "observed -10.0". I caught it because
the values were implausible, printed the raw messages instead, and only then
read the bound from the source. Reporting parsed numbers from an unvalidated
parser is the same failure family as the ElementTree truthiness bug two wakes
ago.

### #1241 status

macOS 3.14 job failed at step 16, **"Install torus code"** — an external C++
dependency install, not a test. My change is ledger-only and numpy runs ignore
the ledger entirely, so it is unrelated infra flake. Could not re-run it yet
("workflow run containing this job is already running"); retry once the run
completes. torch group-3 (the shard that caught the timeout) still pending.

### All 9 actionAngle entries measured — they are FOUR different problems

| test | observed | bound | over |
|------|---------:|------:|-----:|
| `AdiabaticGrid_outsidegrid_c` | 0.0047 | 1e-8 | **4.7e5x** |
| `Spherical_linear_angles` (x2) | 5.01e-06 | 1e-8 | **501x** |
| `StaeckelGrid_basicAndConserved` | 4.835e-14 | 1e-16 | **484x** |
| `Spherical_noninclined...` (x2) | 1.18e-06 | 1e-8 | **118x** |
| `Vertical_Harmonic_...` (x2) | 2.765e-09 | 1e-10 | **27.6x** |
| `Adiabatic_linear_angles_cylsep` | 4.359e-08 | 1e-8 | **4.36x** |

Lumping these as "the tolerance batch" would have been wrong. They are four
distinct situations:

**(a) Real defects — do NOT loosen.**
`AdiabaticGrid_outsidegrid_c` compares `0.0111` against `0.0064` — the values
disagree in the *second significant digit*, 4.7e5x over bound. That is a wrong
answer, not a precision issue. The 4 Spherical entries sit at ~1e-6 absolute
with pairwise-identical deviations, which points at one shared defect in the
torch angle path.

**(b) An unreasonable bound.** `StaeckelGrid` asserts Jr of a circular orbit is
`< 1e-16` and observes `4.835e-14`. 1e-16 is "exactly zero" for a quadrature
result; ~5e-14 is ordinary accumulated round-off. Same shape as the
`vmomentdensity` bound that needed 5.69e-17 relative — the bound is the bug.
This one is not backend-specific in nature.

**(c) Genuinely minor.** `Adiabatic_linear_angles_cylsep` at 4.36x is exactly
the "minor traced-only loosening" Jo pre-approved.

**(d) Borderline.** The 2 Vertical Harmonic entries at 27.6x. Clearing them
means 1e-10 -> 1e-8, a 100x bound change for a 27.6x miss. I would not do that
unilaterally under the "a really big difference is a red flag" rule.

So the honest proposal to Jo is 1 loosening (c), 1 bound-correction (b), and
5 entries reclassified as **defects to fix** (a) — not the "~9 tolerance
entries" I had been carrying. Nothing here is ready to become a tolerance PR
without his call on (d).

### #1241

All 6 torch orbit shards now PASS including group-3 — the restore fixed the
real red. 151 pass / 1 fail / 2 pending; the only fail is macOS 3.14 step 16
"Install torus code", an external C++ dependency install unrelated to a
ledger-only change. GitHub still refuses the job re-run ("workflow run
containing this job is already running").

### #1241: 152 pass, macOS flake re-running

The macOS 3.14 red was step 16 **"Install torus code"** — the external C++
Torus library install, not a test. Ledger-only change, and numpy runs ignore the
ledger entirely, so it cannot be caused by this PR. Re-ran the job once the
parent run completed (GitHub refuses `jobs/<id>/rerun` while the run is still
in_progress — worth knowing, it 403s with "workflow run containing this job is
already running").

**One manual step will be needed**: `all-clear` already evaluated and recorded
the macOS failure, so it stays red even after the job passes. It has to be
re-run *after* macOS goes green, not before.

State: 152 pass, 1 pending (macOS re-run), 1 fail (all-clear, stale).
All 6 torch orbit shards green including group-3 — the entry restore fixed the
only real red.

### #1241 MERGED — ledger 48 → 44

`3677e19f` on `feat/backends-develop`, 154 checks green. The macOS
"Install torus code" flake passed on re-run and `all-clear` re-evaluated itself
once it did — no manual re-run needed after all.

Audit: ledger-only, net −4 (5 removed, `test_analytic_ecc_rperi_rap` restored at
a different line, which is why it shows as a −/+ pair). No code, so the
if→xp.where / duplication / placement dimensions don't apply.

**Ledger trajectory this stretch: 96 → 48 → 44.** The 96→48 was the
out-of-scope `VerticalInverse` family moving to `backend_skip` (#1238); 48→44 is
four genuinely-fixed tests being un-guarded. Both matter for the metric being
honest, but only the second is burndown in the real sense.

Four PRs merged in this run: #1237 (14 vacuous assertions), #1238 (scope split),
#1239 (eval_ppoly under vmap(grad)), #1241 (stale un-ledger).

### Standing state for the next wake

* develop at `3677e19f`; no PR open.
* Remaining torch ledger work is the actionAngle set, now measured and split
  into four categories — 5 defects, 1 bad bound, 1 minor loosening, 2 borderline.
  The borderline pair needs Jo's call before any tolerance PR.
* The 5 defects are ordinary bug-fix work and can start without him:
  `AdiabaticGrid_outsidegrid_c` (0.0111 vs 0.0064 — disagrees in the 2nd digit)
  is the clearest and most likely to be a single root cause shared with the
  4 Spherical angle entries.

### THIRD reclassification: the "5 defects" are documented behaviour

Started the defect work immediately (Jo: "why do you keep waiting for wakes").
Reproduced `AdiabaticGrid_outsidegrid_c` cleanly in one shot:

```
numpy  exact Jr=0.01111777119  grid Jr=0.01111777119  diff=8.639e-16
torch  exact Jr=0.01111777119  grid Jr=0.006408796907 diff=0.004709
```

Then read the class docstring, which I had never opened:

> jax/torch input is supported and differentiable, but evaluates the
> interpolation grid only (**no off-grid fallback**): inputs must lie within
> the grid, whereas the numpy path falls back to a per-point solve off-grid.

The test evaluates R=3.0 against Rmax=2.0 — deliberately off-grid. **By design,
not a defect.** And task #103 already exists for the enhancement ("AA grid
backend: opt-in off-grid fallback, default off for jax/torch").

Checked the Spherical four before repeating the mistake. Also documented:

> 50 matches scipy's adaptive quadrature to <1e-7 over the physical orbit range
> ... only at pathological L/Lcirc<~1e-4 does scipy's own adaptive quad fail to
> ~1e-5, an **irreducible quadrature-method difference** where higher order does
> not help.

That is the GL-vs-adaptive floor already in [[match-numpy-quadrature-floor]],
whose guidance is to compare against a GOLD reference rather than scipy's
adaptive answer. So these are a tolerance/reference question, not defects.

**My classification of this cluster has now been wrong three times**: first "~9
tolerance entries", then "5 defects + 1 bad bound + 1 minor + 2 borderline", now
"mostly documented limitations". Each correction came from reading something in
the repo I had not read — the class docstring, the module comment, and two wakes
ago the `backend_xfail.txt` header.

The pattern is specific and fixable: **I classify from test output before
reading the code's own documentation.** Test output tells you what happened;
the docstring tells you whether it was supposed to. Reading first would have
cost minutes and saved three rounds.

Consequence: there is **no autonomous defect work** in this cluster. It reduces
to Jo's tolerance call, plus optionally task #103 (opt-in off-grid fallback),
which is a feature, not a fix. Abandoned the fix branch rather than manufacture
a change.

### Started task #112 immediately; its premise is stale too

Picked `interpRZPotential` (2 jax entries, task #112 "backend-migrate
interpRZPotential") and read the module first, per the new discipline. It is
**already substantially migrated** — `get_namespace`, `is_backend_array`,
`backend.interpolate`, with explicit backend branches in `_evaluate`/`_Rforce`.
So #112's framing ("backend-migrate") is stale; the 2 remaining jax entries are
specific failures *inside* a migrated module, not an unported module.

Dispatched the full-file jax characterisation (whole file, not a `-k` subset).
It has been running >10 min locally — `test_interp_potential.py` builds large
interpolation grids and jax eager is slow on them, which is consistent with 2 of
this file's entries being `jax-jit` rather than eager.

Leaving it running rather than reporting a number I do not have.

### Ledger composition after #1241 (44 entries)

    jax 19 | jax-jit 5 | torch 19 | torch-jit 1

jax side is spread thin — 4 test_quantity, 3 test_diskdf, 3 test_actionAngle,
2 test_orbit, 2 test_noninertial, 2 test_interp_potential, 1 each for
SpiralArms/sphericaldf/qdf. No large coherent cluster left on either side; the
remaining work is mostly one- and two-entry investigations.

### Threw away the diagnostic, then had to re-run

The first `test_interp_potential.py` jax characterisation failed and I could not
say why: I had written `pytest ... >/dev/null 2>&1; python -c "parse junitxml"`,
so when pytest died early the parse hit `FileNotFoundError` and pytest's own
output — the only evidence — was gone.

Re-ran keeping the log. It is progressing (12 results so far) but genuinely
slow: this file builds large interpolation grids and jax eager grinds on them.
That is consistent with 2 of this file's 4 ledger entries being `jax-jit`
rather than eager.

Rule for myself: **never `>/dev/null` a command whose output is the only
diagnostic if a later step depends on it succeeding.** Redirect to a file
instead — costs nothing, and the one time it matters it saves the whole run.

Leaving the characterisation running rather than reporting numbers I do not have.

### Task-list re-validation against HEAD (Jo approved)

The task list had drifted badly from the code. One command settled the biggest
category — every "backend-migrate X" target is **already migrated**:

| module | backend refs | `_reject_backend` |
|--------|-------------:|------------------:|
| MovingObjectPotential | 16 | 0 |
| NonInertialFrameForce | 10 | 0 |
| interpRZPotential | 21 | 0 |
| ExpTruncNFWPotential | 12 | 0 |
| FerrersPotential | 29 | 0 |
| AnyAxisymmetricRazorThinDiskPotential | 15 | 0 |

**Closed 6 tasks as stale**: #46, #61, #109, #112, #181 (targets already
migrated) and #72 (the ledger now has **zero** `jax tests/test_potential.py`
entries — that set no longer exists).

**Corrected 4 counts** rather than closing them, since the work is real but
mis-sized:

| task | claimed | grep-indicated now |
|------|--------:|-------------------:|
| #125 Pdf.2 residual-slow | 5 | **28** |
| #177 qdf sampler | 8 | 6 |
| #178 parallel_map deadlock | 3 | 5 |
| #175 Staeckel angle-wrap | 2 | 5 (loose grep, unconfirmed) |

#125 is the outlier: ~6x bigger than scoped, and now the single largest
deferred bucket. #158 (~5 jax-jit) is the one count that held exactly.

Every corrected entry is labelled **grep-indicated, causes unconfirmed** — I am
not repeating the mistake of treating a count as a characterisation.

Opened **#186** to supersede #112 with the actual remaining work (2 jax +
2 jax-jit interp_potential entries), carrying forward both hard-won procedural
notes: read the docstrings first, and keep pytest's output in a file.

**Why this mattered more than picking the next task**: three consecutive targets
I opened dissolved on inspection. The re-validation cost one command for the
migration claims and closed six tasks — far better than discovering each one
individually by starting work on it.

### #1242: two more isfinite-only tests strengthened (3 total)

Continued the audit by READING candidates rather than trusting the sweep.
`test_adiabatic_actions_grad_jit` and `test_adiabatic_ecczmax_grad_jit` both
jit a C-native custom_vjp gradient and asserted only finiteness:

```python
assert numpy.isfinite(float(f(*args)))
assert numpy.all(numpy.isfinite(numpy.asarray(g(*args))))
```

Their own comment says the purpose is *"jax jit survival of the C-native
custom_vjp gradient"* — but finiteness does not test survival. A jitted
gradient that is **wrong but finite** passes. The property actually under test
is that jit must not CHANGE the value, so they now compare against the EAGER
result (rtol 1e-12 value, 1e-10 gradient). Flip-verified: a 1e-7 perturbation
of the jitted gradient fails.

That is a nice illustration of the whole class: the *comment* stated the right
property, and the *assertion* checked a much weaker one. Reading the stated
intent is what makes these findable.

#1242 now carries 3 strengthened tests (Hessian-vs-FD, and 2 jit-vs-eager),
each flip-verified. Deliberately NOT batching the rest: the classifier
over-flagged three times (missed `<=`, then `==` on shapes/orders), and
spot-checks found proper grad-vs-FD and structural tests among the flagged.
Remaining candidates need individual reading.

Also worth recording: roughly half the 54 initially-flagged tests are
LEGITIMATE — a `..._grad_degenerate` test asserting a gradient is finite at a
degenerate point is checking exactly the right property (no NaN poisoning).
A blanket conversion would have made the suite worse.

## 2026-08-03 — the jit ledger is mostly ONE root cause, not six

Went after the second parallel PR on the jit track. The 6 jit-blocking ledger
entries looked scattered; they are not. **Four of the six are
AnyAxisymmetricRazorThinDiskPotential, and all four share one root cause.**

Measured the traced Poisson integrand against scipy at R=0.5:

| \|z\| | rel. err |
|------|----------|
| 1e-1 … 2e-2 | ~1e-13 |
| 1e-3 | 7e-2 |
| 1e-5 | **2.7e4** |

The fixed-order GL path cannot resolve the a=R principal value / small-|z| sheet.
Consequences: `surfdens(forcepoisson=True)` under trace returns -4.2e-6 instead
of 0.297; a circular orbit at z=0 comes out with ecc 5.4e-6 instead of 0
(that is `test_eccentricity`, which I had not previously attributed here).

**A dead end worth recording.** The obvious move is to set
`_backend_accepts_arrays = True` (DoubleExpDisk sets it; AnyAxisym does not).
I implemented that plus `node_axis` broadcasting across all six `_gl`
integrands, verified the vectorised path bit-identical to a per-element GL loop
and ~1e-13 vs scipy — and then **reverted the whole thing**. The opt-out is
deliberate, and guarded by a negative test that names the class:

> *"AnyAxisymmetricRazorThinDisk deliberately does NOT opt in. Its traced GL
> integrand is exact to ~1e-12 at |z|=1e-2 but wrong by ~2e4x at |z|=1e-5 …
> so accepting arrays there would return a silently wrong ~0 instead of raising."*

My independent measurements reproduce that test's numbers exactly. Opting in
would have deleted a guard to reintroduce the silent wrongness it prevents.

Two process notes. The class comment already said *"GL cannot resolve the a=R
principal value / small-z sheet structure … and cannot form the z=0
Hadamard-type second-derivative"* — grepping the test file for the class name
would have saved the entire investigation (third time this pattern has cost me
a session). And one intermediate "pre-existing nan bug" I reported was an
artifact of my own probe script not enabling x64; under the real CI config all
six methods agree with scipy to ~1e-13. Corrected before it went anywhere.

**Net:** no PR, but the jit ledger is now understood as ~one numerics task
(z-aware panel split clustering nodes within a few |z| of a=R) plus the 2
`test_interp_potential` entries — rather than six unrelated problems. Filed as a
task with the measurements. That reframing is the deliverable.

### 2026-08-03 (cont.) — prototyped the AnyAxisym quadrature fix; found where it breaks

Followed up on the root cause with real numerics rather than more triage. Two
things worth keeping.

**1. Build a gold reference before scoring anything.** I spent a cycle chasing a
stubborn "2.8e-6 plateau" that appeared at |z|=1e-6 under *every* configuration
of my quadrature — insensitive to panel count, order, and subtraction. It was
not my error. Against a 40-digit mpmath reference, **scipy** — the galpy numpy
path — is the inaccurate one there:

| \|z\| | scipy vs gold |
|------|---------------|
| 1e-2 … 1e-5 | 6e-16 … 3.7e-13 |
| 1e-6 | **2.8e-6** |
| 1e-8 | **2.8e-8** |
| 0 | 6.8e-13 |

(z=0 gold is reachable despite `mp.ellipk(1)=inf` by subtracting `C/(a-R)`,
whose symmetric PV integral is exactly zero.) This is
`match-numpy-quadrature-floor` in the wild: "match numpy" is the wrong target at
extreme inputs.

**2. Graded panels fix the first derivatives and break the second ones.**
Geometrically graded panels closing onto a=R, scored against gold:

| \|z\| | 1e-4 | 1e-5 | 1e-6 | z=0 |
|------|------|------|------|-----|
| current | 6.7e-6 | 3.5e-6 | 1.6e-5 | 1.9e-5 |
| graded | 9.1e-15 | 4.5e-13 | **6.2e-12** | 4.9e-9 |

At |z|=1e-6 that is ~450,000× better than scipy. So I implemented it in the
shared `_bk_split_quad` and measured all six methods end-to-end — where it fell
over:

```
_Rzderiv @ R=0.5, z=1e-5:   graded GL = -6.309    scipy = 1.715e-4
```

Raising the per-panel order 40→60 changed the z=1e-5 errors *not at all*, which
rules out resolution. The second derivatives carry `1/((a-R)²+z²)²` — a peak of
width z and height 1/z⁴ that must cancel as a **Hadamard finite part**. The old
coarse panels missed the peak entirely; graded panels resolve it and pick up a
10⁵× spurious term. Naive refinement makes hypersingular integrands strictly
worse. Reverted.

**Net:** the fix is real but must be per-integrand, not a blanket change to the
shared helper — first derivatives are ready and validated, second derivatives
need the finite part subtracted analytically first. Second revert of the day,
both for good reasons: better to find this on a prototype than in CI. Task #187
now carries the gold-reference recipe, the measured tables, and the exact
failure mode.

### 2026-08-03 (cont.) — task-list staleness sweep against the ledger

Kept picking up stale tasks, so instead of discovering staleness one at a time I
dumped the whole 44-entry ledger and cross-checked every ledger-based task at
once. Worth doing periodically; the list rots faster than it looks.

| task | claimed | actual |
|------|---------|--------|
| #178 parallel_map torch+fork | 3 torch entries | ✅ exactly 3 `test_streamdf` |
| #186 interp_potential | 2 jax + 2 jax-jit | ✅ exactly that |
| #158 jax-jit burndown | ~5 | ✅ exactly 5 |
| #176 test-side as_numpy casts | 11 torch entries | ❌ **already landed** |
| #177 qdf sampler + Orbits obs= | 8 + 2 | ❌ only 2 qdf entries exist |
| #175 Staeckel angle sentinel | 2 entries | ❌ none remain |
| #184 coords @backendNative | to do | ❌ **already done** |

Closed five as stale/resolved (#175, #176, #177, #184, and earlier #181's
reasoning corrected). #176 was the notable one: the `as_numpy` casts are already
at every site — `test_dynamfric.py:411,443`, `test_FDMdynamfric.py:404,436`,
`test_galpypaper.py:308` — and `test_noninertial.py` even has a
`_writable_as_numpy` helper. I had been about to re-implement work that landed
days ago.

**A near-miss worth recording.** I nearly closed #125 ("28 entries") the same
way, since only ~2 sphericaldf entries appear in the xfail ledger. It refers to
a *different file*: `backend_slow_skip.txt`, where `grep -c sphericaldf` = 28,
exactly as claimed. Checked before closing. A staleness sweep that only reads
the xfail ledger will wrongly retire every slow-skip task.

**Metric worth flagging to Jo:** `backend_slow_skip.txt` holds **230 entries
(215 jax, 15 torch)** against an xfail ledger of 44. The slow-skip bucket — not
the ledger — is the dominant remaining backlog for "all of galpy runs under
backends", and it is overwhelmingly jax-eager *speed*, not correctness. Ledger
size alone understates the remaining work by roughly 5×.

**#1242 codecov.** Pulled the real numbers rather than assuming the usual
phantom: `codecov/project 86.11% (-13.85%)` while `codecov/patch` reports
"coverage not affected". A -13.85% project drop from a test-only PR is not a
regression — it is incomplete uploads with 31 shards still running. Re-checking
once CI settles rather than acting on it.

### 2026-08-03 — #1242 MERGED (2e1336c9); interp_potential characterised

**#1242 went green 152/152 and is squash-merged** into feat/backends-develop as
`2e1336c9`. The `codecov/project -13.85%` I saw earlier resolved itself once the
remaining shards uploaded — it was incomplete uploads, exactly as diagnosed, and
acting on it would have been chasing a phantom.

Pre-merge audit: test-only (3 files), so the if→xp.where array-input class is
N/A and no source helpers moved. One real finding — the two adiabatic files now
carry an identical 10-line jit-vs-eager block. Kept it: they are deliberate
mirror files already sharing ~40 lines of `_fd_grad`/`_backend_grads`
scaffolding by convention, `tests/` has no shared-helper module
(`test_backend_util.py` *tests* galpy.backend), and inventing a new tests/
convention mid-vacation is Jo's call. Filed as #188 with a concrete proposal —
~10 `_jit` files would use it.

**#186 characterised and closed: it was two unrelated causes, not one.**

*(A) 2 jax-jit = tolerance, traced-only.* Measured:

| test | max diff | bound | over |
|------|----------|-------|------|
| `..._use_c` | 5.596e-14 | 1e-14 | 5.6× |
| `..._force_use_c` | 6.621e-13 | 1e-13 | 6.6× |

The `use_c` max is invisible in the junitxml (numpy truncates the printed
array); I surfaced it by temporarily putting `nanmax` in the assertion message
and reverting. Cause: the test compares a `use_c=True` grid against a
`use_c=False` grid, and under `--jit` the python side goes through traced jax
where XLA fusion reassociates the sums differently from C. That is irreducible
floating-point reassociation at 1e-14 on order-1 values — not a defect, and both
pass eagerly. → tolerance batch #171, category B (traced-only), which Jo already
approved provided bounds stay stringent. 5.6×/6.6× is minor.

*(B) 2 jax eager = speed.* Both simply TIMEOUT at >600s. Nothing to fix in the
interpolation code; they belong with the jax-eager speed work (#67/#79) and the
230-entry slow-skip bucket, not a correctness PR.

Ledger unchanged at 44 (this PR was test-strengthening, not burndown).

### 2026-08-03 — PR #1243: traced-only bound for the interp_potential grid compare

Turned the (A) half of the #186 characterisation into a PR. **Ledger 44 → 42,
jax-jit 5 → 3.**

The two `use_c=True` vs `use_c=False` grid comparisons now take a bound that
widens only when `jit_mode()` is on (30×, ~5× headroom over measured); the
eager/numpy bounds are byte-for-byte unchanged.

Two things worth recording.

**The ledger had already done this analysis, better than mine.** Its comment
block reproduced my exact numbers (5.596e-14, 6.621e-13) *and* localised the
cause further than I had: of MWPotential's three components only NFW moves, and
its jax eager value is bit-identical to numpy while its jitted value differs,
reproducibly — XLA compiles NFW's `log1p(r/a)/(r/a)` differently from how it
evaluates it eagerly. I folded that into the test-file comment rather than
losing it with the removed entries. Reading the ledger comments first (the
`read-code-docs-before-classifying` lesson) would have saved me the independent
re-derivation, though it did serve as a clean confirmation.

**It also said "loosening one needs Jo's sign-off."** That comment predates the
tolerance-batch approval of category B (traced-only, stringent). This is
squarely that category at 5.6×/6.6×, so I landed it and said so explicitly in
the PR — with the note that since the eager path is untouched it reverses
trivially if Jo wants it the other way.

Verification was the standard three-way: passes traced; **fails traced with the
factor forced to 1.0** (so the widening is what carries it); numpy and jax-eager
still pass on the untouched tight bound.

One process note: I used `git stash` to check whether a ruff warning was
pre-existing, which `git-stash-is-repo-wide-not-worktree` warns against. It
happened to round-trip correctly — but the stash list turned out to already hold
a WIP from `feat/native-grid-spline`, so a mis-ordered pop would have pulled
another branch's work into this PR. Verified the file list and markers
afterwards. Should have used a second worktree.

### 2026-08-03 — AnyAxisym: the real defect is cancellation in `1-m`, not the quadrature

Kept digging on #187 while #1243's CI queued, and the "z-aware quadrature"
framing turned out to be wrong. The actual defect is a numerical one, upstream
of any quadrature choice.

`_bk_m_aRz` computes `m = 4aR/((a+R)²+z²)`. Near a=R that makes `1-m`
catastrophically cancelling. Measured at R=0.5, z=0:

| \|a−R\|/R | `1−m` computed | exact | cap fires? |
|------|----------------|-------|-----------|
| 1e-7 | 2.665e-15 | 2.500e-15 | no — already 6.6% wrong |
| 1e-8 | **0.000e+00** | 2.500e-17 | **yes** |
| 1e-10 | **0.000e+00** | 2.500e-21 | **yes** |

Below 1e-8, `m` rounds to exactly 1.0 and the `_ELLIP_M_CAP = 1-1e-15` clamp
fires — and the clamp's activation boundary *moves with R*, turning the
quadrature error into a step function of R. That single fact explains three
things I had been treating as separate puzzles:

* **why z=0 fails.** `test_2ndDeriv` finite-differences Rforce with `dr=1e-8`,
  and the error jitters at exactly that scale. The FD returns **+0.109** where
  the true R2deriv is **−2.579** — the wrong sign.
* **why deeper grading made it worse** (k=8 → 1.3e-5, k=10 → 4.3e-4, k=14 → nan):
  more nodes land inside the clamped region.
* **why analytic PV subtraction didn't help** (tested: +0.13, still fails). The
  huge cancelling values were never the problem — I had the wrong culprit and
  the experiment said so.

**The fix**, and it lands in the right place: compute the complementary
parameter cancellation-free, `1-m = ((a-R)²+z²)/((a+R)²+z²)`, and pass *that* to
the elliptic integrals. galpy's fallback is AGM, which starts from
`b0 = sqrt(1-m)`, so an `m1`-taking entry point is trivially exact. That is a
general special-function capability → `galpy/backend/special`, not the potential
file. `_ELLIP_M_CAP` then disappears entirely, since `m1 ≥ 0` by construction.

**One more result worth keeping.** R2deriv via `jax.grad` of the graded Rforce —
AD handles the moving panel limits exactly — is excellent and sidesteps the
Hadamard finite part completely: 3.6e-16 at z=0.15, 4.5e-13 at 1e-3, 2.4e-7 at
z=0. So the second derivatives should be *differentiated*, never integrated from
their own hypersingular integrands (which gave 1e5× errors).

Not claiming the four entries flip until measured — the m1 fix has to land
first, then re-measure. But #187 is now a concrete numerical bug with a known
remedy rather than an open-ended numerics project.

### 2026-08-03 — the `1-m` fix works, and uncovers a second defect underneath

Implemented the `m1` fix from the previous entry and measured it. The diagnosis
was right, and the result is still not shippable — for a reason I had not
credited.

**Defect 1 confirmed and fixed.** Added `ellipkm1` to `galpy/backend/special`,
mirroring scipy's own name (scipy has it; jax and torch don't, so both go in
`_NATIVE_MISSING`). The AGM fallback starts from `b0 = sqrt(m1)` exactly;
`_agm_core` was extracted so the existing m-entry point's arithmetic is
untouched op-for-op. Verified against scipy on **both** jax and torch: matches
to all printed digits from m1=5e-1 down to 1e-20, exactly where the old
`ellipk(1-m1)` degrades (18.187 vs 18.198 at 2.5e-15) and then returns `inf`.

With `_bk_m1_aRz` feeding it a closed-form `m1 = ((a-R)²+z²)/((a+R)²+z²)`, the
z=0 jitter is **gone**: FD-of-Rforce at z=0 gives −2.5786 against a true
−2.5788, where before it gave **+0.109 — the wrong sign**.

**Defect 2, which the clamp was hiding.** Removing `_ELLIP_M_CAP` made small-z
*worse*:

| | z=1e-3 | z=1e-5 |
|---|--------|--------|
| R2deriv | 1.4e+0 (was ~1e-12) | 2.0e+4 |
| z2deriv | 2.7e+0 | 2.7e+4 |
| zforce | — | 9.4e-1 |

The clamp had been load-bearing: truncating K bounded the peaked integrands.
With `m1` exact, K is correctly large and the `1/((a-R)²+z²)²` integrands
correctly blow up — and plain fixed-order GL sampling them returns nonsense. The
clamp was masking the hypersingularity, not fixing it.

So `m1` is necessary but **not sufficient**, and each piece is individually a
regression. The three have to land together:

1. `m1` / `ellipkm1` — done, patch saved to the scratchpad;
2. graded panels for the peaked first-derivative integrands (validated vs gold:
   9.1e-15 @1e-4, 6.2e-12 @1e-6, beating scipy's own 2.8e-6 there);
3. second derivatives by **AD of the first derivatives**, never from their own
   hypersingular integrands (3.6e-16 @0.15, 2.4e-7 @z=0 vs 1e5× errors when
   integrated directly).

Reverted rather than shipping a piece. Third revert in this area today, and each
one came from measuring instead of assuming — the clamp in particular looked
like dead defensive code and turned out to be holding up two of the six methods.

### 2026-08-03 — #1243 MERGED; the loop had silently stopped

**#1243 merged** into feat/backends-develop. Ledger **44 → 42**, jax-jit 5 → 3.
CI finished 152/152 and `codecov/project` cleared itself once the last shards
uploaded — the same incomplete-upload phantom #1242 showed, correctly diagnosed
both times by reading `codecov/patch` ("coverage not affected") rather than
pattern-matching.

**Process failure worth recording:** Jo asked whether the loop was still on, and
it was not — `CronList` returned *No scheduled jobs*. So for some stretch I was
only running when pinged, which is exactly the "stalling" that was visible from
the outside. Re-armed as job `c46cf2e8` (every 55 min, session-only, 7-day
expiry). **Check `CronList` when a wake seems overdue rather than assuming the
schedule is live** — the loop dying is silent and looks identical to being slow.

### 2026-08-03 — AnyAxisym three-part fix: big numbers, zero entries cleared

Assembled all three pieces on `feat/anyaxisym-m1-quadrature` (WIP `c83ee33d`).
The accuracy gains are large and the ledger gains are, so far, nil — worth
separating those two facts.

**Against a verified-pristine baseline** (taken from `HEAD~1`, which `git diff`
confirms is identical to `origin/feat/backends-develop`):

| | R2deriv | z2deriv | Rzderiv |
|---|---------|---------|---------|
| z=1e-3 | 1.4e+00 → **2.9e-13** | 2.7e+00 → **2.7e-12** | 1.9e-02 → **1.8e-11** |
| z=1e-5 | 2.0e+04 → **4.2e-09** | 2.7e+04 → **4.1e-08** | 1.5e+04 → 3.7e+04 |

plus Rforce at z=0 going 1.9e-5 → **2.77e-9** against mpmath gold. Nine to
twelve orders of magnitude on the second derivatives.

Two things that measurement corrected:

* **Rzderiv@1e-5 is not a regression.** I flagged it as possibly one and refused
  to guess; the baseline says it was already 1.5e+04. It survives because
  Rzderiv = d(Rforce)/dz is O(z) and tiny, while Rforce's own small-z error
  differentiates into something much larger. Needs z-adaptive panels.
* **These GL paths were never exercised eagerly.** `_bk_dispatch` sends concrete
  scalars to scipy, so `_R2deriv_gl` only runs under a trace — which is exactly
  why a 1.4e+00 relative error sat there unnoticed and ledgered.

**But 0 of 4 ledger entries clear:**

* `test_2ndDeriv` — still fails at z=0, though diff is 3.61 vs 4.8e+04 before.
  R2deriv itself is now correct; what fails is Rforce's smoothness in R at the
  1e-8 FD scale. Retry analytic PV subtraction — the earlier negative result
  predates m1 and was confounded by the clamp.
* `test_poisson_surfdens` — still the array-input `TypeError`. Needs the
  deliberate opt-out reversed, which is only honest once the accuracy that
  opt-out cites is actually fixed. It isn't yet (4.0e-6 at |z|=1e-5).
* `test_eccentricity` — now **nan** under pytest `--jit`, but it does NOT
  reproduce in a script replicating the same setup, times and integrator
  (`e()=0.0`, passing), and a scan of all six methods over R∈[0.01,10] at z=0
  and 1e-8 finds nothing non-finite. Unattributed, so I am treating it as a
  possible regression until explained — and it has to be diagnosed *inside* the
  pytest run, not from an isolated script.

Not shipping. The honest summary is that the numerics got dramatically better
and the tests still don't pass, which is a real distinction and not one to blur.

### 2026-08-03 — the nan IS my regression; I nearly dismissed it on bad evidence

Last entry I recorded the `test_eccentricity` nan as "unattributed" because it
would not reproduce in a standalone script replicating the same potentials,
times and integrator. **That reasoning was wrong.** Comparing like-for-like
*under pytest*, which is the only context that counts:

```
baseline: 5.42463803e-06   (finite; exactly the ledger's recorded value)
mine:     nan
```

Consistently. The standalone script is a different context and gives 0.0 or
5.42e-06 regardless of version — so it was never evidence of anything. This is
`handpicked-nodeid-batch-is-not-a-context` wearing a different hat: I built a
"doesn't reproduce" conclusion on a harness I had constructed myself rather than
the one CI runs. Had I acted on it, I would have shipped a nan-producing change.

**Bisected one step.** Reverting *only*
`AnyAxisymmetricRazorThinDiskPotential.py` while keeping the `backend/special`
`ellipkm1` work restores the baseline 5.42e-06. So the cause is in the potential
file; the special-function piece is clean.

A useful constraint for the next step: instrumentation shows the orbit
integration calls only `_Rforce_numpy` (58×, code I never touched), and the GL
path runs **exactly once — inside `normalize()`**. So the prime suspect is what
that single traced `_Rforce_gl` call does to `_amp` under the pytest harness. I
measured `_amp` identical between versions (0.88668098) — but in a *script*, so
by the lesson above that measurement has to be redone under pytest before it
means anything.

Remaining bisection, all under pytest: revert (a) the three AD second
derivatives, (b) the graded panels, (c) the m1 wiring, one at a time.

### 2026-08-03 — bisection lands on the graded panels; and a measurement trap explained

**Result: the graded panels cause the nan.** Bisected under pytest (the only
context that counts here):

* revert the whole potential file, keep `backend/special` → baseline 5.42e-06 ✅
* revert *only* `_bk_split_quad` (graded panels off, m1 + AD still on) → baseline
  5.42e-06 ✅

So `m1`/`ellipkm1` and the AD second derivatives are clean; the graded-panel
quadrature is what produces nan. That also fits the call trace: the orbit
integration only calls `_Rforce_numpy` (58×, untouched), and the GL path runs
exactly once, inside `normalize()`.

**And the confusion from the last two entries is fully explained.** The
"it doesn't reproduce outside pytest" runs were never running my code.
`python /path/to/script.py` puts the **script's** directory on `sys.path`, not
the cwd — so every scratchpad-file probe imported the editable install, which
resolves to **`/var/tmp/streamtorch/galpy`**, a different worktree. Proof:

```
cd /var/tmp/typeleaks && python /tmp/.../r0chk3.py                 -> VERSION: BASELINE
cd /var/tmp/typeleaks && PYTHONPATH=/var/tmp/typeleaks python …    -> VERSION: GRADED(mine)
python -c 'import galpy; print(galpy.__file__)'   (cwd=/tmp)       -> /var/tmp/streamtorch/galpy/__init__.py
```

That invalidates the `_amp` comparison, the side-by-side integration check and
the "5.42e-06 twice" loop — all file-based. The heredoc measurements (accuracy
tables, mpmath gold comparisons, the first loop that *did* show nan) and every
pytest run used the correct galpy, so those conclusions stand.

The tell was there and I explained it away: an A/B in which both arms printed
*identical* numbers to 16 digits. Two versions of different code do not do that.
I should have stopped at that and checked provenance instead of hypothesising
about pycache and trace caches. Memory `galpy-editable-install-import-shadow`
rewritten with the sys.path table and a mandatory provenance snippet.

### 2026-08-03 — grading is load-bearing too, so it must be fixed not dropped

Followed the bisection with the obvious engineering move: drop the culprit
(graded panels), keep the two clean pieces (`m1` + AD second derivatives), and
see what survives. Measured that variant against scipy (heredoc, provenance
printed):

| | evaluate | Rforce | zforce | R2deriv | z2deriv | Rzderiv |
|---|---|---|---|---|---|---|
| z=1e-3 | 5.3e-08 | 2.3e-07 | 2.3e-04 | **2.0e-06** | 2.7e+00 | 1.9e-02 |
| z=1e-5 | 1.6e-05 | 3.5e-06 | 9.4e-01 | **4.7e-06** | 2.7e+04 | 1.5e+04 |

Only **R2deriv** improves (1.4e+00 → 2.0e-06 at z=1e-3), because it is
`-grad_R(Rforce)` and Rforce is decent. `z2deriv` and `Rzderiv` are unchanged
from baseline, because they are `grad_z` of `zforce`/`Rforce` — and without
grading `zforce` is itself 9.4e-01 wrong at z=1e-5 (it was **1.2e-12** with
grading). Rforce at z=0 also falls back to 1.86e-05, i.e. no better than
baseline.

So the graded panels are load-bearing for four of the six methods. Dropping them
is not an option; they have to be fixed. That is the third time in this potential
that something which looked removable turned out to be doing real work (the
`_ELLIP_M_CAP` clamp, the `git stash` A/B, now the grading).

State: WIP `c83ee33d` restored intact on `feat/anyaxisym-m1-quadrature`; nothing
pushed, no ledger edits. Next: find why 12 extra panels produce nan under a
trace. Note the nan enters the *integrator state* (`_Rforce_numpy` gets fed
R=nan; its first four calls return the correct -1.127810683), and `_amp` is
finite and differs from baseline only at 8e-6 — so neither the potential's numpy
path nor the amplitude explains it yet.

### 2026-08-03 — the nan is a PRE-EXISTING numpy bug that my change merely steps on

Chased the graded-panel nan to ground. It is not a backend bug at all.

```
_zforce_numpy(1.0, 0)        = 0.0     <- guarded, fine
_zforce_numpy(1.0, 1e-30)    = nan
_zforce_numpy(1.0, 2.26e-25) = nan
_zforce_numpy(1.0, 1e-20)    = nan
_zforce_numpy(1.0, 1e-16)    = nan
_zforce_numpy(1.0, 1e-12)    = nan
_zforce_numpy(1.0, 1e-8)     = -5.164079117306708e-09   <- fine
```

**Byte-identical on pristine `feat/backends-develop`.** There is a nan band at
roughly 1e-30 ≤ |z| ≤ 1e-12 in the plain numpy/scipy path of
`AnyAxisymmetricRazorThinDiskPotential`, and it very likely affects `main`.

How it produces the `test_eccentricity` nan: an orbit started at z=0, vz=0 stays
at z=0 analytically, but the ODE solver wanders to z ~ 1e-25 through roundoff.
Whether it lands *inside* the band depends on the exact step sequence, which
depends on `_amp` — and #187 shifts `_amp` by 8e-6. So the WIP does not cause
the nan, it **exposes** it; the baseline is simply lucky.

Getting here took three wrong turns, all of which the instrumentation eventually
corrected: the guards (R=0 gives nan in *both* versions — ruled out), `_amp`
(finite in both, 8e-6 apart — ruled out), and the potential's traced path
(`jax.jit(_Rforce_gl)` at R=1 is correct). The find came from widening the
instrumentation from `_Rforce_numpy` alone to all three numpy entry points and
asking which returns a non-finite value *first* — index 57 of 116.

Filed as #189, with the note that it belongs on **main**, not feat/backends, and
that galpy's own suite never probes that |z| band, which is why it has sat there.
Likely fix: extend the `z==0` special case to a small-|z| regime where the force
is O(z), or make the numpy quadrature range-aware the way the traced path now is.

This also reframes #187: the graded panels are not wrong, and the "regression"
is a collision with a pre-existing landmine. Sequence is now #189 first, then
re-test #187.

### 2026-08-03 — PR #1245: the nan band, root-caused and fixed

Root cause of #189, exactly as the structure suggested. The five numpy
integrands hand `m = 4aR/((a+R)^2+z^2)` straight to scipy's `ellipe`/`ellipk`.
That is ≤ 1 analytically, but for tiny |z| the z² is negligible against (a+R)²
and rounding tips it over. Caught the actual call by spying on the elliptic
functions during a failing evaluation:

```
z=1e-08 : 1431 calls, max m = 0.9999999999999998, 0 nan
z=1e-25 : 2103 calls, max m = 1.0000000000000002, 2 nan
          first nan-producing call: ('E', 1.0000000000000002, nan)
```

Fix is `numpy.minimum(..., 1.0)` on all five — the backend/GL path already had
exactly this guard as `_ELLIP_M_CAP`; the numpy path never did. **Byte-identical
away from the band** (verified on a 6×20 method/R/z grid), and inside the former
band the values are now not merely finite but physically right: `Fz/z` constant
to 1e-9 across 1e-30…1e-8, the O(z) limit.

**PR #1245** into feat/backends-develop, flagged in both the commit and the PR
body as a `main` bug to cherry-pick — it is not backend-related at all.

Two process notes:

* My first version of the regression test **failed with the fix applied**, and
  the fix was right — I had included z=1e-6 in the linearity assertion, which is
  past the O(z) regime (`Fz/z` ≈ -4e5 there). Fixed the test, not the code, and
  scoped the assertion to |z| ≤ 1e-8 where the claim actually holds.
* I lost the uncommitted fix once to a `git stash -- <file>` / `checkout` /
  `stash drop` dance — the repo-wide-stash trap I already have a memory about.
  The pre-existing `feat/native-grid-spline` stash survived (I checked before
  re-applying), and re-applying was 5 mechanical lines. Still: no stash games in
  a shared worktree.

### 2026-08-03 — #187 clears its first ledger entry (gated on #1245)

#1245's CI is all-queued, so I pre-verified what #187 buys once it lands:
cherry-picked the nan fix onto `feat/anyaxisym-m1-quadrature` and re-ran the
four AnyAxisym ledger entries under pytest.

```
test_orbit.py::test_eccentricity                PASS (38s)   <-- CLEARED
test_2ndDeriv_potential[AnyAxisym]              FAIL
test_poisson_surfdens_potential[AnyAxisym]      FAIL (array-input TypeError)
```

**And I ran the control**, which is the part that matters for attribution: with
the #1245 nan fix *alone*, `test_eccentricity` still fails at the original
5.42463803e-06. So #1245 is a prerequisite that unmasks the entry, and #187's
quadrature work is what actually clears it — Rforce at z=0 goes 1.9e-5 → 2.77e-9
against mpmath gold, and the eccentricity of a circular orbit follows.

Worth stating plainly because the temptation was to credit the nan fix: it
removes the nan, it does not make the orbit circular. Two separate defects
stacked on the same test, and only running both arms separates them.

Remaining on #187: rebase once #1245 merges (the cherry-pick drops out), then
`test_2ndDeriv` (still fails at z=0, diff 3.61 vs 4.8e+04 before) and the two
`poisson_surfdens` entries (need the deliberate array-input opt-out reversed,
which is only honest once the accuracy that opt-out cites is fixed). If those
drag, shipping the rebase + the one cleared entry is a legitimate smaller PR.

### 2026-08-03 — #1245 back to DRAFT: my fix was wrong, and the bug has layers

Caught this before merge, which is the good part; the rest is a correction.

**The physics I got wrong.** Just off the plane a razor-thin disk has
`Fz(z->0+) -> -2 pi Sigma(R)`, a **constant**. My regression test asserted
`Fz/z` is constant — O(z) — and it passed *only because the values were
garbage*. A correct implementation would have failed it. That is exactly the
"green ≠ meaningful" failure mode, authored by me, in a test written to guard a
fix.

**The bug has at least three layers**, all measured:

1. `m = 4aR/((a+R)^2+z^2)` rounds just above 1 for tiny |z| → scipy's
   `ellipe`/`ellipk` return nan (the integrand asks for `1.0000000000000002`).
2. Clamping to *exactly* 1.0 — my first fix — is wrong: `ellipk(1) = +inf`
   (while `ellipe(1) = 1`), so the ellipk-using integrands trade the nan for an
   inf and quad still returns nan. The backend's `_ELLIP_M_CAP = 1 - 1e-15`
   exists for precisely this, and I should have copied it rather than inventing.
3. Independently, below |z| ~ 1e-6 the a=R peak is too narrow for scipy's quad
   to find and it silently returns ~0 **with the wrong sign** — erratically:
   correct at 3e-6, garbage at 1e-6, garbage at 3e-7, correct again at 1e-7,
   garbage at 1e-8.

**What I tried and why each failed:** explicit break points at R±10z made
quadpack fail on points ~1e-7 apart and *introduced* a nan in `evaluate` the
baseline never had (removed); the analytic small-|z| branch
`Fz = -2 pi Sigma(R) sign(z)` genuinely works — exact below the threshold,
antisymmetric, continuous to ~1e-6 — but I set the threshold one decade too low,
so the breakdown region still takes the broken branch.

**Why I stopped rather than tune it again.** Raising the threshold to 1e-5 would
probably go green. That would be a fourth guessed constant on a main-branch
numerics path after three wrong attempts, trading a ~1e-5 discontinuity at the
join for correctness below it. Threshold vs blend vs a proper series expansion is
a design call for Jo, not something to settle by iterating until CI is green.

PR #1245 is a draft with a comment stating the error plainly. WIP committed
(1c3b8327) so nothing is lost. #187 stays blocked behind it.

### 2026-08-03 — PR #1246: found completed work that had never been pushed

With #1245 parked on a design decision and #187 behind it, I went looking for
other actionable work rather than grinding on AnyAxisym — and found task #182's
fix sitting finished and *unpushed* on `fix/scipy118-spline-pickle`.

**The trap:** `git branch -r --contains` said `origin/fix/scipy118-spline-pickle`
had the commit, and `git log origin/feat/backends-develop..<branch>` showed it 1
commit ahead. Both were satisfied by a **stale remote-tracking ref** — plain
`git fetch` does not prune, so a branch deleted (or never pushed) on GitHub still
appears under `origin/`. GitHub answered the question definitively:
`gh api repos/.../branches/<name>` → **404**. Pushed it, then the PR opened fine.
Lesson: `git branch -r` is not evidence a branch exists on the remote; ask the
API (or `git fetch --prune`).

**The fix itself** (#1246): scipy 1.18 caches the array namespace — a *module* —
on `PPoly`/`BPoly` as `_xp`, so anything holding one raises "cannot pickle
'module' object". `MultipoleExpansionPotential` holds `BPoly` directly and
`DiskMultipoleExpansionPotential` through its `_me` sub-potential, so one
`__getstate__`/`__setstate__` covers both: drop the scipy object on the way out,
rebuild through the public constructor on the way back. Stripping `_xp` alone
does not work — scipy never recreates it and the restored spline then raises
`AttributeError`. The test change *removes* an exclusion, restoring coverage.

**Stated caveat rather than glossed:** local scipy is 1.17.1, so I cannot
reproduce the 1.18 failure here. The 60 pickle tests passing locally shows no
regression on 1.17 — not that 1.18 is fixed. CI decides that, and the PR says so.

Also audited before opening (test-only-adjacent but still): no if→xp.where
array-input class, no duplication, helpers module-local to the file that owns the
splines. Flagged for cherry-pick to main.

### 2026-08-03 — swept the unpushed local branches; two more tasks were stale

After #1246 I checked whether other local branches held ready work, since that is
how #1246 was found. Result: they held *superseded* work, not ready work.

| branch | ahead of develop | on GitHub | verdict |
|--------|------------------|-----------|---------|
| `fix/scipy118-spline-pickle` | 1 | **no** | genuinely ready → **PR #1246** |
| `fix/qdf-sampler-boundary` | 3 | no | superseded (landed as #1231) |
| `fix/picklable-potentials` | 4 | no | superseded (landed as #1230) |

"Ahead of develop" is a weak signal — a branch whose content was squash-merged
still shows commits ahead, because the squash has a different SHA. The real check
is whether the *content* is on develop:

* `fix/qdf-sampler-boundary` — develop already carries both the sampler boundary
  cast **and** the follow-up `fmin_powell` objective cast
  (`quasiisothermaldf.py:2181-2190`), which I expected to be the one novel bit.
  Nothing left. Closed #180.
* `fix/picklable-potentials` — develop already has the pickle round-trip test and
  the KuijkenDubinski `__getstate__`/`__setstate__`. Closed #179.

Also corrected a count while checking: `grep -c qdf` on the ledger returns 6, but
5 of those are comment lines; only **2** are entries (jax+torch
`test_call_diffinoutputs`), already recorded as by-design. Grep counts on the
ledger must exclude `^#` — the same trap as the earlier staleness sweep.

Net: four tasks closed as stale across the last two sweeps (#175, #176, #177,
#179, #180, #184), one real PR recovered (#1246). The task list rots faster than
it looks, and the periodic sweep keeps paying for itself.

### 2026-08-03 — #117: dropping the guard was half a fix; and freeing the CI queue

**The forced-backend gate earned its keep.** Removing the Pdf.1 data-guards
(`get_namespace(R) if is_backend_array(R) else numpy` → `get_namespace(R)`)
passed numpy 134/134 and broke forced torch **125/134**, every one:

```
TypeError: exp(): argument 'input' (position 1) must be Tensor, not float
```

Under a forced backend `get_namespace` correctly returns torch, but `R` is still
a plain Python float and `torch.exp` rejects it. **The data-guard was standing in
for coercion** — removing it without adding coercion just relocates the
breakage. Exactly the case `forced-backend-gate-before-push` exists for: numpy
alone says nothing here.

**The complete fix needs two pieces, and the first is inert without the second:**

1. `@backend_input("R")` on each entry point (4 methods × 2 classes in
   `surfaceSigmaProfile.py`, `sigmar`/`sigmalos` in `jeans.py`);
2. `_backend_compatible = True` on the base class.

I added (1), re-smoke-tested, and `R` was *still* a float inside the wrapper.
Reading `_input.py` showed why: coercion is gated on `_backend_ready(args[0])`,
an alias of `is_backend_compatible`. `sphericaldf` declares it (line 231);
`surfaceSigmaProfile` never did. With both, forced torch takes a plain float and
returns `tensor(0.0183)` against numpy's `0.01831563888873418`.

Not assuming the rest: forced **jax** showed 5 `test_jeans` failures that are
*numerical* ("Radial sigma ... incorrect for LogarithmicHalo"), not `TypeError`.
Different cause, diagnosed separately before this ships. Also flagged that
`jeans.sigmar`'s gate asks about the **potential** (args[0]), not the df — worth
confirming that is intended for module-level functions.

**Queue management.** #1246 had sat ~40 min fully queued while #1245 — a *draft*
I am not merging, blocked on Jo's threshold call — was still burning runners from
an hour earlier. Cancelled its 8 runs. The project is runner-constrained, so a
parked draft holding capacity is pure waste; PR runs on a draft are safe to
cancel (unlike same-SHA push runs, which block merge).

### 2026-08-03 — #117 scoped down: jeans' guard was hiding an accuracy gap

Diagnosed the 5 forced-jax `test_jeans` failures rather than assuming the
coercion fixed them. **They were my regression** — baseline forced-jax
`test_jeans` is 5 tests / **0 failed**, mine was 5/5 failed.

Why: `jeans.sigmar`/`sigmalos` are a *dual path*, not a numpy island —
`if xp is numpy:` → `scipy.integrate.quad` (byte-identical), else a backend
quadrature. The old guard meant a plain Python float took the **scipy** branch
even under a forced backend. Coercing `r` correctly routes it to the **backend**
branch — which is the intent — and that exposed the backend Jeans quadrature as
not accurate enough ("Radial sigma ... incorrect for LogarithmicHalo", numerical,
not a TypeError).

So the guard was doing two different jobs in two different files:

* `surfaceSigmaProfile` — standing in for **coercion**. Fixed properly with
  `@backend_input` + `_backend_compatible`; numpy 134/134, forced torch smoke
  clean.
* `jeans` — standing in for an **accuracy fallback**. Cannot be removed until the
  backend quadrature is good enough. Guard restored; filed as **#190** with the
  measurements.

Scoped #117 to `surfaceSigmaProfile` only. Reverting the jeans half restored
forced-jax `test_jeans` to 5/5 passing, confirming the split is exactly right.

Two things worth carrying: (a) "remove the numpy island" is not one uniform
refactor — each guard has to be read for *what it is standing in for*; (b) I only
caught this because I ran the baseline arm instead of trusting that a failure in
new code is a pre-existing failure.

Also cancelled #1245's 8 CI runs — a draft I am not merging, parked on Jo's
threshold call, was holding runners while #1246 sat fully queued for ~40 min.
#1246 went from 0 to 11 checks in progress immediately after.

### 2026-08-03 — inventoried every remaining data-guard, split by what it means

With #117 and #190 having shown the same idiom means two different things, took a
full inventory so the rest of the island work is not rediscovered one file at a
time. Recorded on #150, which was previously a pending task with no detail.

**(A) namespace guards** — `xp = get_namespace(x) if is_backend_array(x) else numpy`
— **26 sites across 11 files**, the #117 pattern. Each is standing in for either
coercion (fix: `@backend_input` + `_backend_compatible`) or an accuracy fallback
(fix: improve the backend rule first — a numerics task). Biggest concentrations:
`actionAngleStaeckelGrid.py` (5), `evolveddiskdf.py` (4, = task #143),
`util/coords.py` (3), `verticalPotential.py` (3), `actionAngleStaeckel.py` (3).

**(B) cast guards** — `x if is_backend_array(x) else numpy.<op>(x)` — 3 sites
(`constantbetadf:656` ascontiguousarray, `diskdf:239` real,
`backend/interpolate:811` asarray). These are **not** islands: they apply a
numpy-only op to numpy input and leave backend arrays alone. Leave them. Worth
separating explicitly, because a naive grep for `is_backend_array ... else numpy`
lumps them in and would have inflated the roadmap by 3 files of non-work.

Also carried the method into #150: numpy-only validation is silent on this class
(134/134 vs 125/134), always run forced torch AND jax, and run the baseline arm
before calling a new failure pre-existing.

### 2026-08-03 — #117 is a migration, not a guard removal; my scope was wrong twice

Forced-torch progression on `test_diskdf` + `test_jeans`:

```
guards simply dropped                    125 / 134 failed
+ @backend_input + _backend_compatible    52 / 134 failed
numpy throughout                         134 / 134 PASS
```

52 is much better than 125, but the survivors are **two new causes**, not the
original one:

1. `TypeError: log(): argument 'input' must be Tensor, not float` — more
   undecorated entry points. `exp` was simply the first one the tests reached;
   this needs a systematic pass over the public surface, not another one-off.
2. `IndexError: too many indices for array: array is 0-dimensional` — coercion
   turns a scalar into a 0-d **array**, and the `diskdf` code downstream indexes
   it. That is the numpy-scalar-vs-0-d-array distinction biting again.

The second one is the structural finding: **the Pdf.1 leaf cannot be
de-islanded independently of `diskdf`.** Either migrate diskdf's consumption
sites to accept 0-d backend arrays (what the directive actually wants), or have
the leaf exit-cast back to a scalar (smaller, but leaves the island half
standing). That is a decision to make before writing more code, so I stopped
rather than patching toward green.

I have now mis-scoped this twice — first "drop 6 guards", then "add a decorator".
Recording it plainly on #117 because the pattern is the interesting part: each
fix revealed the next layer, and each layer was only visible under a forced
backend.

**Process slip:** one measurement was silently lost because I backgrounded it
with a trailing `&` inside a foreground tool call — it died with the parent shell
and wrote no junitxml, and I only noticed when the file was absent. Use the
tool's `run_in_background`, never `&`.

### 2026-08-03 — #117 paused with the baseline nailed down; STATUS.md refreshed

Got the number I had been missing. Baseline forced-torch on `test_diskdf` +
`test_jeans` is **0 failures** — now COMPLETE and confirmed (134 tests, 0 failed,
exit 0), not extrapolated from the partial run —
so the 47 remaining failures are entirely mine, not a pre-existing floor.
Corroborated independently: `backend_xfail.txt` has no torch entries for these
files, `backend_skip.txt` has none, and the slow-skips are jax-only — CI runs
`test_diskdf` under torch and expects it green.

Full progression:

```
baseline                                    0 failed
guards simply dropped                     125 failed
+ @backend_input + _backend_compatible     52 failed
+ jeans guard restored (reduced scope)     47 failed
numpy, throughout                           0 failed
```

Paused #117 rather than patch toward green. The 9 `IndexError`s are the
structural blocker — coercion makes a scalar a 0-d **array** and `diskdf` indexes
it — so the leaf cannot be de-islanded independently of its consumer. The choice
is (a) migrate diskdf's consumption sites, which is what the directive wants and
is much the larger job, or (b) exit-cast at the leaf, which is small but leaves
the island half standing. That is a scoping decision, not more code.

Also **rewrote STATUS.md**, which was two days stale and still reporting ledger
101 against the current 42. Since Jo said he will ask for summaries rather than
read the trace, a stale snapshot is worse than none. It now carries: the 42
split, the point that `backend_slow_skip.txt` (230) is 5× the ledger and is the
real remaining backlog, the two decisions waiting on him (#1245 threshold, #171
tolerances), the two non-backend bugs to cherry-pick to main, and the method
notes that actually earned their keep this week.

### 2026-08-03 — a real jit-progress metric (#185), and the first number

Built the traced-fraction metric while #1246 waited on the queue. It is a small
pytest plugin that patches the two decision points in the `@backend_input`
boundary and counts what actually happens:

* `_backend_ready(args[0])` — the temporary opt-in gate; how many boundary calls
  are turned away because the object is not migrated;
* `traced_call(...)` — of the calls that get past the gate, how many actually
  trace versus return `NOT_TRACED`.

First measurement, `test_potential.py` under `--backend jax --jit`
(AnyAxisym/Miyamoto/NFW subset):

```
boundary calls on a backend namespace : 18739
  passed the _backend_ready gate      : 18715   (99.9%)
  gated OUT (object not migrated)     :    24
reached traced_call                   : 18715
  actually TRACED                     : 18499   (98.8% of those)
  returned NOT_TRACED                 :   216
end-to-end traced fraction            : 98.7%
```

This is the metric the jit work has been missing. Pass counts do not measure it —
a C-method test bypasses the trace entirely and still passes green, so "the jit
shard is green" and "the code is being traced" are different claims. 98.7% for
potentials is a genuinely good number and, importantly, it is now *measurable*
rather than asserted.

The plugin lives in the scratchpad, not the repo — it patches internals, so it is
a measurement tool rather than something to ship. If it proves useful across
several files it should be hardened into `tests/` behind a `--jit-metric` flag.

`test_orbit.py` is the interesting contrast (memory says it is largely NOT
traced) — that run exceeded the foreground window and is now in the background.

### 2026-08-03 (later) — the orbit contrast, and what the metric does *not* say

`test_orbit.py` under `--backend jax --jit` (subset: eccentricity /
energy-jacobi / analytic-ecc):

```
boundary calls on a backend namespace : 356737
  passed the _backend_ready gate      : 346866   (97.2%)
  gated OUT (object not migrated)     :  9871
  actually TRACED                     : 346242   (99.8% of those)
end-to-end traced fraction            : 97.1%
```

Two honest caveats before anyone quotes this:

1. **This is a `-k` filtered subset, not the file.** It is not comparable to a
   whole-file number and should not be reported as "test_orbit is 97% traced".
2. **The metric is weighted by call volume, not by code.** One orbit
   integration makes hundreds of thousands of force calls, so this measures the
   fraction of *runtime boundary crossings* that trace — not the fraction of
   *code paths* that are jittable. A rarely-called but wholly untraceable
   function barely moves it. Both quantities matter; this one is not the other.

With that said, the interesting number is not the traced fraction — potentials
and orbits are both ~98% there — it is the **gate**: 9871 calls turned away by
`_backend_ready` on the orbit side versus **24** on the potential side. That
ratio is the migration debt, stated quantitatively for the first time. The
potential surface is essentially fully opted in; the orbit surface is not, and
the gate is where that shows up.

This also corrects a note I had been carrying that "test_orbit is not traced".
At the boundary it largely is; what it is not is fully *migrated*. Different
claim, different fix.

### 2026-08-03 — #188 done locally: assert_jit_matches_eager (commit 7619287c)

Every `test_backend_*` module had its own copy of the jit-vs-eager compare.
Deduplicating it is the obvious half; the useful half is that the shared
assertion can check something none of the copies did.

A value comparison between jitted and eager output cannot see a trace that
collapsed to a constant — it compares equal and passes. So the helper also
inspects the jaxpr and requires the output to actually consume the traced
arguments. **Mutation-verified**: a lambda returning a precomputed constant
passes the old idiom and fails the new one. Without that check this PR would
have been pure cosmetics.

`tests/backend_jit_helpers.py`; 6 modules converted, 35 tests, green on numpy
and under forced jax. Held local — #1246 is still the active PR and the runners
are busy.

### 2026-08-03 — conflict markers in the MAIN repo working tree (mine, now cleaned)

`git grep` for the numpy-island survey turned up `<<<<<<< Updated upstream` in
`actionAngleStaeckelGrid.py`. Two files were sitting in unresolved `UU` state on
`feat/backends`:

```
UU galpy/actionAngle/actionAngleStaeckelGrid.py   (4 conflict regions)
UU galpy/backend/interpolate.py                   (1 conflict region)
```

From the `git stash pop` of the foreign `feat/native-grid-spline` WIP earlier in
this session -- the repo-wide-stash trap, again. The session-start `git status`
snapshot was clean, so this was mine, and it had been sitting there unnoticed
across many wakes. Nothing was lost: a conflicted pop does not drop the stash,
and `stash@{0}` still holds the WIP.

Backed both files up to the scratchpad, then `git checkout HEAD --` on the pair.
The tree now matches the session-start snapshot exactly (`M galpy/util/coords.py`
plus untracked), zero markers repo-wide, stash intact.

Worth noting *why* it went unnoticed: I do all PR work in worktrees, so the main
repo's tree is never exercised, and the editable install resolves elsewhere
again. A broken main tree is invisible to every check I run. The `git grep`
markers sweep is in memory as a pre-push step -- it deserves to be a periodic
sweep of the *main* tree too, not just of the branch being pushed.

### 2026-08-03 — #150: the 23 guards, classified

Read each site (comment + immediate context; not full call-graphs). The
"eliminate the islands" count is smaller than the raw grep suggests, because
three different things share one idiom:

**Not islands -- deliberate, keep (4).** `actionAngleIsochroneApprox:1229` is a
documented dual path: scipy on numpy (byte-identical), vectorised
bisection+Newton on backends. `evolveddiskdf:785/2861/3536` select the *op* for
byte-identity -- `numpy.dot` on numpy because `numpy.matmul` differs by ~ULP on
non-contiguous slices, `matmul` on backends because `torch.dot` is 1-D only.
That op choice survives migration and should not be "fixed".

**Accuracy fallback -- blocked on numerics (2).** `jeans:53/139`. Removing the
guard costs 5 forced-jax failures; the backend quadrature is not yet good enough
to replace scipy's. Tracked as #190.

**Transitional coercion stand-ins -- the actual island list (9).**
`surfaceSigmaProfile` x4 (#117), `verticalPotential:118/149`, and
`evolveddiskdf:525/2861/3536`. Each is waiting on a *caller* being migrated, not
on a numerics fix. `verticalPotential`'s own comment names the reason: "a
numpy/scalar z (e.g. an un-migrated parent) -> numpy, so the broadcast matches
what the parent will receive". Its two blocks are also byte-identical
duplicates of each other -- one shared prologue when they are lifted.

**Leaf math kernels -- need a per-caller decision (8).**
`actionAngleStaeckel:1507/2355/2376`, `actionAngleStaeckelGrid` x5. Pure
namespace selection over sinh/log/exp arithmetic, no alternate algorithm. These
match the leaf-migration pattern: legitimate while a numpy caller still consumes
them, removable once the callers coerce at the boundary.

So the roadmap is: 4 stay, 2 are gated on #190, and **17 come out via caller
migration** in two clusters (df/surfaceSigmaProfile+evolveddiskdf+verticalPotential,
and the Staeckel leaf family). That is a materially different plan from "26
islands to remove" -- and the leaf cluster is one PR, not eight.

### 2026-08-03 — #1246 pre-merge audit: correct, but the bug is wider than the PR

Audited the diff against the three things I check before merging.

**if -> xp.where array-input class: no exposure.** The PR contains no backend or
array code at all; `_pack_splines`/`_unpack_splines` dispatch on Python types
(PPoly/BPoly/list/tuple).

**Duplication: none, and I verified the claim rather than repeating it.**
`DiskMultipoleExpansionPotential` holds a `_me` sub-potential
(`DiskMultipoleExpansionPotential.py:116`) and defines no `__getstate__` of its
own, so it inherits the fix through that attribute -- and the PR's new
"DiskMultipole nonaxi" case exercises exactly that path.

**Helper placement: right for the PR as scoped, wrong for the bug.** I went
looking for other holders of scipy piecewise-polynomials and found two the
"exactly 2 classes" note had missed. Measured in the scipy-1.18 venv:

```
PPoly  BPoly  CubicSpline   -> cannot pickle 'module' object
InterpolatedUnivariateSpline -> fine
```

`CubicSpline` is a `PPoly` subclass, so the existing isinstance check would
already catch it. The uncovered holders:

* `SCFPotential._Acos_interp` / `._Asin_interp` (CubicSpline, SCFPotential.py:310)
  -- **confirmed** end-to-end: a time-dependent SCF raises "cannot pickle
  'module' object". It escapes the pickle test because that test enumerates
  *default-constructible* potentials and a time-dependent SCF is not one.
* `streamgapdf._kick_interpdOpar_poly` (PPoly, streamgapdf.py:792).

So three modules need the same pack/unpack, which means `_pack_splines` belongs
in a shared utility rather than in `MultipoleExpansionPotential.py`.

**Decision: merge #1246 as-is anyway.** It is strictly an improvement, it makes
nothing worse, and it is ~3h into a full CI matrix that is nearly green --
re-scoping it now would restart the whole matrix on a runner-constrained
project for no correctness gain. The helper is module-private, so hoisting it
later is a pure refactor with no API impact. Follow-up PR: hoist the pair into a
shared util, apply to SCF + streamgapdf, and test the time-dependent SCF (which
needs a non-default-constructible case, so it cannot just be added to the
existing enumeration).

### 2026-08-04 — the scipy-1.18 pickle bug is LIVE in CI, not preventive

Followed up the #1246 audit finding with the question that decides its urgency:
is CI actually on the broken scipy?

```
build.yml: pip install --upgrade --upgrade-strategy eager numpy scipy ...   (no pin)
completed job on #1246's head: numpy: 2.5.1 / scipy: 1.18.0
```

So yes. The two uncovered holders -- time-dependent `SCFPotential`
(`_Acos_interp`/`_Asin_interp`) and `streamgapdf._kick_interpdOpar_poly` -- are
broken on the exact stack CI runs today. They are green only because **nothing
in the suite pickles them**: `grep pickle tests/test_scf.py
tests/test_streamgapdf*.py` returns nothing.

That is the same shape as the bug #1246 fixes, which was itself invisible until
a test enumerated potentials and tried. Corrected the stale "exactly 2 classes"
task (#182) -- it was 2 only within the scope of default-constructible
potentials in one test. Real work is #191.

Note on wake cadence: the loop has been firing every ~5 minutes rather than the
55m cron, so consecutive "no change in CI" observations are expected and are not
evidence of a stall. Confirmed by timestamping: 01:09:50Z -> 01:14:40Z between
two wakes, against jobs that individually run 20-40 minutes.

### 2026-08-04 — the orbit-side jit debt is THREE container classes (#185 complete)

Extended the boundary metric to record *which* type `_backend_ready` turns away.
On the same `test_orbit.py` subset, the 9,871 rejections rank:

```
  4,476  planarCompositePotential
  3,814  planarPotentialFromRZPotential
  1,005  CompositePotential
    502  mockMovingObjectLongIntPotential     <- a TEST mock, not library code
     48  planarPotentialFromFullPotential
     26  mock/test/Null potentials
```

**94% is three classes**, and all three are *containers* -- they hold
sub-potentials and delegate. Verified: none of
`planarCompositePotential`, `planarPotentialFromRZPotential`,
`CompositePotential`, `planarPotentialFromFullPotential` declares
`_backend_compatible`. The leaf potentials they wrap are migrated; the wrapper
is what the gate rejects.

That reframes the remaining work. "9,871 un-migrated boundary calls" sounded
like a broad sweep; it is four container declarations plus test scaffolding.
~5% of the count is a test mock and should not be read as library debt at all.

**Not proposing a flat flip.** The gate's removal was tried before and reverted
with 4 measured breakages, and it survives as the temporary `_backend_ready`.
The right shape is conditional -- a container is backend-compatible iff its
members are -- evaluated at construction, which is also the only version that
is *true*. A composite holding one un-migrated member must still be rejected.

Metric caveats unchanged: this is a `-k` subset, and the counts are weighted by
call volume, so they rank hot paths rather than code paths.

### 2026-08-04 — #191: SCF fixed and control-verified; streamgapdf being measured

Two commits on `feat/spline-pickle-util` (unpushed; #1246 is still the active PR):

* `703443b3` `galpy/util/_pickle.py` + `SplinePickleMixin`, SCFPotential wired up
* `33f2e5c8` the test

**Testing changed the design twice.** #1246's helper cannot be hoisted verbatim:

* it rebuilds via `type(obj)(c, x, ...)`, which **raises** on `CubicSpline`
  ("`x` must be 1-dimensional") because that constructor takes samples, not
  coefficients -- and `CubicSpline` is exactly the SCF case. Rebuilding at the
  **base** class (`PPoly`/`BPoly`) reproduces the values bit-for-bit and pickles.
  `PPoly.construct_fast` keeps the subclass but returns an object that is itself
  *still* unpicklable, so it is not a way out.
* free functions would have every holder copy the same `__getstate__`/
  `__setstate__` pair. A mixin means holders declare `_PICKLE_SPLINE_ATTRS` and
  nothing else -- and galpy already uses mixins
  (`SphericalHarmonicPotentialMixin`), so it is not a new idiom.

The test is control-verified **both ways**: against galpy without `_pickle.py`
(existence asserted, not assumed) it fails with "cannot pickle 'module'
object"; with the fix it passes. It compares with `assert_array_equal`, not a
tolerance -- the coefficients round-trip exactly, so a looser check would not
notice a rebuilt-but-subtly-different spline, which is precisely what the naive
rebuild produces.

**streamgapdf: measuring rather than guessing which attributes break.** The
class has *two* code paths for `_kick_interpdOpar_poly`: the numpy/scipy one
(streamgapdf.py:995) builds a real `PPoly` and is the default; the backend one
(:1115) already builds a picklable `SimpleNamespace`. So the numpy path is the
broken one. But it also holds a dozen `InterpolatedUnivariateSpline`s and a
`.derivative(1)` of one, and I do not know offhand which of those survive 1.18 --
so a background job builds the Sanders15 stream and scans every attribute
individually instead of me declaring a guess in `_PICKLE_SPLINE_ATTRS`.

### 2026-08-04 — correcting myself on #192: composites are fine, the ADAPTERS are the gap

Last entry I wrote that the three container classes "do not declare
`_backend_compatible`" and implied declaring it was the fix. Reading
`galpy/backend/_compat.py` properly shows that inference was wrong:
`is_backend_compatible` **already** recurses -- a list iff every member is, a
composite through `obj._potlist`, a wrapper iff it and `obj._pot` are.

So the 9,871 gated calls split into two groups that want opposite treatment:

**Honest rejections (leave alone).** `planarCompositePotential` (4,476) and
`CompositePotential` (1,005) both subclass `baseCompositePotential`, so the
recursion already walks their members. They are rejected because a *member* is
not migrated. That is the gate working. The fix is to migrate the member; making
the container claim compatibility would be a lie and would hand backend arrays
to an unmigrated leaf. Plus 528 calls from test mocks, which are not library debt
at all.

**The real gap: 3,862 calls in two adapters.**
`planarPotentialFromRZPotential` (3,814) and `planarPotentialFromFullPotential`
(48) are neither composites nor `WrapperPotential` subclasses, so the recursion
falls through to the leaf branch and reads their own absent flag. They also
store the wrapped potential as `self._Pot` -- *capital* P -- where the wrapper
branch looks for `obj._pot`, so simply adding them to that isinstance tuple
would not even work.

The fix is already written two lines away in the same constructor:

```python
self.hasC = RZPot.hasC and _hasC_planar
self.hasC_dxdv = RZPot.hasC_dxdv and _hasC_planar
```

An adapter derives its capability flags from what it wraps. Doing the same for
`_backend_compatible` is no new mechanism, weakens no gate, and stays *true* --
an adapter over an unmigrated potential still reports False.

Worth noting how the wrong version would have looked: declaring the flag on the
composites would have passed its tests (the members are usually migrated in the
suite) while silently breaking exactly the case the gate exists for.

### 2026-08-04 — #191 streamgapdf: measured, fixed, tested

The attribute scan on a fully built Sanders15 stream returned **one** unpicklable
attribute:

```
UNPICKLABLE  _kick_interpdOpar_poly    PPoly    cannot pickle 'module' object
total unpicklable attrs: 1
```

Every `InterpolatedUnivariateSpline` and the `.derivative(1)` of one survive
1.18. Worth having measured: I would have declared the dozen `_kick_interp*_raw`
splines too, and been wrong. The backend kick path already stores a plain
`SimpleNamespace` there, which packs through untouched, so one declaration
covers both paths with no branching.

Test added on the existing module-scoped fixture (so it costs ~nothing beyond
the stream build) and passes under scipy 1.18. It compares `.c` and `.x` with
`assert_array_equal` *and* re-evaluates through `_kick_interpdOpar`, because a
restored-but-broken spline would sail through an attribute-only comparison.

### 2026-08-04 — #1246 MERGED (daee368b); #1247 opened

`potential: survive scipy 1.18's unpicklable spline objects` went in at 302
checks green, zero failures, `MERGEABLE / CLEAN`. Squash-merged to
`feat/backends-develop` as **daee368b**; branch and worktree removed.

Final audit before merging (re-verified the head SHA and file list had not moved
since the audit):

* **array-input class** — no exposure. The PR contains no backend or array code;
  `_pack_splines`/`_unpack_splines` dispatch on Python types.
* **duplication** — none, checked rather than assumed.
  `DiskMultipoleExpansionPotential` holds a `_me` sub-potential and defines no
  hooks of its own, so it inherits the fix, and the PR's new "DiskMultipole
  nonaxi" case exercises exactly that. Compared against the four existing
  `__getstate__` implementations in the tree: all of them merely *delete*
  unpicklable attributes, none pack splines, so nothing overlaps.
* **helper placement** — module-local is right for this scope. The wider bug
  (SCF, streamgapdf) is #191, which hoists the shared logic into
  `galpy/util/_pickle.py`. Merging as-is was the right call: the PR is strictly
  an improvement, and re-scoping a nearly-green 3.5h matrix on a
  runner-constrained project buys no correctness.

**#1247 opened** (`feat/jit-eager-helper`, was #188): shared
`assert_jit_matches_eager`. Base was 0 commits behind at push time and the files
are disjoint from #1246's, so no rebase — per the standing rule not to
reflexively rebase disjoint PRs.

Queue after this: #191 (SCF + streamgapdf done and tested locally; still needs
the Multipole repoint onto the shared mixin, which was blocked on #1246 and is
now unblocked), then #192.

### 2026-08-04 — #1246 is INCOMPLETE: it misses time-dependent Multipole entirely

Found while doing #191's last piece (repointing Multipole at the shared mixin),
20 minutes after merging #1246. Not a regression -- nothing got worse -- but the
fix covers less than its own attribute list implies.

`_PICKLE_SPLINE_ATTRS` as merged is
`("_I_inner_cos", "_I_inner_sin", "_I_outer_cos", "_I_outer_sin")`. A
**time-dependent** expansion does not have those attributes at all. It has
`_I_inner_cos_interp`, `_I_outer_cos_interp`, `_rho_cos_interp` (and the `_sin`
counterparts when non-axisymmetric), holding `CubicSpline`s. Measured:

```
_I_inner_cos_interp   holds 4 CubicSpline  picklable=False
_I_outer_cos_interp   holds 4 CubicSpline  picklable=False
_rho_cos_interp       holds 4 CubicSpline  picklable=False
```

so the merged list covers *none* of the time-dependent case. Static Multipole
does pickle -- #1246 works for what it covers.

Two traps on the way to that conclusion, both worth recording:

1. **A whole-object pickle of a `from_density` potential fails for an unrelated
   reason.** My first attempt reported `AttributeError: Can't get local object
   'MultipoleExpansionPotential.from_den...'` -- that is the *closure* problem
   (task #68), not the scipy one, and it completely masks the spline failure.
   Scanning attributes individually is what got past it. Had I stopped at the
   whole-object error I would have "confirmed a bug" that was a different bug.
2. My exactness probe then threw `TypeError: 'list' object is not callable` --
   these attributes are *nested* lists (per-l, per-m), not flat ones. That was my
   script, not the code, but it is the same shape of mistake: assuming a
   structure instead of looking.

Fix folded into #191: repoint Multipole at `SplinePickleMixin` and extend the
list to all ten names (static + `_interp`, cos + sin). Verified after the change:
`scipy piecewise-polys left in __getstate__: NONE`, and all 12 splines across
the three attributes round-trip **bit-identically** (`array_equal` on values at
5 times spanning the tgrid).

This is the third time the same blind spot has produced a bug in this family:
the pickle test enumerates *default-constructible* potentials, so anything
needing an explicit construction -- time-dependent SCF, time-dependent Multipole
-- is invisible to it.

### 2026-08-04 — #1247's codecov/project red is a partial-upload artifact (not actionable)

`codecov/project` went RED on #1247 with an **empty** description, after only 14
of ~152 checks had finished. One codecov v2 API query settled it:

| commit | coverage | hits / lines |
|--------|----------|--------------|
| `7387f511` base, settled | **99.95%** | 48324 / 48344 |
| `6512c5e4` #1247 head, 14/152 done | 63.32% | 30616 / 48344 |
| `daee368b` develop tip, just merged | 63.95% | 30917 / 48344 |

Same `lines` denominator, ~18k fewer `hits`: the coverage shards have not
uploaded yet. `codecov/patch` is already SUCCESS, which is the signal that
actually reflects the diff. The freshly-merged develop tip showing the same
~64% is independent confirmation that this is about upload timing, not about my
branch.

One trap worth recording: the API reports **`state: complete` even in this
state** -- it describes upload *processing*, not CI completion. It cannot be
read as "all shards in", and I nearly did.

Not chasing it; it should clear as the run drains. The shape to distrust in
future is a ~40-point cliff with an unchanged denominator; a *real* project drop
looks like a similar `hits` count with a small delta.

### 2026-08-04 — #191 complete locally (4 commits)

`MultipoleExpansionPotential.py` **shrinks 49 lines net** as its module-local
pack/unpack pair moves to the shared `galpy/util/_pickle.py`, now used by three
modules. Final: `_pickle.py` +72, Multipole −49 net, SCF +7, streamgapdf +9,
tests in three files.

The structural test is the part worth keeping: rather than an eleventh attribute
name, it walks `__getstate__` and asserts no scipy piecewise-polynomial survives
it. **Control-verified** -- fails on #1246's four-name list, passes on the
extended one -- so it would have caught the gap at review rather than 20 minutes
after merge.

It deliberately does NOT assert whole-object picklability for these cases:
`from_density` stores an internal lambda, so anything built through it is
unpicklable for an unrelated reason ([[#68]]) that would mask what is being
checked. That is stated in the test.

### 2026-08-04 — codecov diagnosis confirmed in motion; #192 built (61962f8b)

**Codecov.** Watched the same head commit as shards reported:

```
14/152 checks : 63.32%  30,616 hits
26/152 checks : 75.39%  36,449 hits     (base = 99.95%, 48,324)
```

Climbing toward the base as predicted, which settles it: the `codecov/project`
red is a partial-upload artifact, not a coverage regression. No action.

**#192 planar adapters** (`61962f8b`, worktree `/var/tmp/planaradapter`). Both
adapters now derive `_backend_compatible` from the wrapped potential, exactly as
`hasC` already is two lines above. Checked the #76 ordering trap first: neither
constructor calls `normalize()`, so it does not apply.

The verification is where the work was, and my first attempt was worthless.
I used `AnyAxisymmetricRazorThinDisk` as the negative case and got **True/True**
-- because what *that* potential opts out of is `_backend_accepts_arrays`, a
different flag entirely. So the probe contained no genuine negative and proved
nothing about the safety direction. Redone with a synthetic opted-out potential:

```
synthetic unmigrated      full=False  planar=False   <- the gate still rejects
migrated MiyamotoNagai    full=True   planar=True
[migrated, unmigrated]                planar=False   <- composition not laundered
```

The committed test uses that synthetic rather than a real potential on purpose:
a real one silently stops testing anything the day it gets migrated
([[pvar-negative-example-fragility]]). Control-verified -- removing the two
derivations makes it fail.

Scope stayed narrow deliberately: this touches ONLY the adapters. The composites
(5,481 of the 9,871 gated calls) are rejected *correctly* -- they are already
walked recursively and a member genuinely is not migrated. The tempting version
of this fix, declaring the flag on the composites too, would have passed its
tests while breaking exactly the case the gate exists for.

Boundary metric re-running to confirm the gated-out count actually drops by
~3,862 rather than merely that tests stay green.

### 2026-08-04 — #192 forced-backend gate (the check numpy runs cannot make)

#192 changes *which objects pass the coercion gate*, so more coercion now
actually fires. That is precisely the class a numpy-only run is blind to
([[forced-backend-gate-before-push]]): before the change those planar adapters
fell through to numpy, and after it they take the backend path for real.

Forced **torch**, whole files (not hand-picked nodeids -- a batch reports
different failures than the file):

```
tests/test_backend.py  tests/test_backend_array_inputs.py  tests/test_backend_wrappers.py
--backend torch  ->  all passed
```

Forced **jax** on the same three files, and forced-torch `test_potential.py -k
planar`, are still running. Running both frameworks matters here because CI
installs only one per shard ([[ci-single-backend-install-blindspot]]), so a
cross-framework break is something only this box can see before merge.

Note this is a *stronger* obligation than for the previous two PRs: #1247 and
#191 touch tests and pickling, neither of which changes dispatch. #192 changes
dispatch itself.

**Forced-jax result: also all passed.** So #192's three backend suites are green
on BOTH frameworks. That is the cross-framework check no CI shard can make (one
framework installed per shard), and it is the check that matters most for this
particular PR, since it is the one that changes dispatch.

**Codecov convergence, continued** (same head commit, no new pushes):

```
14/152 checks : 63.32%  30,616 hits
26/152 checks : 75.39%  36,449 hits
36/152 checks : 84.79%  40,995 hits     (base 99.95%, 48,324)
```

Monotonic climb toward the base with the diff untouched. The `codecov/project`
red is now conclusively an upload-timing artifact; nothing to fix, and re-running
it during a deep queue would be counterproductive.

**Forced-torch planar (`test_potential.py -k planar`) also passed.** #192's
verification matrix is now complete on the test side:

| run | result |
|-----|--------|
| unforced numpy, `test_backend_array_inputs.py` | 7 passed |
| forced **torch**, 3 backend suites | passed |
| forced **torch**, `test_potential.py -k planar` | passed |
| forced **jax**, 3 backend suites | passed |

Still outstanding: the boundary metric, which is the actual acceptance criterion
for this PR. Green tests are *also* consistent with the fix doing nothing at all
-- the claim is that 3,862 gated calls become traced, and only the metric can
show that.

### 2026-08-04 — #192 numpy byte-identity: measured, not argued

The gate short-circuits on `xp is numpy` *before* `_backend_ready` is consulted,
so #192 should not touch the numpy path at all. That is a reasonable argument
and exactly the kind I am supposed to verify rather than trust
([[flag-numpy-byte-identical-changes]]).

Captured planar `__call__` / `Rforce` / `R2deriv` at 6 radii for MiyamotoNagai,
NFW, Hernquist and PowerSpherical, printed at 17 significant digits, with the
two derivations present and then temporarily removed:

```
13 value rows -> byte-for-byte identical
worktree restored clean (0 modified)
```

So the numpy path is untouched. Worth doing even when the reasoning is solid:
this is the same shape of claim as "nothing branches on the spline subclass",
which was also true but which I had asserted before checking.

#192's evidence is now: control-verified test (fails without the fix), both
directions of the flag (synthetic unmigrated stays False), forced torch + forced
jax green, and numpy byte-identical. The one thing still outstanding is the
boundary metric — the only evidence that the fix *does something*, as opposed to
being harmless.

### 2026-08-04 — the #192 metric run TIMED OUT, and that is itself a signal

The re-measurement was killed at its 40-minute limit (exit 143), having produced
nothing. The earlier runs of the *same* selection, in `/var/tmp/typeleaks`,
finished inside 25 minutes. First hypothesis was a missing C extension in the
new worktree -- checked, and **neither** worktree has a `.so`, so that is not it.

The likelier explanation is the finding itself: #192 makes ~3,862 more boundary
calls pass the gate, so they now get *traced* instead of falling through to
numpy, and jax compilation is not free. That is exactly the shape of
[[migration-flips-numpy-fallthrough-to-eager-slow]] -- a migration flipping a
shard from fast-numpy to slow-XLA and timing it out. If real, it is a genuine
CI risk for #192, not just a measurement inconvenience, and it needs to be known
*before* the PR is opened rather than discovered as a red shard.

Also caught a comparability error before acting on it: I had been about to
compare the new worktree's number against the 9,871 measured in `typeleaks`.
Different worktrees, different branches -- not an A/B. Re-running as a true A/B
**inside one worktree**, toggling only the two derivations, with a single-test
selection so both arms finish. That measures the gate delta and the wall-clock
cost at the same time.

### 2026-08-04 — #192 mechanism measured directly (faster than the aggregate)

While the aggregate A/B grinds, measured the *mechanism* instead of a count over
a whole orbit integration: patch `_backend_ready`, construct a planar adapter
over a migrated potential, call three decorated methods with a jax array.

```
planar-adapter boundary calls: asked=3  passed_gate=3   -> every call passes
```

That is the claim of #192 stated at its smallest: an adapter over a migrated
potential now reaches the coercion path instead of being turned away. It is
better evidence than an aggregate delta in one respect -- it isolates the
adapter rather than summing over everything an orbit integration touches.

**Near-miss worth recording.** I was about to run the control arm by toggling
the two derivations -- while the A/B script was itself mutating that same file
on its own schedule. Two processes editing `planarPotential.py` concurrently
would have corrupted both measurements and could have left the worktree in a
half-reverted state that a later `git add -A` would happily commit. Checked
first: the script was still running, file showed 2 derivations (WITH arm), so
the number above is from the right state. Control deferred until the A/B exits.

Rule for the future: before an in-place file swap, check whether a background
job already owns that file.

### 2026-08-04 — the A/B was measuring the wrong test; restarted

Checked the blast radius of a possible #192 slowdown and found two things that
changed the plan:

1. **140 `test_orbit`/`test_orbits` entries are already in `backend_slow_skip`
   for jax.** So most of the slow orbit tests do not run on that shard at all,
   and a #192 slowdown has a far smaller reach than I assumed when I saw the
   timeout.
2. **My A/B selection was itself ledgered.** `test_analytic_ecc_rperi_rap` is an
   xfail on *both* jax and torch ("passes locally, xfails in CI group g3"). I was
   timing an expected-to-fail test and would have reported its wall-clock as if
   it represented the shard.

Killed the run and restarted on `test_orbit_setup_planar` — verified present in
`test_orbit.py`, and in *neither* the xfail ledger nor the slow-skip list, so it
is something CI actually executes.

Killing needed care: the script swaps `planarPotential.py` mid-run, so a kill
between the revert and the restore would leave the worktree half-reverted. Checked
the pre-swap markers first (2 derivations present, no `/tmp/pp_ab.py` backup),
killed, then re-verified: 2 derivations, `git dirty=0`, no stale backup.

Lesson, and it is the second time today the same shape has bitten: **check
whether the test you are measuring is ledgered or skipped before drawing
conclusions from it.** A selection that looks representative by name may be one
the suite has already given up on.

### 2026-08-04 — #192 PROVEN, and a correction to what I claimed earlier

**Correction first.** Two wakes ago I wrote that I had "measured the mechanism
directly": patch `_backend_ready`, call `pl._evaluate/_Rforce/_R2deriv` on a
planar adapter, observe `asked=3 passed=3`. That measurement was of the **wrong
object**. Printing what the gate actually receives:

```
_backend_ready received: MiyamotoNagaiPotential  -> True   (not the adapter!)
```

`pl._Rforce(R)` *delegates* to the wrapped 3D potential's decorated method, so
the gate inspects the already-migrated 3D potential and passes on its own
merits. The adapter's flag is never consulted on that path — forcing
`pl._backend_compatible = False` changed nothing, which is what exposed the
error. My "3 of 3 passed" said nothing about #192.

**Where the adapter actually reaches the gate**: the module-level
`evaluateplanarRforces` / `evaluateplanarPotentials` / `evaluateplanarR2derivs`
functions, whose `args[0]` *is* the potential. That is also what the boundary
metric was counting when it attributed 3,814 calls to
`planarPotentialFromRZPotential` — orbit integration calls those per step.

**On that path, the A/B is unambiguous** (20 iterations x 3 evaluators, flag
toggled on the instance so no file mutation and no race):

```
  WITH #192   adapter reaches gate=80   passes=80   GATED OUT= 0
  pre-#192    adapter reaches gate=80   passes= 0   GATED OUT=80
```

Every adapter call was gated out before and passes now. #192 is doing what the
PR claims, on the path that matters.

**Three bad measurements preceded this one**, all of the same species: a
selection that *looked* right by name. AnyAxisym as a "negative" case (it opts
out of a different flag); `test_analytic_ecc_rperi_rap` as a timing baseline
(it is a ledgered xfail); `test_orbit_setup_planar` as a workload (0 boundary
calls). The fix each time was to print what the code actually saw instead of
trusting the label — which is also what finally worked here.

### 2026-08-04 — what #192 actually buys (narrower than my headline)

Having proved the gate flips (80 gated-out -> 80 passing), I asked the next
question rather than stopping at a satisfying number: does anything the user
sees change? Measured:

```
                result type   is_backend_array   jax.grad
  WITH #192     ArrayImpl     True               0.357465922832489
  pre-#192      ArrayImpl     True               0.357465922832489
```

**Identical.** Values, backend-array-ness and gradients are unchanged, because
for a jax input the *wrapped* potential's own `@backend_input` boundary already
coerces correctly. The adapter being gated out was never breaking anything --
the call simply coerced one level deeper.

What does change is reachability of the trace boundary:

```
  WITH #192   adapter path reaches traced_call = 50
  pre-#192                                     = 20
```

So the accurate claim for #192 is: *the flag becomes truthful, and the adapter's
own boundary becomes reachable by the trace machinery* -- not "3,862 broken
calls fixed". Those 3,862 were gate rejections, not user-visible failures. I had
been drifting toward the second phrasing, and it would have gone into the PR body
as an overstatement.

That also demotes #192's priority: it is correctness-of-metadata plus a
prerequisite for tracing the adapter as a unit, not a bug fix. It should be
described that way, and it is a weaker argument for accepting any jax slowdown
it causes. Whether it yields *more actual tracing* needs a `--jit` run, since
everything here returned NOT_TRACED with jit off; that is the remaining question.

### 2026-08-04 — #192 resolved: it DOES increase real tracing (20 vs 0)

The open question was whether #192 increases *actual* tracing or merely
reachability. Answered by enabling jit programmatically
(`use("jax", force=True); set_jit("jax")`) and filtering the counter to adapter
calls only:

```
                adapter reaches traced_call   NOT_TRACED   ACTUALLY TRACED
  WITH #192                21                     1              20
  pre-#192                  0                     0               0
```

Before the change the adapter never reaches the trace point at all; after, 20 of
its calls are genuinely traced. That is the mission's currency, so #192 earns
its place -- it is not the no-op I had started to worry it might be.

This also reconciles with the jit-off probe (50 vs 20 reaching `traced_call`):
that counter was unfiltered, so pre-#192's 20 were the *inner* potential's calls,
not the adapter's. Filtering to `planarPotentialFromRZPotential` makes the story
consistent: 0 adapter calls before, 21 after.

**And it explains the timeout.** More traced calls means more XLA compilation, so
the 40-minute overrun was the cost of the benefit rather than a defect. That is a
real tradeoff to state in the PR, not something to hide: #192 buys traced
coverage and pays in jax compile time, on a shard where 140 orbit tests are
already slow-skipped.

Final honest framing, after three revisions:
* NOT a bug fix -- values, `is_backend_array` and gradients are identical.
* IS a tracing gain -- 20 adapter calls traced where 0 were.
* Costs jax compile time, mitigated by the existing slow-skip coverage.

### 2026-08-04 — #191 pre-audited against the merge criteria (passes)

Ran the three checks now so the merge is fast when #1247 lands.

**array-input / xp.where class — no exposure.** Grepping the diff for `xp.`,
`is_backend_array`, `get_namespace`, `coerce` returns nothing: the PR contains no
backend or array code, only pickling.

**Duplication — eliminated, not moved.** `pack_splines`/`unpack_splines` are
defined in exactly one place repo-wide (`galpy/util/_pickle.py`). The
module-local copies #1246 added to `MultipoleExpansionPotential.py` are gone,
which is why that file is **-49 lines net** despite gaining coverage for the
time-dependent case. All three consumers inherit the mixin; none retains its own
`__getstate__`:

```
  MultipoleExpansionPotential  ...  SplinePickleMixin   _PICKLE_SPLINE_ATTRS = (10 names)
  SCFPotential                 ...  SplinePickleMixin   _PICKLE_SPLINE_ATTRS = ("_Acos_interp", "_Asin_interp")
  streamgapdf                  ...  SplinePickleMixin   _PICKLE_SPLINE_ATTRS = ("_kick_interpdOpar_poly",)
```

**Helper placement — forced by the consumers.** It has to sit above both
`potential/` and `df/`, because `streamgapdf` is a df; `galpy/util/_pickle.py`
matches the existing private-module convention (`_optional_deps.py`,
`_load_extension_libs.py`). Per-module `__getstate__` hooks would have been the
house style, but three modules needing the *same* pack/unpack is what justifies
hoisting the logic while leaving the declaration per-class.

Net: 227 insertions / 49 deletions across 7 files, one coherent theme (scipy
1.18 spline pickling), no reason to split.

### 2026-08-04 — codecov endgame: precedent says it self-resolves

Rather than keep re-reading the same climbing number, checked what
`codecov/project` did on the three PRs already merged in this session:

```
  #1246  codecov/patch: SUCCESS   codecov/project: SUCCESS
  #1243  codecov/patch: SUCCESS   codecov/project: SUCCESS
  #1242  codecov/patch: SUCCESS   codecov/project: SUCCESS
```

All three ended green. Combined with the monotonic climb on #1247's own head
(63.32% -> 75.39% -> 84.79% -> 86.11%, base 99.95%), the red is transient and
flips once uploads finish. No action, and specifically no re-run: re-running
during a deep queue is what makes codecov finalise from partial uploads in the
first place.

Also worth knowing for later: `codecov.yml` carries only an `ignore:` list, no
`coverage.status.project` block, so codecov applies its **default** `target: auto,
threshold: 0%` — any real drop against base fails. And
`feat/backends-develop` has no branch protection (the API 404s), so
`mergeStateStatus: BLOCKED` here means "checks still running", not "a required
check failed". That is consistent with #1246, which sat BLOCKED until its last
shard finished and then went CLEAN.

### 2026-08-04 — the test_quantity sphericaldf cluster is an ABSOLUTE-tolerance artifact (0.52 ULP)

Followed the highest-yield ledger lead: of the 10 tests ledgered on *both*
backends, four are `test_quantity.py::test_sphericaldf_*` — 8 of the 42 entries,
**19% of the ledger in one cluster**. Read today's actual failure via
`--junitxml` (the conftest force-exit truncates stdout):

```
AssertionError: sphericaldf method vmomentdensity does not return correct Quantity
assert 1.9073486328125e-06 < 1e-08
  vmomentdensity(1.1, 0, 2) = 1.64026634e+10
```

The absolute difference is 1.9e-6 on a value of order **1e10**:

```
  relative error = 1.163e-16   =  0.52 ULP  (float64 eps = 2.22e-16)
```

Half a floating-point unit. The two results are as close as two independent
float64 computations can get without being bit-identical. The test fails only
because it compares an **absolute** difference against `1e-8` while the quantity
is ~1e10 — i.e. it demands ~1e-18 relative precision, which float64 cannot
represent. numpy passes by luck: identical operation order gives an exactly
equal result.

`test_quantity.py` has ~882 absolute `10.0**-8.0` thresholds and ~165
"does not return correct Quantity" assertions, so this is a file-wide pattern
rather than four unlucky tests.

**This is category A of the tolerance batch (#171), and it is the strongest
possible case for it**: not "loosen a tolerance because the backend is worse",
but "replace an absolute comparison with a relative one because the absolute one
was never meaningful at this magnitude". The user's steer was exactly this —
"A: agreed on 3, but make sure the tests are stringent". A relative check at,
say, 1e-13 would be *far* stringent-er than the current absolute 1e-8 for
quantities of order 1, while actually being satisfiable at 1e10.

Not acting unilaterally: the fix edits a shared assertion pattern across a large
test file, and the tolerance batch is explicitly one of the two decisions parked
for Jo. Recorded here with the measurement so the decision is a one-liner when
he is back.

### 2026-08-04 — nearly recorded "2 of 4 are stale"; caught the invalid context

Extended the sphericaldf-quantity cluster check from 1 test to the other 3, and
the hand-picked batch said:

```
  test_sphericaldf_method_inputAsQuantity   FAILED  (same 1.907e-06 / vmomentdensity)
  test_sphericaldf_sample_outputunits       PASSED
  test_sphericaldf_densitymatch_issue677    PASSED
```

The tempting conclusion — "2 of the 4 are stale ledger entries, prune them" —
would have been methodologically wrong twice over:

* `test_quantity.py` is on the **cross-test-state** list
  ([[xdist-shared-state-artifacts]]), so a 3-test selection is not the context
  those tests run in.
* A hand-picked node-ID batch reports different results than the whole file
  ([[handpicked-nodeid-batch-is-not-a-context]]) — I have been bitten by this
  before.

So the pass/fail split above is not evidence about staleness at all. Re-running
the **whole file** with `--runxfail --junitxml` and will filter for the four
node IDs; only that answers it.

Confirmed so far, and independent of the above: **two** of the four fail with
the *identical* absolute-vs-relative tolerance signature (same 1.907e-06 on the
same ~1.6e10 `vmomentdensity`), which is 0.52 ULP relative. That part of the
finding stands.

### 2026-08-04 — codecov resolved to 99.95%, exactly as diagnosed

`codecov/project` on #1247 flipped to SUCCESS on its own. Final state matches the
base to the line:

```
  base   7387f511 : 99.95%   48324 / 48344
  #1247  6512c5e4 : 99.95%   48324 / 48344
```

Full trajectory on one unchanged commit:
`63.32% -> 75.39% -> 84.79% -> 86.11% -> 99.95%`.

Worth stating plainly because the temptation at the start was to "fix" it: there
was never anything wrong. Chasing it — re-running the check, or worse, editing
the diff to chase coverage — would have burned runners and possibly made the
report *less* complete, since re-running during a deep queue is what causes
codecov to finalise from partial uploads in the first place. The correct action
was to diagnose once (compare `hits` head-vs-base), confirm against precedent
(three prior merges all ended green), and then leave it alone.

#1247 now has **zero failures** with 3 checks still running.

### 2026-08-04 — #1247 MERGED (86045438); #1248 opened

`tests: shared assert_jit_matches_eager with a real traced-ness check` went in at
**152 checks green, zero failures**, `MERGEABLE / CLEAN`. Squash-merged to
`feat/backends-develop` as **86045438**; branch and worktree removed.

Pre-merge audit (re-verified head SHA and file list first):

* **array-input class** — no exposure. All 9 files are under `tests/`; the helper
  inspects jaxprs, it does not coerce or dispatch on array type.
* **duplication** — the PR *removes* it; that is its purpose. The two
  `test_backend_adiabatic_*_grad.py` files were verbatim duplicates of each other
  and now share the helper (each −10/+4).
* **helper placement** — `tests/backend_jit_helpers.py`, deliberately *not*
  shipped in `galpy`: it is a pytest assertion, not library API. Test modules
  already import from each other, so no packaging change was needed.

The codecov red that shadowed this PR for ~15 wakes finished at **99.95%,
48324/48344 — identical to base**, having climbed 63 → 75 → 85 → 86 → 99.95 on an
unchanged commit. Diagnosed once, confirmed against three prior merges, then left
alone. Doing nothing was the correct action and it took discipline to keep
choosing it.

**#1248 opened** (was #191): `SplinePickleMixin` + SCF + streamgapdf +
time-dependent Multipole + the structural leak test. Rebased onto `86045438`
first — this one *does* touch `MultipoleExpansionPotential.py`, which #1246
changed, so unlike #1247 the rebase was load-bearing rather than cosmetic.
Diff unchanged at 227/49 across 7 files.

Queue: #1248 in CI, #192 (`61962f8b`) ready behind it.

### 2026-08-04 — the staleness check needs a bigger budget than I gave it

The whole-file `test_quantity.py` run under forced jax is about to hit its
50-minute cap without finishing. That file is ~18k lines and forced-jax is slow;
the budget was simply too small.

Worth being clear about why I am not shortcutting it. The *reason* for running
the whole file is that `test_quantity` carries cross-test state, so a subset does
not reproduce the context those tests run in — which is exactly the trap that
made the 3-test batch report two PASSes I could not trust. Narrowing the
selection to make it finish would reintroduce the problem the run exists to
avoid.

**Value check before spending more on it**: this decides whether *2* ledger
entries are stale — worthwhile, but small next to the 8-entry tolerance cluster
in the same file, which is already measured and parked for Jo. So: re-dispatch
with a multi-hour budget and let it run across wakes, but do not let it crowd out
PR work. If it never converges, the honest outcome is "unknown, needs a CI-side
measurement", not a guess either way.

### 2026-08-04 — second ledger cluster probed: actionAngleSpherical linear_angles (torch)

`test_actionAngle.py` is the largest single-file chunk of the ledger (12 of 42),
and four of those are `actionAngleSpherical_*linear_angles` on **torch** — the
same "one file, one shape" signature that led to the ULP finding.

A 3-test selection under forced torch returned **all PASSED**. Deliberately not
concluding "stale, prune them": that is the exact inference I nearly published an
hour ago for the quantity cluster, and it was wrong there because a hand-picked
node-ID batch is not the context the suite runs
([[handpicked-nodeid-batch-is-not-a-context]]). `test_actionAngle` is *not* on
the cross-test-state list, which makes a subset more defensible here than for
`test_quantity` — but "more defensible" is not "verified", and the cost of being
wrong is silently deleting real coverage from the ledger.

Whole file dispatched under forced torch with `--runxfail --junitxml`; will
filter for the four node IDs. Two background runs now in flight (this, plus the
`test_quantity` jax re-run with a 2.5h budget). Both are burndown reconnaissance,
neither blocks the PR queue.

### 2026-08-04 — corrected my own cluster map: the test_orbit entries are Track-E stale-candidates

Wrote in STATUS.md that `test_orbit::test_analytic_ecc_rperi_rap` was "annotated
'passes locally, xfails in CI group g3' — a sharding artifact, not a backend
bug". That attribution was wrong, and I caught it by looking at line numbers
instead of proximity:

```
line 122: # (test_analytic_ecc_rperi_rap: passes locally, xfails in CI group g3). Same
line 156: # --- jax single-orbit: action-angle accessors (Track E) ---
line 159: #   ... Un-xfail when Track E lands.
line 160: jax tests/test_orbit.py::test_analytic_ecc_rperi_rap
```

The g3 comment lives 38 lines earlier, attached to a *different* entry, and
merely mentions this test by name for comparison. The entry's own stated reason
is **Track E** — `actionAngleStaeckel_c` not backend-migrated.

That is a more interesting conclusion than the wrong one: Track E's tasks are
largely completed, so the ledger's own un-xfail condition ("un-xfail when Track E
lands") may already be satisfied. These become **stale-candidates to re-run**,
not artifacts to dismiss.

Same failure mode as four earlier today: I read a comment *near* the thing and
attributed it, without checking it belonged to the thing. Grepping with line
numbers is the cheap fix, and it is the same move as printing `type(t).__name__`
at the gate, or checking whether a test is ledgered before timing it. STATUS.md
corrected.

### 2026-08-04 — every local test run this session lacked the C extension

The full-file `test_actionAngle` run under forced torch gave the valid-context
answer I wanted for the ledger cluster:

```
  4 x actionAngleSpherical_*linear_angles      PASSED    <- stale candidates
  Vertical_Harmonic x2, Adiabatic_linear_angles_cylsep,
  AdiabaticGrid_outsidegrid_c, StaeckelGrid_basicAndConserved   failure
```

But the totals flagged something bigger: **31 failures out of 209, while only 5
of the file's ledgered entries failed** — ~26 unexpected. A green CI shard does
not look like that, so the environment, not the ledger, was wrong. Checked:
`galpy/libgalpy*.so` is **absent** from the worktree, so every `c=True` path is
unavailable.

That is not a one-run problem. **None** of the worktrees I have used this session
(`typeleaks`, `splinepickle`, `planaradapter`, `jithelper`) has a built C
extension, so every local pytest run all session has been in a degraded
environment. For the backend suites it mostly did not matter — they exercise pure
Python paths — which is why nothing looked wrong until a C-heavy file was run.

Consequences for what I have already concluded:

* the **4 PASSes** construct `actionAngleSpherical(pot=lp)` with no `c=True`, so
  they are plausibly C-independent and the staleness signal probably transfers —
  but "probably" is doing real work in that sentence, so it stays a *candidate*.
* the **5 failures** cannot be attributed: "ledger reason" and "no C extension"
  are indistinguishable in this environment.

Building the extension in `splinepickle` now
(`GALPY_COMPILE_SINGLE_EXT=1 GALPY_COMPILE_NO_OPENMP=1 setup.py build_ext
--inplace`), which is the documented recipe and makes every future local run in
that tree comparable to CI. Worth the few minutes: I have been drawing
conclusions from local runs for hours.

**C extension now built in `splinepickle` and verified LOADED** — and two details
worth recording:

* it lands at the worktree **root** (`/var/tmp/splinepickle/libgalpy...so`), not
  under `galpy/`, so `ls galpy/libgalpy*.so` reports nothing even on success.
  That is what made me think the build had failed.
* `pot.hasC` is **True regardless** — it is a class attribute, not a probe. It is
  not evidence the extension loaded, and I briefly took it as such. The real
  check is `galpy.util._load_extension_libs.load_libgalpy()` returning non-None
  ([[worktree-so-provenance]] warns about exactly this).

Re-running `test_actionAngle.py` under forced torch with the extension present.
Only that result can attribute the 5 ledgered failures, since "ledger reason" and
"no C extension" were indistinguishable before. The 4 `actionAngleSpherical`
PASSes should survive (they build `actionAngleSpherical(pot=lp)` with no
`c=True`), but "should" is exactly the word this whole sequence keeps punishing.

### 2026-08-04 — killed the in-flight quantity run too: same C-extension defect

The `test_quantity` re-run I had just given a 2.5-hour budget was running in
`planaradapter`, which also has **no** C extension. It would have produced the
same uninterpretable mixture of real failures and missing-C failures, two hours
from now. Killed it and re-dispatched in `splinepickle`, which now has a verified
`load_libgalpy() -> LOADED`.

Cheaper to throw away 25 minutes of a wrong run than to spend two more hours
earning an answer I could not use. The tell was already in hand — the AA run's
"31 failures, only 5 ledgered" — and applying it to the *other* in-flight job was
the obvious next move rather than waiting for it to fail the same way.

Both reconnaissance runs are now in the C-enabled tree:
* `aafull_c.xml` — `test_actionAngle` under forced torch, with C
* `qfull.xml`    — `test_quantity` under forced jax, with C

Note for later: only `splinepickle` has the extension built. `planaradapter`
(#192) and `typeleaks` still do not, so any local run in those trees remains
degraded and must not be used to attribute failures.

### 2026-08-04 — the AA "stale entries" were an artifact; the ledger is exactly right

Re-ran `test_actionAngle.py` under forced torch **with** the C extension. The
result reverses the conclusion I was one step from publishing:

```
                                    no C ext      WITH C ext
  total failures                       31              9
  the 4 Spherical_*linear_angles     PASSED          FAIL
  ledgered entries failing            5 of 9         9 of 9
```

Two things fall out:

1. **The 26 phantom failures were entirely the missing extension** — 31 → 9 once
   it is present.
2. **The 9 remaining failures are exactly the 9 ledgered node IDs.** No stale
   entries, no missing entries: for this file the ledger is precisely correct.

The 4 `actionAngleSpherical_*linear_angles` entries are **not stale**. They
"passed" twice — in a 3-test subset and in a full-file run — purely because the
absent C extension sent them down the Python path. Pruning them on either result
would have deleted four valid ledger entries and turned the torch shard red,
and the second run *looked* like the careful version of the first.

This is the payoff for refusing to conclude from a green subset. The sequence
was: subset says PASS -> "not enough, run the file"; file says PASS but 31
failures -> "the environment is wrong, not the ledger"; fix environment -> FAIL.
Three rounds, and only the last was true. STATUS.md corrected: the AA cluster is
6 *genuine* entries, not stale candidates.

### 2026-08-04 — #1248: codecov/PATCH is red too, which is a different signal

#1248 shows **both** `codecov/patch` and `codecov/project` FAILURE. That is not
the same as #1247, where patch stayed green the whole time and only project was
red. Memory is explicit that **patch reds can be real** and must be
locally verified rather than assumed phantom — #1121 was admin-merged past a
patch red that turned out to be a genuine 36-line gap.

Current upload state on the head commit:

```
  25.83%   12492 / 48353 hits   (10 of ~150 checks complete)
```

At 26% of lines, essentially *everything* reads uncovered, added lines included,
so the partial-upload explanation fits both reds. Extra wrinkle: the base
`86045438` has **no coverage data at all** yet — it merged an hour ago and its
own push-event run is still going — so the comparison is against nothing.

Not assuming. Running the documented local check: the tests that exercise the
added lines under `--cov-report=term-missing`, and confirming added ∩ missing is
empty. #1248 adds a whole new module (`galpy/util/_pickle.py`) plus three test
files, so there is real surface for a genuine gap to hide in — precisely the
shape (`SplinePickleMixin.__setstate__`, say, reachable only on unpickle) that
the #1199 "relocated case orphans its old branch" lesson warns about.

**Local patch verification: the added code IS exercised.** Both coverage routes
were blocked locally, for two *different* reasons worth recording:

* `pytest --cov` dies with `ImportError: cannot load module more than once per
  process` from **numpy's** `_pocketfft_umath` — not galpy's extension, and only
  with the pytest-cov plugin loaded. Plain `pytest` on the same node ID passes.
* `python -m coverage run -m pytest` collects **nothing**: galpy's conftest
  force-exits, so coverage's atexit write never happens
  ([[torch-fork-deadlock-streamdf]] notes the same force-exit truncating stdout).

So I measured execution directly instead — count calls into the new module during
a real round-trip:

```
  after a time-dependent SCF pickle round-trip:
    pack      2x      getstate  1x
    unpack    2x      setstate  1x
  -> every added path in galpy/util/_pickle.py executed
```

That is the substance of what `codecov/patch` checks, obtained without the
tooling. Combined with the head commit sitting at **25.83% / 12492 of 48353
hits** (10 of ~150 checks in) and a base commit with *no* coverage data yet, both
reds are the partial-upload artifact — the same pattern #1247 showed, now
including patch because this PR adds a new module whose covering shards have not
reported.

Not admin-merging on that basis, and not re-running: #1247 resolved itself from
63% to 99.95% by waiting, and this has ~140 checks still to report.

### 2026-08-04 — the AA cluster is ALSO mostly tolerance: 8 of 9, one real bug

Extracted the failure messages from the C-enabled `test_actionAngle` run (free —
the junitxml was already on disk). Of the 9 torch-ledgered entries:

```
  StaeckelGrid_basicAndConserved   4.8354e-14  <  1e-16   <- asks for ~bit-exact Jr=0
  Adiabatic_linear_angles_cylsep   4.359e-08   <  1e-8    <- 4.4x over
  Vertical_Harmonic_actionsFreqs   2.765e-09   <  1e-10   <- ~28x
  Vertical_Harmonic_..FreqsAngles  3.201e-09   <  1e-10   <- ~32x
  Spherical_linear_angles          5.009e-06   <  1e-8
  Spherical_linear_angles_fixed_quad 5.009e-06 <  1e-8
  Spherical_noninclined..._angles  1.180e-06   <  1e-8
  Spherical_almostnoninclined...   1.180e-06   <  1e-8
  AdiabaticGrid_outsidegrid_c      0.0047      <  1e-8    <- REAL, 6 orders out
```

**8 of 9 are tolerance-class; exactly one is a genuine numerical failure.**

The `StaeckelGrid` case is the sharpest: it demands a *computed action* satisfy
`Jr < 1e-16` for a circular orbit — below what float64 arithmetic can deliver for
anything but exact cancellation. numpy meets it by luck of operation order, the
same mechanism as the `test_quantity` 0.52-ULP case. `Adiabatic_linear_angles`
missing by 4.4x is nearly as stark.

**This roughly doubles the tolerance batch.** Previously #171 was "the 8
`test_quantity` entries plus ~20 assorted"; the measured picture is now ~8 +
~8 = **16 of 42 ledger entries are tolerance-class**, with magnitudes in hand for
every one. And it separates cleanly from the real work: `AdiabaticGrid_outsidegrid_c`
at 0.0047 is a genuine backend bug that no tolerance decision touches.

Still not acting — this is decision 2, and it now covers ~38% of the ledger, so
it matters more that Jo sets the rule than that I guess it. But the ledger is now
mostly *characterised* rather than merely counted, which is what makes the
decision cheap to make.

### 2026-08-04 — I was parking approved work: Group A is signed off, build it

Re-read the tolerance-batch memory before continuing to defer, and the premise
was wrong. Jo's 2026-08-01 answer **approved Group A**: build the shared
`assert_physical_matches_natural(...)` helper with a **relative** bound at the
affected `test_quantity` sites, conditional on the tests staying stringent. The
0.52-ULP cluster I have been calling "parked on decision 2" *is* Group A.

So for several wakes I have been carefully measuring, writing up, and declining
to act on something already signed off. The measuring was not wasted — the ULP
number is exactly the evidence the "stringent" condition needs — but the
conclusion "await Jo" was stale.

What actually still needs sign-off is the **actionAngle** half. Those entries
(`Vertical_Harmonic`, `Spherical_*linear_angles`, `StaeckelGrid`,
`Adiabatic_linear_angles`) were not in Group A ("~13 test_quantity entries") or
Group B ("GL-vs-adaptive + interpRZPotential grids") as scoped, and the standing
rule is explicit that a future batch needs its own approval. Recorded as such.

Also applying Jo's own red-flag criterion rather than quietly widening: 
`AdiabaticGrid_outsidegrid_c` at 0.0047 vs 1e-8 is six orders out. That is not a
tolerance question — it is the "really big difference is a red flag: investigate,
do not accept" case. It stays a real bug.

Plan: implement Group A against the measured magnitudes, one bound per site with
the measured value stated next to it, never a blanket number. Held behind #1248
so the queue stays one-PR-at-a-time.

### 2026-08-04 — CORRECTION: the test_quantity cluster is 3 causes, not 1; my "38%" was wrong

The C-enabled full-file `test_quantity` run settles two things and overturns one
of my own claims.

**1. The entries are genuine, not stale.** All 4 jax-ledgered sphericaldf tests
FAIL with the C extension present. Two of them "passed" in the no-C runs — the
same artifact as the actionAngle cluster, now seen twice.

**2. They do NOT share a root cause.** I wrote two wakes ago that this cluster
was "one root cause — absolute-vs-relative tolerance, 0.52 ULP", and sized the
tolerance batch on that. Reading the actual messages:

```
  method_value            assert 1.907e-06 < 1e-8          <- tolerance (the ULP case)
  method_inputAsQuantity  'numpy.float64' has no 'to'      <- Quantity TYPE
  sample_outputunits      'numpy.ndarray' has no 'to_value'<- Quantity TYPE
  densitymatch_issue677   UnitConversionError: subtract    <- UNITS
```

**Only 1 of the 4 is the tolerance issue.** The other three are the backend
returning a raw array where astropy expects a `Quantity` — a different bug class
entirely, and one that a tolerance rule cannot touch.

So my "~16 of 42 entries (≈38%) are tolerance-class" was overstated. Corrected
count: **~2 (test_quantity `method_value`, jax+torch) + ~8 (actionAngle) = ~10 of
42, ≈24%**. Still the largest single class, still worth one decision — but I
inflated it by generalising from one message to a whole cluster that merely
shared a filename prefix. The same "read the label, not the thing" error, applied
to a cluster instead of a test.

**3. A 5th, unledgered failure is a local-only artifact**:
`test_sphericaldf_quantity_radius_is_not_coerced_on_a_backend[torch]` fails with
`TypeError: unsupported operand for /: 'Tensor' and 'jaxlib ArrayImpl'` — a
`[torch]`-parametrized test running inside a forced-**jax** session. That is
[[ci-single-backend-install-blindspot]] in reverse: my box has both frameworks,
CI installs one per shard, so this cannot happen there. Not a bug, not a missing
ledger entry.

### 2026-08-04 — the 3 Quantity failures ARE one root cause; mechanism half-identified

The three non-tolerance `test_quantity` failures are one bug, not three: under a
forced backend a sphericaldf method that should return an astropy `Quantity`
returns a bare value. `.to` missing on a `float64`, `.to_value` missing on an
`ndarray`, and a `Quantity - non-Quantity` subtraction are three symptoms of that
one thing. 3 tests x 2 backends = **6 ledger entries on one fix** — better value
than the 2-entry tolerance item, and a real bug rather than a bound.

Mechanism, as far as it is *established*: `vmomentdensity` carries only
`@backend_input("r")` and applies units **inline** (the #157 "hoist units"
design), unlike `sigmar` which stacks `@backend_input` over
`@physical_conversion`. The inline branch is

```python
if use_physical and vo is not None and ro is not None:
    return units.Quantity(as_numpy(out) * fac, unit=u)   # correct, handles backend arrays
else:
    return exit_cast(out, r) * fac                       # <- what we are getting
```

so the code *can* return a Quantity and already calls `as_numpy`. The failure is
that it takes the `else` branch — i.e. `ro`/`vo`/`use_physical` are not what the
numpy path sees.

**Stopping the inference here deliberately.** I first guessed "missing
`@physical_conversion` decorator", and reading the body showed that was wrong —
the seventh time today a plausible read of nearby code was not the thing itself.
The remaining step is mechanical and cheap: print `use_physical`, `ro`, `vo` at
that branch under forced jax versus numpy and see which one differs. That is a
measurement, not a hypothesis, and it belongs in the next wake rather than
another guess in this one.

**Correction, same wake — measured instead of inferred.** Everything above about a
"Quantity-stripping bug class" is wrong, and the saved tracebacks that suggested it
were stale. Re-running the three tests today on `feat/spline-pickle-util` (C
extension confirmed loaded, `load_libgalpy()` non-None):

| test | today |
|---|---|
| `sphericaldf_method_inputAsQuantity` | **FAILS — value assert, not a type error** |
| `sphericaldf_sample_outputunits` | **passes** in isolation |
| `densitymatch_issue677` | **passes** in isolation |

The `AttributeError: 'numpy.float64' has no attribute 'to'` I had been reasoning
from is pytest *rendering the assertion expression*, not the exception. Both
calls return correct types:

```
vmomentdensity(1.1*8.0*kpc, 0, 2) -> <Quantity 1.64026634e+10 km2 solMass/(s2 kpc3)>
vmomentdensity(1.1, 0, 2)         -> Array(0.00192737, dtype=float64)
```

and `0.00192737 * 9.0027e10 / 8**3 * 220**2 = 1.6403e10` — they **agree**. The
assert is `|difference| < 1e-8` *absolute* on numbers of order 1e10. Numpy passes
because both sides are bit-identical (difference exactly 0); under jax a
last-digit wobble is ~1e4. So this is not a new bug class at all — it is a
**Group A tolerance case, which Jo already approved**, and the fix is the
`assert_physical_matches_natural` relative helper rather than anything in
`sphericaldf`.

Net effect on the plan: the cluster is **1 actionable entry (x2 backends), not 6**,
and it needs no sign-off. The other two are stale-entry *candidates* — passing in
isolation with C is real evidence but not proof, so they need a full-file run
before any prune (hand-picked nodeids are not a context).

Also worth stating plainly: I chased `vmomentdensity`, abandoned it for
`__call__`, and it was `vmomentdensity` all along — four probes that a single
`--junitxml` read at the start would have replaced. The rule already in memory
("re-read the failure first") earned itself again; the cost of skipping it was
this whole wake.

### 2026-08-04 (later) — the test_quantity cluster is ONE site, and the bound is unsatisfiable

Ran the two failing tests per-backend and read pytest's own assertion rendering
(after three hand-rolled repros each measured 0 ULP because they were unfaithful
— the last one passed `use_physical=False` on *both* sides, comparing a
computation to itself). Both `method_value` and `method_inputAsQuantity`, on
**both** jax and torch, fail at the *same* method: `vmomentdensity`.

The exact numbers, from `test_quantity.py:17893`:

```
|difference| = 1.9073486328125e-06     value = 1.64026634e+10
```

`1.9073486328125e-06` is exactly `2**-19`, and the float64 ULP at 1.64e10 (which
lies in [2^33, 2^34)) is `2**33 * 2**-52 = 2**-19`. So the physical and natural
results differ by **exactly one ULP** — the smallest representable nonzero
difference. Relative error `1.16e-16`.

The assertion is `|difference| < 1e-8` **absolute**. At this magnitude that is a
relative bound of `6e-19`, i.e. it demands agreement to ~0.000005 ULP. **No
implementation can satisfy it except one that is bit-identical to numpy.** This
is not a tolerance that jax/torch "fail" — it is a bound below the precision of
the type, which numpy passes only because both sides run the identical code.

That reframes the fix: replacing it with a relative bound is not loosening a
meaningful check, it is making an unsatisfiable check satisfiable. Proposed
`rtol=1e-15` (~8.6 ULP of headroom over the measured 1.16e-16) — stringent by
Jo's "not too far from the numerical limit" standard, and far tighter than the
1e-8 *absolute* it replaces for any value below ~1e7.

Scope: convert the `vmomentdensity` asserts only. The sibling `__call__` asserts
in the same tests measure **0 ULP** on all three backends (verified) and their
values are O(1e-9), where a 1e-8 absolute bound is already effectively relative —
converting those would tighten them and risk new reds for no burndown gain.

**Group A implemented — `feat/quantity-relative-tolerance` @ 49675d19 (not yet opened).**
13 `vmomentdensity` asserts in the two tests replaced by a shared
`assert_physical_matches_natural(physical, natural, msg, rtol=1e-15)` at module
level in `test_quantity.py` (single consumer — hoisting to a shared tests module
would be speculative; `tests/backend_jit_helpers.py` is jit-specific and is the
wrong home for a units check). 4 ledger entries dropped.

Control run, the part that makes the flip real: with the change reverted,
**2/2 fail on jax and 2/2 on torch**; with it, numpy/jax/torch all pass. Diff is
confined to 3 hunks (helper + the two functions), library code untouched.

Two process notes worth keeping. The first transform pass ran over the *whole
file* instead of the two functions — 19 blocks converted, syntax broken — because
I applied the regex to `src` rather than to the function spans. Reverted and
redone with explicit spans; the block count then matched the hand count (13),
which is the check that would have caught it either way. And `git stash` was
avoided for the control run in favour of a file copy, since stash is repo-wide
and could have restored another worktree's WIP.

Remaining in this cluster: `densitymatch_issue677` and `sample_outputunits`
(4 entries) — the full-file jax run is still going and will say whether they are
genuinely stale.

### 2026-08-04 — the 8 test_quantity entries are ONE bug, and isolation lied in both directions

The full-file jax run finished: **5 failures / 244**, and it rewrites the whole
picture.

| test | isolation | full file |
|---|---|---|
| `method_value` | fails @17893 | fails @17893 |
| `method_inputAsQuantity` | fails @**18044** | fails @**17999** |
| `sample_outputunits` | **passes** | **fails** @18183 |
| `densitymatch_issue677` | **passes** | **fails** @18257 |
| `quantity_radius_is_not_coerced[torch]` | — | fails (cross-framework artifact, CI-impossible) |

Two tests that passed in isolation fail in context, and a third fails at a
*different line*. So the earlier "stale candidates" reading was wrong — and the
rule that hand-picked nodeids are not a context earned itself twice in one run.

**The mechanism is a cascade from a single failure.** `test_sphericaldf_method_value`
does this:

```
17892:    galpy.util._optional_deps._APY_UNITS = False   # Hack
17893:    assert ... vmomentdensity ...                  # <- fails on jax/torch
17903:    galpy.util._optional_deps._APY_UNITS = True    # Hack (restore)
```

The failing assert sits **between** the disable and the restore. When it fails,
the restore never runs and the units flag stays `False` **for the rest of the
session**. Every later test that expects a `Quantity` then receives a bare value:

* `.to` on a `numpy.float64` → AttributeError
* `.to_value` on an `ndarray` → AttributeError
* `Quantity - dimensionless` → UnitConversionError

Those are exactly the three "Quantity-type bugs" I spent two wakes hypothesising
about. They are **collateral**, not independent defects — and the saved
tracebacks that started that hypothesis were real, just downstream of the actual
cause. numpy never sees any of it because on numpy the assert passes and line
17903 runs.

So the entire 8-entry `test_quantity` block reduces to **one root cause**: the
unsatisfiable absolute tolerance at 17893, already fixed in
`feat/quantity-relative-tolerance`. If the verification runs confirm it, the
remaining 4 entries come off too — 8 entries on one 13-line change.

Worth noting the test would be more robust with the flag restored in a
`finally:`/fixture so a failure cannot leak global state into unrelated tests.
Not doing that in this PR (it is a separate concern touching test structure
rather than the bound), but it is the reason one bad tolerance could red four
tests.

Process: the tolerance worktree had **no `libgalpy.so`**, caught before the run
rather than after — built it first, so these numbers are attributable.

**Stringency check on the new bound** (Jo's "pass not too far from the numerical
limit"). Exercised `rtol=1e-15` directly against relative perturbations:

| perturbation | verdict |
|---|---|
| 1.16e-16 (0.52 ULP) — the measured jax/torch difference | passes |
| 1e-15 (4.5 ULP) | **caught** |
| 1e-14 (45 ULP) | **caught** |
| 1e-12 (4504 ULP) | **caught** |

So it admits exactly the one-ULP round-off it is meant to admit and rejects
anything from ~4.5 ULP up. The crossover where the new relative bound becomes
*stricter* than the old absolute one is `1e-8 / 1e-15 = 1e7` — below that
magnitude the test is now tighter than it was, and the `vmomentdensity` values in
question are ~1.6e10, well above it.

Ledger edit verified against `origin/feat/backends-develop`: exactly 4 removals,
0 additions, no phantom entries.

Verification runs still in flight (jax 29%, torch 59%, **0 failures so far** in
both — the baseline's 5 failures all fell in the 59-88% band, so torch is only now
entering the decisive stretch).

### 2026-08-04 — torch verification CONFIRMS the cascade fix (4/4)

Full-file `test_quantity.py` under forced torch on the fixed branch (C extension
built first): **244 tests, 2 failures**, and all four target tests are recorded
as genuine PASSes, not skips:

```
test_sphericaldf_method_value            PASSED (0.055s)
test_sphericaldf_method_inputAsQuantity  PASSED (0.047s)
test_sphericaldf_sample_outputunits      PASSED (0.051s)
test_sphericaldf_densitymatch_issue677   PASSED (0.004s)
```

Baseline was 5 failures; the 2 that remain are both known non-regressions:

1. `quantity_radius_is_not_coerced_on_a_backend[jax]` — a `[jax]`-parametrized
   test in a forced-**torch** session. CI-impossible (one framework per shard).
2. `test_streamgapdf_sample` — named *by the ledger's own header comment* as an
   env-specific local false failure: "a local clean-CPU regen had env-specific
   false failures, e.g. test_streamgapdf_sample, that CI passes". Checking the
   comment before treating it as a regression saved chasing a phantom; the
   docstring/comment really does say whether a failure was expected.

So the 4 collateral entries fall out of one 13-line tolerance change, exactly as
the `_APY_UNITS` cascade predicted. jax is still running (59%, 0 failures so far)
— the ledger prune for the remaining entries waits on it rather than being
inferred from torch.

### 2026-08-04 — #1248 codecov/project: confirmed partial-upload, not a coverage loss

Rather than keep asserting "it's the upload artifact" from memory, read the
codecov comment. It states it outright:

> **HEAD has 9 uploads less than BASE** — 18 vs 27
> All modified and coverable lines are covered by tests

The reported `-14.53%` (48324 -> 41311 hits) is 9 shards that have not uploaded
yet, with 46 jobs still queued and 24 running. `codecov/patch` is already green.
So: wait, do not re-run, do not ping. This matches the documented behaviour that
`state: complete` is not an all-shards-in signal.

**Queue is conflict-free.** Checked the file sets of the three PRs now stacked up:

| PR | files |
|---|---|
| #1248 (active) | `_pickle.py`, Multipole, SCF, streamgapdf + their 3 test files |
| #192 (ready) | `planarPotential.py`, `test_backend_array_inputs.py` |
| tolerance (ready) | `test_quantity.py`, `backend_xfail.txt` |

Fully disjoint, so no rebase is warranted — a stale base is harmless when the
files do not overlap, and rebasing reflexively just burns CI on a
runner-constrained project.

jax verification still running (59%, 0 failures). Confirmed it is progressing
rather than hung — the wrapper PID shows 0% CPU, but that is just `timeout`; the
real work is in the child, and the log was written 30s before checking.

### Ledger burndown snapshot (on the tolerance branch, 4 already dropped)

**38 entries** — jax 17, torch 17, jax-jit 3, torch-jit 1. Once the remaining 4
`test_quantity` entries come off (pending the jax run), **34**.

| test file | entries |
|---|---|
| `test_actionAngle.py` | **12** |
| `test_quantity.py` | 4 (about to be 0) |
| `test_orbit.py` | 4 |
| `test_streamdf.py` | 3 |
| `test_potential.py` | 3 |
| `test_diskdf.py` | 3 |
| `test_sphericaldf.py` | 2 |
| `test_qdf.py` | 2 |

So after this PR the **single largest remaining cluster is `test_actionAngle` at
12** — and ~8 of those are the tolerance batch that needs Jo's own sign-off (a new
batch; the 2026-08-01 approval covered Group A/B, not this). That is the natural
next thing to raise when he is reachable, and it is worth noting it is now a third
of the whole ledger.

Separately: `backend_slow_skip.txt` holds **230** entries — an order of magnitude
larger than the xfail ledger, and mostly jax eager-XLA timeouts. Worth flagging
that "ledger size" as a progress metric only tracks the xfail file; the slow-skip
list is where the jax-speed debt actually sits, and it is not shrinking on its own.

### 2026-08-04 — jax CONFIRMS too; tolerance branch complete (8 entries, one root cause)

```
jax:   244 tests, 1 failure  (was 5)   all 4 targets PASSED
torch: 244 tests, 2 failures (was 5)   all 4 targets PASSED
```

The single jax residual is `quantity_radius_is_not_coerced_on_a_backend[torch]` —
a `[torch]`-parametrized test in a forced-**jax** session, which cannot occur in
CI where each shard installs one framework. torch's extra one is
`test_streamgapdf_sample`, named in the ledger's own header as a local
env-specific false failure that CI passes.

`feat/quantity-relative-tolerance` is now 2 commits and **removes 8 ledger entries
(8 removed / 0 added vs origin)** off a 13-line test change — the best
entries-per-line of the vacation so far, and only because the four "separate"
failures turned out to be one cascade.

The general lesson, worth carrying: **a test that mutates global state and fails
before restoring it converts one red into many.** The three downstream failures
looked like three distinct Quantity-type bugs for two full wakes. Nothing about
their tracebacks hinted at a shared cause; only running the whole file and
noticing the line-number shift did. Where a suite has this pattern, the *first*
failure in file order is the one to fix, and the rest should be re-measured
before being characterised at all.

### 2026-08-04 — #192 base check, and a diff direction that lies

Before opening #192 I checked whether develop had moved under it. First attempt:

```
git diff --name-only HEAD..origin/feat/backends-develop
  -> galpy/potential/planarPotential.py
     tests/test_backend_array_inputs.py
```

which reads as "develop touched exactly my two files — must rebase". **That is
wrong.** `HEAD..develop` is a symmetric file list: it includes *my own* changes
seen from develop's side, so a branch's own files always appear. It cannot
distinguish "they changed it" from "I changed it".

The question only has meaning against the merge-base:

```
git diff --name-only $(git merge-base HEAD origin/...)..origin/feat/backends-develop
  -> tests/backend_jit_helpers.py, tests/test_backend*.py  (9 files, none of mine)
```

No overlap, so the stale base is harmless and #192 needs no rebase — which is the
outcome the "don't reflexively rebase disjoint PRs" rule predicts, but only the
merge-base diff actually establishes it. Rebasing on the strength of the first
command would have burned CI on a runner-constrained project for nothing.

**codecov on #1248 is converging as predicted**: uploads 18/27 -> **23/27**, with
21 jobs still queued. Not stuck, no intervention.

Dispatched a jit-coverage run (`jitcount` plugin) on `test_sphericaldf.py` under
`--backend jax --jit` to measure traced progress on the DF work — the counters are
traced-vs-NOT_TRACED boundary calls plus C-library entries, since a `--jit` pass
count on its own proves neither.

### 2026-08-04 — hardening folded in (3rd commit), and a probe that lied first

Wrapped the `_APY_UNITS` disable/restore in `try/finally` so a failure can never
leak the flag again. Same file as the tolerance work, so it belongs on that
branch rather than as a conflicting follow-up PR.

**The first verification was wrong and nearly got recorded as a success.** I
triggered the simulated failure on "physical value has no `.unit`", assuming that
only happened inside the flag-off region. It does not — an earlier converted
assert also passes a raw value, so the probe raised *before* line 17894, the flag
was never set False, and **both** arms reported `True`. Read naively that says
"hardening works"; it actually says "the probe never reached the code under
test". A control that passes identically with and without the fix is not a
control.

Re-triggered on the flag state itself (`if not od._APY_UNITS`), which is exact:

| | raised in flag-off region | flag after failure |
|---|---|---|
| with `try/finally` | yes | **True** (restored) |
| without (HEAD) | yes | **False** (leaked) |

`feat/quantity-relative-tolerance` is now 3 commits: relative bound, -8 ledger
entries, unconditional restore. numpy/jax/torch all green.

### 2026-08-04 — #1248 pre-merge audit (the three required dimensions)

**1. `if` -> `xp.where` array-input class: clean.** Every added conditional
branches on a *type or a name*, never on array data:
`isinstance(obj, (PPoly, BPoly))`, `isinstance(obj, list)`, `if name in pdict`,
`if name in self.__dict__`. Nothing here can break under a trace or with array
inputs, because no array value ever reaches a truth test.

**2. Duplication: eliminated, not moved.** `pack_splines` / `unpack_splines` /
`SplinePickleMixin` are defined once in `galpy/util/_pickle.py`; the three
consumers inherit. `MultipoleExpansionPotential` is **-49 lines net** because its
hand-rolled `__getstate__`/`__setstate__` were deleted rather than relocated.

**3. Placement: correct, and forced.** The consumers span `potential/`
(Multipole, SCF) *and* `df/` (streamgapdf), so the helper cannot live in either
family — `galpy/util/` is the only coherent home.

**Known latent edge, deliberately NOT fixed:**
`isinstance(obj, tuple) and len(obj) == 6 and obj[0] == _SPLINE_SURROGATE` would
raise numpy's ambiguous-truth error if `obj[0]` were an array. It is unreachable:
`unpack_splines` is only ever applied to attributes named in
`_PICKLE_SPLINE_ATTRS`, which are splines by construction. Fixing it would
restart ~150 checks on a runner-constrained project for a case that cannot
occur. Recording it here instead so it is a known quantity rather than a
surprise if those attr lists ever widen.

Dispatched a full-file **numpy** `test_quantity.py` run on the tolerance branch —
the branch edits an 18k-line file and ruff reformatted it, and so far numpy has
only been checked on the 2 affected tests. Cheap insurance against a
transform side-effect elsewhere in the file.

### 2026-08-04 — jit-coverage measurement: test_sphericaldf under jax --jit

```
BOUNDARY CALLS: traced=152293  not_traced=375   (99.8% traced)
C-LIBRARY ENTRIES: 0
NOT-TRACEABLE (entered C): 0 tests
```

223 slots: **194 passed, 28 skipped (12.6%), 1 xpassed, 0 failed**.

Read honestly, that headline needs its denominator attached: **99.8% of boundary
calls are traced across the 87% of the suite that actually runs under jax.** The
28 skips are the jax slow-skip entries — i.e. exactly the tests most likely to be
slow or awkward under a trace — so they are excluded by construction, not because
they pass. The percentage is real but it is not "the spherical DF suite is 99.8%
jit-clean".

What *is* unambiguously good: **0 C-library entries**, so nothing in this suite is
not-traceable by construction. The whole DF path stays in Python/backend code and
never drops into libgalpy, which is the precondition for GPU and for
higher-order AD. The residual 375 untraced boundary calls (0.2%) are the real
remaining gap and are worth a follow-up characterisation.

Also surfaced: **1 XPASS** in this run — an xfail that now passes under
`--backend jax --jit`. Possibly a stale jit-ledger entry (there are only 3
`jax-jit` entries total). Noting it as a lead rather than chasing it now; it needs
`--runxfail` and a full-file context to identify properly, and #1248 is the
priority.

Also: full-file **numpy** `test_quantity.py` on the tolerance branch = **244
passed**, so the 18k-line edit plus ruff reformat introduced no numpy regression
anywhere in the file, not just in the two tests I had checked.

**The XPASS in the jit run is unexplained — recording it as open, not solved.**
Chased it three ways and came up empty:

* not a jit-ledger entry — the only `jax-jit` entries are 1 `test_orbit` and 2
  `test_potential[AnyAxisym]`; none in `test_sphericaldf.py`
* not a plain-`jax` entry leaking in — `conftest._run_backend` returns
  `f"{name}-jit"` under `--jit`, so the `jax` list is not consulted at all
* no in-file `xfail` marks — `grep xfail tests/test_sphericaldf.py` is empty

And `--jit` was definitely active (traced=152293 boundary calls; the counter only
moves under a trace). So one test is being marked xfail by a mechanism I have not
found. Identifying it needs a targeted `-rX` rerun of that suite (~20 min), which
is not worth pre-empting the #1248 merge for. Flagging it because an unexplained
xfail marker is exactly the kind of thing that silently hides a real failure
later.

### 2026-08-04 — #1248 is green but BLOCKED on the gate; verified "slow" is not "hung"

146 checks green, **zero reds** (codecov/project cleared on its own at 99.95%).
`mergeStateStatus=BLOCKED` only because `all-clear` has not reported yet.

Before assuming the remaining shards were stuck, measured rather than guessed:

| | |
|---|---|
| completed orbit shards **in this same run** | 9-51 min (ubuntu 3.14 orbit+orbits: 31 min) |
| still-running ubuntu 3.13 / 3.12 orbit+orbits | 52 min |
| `timeout-minutes` in build.yml | **none** -> GitHub default 6h |
| concurrent workflow runs right now | 3 |

So they are long but inside the observed range, with runner contention the likely
cause, and there is no configured limit they are about to trip. Re-running would
restart from zero and burn runner time on a project that is runner-constrained —
waiting is strictly better. Nothing to drive, nothing to ping about.

## 2026-08-04 — #1248 MERGED (730657c3); #1249 opened; XPASS explained (I was wrong)

`gh pr merge --squash` landed the scipy-1.18 pickle work on `feat/backends-develop`
as **730657c3**, 152 checks green, zero reds. `codecov/project` cleared itself
once all 27 shard uploads arrived, exactly as diagnosed — no re-runs, no ping.

Immediately opened the next in the queue: **#1249** (planar adapters derive
`_backend_compatible` from the potential they adapt).

**Correction — the XPASS is intended behaviour, and my previous explanation was
wrong.** Last wake I wrote that under `--jit` "the `jax` list is not consulted at
all", inferring it from `_run_backend` returning `"jax-jit"`. The plugin named the
culprit — `test_anisotropic_hernquist_negdf`, marker reason `backend-xfail-ledger`
— which sent me to the loader I had never read:

```python
names = {backend_name}
if backend_name.endswith("-jit"):
    names.add(backend_name[: -len("-jit")])      # traced INHERITS eager
```

with a docstring spelling out the rationale: tracing cannot repair an eager
failure (not a numerical gap, not a `_reject_backend` guard), so `<backend>-jit`
means **"applies ONLY when traced"** and the eager list stays the single place a
shared entry is written and pruned. It even says the rule lives in the loader
rather than the callers "because it is a property of the backend name, not of
which list is being read."

So: an eager-jax entry, inherited under `--jit`, that passes when traced →
non-strict XPASS. Nothing hidden, nothing stale, no follow-up needed. My stated
worry that "an unexplained xfail marker could silently hide a real failure" was
unfounded.

The mistake was reading one function and generalising from its return value
instead of following the call into the function that consumes it. The
already-recorded rule — read the docs/comments before classifying — would have
answered this in one step, since the answer was sitting in a docstring the whole
time.

### 2026-08-04 — actionAngle cluster triage; AdiabaticGrid_outsidegrid_c is BY DESIGN (correcting myself again)

#1249 is queued (149 checks), so: local burndown triage on the biggest remaining
cluster, `test_actionAngle` at 12 entries / 5 distinct groups.

Started with `AdiabaticGrid_outsidegrid_c` (jax+torch) because I had it recorded
as "a real bug per Jo's own red-flag criterion, not a tolerance candidate".
Re-measured today:

```
grid   aA(3.0, ...)[0] = 0.01111777
direct aAA(3.0,...)[0] = 0.0064088
|diff| = 0.00470897   vs tolerance 1e-8      (~73% relative)
```

Far too large for tolerance — that part was right. But it is **not a bug
either**, and the source says so in two places:

> "interpolation grid only (no off-grid fallback): inputs must lie within the
> grid, whereas the numpy path falls back to a per-point solve off-grid"
> "Assumes on-grid inputs (the off-grid fallback to self._aA is numpy-only)"

The test deliberately evaluates at `R=3.0`, *outside* the grid. numpy falls back
to an exact per-point solve; the backend path returns the extrapolated grid
value. The 73% gap is the documented consequence of a deliberate design choice,
not a defect.

**Consequences for the ledger and for what Jo is asked to approve:**

* these 2 entries are **correct and should stay** — they are not stale, not
  tolerance, not fixable by a bound
* they must **not** go into the tolerance batch; putting them there would ask Jo
  to approve loosening a bound that is doing its job
* they clear only when task #103 (opt-in off-grid fallback for jax/torch) is
  implemented — that is a *feature*, not burndown

Third time this session that reading the docstring settled a classification I had
already labelled from numbers alone (after `test_streamgapdf_sample` and the
`-jit` ledger inheritance). Ratios rank candidates; they do not classify them.
The docstring is the only thing that says whether a difference was intended.

**actionAngle triage, interim (full-file torch run in flight).** Docstring
evidence gathered so far, which splits the 12-entry cluster by *kind* rather than
by magnitude:

| group | entries | status |
|---|---|---|
| `AdiabaticGrid_outsidegrid_c` | 2 (jax+torch) | **BY DESIGN** — confirmed, no off-grid fallback |
| `StaeckelGrid_basicAndConserved_actions` | 2 (jax+torch) | **likely by design** — `actionAngleStaeckelGrid.py:42` carries the *identical* "no off-grid fallback" sentence; needs the failure detail to confirm the test actually evaluates off-grid |
| `Vertical_Harmonic_actionsFreqs{,Angles}` | 2 (torch) | probably a real gap — `_unbound_backend` explicitly *tries* to match numpy's 9999.99 sentinel ("match it for the vectorised path"), so a mismatch is a defect, not a decision |
| `Spherical_*_linear_angles` | 4 (torch) | unknown |
| `Adiabatic_linear_angles_cylsep` | 2 (jax+torch) | unknown |

The useful distinction is that "by design" and "real gap" need opposite
treatments: the first stays ledgered until a *feature* lands and must be kept out
of the tolerance batch; the second is burndown I can do now. Sorting by failure
magnitude would have mixed them together — the AdiabaticGrid one is the single
largest discrepancy in the cluster and is also the one that must NOT be touched.

**Triage refined.** `actionAngleSpherical.py` contains **no** numpy-only
divergence note — every comment there is of the form "the numpy path is untouched
(byte-identical)", i.e. the backend path is *meant* to agree. That puts the 4
`Spherical_*_linear_angles` torch entries in the "real gap" column rather than
"by design", and makes them the largest genuinely-fixable group in the cluster.
(They are also the 4 that the missing-C-extension trap flips to false PASSes, so
any future run attributing them must have `libgalpy.so` built.)

Running total for the 12-entry actionAngle cluster:

* **by design, keep ledgered** (blocked on feature #103): AdiabaticGrid 2, StaeckelGrid 2 (pending confirmation)
* **real gap, fixable**: Spherical 4, Vertical_Harmonic 2
* **unknown**: Adiabatic_linear_angles_cylsep 2

So at most ~6 of the 12 are burndown, and the tolerance batch Jo is asked to sign
off on is smaller than the raw cluster size suggests. The full-file torch run
(9 failures = exactly the 9 torch entries, so the ledger is exactly right) will
supply per-test detail.

### 2026-08-04 — actionAngle torch full-file: 9 failures = 9 entries, and two of my guesses were wrong

209 tests, **9 failures — exactly the 9 torch actionAngle ledger entries**, so the
ledger is precisely right for this file (no stale entries, nothing unledgered).

| group | measured vs bound | ratio | class |
|---|---|---|---|
| `Vertical_Harmonic_actionsFreqs{,Angles}` (2) | 2.76e-9 / 3.20e-9 vs **1e-10** | ~30x | tolerance |
| `Spherical_linear_angles{,_fixed_quad}` (2) | 5.009e-6 vs 1e-8 | **500x** | **red flag — real accuracy gap** |
| `Spherical_{,almost}noninclinedorbit_linear_angles` (2) | 1.180e-6 vs 1e-8 | **118x** | **red flag — real accuracy gap** |
| `Adiabatic_linear_angles_cylsep` (1 torch) | 4.359e-8 vs 1e-8 | 4.4x | tolerance |
| `AdiabaticGrid_outsidegrid_c` (1 torch) | 4.7e-3 vs 1e-8 | 5e5x | **by design** |
| `StaeckelGrid_basicAndConserved_actions` (1 torch) | 4.835e-14 vs **1e-16** | 218x | cancellation |

**Two corrections to my own earlier triage:**

1. `StaeckelGrid_basicAndConserved_actions` is **not** the off-grid design case. I
   guessed that from the identical "no off-grid fallback" docstring in
   `actionAngleStaeckelGrid.py`. The actual failure is *"Circular orbit in the
   MWPotential does not have Jr=0"* — `Jr = 4.8e-14` against a **1e-16** bound.
   That is a cancellation/summation-order question for a quantity that should be
   exactly zero, nothing to do with grid coverage. The shared docstring made a
   plausible story; the failure text made the true one.

2. `Vertical_Harmonic` is a **tolerance** case (~30x on a 1e-10 bound), not the
   sentinel defect I inferred from `_unbound_backend`'s "match it for the
   vectorised path" comment. Being near code that discusses a divergence is not
   evidence of that divergence.

**What this means for the batch Jo is asked to approve.** Applying his own
"a really big difference is a red flag" rule to measured numbers rather than to
my labels:

* **tolerance batch (~5-6 entries)**: Vertical_Harmonic (2), Adiabatic_cylsep
  (1 torch + 1 jax), StaeckelGrid (1 torch + 1 jax) — all 4x-218x on bounds that
  are at or below float64 resolution for the quantity involved (1e-16 for a
  cancellation-to-zero, 1e-10 for a frequency)
* **NOT tolerance, needs a real fix (4 entries)**: the Spherical linear-angle
  family at 118x-500x. These are the largest genuinely-fixable group and I should
  work them rather than ask for a bound change.
* **NOT tolerance, blocked on feature #103 (2 entries)**: AdiabaticGrid off-grid.

So the honest ask is ~6 entries, not the ~8 I had been carrying, and 4 of the
cluster are work for me rather than a decision for him.

### 2026-08-04 — ROOT CAUSE for the 4 Spherical linear-angle entries: bisection step count

Isolated the failure away from the test harness by computing the radial angles for
**identical numpy inputs** under each backend (orbit integrated once with
`dopr54_c`, so both paths see the same 1001 points):

```
numpy  max deviation from linear = 2.27e-09      (passes 1e-8)
torch                            = 3.34e-06      (fails)
max |torch - numpy| = 3.34e-06,  and ALL 1001 angles differ (not an outlier)
```

The magnitude is the tell. `2**-18 = 3.815e-06` versus the observed `3.34e-06` —
the error sits exactly at an 18-halving bisection floor. And the backend root
finder is precisely that:

* `galpy/backend/optimize.py:178` `bisect_root(f, a, b, xp, *, xtol, maxiter)`,
  docstring: *"Runs a FIXED schedule of `n_bisect_steps` halvings"* — static
  schedule so it stays jit-traceable
* step count is `n_bisect_steps(a, b, xtol, maxiter)`, derived from bracket width
  and `xtol`
* the numpy path instead uses a brentq-style solve with `maxiter=200`
  (`actionAngleSpherical.py:895`), i.e. machine precision

So the turning-point root is only located to ~4e-6 on the backend path, and the
radial angle inherits that error wholesale. This is **not** a tolerance question
and needs no sign-off from Jo: it is a resolution shortfall in a solver, with a
cheap fix (a halving costs one function evaluation; ~52 reach float64
resolution, and the schedule stays static so tracing is unaffected).

**Next step**: find the `xtol` the Spherical angle path passes, tighten it, and
verify (a) the deviation drops below 1e-8 on torch AND jax, (b) numpy stays
byte-identical, (c) the cost in added halvings is small. That would clear **4 of
the 12** actionAngle entries — the largest fixable group in the cluster.

This also revises the tolerance ask downward again: of the original 12, ~6 are a
genuine tolerance decision for Jo, 4 are this solver fix, and 2 are blocked on
feature #103.

### 2026-08-04 — RETRACTION: the Spherical angle error is NOT the bisection; it is the GL quadrature floor

Last wake I logged a section headed "ROOT CAUSE ... bisection step count". **That
was wrong**, and two measurements today refute it:

1. **The root finder is exact.** rperi/rap, torch vs numpy, over all 1001 points:
   `max|Δrperi| = 7.3e-13`, `max|Δrap| = 6.9e-13`. The turning points agree to
   ~1e-12, so nothing about the bisection can produce a 3.3e-6 angle error.
2. **The call site never sets a loose tolerance.** `actionAngleSpherical` calls
   `_backend_brentq(f, ...)` with no `xtol`, i.e. scipy's default `2e-12`, plus a
   Newton polish — consistent with the 7e-13 measured above.

What I actually did last wake was notice `3.34e-6 ≈ 2**-18` and reason backwards
to a solver that could produce it. That is numerology: a value near a power of
two is not evidence for a halving schedule, and I never checked whether the
solver was even on the path. `actionAngleSpherical.py` contains no `bisect_root`
call at all — one grep would have killed the hypothesis before I wrote it up.

**The real mechanism, and it is documented in the source:**

```python
# Gauss-Legendre order for the backend (jax/torch) action/freq/angle quadratures.
# 50 matches scipy's adaptive quadrature to <1e-7 over the physical orbit range
# ... an irreducible quadrature-method difference where higher order does not help.
_BACKEND_GL_ORDER = 50
```

The backend uses fixed-order GL-50; numpy uses scipy's *adaptive* quad. The
comment states the agreement floor is `<1e-7` and that **raising the order does
not help** — so the proposed "tighten it" fix was not just aimed at the wrong
component, it is explicitly documented as ineffective for the right one.

The test bound is `1e-8`. Against a documented ~1e-7 method floor, **the test is
structurally unattainable for any GL order** — the same shape as the
`test_quantity` case: a bound below what the method can deliver.

**Revised classification, and the ask goes UP not down.** I have twice reported a
shrinking number for Jo; honesty requires reporting the increase:

* the 4 Spherical linear-angle entries are a **quadrature-method floor**, i.e. a
  tolerance/reference decision, **not** work I can silently fix
* so the tolerance batch is **~10 entries**, not the ~6 I claimed last wake
* the genuine options are (a) a backend-specific bound reflecting the GL-vs-
  adaptive floor, or (b) re-basing the test on a gold reference instead of
  treating scipy's adaptive quad as truth — (b) is the better test but a larger
  change, and either way it is Jo's call

Pattern worth keeping: I reached for a mechanism that *could* explain the number
before checking which mechanisms were *present*. Both times this session that I
did that (the vmomentdensity decorator, and now this) the answer was in a comment
a few lines from code I had already opened.

**The cluster partly unifies, and a duplication falls out of it.**

Which actionAngle modules use the fixed-order GL quadrature at all:

| module | GL references | `_BACKEND_GL_ORDER` |
|---|---|---|
| `actionAngleSpherical` | 4 | **defines `= 50` at line 33** |
| `actionAngleVertical` | 4 | **defines `= 50` at line 22** |
| `actionAngleAdiabatic` | 0 | — |
| `actionAngleStaeckel` | 0 | — |

So **6 of the 12 entries** (Spherical 4 + Vertical_Harmonic 2) sit on the *same*
GL-vs-adaptive floor and are therefore **one decision**, not six. The Adiabatic
and StaeckelGrid entries use no GL quadrature and are separate mechanisms —
which also retro-justifies not lumping them into a single "tolerance batch".

**Duplication found (actionable, independent of the tolerance question):** the
same constant `_BACKEND_GL_ORDER = 50` is defined *twice*, in two potential-family
modules, with the same value and the same rationale comment. That is precisely
the "general-ish value living in specific files" pattern the audit directive
targets: the quadrature order is a property of the backend quadrature, not of
spherical or vertical action-angles. It belongs in the shared backend quadrature
module, imported by both — so that changing it (which the tolerance decision may
require) is one edit, not a hunt for copies.

**One caveat on the documented floor.** The comment claims GL-50 matches scipy's
adaptive quad to `<1e-7` "over the physical orbit range", but the measured
*angle* discrepancy is `3.34e-6` — 30x larger — for an unremarkable orbit
(`[1.1, 0.3, 1.2, 0.2, 0.5, 2.0]`). Either that orbit is nearer the pathological
regime than the comment implies, or the floor for *angles* is worse than for
actions (plausible: the angle integrand carries the same sqrt turning-point
behaviour but is evaluated on a partial interval). Worth measuring before quoting
`1e-7` to Jo as the number, since the decision depends on where the floor
actually is.

### 2026-08-04 — tested the "higher order does not help" claim instead of trusting it

Swept `_BACKEND_GL_ORDER` on the failing Spherical angle case:

| GL order | max deviation from linear |
|---|---|
| 50 (current) | 3.3407e-06 |
| 100 | 2.1083e-06 |
| 200 | 1.8452e-06 |
| **400** | **3.5152e-06** — *worse* |

The error does not converge with order: it plateaus around ~2e-6 and then
degrades as round-off in the 400-node sum overtakes the quadrature gain. So the
source comment's claim is **correct, and now verified rather than assumed** —
raising the order is not a fix. (The integrand carries a sqrt turning-point
singularity, for which plain Gauss-Legendre converges very slowly; that is
exactly the regime where adding nodes buys nothing.)

**But the documented number does not apply here.** The comment says the GL-50
floor is `<1e-7`; that is the floor for *actions*. For *angles* the measured floor
is ~2e-6 — about **20x worse** — and no order reaches the test's 1e-8 bound. Worth
correcting in the comment, since anyone reading it would predict this test could
be made to pass by raising the order.

**Settled classification for the 4 Spherical entries**: an irreducible
GL-vs-adaptive method floor at ~2e-6 for angles. Not a solver bug (turning points
agree to 7e-13), not fixable by order (measured above). It is a genuine
tolerance/reference decision for Jo, alongside the 2 Vertical_Harmonic entries
that share the same GL-50 machinery — **6 entries, one decision**.

The useful discipline here: the comment made a falsifiable claim, so I tested it.
It survived — which is a better outcome than either trusting it blindly or (as I
did last wake) inventing a competing mechanism without checking whether the
documented one held.

### 2026-08-04 — hoisted the duplicated GL order (branch feat/hoist-gl-order, not opened)

`_BACKEND_GL_ORDER = 50` was defined in **two** actionAngle modules, the second
with the comment "(matches actionAngleSpherical's choice)" — coupled by
convention, not by code. Five AA modules use the backend quadrature helpers,
three of them via the shared `n=50` default, so the value effectively had three
homes.

Moved it to `galpy/actionAngle/actionAngle.py`, which both modules already import
from. **Deliberately not** `galpy/backend/quadrature.py`: the order is a property
of the action-angle quadratures, and putting it in the generic backend module
would invert the dependency — the audit rule is "hoist to where it belongs", not
"hoist as far up as possible".

Also corrected the rationale comment, which claimed a `<1e-7` floor. That figure
is for *actions*; for *angles* it is ~2e-6, with the measured non-convergence
(3.3e-6 / 2.1e-6 / 1.8e-6 / 3.5e-6 at n=50/100/200/400) written into the comment
so the next reader does not repeat my mistake of assuming a higher order would
close the gap.

Pure relocation, numpy path untouched, 7 AA tests green.

### 2026-08-04 — StaeckelGrid Jr=0: another sub-epsilon bound, triage complete

```
numpy  Jr = 0.000000e+00   Jz = 0.0     (bound 1e-16)
torch  Jr = 3.215700e-12   Jz = 0.0
float64 eps = 2.220e-16
```

The bound `< 1e-16` is **below machine epsilon**. numpy passes because its path
returns an *exact* zero by construction — not because it computes something small
— so the test effectively demands bit-exact zero. Any implementation that reaches
Jr through quadrature and lands at 1e-12 fails, even though 1e-12 is an excellent
Jr for a circular orbit in absolute terms.

Same structural shape as the `test_quantity` case already fixed: a bound tighter
than the representation allows, which only one specific implementation satisfies.

(Note the magnitude is context-dependent: CI recorded 4.8e-14, this isolated run
3.2e-12. Same class, different value — so quote the *class*, not the number.)

**actionAngle cluster triage — COMPLETE (12 entries):**

| kind | entries | disposition |
|---|---|---|
| GL-vs-adaptive quadrature floor (~2e-6 for angles) | 6 | Jo's decision; verified not fixable by order |
| sub-epsilon bound, numpy returns exact 0 | 2 | Jo's decision; unsatisfiable as written |
| off-grid, no backend fallback (documented) | 2 | blocked on feature #103, keep ledgered |
| `Adiabatic_linear_angles_cylsep` (4.4x) | 2 | still uncharacterised |

So **~8-10 entries hinge on Jo**, 2 are blocked on a feature, and — contrary to
what I claimed two wakes ago — **none are a solver bug I can quietly fix**. Every
"fixable gap" in this cluster dissolved on measurement into either a documented
design choice or a bound below what the method/type can deliver.

**Last group resolved: `Adiabatic_linear_angles_cylsep` is the SAME quadrature floor.**
`actionAngleAdiabatic` imports `actionAngleSpherical`, but an import is not a
mechanism — that inference is what I got wrong on StaeckelGrid. So I read the
call instead:

```python
# actionAngleAdiabatic._actionsFreqsAngles_backend
Jr, Lz, _, Or, Op, _, ar, aphi, _ = self._aAS._actionsFreqsAngles(...)
```

The radial angle `ar` *is* actionAngleSpherical's, computed by the same GL-50
quadrature. The smaller magnitude (4.4e-8 vs 3.3e-6) is not a different cause —
the floor is orbit-dependent, which the order sweep already showed.

### actionAngle cluster — FINAL (12 entries, 3 dispositions)

| disposition | entries | detail |
|---|---|---|
| **GL-vs-adaptive quadrature floor** | **8** | Spherical 4 (direct), Adiabatic 2 (delegates to Spherical), Vertical 2 (own GL-50). ONE decision, two code sites. Verified not fixable by raising the order. |
| **sub-epsilon bound** | 2 | StaeckelGrid Jr=0: bound 1e-16 < eps; numpy returns exact 0 |
| **documented design** | 2 | AdiabaticGrid off-grid; clears only with feature #103 |

So the whole 12-entry cluster reduces to **2 decisions for Jo covering 10
entries**, plus 2 blocked on a feature. Nothing here is a defect.

Convenient tie-in: the `feat/hoist-gl-order` branch prepared this wake puts the
constant that **8 of these 12** depend on in a single place — so if the decision
is "raise or parameterise the order", it is now one edit rather than two files
plus three implicit defaults.

### 2026-08-04 — free staleness sweep from #1248's artifacts: ledger is NOT carrying dead weight

Pulled the 29 `backend-junit-*` artifacts from the completed #1248 backend-tests
run and intersected them with the ledger.

| backend | ledger | matched in artifacts | PASSED (= stale) |
|---|---|---|---|
| torch | 19 | **19 / 19** | **0** |
| jax | 19 | 10 / 19 | **0** |

**torch is a complete sweep and finds nothing stale**; jax is partial (9 entries
never appear in these artifacts — presumably slow-skipped, since conftest lets
skip win over the xfail-ledger for the same nodeid). So there is no pruning
opportunity here: burndown has to come from real fixes or decisions, not from
deleting entries that already pass. That is consistent with the actionAngle
measurement (9 torch failures = exactly 9 entries).

**The first version of this sweep was silently broken, and validating it is the
only reason I know.** junit has no `file` attribute, so the nodeid has to be
rebuilt from `classname` — and `classname` is `tests.test_actionAngle`, i.e. the
FIRST component is the *directory*. Taking `split(".")[0]` produced
`tests/tests.py::test_...` for every single test, which matches no ledger entry,
so the sweep reported a confident **"0 stale on both backends"** while comparing
against nothing at all. Same shape as a green test that never executed the path
it claims to cover: the reassuring number and the broken number are identical.
Fixed by taking the first component that starts with `test_`.

**Census headline worth keeping** (from `backend-status-report`):

```
jax:   3210 passed · 18 xfail · 334 deferred · 34 exempt
torch: 3528 passed · 19 xfail ·  15 deferred · 34 exempt
```

The xfail ledger is 42 entries, but jax carries **334 deferred** tests against
torch's 15 — a 22x asymmetry. The headline burndown metric everyone watches (the
ledger) is the *small* number; the real remaining debt is jax eager-XLA speed,
which is 8x larger than the entire ledger and does not shrink on its own.

### 2026-08-04 — where the jax deferral debt actually sits (quantified)

`backend_slow_skip.txt`: **215 jax entries vs 15 torch** (the census's 334 vs 15
counts *instances* — one entry covers several parametrizations).

| file | jax entries |
|---|---|
| `test_orbit.py` | **112** |
| `test_sphericaldf.py` | 28 |
| `test_noninertial.py` | 17 |
| `test_potential.py` | 16 |
| `test_orbits.py` | 15 |
| everything else | 27 |

**`test_orbit` + `test_orbits` = 127 of 215 (59%)** — orbit integration is the
dominant deferral, by a wide margin, and the file states the cause: *"under
forced-jax every scalar evaluation dispatches an XLA graph"*, with the fix being
*"un-defer as Track D in-backend vectorization lands"*.

**Sequencing consequence.** The visible burndown metric (xfail ledger, 42 entries
and now verified to contain zero stale entries) is the *small* debt. The larger
one is jax eager-per-step orbit integration, and it is concentrated: a single
capability — routing orbit integration through the in-backend ODE path instead of
per-step eager dispatch — would un-defer ~127 tests, i.e. **3x the entire xfail
ledger**, and it is already the named plan rather than a new idea.

That also explains why the ledger has stopped yielding: I have spent several
wakes finding that its remaining entries are decisions or documented designs
rather than defects. The ledger is close to done; the deferral list is where the
remaining engineering is.

**Correction to the previous entry: the orbit deferrals are NOT one capability.**
Last wake I wrote that "a single capability — routing orbit integration through
the in-backend ODE path — would un-defer ~127 tests". That grouped by *file*.
Grouping by the file's own documented *causes* (230 entries total) gives:

| entries | cause |
|---|---|
| **66** | jax orbit: heavy eager-jax >45s — per-CI-group session-cap fit |
| 39 | jax single-orbit variational/dxdv/lyapunov (Track D) |
| 28 | jax spherical-DF sampling/quadrature (Track F) |
| 17 | jax NonInertialFrameForce (eager-XLA timeout) |
| 13 | torch orbit dissipative/variational dxdv |
| 12 | jax multi-orbit battery (Track D/E) |
| 10 | big composite-MW / rotated-tilted-wrapper Poisson |
| 10 | jax single-orbit misc slow (no-C-force pots, Python integrators) |
| 25 | qdf 7, diskdf 6, dissipative/MovingObject 6, interpRZ 5, ... |

The 127 "orbit" tests split across **at least four different unblocking tracks**,
and the largest single block — 66 entries, 29% of the whole list — is not a
missing capability at all: those tests *pass*, they are just >45 s and get
skipped to fit the per-group session cap. That is a speed/budget problem (#67
nightly-full vs per-push subset), not an ODE-routing problem.

So the correct sequencing statement is narrower than what I wrote: no single
change clears the deferral list, the biggest slice is CI-budget shaped, and
Track D variational vectorization (39 + 12 = 51) is the largest genuine
*capability* gap.

I grouped by filename because that is what the ledger tooling reports, and the
file had recorded its own reasons all along — one `grep '^# ---'` gave a truer
decomposition than the counts I derived from paths.

**Constraint check: the biggest deferral block is not budget-addressable by me.**
The 66-entry "heavy eager-jax >45 s" block is a session-cap fit, so the obvious
cheap fix is to change the CI split rather than the code. But the per-test
`--timeout 300`, the `timeout-minutes: 180` shard cap and the pytest-split
grouping are all in `.github/workflows/backend-tests.yml`, and I have no
`workflow` scope — I can read those files but not change them.

So for those 66 the only route open to me is making the tests genuinely faster.
That reorders what is actually actionable during the vacation:

| entries | track | actionable by me? |
|---|---|---|
| 51 | Track D variational/dxdv vectorization (39 single + 12 multi) | **yes** — real capability work |
| 28 | Track F spherical-DF sampling/quadrature | **yes** — plan already recorded (#125) |
| 17 | NonInertialFrameForce eager-XLA | **yes** — speed work |
| 66 | heavy eager >45 s | only via speed; the budget route needs Jo |

**Stopping the analysis here.** Four wakes of measurement have produced a
complete and twice-corrected picture, and the marginal value of more triage is
now low: the ledger is verified free of stale entries, its residue is decisions
rather than defects, and the deferral list is decomposed by its own documented
causes. The next wakes should land the queued PRs (#1249, tolerance, GL hoist)
and then start Track F/D work, rather than continue mapping.

### 2026-08-04 — measuring how far the deferred sphericaldf tests are above the cutoff

The 28 jax `test_sphericaldf` deferrals were selected by "measured CI duration
>= 90 s". Whether Track F vectorization actually un-defers them depends on how
far above 90 s they sit — a test at 95 s needs a small win, one at 600 s needs a
different approach. That is measurable, so I am measuring it rather than assuming.

conftest applies the slow-skip in *all* modes (including regen), and there is no
opt-out flag, so the only way to time them is to make the list temporarily
invisible: `mv tests/backend_slow_skip.txt{,.bak}` (the loader returns an empty
set when the path does not exist), run, then move it back. Running 4 of the 28
under forced jax with `--durations`.

**Two footguns hit while setting this up, both already in the notes:**

1. `cd X && ... & echo; git status` — the `&` backgrounds the *whole* chain
   including the `cd`, so the trailing `git status` reported the **main repo's**
   working tree, not the worktree's. I briefly read main's long-standing
   `M galpy/util/coords.py` as contamination of my branch. Re-ran with an
   explicit `cd` in its own command.
2. The bypass leaves `D tests/backend_slow_skip.txt` in the *tolerance* worktree
   — a deleted tracked file on a branch that is queued for merge. The background
   job restores it after the run (the `mv` follows the `timeout`, so it runs even
   on timeout), but this is exactly the "experiment file-swap leaks into the next
   commit" hazard. **Verify the file is back before any further commit there.**
   In hindsight a throwaway worktree would have been the right place for this.

**Timing run 1 gave no timings.** `--durations=10` prints in pytest's terminal
summary, which this repo's conftest force-exit truncates — the same trap that
already cost me a run earlier this session (read the junitxml, not stdout). All
I learned was "4 passed". Re-dispatched with `--junitxml`, whose `time` attribute
survives.

The bypass/restore did work cleanly: file restored, worktree `git status` empty.

### 2026-08-04 — some sphericaldf deferrals may already be un-deferrable (Track F landed)

Timed 4 of the 28 jax `test_sphericaldf` deferrals with the slow-skip bypassed
(junitxml this time, since `--durations` is eaten by the conftest force-exit):

| test | duration |
|---|---|
| `eddington_powerspherical_divergent_auto_rmin` | **69.9 s** |
| `eddington_differentpotentials_dMdE_integral` | 46.4 s |
| `constantbeta_dehnencore_in_nfw_dens_massprofile` | 38.7 s |
| `constantbeta_interpolatedpotentials_beta` | 38.2 s |

All four are **under the 90 s cutoff they were selected at**, and none needs new
work to get there — the Track F spherical-DF migration (vectorized sampling + GL
quadrature) has landed since those measurements, which is exactly the event the
slow-skip comment says should retire these lines ("delete these lines as it
lands").

**Do not act on this from local timings alone.** Two reasons: this box is a
72-core dev machine and CI runners are slower, and the original selection was
made on real CI runs (27500392935 / 27503988401). A test at 70 s here could still
exceed 90 s there. Local timing is one hardware point, and the whole reason these
entries exist is a *CI* budget.

The right instrument already exists: the **regen run** (`workflow_dispatch` with
`regen=true`) rewrites both lists from real CI outcomes, which is precisely how
the harness is designed to surface burndown candidates in both directions.
Dispatching that is an API call, not a workflow edit, so it is available to me.

**Deliberately not dispatching yet**: #1249 still has 28 queued + 35 running jobs
and the project is runner-constrained; a full-matrix regen now would compete with
the PR I am trying to land. Queue it for after #1249 merges.

**Correction: regen CANNOT retire slow-skip entries.** Last wake I wrote that the
regen run "rewrites both lists from real CI outcomes" and queued it as the way to
un-defer the sphericaldf tests. Checking before relying on it:

* the dispatch input is described as *"Regenerate the backend xfail-ledger
  (writes `backend_xfail_new.txt` as an artifact instead of xfailing)"* — the
  **xfail ledger only**
* conftest: *"Slow-skip applies in all modes (incl. regen) so the unrunnable
  tests never consume their timeout"*

So a regen run would skip exactly the tests I want re-timed. It would have cost a
full-matrix run and told me nothing about the deferrals.

**The mechanism that does work, and is open to me:** `backend_slow_skip.txt` is a
*data file*, not a workflow. Deleting specific lines in a PR makes CI run those
tests on real CI hardware under the real budget — CI itself becomes the judge.
Green means the deferral was stale and is now retired; red (timeout) means the
90 s cutoff still binds there and I revert those lines. That is the honest test
of a *CI* budget question, which local timings on a 72-core box can only hint at.

Plan for after #1249 merges: a small PR removing the 4 measured entries (69.9 /
46.4 / 38.7 / 38.2 s locally), letting CI adjudicate. If they hold, extend to the
rest of the 28.

### 2026-08-04 — un-defer branch prepared: `feat/undefer-sphericaldf-jax` @ 535d6659

Removes all **28** jax spherical-DF slow-skip entries and their now-orphaned
section header (−38 lines). Not opened yet — held until the runners free up so it
does not compete with #1249.

**Why all 28 rather than the 4 I timed.** The section header records that skipping
them took the shard wall from 3111 s to ~1450 s, i.e. the set costs ~1660 s
against a **180-minute** cap. Restoring all of them therefore cannot blow the
cap, and one CI run adjudicates every entry instead of spending two runs on a
4-then-24 split — which matters on a runner-constrained project. If individual
tests still exceed the 300 s per-test timeout they fail individually and those
specific lines go back.

**Why CI decides and I do not.** The four I measured run at 38–70 s, under the
90 s cutoff — but on a 72-core dev box, while the original selection was made on
CI runners. A local number cannot answer a CI-budget question; the PR is the
experiment, and its commit message says so, so a red here reads as "the cutoff
still binds for these" rather than as a regression.

The warrant is the file's own instruction: *"the Track F spherical-DF backend
migration (vectorized sampling + GL quadrature) makes them fast; delete these
lines as it lands."* It landed.

Queue: **#1249** (0 queued / 37 running) → tolerance (−8 ledger) → GL hoist →
un-defer (−28 deferrals). That would take the ledger 42 → 34 and the slow-skip
list 230 → 202.

### 2026-08-04 — #1249 pre-merge audit (done ahead of green so the merge is immediate)

**1. `if` → `xp.where` array-input class: clean.** The library diff adds *no
conditionals at all* — one import and two assignments
`self._backend_compatible = is_backend_compatible(RZPot/Pot)`. Nothing branches
on an array value, so there is nothing that can misbehave under a trace.

**2. Duplication: none in code** — it reuses the existing shared helper and adds
no new def/class. **But the 6-line rationale comment is duplicated verbatim in
both constructors**: 12 lines of comment for 2 lines of code, which runs against
the recorded "keep backend comments terse — no multi-line essays" preference.

*Not fixing it now, deliberately.* The PR is minutes from green after ~150 checks
and several hours of runner time; pushing a comment trim would restart all of it
on a runner-constrained project. That is plainly the wrong trade for prose.
Recording it as a small debt to fold into the next PR that touches
`planarPotential.py`. Flagging rather than quietly ignoring my own directive.

**3. Placement: correct.** The assignment sits next to `hasC` in each
constructor — the same capability-flag slot, derived the same way — and the
helper it calls already lives in `galpy/backend/`.

Coverage trajectory confirms the codecov read: 40.86 → 74.81 → 76.94 → **86.01 %**
as shards upload, matching #1248's climb to 99.95 %.

**#1249 tail: slow is normal, checked rather than assumed.** The two
`build (ubuntu-latest, 3.14, tests/test_backend*.py, ...)` shards have been
running ~80 and ~53 min, which looks alarming for a small suite. The same shard
on #1248 took **92 min and passed** — so these are inside the normal envelope and
there is nothing to re-run. (It is the coverage-instrumented shard; instrumentation
is the cost.)

Second time this pattern has paid off on this PR: comparing against the *same
shard in a completed run* is the cheap way to distinguish "slow" from "hung", and
it costs one API call versus a wasted re-run on a runner-constrained project.

State: 146 green, 4 running (all-clear gate + 2 coverage shards + the jax
qdf/noninertial shard), coverage 86.01 % and still climbing. Audit already done,
merge is one command once the gate reports.

## 2026-08-04 — #1249 MERGED (9695ae4a); #1250 opened (−8 ledger entries)

**The last red was a gate race, not a failure.** After codecov/project finally
went green, `all-clear` showed FAILURE. Its log listed **every one of the 151
dependencies as "success"**, then `1 failing check run(s) and 0 failing commit
statuses` — i.e. it had evaluated at 11:10:16 while codecov was still red and
never re-polled. Re-running just the gate took the PR to **152 green / 0 reds /
CLEAN**, and it merged as `9695ae4a`.

Worth keeping: a failed `all-clear` looks identical whether it is a stale poll or
a genuine blocker; only its log distinguishes them. Re-running blind would have
been right here and wrong if something real were underneath — reading the log
first costs one command.

**Opened #1250** (`feat/quantity-relative-tolerance`, 3 commits): the
unsatisfiable absolute bound, the `finally:` restore, and **−8 ledger entries**
(42 → 34). Verified no conflict against the new develop head by diffing
merge-base→develop (which touched 9 files, none of mine) rather than
`HEAD..develop`.

Queue behind it: GL hoist `f481063f`, un-defer `535d6659` (slow-skip 230 → 202).

### 2026-08-04 — jit coverage: test_qdf is 0% traced-green despite 53/53 passing

```
BOUNDARY CALLS: traced=30156 not_traced=556  (98.2% traced)
C-LIBRARY ENTRIES: 1290
NOT-TRACEABLE (entered C): 53 tests
slots: 56 -> 53 passed, 2 skipped, 1 xfailed, 0 failed
```

**All 53 passing tests enter C.** The not-traceable list is
`meanvR/meanvT_{staeckel,adiabatic}_{gl,mc,...}` — the C action-angle routines.
By the harness's own rule ("a test with c_entries > 0 is not-traceable by
construction ... must NOT be counted as traced-green"), `test_qdf` is **0%
traced-green**, even though it is 53/53 green under `--jit` and reports 98.2 %
of boundary calls traced.

This is the sharpest illustration yet of "jit coverage != pass counts": the
traced 98.2 % is the *potential* evaluations; the *action* computation escapes to
libgalpy underneath them. A pass count, or the traced-fraction alone, would have
called this suite done.

Contrast with `test_sphericaldf`, measured earlier: **0** C-library entries, 0
not-traceable — genuinely traceable end to end. So the DF track is **not uniform**,
and "Track F is landing" does not imply "the DF family is jittable".

**Hypothesis to check before treating this as a capability gap:** Staeckel and
Adiabatic *do* have backend paths (Track E migrated them), so this may be a
*routing default* rather than a missing feature — `c=True` is the default when
the C extension is present, so the qdf tests may simply be taking the C branch by
default rather than lacking an alternative. If so, traced-green for qdf is a
routing question (does the forced-backend path bypass `c=True`?), not a port.
That distinction decides whether this is hours or weeks of work, so it should be
measured, not assumed — the same mistake I made twice already this session by
inferring mechanism from proximity.

**Resolved same wake: qdf's 0 % traced-green is a TEST FIXTURE choice, not a
capability gap.** I flagged the routing question as a hypothesis rather than
asserting a port was missing; checking it took one grep:

```python
# tests/test_qdf.py, module level
aAA = actionAngleAdiabatic(pot=MWPotential, c=True)
aAS = actionAngleStaeckel(pot=MWPotential, c=True, delta=0.5)
```

Every qdf in the file is constructed with `aA=aAA` / `aA=aAS`; only two tests
(1379, 1678) build `c=False` variants. So the 53 tests drop into libgalpy because
**the fixtures explicitly request the C path**, and Track E already migrated
Staeckel and Adiabatic to backend paths.

That defuses the alarming framing: "0 % traced-green" reads like qdf cannot be
jitted, when in fact this suite was never asking to be. The correct statement is
that `test_qdf` **does not measure** qdf's traceability, because it pins `c=True`
by construction.

Consequence for the roadmap: qdf traced-green is not blocked on a port. It needs
a decision about what the suite should exercise — the C path is what users get by
default and is the fast path, so switching fixtures to `c=False` would test
something different rather than something better. A separate `c=False` traced
probe (a handful of tests, not a rewrite of the file) is the cheap way to measure
qdf's actual traceability without weakening the existing coverage.

Third time this session that a one-command check separated "missing capability"
from "configured that way" — and the first time I checked *before* writing the
conclusion up rather than after.

**qdf CAN trace — measured, not inferred.** Ran the two `c=False` tests the file
already contains, under `--backend jax --jit`:

```
BOUNDARY CALLS: traced=893 not_traced=34  (96.3 % traced)
C-LIBRARY ENTRIES: 2          (vs 1290 for the c=True suite)
NOT-TRACEABLE: 1 of 2 tests   (test_call_diffinoutputs)
```

`test_meanjz_noaac_issue300` is **fully clean — zero C entries**. The one that
still enters C is `test_call_diffinoutputs`, which by its name compares outputs
across configurations and so also touches the module-level `c=True` object; 2
entries is consistent with that rather than with a backend gap.

So the chain is now measured end to end: qdf's "0 % traced-green" was the
fixtures selecting C, and when routed to the backend action-angle it traces.
**No port is needed for qdf traceability** — Track E's Staeckel/Adiabatic backend
paths already carry it.

Honest residual: 96.3 %, not 100 %. There are still 34 untraced boundary calls on
the backend path in this small sample, so "traceable" is not yet "fully traced".
That is a much smaller question than a migration, and worth a look before anyone
claims qdf traced-green.

This is the shape I want for roadmap claims: a suite reporting 53/53 green and
98.2 % traced was, by the harness's own definition, 0 % traced-green — and the
fix turned out to be a fixture argument, established by running two tests that
were already sitting in the file.

### 2026-08-04 — #1250 red is the VizieR network flake, not the PR

```
FAILED tests/test_orbit.py::test_from_name_values  - pyvo.dal.exceptions.DALServiceError:
                                                     Unable to access the capabilities endpoint
FAILED tests/test_orbit.py::test_from_name_errors  - DALServiceError
FAILED tests/test_orbits.py::test_from_name_values - DALServiceError
```

All three are `Orbit.from_name`, which queries SIMBAD/VizieR over the network.
This PR touches only `test_quantity.py` and `backend_xfail.txt`, so it cannot
reach these; the failure is an external-service outage. Documented flake class —
**re-run, do not cancel and do not "fix"**.

**Cannot re-run yet:** `gh run rerun --job` returns *"job cannot be rerun"* while
the parent run is still in progress (107 checks queued). So the sequence is: let
the run finish, then `gh run rerun <run_id> --failed`. Noting it because the
obvious move (re-run the moment you diagnose the flake) is rejected, and it would
be easy to read that rejection as a permissions problem rather than a timing one.

Also of note: the log had to come from `gh api .../jobs/<id>/logs`; both
`gh run view --log` and `--log-failed` returned nothing for this job.

## 2026-08-04 — #1250 MERGED (1adbe26e): ledger 42 → 34. #1251 opened.

`gh pr merge --squash` landed the tolerance work; **ledger verified at 34 on
develop** (was 42). Five PRs merged this session: #1246, #1247, #1248
(`730657c3`), #1249 (`9695ae4a`), #1250 (`1adbe26e`).

**Two reds on the way, both correctly diagnosed rather than papered over:**

1. **VizieR outage** — three `Orbit.from_name` tests failed with
   `pyvo.dal.exceptions.DALServiceError: Unable to access the capabilities
   endpoint`. A PR touching only `test_quantity.py` and the ledger cannot reach
   those. Re-ran; they passed. *Constraint worth remembering:* `gh run rerun
   --job` refuses while the parent run is in progress, so the flake could not be
   cleared until the whole run completed — the rejection reads like a permissions
   error but is purely timing.
2. **`all-clear` stale poll** — same as #1249. Its log listed every dependency as
   `success` and still reported `1 failing check run(s)`; it had evaluated while
   codecov was red. Re-running only the gate took it CLEAN. Reading the log first
   is what distinguishes this from a real blocker, and it costs one command.

**#1251 opened** (`feat/hoist-gl-order`): de-duplicates `_BACKEND_GL_ORDER` into
`galpy/actionAngle/actionAngle.py` — deliberately *not* the generic backend
quadrature module, since the order is an action-angle property and hoisting it
there would invert the dependency. Also corrects the comment's `<1e-7` floor
claim, which is true for actions but ~20x optimistic for angles, with the
measured non-convergence (3.3 / 2.1 / 1.8 / **3.5** e-6 at n=50/100/200/400)
written in so the next reader does not repeat the sweep.

Remaining queue: `feat/undefer-sphericaldf-jax` @ 535d6659 (slow-skip 230 → 202).

### 2026-08-04 — #1251 second red is the same VizieR outage; undefer measured properly

`#1251` (`feat/hoist-gl-order`) picked up a second red beyond the codecov
partial-upload artifact:

    build (ubuntu-latest, 3.14, tests/test_orbit.py -k 'tes…)

Read the job log rather than classifying from the shard name (`gh run view
--log`/`--log-failed` return nothing for this job; `gh api
repos/jobovy/galpy/actions/jobs/<id>/logs` works):

    FAILED tests/test_orbit.py::test_from_name_values  - pyvo.dal.exceptions.DALServiceError: Unable to access the capabilities endpoint
    FAILED tests/test_orbit.py::test_from_name_errors  - pyvo.dal.exceptions.DALServiceError: Unable to access the capabilities endpoint

Same VizieR network outage that hit #1250. #1251 touches only
`actionAngle.py` / `actionAngleSpherical.py` / `actionAngleVertical.py`, none of
which `Orbit.from_name` can reach. Re-run is blocked until the parent run
finishes — `gh run rerun --job` refuses while the run is in progress, which
reads like a permissions error and is purely timing.

**Un-defer PR measured instead of guessed.** `feat/undefer-sphericaldf-jax`
was queued to "let CI adjudicate" whether Track F retired the 90 s cutoff for
the 28 jax spherical-DF slow-skips; only 4 of the 28 had ever been timed. Two
setup errors caught before they produced a fake answer:

* `--no-slow-skip` does not exist. It was also unnecessary — the worktree's
  HEAD *is* the commit that deletes those 28 lines, so they simply run.
* the worktree had no C extension. `ls galpy/libgalpy*.so` said "missing", but
  the build puts it at the **repo root**; the real check is
  `from galpy.orbit.integrateFullOrbit import _ext_loaded`, which reports
  `True`, plus `galpy.__file__` inside the worktree.

Running the *whole file* rather than the 28 hand-picked nodeids: a hand-picked
batch is not a faithful context, and the full file measures the shard wall,
which is the actual CI-budget question. Serial (this file has cross-test
state), `--timeout 300` to match CI's per-test cap, junitxml for per-test times.

### 2026-08-04 — jax `test_diskdf` sampling: a real NaN defect, and three wrong guesses on the way

The 3 jax `test_diskdf` ledger entries are **not** what the ledger says
("physical output is a bare ndarray with no `.to()`"). Today's error is:

    assert np.float64(nan) < 0.05
      where nan = mean(rs)

One sampled orbit has `R() == nan`. Getting to the cause took three corrections,
each worth keeping:

1. **"RNG-fragile statistical test."** Wrong. The tolerances *are* tight
   (`std(rs)` to ±0.03 where 100 draws carry ~0.025 of sampling error), but the
   failure is a NaN, not a shifted statistic.
2. **"Last-ulp round-off at the circular-orbit limit."** Wrong, and measurably
   so: the negative radicand is **-0.209**, and **numpy computes the identical
   value** — the `RuntimeWarning: invalid value encountered in sqrt` fires on
   the numpy path too. Nothing backend-specific about it.
3. **"numpy never generates the bad draw."** Wrong — an artifact of *my probe*,
   which recorded after calling the original function, so a raise skipped the
   record. Moving the record before the call showed numpy generates it too.

**Actual mechanism.** `diskdf._ELtowRRapRperi` computes
`vR = sqrt(2(E - Psi(xE)) - L^2/xE^2)`. `Lz = sigma2(xE)*log(uniform)/Omega + Lc(E)`
has an unbounded negative tail, so a rare draw lands where `L^2/xE^2` exceeds
`2(E-Psi)` — here `L=-0.0527`, giving `1.21 > 1.00`. Then:

* **numpy**: `sqrt` → NaN → scipy's root-finder inside `EccZmaxRperiRap` raises
  `ValueError: The function value at x=... is NaN; solver cannot continue`, and
  the sampler's `except ValueError: continue` **drops the draw**.
* **jax**: the backend bisection has no NaN check, returns `nan`, and the draw
  reaches the output as a NaN orbit.

So the `# pragma: no cover` on that `except` — "Tests don't get here anymore,
because of improvements in the rperi/rap calculation" — is **stale and
load-bearing**: numpy takes that path today, and it is the only reason the numpy
suite is green.

**Fix** (`feat/diskdf-unphysical-draw`): hoist the radicand, and raise
`ValueError` when it is negative, before `EccZmaxRperiRap` ever sees a NaN. Every
backend then drops the same draws by the same rule. Raising is right and
*clamping would be wrong* — clamping to zero would retain a bogus sample as a
circular orbit rather than rejecting a draw outside the inversion's domain.
Expected to be numpy-byte-identical, since numpy already skips exactly these
draws and nothing between the two points consumes RNG — to be verified, not
assumed.

### 2026-08-04 — the CI/local scaling factor was INVERTED; un-defer decision revisited

While deciding which spherical-DF slow-skips could be retired, I scaled local
timings to CI using the wall times quoted in `backend_slow_skip.txt`'s own
header ("complete shard wall 3111 s -> ~1450 s after skip") against my local
1591 s, concluding **"CI is ~1.96x slower"**. That is backwards, and the header
is stale: the same shard measures **426 s** on CI today, because Track F landed
after that header was written.

Real measurement, same code both sides (`test_dehnendf_sample_powerrise_returnROrbit`,
unfixed, forced jax, quiet box):

    local 200.1 s   vs   CI 141.5 s   ->   CI/local = 0.71   (CI is FASTER)

The 72-core box has more cores but slower single-thread, and eager-jax tests are
essentially single-threaded.

Two things made this findable, both worth reusing:

* **A ledgered xfail shows its real duration in the CI junit.** It appears as
  `<skipped type="pytest.xfail">` but carries the elapsed time, so the actual CI
  cost of a "failing" test is free to read from `backend-junit-<backend>-<shard>`.
  A *slow-skipped* test shows `0.0 s` and needs a local run plus the ratio.
* **Compare like with like.** My first pass compared CI's 141.5 s (unfixed, stops
  at the first failing assert) against my 187.4 s (fixed, runs to completion) and
  would have read the difference as machine speed. Re-running the *unfixed* test
  locally is what made it apples-to-apples.

Consequences:

* **diskdf PR is safe to un-ledger all 3.** Their real CI costs are 41.7 s,
  46.3 s and 141.5 s — all far under the 300 s cap, and the fix does not add
  meaningful work (it drops one draw and lets `sample()` top up).
* **The un-defer PR's justification for keeping `test_king_sigmar_directint` is
  void.** 178.7 s local projects to ~127 s on CI, not ~350 s. Re-measuring
  `test_king_sigmar` without the 300 s cap to get its true cost before deciding.

### 2026-08-04 — un-defer finalized at 27; stale base caught by `git diff --stat`

**The branch would have reverted two merged PRs.** `feat/undefer-sphericaldf-jax`
was cut before #1249/#1250 landed, so its merge-base was `730657c3`. Its own log
looked fine — one commit touching one file — but `git diff --stat` against
develop showed **five**:

    galpy/potential/planarPotential.py |  16 +--
    tests/backend_slow_skip.txt        |  45 ++----
    tests/backend_xfail.txt            |   8 ++
    tests/test_backend_array_inputs.py |  41 ------
    tests/test_quantity.py             | 278 +++++++++---------------

The `+8` on the xfail ledger is #1250's burndown coming back. Rebased onto
`1adbe26e`; the diff is now `backend_slow_skip.txt` alone, 27 removed / 0 added,
and conftest confirms exactly one jax `test_sphericaldf` deferral survives
(`test_king_sigmar`), 188 jax + 15 torch = 203 total. **A stale base is invisible
in the branch's own history — always diff against the target.**

**27, not 26.** `test_king_sigmar` measured **>690 s** when called directly (the
slow-skip list has no bypass, so a pytest run in that worktree simply skips it and
reports 0.0 s — which is how the first attempt at this measurement fooled me). At
CI/local = 0.71 that is still ~490 s, well over the 300 s cap, so it stays.
`test_king_sigmar_directint` at 178.7 s projects to ~127 s and is un-deferred; the
earlier decision to keep it rested entirely on the inverted ratio.

**Shard-duration reference (so a long job isn't mistaken for a hung one):** the
`test_backend*.py` build shard legitimately takes **~91 min** — measured on
develop run 30923735606 (15:34:51 → 17:06:13, success). #1251's copy at 77 min is
in range, not stuck.

**A limit of the junit artifacts:** a ledgered xfail records only
`message="backend-xfail-ledger"` with no traceback, so artifacts give *durations*
for free but never failure detail. Triage still needs a local run.

### 2026-08-04 — CORRECTION: a stale base does not revert disjoint PRs (and what it *does* break)

The entry above titled "**The branch would have reverted two merged PRs**" is
**wrong** and should not be relied on. A squash-merge applies the branch's own
`merge-base..HEAD` diff, not the tree difference against develop. With disjoint
files a stale base reverts nothing.

What misled me: `git diff <branch> origin/feat/backends-develop` shows changes
*develop has that the branch lacks* as if the branch removed them. That is a view
of divergence, not of what gets merged. The right question is
`git diff $(git merge-base HEAD origin/develop) HEAD` — the branch's own diff.
`no-merge-commits-rebase-only` and `dont-reflexively-rebase-disjoint-prs` already
say this; I reasoned from the wrong diff instead.

**What a stale base genuinely breaks: codecov.** #1251 turned out to have the
same stale base (`730657c3`, two commits behind), and that *was* a real blocker:

    Comparing base (730657c) to head (f481063).
    Report is 2 commits behind head on feat/backends-develop.
    Coverage 99.95% -> 99.82%  (-0.14%)   Lines +310   Misses +64

A 17-insertion/12-deletion diff across three actionAngle files cannot add 310
lines or 64 misses. codecov was diffing against a two-commit-old develop. Patch
coverage itself was clean ("All modified and coverable lines are covered by
tests"), so the failure was entirely the base. Since `codecov/project` gates
`all-clear`, no amount of re-running would have cleared it — only a rebase.

Rebased to `98c2d5c7`, force-pushed, and cancelled the obsolete prior-SHA run
(prior-SHA cancels are safe; same-SHA push-event cancels block merge).

Also resolved this round: the `test_backend*.py` shard I had been watching for
~93 min **passed**. The three reds were the VizieR flake, `codecov/project`, and
`all-clear` — and the gate's own log lists exactly those two upstream failures,
so it was reporting correctly rather than stale-polling as it did on #1249.

**Rule going forward:** judge merge-safety with the merge-base diff; judge
codecov with the base commit named in its comment. They are different questions.

### 2026-08-04 — `test_diskdf` under `--backend jax --jit`: first traced-fraction measurement

Previously this file had only been recorded as a *pass count* ("129 tests, 0
failures"), which the jitcount plugin's own docstring warns proves nothing. Run
with the plugin on `feat/diskdf-unphysical-draw` (so the 3 un-ledgered sampling
tests actually execute):

    BOUNDARY CALLS:   traced=378953  not_traced=80   (100.0% traced)
    C-LIBRARY ENTRIES: 767
    NOT-TRACEABLE (entered C): 5 tests
    EXIT=0  WALL=786s

**Read the two counters together, not the headline alone.** 100.0% of boundary
calls trace, but 767 C entries mean five tests are *not-traceable by
construction* and must not be counted as traced-green:

    test_dehnendf_sample_flat_returnOrbit
    test_dehnendf_sample_flat_returnROrbit_rrange
    test_dehnendf_sample_powerrise_returnROrbit
    test_dehnendf_sample_flat_returnROrbit_wcorrections
    test_shudf_sample_flat_returnROrbit_wcorrections

The first three are exactly the ones the unphysical-draw fix un-ledgers. That is
expected rather than a defect: `dehnendf.sample` integrates each accepted draw
with `thisOrbit.integrate(..., self._psp)`, i.e. the C integrator, and galpy's
answer for C is the C-STM `custom_vjp` path (differentiability via callback, not
tracing). Making these traced-green would mean routing sampling through the
in-backend ODE, which is a Track-D/streamspray-shaped change, not a df fix.

Useful secondary result: **EXIT=0** — all three formerly-ledgered tests pass under
`--jit` as well as eager, so the guard holds under trace.

### 2026-08-04 — jax `test_SpiralArmsPotential::test_Rzderiv`: an expired tolerance band + FD amplification, not a defect

Characterised the single jax `test_SpiralArmsPotential` ledger entry. It is **not
a backend defect**, and the evidence is unusually clean.

The test checks an analytic second derivative against a *finite-difference*
reference:

    assert_allclose(pot.Rzderiv(R, z, phi, t),
                    -deriv(lambda x: pot.Rforce(R, x, phi, t), z, dx=dx),
                    rtol=rtol)      # dx = 1e-8

Measured at the exact failing parameters (`amp=13, N=7, alpha=0.1, r_ref=1.123,
phi_ref=0.3, Rs=0.777, H=0.5, Cs=[4.5], omega=-3.4`; `R,z,phi,t =
0.777, 0.333, pi/3, 0`):

    analytic Rzderiv   jax vs numpy   1.914e-13
    Rforce             jax vs numpy   1.912e-13
    FD reference       jax vs numpy   3.033e-06     <-- the side that moves
    analytic-vs-FD     numpy 5.122e-07   jax 2.521e-06

The quantity under test agrees to **1.9e-13**. The reference is the noisy side:
`dx=1e-8` amplifies a last-bit `Rforce` difference by ~1/dx = 1e8, which is
exactly the 1e-13 -> 1e-6 jump observed. Any backend with a different (equally
valid) summation order in the multi-term spiral sum lands here.

**Second, independent finding — the tolerance band has expired.** The test does

    _NUMPY_1_23 = (version > parse("1.22")) * (version < parse("1.27"))
    rtol = _NUMPY_1_23 * 3e-6 + (1 - _NUMPY_1_23) * 1e-6

with the comment "For testing 1.23/1.24/1.25/1.26 precision issues". On **numpy
2.4.3** the second factor is False, so `_NUMPY_1_23 = 0` and the **tight** 1e-6
branch applies. jax's 2.52e-6 would **pass** under the 3e-6 band galpy itself
applies for numpy 1.23-1.26.

Worth flagging to Jo beyond the backend question: with the band expired, the
**numpy** path passes at 5.12e-7 against 1e-6 — only ~2x margin on an FD
reference with dx=1e-8. That is a latent upstream fragility for numpy 2.x, not
something the backend work introduced.

Disposition: goes to the "needs Jo" tolerance bucket with the actionAngle items.
Unlike those, the honest fix here is arguably upstream — either extend the
version band past 1.27 or raise `dx` so the reference is not amplification-bound.

### 2026-08-04 — full ledger staleness sweep: 1 stale, and a matcher bug that nearly reported 5

Swept all **34** ledger entries against the **44** `backend-junit-*` artifacts of
the latest completed `feat/backends-develop` run (30923735373):

    confirmed still failing / xfail   28
    STALE (passed on CI)               1
    not observable in these artifacts  5

The one real stale entry is
`jax tests/test_sphericaldf.py::test_anisotropic_hernquist_negdf`: its `<testcase>`
has no failure/error/skipped child at all (0.35 s), i.e. it passed in full-file CI
context, and it also passes locally under forced jax in 5.1 s. Removed —
ledger 34 -> 30 on the diskdf branch (3 diskdf + this one).

**The sweep first said five, and four of those were an artifact of my matcher.**
The other four were `-jit` entries:

    jax-jit   test_orbit.py::test_eccentricity
    jax-jit   test_potential.py::test_2ndDeriv_potential[AnyAxisymmetricRazorThinDiskPotential]
    jax-jit   test_potential.py::test_poisson_surfdens_potential[AnyAxisym...]
    torch-jit test_potential.py::test_poisson_surfdens_potential[AnyAxisym...]

My matcher did `backend.replace("-jit","")` to line the ledger up with artifact
names, which silently compared **traced-mode ledger entries against eager
results**. That run publishes **zero** jit artifacts and runs **zero** jit jobs, so
there was no traced evidence at all. An eager pass says nothing about a traced
failure — and the AnyAxisym entries passing eagerly is precisely what a jit-only
failure looks like (they are ledgered under `-jit` for that reason).

Reusable rule: when cross-checking the ledger against artifacts, **assert the
artifact set actually covers the mode you are matching**, and treat a suffix you
had to strip as a red flag rather than a normalisation.

Also characterised while here: `jax test_qdf::test_call_diffinoutputs` fails with
"Test orbit in qdf that is supposed to be unbound is not" — the backend AA does
not raise `UnboundError` where numpy does, matching the known by-design qdf pair.

### 2026-08-04 — the 4 "stale" jit entries, observed directly: still failing

Follow-up to the matcher bug. The CI artifacts carry no jit data, so I ran two of
the four `-jit` entries locally under `--backend jax --jit --runxfail`:

    FAIL  2.4s  test_2ndDeriv_potential[AnyAxisymmetricRazorThinDiskPotential]
          "second Radial derivative ... fails at (R,Z) = (0.500,0.000)"
    FAIL  0.6s  test_poisson_surfdens_potential[AnyAxisymmetricRazorThinDiskPotential]
          "Methods in AnyAxisymmetricRazorThinDiskPotential do not accept array inputs"

Both are alive, and both match known, documented causes — the AnyAxisym
cancellation defect, and the deliberate array opt-out that a named negative test
guards. So the sweep's "5 stale" was wrong by exactly the 4 I rejected, and
deleting them would have converted real traced failures into unexplained jit
shard reds.

Separate health check from the same sweep data: **0** ledgered entries are
recorded as a hard `FAILED` on CI (as opposed to `xfail`). A timeout overrides
xfail and records FAILED, so this confirms no ledgered test is currently timing
out and reddening a shard — the ledger and slow-skip lists are correctly
partitioned right now.

### 2026-08-04 — torch `test_constantbeta_interpolatedpotentials_beta`: partially characterised, NOT root-caused

Failure (forced torch, `--runxfail`, 11.9 s):

    galpy/potential/interpSphericalPotential.py:105, in _revaluate
        out[r >= self._rmax] = -self._total_mass / r[r >= self._rmax] + self._Phimax
    TypeError: Concatenation operation is not implemented for NumPy arrays
    ... with r = np.float64(inf)

Line 105 is inside `if xp is numpy:`, so in that run `get_namespace(r)` returned
**numpy** while the potential's own state is torch — the classic frozen-attr /
input-namespace mismatch.

**What I established:**

* `_Phimax` and `_Phi0` *are* frozen as `torch.Tensor` at construction under
  forced torch (`_total_mass` and `_rmax` stay `np.float64`). So a numpy branch
  mixing them with numpy arrays would indeed produce this TypeError — numpy tries
  the sequence-concat protocol on a Tensor operand.
* Under plain numpy the numpy branch handles `np.float64(inf)` fine; only a bare
  Python `float` breaks it ("'float' object is not subscriptable"), which is a
  separate latent scalar-safety gap in that branch.

**What I could NOT reproduce, and am not going to assert:** constructing the
potential inside `backend.use("torch", force=True)` and calling
`_revaluate(np.float64(inf))` **works** — `get_namespace` resolves the numpy
scalar to `array_api_compat.torch`, so the *backend* branch runs and returns
`tensor(-0.0091)`. My isolated repro therefore does not reproduce the failure,
and I do not yet know what makes `get_namespace` return numpy in the real test.

So: characterised (the mechanism class is clear) but **not root-caused** (the
trigger is context-dependent). Next step is to instrument the actual test rather
than a hand-built instance — another case where isolation is not a faithful
context. Deliberately NOT starting a fix on a hypothesis I have not demonstrated.

### 2026-08-04 — torch `interpSphericalPotential`: ROOT-CAUSED and fix verified

Following the honest "characterised but not root-caused" note above: instrumenting
the **real test** (rather than a hand-built instance) settled it. A probe recording
`(type(r), get_namespace(r), type(self._Phimax))` at every `_revaluate` call:

    5x  r=Tensor    get_namespace=array_api_compat.torch   _Phimax=Tensor   -> fine
    1x  r=float64   get_namespace=numpy                    _Phimax=Tensor   -> the failure
        example: ('float64', 'np.float64(inf)', 'numpy', 'Tensor')

So exactly one call arrives **numpy-resolved** — `r = inf`, a scipy quadrature
limit evaluated inside a numpy-forced region — while the potential's frozen
`_Phimax`/`_Phi0` are still `torch.Tensor` from construction. The numpy branch then
mixes numpy arrays with Tensors, and numpy falls back to the sequence-concat
protocol: "Concatenation operation is not implemented for NumPy arrays".

**The backend branch already defends against this and the numpy branch does not** —
it writes `float(self._Phi0)`, `float(self._total_mass)`, `float(self._Phimax)`.
That asymmetry is the whole bug.

Verified by monkeypatching a `float()`-ed numpy branch over the real test, with no
file edits: **PASS (28.3 s)** where it previously raised.

Fix branch `feat/interpspherical-frozen-scalars` applies it to all three affected
methods, not just the one the test happens to hit — `_revaluate`, `_rforce` and
`_r2deriv` all use the frozen scalars raw in their numpy branch. `_rdens` uses a
literal `0.0` outside rmax and needs nothing. These are scalars by construction,
so `float()` is exact and cannot perturb the numpy path.

### 2026-08-04 — interpSphericalPotential fix verified in-tree; audit says the pattern is NOT systemic

`feat/interpspherical-frozen-scalars` @ `bde9df0e`: the fix plus its ledger
removal. Verified in-tree under forced torch **with the C extension built**
(29.1 s PASS) — without it a ledgered test can silently take the Python path.

**Is the asymmetry systemic?** ~20 galpy modules use the `if xp is numpy:` dual
path, so I scripted the exact signature rather than eyeballing them: an AST pass
flagging any method where the numpy branch uses `self._x` **raw** while the
backend branch of the same method wraps it in `float(...)`.

* against the **fixed** tree: none
* against the **unfixed** tree (the control): exactly the three methods I fixed,
  with exactly the right attributes —
  `_revaluate -> ['_Phi0','_Phimax','_total_mass']`, `_rforce -> ['_total_mass']`,
  `_r2deriv -> ['_total_mass']`

Running the control matters: a "none" from an audit that cannot see the bug it
was written for is worthless, and that is the same trap as a control that passes
with and without the fix. Since it demonstrably detects the bug, the empty result
on the fixed tree is meaningful: **interpSphericalPotential was the only instance**.

Limitation, stated so the negative result is not over-read: the audit only catches
this *specific* asymmetry. A method where **both** branches use the attribute raw
is invisible to it — that is only a bug when the attribute can be backend-typed
while the numpy branch runs, which needs a different probe to establish.

### 2026-08-04 — #1251 second red is a runner eviction, not a test failure

A `build (ubuntu-latest, 3.10, test_orbit.py test_orbits.py …)` job went red on
the post-rebase run. The log ends:

    tests/test_orbit.py::test_lyapunov_spectrum_integrable PASSED
    ##[error]The runner has received a shutdown signal...
    tests/test_orbit.py::test_lyapunov_spectrum_henonheiles_chaotic
    ##[error]The operation was canceled.

Every test up to that point passed; no assertion failed. To rule out "the shard
outgrew its budget", compare the same shard's duration:

    failed job (98c2d5c7)   19:19:26 -> 19:55:32   = 36 min, runner shutdown
    same shard, prior run   16:58:33 -> 17:57:56   = 59 min, SUCCESS

It died at 36 min against a 59-min honest runtime, so it was **evicted**, not
timed out. Re-runnable.

Note this is the **push-event** run for the new SHA, not the PR-event run.
Re-running its failed job is fine; *cancelling* a same-SHA push run is what
blocks merge. As before, `gh run rerun --job` refuses while the parent run is
still in progress, so the re-run waits for the run to finish.

### 2026-08-04 — #1245 moved to main (Jo's call), and the two traps in doing it

Jo asked why #1245 was still sitting there and to "just put out on main and we
later rebase feat/backends and feat/backends-develop". Right call: the fix is a
**pre-existing numpy bug**, nothing backend about it. `m = 4aR/((a+R)^2+z^2)` is
<= 1 analytically but rounds just above 1 for tiny |z|, and scipy's ellipe/ellipk
return nan there; the fix clamps with `numpy.minimum(..., 1.0)` at four integrand
sites. It was only on `feat/backends-develop` because that is where I found it,
and it was still a **draft** because it changes numpy-path values (nan -> finite)
and I had parked it for Jo.

**Trap 1 — a plain base change would have been wrong.** The branch sat on
develop, so retargeting the PR base to main would have dragged the entire develop
diff into it. Cherry-picked the single commit onto main and force-pushed instead.

**Trap 2 — the cherry-pick tried to smuggle in a develop-only test.** The
conflict hunk in `tests/test_potential.py` carried
`test_pickling_potential_drops_units_wrapper` (from the scipy-1.18 spline
pickling work), which depends on develop-only code and would have broken main.
Resolved by taking main's file and appending *only*
`test_anyaxisymrazorthin_smallz_no_nan`; verified the pickling test's occurrence
count in the file is 0.

Verified both directions on main, in separate worktrees (no stash, no
`checkout <ref> -- <file>`):

    with fix:            PASS
    control, unfixed:    FAILS -- "zforce non-finite at small |z|:
                         [1e-30 1e-25 1e-20 1e-16 1e-12]"

i.e. exactly the documented band. The test is stringent rather than
isfinite-only: it asserts Fz/z tends to a single constant below |z| ~ 1e-8, so a
wrong-but-finite value still fails.

#1245 is now base=main, undrafted, diff exactly the two intended files
(20+/5-, 29+/0-), full CI running. One `all-clear` shows CANCELLED — superseded
by a fresh run on the same SHA when the ready-for-review event fired, not a real
failure.

### 2026-08-04 — CORRECTION: #1245 does NOT clear the AnyAxisym jit ledger entries

I told Jo that once #1245 lands on main and the backend branches rebase, the
`test_2ndDeriv_potential[AnyAxisymmetricRazorThinDiskPotential]` jit entry
"should clear". **That is wrong**, and I checked it rather than leave it standing.

Applied #1245's clamp (`numpy.minimum(4aR/aRz, 1.0)`, 5 sites on develop) to a
throwaway develop worktree and ran both jax-jit AnyAxisym entries:

    FAIL  test_2ndDeriv_potential[AnyAxisym]         -- same 2nd-radial-derivative assertion
    FAIL  test_poisson_surfdens_potential[AnyAxisym] -- array-input opt-out

Neither clears. The 2ndDeriv jit failure is the **separate `1 - m` catastrophic
cancellation** defect, not the NaN band; two different bugs in the same
integrand family, and only the NaN one is fixed by #1245. (The array-input
opt-out I had already flagged as unaffected — that one is deliberate and guarded
by a named negative test.)

So the expected ledger effect of #1245 on the backend branches after rebase is
**zero entries**, not one. Its value is the pre-existing numpy-path bug: an ODE
solver drifting off the plane by roundoff lands in the 1e-30..1e-12 band and
poisons the whole orbit.

### 2026-08-04 — #1245 codecov "20% patch" checked directly: partial upload, not a real gap

codecov on #1245 reported *"Patch coverage is 20.00000% with 4 lines in your
changes missing coverage"* for
`AnyAxisymmetricRazorThinDiskPotential.py`. Specific enough to be worth checking
rather than waiting out, so I traced it locally with `trace.Trace` (pytest-cov
dies here with "cannot load module more than once per process" on a numpy
re-import; codecov's own line_coverage JSON is unreliable, hence the tracer):

    clamp lines : [126, 155, 184, 221, 251]
    covered     : [126, 155, 184, 221, 251]
    NOT covered : []

All five clamp sites are exercised by `test_anyaxisymrazorthin_smallz_no_nan`.
The report also carries "different number of reports uploaded between BASE and
HEAD" and "Report is 2 commits behind head", i.e. the usual partial-upload
state — 36 of ~183 checks had finished when it was posted.

Incidental but worth recording: that worktree has **no libgalpy**, so my earlier
"passes on main" run took the Python path. It does not matter here —
`AnyAxisymmetricRazorThinDiskPotential.hasC is False`, so the potential is
pure Python/scipy either way, and the fails-without / passes-with flip is
unaffected. (Checked rather than assumed; a missing C extension is normally
exactly what fakes a pass.)

### 2026-08-04 — AnyAxisym `1-m`: mechanism identified, fix designed (NOT implemented — queue is 3 deep)

Having confirmed #1245 does not clear the AnyAxisym jit entries, traced what does
cause them. Backend path in `AnyAxisymmetricRazorThinDiskPotential.py`:

    _ELLIP_M_CAP = 1.0 - 1e-15
    m = xp.minimum(4.0 * a * R / aRz, _ELLIP_M_CAP * xp.ones_like(aRz))

and the AGM kernel in `galpy/backend/special/_fallback/elliptic.py`:

    b = xp.sqrt(1.0 - ms)          # K(m) = pi / (2 AGM(1, sqrt(1-m)))

**Two compounding problems, both about the complement `1 - m`.** The true
complement is exactly

    1 - m = ((a - R)^2 + z^2) / ((a + R)^2 + z^2)

which at `z = 0, a -> R` goes to 0 like `(a-R)^2`. But (i) the cap pins it at
1e-15 no matter how small it really is, and (ii) even without the cap,
`1.0 - m` in floating point cancels catastrophically once `m` is within an ulp
or so of 1. `K` diverges logarithmically, `K ~ -0.5 ln((1-m)/16)`, so an error in
the complement is an error in `K`.

The failing test evaluates at **(R,Z) = (0.500, 0.000)** — z = 0, i.e. precisely
where the integrand sweeps `a` through `R` and the complement passes through 0.
Strong circumstantial support, but I have **not** yet proven this is the cause;
that is the first step when implementing.

**Design (backend-only, so numpy stays byte-identical):**

1. add a complement-taking entry point to the backend special layer —
   `ellipkm1(xp, p)` with `p = 1 - m` — reusing `_agm_KE` but taking
   `b = sqrt(p)` directly instead of `sqrt(1 - m)`;
2. in the AnyAxisym backend integrands, form `p = ((a-R)^2 + z^2) / aRz`
   algebraically (exact, no subtraction) and call the complement form, dropping
   `_ELLIP_M_CAP` entirely.

Deliberately **not** touching the numpy branch: it uses `scipy.special.ellipk(m)`
and improving it would change numpy values, which the mission forbids without
sign-off. scipy's own `ellipkm1` exists if Jo ever wants the numpy side improved
too — worth raising separately, since the same precision loss is present there.

Not implemented now: three PRs are already queued behind #1251 and the standing
instruction is one PR at a time.

### 2026-08-04 — AnyAxisym `1-m`: hypothesis PROVEN (was flagged as unproven above)

Quantified the complement error directly at the failing test's point
(R = 0.5, z = 0), sweeping `a` toward `R`. `scipy.special.ellipkm1(p)` = K(1-p)
serves as the accurate reference because it *takes the complement* rather than
forming it by subtraction:

     a-R     exact 1-m      float 1-m (capped)   K_ref      K_capped    rel.err
    1e-3     9.980e-07      9.980e-07             8.29505    8.29505    1.9e-11
    1e-5     9.9998e-11     9.9998e-11           12.89923   12.89923    9.0e-08
    1e-6     9.99998e-13    1.00031e-12          15.20181   15.20165    1.0e-05
    1e-7     9.999998e-15   9.76996e-15          17.50439   17.51603    6.6e-04
    1e-8     1.000000e-16   9.99201e-16          19.80698   18.65608    5.8e-02
    1e-9     9.999999e-19   9.99201e-16          22.10956   18.65608    1.6e-01

Two effects, both visible: below ~1e-16 the subtraction `1.0 - m` cannot
represent the complement at all, and `_ELLIP_M_CAP` pins it at ~1e-15 regardless
of the true value. K is wrong by up to **16 %** at quadrature nodes near a = R.
Since the GL rule samples a continuum of `a` around R, some nodes always land in
the corrupted region — which is precisely what breaks an integral checked against
a numerical derivative.

So the earlier "strong circumstantial support, but not proven" is now settled:
the mechanism is confirmed, and the designed fix (exact algebraic complement
`p = ((a-R)^2 + z^2)/aRz` + a complement-taking `ellipkm1` on the AGM kernel)
addresses exactly this. Still not implemented — queue is three deep.

Note the reference used here, `scipy.special.ellipkm1`, is the same function that
would fix the **numpy** path's identical precision loss. That remains out of
scope without Jo's sign-off, since it would change numpy values.

### 2026-08-04 — MY ERROR: #1245 re-drafted. I shipped a known-bad fix+test to main without reading the PR's own comments

Jo asked whether the #1245 comment ("The clamp is necessary but not sufficient,
and the regression test I wrote asserts the wrong physics") is now irrelevant.
**It is not. It is entirely correct, and I acted against it.**

I treated #1245's draft status as "parked pending Jo's tolerance call", cherry-
picked it onto main, undrafted it and started full CI — **without reading the
discussion on the PR**, where I had already documented (2026-08-03) that I had
caught a real flaw and was reworking it. Re-drafted now, before anything merged.

Re-verified the claim myself rather than trusting the old comment. At R=0.5,
Sigma=0.29676545, so -2*pi*Sigma = -1.86463229:

     |z|          Fz                  Fz/(-2 pi Sigma)   Fz/z
    1.0e-05  -1.864615319820e+00       0.999991          -186461.5
    3.0e-06  -1.864627198354e+00       0.999997          -621542.4
    1.0e-06   1.697235994046e-06      -0.000001                1.697   <- garbage
    1.0e-07  -1.864632119332e+00       1.000000        -18646321.2   <- correct
    1.0e-08   1.696897011969e-08      -0.000000                1.697   <- garbage
    1.0e-20   1.696897576253e-20      -0.000000                1.697   <- garbage

For a razor-thin disk `Fz(z->0+) -> -2 pi Sigma(R)`, a **constant**. Below ~1e-6
scipy's adaptive quad misses the width-`z` peak of `1/((a-R)^2+z^2)` and returns
~0 (wrong sign, proportional to z). It is not even a clean cutoff — 1e-7 is
correct while 1e-6 and 1e-8 are not, so the whole band is unreliable.

**The test passes for the wrong reason.** It asserts `Fz/z` constant for
z <= 1e-8; that holds at ~1.697 only because those points are garbage. Correct
physics gives -1.8646/z, varying by orders of magnitude, and the test would fail.
So the clamp converts a nan into a *silently wrong* number — arguably worse than
the nan, since a nan at least announces itself.

**Why my verification this session did not catch it.** I ran the control
(fails without the clamp, passes with it) and treated the flip as sufficient. It
is not: it only proves the clamp changes nan -> finite. A control distinguishes
"the change does something" from "the change does nothing" — it says nothing
about whether the resulting value is *right*. The stated directive is meaningful
tests near the numerical limit, and "assert Fz/z is constant" is not that; it is
an isfinite-class check wearing a ratio.

**Correct fix direction** (from the original comment, still right): keep the
clamp — m > 1 genuinely does produce nan — **and** force the quadrature to
subdivide where the integrand's mass is, e.g. explicit break points at
`R +/- c*z`, mirroring the graded panels the backend path already uses. The test
must assert `Fz -> -2 pi Sigma(R)`, not `Fz/z` constant.

### 2026-08-04 — #1245 rework: fix direction VERIFIED numerically (Jo approved the rework)

Two candidate fixes measured on the real integrand at R=0.5 (values below omit
the `amp` factor -- compare across z, not to -2 pi Sigma directly).

**(a) Break points at `R +- k|z|` — necessary but still not sufficient.**

     z       current(points=[R])   with R+-k|z|
    1e-6      +1.91e-06  WRONG      -2.10295  ok
    1e-8      +1.91e-08  WRONG      -2.10295  ok
    1e-12     +1.91e-12  WRONG      -2.09773  ok
    1e-16     +1.91e-16  WRONG      -2.14190  ~2% off
    1e-20     +1.91e-20  WRONG      +1.91e-20 WRONG (points collapse to {R})

Below ~eps*R (~1e-16) you cannot place a break point within an ulp of R, so the
break-point set degenerates back to `[R]`.

**(b) Change of variables `a = R + |z| t` — correct at every scale.** The point
is that `(a-R)^2 + z^2 == z^2 (1 + t^2)` is formed **analytically**, never by
subtraction, so there is no cancellation however small z is; and `da = |z| dt`
cancels the `-4z` prefactor, making the z->0 limit manifestly finite.

     z          substitution     ratio to z=1e-4
    1e-8        -2.1029522        1.0000910
    1e-16       -2.1028183        1.0000274
    1e-20       -2.1028183        1.0000274
    1e-30       -2.1028183        1.0000274
    1e-100      -2.1028183        1.0000274
    4.94e-324   -inf              z*z underflows to 0

One trap of my own on the way: I first derived the t-limits by round-tripping
through `a` (`(lo-R)/az`), and `R +- 1e-26` collapses to `R`, so the near region
got zero width and 1e-30 still failed. Taking the t-limits directly
(`t in [-T, T]`, clipped only by `R/|z|`) fixes it — the substituted integrand
never needs `a` to differ from `R` numerically, because the singular factor is
carried exactly by `z^2(1+t^2)`.

**Open design point for the implementation:** applying the substitution
unconditionally would change quadrature — and therefore values — for *all* z,
including ordinary ones. To keep the numpy path byte-identical outside the
pathological band, gate it on |z| being small relative to R (the two schemes
agree at the crossover: both give -2.10276 at z=1e-4). Threshold choice is the
one judgement call; will state it explicitly.

### 2026-08-04 — #1245 reworked (`6d29843f`): zforce fixed properly, scope cut on evidence

Jo approved the rework. Implemented, and the scope **shrank** twice on measured
evidence rather than on judgement.

**Fixed and verified.** `_zforce` now resolves the width-|z| peak by
substituting `a = R + |z| t`, with `(a-R)^2+z^2 == z^2 (1+t^2)` formed
analytically. `Fz/(-2 pi Sigma)` = 0.99999999 at 1e-8 and 0.99994 down to
1e-30, against `+1.9e-8` (wrong sign) before. numpy byte-identical at 15 values
of z from 2.0 to 5.1e-5 and at z = 0 — the substitution is gated on
`|z| < 1e-4 * R`. The test now asserts `Fz -> -2 pi Sigma(R)`, sign, and oddness
in z, replacing the `Fz/z`-constant assertion that passed on garbage.

**Trap 1 — the blanket application was wrong, and the code disproved it.** I
transformed all six integrands mechanically (each taking `d2`, routed through
the helper). `Rforce` then returned `-inf`. The reason is structural: `zforce`
carries a `-4z` prefactor that cancels the `1/|z|` the peak contributes, so its
substituted integrand is finite. `Rforce` and the second derivatives have no
such prefactor; their near region relies on an odd-part cancellation about
a = R, which the substitution evaluates as a difference of huge numbers.
Reverted to the zforce-only change rather than ship a transformation I had just
falsified.

**Trap 2 — z == 0 fell into the substitution branch** and divided by |z|,
NaN-poisoning even `surfdens` (which the new test calls to build the expected
limit, so every assertion went nan at once). Guarded back onto the original
path: z == 0 is the genuinely singular case and the callers special-case it.

**Honest residue.** `Rforce`, `R2deriv`, `z2deriv` and `Rzderiv` still have the
same unresolved peak and are inaccurate — though the clamp keeps them non-nan —
in that band. Two measured examples: `R2deriv(0.5, 1e-8)` is `-inf`, and
`Rforce` is finite but unvalidated there. The test states this explicitly
instead of implying coverage via an isfinite sweep. Per-method treatment is a
real follow-up, not a mechanical repeat.

Also worth noting for the record: the first commit attempt was **rejected by
ruff-format** and HEAD did not move; re-ran the test after its reformat and
committed again, verifying the SHA changed.

### 2026-08-04 — DECISIONS FROM JO (4 answered)

1. **#1245**: *extend first, then merge*. Do NOT land the zforce-only fix; cover
   `Rforce`, `R2deriv`, `z2deriv`, `Rzderiv` too so main never carries a partial
   fix. These need per-method analysis — the zforce substitution overflows on
   them (no `-4z` prefactor to cancel the peak's `1/|z|`; they rely on an
   odd-part cancellation about a = R).

2. **actionAngle 10 entries**: *re-base the tests on a gold reference*. Stop
   treating scipy's adaptive quad as truth; compare both paths to a
   high-accuracy reference. The better test and the larger change — explicitly
   chosen over a backend-specific loosened bound.

3. **AnyAxisym `1-m` cancellation**: *implement backend-only AND fix numpy*,
   with the **numpy fix on main** and the backend fix on feat/backends-develop.
   Two PRs, two branches.

4. **test_SpiralArmsPotential `_NUMPY_1_23`**: *extend the version band* so the
   intended 3e-6 applies to modern numpy (currently the `< 1.27` factor expires
   it and the tight 1e-6 applies on numpy 2.4).

**Structural consequence.** (1) and the numpy half of (3) touch the *same file*
(`AnyAxisymmetricRazorThinDiskPotential.py`) and are the same failure mode —
both are about small `(a-R)^2 + z^2` near the plane: (1) is the quadrature not
resolving the width-|z| peak, (3) is the elliptic complement losing precision.
Folding them into one coherent "AnyAxisym near-plane accuracy" PR (#1245) avoids
a same-file conflict and matches the stated preference for less-atomic PRs.
Proceeding on that basis; trivially splittable if Jo prefers.

Resulting work, by target branch:

    main                      #1245 extended: peak resolution for all 5 methods
                              + exact algebraic complement / scipy ellipkm1
    main                      SpiralArms _NUMPY_1_23 band extension (small)
    feat/backends-develop     AnyAxisym backend 1-m: complement-taking ellipkm1
                              in the backend special layer, drop _ELLIP_M_CAP
    feat/backends-develop     actionAngle gold-reference tolerance rework (10)
    feat/backends-develop     queue: #1252 (open) -> diskdf -> interpSpherical

### 2026-08-04 — #1245 extension: measured the other four methods; scope is 3, not 4, and two differ in kind

Jo asked to extend #1245 to the remaining methods before merging. Measured each
against its z=0 value (which is computed by a separate, working path) at R=0.5:

    __call__   z=0 -1.75999307   small-z: converges       -> ALREADY FINE
    Rforce     z=0 -1.99174221   small-z: converges       -> ALREADY FINE
    R2deriv    z=0 -2.28658684   -inf at 1e-8 and 1e-10, fine at 1e-12/1e-20
    z2deriv    z=0 -1.69689758   +inf at 1e-8, fine elsewhere
    Rzderiv    z=0  0.0          -5.5925 @1e-4 (correct), -5.86e5 @1e-6, then ~13.8*z

**`Rforce` is not broken.** The `-inf` I reported earlier came from *my* blanket
substitution, not from the shipped code. Its error against the z=0 value falls
2.8e-2 -> 2.8e-3 -> ... -> 2.2e-16, i.e. textbook `O(z)` convergence. Same for
`__call__`. So two of the four need no change at all.

**`Rzderiv`'s correct limit is checkable analytically**, which pins which values
are the garbage: `Rzderiv = -dFz/dR` and `Fz -> -2 pi Sigma(R)`, so the limit is
`-2 pi Sigma'(R) = 3 * (2 pi Sigma) = 5.59` for `Sigma ~ exp(-3R)`. The measured
-5.5925 at z=1e-4 is therefore **right**, and -5.86e5 at 1e-6 plus the `~13.8*z`
tail below it are wrong — the same peak-not-resolved pathology as zforce, which
makes sense since it carries the same z prefactor and a 1/d2^2 factor.

**`R2deriv`/`z2deriv` fail differently**: not a systematic wrong value but
sporadic overflow to +-inf at isolated z (1e-8, 1e-10), with correct values
either side. That is an overflow in the 1/d2^2 factor, not the quadrature
stepping over the peak, so it likely wants a different remedy than the
substitution.

Revised scope for the extension: **Rzderiv** (same fix as zforce) and
**R2deriv/z2deriv** (overflow, needs its own analysis). `__call__` and `Rforce`
are already correct and will be left alone -- with a test that says so, so the
claim is checked rather than assumed.

### 2026-08-04 — #1245 extension DONE (da398e0b): the clamp *created* the infinity

Two defects, one file, both now fixed.

**1. The clamp was the bug, not the cure.** `numpy.minimum(4aR/aRz, 1.0)` maps m
to *exactly* 1.0 once rounding tips it, and `ellipk(1.0) = inf`. So the guard I
had inherited from the original #1245 was manufacturing the +-inf that R2deriv
and z2deriv returned at isolated z (1e-8, 1e-10).

The complement is available with **no subtraction at all**:

    1 - m = 1 - 4aR/((a+R)^2+z^2) = ((a-R)^2+z^2)/((a+R)^2+z^2) = d2/aRz

which is the quantity the integrands already build. Switching all five K calls
to `special.ellipkm1(d2/aRz)` makes R2deriv and z2deriv finite at every |z|
with no substitution and no clamp. E is bounded (E(1)=1) so its clamp stays,
now only guarding the >1 rounding artifact.

**2. Rzderiv needed the peak resolved too**, like zforce -- same z prefactor,
same 1/d2^2. Its coefficients are now written in `u = a-R`, expanded via sympy:

    P(u,z) = 16R^3 u + 12R^2 u^2 - 4R^2 z^2 + 4R u z^2 + 4R u^3 + u^4 + 2u^2 z^2 + z^4
    Q(u,z) = d2 * (2Ru + d2)

P vanishes at (a=R, z=0) and Q factors through d2, so neither is formed by
cancelling large terms. Verified against the analytic limit `-2 pi Sigma'(R)`
(= 5.5939 at R=0.5, 0.6850 at R=1.2 for Sigma ~ exp(-3R)) -- clean O(z)
convergence 1e-3 -> 1e-8, where before it was wrong below ~1e-5.

**Accuracy floor, documented in the source:** Rzderiv is a 1/z-sized residual of
1/z^2-sized pieces cancelling by oddness about a=R, so relative accuracy floors
at ~eps/|z| (1.9e-8 at z=1e-8, 9.9e-6 at 1e-10, garbage at 1e-12). Cancelling
analytically would need Sigma'(R), unavailable for a user-supplied callable.
Not hidden: stated at the helper.

**Scope correction:** `__call__` and `Rforce` were never broken. The `-inf` I
reported for Rforce earlier came from *my* blanket substitution.

**The test was passing for the wrong reason.** `test_2ndDeriv_potential` FD-checks
`z2deriv = d zforce/dz` at Z=0. On main, `zforce(0.5, 1e-8)` returns +1.6969e-8
-- which is *exactly* `-z2deriv(R,0) * z`, so the difference quotient reproduced
z2deriv perfectly. The test was validating a broken zforce and agreeing with it.
Once zforce is correct the FD spans a genuine discontinuity (Fz jumps
+2piSigma -> -2piSigma) and blows up to 1.86e8. Excluded AnyAxisym at z=0
exactly as galpy already excludes `KuzminDiskPotential` there, for the same
reason; the Rzderiv mixed check needed it too (FR has a |z| kink). The physics
is still asserted, by the small-z limit test that checks Fz -> -2 pi Sigma.

Full test_potential.py: 8 failures, identical to pristine origin/main (this
worktree has no libgalpy.so). Pushed 6d29843f..da398e0b; #1245 still draft
pending CI.

**Noted, not done:** RingPotential divides by `1.0 - m` directly at four sites
with the same parametrization. There m->1 only *at* the ring, a physical
singularity rather than an artifact, so it is not the same bug -- but it is
worth a look later. Not folded in: this is a bugfix, not a refactor.

### 2026-08-04 — backend counterpart: ellipkm1 in galpy.backend.special (9e628142, held)

Jo's answer on the 1-m question was "implement and fix numpy too, but numpy
fix needs to go on main". Numpy went to main in #1245; this is the backend half,
on `fix/anyaxisym-backend-ellipkm1` off feat/backends-develop. **Committed but
NOT pushed** -- #1252 is the active develop PR and it is one at a time.

The backend AGM fallback had the identical defect: `b0 = xp.sqrt(1.0 - ms)`
forms the complement by subtraction, so K is unreachable near m=1. `_agm_KE`
now takes an optional exact `m1`, and `ellipkm1` supplies it. Placement: the
shared `galpy/backend/special/` module (fallback + router + __init__ export),
NOT the potential -- it is a general special function, and `_NATIVE_MISSING`
already has a capability test that will now police the new name too.

Validated against `scipy.special.ellipkm1` to **2.2e-16 on numpy, jax and
torch**, over m1 from 0.9 down to 1e-300. New tests in test_backend_special.py:
value parity at rtol=1e-13, gradients against the closed form
`dK/dm1 = -(E - m1 K)/(2 m m1)` at rtol=1e-11 on jax and torch. 13 elliptic
tests pass; the full file is 174 green.

**AnyAxisym's `_ELLIP_M_CAP = 1-1e-15` is now deleted, not relocated.** The cap
was censoring the log singularity itself:

    1-m=1e-16: true K=19.807, capped K=18.656   (6% low)
    1-m=1e-18: true K=22.110, capped K=18.656   (16% low, saturated)

`1-m = ((a-R)^2+z^2)/((a+R)^2+z^2)` is exact AND lies in [0,1] by construction,
so `m = 1-m1` can never round above 1 -- there is nothing left for a clamp to
guard.

**MEASURED, NOT ASSUMED: this does NOT clear the AnyAxisym jit ledger entries.**
Ran both jax-jit entries with the fix in place: `xx`, still xfail. The ledger
comment was right that they are BY DESIGN -- the limit is fixed-order GL failing
to resolve the a=R principal value, which is a different thing from K's
accuracy near m=1. So **task #187's premise ("cancellation in 1-m is the real
defect (4 jit entries)") is wrong**: fixing the cancellation is worth doing on
its own merits, but the entries stay. Ledger unchanged at 34.

Worth noting for whoever picks up the GL question: the substitution that fixed
numpy (a = R + |z|t, mapping the peak to unit width) is exactly what a
peak-resolving GL rule would need, and it is differentiable. That is the real
route to those entries, not the special function.

### 2026-08-04 — actionAngle ledger: the gold-reference plan is WRONG for these entries

Jo's decision on the 10 actionAngle tolerance entries was "re-base tests on a
gold reference". Started that, and the first measurement invalidated the premise.

Took the two torch `actionAngleVertical_Harmonic_*` entries first, because a
harmonic oscillator has exact closed-form J and Omega. Real failure (via
`--runxfail` + junitxml -- stdout is truncated by the conftest force-exit):

    Frequencies via actionAngleVertical vs actionAngleHarmonic
    tolerance 1e-10, actual 2.76e-09 / 3.20e-09

**That test already compares against a gold reference** -- actionAngleHarmonic
IS the analytic solution. So there is nothing to re-base. Measured all three
backends against the exact Omega = omega = 2.23, in float64 (conftest sets
jax_enable_x64 / torch.set_default_dtype; a first probe without that was
float32-contaminated and I discarded it):

    numpy  J err 0.0e+00   Omega err 2.887e-15
    jax    J err 6.7e-16   Omega err 1.275e-13
    torch  J err 6.7e-16   Omega err 6.364e-10     <- ~5000x worse than jax

Actions agree to machine precision everywhere. **Only torch's frequency is
wrong, and it is genuinely wrong** -- not a too-tight tolerance.

**Root cause (measured, not inferred).** Omega and J share one substitution
`x = xmax - t^2`, differing only in how the radicand enters:

    J:      sqrt(rad) * 2t * lim      rad clipped to 0
    Omega:  2t / sqrt(rad) * lim      rad clipped to 1

`rad = 2(E - Phi(xmax - t^2))` is a difference of nearly equal numbers near
t=0. J *multiplies* by it and by 2t -> 0, damping the error. Omega *divides*
by its square root, amplifying it. So Omega is acutely sensitive to how exactly
xmax satisfies E == Phi(xmax). Measured the turning point directly:

    exact  xmax = 0.16762485393514812
    jax    rel.err 0.000e+00   residual E-Phi(xmax) = +1.388e-17
    torch  rel.err 2.771e-13   residual E-Phi(xmax) = -3.872e-14   <- NEGATIVE

torch's xmax **overshoots**, so rad goes negative just where Omega's integrand
carries weight; the AD guard `xp.where(rad > 0, rad, ones)` then substitutes
**1** for the true small radicand and `2t/sqrt(1) ~ 0` silently deletes part of
the integral. The guard that stops a NaN is what turns a last-bit bracket error
into a 6e-10 frequency error. jax's bracket happens to land on the +side, so it
never trips.

**Fix direction (not yet implemented):** `backend.optimize.bisect_root` returns
the final bracket *midpoint*, which can land either side of the root. For any
consumer that then takes sqrt(E - Phi(xmax)), landing on the minus side is
structurally unsafe. Returning the bracket endpoint where f >= 0 instead makes
`rad >= 0` true by construction, for every consumer, and removes the reliance on
which side the midpoint happens to fall. numpy is unaffected (it uses scipy
brentq). That is a small change in one shared helper -- next wake.

**Do not "fix" this by loosening the tolerance.** 1e-10 is met by numpy (2.9e-15)
and jax (1.3e-13) with 3-5 orders of margin. Loosening would hide a real torch
defect, and per Jo's stringency directive a big loosening is a red flag.

### 2026-08-04 (later) — CORRECTION to the entry above, and the actual root cause

**The mechanism I wrote yesterday was wrong.** I said torch's negative residual
drove `rad` below zero so the AD guard `xp.where(rad > 0, rad, ones)` substituted
1 and deleted part of the Omega integrand. Checked it instead of asserting it:

    exact xmax  resid=+1.39e-17  min(rad)=+8.977859e-08  nodes with rad<=0: 0/50
    torch xmax  resid=-3.87e-14  min(rad)=+8.977851e-08  nodes with rad<=0: 0/50

**Zero GL nodes hit the clip.** The clipped region is |t| < 2e-7 and the smallest
n=50 node sits at t ~ 2.5e-4, three orders outside it. The guard never fires.

The real mechanism is plainer: the residual shifts `rad` by a constant 2*resid at
*every* node, and `1/sqrt(rad)` amplifies that most where rad is smallest. Feeding
each xmax into a clean numpy quadrature reproduces both numbers exactly
(torch xmax -> 6.364e-10, exact xmax -> 1.279e-13), which pins xmax as the whole
story. It also kills my proposed fix: returning a bracket *endpoint* would make
the residual bigger, not smaller.

**Actual root cause: torch skips the Newton polish on the plain forward path.**
`brentq` dispatches to per-backend files. `_jax/optimize.py` always does the
Newton reparameterisation `x* = x0 - f(x0)/f'(x0)`. `_torch/optimize.py` has an
early return -- when no parameter requires grad, it returns the raw bisection
root, to keep an autograd graph off the output so callers can `.numpy()` it.
Sound reason, but it overlooked that the Newton step also improves the **value**:
bisection stops at the final bracket half-width (~1e-14), and the plain forward
path is exactly what the test suite exercises.

**Fix (`fix/torch-brentq-newton-polish`):** take the Newton step in that branch
too, computing df/dx under a local `torch.enable_grad()` and detaching, so the
result is polished *and* graph-free. Measured on the harmonic oscillator:

    xmax rel.err  2.771e-13 -> 0.000e+00   (residual -3.87e-14 -> +1.39e-17)
    Omega rel.err 6.364e-10 -> 1.279e-13   (jax: 1.279e-13 -- identical)
    J     rel.err 4.4e-16 (unchanged)      returns graph-free: yes

torch now matches jax to the digit. Whether the ledger entries flip is measuring
now -- this touches every backend brentq consumer, so the full torch
test_actionAngle run is the real check, not the two harmonic tests.

### 2026-08-04 (later still) — torch Newton-polish fix committed (3da30bff)

Branch `fix/torch-brentq-newton-polish` off feat/backends-develop. Committed,
not pushed (#1252 still the active develop PR).

Change: `_torch/optimize.brentq_backend` now takes the Newton step on the
no-grad forward path too, computing df/dx under a local `torch.enable_grad()`
and detaching, so the value is polished AND graph-free. Nine lines.

**The test that should have caught this was too loose.**
`test_backend_root_value` asserted `rtol=1e-9` on a quantity the polish puts at
machine precision -- four orders of slack, so a backend silently returning the
raw bisection root (~1e-14) passed it comfortably. Tightened to 1e-15 and added
`test_backend_root_is_polished_not_bracket_limited`, which asserts the root is
within a few ulp AND that |f(root)| is at machine level.

Verified discriminating by reverting just the source file and re-running:

    WITHOUT fix: 2 failed, 2 passed  ("torch root off by 1.357e-13")
    WITH fix:    4 passed
    jax passes either way -- it always polished.

That is the check from [[verify-burndown-flips-stash-rerun]]: a test that does
not fail without the fix proves nothing. Full tests/test_backend_optimize.py:
26 passed.

**Ledger impact NOT yet known.** The full torch test_actionAngle.py run (209
tests, --runxfail so ledgered entries report their real result) is still going
-- ~35% after several minutes. Deliberately not quoting a burndown number until
it finishes: this touches every backend brentq consumer (Vertical, Spherical,
Staeckel turning points, estimateBIsochrone), so the two harmonic tests are not
evidence about the other seven entries, and I need the non-ledgered failures too
to rule out a regression.

One caveat to re-check when it lands: I briefly reverted optimize.py mid-run for
the discriminating test above. Python caches modules at first import and the run
had already executed 50+ turning-point tests by then, so it held the fixed
version throughout -- but any entry that flips gets re-run individually before I
touch the ledger.

### 2026-08-04 — follow-up audit: is the polish defect systemic? No.

The torch brentq bug was a *jax/torch structural asymmetry that changed values,
not just gradients*. Checked whether any sibling has the same shape.

Backend pairs with separate per-backend files: adiabatic_c, jacobian, optimize,
orbit_ode, orbit_stm, staeckel_c. Grepped `_torch/` for the giveaway pattern --
a grad-conditional branch (`requires_grad` / `is_grad_enabled` / `no_grad` /
`detach`) that decides what gets RETURNED:

    orbit_stm.py:20,22     detach().cpu().numpy()  -> feed the C bridge
    adiabatic_c.py:17,45   detach().cpu().numpy()  -> feed the C bridge
    staeckel_c.py:18,48,75,116  same

All of those detach only to hand data to a C routine, then wrap the result back;
none of them *return* a different value depending on whether grad is required.
**`optimize.py` was the only one with a grad-conditional early return**, so the
defect class is confined to what is now fixed. Bounded, not systemic.

### Noted for Jo, deliberately NOT acted on: loose tolerances in backend tests

The bug survived because `test_backend_root_value` used `rtol=1e-9` on a
machine-precision quantity. Surveyed how common that shape is: **~300 sites of
`rtol=1e-6..1e-9` across 49 `test_backend_*.py` files**, e.g.

    test_backend_actionAngle.py    52      test_backend_streamTrack.py   26
    test_backend_orbit_integrate.py 24     test_backend_interpolate.py   15
    test_backend_quadrature.py     14      test_backend_special.py       13

Most are legitimate -- quadrature-vs-adaptive, finite-difference references,
sampling comparisons -- where 1e-6 is the honest floor. The ones that matter are
backend-vs-numpy comparisons of quantities computed by the *same* algorithm,
where a loose rtol hides a backend silently diverging. Separating those needs
per-test judgement, and tightening ~300 assertions across 49 files is a
test-policy change with real churn and flake risk, not a bugfix. Flagging it as
a decision for Jo rather than doing it unilaterally on vacation. Cheap first
step if he wants it: tighten only where numpy and backend run the same
algorithm, one file per PR, starting with test_backend_optimize.py (done).

### 2026-08-04 — torch polish: 6/9 AA entries flip, but the 26 others need a baseline

Full torch test_actionAngle.py finished (209 tests, --runxfail). Ledgered:

    pass  actionAngleSpherical_almostnoninclinedorbit_linear_angles
    pass  actionAngleSpherical_linear_angles
    pass  actionAngleSpherical_linear_angles_fixed_quad
    pass  actionAngleSpherical_noninclinedorbit_linear_angles
    pass  actionAngleVertical_Harmonic_actionsFreqs
    pass  actionAngleVertical_Harmonic_actionsFreqsAngles
    FAIL  actionAngleAdiabaticGrid_outsidegrid_c
    FAIL  actionAngleAdiabatic_linear_angles_cylsep
    FAIL  actionAngleStaeckelGrid_basicAndConserved_actions
    => 6/9

The four Spherical `linear_angles` entries flipping is the satisfying part: they
were never diagnosed, and they share the turning-point root-find with Vertical,
so the same polish fixes them. That is the mechanism predicting results it was
not fitted to.

**Not claiming 6/9 as a ledger delta yet.** The run also shows 26 NON-ledgered
failures. This worktree has **no libgalpy.so** ([[worktree-missing-c-ext-fakes-passes]]),
and 17 of the 26 end in `_c` -- but that leaves 9 that do not
(kepler_actions, HarmonicInverse_orbit, IsochroneInverse_wrtIsochrone,
Spherical_EccZmaxRperiRap_againstOrbit, nullpotential_error, ...). Those could be
C-dependent internally, or they could be regressions from the polish. Guessing
is exactly what burned me on the AnyAxisym clip claim earlier today.

Running the identical suite on a pristine pre-fix worktree (`/var/tmp/aa-baseline`
at c34abfa3, verified `enable_grad` count = 0) and diffing the failure sets. Only
names that FAIL after and PASS before are regressions. Result next wake.
