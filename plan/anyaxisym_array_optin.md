# AnyAxisymmetricRazorThinDisk: opting in to backend array input

**Status: proposal, needs Jo's decision on one point.** Written 2026-09-12.
Burns the last 2 burnable xfail rows; the other 3 are permanent by construction.

---

## What blocks the two rows today

    jax-jit   tests/test_potential.py::test_poisson_surfdens_potential[AnyAxisymmetricRazorThinDiskPotential]
    torch-jit  "

Reproduced 2026-09-12, `--backend jax --jit --runxfail`:

    TypeError: Methods in AnyAxisymmetricRazorThinDiskPotential do not accept
               array inputs. Please input scalars

Not a wrong number — a refusal. `Potential.py`'s array gate has an opt-in escape
hatch, `_backend_accepts_arrays`, which this class deliberately does not set.
Only `RotateAndTiltWrapperPotential` and `DoubleExponentialDiskPotential` do.

## The numerical objection is gone; I re-measured it

The opt-out's stated reason was that the traced GL path was catastrophically
wrong as z -> 0, so opting in would turn a loud `TypeError` into a silently
wrong number. That was true in 2026-08 (2.7e+04 relative at |z|=1e-5). It is not
true now — `_bk_split_quad`'s geometric ladder, gh#1330's `finite_part_quad` log
subtraction and gh#1331's relative zforce floor fixed it. Traced vs scipy at
R=1.0, measured today:

    |z|         Rforce      zforce     R2deriv     z2deriv
    1e-1     1.451e-16   1.226e-15   3.613e-14   4.998e-15
    1e-2     7.730e-16   6.185e-15   3.163e-13   7.675e-14
    1e-3     2.928e-15   3.460e-15   2.477e-12   8.505e-13
    1e-5     4.680e-11   5.923e-13   1.204e-10   3.126e-11
    1e-7     1.323e-07   9.360e-12   8.652e-08   2.171e-08
    0        5.215e-05   0.000e+00   3.864e-10   9.028e-11

The 5e-5 only appears at z = 0 exactly (the `a=R` principal value), and
`test_poisson_surfdens_potential` evaluates at `Zs = [0.125, 0.25, 1.0, 10.0]`
against a **1e-4 relative** bar (`tol["default"] = -8.0` on the squared
residual, no AnyAxisym entry). At those z the traced forces are exact to ~1e-15.
So the accuracy case is comfortable.

## What the work actually is — the flag alone is NOT enough

The old note said the remaining blocker was just the flag. **I tested that and it
is wrong.** Setting `pot._backend_accepts_arrays = True` and tracing the Poisson
surfdens gives:

    ValueError: Cannot stack arrays with different numbers of dimensions:
                got (), (100,), (100,), ...

from `xp.stack(lo, axis=-1)` in `_bk_split_quad`. With an array `z`, `dmin` and
therefore every panel edge becomes `(100,)`, while `lo[0] = zeros_like(Rp)` stays
`()`. Fixing the stack is not enough either: `fixed_quad` would then hand the
integrands `a` of shape `(100, 2K+2, n)` while `R`/`z` are `(100,)`, so each of
the six `_gl` integrands needs its captured `R`/`z` given trailing axes.

So the real task, in order:

1. broadcast `Rp`/`z` to a common shape in `_bk_split_quad` before building the
   panel ladder;
2. thread trailing node axes through to the six `_gl` integrands (`_evaluate`,
   `_Rforce`, `_zforce`, `_R2deriv`, `_z2deriv`, and the `_phitorque`/mixed one);
3. `_backend_accepts_arrays = True`;
4. **repoint the negative test** —
   `tests/test_backend_array_inputs.py::test_not_opted_in_potential_still_rejects_backend_arrays`
   names this class explicitly;
5. drop the 2 xfail rows.

Step 2 is the delicate part: the panel grading is `dmin = 4|z|` per point, so
with an array `z` every point wants its OWN ladder. That is exactly the
mechanism the accuracy above depends on, and it is the part a previous attempt
(2026-08-03) implemented and then reverted.

## The decision I need from you

**Step 4.** The negative test deliberately names `AnyAxisymmetricRazorThinDisk`
and its comment still quotes the stale "wrong by ~2e4x at |z|=1e-5" figure.
Opting in means repointing it at some other still-opted-out potential. That
changes a test written to encode a decision, so it is yours to make, not mine to
slip into a refactor.

Two sub-questions worth answering together:

- is "loud TypeError" still the behaviour you want for a potential whose traced
  path is *deliberately* less accurate than its concrete one? The class already
  ships that asymmetry for scalars; the gate only stops it for arrays.
- if yes, these 2 rows are permanent and the xfail ledger's floor is 5, not 3.
  That is a fine answer — it just wants saying, because the current ledger
  comments imply the rows are pending work.

## Why I did not just do it

It is the most numerically delicate code in the file, the payoff is 2 jit-only
rows that CI never runs (`--jit` is workflow_dispatch only), a previous attempt
was reverted, and it ends in a decision I cannot make. Doing steps 1-3
unsupervised and landing on step 4 would have been the wrong use of a weekend.
