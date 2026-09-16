# Vacation status snapshot — 2026-09-16 (overnight, updated)

**This file is OVERWRITTEN, not appended.** "Where are things now";
`log.md` is the trace. Refresh on every merge / branch change.

---

## Now

    feat/backends  0c05087f3   main
    merged today   #1496 #1497 #1498 #1499 #1500 #1503

Jo is asleep; instruction is to keep pushing the streamdf work through the
night. One PR at a time; Jo merges.

## RESOLVED as a diagnosis: the feat/backends jax red is a bit-identity test

Perturbing `sigv` by 1 ulp and re-sampling moves `streamgapdf.sample(1, lb=True)`
by 1.2e-5 - 5.4e-5 -- and the response is FLAT from 1 ulp to 1e6 ulp, six orders
of magnitude. That is a discrete jump, not a derivative. So the 1e-5 bar has no
margin at all: the test passes only while the two objects agree BIT-FOR-BIT, and
any last-bit difference lands over the bar.

Measured identically (numpy 1.24e14 apparent amplification) at 12f37425f and at
8c99e2034, so the fragility is PRE-EXISTING -- #1499 only moved the last bit.
Not reproducible off-CI in six configurations (isolated + full shard, each at
jax 0.10.1 and CI's jax 0.11.1 + numpy 2.5.3, plus astropy 8).

Conclusion: do NOT chase this as a precision regression in #1499. It is a
test-design decision -- compare a stable quantity, or widen this one comparison.
`_NUMPY_1_22` is not broken; 1e-5 is deliberate for modern numpy.

## Original evidence trail: a real red on feat/backends, from #1499

    backend jax  tests/test_quantity.py tests/test_coords.py
    test_streamgapdf_sample  ->  1.7650297e-05  vs a 1e-05 bar

* Bisected to **#1499** (streamdf C.3): `12f37425f` green, `8c99e2034` red,
  and that range is exactly one commit.
* **Deterministic, not a flake**: the base and #1503 produced the value
  *byte-identically* (1.7650297e-05 both times, 19 h apart).
* #1503 `9629bcc83` looked green only because that revision still had the
  force-PROBE guard, whose cache side effect perturbed the sampled values.
  Removing the probe (correct) un-masked the base red. #1503 was the
  messenger, not the cause.
* **jax-only**: the torch and plain-numpy shards pass the same test at the
  same commit.
* Does **not** reproduce locally: full shard at jax 0.10.1 = 344 passed;
  isolated probe = 0.0 difference on BOTH numpy and jax, and still 0.0 after
  upgrading to CI's jax 0.11.1 + numpy 2.5.3. The remaining untested cell,
  full shard x jax 0.11.1, is running now (`/var/tmp/jax011` target-dir
  install; does not touch any existing env).
* On numpy the unitful and non-unitful objects agree to **exactly 0.0**, so
  the 1.8e-5 is entirely a jax-path artifact, not a pre-existing tolerance.

## streamdf differentiable `__init__` — layers 9-16

Chain of `theta`-gradient blockers cleared tonight in `/var/tmp/spray`
(branch `backend/aa-parse-args-backend`, uncommitted):

     9  Orbits `_call_internal`      traced self.t kept off numpy.asarray
    10  `_call_internal_backend_interp`  traced grid ordered by argsort, not a
                                    concrete `descending` compare (concrete
                                    path untouched -> numpy byte-identical)
    11  `cubic_spline_coeffs`       backend knots kept; tridiagonal A built
                                    from one-hot rows when the knots are traced
    12  `_deltaAngleTrack`          dAT * linspace(0,1) (torch drops endpoint grad)
    13  `Spline1D` + streamdf 2409  traced knots survive into the spline
    14  coords `galcencyl_to_XYZ`,  marked @backendNative (innermost, under
        `galcencyl_to_vxvyvz`      @scalarDecorator) -- their rect twins
                                    already were
    15  `_cart_and_interp_cov_backend`  the twin knot site
    16  diffrax adaptive + traced t0/t1 -> `checkpointed_while_loop` cannot
        reverse-differentiate. NOT a defect: galpy already documents that
        constant stepping fixes this, and the backend track path already
        forwards `self._aA._integrate_kwargs`. The probe just had to opt in
        (`nsteps`).

Verified along the way, against FD rather than finite-and-nonzero:

    traced-vs-concrete spline coeffs   5.55e-17  (both bc)
    d(coeffs)/d(knots) AD vs FD        8.97e-10 -> 5.02e-11  (converges with h)
    galcencyl_to_XYZ numpy-vs-jax      5.55e-17 ; d(XYZ)/dR AD-vs-FD 3.22e-10
    galcencyl_to_vxvyvz numpy-vs-jax   3.47e-18

## Disk was at 98%; now 79%

`/dev/sda6` on `/` had 780 MB free. **368 stale galpy worktrees** under
/var/tmp (of 416) removed with plain `git worktree remove` -- no `--force`, so
the **44 with uncommitted work were skipped and kept**; branches are all still
in the repo. Plus the duplicate `venv-p23` (identical to `venv-p24`: py3.13 /
numpy 2.4.6 / jax 0.10.1 / scipy 1.17.1). 780 MB -> 22 GB free.

Still large, deliberately untouched: `galpy-gpu` 8.8 G (the pinned TITAN Xp
env), `galax-src` + `galax-venv` 6.1 G, and ~4.3 G under /tmp belonging to
other projects.

## Care note

A heredoc `open(p,"w")` hit ENOSPC mid-write and **truncated**
`galpy/backend/interpolate.py` to zero. Its diff vs HEAD was exactly 1655
deletions / 0 insertions, so nothing uncommitted was lost and
`git checkout --` restored it byte-for-byte. All three worktrees were then
scanned for other zero-length files (none). Every edit since goes through
write-temp + `os.replace`.


## Tonight's landings

* **#1504 opened** (`backend/traced-time-grid` -> feat/backends): orbit
  integrate/spline/read-back on a TRACED time grid. Gates 498 passed under each
  of numpy / jax / torch; numpy byte-identical (digest a0d3c99f6b0e6b845fc42ce6,
  27 arrays).
* **PR B committed** (`backend/streamdf-traced-track`, stacked on #1504):
  streamdf track determination on a theta-dependent grid. Gate running.

## streamdf __init__ now RUNS under jax.grad -- but the gradient is frozen-frame

All 16 layers cleared; `jax.grad(loss)(0.9)` returns. Reverse-mode needs CONSTANT
stepping (traced t0/t1 make the adaptive loop bound dynamic, which equinox's
checkpointed_while_loop cannot reverse-differentiate); the backend track path
already forwards `_aA._integrate_kwargs`, so a caller opts in with
{"nsteps": 200, "adjoint": "direct", "max_steps": 260}.

The number it returns is NOT the total derivative:

    AD, eigenvectors frozen (current)     -8.5110
    AD, eigenvectors unfrozen             -4.9518
    central FD, Richardson-extrapolated   -5.008

real_eig's stop_gradient on the eigenvector rotation costs **70%** of the
derivative -- the code comment in test_backend_streamdf.py says "~10%", which
understates it 7x. Unfreezing recovers the FD value to ~1%. stop_gradient is the
identity in the forward pass, so both AD arms share those FD values.

Deciding how to differentiate that rotation (it diverges as 1/gap at degeneracy,
which is why it was frozen) is the next real question, and it is yours.

## An end-to-end streamdf grad test is NOT CI-viable

Even at nTrackChunks=3, tintJ=8, nsteps=40 the FORWARD pass alone is 359 s,
over the 300 s per-test cap, before any gradient. PR B therefore tests the grid
CONSTRUCTION (cheap, exact) and documents the end-to-end numbers instead.


## Morning state (2026-09-16)

    #1504  backend/traced-time-grid        GREEN 104/104, all-clear + codecov OK,
                                           MERGEABLE/CLEAN.  Yours to merge.
    #1505  backend/streamdf-traced-track   98 green / 1 red, stacked on #1504.
           codecov/patch 100% (20/20), head misses == base misses == 14.

### #1505's single red, stated carefully

    feat/backends (base)   RED   1.7650297e-05
    #1505 0af95e10e        RED   1.7650297e-05
    #1505 2ff462090        GREEN (343 passed)
    #1505 ca05b3f72        RED   1.7650297e-05   <- current

The _deltaAngleTrack regression WAS real (CI caught it; _ns_coerce resolves the
AMBIENT namespace, so a forced backend turned a derived scalar into a tensor and
the numpy track path died). Fixing it is what made 2ff462090 green. Removing the
unverified _firstFlip concat arm afterwards -- correct on its own terms -- put the
value back to EXACTLY the base's 1.7650297e-05, 8 digits. At the measured ~7e8
relative amplification an 8-digit coincidence is not plausible, so #1505 now
reproduces the base's failure rather than a new one.

Four hypotheses tried; three disproved by measurement (float() grid bits are
identical; _bk_assemble; the numpy-path thetasTrack reader). No confirmed
mechanism for each flip -- only the established fact that the test has ZERO
tolerance margin, so its outcome tracks incidental arithmetic anywhere in the
jax stream path.

### Also landed

* _firstFlip (a TRAILING stream) with a backend orbit now raises
  NotImplementedError. It never worked -- it died with "JAX arrays are
  immutable" before this branch existed -- and reproducing the numpy
  reversed-buffer order on top of that is unverified (measured ~10% off). A
  silently different orbit is worse than a refusal. Trailing streams with a
  backend progenitor are an open follow-up.
* _span_grid / _ns_sqrt helpers: the traced arms are now reachable by a 14 s
  unit test instead of a 359 s streamdf build, and the three call sites no
  longer duplicate the if/else. Both gate on TRACEDNESS, not backend-ness.

### Unpushed: backend/unfreeze-stream-frame

Unfreezes ONLY the well-separated decomposition (top eigenvalue separated by
99.8% of the spectrum norm at q=0.9 and 0.95; the near-degenerate pair feeds
nothing). AD -8.5110 -> -4.9518 against an FD-converged -5.008, identical to a
FULL unfreeze to every digit -- so that one decomposition is the whole 70%.
298 passed under jax; numpy digest unchanged. Held back because it reverses a
deliberate #1499 design decision.
