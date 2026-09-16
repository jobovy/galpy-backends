# Vacation status snapshot — 2026-09-16 (overnight)

**This file is OVERWRITTEN, not appended.** "Where are things now";
`log.md` is the trace. Refresh on every merge / branch change.

---

## Now

    feat/backends  0c05087f3   main
    merged today   #1496 #1497 #1498 #1499 #1500 #1503

Jo is asleep; instruction is to keep pushing the streamdf work through the
night. One PR at a time; Jo merges.

## THE open blocker: a real red on feat/backends, from #1499

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
