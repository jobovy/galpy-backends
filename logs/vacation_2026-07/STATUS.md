# Vacation status snapshot — 2026-09-12 (afternoon)

**This file is OVERWRITTEN, not appended.** "Where are things now";
`log.md` is the trace. Refresh on every merge / branch change.

---

## Now

    feat/backends  cff1a1dc1   main  (rebase landed 2026-09-12)
    ledger         5 slow-skip, 5 xfail   (was 6 / 6 yesterday; 42 / 31 on 09-09)

Jo is away for the weekend and reviews Monday. Instruction: keep stacking PRs,
keep going, at least two open at a time.

## The stack — gh stack #1488, all on feat/backends, all Jo's to merge

    #1485  backend/oblatestaeckel-ntab-backend      -> feat/backends
      OblateStaeckelWrapperPotential ntab= on the backend, differentiably.
      Spline1D.from_ppoly keeps main's exact CubicSpline (refitting would break
      numpy byte-identity AND the GSL-cspline/C agreement). Then, because CI
      showed the cost of actually USING the tables under jax, eval_ppoly itself:
      the table was re-uploaded per call, the 4 coefficient rows were gathered
      separately, and unpacking + reshapes each cost a dispatch.
        eval per spline    6.7 ms -> 0.7 ms   (~10x)
        evaluateRforces   19.10 -> 7.22 ms    (2.6x)
        the ntab test    464.7 -> 258.1 s     (so NO slow-skip row added)
      bit-exact vs the old formulation on numpy/jax/torch, nu=0/1/2.

    #1486  backend/orbit-returns-backend-array      -> #1485
      A backend IC gets a backend trajectory back for EVERY integrator (symplectic
      C incl. the default, ias15_c, the Python steppers), at 1.00-1.02x and with
      0.0 difference from the numpy values. Routing them through the C-STM instead
      was measured at 801x and rejected. Continuation made namespace-aware.

    #1489  backend/cstm-continuation                -> #1486
      The C-STM and in-backend routes silently DROPPED the first leg of a
      continued integration (101 points where numpy gets 201). Pre-existing.
      Merged trajectory stays differentiable: jacobian vs FD 1e-12.

    #1490  backend/traced-ecc-tolerance             -> #1489
      test_eccentricity traced-only bar. ledger -1.

    #1491  backend/interp2d-dispatch                -> #1490
      eval_rect_ppoly: 16 fancy gathers -> 1. 25.5 -> 1.8 ms; interpRZPotential
      on backend coords 32.7 -> 2.8 ms for Rforce. Bit-exact.

    READY, NOT PUSHED (waiting for the CI queue to drain):
    backend/fdm-ledger-burn @ 2ac500eb1  -> #1491
      slow-skip 6 -> 5, and three stale notes corrected.

## Two bugs CI found in my own interpolation work

Both in the Spline1D device cache / unstack, both invisible to the eager suite:

1. a conversion made INSIDE a jax trace is a tracer, so caching it leaked one
   trace into the next (`x = JitTracer(float64[499])`, surfacing a module away in
   jacfwd over a friction force). Gated on `under_trace`.
2. array-api-compat implements torch's `unstack` as `moveaxis`, which has no vmap
   batching rule. The existing `vmap(grad)` guard PASSES with the bug — only
   BARE vmap fails. `unstack` is now jax-only.

## Measured constants earned today

    torch CI/local      0.62   (median of 7, range 0.60-0.65) — was assumed 0.71
    jax qdf shard       BIMODAL: 0.74-0.95 for python_vs_c/linacc/impulse,
                        0.36-0.40 for the qdf family. The slow-skip file's claim
                        that "a CI shard runs ~3x slower" is wrong by ~8x.

## The 5 remaining xfail rows are at their floor, not their target

    3  pickling / unitful dens — a units-based density routes through as_numpy(),
       which a tracer cannot satisfy. Non-traceable BY CONSTRUCTION.
    2  poisson_surfdens[AnyAxisym] — the memory said only the flag blocks these.
       TESTED: not true yet. Setting _backend_accepts_arrays still dies in
       _bk_split_quad ("Cannot stack arrays with different numbers of dimensions:
       got (), (100,)"). Needs trailing node axes threaded through the panel
       ladder and all six _gl integrands, THEN Jo's call on repointing the
       negative test. Feasibility is good: the test's bar is 1e-4 and its z are
       all >= 0.125, where the traced forces are exact to ~1e-15.

## The 5 remaining slow-skip rows

    torch     FDM central_limit        291.9s in-shard, FLIPS to a 300s timeout
    torch-jit FDM central_limit        305.7s
    jax       qdf sampleV_interpolate  2837.5s in-shard (244s isolated!)
    jax-jit   qdf sampleV_interpolate
    jax-jit   orbit ecc_rperi_rap[ExpTruncNFW]   296.2s  <- NOT yet re-measured

FDM is fully characterised now: leg1 (dop853_c) is 4.7s on numpy AND torch; ALL
the cost is leg2's 1001-step PYTHON stepper (55.5 -> 272.1 s). The force is down
to 22 torch ops, so there is no eval_ppoly-style win left — retiring it needs
fewer force evaluations or an in-backend stepper.
