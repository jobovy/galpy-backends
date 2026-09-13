# Vacation status snapshot — 2026-09-12 (evening)

**This file is OVERWRITTEN, not appended.** "Where are things now";
`log.md` is the trace. Refresh on every merge / branch change.

---

## Now

    feat/backends  cff1a1dc1   main  (rebase landed 2026-09-12)
    ledger         5 slow-skip, 5 xfail   (was 6 / 6 yesterday; 42 / 31 on 09-09)

Jo is away for the weekend and reviews Monday. Instruction: keep stacking PRs,
keep going, at least two open at a time.

## The stack — gh stack #1488, 8 PRs, all on feat/backends, all Jo's to merge

    #1485  oblatestaeckel-ntab-backend   GREEN 106/106   ntab on the backend +
           eval_ppoly ~10x (table cached per namespace/device, one batched gather,
           unstack, no needless reshapes). Bit-exact vs the old formulation.
    #1486  orbit-returns-backend-array   GREEN 106/106   every integrator hands a
           backend IC a backend trajectory, 1.00-1.02x, 0.0 difference. Routing
           them through the C-STM instead was 801x and rejected.
    #1489  cstm-continuation             GREEN 106/106   backend routes silently
           dropped the first leg of a continued integration. Jacobian vs FD 1e-12.
    #1490  traced-ecc-tolerance          GREEN 106/106   xfail -1.
    #1491  interp2d-dispatch             eval_rect_ppoly 16 gathers -> 1; 25.5 ->
           1.8 ms; interpRZ on backend coords 32.7 -> 2.8 ms Rforce.
    #1492  fdm-ledger-burn               slow-skip -1, three stale notes corrected.
    #1493  jac-openmp                    actionsFreqsAnglesJac 2.107 -> 0.299 s at
           16 threads; bit-identical 1-vs-16 and vs base. NOT a main PR: that
           entry does not exist on main.
    #1494  sici-cumprod                  1860 -> 272 us (6.8x), accuracy unchanged.
           Fixed its cache guard: under_jax_trace -> under_trace, because
           sici_fallback is TORCH-only so the jax predicate guarded nothing.

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
    jax-jit   orbit ecc_rperi_rap[ExpTruncNFW]   285.4s PASS / 300.1s TIMEOUT

EVERY row is now measured. Two of them FLIP: torch FDM central_limit read
291.85 s then timed out, and ExpTruncNFW read 285.41 s then timed out, both on
the same idle box in the same shard/group. A row within ~5% of the cap is not
retirable on one passing measurement -- that is what gh#1284 got wrong.
ExpTruncNFW has no dispatch win either: it is 19x plain NFW ON NUMPY
(0.573 vs 0.030 ms per Rforce), so its cost is the truncated-mass computation.

FDM is fully characterised now: leg1 (dop853_c) is 4.7s on numpy AND torch; ALL
the cost is leg2's 1001-step PYTHON stepper (55.5 -> 272.1 s). The force is down
to 22 torch ops, so there is no eval_ppoly-style win left — retiring it needs
fewer force evaluations or an in-backend stepper.
