**Backend-port burndown.** Number of tests still carried in the all-backend
`xfail(strict=False)` ledger — i.e. still failing under each backend (jax, torch)
— versus time, reconstructed exactly from the git history of
`tests/backend_xfail.txt` + `tests/backend_slow_skip.txt` on `feat/backends`.
This count is the *remaining* backend-port work; it reaches zero when the entire
existing galpy suite passes under jax and torch at numpy tolerances. Dashed lines
mark migration milestones, and the series extends one point per ledger-touching
commit as each subsystem (potentials, orbit, action-angle, df) is ported and its
tests move xfail→pass at a milestone review. The early torch≫jax gap is the
systemic torch scalar-rejection fixed by the central coordinate-coercion work.

This is the **Pillar 1** metric (the *existing* suite green under every backend).
Two caveats on absolute heights (the descent/trend is unaffected — only genuinely
ledgered, i.e. backend-exercising, tests move the count): (i) the project-start
anchor at the full suite size (2466) is an **upper bound**, since some tests are
backend-inert (pure-numpy inputs resolve to the numpy namespace and never failed
under jax/torch), so the real starting height is lower; (ii) "passing under a
backend" therefore includes such inert passes. The guarantee that the backend is
genuinely exercised and correct is **Pillar 2** (the new derivative / parity /
gradcheck tests), tracked separately.
