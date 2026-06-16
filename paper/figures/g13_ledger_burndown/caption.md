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
