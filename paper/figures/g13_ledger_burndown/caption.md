**Backend port of galpy's existing test suite.** Top two panels: the size of the
existing suite (`tests/` minus the dedicated `test_backend*.py` modules; grey) and
the number of its tests that run and pass under torch and jax (solid), versus time.
Each point is a commit on `feat/backends` that touched the backend ledgers
(`backend_xfail.txt`, `backend_slow_skip.txt`, `backend_skip.txt`), collected at
that PR's head, and the last is the branch's tip at the time of writing; "passing" is the suite minus the tests that commit's own
`tests/conftest.py` takes out of the run (xfail-ledgered, slow-skipped or
permanently skipped), with each ledger entry expanded to every test it matches.
The shaded band is the remaining port work. The open circle is the pre-migration
anchor (2466 tests, nothing yet running under jax/torch); the dotted segment
bridges to the first ledger snapshot (2026-06-14). The large early steps are
milestone prunes of the ledger. The suite grows by ~1850 tests (+1094 on
2026-06-19, when `test_potential` was split into one test per potential); new
tests that passed from the start never entered the ledger.

Dashed: the same suite run traced (`--jit`: every galpy entry point under
`jax.jit` / `torch.compile`), from the mode's introduction on 2026-07-26. A
`--jit` run applies its own `<backend>-jit` ledger rows *plus* eager rows it
inherits, by rules that changed over time and are read from each snapshot's
conftest: until #1210 (2026-07-27) the `-jit` rows only; from #1210 all three
eager lists; from #1279 (2026-08-09) the eager xfail and permanent-skip lists but
not the eager slow-skips, `--jit` keeping its own measured slow-skip list. Hence
jax `--jit` runs ~170 orbit tests in August that eager jax still slow-skipped.
At the end the expected-failure and slow-test ledgers are empty, and 4311 of the
4315 tests pass in every mode; the other four test the out-of-scope torus mapper
and are permanently skipped. Bottom: tests in the dedicated backend modules (11097
in 92 files at the end), which check gradients, parity with numpy and traceability directly.
