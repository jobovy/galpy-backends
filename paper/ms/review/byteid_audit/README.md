# numpy byte-identity audit (review round 1, 2026-10-28)

`compare_pots.py <worktree> <out.pkl>` dumps every parameter-free potential's
`__call__`, forces, density, second derivatives and enclosed mass on a 40x9
(R, z) grid; `compare_pots2.py` the same for a set of non-default parameters;
`triax_dump.py` the ellipsoidal density helpers and raw second-derivative
integrals. Run each against a `main` worktree and a `feat/backends` worktree
(`PYTHONPATH=<worktree>`; the script asserts `galpy.__file__`), then compare
the pickles entry by entry with `numpy.array_equal(..., equal_nan=True)`.

Result (main 9c129abcb vs feat/backends db33e5fcf): 654 of 691 comparable
entries identical; the King model deliberately changed (ODE tolerance); NFW,
Burkert and the eight ellipsoidal potentials' second derivatives off by a
few ulp from operation-order drift, restored by galpy PR #1768; Einasto below
its scale radius and PowerSphericalPotentialwCutoff off by <= 5e-15 (2e-14 in
one second derivative) because their force formulas changed (documented in
the paper's Sec. 2.1 instead).
