#!/usr/bin/env python3
"""PREPARED, HELD until gh#1364 (as_backend_constant default-device guard) merges.

Un-defers the noninertial entries that clear the ledger's 240 s CI margin ONCE
gh#1364's guard is in the base. Both halves were measured as CONTROLLED A/Bs
(same file, same command, same box; the two trees differ only in
_coerce.py + _namespaces.py):

    jax eager   BASE 5146 s -> FIXED  ... , 3 of 8 entries clear the margin
    jax --jit   BASE 5894 s -> FIXED 4543 s, 5 of 9 entries clear the margin

DO NOT APPLY BEFORE gh#1364 IS IN THE BASE -- without the guard these run at
their BASE times and would flake or time out.

Matches on the ENTRY FIELD (backend + nodeid), ignoring trailing "# 616s"
comments, and requires an EXACT nodeid match -- a prefix match on
"::test_arbitraryaxisrotation" would also swallow _omegafunc / _omegadot /
_omegafunc_nullpotential. Verifies the delta before writing.
"""
import pathlib
import sys

LEDGER = pathlib.Path("tests/backend_slow_skip.txt")

# (backend, nodeid) -> measured FIXED seconds, CI estimate at the 0.71 factor
UNDEFER = {
    # jax eager: 8 -> 5
    ("jax", "tests/test_noninertial.py::test_python_vs_c_linacc_changingacc_xyz_accellsrframe_vecomegaz"): (301.0, 213.7),
    ("jax", "tests/test_noninertial.py::test_arbitraryaxisrotation_omegafunc_nullpotential"): (280.4, 199.1),
    ("jax", "tests/test_noninertial.py::test_python_vs_c_arbitraryaxisrotation_funcomega"): (248.8, 176.7),
    # jax --jit: 9 -> 4
    ("jax-jit", "tests/test_noninertial.py::test_python_vs_c_linacc_changingacc_xyz_accellsrframe_vecomegaz"): (297.5, 211.2),
    ("jax-jit", "tests/test_noninertial.py::test_linacc_changingacc_xyz_accellsrframe_scalarfuncomegaz"): (307.1, 218.0),
    ("jax-jit", "tests/test_noninertial.py::test_arbitraryaxisrotation_omegafunc_nullpotential"): (247.2, 175.5),
    ("jax-jit", "tests/test_noninertial.py::test_python_vs_c_arbitraryaxisrotation_funcomega"): (243.3, 172.8),
    ("jax-jit", "tests/test_noninertial.py::test_python_vs_c_linacc_changingacc_xyz_accellsrframe_scalarfuncomegaz"): (235.3, 167.0),
}

ANNOTATION = """#
# RE-MEASURED 2026-08-22 as CONTROLLED A/Bs, after the as_backend_constant
# default-device guard (gh#1364). Whole file, serial, C extension active; the
# BASE arm is develop without the guard, everything else identical. BASE
# reproduces this ledger's own historical numbers to 0.85-1.02, so the arms are
# comparable. CI estimated at the measured CI/local factor 0.71; entries are
# kept unless they clear the 240 s CI margin.
#
#   jax --jit          BASE   FIXED     x     CI~
#     test_arbitraryaxisrotation          525.6  490.7  1.07  348.4   stays
#     ..._omegafunc                       523.9  400.5  1.31  284.4   stays
#     ..._accellsrframe_funcomegaz        502.6  351.9  1.43  249.9   stays
#     ..._omegadot                        387.3  352.3  1.10  250.1   stays
#     test_python_vs_c_..._vecomegaz      486.7  297.5  1.64  211.2   un-deferred
#     ..._accellsrframe_scalarfuncomegaz  399.1  307.1  1.30  218.0   un-deferred
#     ..._omegafunc_nullpotential         372.8  247.2  1.51  175.5   un-deferred
#     test_python_vs_c_...funcomega       383.1  243.3  1.57  172.8   un-deferred
#     test_python_vs_c_..._scalarfunc     340.1  235.3  1.45  167.0   un-deferred
#
# The speedup is HETEROGENEOUS (1.07x-1.64x): the two slowest tests barely move,
# so they stay deferred on their own merits, not on an averaged figure.
# jax eager (same treatment): 8 -> 5, un-deferring the three at CI~213.7 /
# 199.1 / 176.7 s.
"""


def entry_of(line):
    """(backend, nodeid) for a ledger line, or None for blank/comment lines."""
    bare = line.split("#", 1)[0].strip()
    if not bare:
        return None
    parts = bare.split(None, 1)
    return (parts[0], parts[1].strip()) if len(parts) == 2 else None


def main():
    if not LEDGER.exists():
        sys.exit(f"run from the repo root; {LEDGER} not found")
    lines = LEDGER.read_text().splitlines(keepends=True)
    seen = [entry_of(ln) for ln in lines]
    for key in UNDEFER:
        n = seen.count(key)
        if n != 1:
            sys.exit(f"expected exactly 1 ledger line for {key}, found {n}")
    before = sum(1 for e in seen if e)
    kept, anchored = [], False
    for ln, e in zip(lines, seen):
        if e in UNDEFER:
            continue
        kept.append(ln)
        if not anchored and e == (
            "jax-jit",
            "tests/test_noninertial.py::test_arbitraryaxisrotation",
        ):
            kept.append(ANNOTATION)
            anchored = True
    if not anchored:
        sys.exit("annotation anchor not found; re-derive by hand")
    LEDGER.write_text("".join(kept))
    after = sum(1 for e in (entry_of(l) for l in kept) if e)
    print(f"slow_skip entries {before} -> {after}  (expected -{len(UNDEFER)})")
    if before - after != len(UNDEFER):
        sys.exit("delta wrong; inspect before committing")


if __name__ == "__main__":
    main()
