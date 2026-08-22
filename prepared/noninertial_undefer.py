#!/usr/bin/env python3
"""PREPARED, HELD until gh#1364 (as_backend_constant device guard) merges.

Un-defers the 3 jax-eager noninertial entries that clear the ledger's 240 s CI
margin ONCE gh#1364's guard is in the base. Measured 2026-08-22, whole file,
serial, C extension active, quiet box:  53 passed in 5145.98 s.

DO NOT APPLY BEFORE gh#1364 IS IN THE BASE -- without the guard these run
~415/370 s local (CI ~294/263) and would flake or time out.

Content-addressed (matches on entry text, not line numbers) so it survives a
moved base. Run from the repo root; verifies before and after.
"""
import sys, pathlib

LEDGER = pathlib.Path("tests/backend_slow_skip.txt")

UNDEFER = [
    "jax tests/test_noninertial.py::test_python_vs_c_linacc_changingacc_xyz_accellsrframe_vecomegaz",
    "jax tests/test_noninertial.py::test_arbitraryaxisrotation_omegafunc_nullpotential",
    "jax tests/test_noninertial.py::test_python_vs_c_arbitraryaxisrotation_funcomega",
]

ANNOTATION = """#
# RE-MEASURED 2026-08-22, AFTER the as_backend_constant default-device guard
# (gh#1364): whole file, --backend jax, serial, C extension active -- 53 passed
# in 5145.98 s (the baseline recorded above for the same file was 6308 s). All 8
# jax entries PASS; none was stale. CI estimated at the measured CI/local 0.71:
#
#     CI~397.4  local 559.8  test_arbitraryaxisrotation                 (stays)
#     CI~325.9  local 459.0  test_arbitraryaxisrotation_omegafunc       (stays)
#     CI~320.4  local 451.2  ..._accellsrframe_funcomegaz               (stays)
#     CI~288.3  local 406.0  test_arbitraryaxisrotation_omegadot        (stays)
#     CI~284.4  local 400.6  ..._accellsrframe_scalarfuncomegaz         (stays)
#     CI~213.7  local 301.0  test_python_vs_c_..._vecomegaz             UN-DEFERRED
#     CI~199.1  local 280.4  ..._omegafunc_nullpotential                UN-DEFERRED
#     CI~176.7  local 248.8  test_python_vs_c_arbitraryaxisrotation_funcomega  UN-DEFERRED
#
# The three removed clear the 240 s CI margin; the two at CI~284-288 do not and
# stay deferred even though they are nominally under 300 s. Cross-check that the
# numbers are not a context artifact: ..._omegafunc_nullpotential measured
# 273.58 s ALONE and 280.4 s inside the full 53-test file (2.5% apart), against a
# pre-guard 414.5 s. The un-ledgered sibling omegadot_nullpotential moved only
# 159.1 -> 146.8 s (7.7%), i.e. the win scales with time spent in the force path,
# which is what the guard predicts.
#
# The 9 jax-jit noninertial entries are deliberately NOT touched here: under
# trace as_backend_constant runs once per trace, so the guard buys far less.
"""


def main():
    if not LEDGER.exists():
        sys.exit(f"run from the repo root; {LEDGER} not found")
    text = LEDGER.read_text()
    for entry in UNDEFER:
        if text.count(entry + "\n") != 1:
            sys.exit(f"expected exactly one bare occurrence of:\n  {entry}")
    before = len([l for l in text.splitlines() if l.strip() and not l.startswith("#")])
    for entry in UNDEFER:
        text = text.replace(entry + "\n", "", 1)
    marker = "jax tests/test_noninertial.py::test_linacc_changingacc_xyz_accellsrframe_funcomegaz\n"
    if marker not in text:
        sys.exit("anchor entry for the annotation is gone; re-derive by hand")
    text = text.replace(marker, marker + ANNOTATION, 1)
    LEDGER.write_text(text)
    after = len([l for l in text.splitlines() if l.strip() and not l.startswith("#")])
    print(f"slow_skip entries {before} -> {after}  (expected -3)")
    if before - after != 3:
        sys.exit("delta is not -3; inspect before committing")


if __name__ == "__main__":
    main()
