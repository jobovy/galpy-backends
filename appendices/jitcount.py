"""pytest plugin: measure REAL jit coverage, per test.

Two independent counters, because a `--jit` pass count proves neither:

* boundary calls  traced vs NOT_TRACED  -- did the trace actually happen?
* C-library entries                     -- did the test drop into libgalpy,
                                           which cannot be traced at all?

Usage:  PYTHONPATH=<dir-of-this-file> pytest ... --backend jax --jit -p jitcount

A test with c_entries > 0 is `not-traceable` by construction: galpy's answer for
C is the C-STM custom_vjp path, which buys differentiability via callbacks, not
tracing. Such tests must NOT be counted as traced-green.
"""

import importlib

import galpy.backend._jit as J
import galpy.backend._input as I  # imports traced_call by name -- patch HERE

# Every module that holds its own module-level reference to the ctypes CDLL.
_C_MODULES = (
    "galpy.orbit.integrateFullOrbit",
    "galpy.orbit.integrateLinearOrbit",
    "galpy.orbit.integratePlanarOrbit",
    "galpy.actionAngle.actionAngleAdiabatic_c",
    "galpy.actionAngle.actionAngleStaeckel_c",
    "galpy.potential.interpRZPotential",
)

STATS = {"traced": 0, "not_traced": 0, "c": 0}
PER_TEST = {}
_orig_traced_call = J.traced_call


def _counting_traced_call(method, args, kwargs, slots, nargs):
    res = _orig_traced_call(method, args, kwargs, slots, nargs)
    STATS["not_traced" if res is J.NOT_TRACED else "traced"] += 1
    return res


class _CountingLib:
    """Counts C-symbol *resolution* and returns the real ctypes function.

    Resolution, not invocation: call sites do ``fn = _lib.foo`` and then set
    ``fn.argtypes`` before calling, so wrapping the callable breaks ctypes.
    These sites resolve immediately before calling, so resolution is a faithful
    proxy and costs zero behaviour change.
    """

    def __init__(self, lib):
        object.__setattr__(self, "_lib", lib)

    def __getattr__(self, name):
        STATS["c"] += 1
        return getattr(object.__getattribute__(self, "_lib"), name)


def pytest_configure(config):
    I.traced_call = _counting_traced_call
    for name in _C_MODULES:
        try:
            mod = importlib.import_module(name)
        except ImportError:  # pragma: no cover
            continue
        if getattr(mod, "_lib", None) is not None:
            mod._lib = _CountingLib(mod._lib)


def pytest_runtest_logreport(report):
    if report.when != "call":
        return
    PER_TEST[report.nodeid] = (STATS["traced"], STATS["not_traced"], STATS["c"])


def pytest_sessionfinish(session, exitstatus):
    t, n, c = STATS["traced"], STATS["not_traced"], STATS["c"]
    tot = t + n
    pct = (100.0 * t / tot) if tot else 0.0
    print(f"\n=== BOUNDARY CALLS: traced={t} not_traced={n} ({pct:.1f}% traced) ===")
    print(f"=== C-LIBRARY ENTRIES: {c} ===")
    # per-test deltas -> the not-traceable list
    prev = (0, 0, 0)
    not_traceable = []
    for nodeid, cur in PER_TEST.items():
        if cur[2] > prev[2]:
            not_traceable.append(nodeid)
        prev = cur
    print(f"=== NOT-TRACEABLE (entered C): {len(not_traceable)} tests ===")
    for nodeid in not_traceable:
        print(f"    {nodeid}")
