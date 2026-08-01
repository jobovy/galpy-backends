"""pytest plugin: count how many @backend_input boundary calls actually trace."""
import galpy.backend._jit as J
import galpy.backend._input as I

STATS = {"traced": 0, "not_traced": 0}
_orig = J.traced_call

def counting(method, args, kwargs, slots, nargs):
    res = _orig(method, args, kwargs, slots, nargs)
    if res is J.NOT_TRACED:
        STATS["not_traced"] += 1
    else:
        STATS["traced"] += 1
    return res

def pytest_configure(config):
    I.traced_call = counting          # _input.py imported the name directly

def pytest_sessionfinish(session, exitstatus):
    t, n = STATS["traced"], STATS["not_traced"]
    tot = t + n
    pct = (100.0 * t / tot) if tot else 0.0
    print(f"\n=== BOUNDARY CALLS: traced={t}  not_traced={n}  ({pct:.1f}% traced) ===")
