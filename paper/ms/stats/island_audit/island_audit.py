"""pytest plugin: where does galpy's computation leave the backend under a forced
backend? Load with  -p island_audit  (this directory on PYTHONPATH).

Records, per test, every galpy call site (file:line:function of the first frame
in galpy/ outside galpy/backend/) at which
  * implicit: numpy converts a torch Tensor / jax Array (Tensor.__array__,
    ArrayImpl.__array__), i.e. numpy or scipy read backend data;
  * explicit: galpy.backend.as_numpy / to_host / host_eval was given backend data;
  * pin: galpy entered use("numpy", force=True).
Writes ISLAND_AUDIT_OUT/<pid>.json at session end. galpy is not imported here:
the patches go in after collection (the test modules import galpy)."""

import collections
import json
import os
import sys

_EVENTS = collections.defaultdict(lambda: collections.defaultdict(int))  # site -> {kind|test: n}
_TESTS = collections.defaultdict(set)  # site|kind -> tests
_CUR = {"test": None}
_ALL = {}
_GALPY = os.sep + "galpy" + os.sep
_BACKEND = os.sep + "galpy" + os.sep + "backend" + os.sep


def _site(depth=2):
    f = sys._getframe(depth)
    while f is not None:
        fn = f.f_code.co_filename
        if _GALPY in fn and _BACKEND not in fn and os.sep + "tests" + os.sep not in fn:
            i = fn.rindex(_GALPY) + 1
            return f"{fn[i:]}:{f.f_lineno}:{f.f_code.co_name}"
        f = f.f_back
    return None


def _record(kind, depth=3):
    if _CUR["test"] is None:
        return
    s = _site(depth)
    if s is None:
        return
    _EVENTS[s][kind] += 1
    _TESTS[(s, kind)].add(_CUR["test"])


def _install():
    if "galpy.backend" not in sys.modules:
        return False
    import galpy.backend as gb
    import galpy.backend._coerce as gc
    import galpy.backend._namespaces as gn
    import galpy.backend._resolver as gr

    is_bk = gn.is_backend_array

    def _has_bk(x):
        if isinstance(x, (list, tuple)):
            return any(_has_bk(e) for e in x)
        return is_bk(x)

    def wrap_explicit(name, orig):
        def w(*a, **k):
            if a and _has_bk(a[0]):
                _record("explicit:" + name)
            return orig(*a, **k)
        w.__wrapped__ = orig
        return w

    orig_use = gr.use

    def use(name=None, *a, **k):
        if name == "numpy" and (k.get("force") or (a and a[0] is True)):
            _record("pin")
        return orig_use(name, *a, **k)

    patches = {}
    for name, mod in (("as_numpy", gn), ("to_host", gn), ("host_eval", gc)):
        if hasattr(mod, name):
            patches[getattr(mod, name)] = wrap_explicit(name, getattr(mod, name))
    patches[orig_use] = use
    for m in list(sys.modules.values()):
        if m is None or not getattr(m, "__name__", "").startswith("galpy"):
            continue
        for attr, val in list(vars(m).items()):
            try:
                hit = val in patches
            except Exception:  # noqa: BLE001 -- unhashable/odd attributes
                continue
            if hit:
                setattr(m, attr, patches[val])
    try:
        import torch

        orig_t = torch.Tensor.__array__

        def t_array(self, *a, **k):
            _record("implicit:torch")
            return orig_t(self, *a, **k)

        torch.Tensor.__array__ = t_array
    except ImportError:
        pass
    try:
        from jax._src.array import ArrayImpl

        orig_j = ArrayImpl.__array__

        def j_array(self, *a, **k):
            _record("implicit:jax")
            return orig_j(self, *a, **k)

        ArrayImpl.__array__ = j_array
    except ImportError:
        pass
    return True


_INSTALLED = {"done": False}


def pytest_collection_finish(session):
    _INSTALLED["done"] = _install()


def pytest_runtest_setup(item):
    if not _INSTALLED["done"]:
        _INSTALLED["done"] = _install()
    _CUR["test"] = item.nodeid


def pytest_runtest_teardown(item):
    _CUR["test"] = None


def pytest_runtest_logreport(report):
    if report.when == "call" or (report.when == "setup" and report.outcome != "passed"):
        _ALL[report.nodeid] = report.outcome


def pytest_sessionfinish(session):
    out = os.environ.get("ISLAND_AUDIT_OUT")
    if not out:
        return
    os.makedirs(out, exist_ok=True)
    rec = {
        "events": {s: dict(v) for s, v in _EVENTS.items()},
        "tests": {f"{s}|{k}": sorted(t) for (s, k), t in _TESTS.items()},
        "all": _ALL,
    }
    with open(os.path.join(out, f"{os.getpid()}.json"), "w") as fh:
        json.dump(rec, fh)
