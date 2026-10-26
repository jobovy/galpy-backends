import sys, time, timeit, warnings, inspect, json, galpy, numpy
tree=sys.argv[1]; assert galpy.__file__.startswith(tree); warnings.simplefilter("ignore")
from galpy import potential
from galpy.potential.Potential import _evaluatePotentials, _evaluateRforces, _evaluatezforces
from galpy.potential import Potential
def T(f, budget):
    t0 = time.perf_counter(); f(); one = time.perf_counter() - t0
    n = max(1, min(2000, int(budget / max(one, 1e-7))))
    return min(timeit.repeat(f, number=n, repeat=3)) / n * 1e6
out = open(sys.argv[2], "w")
names = sorted(n for n in dir(potential) if n.endswith("Potential") and inspect.isclass(getattr(potential, n)) and issubclass(getattr(potential, n), Potential))
R = numpy.linspace(0.5, 2, 1000); z = numpy.linspace(-0.3, 0.3, 1000)
for n in names:
    try:
        t0 = time.perf_counter(); p = getattr(potential, n)(); tc = time.perf_counter() - t0
        if tc > 20 or p.dim != 3: continue
        p._Rforce(1.1, 0.2, phi=0.3, t=0.5)
    except Exception:
        continue
    row = {}
    for lab, fn in (("Phi", _evaluatePotentials), ("Rf", _evaluateRforces), ("zf", _evaluatezforces)):
        try:
            row[lab] = (T(lambda: fn(p, 1.1, 0.2, phi=0.3, t=0.5), 0.02), T(lambda: fn(p, R, z, phi=0.3, t=0.5), 0.1))
        except Exception:
            pass
    out.write(json.dumps([n, row]) + "\n"); out.flush()
