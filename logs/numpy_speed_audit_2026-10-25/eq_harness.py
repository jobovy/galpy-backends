import sys, time, warnings, inspect, pickle, numpy, galpy
tree = sys.argv[1]; assert galpy.__file__.startswith(tree); warnings.simplefilter("ignore")
numpy.seterr(all="ignore")
from galpy import potential
from galpy.potential import Potential
only = sys.argv[3].split(",") if len(sys.argv) > 3 else None
names = sorted(n for n in dir(potential) if n.endswith("Potential") and inspect.isclass(getattr(potential, n)) and issubclass(getattr(potential, n), Potential))
if only: names = [n for n in names if n in only]
meths = ["_evaluate", "_Rforce", "_zforce", "_phitorque", "_R2deriv", "_z2deriv", "_Rzderiv", "_phi2deriv", "_Rphideriv", "_phizderiv", "_dens"]
pts = [(1.1, 0.2, 0.3, 0.5), (0.0, 0.0, 0.3, 0.5), (0.0, 0.3, 0.3, 0.5), (0.4, 0.0, 0.0, 0.0), (numpy.inf, 0.2, 0.3, 0.5), (5.0, -2.0, 2.0, 3.0), (0.01, 0.001, 0.1, -1.0)]
R = numpy.array([0.0, 0.01, 0.5, 1.1, 2.0, 5.0, numpy.inf]); z = numpy.array([0.0, 0.001, 0.0, 0.2, -0.3, -2.0, 0.2])
out = {}
def rec(v):
    try:
        return (type(v).__name__, getattr(v, "dtype", None).__str__(), numpy.asarray(v, dtype=float).tobytes(), numpy.shape(v))
    except Exception as e:
        return ("ERR", type(v).__name__, repr(v)[:80])
EXTRAS = {
    "X_MWP14": lambda: potential.MWPotential2014,
    "X_RaT": lambda: potential.RotateAndTiltWrapperPotential(pot=potential.MWPotential2014, zvec=[0.1, 0.2, 1.0], galaxy_pa=0.3, offset=[0.1, 0.0, 0.05]),
    "X_SmoothBar": lambda: potential.DehnenSmoothWrapperPotential(pot=potential.DehnenBarPotential(), tform=-5.0, tsteady=2.0),
    "X_ETNFW": lambda: potential.ExpTruncNFWPotential(amp=1.7, a=1.3, rc=8.0),
    "X_Einasto_small": lambda: potential.EinastoPotential(h=3.0, n=2.0),
    "X_PSPwC": lambda: potential.PowerSphericalPotentialwCutoff(alpha=1.8, rc=1.9),
}
if not only: names = names + list(EXTRAS)
else: names = names + [n for n in EXTRAS if n in only]
for n in names:
    try:
        t0 = time.perf_counter(); p = EXTRAS[n]() if n in EXTRAS else getattr(potential, n)()
        if time.perf_counter() - t0 > 20: continue
    except Exception:
        continue
    for m in meths:
        f = getattr(p, m, None)
        if f is None: continue
        for i, (r_, z_, ph, t) in enumerate(pts + [(R, z, 0.3, 0.5), (1.1, 0.2, 0.3, numpy.array([-1.0, 0.5, 50.0]))]):
            try:
                v = f(r_, z_, phi=ph, t=t)
                out[(n, m, i)] = rec(v)
            except Exception as e:
                out[(n, m, i)] = ("EXC", type(e).__name__, str(e)[:80])
pickle.dump(out, open(sys.argv[2], "wb"))
print(len(out), "records")
