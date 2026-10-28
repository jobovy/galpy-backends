import sys, warnings, pickle, numpy; warnings.filterwarnings("ignore")
import galpy; assert galpy.__file__.startswith(sys.argv[1])
import galpy.potential as gp, galpy.potential.EllipsoidalPotential as ep
out = {}
m = numpy.geomspace(0.01, 30.0, 25)
for name in ("TriaxialNFWPotential", "PerfectEllipsoidPotential", "TriaxialGaussianPotential", "PowerTriaxialPotential", "TriaxialHernquistPotential"):
    p = getattr(gp, name)()
    out[name + ".mdens"] = numpy.asarray(p._mdens(m)); out[name + ".mdens_deriv"] = numpy.asarray(p._mdens_deriv(m))
    out[name + ".glx"] = numpy.asarray(p._glx); out[name + ".glw"] = numpy.asarray(p._glw)
    out[name + ".b2c2"] = (p._b2, p._c2, type(p._b2).__name__)
    x, y, z = 1.3, 0.0, 0.4
    kw = dict(glx=p._glx, glw=p._glw)
    try:
        r = ep._2ndDerivInt_all(x, y, z, lambda mm: p._mdens(mm), lambda mm: p._mdens_deriv(mm), p._b2, p._c2, **kw)
    except TypeError:
        r = ep._2ndDerivInt_all(x, y, z, lambda mm: p._mdens(mm), lambda mm: p._mdens_deriv(mm), p._b2, p._c2, numpy, **kw)
    out[name + ".2nd"] = numpy.asarray(r)
    out[name + ".R2deriv"] = numpy.asarray(p.R2deriv(1.3, 0.4)); out[name + ".z2deriv"] = numpy.asarray(p.z2deriv(1.3, 0.4))
    out[name + ".R2deriv_arr"] = numpy.asarray(p.R2deriv(numpy.array([1.3, 2.0]), numpy.array([0.4, 0.1])))
pickle.dump(out, open(sys.argv[2], "wb")); print("ok")
