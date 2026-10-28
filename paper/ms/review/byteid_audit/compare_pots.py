"""Values of every parameter-free potential's methods on a grid -> pickle (one worktree per run)."""
import sys, warnings, pickle, numpy
warnings.filterwarnings("ignore")
import galpy; assert galpy.__file__.startswith(sys.argv[1]), galpy.__file__
import galpy.potential as gp
R = numpy.geomspace(0.03, 30.0, 40); z = numpy.linspace(-3.0, 3.0, 9)
RR, ZZ = numpy.meshgrid(R, z, indexing="ij")
methods = ["__call__", "Rforce", "zforce", "phitorque", "dens", "R2deriv", "z2deriv", "Rzderiv", "phi2deriv", "Rphideriv", "zphideriv", "mass"]
out = {}
names = [n for n in dir(gp) if n.endswith("Potential") and n not in ("Potential", "planarPotential", "linearPotential", "RZPotential")]
for n in sorted(names):
    cls = getattr(gp, n)
    try:
        p = cls()
    except Exception:
        continue
    for m in methods:
        try:
            if m == "mass":
                v = numpy.asarray(p.mass(R))
            elif m == "__call__":
                v = numpy.asarray(p(RR, ZZ, phi=0.3))
            else:
                v = numpy.asarray(getattr(p, m)(RR, ZZ, phi=0.3))
            out[f"{n}.{m}"] = v
        except Exception as e:
            out[f"{n}.{m}"] = f"ERR {type(e).__name__}"
pickle.dump(out, open(sys.argv[2], "wb")); print("wrote", len(out), "entries")
