"""Second parameter set (non-default shapes/amplitudes) for the byte-identity audit."""
import sys, warnings, pickle, numpy
warnings.filterwarnings("ignore")
import galpy; assert galpy.__file__.startswith(sys.argv[1]), galpy.__file__
import galpy.potential as gp
R = numpy.geomspace(0.01, 50.0, 45); z = numpy.linspace(-4.0, 4.0, 11)
RR, ZZ = numpy.meshgrid(R, z, indexing="ij")
methods = ["__call__", "Rforce", "zforce", "phitorque", "dens", "R2deriv", "z2deriv", "Rzderiv", "phi2deriv", "Rphideriv", "mass"]
pots = {
 "NFW": lambda: gp.NFWPotential(amp=1.7, a=3.1), "NFW_small_a": lambda: gp.NFWPotential(amp=0.3, a=0.2),
 "Burkert": lambda: gp.BurkertPotential(amp=2.2, a=0.7), "Burkert_big_a": lambda: gp.BurkertPotential(amp=0.9, a=5.0),
 "Einasto": lambda: gp.EinastoPotential(amp=1.1, h=2.0, n=1.5), "Einasto_n4": lambda: gp.EinastoPotential(amp=1.1, h=0.8, n=4.0),
 "PSPwC": lambda: gp.PowerSphericalPotentialwCutoff(amp=1.3, alpha=1.8, rc=1.9), "PSPwC_a0.5": lambda: gp.PowerSphericalPotentialwCutoff(amp=0.7, alpha=0.5, rc=0.6),
 "TriaxialNFW": lambda: gp.TriaxialNFWPotential(amp=1.2, a=1.8, b=0.8, c=0.6), "TriaxialHernquist": lambda: gp.TriaxialHernquistPotential(amp=1.2, a=1.8, b=0.7, c=0.5),
 "TriaxialJaffe": lambda: gp.TriaxialJaffePotential(amp=1.2, a=1.8, b=0.9, c=0.4), "TwoPowerTriaxial": lambda: gp.TwoPowerTriaxialPotential(amp=1.2, a=1.8, alpha=1.3, beta=3.5, b=0.8, c=0.6),
 "PerfectEllipsoid": lambda: gp.PerfectEllipsoidPotential(amp=1.2, a=1.8, b=0.8, c=0.6), "TriaxialGaussian": lambda: gp.TriaxialGaussianPotential(amp=1.2, sigma=1.8, b=0.8, c=0.6),
 "PowerTriaxial": lambda: gp.PowerTriaxialPotential(amp=1.2, alpha=1.3, b=0.8, c=0.6), "Hernquist": lambda: gp.HernquistPotential(amp=1.5, a=2.3),
 "Jaffe": lambda: gp.JaffePotential(amp=1.5, a=2.3), "TwoPower": lambda: gp.TwoPowerSphericalPotential(amp=1.5, a=2.3, alpha=1.2, beta=3.3),
 "ExpTruncNFW": lambda: gp.ExpTruncNFWPotential(amp=1.5, a=2.3, rc=6.0) if hasattr(gp, "ExpTruncNFWPotential") else None,
}
out = {}
for n, mk in sorted(pots.items()):
    try:
        p = mk()
    except Exception as e:
        continue
    if p is None: continue
    for m in methods:
        try:
            v = numpy.asarray(p.mass(R)) if m == "mass" else numpy.asarray(p(RR, ZZ, phi=0.7) if m == "__call__" else getattr(p, m)(RR, ZZ, phi=0.7))
            out[f"{n}.{m}"] = v
        except Exception as e:
            out[f"{n}.{m}"] = f"ERR {type(e).__name__}"
pickle.dump(out, open(sys.argv[2], "wb")); print("wrote", len(out), "entries")
