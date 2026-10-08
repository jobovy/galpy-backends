"""One-time: dump galpy's C parse of MWPotential2014 + GSL constants to JSON.

Needs galpy (with its C extension) and the libgsl galpy links against:
    PYTHONPATH=/var/tmp/fbp python make_params.py
"""

import ctypes
import json
import os

from galpy.orbit.integrateFullOrbit import _parse_pot
from galpy.potential import MWPotential2014

GSL = os.environ.get("GSL_LIB", "/home/bovy/miniforge3/envs/py313/lib/libgsl.so.28")
gsl = ctypes.CDLL(GSL)
gsl.gsl_sf_gamma.restype = ctypes.c_double
gsl.gsl_sf_gamma.argtypes = [ctypes.c_double]
gsl.gsl_sf_lngamma.restype = ctypes.c_double
gsl.gsl_sf_lngamma.argtypes = [ctypes.c_double]

npot, pot_type, pot_args, _ = _parse_pot(MWPotential2014)[:4]
pot_type = [int(x) for x in pot_type]
pot_args = [float(x) for x in pot_args]
gam, lg1 = {}, {}
k = 0
for ty in pot_type:
    if ty == 15:
        a = 1.5 - 0.5 * pot_args[k + 1]
        gam[a.hex()] = gsl.gsl_sf_gamma(a).hex()
        lg1[a.hex()] = gsl.gsl_sf_lngamma(a + 1.0).hex()
        k += 5
    else:
        k += {5: 3, 9: 2}[ty]
out = {
    "potential": "MWPotential2014",
    "pot_type": pot_type,
    "pot_args": pot_args,
    "pot_args_hex": [x.hex() for x in pot_args],
    "gamma_a_hex": gam,
    "lngamma_a1_hex": lg1,
}
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mw2014_params.json"), "w") as fh:
    json.dump(out, fh, indent=1)
print(out)
