"""Sensitivity as GENERATIVE perturbed tracks: the particles xv(theta) FLOW
(re-sampled at each perturbed parameter, common random numbers), only the
discrete reconstruction STRUCTURE (closest-point assignment, GCV smoother) is
frozen. This is the physical d(track)/d(param) -- the whole stream re-forms."""
import sys, json, base64, io
WT = "/var/tmp/cint-param"; sys.path.insert(0, WT)
import numpy, galpy
assert galpy.__file__.startswith(WT)
from scipy import interpolate
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from galpy.orbit import Orbit
from galpy.potential import LogarithmicHaloPotential
from galpy.df import fardal15spraydf
from galpy.util import conversion, coords
from galpy.df.streamTrack import _closest_point_on_curve, _bin_by_tp, _DiffSpline
OUT = "/tmp/claude-21516/-home-bovy-Repos-galpy/e83d6165-0836-4adf-9f62-52251f942f9d/scratchpad/trackg"
AMP0, Q0 = 1.0, 0.9; IC = [1.2, 0.15, 0.85, 0.08, 0.05, 0.0]
RO, VO = 8.0, 220.0; SEED = 20260711
M = 3*10.0**4.0/conversion.mass_in_msol(VO, RO); TD = 4.5/conversion.time_in_Gyr(VO, RO)
T, HD = 2.0, 601
t_fwd = numpy.linspace(0.0, T, HD); t_back = numpy.linspace(0.0, -T, HD)
tg = numpy.concatenate([t_back[::-1], t_fwd[1:]])

def curve(amp, q, ic):
    pot = LogarithmicHaloPotential(amp=amp, q=q)
    of = Orbit(list(ic)); of.turn_physical_off(); of.integrate(t_fwd, pot, method="dop853_c")
    ob = Orbit(list(ic)); ob.turn_physical_off(); ob.integrate(t_back, pot, method="dop853_c")
    cart = lambda o, ts: numpy.column_stack([o.x(ts), o.y(ts), o.z(ts), o.vx(ts), o.vy(ts), o.vz(ts)])
    return numpy.concatenate([cart(ob, t_back)[::-1], cart(of, t_fwd)[1:]], axis=0)

def interp(cv, tpq):
    sp = [interpolate.InterpolatedUnivariateSpline(tg, cv[:, i], k=3) for i in range(6)]
    return numpy.column_stack([s(numpy.atleast_1d(tpq)) for s in sp])

def sample(amp, q, ic):  # GENERATIVE: re-sample the stream (common random numbers)
    sp = fardal15spraydf(M, progenitor=Orbit(list(ic)), pot=LogarithmicHaloPotential(amp=amp, q=q), tdisrupt=TD)
    numpy.random.seed(SEED)
    xv, _ = sp._sample_tail(600, True, leading=True)
    return coords.galcencyl_to_galcenrect(*numpy.asarray(xv, float))

# --- FROZEN structure from the base stream (assignment ta + GCV smoother) ---
base_cv = curve(AMP0, Q0, IC); pc0 = sample(AMP0, Q0, IC)
sign = numpy.broadcast_to((tg >= 0)[None, :], (pc0.shape[0], tg.size))
ta = _closest_point_on_curve(pc0, base_cv, tg, mask=sign, velocity_weight=1.0)
keep = numpy.abs(ta - tg[-1]) > 1e-3*abs(tg[-1]-tg[0]); ta = ta[keep]
tp_hi = float(numpy.percentile(ta, 99.0))
tp_grid = numpy.linspace(0.0, tp_hi, 200); tp_nodes = numpy.linspace(0.0, tp_hi, 24)
off0 = pc0[keep] - interp(base_cv, ta)
means0, cov0, cnt0 = _bin_by_tp(ta, off0, tp_nodes)
with numpy.errstate(invalid="ignore"):
    per0 = numpy.sqrt(numpy.clip(numpy.einsum("mii->mi", cov0), 0.0, None))
    sig = numpy.where(cnt0[:, None] > 1, per0/numpy.sqrt(numpy.maximum(cnt0[:, None], 1)), numpy.nan)
gm = [_DiffSpline(means0[:, i], tp_nodes, sig[:, i], None, 1.0, "numpy")._build(tp_grid)(means0[:, i]) for i in range(6)]

def track_gen(amp, q, ic):  # particles FLOW (re-sample), structure frozen
    cv = curve(amp, q, ic); pcp = sample(amp, q, ic)[keep]
    means, _, _ = _bin_by_tp(ta, pcp - interp(cv, ta), tp_nodes)
    smooth = numpy.column_stack([gm[i] @ numpy.nan_to_num(means[:, i]) for i in range(6)])
    return interp(cv, tp_grid) + smooth

def ic_pert(comp, d): x = list(IC); x[comp] += d; return x
base = track_gen(AMP0, Q0, IC)
D = {"amp": 0.30, "q": 0.11, "vR": 0.10, "vz": 0.10}
pert = {"amp": track_gen(AMP0+D["amp"], Q0, IC), "q": track_gen(AMP0, Q0+D["q"], IC),
        "vR": track_gen(AMP0, Q0, ic_pert(1, D["vR"])), "vz": track_gen(AMP0, Q0, ic_pert(4, D["vz"]))}
NUM = {}
for k in D:
    dmax = float(numpy.abs(pert[k][:, [0, 2]] - base[:, [0, 2]]).max())
    NUM[f"gen_deform_{k}"] = dmax
    print(f"  GENERATIVE {k}: max|deform| = {dmax:.4f}")
json.dump(NUM, open(f"{OUT}/gen_sens_numbers.json", "w"), indent=2)

# --- frozen-XV comparison (the OLD, non-physical deformation: particles NOT re-sampled) ---
pc0k = pc0[keep]
def track_frozenxv(amp, q, ic):
    cv = curve(amp, q, ic)
    means, _, _ = _bin_by_tp(ta, pc0k - interp(cv, ta), tp_nodes)  # FROZEN base particles
    smooth = numpy.column_stack([gm[i] @ numpy.nan_to_num(means[:, i]) for i in range(6)])
    return interp(cv, tp_grid) + smooth
fz = {"amp": track_frozenxv(AMP0+D["amp"], Q0, IC), "q": track_frozenxv(AMP0, Q0+D["q"], IC),
      "vR": track_frozenxv(AMP0, Q0, ic_pert(1, D["vR"])), "vz": track_frozenxv(AMP0, Q0, ic_pert(4, D["vz"]))}
base_fz = track_frozenxv(AMP0, Q0, IC)
print("=== deformation: generative (xv flows) vs frozen-xv (old) ===")
for k in D:
    gd = float(numpy.abs(pert[k][:, [0,2]] - base[:, [0,2]]).max())
    fd = float(numpy.abs(fz[k][:, [0,2]] - base_fz[:, [0,2]]).max())
    NUM[f"frozenxv_deform_{k}"] = fd
    print(f"  {k}: generative={gd:.4f}  frozen-xv={fd:.4f}  ratio={gd/max(fd,1e-9):.2f}x")
numpy.savez(f"{OUT}/sens_tracks.npz", tp_grid=tp_grid, base=base,
            **{f"pert_{k}": pert[k] for k in D})
json.dump(NUM, open(f"{OUT}/gen_sens_numbers.json", "w"), indent=2)
print("saved sens_tracks.npz + numbers")
