"""GS3 -- one AD pass predicts the smooth stream track for LARGE halo-shape changes.

The expensive part of a spray stream model is integrating every stripped particle.
This asks whether a single forward-mode AD pass at one halo flattening can predict
the resulting SMOOTH TRACK at a genuinely different flattening:

    particles_AD(q0+dq) = particles(q0) + dq * d(particles)/dq     [ONE jvp pass]
    track_AD           = streamTrack(particles=particles_AD)
    track_direct       = streamTrack(particles=sample(q0+dq))      [full re-run]

Both sides go through the SAME track machinery (closest-point projection onto the
progenitor orbit at that q, GCV-smoothed spline fit), so the only difference is
AD-predicted versus directly-integrated particles -- the comparison isolates the
gradient, not the fitter. The progenitor orbit is one cheap integration and is
computed exactly on both sides; what AD predicts is the expensive N-particle spray.

Common random numbers throughout: the same key at every q, so a difference is the
pathwise response to the halo, not sampling noise.

WHY FORWARD MODE: reverse mode would need one pass per output, i.e. per particle.
Forward mode gives the whole (6, N) tangent in ONE pass, because q is a single
scalar. It requires an adjoint that differentiates the solver's own operations
(`adjoint="direct"`); the default checkpointed adjoint is a custom_vjp and is
reverse-only. That option reaches the sampler through `integrate_kwargs`.

VALIDATION of the tangent (N=120, against a common-random-number central
difference of the same sampler): max|FD-AD|/max|AD| = 5.9e-02, 6.0e-04, 1.6e-05
at h = 1e-2, 1e-3, 1e-4 -- second-order convergence onto the AD value.

Reproducibility: the spray and the track fits dominate, so measure->json and the
plot is byte-stable from it. Delete the JSON to re-measure.
"""

import json
import os

import numpy

from paper.figures._harness.figure import figure

_DATA = os.path.join(os.path.dirname(__file__), "data")
_JSON = os.path.join(_DATA, "gS3_track_extrapolation.json")
# journal version (default): no env stamp / long titles or notes; the caption
# carries the explanations. GALPY_JAX_PAPER=0 restores the annotated figure.
PAPER = os.environ.get("GALPY_JAX_PAPER", "1") == "1"

# The orbit must actually PROBE the flattening: q flattens the halo in z, so a
# near-planar orbit is insensitive to it by construction. This is galpy's standard
# inclined stream progenitor (z=0.887, vz=-0.477), disrupting for 4.5 Gyr so the
# halo shape has time to imprint on the track.
_IC = [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596]
_Q0 = 0.9
_N = 1500
_SEED = 20260707
_TD_GYR = 4.5
_DQS = (-0.2, -0.1, 0.1, 0.2)          # shown as tracks: q = 0.7 .. 1.1
# a slope fitted to two points is not a measurement of scaling -- sweep a decade
_DQS_ERR = (0.025, 0.05, 0.1, 0.2)
_IKW = {"adjoint": "direct", "max_steps": 100000}


def _measure():
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from galpy.backend import random as grandom
    from galpy.df import fardal15spraydf
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential
    from galpy.util import conversion

    td = _TD_GYR / conversion.time_in_Gyr(220.0, 8.0)
    # ONE stream frame for every fit (phi1 along the progenitor's orbital plane,
    # phi2 across it), from the q-independent initial condition: differences
    # between tracks are then differences in the same coordinates
    T = numpy.asarray(Orbit(_IC).align_to_orbit(), dtype=float)
    mass = 2.0e4 / conversion.mass_in_msol(220.0, 8.0)
    key = grandom.key(_SEED, backend="jax")

    def make_df(q, ikw=_IKW):
        lp = LogarithmicHaloPotential(normalize=1.0, q=q)
        prog = Orbit(jnp.stack([jnp.asarray(v) for v in _IC]))
        prog.turn_physical_off()
        return fardal15spraydf(progenitor=prog, pot=lp, tdisrupt=td,
                               integrate_kwargs=ikw, progenitor_mass=mass)

    def particles(q):
        return make_df(q).sample(n=_N, integrate=True, key=key, return_orbit=False)

    # ONE forward pass -> present-day particles at q0 and their q-tangents
    p0, dp = jax.jvp(particles, (jnp.asarray(_Q0),), (jnp.asarray(1.0),))
    p0 = numpy.asarray(p0, dtype=float)
    dp = numpy.asarray(dp, dtype=float)

    def fit(q, xv, smoothing=None):
        """Both sides use the SAME machinery: the df at q (hence that q's
        progenitor curve) fitting the supplied particles.

        `smoothing` is pinned to the base fit's GCV-selected values for every
        subsequent fit. Letting GCV re-tune per fit makes the spline itself a
        source of scatter, which then swamps the quantity being measured -- the
        first version of this figure had the fitter's noise comparable to the
        track shift being predicted.
        """
        tr = make_df(q, ikw=None).streamTrack(particles=jnp.asarray(xv),
                                              smoothing=smoothing,
                                              custom_sky_transform=T)
        tp = numpy.asarray(tr.tp_grid(), dtype=float)
        return ({"tp": tp.tolist(),
                 "R": numpy.asarray(tr.R(tp), dtype=float).tolist(),
                 "z": numpy.asarray(tr.z(tp), dtype=float).tolist(),
                 "phi1": numpy.asarray(tr.phi1(tp), dtype=float).tolist(),
                 "phi2": numpy.asarray(tr.phi2(tp), dtype=float).tolist()},
                tr.smoothing_s)

    def dphi2(t1, t2):
        """max |phi2_1 - phi2_2| [deg] at matched phi1, over the common range."""
        def srt(t):
            o = numpy.argsort(t["phi1"])
            return numpy.asarray(t["phi1"])[o], numpy.asarray(t["phi2"])[o]
        x1, y1 = srt(t1)
        x2, y2 = srt(t2)
        g = numpy.linspace(max(x1[0], x2[0]), min(x1[-1], x2[-1]), 400)
        return float(numpy.max(numpy.abs(numpy.interp(g, x1, y1) - numpy.interp(g, x2, y2))))

    base, smooth = fit(_Q0, p0)          # GCV once; every other fit reuses it
    out = {"q0": _Q0, "n": _N, "dqs": list(_DQS), "dqs_err": list(_DQS_ERR),
           "seed": _SEED,
           "base": base, "ad": {}, "direct": {}, "err": {}}
    b = out["base"]
    scale = float(numpy.hypot(numpy.ptp(b["R"]), numpy.ptp(b["z"])))
    out["scale"] = scale
    for dq in sorted(set(_DQS) | set(_DQS_ERR) | {-d for d in _DQS_ERR}):
        q = _Q0 + dq
        a, _ = fit(q, p0 + dq * dp, smoothing=smooth)              # AD-predicted
        d, _ = fit(q, numpy.asarray(particles(jnp.asarray(q)), dtype=float),
                   smoothing=smooth)                                # direct
        out["ad"][f"{dq:+.3f}"] = a
        out["direct"][f"{dq:+.3f}"] = d
        # compare on the shorter common tp range, resampled to a common grid
        lo = max(min(a["tp"]), min(d["tp"]))
        hi = min(max(a["tp"]), max(d["tp"]))
        g = numpy.linspace(lo, hi, 400)
        dR = numpy.interp(g, a["tp"], a["R"]) - numpy.interp(g, d["tp"], d["R"])
        dz = numpy.interp(g, a["tp"], a["z"]) - numpy.interp(g, d["tp"], d["z"])
        out["err"][f"{dq:+.3f}"] = float(numpy.max(numpy.hypot(dR, dz)) / scale)
        # how far the track moved at all -- an error only means something next to it
        mR = numpy.interp(g, d["tp"], d["R"]) - numpy.interp(g, b["tp"], b["R"])
        mz = numpy.interp(g, d["tp"], d["z"]) - numpy.interp(g, b["tp"], b["z"])
        out.setdefault("moved", {})[f"{dq:+.3f}"] = float(
            numpy.max(numpy.hypot(mR, mz)) / scale
        )
        out.setdefault("err_phi2", {})[f"{dq:+.3f}"] = dphi2(a, d)
        out.setdefault("moved_phi2", {})[f"{dq:+.3f}"] = dphi2(d, b)
    return out


_CB = "#111827"
_CD = {"-0.200": "#1d4ed8", "-0.100": "#0891b2", "+0.100": "#ea580c",
       "+0.200": "#b91c1c"}


@figure(
    name="gS3_track_extrapolation",
    track="G",
    seed=_SEED,
    description="one forward-mode AD pass predicts the streamspraydf smooth track for LARGE halo-flattening changes (q=0.7-1.1), against a direct re-run (CPU float64) [measure->json, plot byte-stable]",
)
def make():
    import matplotlib.pyplot as plt

    if os.path.exists(_JSON):
        data = json.load(open(_JSON))
    else:
        data = _measure()
        os.makedirs(_DATA, exist_ok=True)
        with open(_JSON, "w") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
            fh.write("\n")

    # PAPER: final printed width of a figure* (7.0 in); text sizes are then the
    # printed sizes (9-10 pt labels/ticks, 10 pt titles, 8 pt legends/notes).
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 3.3) if PAPER else (9.8, 4.0))
    _ts = {"fontsize": 10} if PAPER else {}
    _fs_leg = 8 if PAPER else 7
    b = data["base"]
    ax1.plot(b["phi1"], b["phi2"], "-", color=_CB, lw=2.4, label=rf"$q_0={data['q0']}$")
    for k in [f"{d:+.3f}" for d in data["dqs"]]:
        d, a = data["direct"][k], data["ad"][k]
        c = _CD[k]
        ax1.plot(d["phi1"], d["phi2"], "-", color=c, lw=1.6, alpha=0.95,
                 label=rf"$q={data['q0'] + float(k):.1f}$")
        ax1.plot(a["phi1"], a["phi2"], "--", color=c, lw=1.5, dashes=(4, 2))
    ax1.set_xlabel(r"$\phi_1$ [deg]" if PAPER
                   else r"$\phi_1$ [deg] (along the progenitor's orbital plane)")
    ax1.set_ylabel(r"$\phi_2$ [deg]")
    ax1.set_title("(a) direct (solid) vs AD (dashed)" if PAPER
                  else rf"solid: direct re-run   dashed: one AD pass ($N={data['n']}$)", **_ts)
    ax1.legend(fontsize=_fs_leg, framealpha=0.92, ncol=2)

    dqs = numpy.array(sorted(data.get("dqs_err", [abs(float(k)) for k in data["err"]])))
    def _rms_dphi2(t1, t2):
        """rms |phi2_1 - phi2_2| [deg] at matched phi1 over the common range:
        the max is dominated by the sparse far tail ends, where any two spline
        fits scatter."""
        def srt(t):
            o = numpy.argsort(t["phi1"])
            return numpy.asarray(t["phi1"])[o], numpy.asarray(t["phi2"])[o]
        x1, y1 = srt(t1)
        x2, y2 = srt(t2)
        g = numpy.linspace(max(x1[0], x2[0]), min(x1[-1], x2[-1]), 400)
        r = numpy.interp(g, x1, y1) - numpy.interp(g, x2, y2)
        return float(numpy.sqrt(numpy.mean(r**2)))

    def _worst(d, fn):  # worse of +/- dq
        return max(fn(f"{+d:+.3f}"), fn(f"{-d:+.3f}"))

    errs = numpy.array([_worst(d, lambda k: _rms_dphi2(data["ad"][k], data["direct"][k])) for d in dqs])
    moved = numpy.array([_worst(d, lambda k: _rms_dphi2(data["direct"][k], data["base"])) for d in dqs])
    # the error only means something next to the shift it is predicting
    ax2.loglog(dqs, moved, "s-", color="#111827", ms=5, lw=1.4,
               label="track shift being predicted")
    ax2.loglog(dqs, errs, "o-", color="#1d4ed8", ms=6, lw=1.4,
               label="residual after one AD pass")
    for x, m, e in zip(dqs, moved, errs):
        ax2.annotate(f"{100 * (1 - e / m):.0f}%", (x, e), textcoords="offset points",
                     xytext=(0, -13), ha="center", fontsize=_fs_leg, color="#1d4ed8")
    ax2.set_xlabel(r"$|\Delta q|$")
    ax2.set_ylabel(r"rms $\Delta\phi_2$ at fixed $\phi_1$ [deg]")
    frac = 100.0 * (1.0 - errs / moved)
    ax2.set_title("(b) shift and residual" if PAPER
                  else f"one AD pass removes {frac.min():.0f}-{frac.max():.0f}% of the shift", **_ts)
    ax2.legend(fontsize=_fs_leg, framealpha=0.92, loc="upper left")
    ax2.grid(True, which="both", alpha=0.25)
    ax2.set_ylim(bottom=0.6 * errs.min())  # room for the lowest "%" callout
    if PAPER:  # plain ticks at the measured |dq| (the auto log labels overlap)
        ax2.set_xticks(dqs)
        ax2.set_xticklabels([f"{d:g}" for d in dqs])
        ax2.xaxis.set_minor_formatter(plt.NullFormatter())
        ax2.set_xlim(dqs.min() / 1.3, dqs.max() * 1.3)

    fig.tight_layout()
    return fig
