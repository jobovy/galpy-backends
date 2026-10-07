"""GS3 -- one AD pass predicts the smooth stream track for LARGE halo-shape changes.

The expensive part of a spray stream model is integrating every stripped particle.
This asks whether a single forward-mode AD pass at one halo flattening q0 predicts
the stream's SMOOTH TRACK at a genuinely different flattening:

    particles_AD(q0+dq) = particles(q0) + dq * d(particles)/dq     [ONE jvp pass]
    particles_direct    = sample(q0+dq)                              [full re-run]

Both tails (leading + trailing, N/2 each), the same random key at every q (common
random numbers: a difference is the pathwise response to the halo, not sampling
noise), and ONE stream frame (phi1 along the progenitor's orbital plane, phi2
across it) from the q-independent initial condition (Orbit.align_to_orbit).

(a) galpy's streamTrack (default GCV smoothing) fitted to the direct and to the
    AD-predicted particles, at q = 0.7 .. 1.1.
(b) the like-for-like accuracy test. A first-order prediction must leave a
    residual O(dq^2), so the fraction of the track shift it removes must tend to
    100% as dq -> 0. That holds only if the track is a SMOOTH function of the
    particles with everything else held fixed. Here: the same key, the same frame,
    a fixed phi1 grid (the inner 5-95% of each tail at q0) and a fixed linear
    smoother (Gaussian-kernel mean phi2 at each grid phi1, fixed bandwidth). The
    residual then falls as dq^2 down to dq = 0.003.

WHY THE FIRST VERSION FLOORED AT ~85%: it compared two INDEPENDENT streamTrack
re-fits. streamTrack is, by design, a discontinuous function of the particles:
hard tp bins, a closest-point argmin on a discrete progenitor grid, a percentile
trim of the tp range, data-driven time range / velocity weight, and (with a
pinned s) FITPACK's adaptive knots. The AD and direct particles differ by O(dq^2)
each, but every particle that lands in a different bin moves a bin mean by a
finite jump; the random sum of those jumps grows as sqrt(#crossings) ~ dq, so the
residual is ~dq -- a constant removed fraction, i.e. a floor. Measured on these
particles: hard phi1 bins (30 or 60 per tail; `hardbin*` in the JSON) plateau at
82-93% removed at every dq <= 0.025, the Gaussian kernel on the same particles
reaches 99%; the independent streamTrack re-fits (grey points in b) scatter
between 82% and 95% at dq <= 0.025.

WHY FORWARD MODE: q is one scalar, so forward mode gives the whole (6, N) tangent
in ONE pass (reverse mode would need one pass per output). It needs an adjoint
that differentiates the solver's own operations (`adjoint="direct"`), reached
through `integrate_kwargs`.

VALIDATION of the tangent: per particle, the rms phi2 residual of the AD
prediction falls as dq^2 (stored as `particle_phi2`); and (first version, N=120)
max|FD-AD|/max|AD| = 5.9e-02, 6.0e-04, 1.6e-05 at h = 1e-2, 1e-3, 1e-4 against a
common-random-number central difference.

Reproducibility: measure -> json, the plot is byte-stable from it. Delete the JSON
to re-measure (~10 min on 12 CPU cores).
"""

import json
import os

import numpy

from paper.figures._harness import layout
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
_N = 6000                              # both tails, 3000 each
_SEED = 20260707
_TD_GYR = 4.5
_DQS = (-0.2, -0.1, 0.1, 0.2)          # shown as tracks: q = 0.7 .. 1.1
# |dq| halving from 0.2 to 0.003: two decades to see the dq^2 scaling
_DQS_ERR = tuple(0.2 / 2**k for k in range(7))
_H_DEG = 2.0                           # fixed kernel bandwidth [deg]
_NGRID = 200                           # fixed phi1 grid points per tail
_IKW = {"adjoint": "direct", "max_steps": 100000}


def _key(dq):
    return f"{dq:+.6f}"


def _measure():
    import warnings

    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from galpy.backend import random as grandom
    from galpy.df import fardal15spraydf
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential
    from galpy.util import conversion, coords

    warnings.simplefilter("ignore")
    td = _TD_GYR / conversion.time_in_Gyr(220.0, 8.0)
    # ONE stream frame for every q, from the q-independent initial condition
    T = numpy.asarray(Orbit(_IC).align_to_orbit(), dtype=float)
    mass = 2.0e4 / conversion.mass_in_msol(220.0, 8.0)
    key = grandom.key(_SEED, backend="jax")
    nl = _N // 2
    tails = (slice(0, nl), slice(nl, _N))

    def make_df(q, sampling=True):
        lp = LogarithmicHaloPotential(normalize=1.0, q=q)
        if sampling:  # backend progenitor + differentiable integrator
            prog = Orbit(jnp.stack([jnp.asarray(v) for v in _IC]))
            ikw = _IKW
        else:          # track fits: plain numpy progenitor curve at this q
            prog = Orbit(_IC)
            ikw = None
        prog.turn_physical_off()
        return fardal15spraydf(progenitor=prog, pot=lp, tdisrupt=td, tail="both",
                               integrate_kwargs=ikw, progenitor_mass=mass)

    def particles(q):
        return make_df(q).sample(n=_N, integrate=True, key=key, return_orbit=False)

    # ONE forward pass -> present-day particles at q0 and their q-tangents
    p0, dp = jax.jvp(particles, (jnp.asarray(_Q0),), (jnp.asarray(1.0),))
    p0 = numpy.asarray(p0, dtype=float)
    dp = numpy.asarray(dp, dtype=float)
    dqs = sorted({s * d for d in _DQS_ERR for s in (-1, 1)})
    direct = {_key(dq): numpy.asarray(particles(jnp.asarray(_Q0 + dq)), dtype=float)
              for dq in dqs}

    def sky(p):
        o = Orbit(p.T)
        p12 = coords.radec_to_custom(o.ra(), o.dec(), T=T, degree=True)
        return p12[:, 0], p12[:, 1]

    # fixed phi1 grid per tail: the inner 5-95% of the q0 particles
    x0, y0 = sky(p0)
    grids = [numpy.linspace(*numpy.percentile(x0[sl], [5.0, 95.0]), _NGRID)
             for sl in tails]

    def kernel_track(p):
        """FIXED linear smoother: Gaussian-kernel mean phi2 on the fixed grid --
        a smooth function of the particles, so a first-order prediction of the
        particles is a first-order prediction of this track."""
        x, y = sky(p)
        out = []
        for sl, g in zip(tails, grids):
            w = numpy.exp(-0.5 * ((g[:, None] - x[None, sl]) / _H_DEG) ** 2)
            out.append((w * y[None, sl]).sum(1) / w.sum(1))
        return numpy.concatenate(out)

    def hardbin_track(p, nb):
        """the same fixed grid, but hard phi1 bins: a DIScontinuous estimator
        (kept in the JSON, not plotted -- it reproduces the re-fit plateau)."""
        x, y = sky(p)
        out = []
        for sl, g in zip(tails, grids):
            e = numpy.linspace(g[0], g[-1], nb + 1)
            i = numpy.digitize(x[sl], e) - 1
            ok = (i >= 0) & (i < nb)
            m = numpy.bincount(i[ok], y[sl][ok], nb) / numpy.bincount(i[ok], None, nb)
            out.append(numpy.interp(g, 0.5 * (e[1:] + e[:-1]), m))
        return numpy.concatenate(out)

    def st_fit(q, p):
        """galpy's streamTrack, default (GCV) smoothing, both tails."""
        tr = make_df(q, sampling=False).streamTrack(
            particles=p, tail="both", custom_sky_transform=T)
        arms = []
        for arm in (tr.leading, tr.trailing):
            tp = numpy.asarray(arm.tp_grid(), dtype=float)[::4]
            arms.append({"phi1": numpy.asarray(arm.phi1(tp), dtype=float).tolist(),
                         "phi2": numpy.asarray(arm.phi2(tp), dtype=float).tolist()})
        return arms

    def st_on_grid(arms):
        out = []
        for a, g in zip(arms, grids):
            x, y = numpy.asarray(a["phi1"]), numpy.asarray(a["phi2"])
            o = numpy.argsort(x)
            out.append(numpy.interp(g, x[o], y[o], left=numpy.nan, right=numpy.nan))
        return numpy.concatenate(out)

    def rms(a):
        a = a[numpy.isfinite(a)]
        return float(numpy.sqrt(numpy.mean(a**2)))

    out = {"q0": _Q0, "n": _N, "seed": _SEED, "h_deg": _H_DEG,
           "dqs": list(_DQS), "dqs_err": list(_DQS_ERR),
           "grid_phi1": [[float(g[0]), float(g[-1])] for g in grids],
           "kernel": {}, "streamtrack_refit": {}, "hardbin30": {}, "hardbin60": {},
           "particle_phi2": {},
           "tracks": {"ad": {}, "direct": {}}}
    kb = kernel_track(p0)
    sb = st_fit(_Q0, p0)
    out["tracks"]["base"] = sb
    sbg = st_on_grid(sb)
    for dq in dqs:
        k = _key(dq)
        pa, pd = p0 + dq * dp, direct[k]
        kd, ka = kernel_track(pd), kernel_track(pa)
        out["kernel"][k] = {"shift": rms(kd - kb), "resid": rms(ka - kd)}
        for nb in (30, 60):
            hb, hd, ha = (hardbin_track(p, nb) for p in (p0, pd, pa))
            out[f"hardbin{nb}"][k] = {"shift": rms(hd - hb), "resid": rms(ha - hd)}
        sd, sa = st_fit(_Q0 + dq, pd), st_fit(_Q0 + dq, pa)
        sdg, sag = st_on_grid(sd), st_on_grid(sa)
        out["streamtrack_refit"][k] = {"shift": rms(sdg - sbg), "resid": rms(sag - sdg)}
        # test (1): per particle, same key
        _, ya = sky(pa)
        _, yd = sky(pd)
        out["particle_phi2"][k] = {"shift": rms(yd - y0), "resid": rms(ya - yd)}
        if any(abs(dq - d) < 1e-12 for d in _DQS):
            out["tracks"]["ad"][k], out["tracks"]["direct"][k] = sa, sd
    return out


@figure(
    name="gS3_track_extrapolation",
    track="G",
    seed=_SEED,
    description="one forward-mode AD pass predicts the streamspraydf smooth track (both tails) for LARGE halo-flattening changes (q=0.7-1.1), against a direct re-run; like-for-like residual falls as dq^2 (CPU float64) [measure->json, plot byte-stable]",
)
def make():
    import matplotlib.pyplot as plt

    if os.path.exists(_JSON):
        data = json.load(open(_JSON))
    else:
        data = _measure()
        os.makedirs(_DATA, exist_ok=True)
        with open(_JSON, "w") as fh:
            json.dump(data, fh, indent=1, sort_keys=True)
            fh.write("\n")

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=layout.figsize(PAPER, (7.0, 3.1), (9.8, 4.0)),
        gridspec_kw={"width_ratios": [1.25, 1.0]})
    fs_leg = layout.FS["legend"] if PAPER else 7.5
    fs_note = layout.FS["note"] if PAPER else 7.5
    q0 = data["q0"]

    # ---- (a) tracks: a sequential colormap in q ----
    cmap = plt.get_cmap("viridis")
    qs = [q0 + d for d in data["dqs"]] + [q0]
    qlo, qhi = min(qs), max(qs)

    def qcol(q):
        return cmap(0.9 * (q - qlo) / (qhi - qlo))

    for dq in sorted(data["dqs"] + [0.0]):
        k = _key(dq)
        c = qcol(q0 + dq)
        if dq == 0.0:  # the reference q0, heavier
            for arm in data["tracks"]["base"]:
                ax1.plot(arm["phi1"], arm["phi2"], "-", color=c, lw=2.2, zorder=3)
            ax1.plot([], [], "-", color=c, lw=2.2, label=rf"$q_0={q0:.1f}$")
            continue
        for arm in data["tracks"]["direct"][k]:
            ax1.plot(arm["phi1"], arm["phi2"], "-", color=c, lw=1.4)
        for arm in data["tracks"]["ad"][k]:
            ax1.plot(arm["phi1"], arm["phi2"], "--", color=c, lw=1.4, dashes=(3, 1.6))
        ax1.plot([], [], "-", color=c, lw=1.4, label=rf"$q={q0 + dq:.1f}$")
    ax1.set_xlabel(r"$\phi_1$ [deg]")
    ax1.set_ylabel(r"$\phi_2$ [deg]")
    ax1.set_title("(a) direct (solid) vs one AD pass (dashed)")
    # the legend sits inside the arch, below the two tails' near ends
    ax1.legend(fontsize=fs_leg, loc="lower center", ncol=1, framealpha=0.92,
               handlelength=1.6, handletextpad=0.5, borderaxespad=0.6)

    # ---- (b) like-for-like: shift vs residual, worse of +/- dq ----
    dqs = numpy.array(sorted(data["dqs_err"]))

    def worst(block, field):
        return numpy.array([max(data[block][_key(+d)][field], data[block][_key(-d)][field])
                            for d in dqs])

    shift = worst("kernel", "shift")
    resid = worst("kernel", "resid")
    refit = worst("streamtrack_refit", "resid")
    ax2.loglog(dqs, shift, "s-", color="#111827", ms=4.5, lw=1.3,
               label="shift being predicted")
    cres = "#c2410c"
    ax2.loglog(dqs, resid, "o-", color=cres, ms=5, lw=1.3,
               label="residual after one AD pass")
    ax2.loglog(dqs, refit, "x", color="0.55", ms=5, mew=1.1,
               label="residual, two independent\nstreamTrack fits")
    # dq^2 guide, offset below the residual so both stay visible
    g = resid[0] / 3.0 * (dqs / dqs[0]) ** 2
    ax2.loglog(dqs, g, ":", color="0.35", lw=1.1)
    ax2.annotate(r"$\propto\Delta q^2$", (dqs[2], g[2]), textcoords="offset points",
                 xytext=(4, -4), ha="left", va="top", fontsize=fs_note, color="0.35")
    # removed fraction as a row along the bottom, one entry per |dq|
    ylo = 0.02 * min(resid.min(), refit.min())
    for x, m, e in zip(dqs, shift, resid):
        ax2.text(x, 1.6 * ylo, f"{100 * (1 - e / m):.0f}%", ha="center", va="bottom",
                 fontsize=fs_note, color=cres)
    ax2.text(dqs[0] / 1.33, 1.6 * ylo * 3.2, "removed:", ha="left", va="bottom",
             fontsize=fs_note, color=cres)
    ax2.set_xlabel(r"$|\Delta q|$")
    ax2.set_ylabel(r"rms $\Delta\phi_2$ at fixed $\phi_1$ [deg]")
    ax2.set_title("(b) shift and residual")
    ax2.legend(fontsize=fs_leg, loc="upper left", framealpha=0.92, handlelength=1.6)
    ax2.set_xticks([0.003, 0.01, 0.03, 0.1, 0.2])
    ax2.set_xticklabels(["0.003", "0.01", "0.03", "0.1", "0.2"])
    ax2.xaxis.set_minor_formatter(plt.NullFormatter())
    ax2.set_xlim(dqs.min() / 1.4, dqs.max() * 1.4)
    ax2.set_ylim(ylo, 1500 * shift.max())  # legend above the data

    fig.tight_layout()
    return fig
