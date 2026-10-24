"""GS3 -- the derivative of a spray stream's fitted track predicts the track in a
halo of different flattening.

    track_pred(q0+dq) = track(q0) + dq * d(track)/dq     [ONE jit(jvp) pass]
    track_direct      = track(q0+dq)                      [full re-run]

track(q) is the whole pipeline, galpy's own API: fardal15spraydf(...).streamTrack(
n=6000, key=key, tail='both') draws and integrates the stream in a logarithmic
halo with flattening q (4.5 Gyr) and fits the track (method="pspline", the
default since galpy #1731: REML smoothing).
It returns phi2 of the fitted track on a FIXED phi1 grid per tail (the q0
particles' 0.5-99.5% range), in ONE stream frame (phi1 along the progenitor's
orbital plane, phi2 across it) from the q-independent initial condition
(Orbit.align_to_orbit). The same random key at every q (common random numbers).
The phi1 interpolation uses a fixed window of the track's (normalized) tp grid
around the progenitor.

(a) tracks at q = 0.7 .. 1.1: direct (solid) and first-order prediction (dashed).
(b) rms shift of the track at fixed phi1 (inner 10-90% of each tail's q0
    particles) and the residual of the prediction vs |dq|, both signs; grey
    crosses: the earlier route, particles(q0) + dq * dparticles/dq re-fitted
    with the same streamTrack at q0+dq.

THE FIT IS SMOOTH: with method="pspline" every step of the fit is a smooth
function of the particles (Newton projection onto the progenitor's orbit, REML
smoothing with its implicit derivative, kernel-smoothed range), and it is the
same eagerly and under jit. Every track here (q0, direct, FD, re-fit) is the
jax.jit-ted fit, and the derivative is jit(jvp) of that same function. (Before
#1731 the traced fit froze each particle's closest point and the smoothing
level, which put a ~1%-of-shift floor under the residual at small dq and made
FD at N=6000 stall at ~3%.)

VALIDATION (`fd`, `fd_small_n` in the JSON: max|FD-AD|/max|AD| over the metric
range): central FD of the same jitted track, at N=6000 and N=600. FD converges
as h^2 down to the integration noise (~1e-5 relative below h ~ 1e-5).

sample(tail='both') is not jit-able (galpy concatenates the stripping times
with numpy), so the particle route's jvp runs eagerly; its q0 fit equals the
base track (`refit_q0_vs_track_maxdiff_deg`).

WHY FORWARD MODE: q is one scalar, so forward mode gives the whole tangent in
ONE pass. It needs an adjoint that differentiates the solver's own operations
(`adjoint="direct"`), reached through `integrate_kwargs`.

Reproducibility: measure -> json, the plot is byte-stable from it. Delete the
JSON to re-measure (~45 min on 4 CPU cores).
"""

import json
import os

import numpy

from paper.figures._harness import layout
from paper.figures._harness.figure import figure

_DATA = os.path.join(os.path.dirname(__file__), "data")
_JSON = os.environ.get("GS3_JSON", os.path.join(_DATA, "gS3_track_extrapolation.json"))
# journal version (default): no env stamp / long titles or notes; the caption
# carries the explanations. GALPY_JAX_PAPER=0 restores the annotated figure.
PAPER = os.environ.get("GALPY_JAX_PAPER", "1") == "1"

# The orbit must actually PROBE the flattening: q flattens the halo in z, so a
# near-planar orbit is insensitive to it by construction. This is galpy's standard
# inclined stream progenitor (z=0.887, vz=-0.477), disrupting for 4.5 Gyr so the
# halo shape has time to imprint on the track.
_IC = [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596]
_Q0 = 0.9
_N = int(os.environ.get("GS3_N", 6000))   # both tails, N/2 each
_SEED = 20260707
_TD_GYR = 4.5
_DQS = (-0.2, -0.1, 0.1, 0.2)          # shown as tracks: q = 0.7 .. 1.1
# |dq| halving from 0.2 to 0.003: two decades for the residual's scaling
_DQS_ERR = tuple(0.2 / 2**k for k in range(7))
_FD_H = (1e-3, 1e-4, 1e-5, 1e-6, 1e-7)  # FD validation steps
_N_FD = 600                             # FD check also at this N
_NGRID = 300                            # fixed phi1 grid points per tail
_PCT = (10.0, 90.0)                     # metric: inner 80% of each tail
_NCURVE = 250                           # tp samples per arm stored for (a)
_MARGIN = 3.0                           # deg beyond the particles' phi1 range
_IKW = {"adjoint": "direct", "max_steps": 100000}


def _key(dq):
    return f"{dq:+.6f}"


def _measure(n=_N, fd_only=False):
    import time
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
    t_start = time.time()
    td = _TD_GYR / conversion.time_in_Gyr(220.0, 8.0)
    # ONE stream frame for every q, from the q-independent initial condition
    T = numpy.asarray(Orbit(_IC).align_to_orbit(), dtype=float)
    mass = 2.0e4 / conversion.mass_in_msol(220.0, 8.0)
    key = grandom.key(_SEED, backend="jax")
    nl = n // 2
    tails = (slice(0, nl), slice(nl, n))

    def make_df(q):
        lp = LogarithmicHaloPotential(normalize=1.0, q=q)
        prog = Orbit(jnp.stack([jnp.asarray(v) for v in _IC]))
        prog.turn_physical_off()
        return fardal15spraydf(progenitor=prog, pot=lp, tdisrupt=td, tail="both",
                               integrate_kwargs=_IKW, progenitor_mass=mass)

    def particles(q):
        return make_df(q).sample(n=n, integrate=True, key=key, return_orbit=False)

    def arms(tr):
        return (tr.leading, tr.trailing)

    def fit(q, p=None):
        """galpy's streamTrack (default pspline fit), both tails; p=None draws
        the particles inside (streamTrack(n=, key=)), as in the paper snippet."""
        return make_df(q).streamTrack(n=n, particles=p, key=key, tail="both",
                                      custom_sky_transform=T)

    def npy(t):
        return jax.tree_util.tree_map(lambda a: numpy.asarray(a, dtype=float), t)

    # particle route (eager jvp: sample() is not jit-able for tail='both'):
    # particles(q0) and their tangent; their q0 fit must equal the base track
    pp0, dpp = npy(jax.jvp(particles, (jnp.asarray(_Q0),), (jnp.asarray(1.0),)))
    o0 = Orbit(pp0.T)
    x0 = coords.radec_to_custom(o0.ra(), o0.dec(), T=T, degree=True)[:, 0]
    pct = [numpy.percentile(x0[sl], [0.5, *_PCT, 99.5]) for sl in tails]
    # fixed phi1 grid per tail over the particles' 0.5-99.5% range; the metric
    # uses its inner _PCT part
    grids = [numpy.linspace(p[0], p[-1], _NGRID) for p in pct]
    metric = numpy.concatenate([(g >= p[1]) & (g <= p[2]) for g, p in zip(grids, pct)])

    # A FIXED window of the (normalized, concrete) tp grid: from the progenitor
    # (tp=0) out to where the q0 track leaves the particles' phi1 range by
    # _MARGIN deg.
    full = npy(jax.jit(lambda q: [(a.tp_grid(), a.phi1(a.tp_grid()))
                                   for a in arms(fit(q))])(jnp.asarray(_Q0)))
    windows = []
    for (tpg, x), p in zip(full, pct):
        i0 = int(numpy.argmin(numpy.abs(tpg)))
        step = 1 if tpg[-1] > 0 else -1
        inside = (x >= p[0] - _MARGIN) & (x <= p[-1] + _MARGIN)
        i = i0
        while 0 <= i + step < len(x) and inside[i + step]:
            i += step
        windows.append(numpy.arange(min(i0, i), max(i0, i) + 1))

    def curves(tr):
        """phi1, phi2 along each arm at _NCURVE tp samples in its window (panel a)."""
        out = []
        for arm, w in zip(arms(tr), windows):
            tp = arm.tp_grid()[w[numpy.linspace(0, len(w) - 1, _NCURVE).astype(int)]]
            out.append(jnp.stack([arm.phi1(tp), arm.phi2(tp)]))
        return jnp.stack(out)  # (2 arms, 2, _NCURVE)

    def on_grid(tr):
        """phi2 of the fitted track at the fixed phi1 grids (2*_NGRID), plus each
        arm's phi1 extent in its window (to mask grid points outside a track)."""
        vals, ext = [], []
        for arm, g, w in zip(arms(tr), grids, windows):
            tp = arm.tp_grid()[w]
            x, y = arm.phi1(tp), arm.phi2(tp)
            o = jnp.argsort(x)
            vals.append(jnp.interp(jnp.asarray(g), x[o], y[o]))
            ext.append(jnp.stack([jnp.min(x), jnp.max(x)]))
        return jnp.concatenate(vals), jnp.stack(ext)

    def track(q):
        tr = fit(q)
        return on_grid(tr), curves(tr)

    track_jit = jax.jit(track)
    jvp_jit = jax.jit(lambda q: jax.jvp(track, (q,), (jnp.asarray(1.0),)))
    refit_jit = jax.jit(lambda q, p: on_grid(fit(q, p)))

    def masked(vals, ext):
        g = numpy.concatenate(grids)
        lo = numpy.repeat(ext[:, 0], _NGRID)
        hi = numpy.repeat(ext[:, 1], _NGRID)
        return numpy.where((g >= lo) & (g <= hi), vals, numpy.nan)

    def rms(a):
        a = a[metric]
        a = a[numpy.isfinite(a)]
        return float(numpy.sqrt(numpy.mean(a**2)))

    # ---- ONE forward pass: the q0 track and its q-derivative ----
    t0 = time.time()
    ((g0, e0), cb), ((dg, _), dc) = npy(jvp_jit(jnp.asarray(_Q0)))
    t_jvp = time.time() - t0
    b = masked(g0, e0)
    # same-function check: the jvp primal vs a plain jitted call
    (g0p, _), _ = npy(track_jit(jnp.asarray(_Q0)))
    primal_diff = float(numpy.nanmax(numpy.abs(g0p - g0)))
    gr0, _ = npy(refit_jit(jnp.asarray(_Q0), jnp.asarray(pp0)))
    refit_q0_diff = float(numpy.nanmax(numpy.abs(gr0 - g0)))

    out = {"q0": _Q0, "n": n, "seed": _SEED, "dqs": list(_DQS),
           "dqs_err": list(_DQS_ERR), "pct": list(_PCT), "ngrid": _NGRID,
           "grid_phi1": [[float(g[0]), float(g[-1])] for g in grids],
           "metric_phi1": [[float(p[1]), float(p[2])] for p in pct],
           "window_len": [len(w) for w in windows],
           "primal_vs_jit_maxdiff_deg": primal_diff,
           "refit_q0_vs_track_maxdiff_deg": refit_q0_diff, "t_jvp_s": t_jvp,
           "track": {}, "particle_refit": {}, "fd": {},
           "tracks": {"base": {"curve": cb.tolist(), "phi2": g0.tolist()},
                      "dphi2_dq": dg.tolist(), "pred": {}, "direct": {}},
           "grids": [g.tolist() for g in grids]}

    # ---- FD validation of the derivative, same jitted function ----
    for h in _FD_H:
        (gp, _), _ = npy(track_jit(jnp.asarray(_Q0 + h)))
        (gm, _), _ = npy(track_jit(jnp.asarray(_Q0 - h)))
        fd = (gp - gm) / (2 * h)
        out["fd"][f"{h:.0e}"] = {
            "maxrel": float(numpy.max(numpy.abs(fd - dg)) / numpy.max(numpy.abs(dg))),
            "maxrel_metric": float(numpy.max(numpy.abs(fd - dg)[metric])
                                   / numpy.max(numpy.abs(dg[metric])))}
        print("FD", h, out["fd"][f"{h:.0e}"], flush=True)
    if fd_only:
        return out["fd"]

    # ---- direct re-runs vs the derivative prediction ----
    dqs = sorted({s * d for d in _DQS_ERR for s in (-1, 1)} | set(_DQS))
    for dq in dqs:
        k = _key(dq)
        (gd, ed), cd = npy(track_jit(jnp.asarray(_Q0 + dq)))
        d = masked(gd, ed)
        pred = g0 + dq * dg
        gr, er = npy(refit_jit(jnp.asarray(_Q0 + dq), jnp.asarray(pp0 + dq * dpp)))
        r = masked(gr, er)
        shift = rms(d - b)
        out["track"][k] = {"shift": shift, "resid": rms(pred - d),
                           "removed": 1.0 - rms(pred - d) / shift}
        out["particle_refit"][k] = {"resid": rms(r - d),
                                    "removed": 1.0 - rms(r - d) / shift}
        print(k, out["track"][k], out["particle_refit"][k], flush=True)
        if any(abs(dq - x) < 1e-12 for x in _DQS):
            out["tracks"]["direct"][k] = {"curve": cd.tolist(),
                                          "phi2": numpy.where(numpy.isfinite(d), d,
                                                              None).tolist()}
            out["tracks"]["pred"][k] = pred.tolist()

    # measured log-log slope of the residual vs |dq| (both signs pooled)
    def slope(block, lo=0.0, hi=1.0):
        x = [abs(float(k)) for k in out[block] if lo <= abs(float(k)) <= hi]
        y = [out[block][k]["resid"] for k in out[block] if lo <= abs(float(k)) <= hi]
        return float(numpy.polyfit(numpy.log(x), numpy.log(y), 1)[0])

    out["slope"] = {"track_all": slope("track"),
                    "track_le_0.025": slope("track", hi=0.0251),
                    "track_ge_0.025": slope("track", lo=0.0249),
                    "particle_refit_all": slope("particle_refit")}
    out["t_total_s"] = time.time() - t_start
    return out


@figure(
    name="gS3_track_extrapolation",
    track="G",
    seed=_SEED,
    description="the forward-mode derivative of the streamspraydf fitted track (both tails, jitted streamTrack pspline fit) predicts the track for LARGE halo-flattening changes (q=0.7-1.1), against a direct re-run (CPU float64) [measure->json, plot byte-stable]",
)
def make():
    import matplotlib.pyplot as plt

    if os.path.exists(_JSON):
        data = json.load(open(_JSON))
    else:
        data = _measure()
        # the FD check also at a smaller N
        data["fd_small_n"] = {"n": _N_FD, "fd": _measure(_N_FD, fd_only=True)}
        os.makedirs(os.path.dirname(_JSON), exist_ok=True)
        with open(_JSON, "w") as fh:
            json.dump(data, fh, indent=1, sort_keys=True)
            fh.write("\n")

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=layout.figsize(PAPER, (7.0, 3.1), (9.8, 4.0)),
        gridspec_kw={"width_ratios": [1.25, 1.0]})
    fs_note = layout.FS["note"] if PAPER else 7.5
    q0 = data["q0"]
    ng = data["ngrid"]
    grids = [numpy.asarray(g) for g in data["grids"]]

    # ---- (a) tracks: a sequential colormap in q, labelled directly ----
    cmap = plt.get_cmap("viridis")
    qs = [q0 + d for d in data["dqs"]] + [q0]
    qlo, qhi = min(qs), max(qs)

    def qcol(q):
        return cmap(0.9 * (q - qlo) / (qhi - qlo))

    def arms_of(v):
        v = numpy.array([numpy.nan if x is None else x for x in v], dtype=float)
        return v[:ng], v[ng:]

    lbl = []
    for dq in sorted(data["dqs"] + [0.0]):
        k = _key(dq)
        c = qcol(q0 + dq)
        if dq == 0.0:  # the reference q0, heavier
            for arm in data["tracks"]["base"]["curve"]:
                ax1.plot(arm[0], arm[1], "-", color=c, lw=2.4, zorder=3)
            ye = data["tracks"]["base"]["curve"][1][1]
            lbl.append((q0, c, True))
            continue
        for arm in data["tracks"]["direct"][k]["curve"]:
            ax1.plot(arm[0], arm[1], "-", color=c, lw=1.3)
        for g, y in zip(grids, arms_of(data["tracks"]["pred"][k])):
            ax1.plot(g, y, "--", color=c, lw=1.3, dashes=(3, 1.6))
        lbl.append((q0 + dq, c, False))
    # direct labels at the trailing tail's far end (phi1 max), spread apart
    ends = []
    for q, c, thick in lbl:
        cur = (data["tracks"]["base"]["curve"] if thick
               else data["tracks"]["direct"][_key(q - q0)]["curve"])[1]
        x, y = numpy.asarray(cur[0]), numpy.asarray(cur[1])
        i = numpy.nanargmax(x)
        ends.append([y[i], x[i], q, c, thick])
    ends.sort(key=lambda e: -e[0])
    for j in range(1, len(ends)):  # >= 0.75 deg apart in phi2, top down
        ends[j][0] = min(ends[j][0], ends[j - 1][0] - 0.75)
    xr = max(e[1] for e in ends)
    for ylab, xe, q, c, thick in ends:
        ax1.text(xr + 4, ylab, rf"$q_0={q:.1f}$" if thick else rf"$q={q:.1f}$",
                 ha="left", va="center", fontsize=fs_note, color=c,
                 fontweight="bold" if thick else "normal")
    ax1.set_xlabel(r"$\phi_1$ [deg]")
    ax1.set_ylabel(r"$\phi_2$ [deg]")
    ax1.set_title("(a) direct (solid) vs track derivative (dashed)")
    xl = ax1.get_xlim()
    ax1.set_xlim(xl[0], xl[1] + 0.16 * (xl[1] - xl[0]))  # room for labels

    # ---- (b) shift and residual, both signs ----
    def series(block, field, sign):
        d = numpy.array(sorted(data["dqs_err"]))
        return d, numpy.array([data[block][_key(sign * x)][field] for x in d])

    cres = "#c2410c"
    for sign, mk in ((+1, "o"), (-1, "v")):
        d, shift = series("track", "shift", sign)
        _, resid = series("track", "resid", sign)
        _, refit = series("particle_refit", "resid", sign)
        ax2.loglog(d, shift, "s-" if sign > 0 else "s--", color="#111827", ms=3.5,
                   lw=1.1, mfc="#111827" if sign > 0 else "white")
        ax2.loglog(d, resid, mk + ("-" if sign > 0 else "--"), color=cres, ms=4,
                   lw=1.2, mfc=cres if sign > 0 else "white")
        ax2.loglog(d, refit, "x", color="0.6", ms=4, mew=0.9, alpha=0.8)
    d, shift = series("track", "shift", +1)
    _, resid = series("track", "resid", +1)
    _, refit = series("particle_refit", "resid", +1)
    # direct labels
    ax2.annotate("shift", (d[3], shift[3]), textcoords="offset points",
                 xytext=(-4, 5), ha="right", va="bottom", fontsize=fs_note,
                 color="#111827")
    ax2.annotate("residual of the\ntrack derivative", (d[4], resid[4]),
                 textcoords="offset points", xytext=(9, -9), ha="left", va="top",
                 fontsize=fs_note, color=cres)
    ax2.annotate("particles + re-fit", (d[0], refit[0]), textcoords="offset points",
                 xytext=(6, -2), ha="left", va="top", fontsize=fs_note,
                 color="0.5")
    # short dq^2 guide over the small steps, above the residual
    gx = d[d <= 0.0251]
    g = resid[3] * 4.0 * (gx / gx[-1]) ** 2
    ax2.loglog(gx, g, ":", color="0.35", lw=1.0)
    ax2.annotate(r"$\propto\Delta q^2$", (gx[1], g[1]), textcoords="offset points",
                 xytext=(-3, 4), ha="right", va="bottom", fontsize=fs_note, color="0.35")
    # fraction of the shift removed by the track derivative (worse of +/- dq)
    rem = numpy.minimum(*(series("track", "removed", s)[1] for s in (1, -1)))
    ylo = 0.025 * min(resid.min(), refit.min())
    for x, r in zip(d, rem):
        ax2.text(x, ylo * 1.25, f"{100 * r:.0f}%", ha="center", va="bottom",
                 fontsize=fs_note, color=cres)
    ax2.text(d[0] / 1.33, ylo * 1.25 * 4.0, "shift removed:", ha="left",
             va="bottom", fontsize=fs_note, color=cres)
    ax2.set_ylim(ylo, 3.0 * shift.max())
    ax2.set_xlabel(r"$|\Delta q|$  (filled: $\Delta q>0$, open: $\Delta q<0$)")
    ax2.set_ylabel(r"rms $\Delta\phi_2$ at fixed $\phi_1$ [deg]")
    ax2.set_title("(b) shift and residual")
    ax2.set_xticks([0.003, 0.01, 0.03, 0.1, 0.2])
    ax2.set_xticklabels(["0.003", "0.01", "0.03", "0.1", "0.2"])
    ax2.xaxis.set_minor_formatter(plt.NullFormatter())
    ax2.set_xlim(d.min() / 1.4, d.max() * 1.4)

    fig.tight_layout()
    return fig
