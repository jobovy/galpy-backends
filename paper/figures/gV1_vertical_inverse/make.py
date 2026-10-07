"""GV1 -- the differentiable 1D inverse action-angle map: d(torus)/d(potential).

``actionAngleVerticalInverse`` maps (J, theta) -> (z, v_z) for vertical motion
in a 1D potential through a "momentum-matched" canonical torus map: a family of
tori is built on an energy grid (turning points by a root find, a spectral sweep
of each torus for its action, frequency and anomaly map, Gauss-Newton fits of
the anomaly-map coefficients, Hermite interpolants in J), and each evaluation
inverts the angle relation for the anomaly with a bracketed root find. Both the
CONSTRUCTION and the EVALUATION now run natively under jax/torch, so the map is
differentiable in the potential's parameters (through the construction) as well
as in J and theta (through the evaluation), and the whole pipeline -- potential
parameter in, (z, v_z, Omega) out -- traces under a single ``jax.jit``.

Setup: the self-gravitating isothermal disk ``IsothermalDiskPotential(amp=1,
sigma)`` (Phi = 2 sigma^2 ln cosh(z / 2H), H = sigma / sqrt(8 pi amp)),
sigma0 = 0.5, family on 33 energies E in [0, 1] with setup_interp=True (the
default mm_npt=40 harmonics). Every derivative below is jax forward-mode
(``jax.jvp``) of the jitted function sigma |-> build family -> xvFreqs(J,
theta), float64 on the CPU.

(a) Phase-space tori (z, v_z) at five fixed actions for sigma0 (solid), the
    tori REBUILT at sigma0 + dsigma with dsigma = 0.2 sigma0 (dashed), and the
    first-order AD prediction torus(sigma0) + dsigma d(z, v_z)/dsigma (dots),
    i.e. the derivative used the way a fitter would use it. The residual
    mismatch at this large step is the second-order term.

(b) The stringent check, per angle along one torus (J = 0.06, 64 angles
    including both turning points and the midplane crossings): |AD - FD| /
    max|AD| for d(z, v_z)/dsigma, d Omega/dsigma (a single number per torus,
    drawn as a horizontal line), d(z, v_z)/dJ and d(z, v_z)/dtheta, each
    against a 4th-order central (5-point) finite difference at its best step
    (each stencil point a full re-evaluation); the sigma FD REBUILDS
    the whole family at sigma0 +- h. Also drawn: |{z, v_z} - 1|, the Poisson
    bracket dz/dtheta dv/dJ - dz/dJ dv/dtheta evaluated purely from the AD
    Jacobian -- the map is canonical, so this tests the J and theta derivatives
    jointly against an exact identity (no FD involved).

(c) Why those are the right FD steps: the max-over-angle relative error of the
    5-point central FD against AD versus the step h. Each curve falls as h^4
    (truncation) to a round-off floor ~ eps_eff / h; the AD derivative sits at
    the bottom of every V, i.e. agrees with the FD to the FD's own floor.

(d) The derivatives at work: fitting the disk. 120 "stars" are placed at
    random angles on three tori of known action (J = 0.02, 0.08, 0.2) of the
    true disk (sigma, amp) = (0.5, 1), with Gaussian noise 0.01 on z and v_z.
    (sigma, amp) are fit jointly with the 120 unknown angles (122 parameters,
    the angles initialised at each star's nearest point on its torus at the
    starting guess (0.7, 0.6)) by Levenberg-Marquardt whose Jacobian is pure AD:
    jax.jacfwd in (ln sigma, ln amp) through the native family CONSTRUCTION and
    the diagonal d/dtheta_i (one jvp) through the EVALUATION. Shown: the LM
    path, the truth, and the 68/95% contours of the angle-marginalised
    covariance from the same AD Jacobian; inset, the data and the fitted tori.
    Near the midplane the isothermal disk is harmonic with nu^2 = 4 pi amp,
    independent of sigma, so sigma is constrained only by the anharmonic outer
    tori -- hence the elongated, anti-correlated contours.

Small tori are nearly insensitive to sigma for the same reason, so in (a) the
sigma0 and rebuilt tori differ visibly only for the larger actions (the zoom).

Reproducibility: measure -> data/gV1_vertical_inverse.json (~2 min on 4 CPU
cores, dominated by the one-time jit compiles); the plot is byte-stable from
the JSON. Re-measure by deleting the JSON.
"""

import json
import os

import numpy

from paper.figures._harness.figure import figure

_DATA = os.path.join(os.path.dirname(__file__), "data")
_JSON = os.path.join(_DATA, "gV1_vertical_inverse.json")
# journal version (default): no env stamp / long titles or notes; the caption
# carries the explanations. GALPY_JAX_PAPER=0 restores the annotated figure.
PAPER = os.environ.get("GALPY_JAX_PAPER", "1") == "1"

_AMP = 1.0
_SIG0 = 0.5
_DSIG = 0.1                        # sizeable step drawn in panel (a): 20%
_ES = numpy.linspace(0.0, 1.0, 33)  # energy grid of the family
_JSHOW = (0.01, 0.04, 0.09, 0.16, 0.25)
_NSHOW = 241
_JCHK = 0.06
_NCHK = 64                         # multiple of 4: hits both turning points
_HS = (1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6)

# (d) the fit: noisy (z, v_z) of stars on three tori of known action, unknown
# angles; fit (ln sigma, ln amp) jointly with the per-star angles by
# Levenberg-Marquardt on the AD Jacobian
_FIT_ES = numpy.linspace(0.0, 1.5, 33)
_FIT_J = (0.02, 0.08, 0.2)
_FIT_NP = 40                       # stars per torus
_FIT_NOISE = 0.01                  # Gaussian noise on z and v_z
_FIT_TRUE = (0.5, 1.0)             # (sigma, amp)
_FIT_START = (0.7, 0.6)
_FIT_SEED = 1
_FIT_NGRID = 256                   # angle grid for the initial angle guesses

_QUANT = [  # key, label, colour, linestyle
    ("dx_ds", r"$\partial z/\partial\sigma$", "#0072B2", "-"),
    ("dv_ds", r"$\partial v_z/\partial\sigma$", "#56B4E9", "-"),
    ("dO_ds", r"$\partial\Omega/\partial\sigma$", "#000000", "--"),
    ("dx_dJ", r"$\partial z/\partial J$", "#D55E00", "-"),
    ("dv_dJ", r"$\partial v_z/\partial J$", "#E69F00", "-"),
    ("dx_dth", r"$\partial z/\partial\theta$", "#009E73", "-"),
    ("dv_dth", r"$\partial v_z/\partial\theta$", "#CC79A7", "-"),
]


def _measure():
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from galpy.actionAngle import actionAngleVerticalInverse
    from galpy.potential import IsothermalDiskPotential

    def f(s, J, a):
        aA = actionAngleVerticalInverse(
            pot=IsothermalDiskPotential(amp=_AMP, sigma=s), Es=_ES, setup_interp=True
        )
        x, v, O = aA.xvFreqs(J, a)
        return x, v, O

    fj = jax.jit(f)

    def _jvp(argnum):
        def g(s, J, a):
            args = [s, J, a]
            t = [jnp.zeros_like(s), jnp.zeros_like(J), jnp.zeros_like(a)]
            t[argnum] = jnp.ones_like(args[argnum])
            # theta: each output depends on its own angle only, so the all-ones
            # tangent gives the per-angle derivative
            return jax.jvp(f, tuple(args), tuple(t))

        return jax.jit(g)

    jv_s, jv_J, jv_a = _jvp(0), _jvp(1), _jvp(2)
    s0 = jnp.asarray(_SIG0)
    npf = lambda t: [numpy.asarray(o, dtype=float) for o in t]  # noqa: E731

    # (a) tori at sigma0, rebuilt at sigma0+dsigma, and the AD prediction
    ash = jnp.asarray(numpy.linspace(0.0, 2.0 * numpy.pi, _NSHOW))
    tori = []
    for J in _JSHOW:
        Jj = jnp.asarray(J)
        (x0, v0, O0), (dx, dv, dO) = jv_s(s0, Jj, ash)
        x1, v1, O1 = fj(s0 + _DSIG, Jj, ash)
        x0, v0, dx, dv, x1, v1 = npf((x0, v0, dx, dv, x1, v1))
        tori.append({
            "J": J, "x0": x0.tolist(), "v0": v0.tolist(),
            "x_ad": (x0 + _DSIG * dx).tolist(), "v_ad": (v0 + _DSIG * dv).tolist(),
            "x1": x1.tolist(), "v1": v1.tolist(),
            "O0": float(O0), "O1": float(O1), "O_ad": float(O0 + _DSIG * dO),
        })

    # (b, c) AD vs 5-point central FD along one torus
    ach = numpy.arange(_NCHK) * 2.0 * numpy.pi / _NCHK
    a = jnp.asarray(ach)
    Jc = jnp.asarray(_JCHK)
    ad = {}
    _, (ad["dx_ds"], ad["dv_ds"], ad["dO_ds"]) = jv_s(s0, Jc, a)
    _, (ad["dx_dJ"], ad["dv_dJ"], _dOdJ) = jv_J(s0, Jc, a)
    _, (ad["dx_dth"], ad["dv_dth"], _) = jv_a(s0, Jc, a)
    ad = {k: numpy.atleast_1d(numpy.asarray(v, dtype=float)) for k, v in ad.items()}
    pb = ad["dx_dth"] * ad["dv_dJ"] - ad["dx_dJ"] * ad["dv_dth"]

    def _d5(fun, h):
        # 4th-order central (5-point) stencil, each point a full re-evaluation
        vals = [npf(fun(c * h)) for c in (2.0, 1.0, -1.0, -2.0)]
        return [(-vals[0][i] + 8.0 * vals[1][i] - 8.0 * vals[2][i] + vals[3][i])
                / (12.0 * h) for i in range(3)]

    fd = {k: [] for k in ad}
    for h in _HS:
        # sigma: REBUILD the whole family at sigma0 + c h
        d = _d5(lambda e: fj(s0 + e, Jc, a), h)
        for k, i in (("dx_ds", 0), ("dv_ds", 1), ("dO_ds", 2)):
            fd[k].append(numpy.atleast_1d(d[i]))
        d = _d5(lambda e: fj(s0, Jc + e, a), h)
        for k, i in (("dx_dJ", 0), ("dv_dJ", 1)):
            fd[k].append(d[i])
        d = _d5(lambda e: fj(s0, Jc, a + e), h)
        for k, i in (("dx_dth", 0), ("dv_dth", 1)):
            fd[k].append(d[i])

    rel = {}   # [ih][angle]: |AD - FD(h)| / max|AD|
    hconv = {}  # [ih]: max over angles
    best = {}
    for k in ad:
        scale = numpy.max(numpy.abs(ad[k]))
        r = numpy.array([numpy.abs(ad[k] - fdh) / scale for fdh in fd[k]])
        rel[k] = r
        hconv[k] = r.max(axis=1).tolist()
        best[k] = int(numpy.argmin(r.max(axis=1)))

    fit = _measure_fit(jax, jnp, actionAngleVerticalInverse, IsothermalDiskPotential)

    return {
        "fit": fit,
        "amp": _AMP, "sig0": _SIG0, "dsig": _DSIG, "Es": _ES.tolist(),
        "tori": tori, "Jchk": _JCHK, "angles": ach.tolist(), "hs": list(_HS),
        "ad_scale": {k: float(numpy.max(numpy.abs(v))) for k, v in ad.items()},
        "rel_best": {k: rel[k][best[k]].tolist() for k in ad},
        "h_best": {k: _HS[best[k]] for k in ad},
        "hconv": hconv,
        "poisson_minus_1": numpy.abs(pb - 1.0).tolist(),
    }


def _measure_fit(jax, jnp, actionAngleVerticalInverse, IsothermalDiskPotential):
    """Joint Levenberg-Marquardt fit of (ln sigma, ln amp) and the 3*_FIT_NP
    per-star angles to noisy (z, v_z); the Jacobian is pure AD: jacfwd in the
    two potential parameters (through the native construction) and the
    diagonal d/dtheta_i (one jvp with an all-ones tangent, through the
    evaluation)."""
    rng = numpy.random.default_rng(_FIT_SEED)
    NP, n = _FIT_NP, 3 * _FIT_NP

    def build(lp):
        return actionAngleVerticalInverse(
            pot=IsothermalDiskPotential(amp=jnp.exp(lp[1]), sigma=jnp.exp(lp[0])),
            Es=_FIT_ES, setup_interp=True,
        )

    truth = jnp.log(jnp.asarray(_FIT_TRUE))
    aT = build(truth)
    xd, vd, tht = [], [], []
    for J in _FIT_J:
        th = rng.uniform(0.0, 2.0 * numpy.pi, NP)
        x, v, _ = aT.xvFreqs(J, th)
        xd.append(numpy.asarray(x) + _FIT_NOISE * rng.normal(size=NP))
        vd.append(numpy.asarray(v) + _FIT_NOISE * rng.normal(size=NP))
        tht.append(th)
    xd, vd = jnp.asarray(numpy.concatenate(xd)), jnp.asarray(numpy.concatenate(vd))

    def model(lp, th):
        aA = build(lp)
        out = [aA.xvFreqs(J, th[k * NP:(k + 1) * NP])[:2] for k, J in enumerate(_FIT_J)]
        return jnp.concatenate([o[0] for o in out]), jnp.concatenate([o[1] for o in out])

    def resid(lp, th):
        x, v = model(lp, th)
        return jnp.concatenate([(xd - x) / _FIT_NOISE, (vd - v) / _FIT_NOISE])

    rj = jax.jit(resid)
    jac_p = jax.jit(jax.jacfwd(resid, argnums=0))
    jac_t = jax.jit(lambda lp, th: jax.jvp(lambda t: resid(lp, t), (th,),
                                           (jnp.ones_like(th),))[1])
    # initial angles: nearest point of each star's torus at the starting guess
    lp = jnp.log(jnp.asarray(_FIT_START))
    grid = jnp.asarray(numpy.linspace(0.0, 2.0 * numpy.pi, _FIT_NGRID, endpoint=False))
    aA0 = build(lp)
    th0 = []
    for k, J in enumerate(_FIT_J):
        x, v, _ = aA0.xvFreqs(J, grid)
        d2 = ((xd[k * NP:(k + 1) * NP, None] - x[None, :]) ** 2
              + (vd[k * NP:(k + 1) * NP, None] - v[None, :]) ** 2)
        th0.append(grid[jnp.argmin(d2, axis=1)])
    th = jnp.concatenate(th0)
    r = numpy.asarray(rj(lp, th))
    path, chi2 = [numpy.exp(numpy.asarray(lp)).tolist()], [float(r @ r)]
    lam = 1e-3
    idx = numpy.arange(n)
    for _ in range(40):
        r = numpy.asarray(rj(lp, th))
        c2 = float(r @ r)
        A = numpy.zeros((2 * n, 2 + n))
        A[:, :2] = numpy.asarray(jac_p(lp, th))
        dt = numpy.asarray(jac_t(lp, th))
        A[idx, 2 + idx], A[n + idx, 2 + idx] = dt[:n], dt[n:]
        H, g = A.T @ A, A.T @ r
        acc = False
        while lam < 1e10:
            step = -numpy.linalg.solve(H + lam * numpy.diag(numpy.diag(H)), g)
            lpn, thn = lp + step[:2], th + step[2:]
            rn = numpy.asarray(rj(lpn, thn))
            if float(rn @ rn) < c2:
                lp, th, acc = lpn, thn, True
                lam = max(lam / 5.0, 1e-9)
                break
            lam *= 5.0
        if not acc:
            break
        path.append(numpy.exp(numpy.asarray(lp)).tolist())
        chi2.append(float(rn @ rn))
        if c2 - chi2[-1] < 1e-9 * c2:
            break
    # marginal covariance of (ln sigma, ln amp), angles profiled, at the optimum
    r = numpy.asarray(rj(lp, th))
    A = numpy.zeros((2 * n, 2 + n))
    A[:, :2] = numpy.asarray(jac_p(lp, th))
    dt = numpy.asarray(jac_t(lp, th))
    A[idx, 2 + idx], A[n + idx, 2 + idx] = dt[:n], dt[n:]
    cov = numpy.linalg.inv(A.T @ A)[:2, :2]
    # data + fitted tori for the inset
    aF = build(lp)
    ash = numpy.linspace(0.0, 2.0 * numpy.pi, 121)
    ft = [[numpy.asarray(o).tolist() for o in aF.xvFreqs(J, ash)[:2]] for J in _FIT_J]
    return {
        "J": list(_FIT_J), "np": NP, "noise": _FIT_NOISE, "true": list(_FIT_TRUE),
        "start": list(_FIT_START), "seed": _FIT_SEED, "path": path, "chi2": chi2,
        "chi2_true_angles": float(numpy.sum(numpy.asarray(
            rj(truth, jnp.asarray(numpy.concatenate(tht)))) ** 2)),
        "best": numpy.exp(numpy.asarray(lp)).tolist(), "cov_log": cov.tolist(),
        "ndof": int(2 * n - (2 + n)),
        "xd": numpy.asarray(xd).tolist(), "vd": numpy.asarray(vd).tolist(),
        "fit_tori": ft,
    }


@figure(
    name="gV1_vertical_inverse",
    track="E",
    seed=0,
    description="differentiable 1D inverse action-angle map (actionAngleVerticalInverse, momentum-matched): tori of an isothermal disk + AD d/dsigma through the native construction, AD vs rebuilt central FD per angle, FD h-convergence, Poisson bracket (jax, jit, CPU float64) [measure->json, plot byte-stable]",
)
def make():
    import matplotlib.pyplot as plt

    if os.path.exists(_JSON):
        with open(_JSON) as fh:
            data = json.load(fh)
    else:
        data = _measure()
        os.makedirs(_DATA, exist_ok=True)
        with open(_JSON, "w") as fh:
            json.dump(data, fh, indent=1, sort_keys=True)
            fh.write("\n")

    # PAPER: final printed width of a figure* (7.0 in); text sizes are then the
    # printed sizes (9-10 pt labels/ticks, 10 pt titles, 8 pt legends/notes).
    fig, axs = plt.subplots(2, 2, figsize=(7.1, 7.0) if PAPER else (10.4, 8.6))
    ax1, ax2, ax3, ax4 = axs.ravel()
    _ts = {"fontsize": 10} if PAPER else {}
    _fl = (lambda d: 8) if PAPER else (lambda d: d)  # legend / note sizes

    # (a) the tori, the rebuilt tori, and the first-order AD prediction
    s0, ds = data["sig0"], data["dsig"]
    cols = plt.cm.viridis(numpy.linspace(0.05, 0.85, len(data["tori"])))
    ins = ax1.inset_axes([0.665, 0.08, 0.325, 0.60] if PAPER else [0.60, 0.10, 0.38, 0.62])
    for t, c in zip(data["tori"], cols):
        for ax, ev in ((ax1, 8), (ins, 2)):
            ax.plot(t["x0"], t["v0"], "-", color=c, lw=1.4)
            ax.plot(t["x1"], t["v1"], "--", color=c, lw=1.0)
            ax.plot(t["x_ad"][::ev], t["v_ad"][::ev], "o", color=c, ms=2.4, mew=0)
        if PAPER:  # below each torus: the legend takes the top
            ax1.text(0.0, min(t["v1"]) * 1.02, rf"$J={t['J']:g}$", color=c, fontsize=_fl(6.5),
                     ha="center", va="top")
        else:
            ax1.text(0.0, max(t["v1"]) * 1.02, rf"$J={t['J']:g}$", color=c, fontsize=_fl(6.5),
                     ha="center", va="bottom")
    ins.set_xlim(0.27, 0.49)
    ins.set_ylim(-0.42, 0.42)
    ins.tick_params(labelsize=_fl(6))
    ins.set_title(r"$+z$ turning points" if PAPER else r"zoom: $+z$ turning points",
                  fontsize=_fl(6.5), pad=2)
    ax1.indicate_inset_zoom(ins, edgecolor="0.4")
    ax1.plot([], [], "-", color="0.3", label=rf"$\sigma_0={s0:g}$")
    ax1.plot([], [], "--", color="0.3", label=rf"rebuilt at $\sigma_0+{ds:g}$")
    ax1.plot([], [], "o", color="0.3", ms=3,
             label=r"AD: torus$(\sigma_0)+\Delta\sigma\,\partial_\sigma$torus")
    ax1.set_xlim((-0.5, 2.05) if PAPER else (-0.47, 1.12))
    ax1.set_xlabel(r"$z$")
    ax1.set_ylabel(r"$v_z$")
    ax1.set_title(r"(a) tori and their $\sigma$-derivative" if PAPER else
                  "(a) isothermal-disk tori at fixed $J$ and their\n"
                  rf"AD sensitivity to $\sigma$ ($\Delta\sigma={ds / s0:.1f}\,\sigma_0$)", **_ts)
    ax1.legend(fontsize=_fl(6.5), loc="upper right", framealpha=0.92)
    ax1.margins(y=0.12)

    # (b) per-angle AD vs central FD (best h each) + Poisson bracket
    ang = numpy.asarray(data["angles"])
    for key, lab, col, ls in _QUANT:
        r = numpy.maximum(numpy.asarray(data["rel_best"][key]), 1e-17)
        hb = data["h_best"][key]
        if key == "dO_ds":
            ax2.axhline(r[0], color=col, ls=ls, lw=1.1,
                        label=lab + rf" ($h=10^{{{numpy.log10(hb):.1f}}}$)")
        else:
            ax2.semilogy(ang, r, ls, color=col, lw=1.0, marker="o", ms=2.0,
                         label=lab + rf" ($h=10^{{{numpy.log10(hb):.1f}}}$)")
    pb = numpy.maximum(numpy.asarray(data["poisson_minus_1"]), 1e-17)
    ax2.semilogy(ang, pb, ":", color="0.45", lw=1.3, marker="s", ms=2.0,
                 label=r"$|\{z,v_z\}-1|$ (AD only)")
    ax2.set_yscale("log")
    for xa in (0.5 * numpy.pi, 1.5 * numpy.pi):
        ax2.axvline(xa, color="0.8", lw=0.8, zorder=0)
    ax2.set_xlabel(r"angle $\theta$")
    ax2.set_xticks([0, 0.5 * numpy.pi, numpy.pi, 1.5 * numpy.pi, 2 * numpy.pi])
    ax2.set_xticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    ax2.set_ylabel(r"$|{\rm AD}-{\rm FD}|\,/\,\max|{\rm AD}|$")
    ax2.set_title(rf"(b) AD vs FD along the $J={data['Jchk']:g}$ torus" if PAPER else
                  rf"(b) AD vs 5-point central FD along the $J={data['Jchk']:g}$ torus" "\n"
                  r"($\sigma$-FD rebuilds the family at $\sigma_0\pm h$)", **_ts)
    ax2.legend(fontsize=_fl(6.0), ncol=2, framealpha=0.92, loc="upper center")
    ax2.set_ylim(1e-17, 1e-4)

    # (c) h-convergence of the FD onto AD
    hs = numpy.asarray(data["hs"])
    for key, lab, col, ls in _QUANT:
        ax3.loglog(hs, numpy.maximum(data["hconv"][key], 1e-17), ls, color=col,
                   marker="o", ms=3, lw=1.1, label=lab)
    ax3.set_xlabel("finite-difference step $h$")
    ax3.set_ylabel(r"$\max_\theta |{\rm AD}-{\rm FD}(h)|\,/\,\max|{\rm AD}|$")
    ax3.set_title("(c) FD convergence onto AD" if PAPER else
                  "(c) the 5-point central FD converges onto AD\n"
                  r"($\propto h^4$) down to its round-off floor", **_ts)
    ax3.legend(fontsize=_fl(6.0), ncol=2, framealpha=0.92, loc="upper center")
    ax3.grid(True, which="both", alpha=0.25)

    # (d) the fit in the (sigma, amp) plane
    _plot_fit(fig, ax4, data["fit"])

    fig.tight_layout()
    return fig


def _plot_fit(fig, ax, ft):
    _ts = {"fontsize": 10} if PAPER else {}
    _fl = (lambda d: 8) if PAPER else (lambda d: d)
    best, cov = numpy.asarray(ft["best"]), numpy.asarray(ft["cov_log"])
    # 1- and 2-sigma (68/95% for 2 parameters) ellipses of (ln sigma, ln amp),
    # mapped to (sigma, amp)
    w, V = numpy.linalg.eigh(cov)
    t = numpy.linspace(0.0, 2.0 * numpy.pi, 200)
    for d2, ls in ((2.30, "-"), (6.18, "--")):
        e = (V @ (numpy.sqrt(d2 * w)[:, None] * numpy.array([numpy.cos(t), numpy.sin(t)])))
        ax.plot(best[0] * numpy.exp(e[0]), best[1] * numpy.exp(e[1]), ls,
                color="#0072B2", lw=1.1)
    path = numpy.asarray(ft["path"])
    ax.plot(path[:, 0], path[:, 1], "-o", color="#D55E00", ms=3.5, lw=1.0,
            label=f"LM path ({len(path) - 1} steps)", zorder=3)
    ax.plot(*ft["start"], "s", color="#D55E00", ms=6, mfc="w", zorder=4, label="start")
    ax.plot(*ft["true"], "*", color="k", ms=11, zorder=5, label="truth")
    ax.plot(*best, "o", color="#0072B2", ms=5, zorder=5,
            label=r"best fit, 68/95% contours")
    es = numpy.sqrt(numpy.diag(cov))
    ax.text(0.97, 0.97,
            rf"$\sigma={best[0]:.3f}\pm{best[0] * es[0]:.3f}$" "\n"
            rf"amp$\,={best[1]:.3f}\pm{best[1] * es[1]:.3f}$" "\n"
            rf"$\chi^2={ft['chi2'][-1]:.1f}$ for {ft['ndof']} dof",
            transform=ax.transAxes, ha="right", va="top", fontsize=_fl(7.5),
            bbox=dict(fc="w", ec="0.8", alpha=0.9))
    ax.set_xlabel(r"$\sigma$")
    ax.set_ylabel(r"amp")
    ax.set_title("(d) Levenberg-Marquardt fit" if PAPER else
                 "(d) fitting the disk from noisy $(z, v_z)$ on 3 tori:\n"
                 r"Levenberg-Marquardt on the AD Jacobian ($\partial_\sigma$, "
                 r"$\partial_{\rm amp}$, $\partial_{\theta_i}$)", **_ts)
    ax.legend(fontsize=_fl(7), loc="lower left", framealpha=0.92)
    # inset: the data and the fitted tori
    ins = ax.inset_axes([0.62, 0.36, 0.36, 0.34])
    xd, vd, n = numpy.asarray(ft["xd"]), numpy.asarray(ft["vd"]), ft["np"]
    cols = ["#009E73", "#CC79A7", "#E69F00"]
    for k, (tx, tv) in enumerate(ft["fit_tori"]):
        ins.plot(tx, tv, "-", color=cols[k], lw=0.9)
        ins.plot(xd[k * n:(k + 1) * n], vd[k * n:(k + 1) * n], ".", color=cols[k], ms=2.2)
    ins.set_xticks([])
    ins.set_yticks([])
    _js = ', '.join(f'{j:g}' for j in ft['J'])
    ins.set_title((rf"{3 * n} stars" "\n" rf"$J={_js}$") if PAPER
                  else rf"{3 * n} stars, $J={_js}$", fontsize=_fl(6.0), pad=2)
