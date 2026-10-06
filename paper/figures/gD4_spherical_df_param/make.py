"""GD4 -- derivatives through spherical-DF CONSTRUCTION and SAMPLING.

gD1 differentiates DF quantities of an analytic DF (isotropicHernquistdf f(E))
and the Jeans quadrature; gD2 differentiates a kernel estimate over a FIXED
sample. Neither differentiates the DF *machinery* itself. Here the parameter
enters upstream of everything -- the ODE solve or Eddington-type inversion that
builds the DF, and the inverse-CDF sampler that draws particles from it -- and
the derivative flows through all of it.

Row 1 -- King models: d/dW0 through the King ODE solve.
    ``kingdf(W0, M=1, rt=10)`` solves Poisson's equation for the scale-free
    model W(r) with central potential W0, then rescales to (M, rt). W0 sets the
    SHAPE, so every table depends on it through the ODE solution. We take
    d ln rho(r) / dW0 at fixed (M, rt) on an r grid two ways:
      * eager jax (``jax.jacfwd``; W0 a jax scalar): scipy's DOP853 solve plus
        the forward sensitivity equations grafted on as the derivative;
      * ``jax.jit(jax.jacfwd(...))``: the whole construction is traced, the ODE
        solved in-graph with diffrax and differentiated by AD through the
        discretized solve (a completely different code path).
    (a) rho(r) for W0 = 3, 5, 7, 9; (b) d ln rho / dW0, eager AD (solid) and jit
    AD (dashed) vs a central finite difference that REBUILDS ``kingdf(W0 +- h)``
    on the plain numpy/scipy path (open circles, h = 1e-3); (c) the max-over-r
    |AD - FD| vs h: the V-shape (truncation ~h^2 on the right, ODE-tolerance
    noise ~1/h on the left) bottoming out at ~1e-7 is the signature of a
    correct derivative. W0 = 2 is avoided on purpose: the scipy solve switches its
    inner/outer split radius at W0 = 2 (rbreak = r0/100 below, r0 above), which
    moves the interpolation nodes and makes rho jump by ~5e-7 there.

Row 2 -- differentiable SAMPLING (common random numbers).
    ``osipkovmerrittdf(pot=HernquistPotential(amp=2, a), ra=1.5)`` (M = 1, an
    anisotropic f(Q) built numerically by the Eddington-type inversion) is
    CONSTRUCTED AND SAMPLED inside one ``jax.jit``-ed function of the Hernquist
    scale radius a, with a FIXED jax PRNG key passed as ``sample(key=...)``.
    With the key fixed, every uniform the sampler draws is a constant, and each
    particle's (r_i, v_i) is a deterministic, almost-everywhere smooth function of
    a: the radial inverse-CDF r(u; a), the analytic eta inversion, and the
    bilinear p(v|r) inverse-CDF table built from f(Q; a). ``jax.jvp`` gives
    d(r_i, v_r,i)/da for every particle in one forward pass; the finite difference
    re-runs the SAME jitted construct-and-sample at a +- h with the SAME key.

    The binned estimator. Particles are assigned to radial bins by their radius
    at the nominal a = 1. Because r(u; a) is monotone in u for every a, this is
    a binning in the radial uniform u (i.e., in enclosed-mass fraction), so bin
    membership does not change with a and the binned statistic
    sigma_r,bin(a) = sqrt(<v_r^2>_bin) is differentiable. Its expectation given
    the drawn radii is sqrt(mean_i sigma_r^2(r_i; a)), with sigma_r^2 from the
    Osipkov-Merritt Jeans equation for a Hernquist sphere (an independent scipy
    quadrature, truncated at the DF's rmax = 1e4), and d/da of the expectation
    includes the drift dr_i/da of the drawn radii. (d) sigma_r,bin from the
    sample vs this expectation; (e) d sigma_r,bin / da: jvp through the sample
    (filled), FD of the sample with the same key (crosses), expectation (open
    squares); error bars are the scatter of 16 independent keys / sqrt(16).
    (f) AD vs FD vs h: per-particle (median and 99.9th percentile of
    |FD - AD| / rms(AD)) and for the binned d sigma_r/da (max over bins).

    What (e) shows, and what it does not. AD == FD of the sample to ~1e-8 is a
    statement about the pathwise derivative of THIS sample (exact, no noise);
    sample vs expectation agrees to within MC noise, which is the statement that
    the pathwise (reparameterization) derivative is an unbiased estimator of
    d E[sigma_r] / da. A few particles in 1e3 sit within ~1e-5 of a kink of the
    piecewise-linear inverse-CDF tables (a CDF knot crossing a tabulated u level
    as a changes): there the FD straddles the kink while AD gives the one-sided
    derivative (at h = 1e-6 the FD converges onto AD for them too), which is
    what lifts the 99.9th percentile in (f) at small h. The kinks average out in
    the binned estimator.

Reproducibility: the jit compiles (~20 s each) and the 16 x (1 jvp + 22 FD)
construct-and-sample calls dominate (~10 min on 4 cores), so measure->json and
the plot is byte-stable from data/gD4_spherical_df_param.json. Re-measure by
deleting the JSON. CPU float64 (``jax_enable_x64``).
"""

import json
import os

import numpy

from paper.figures._harness.figure import figure

_DATA = os.path.join(os.path.dirname(__file__), "data")
_JSON = os.path.join(_DATA, "gD4_spherical_df_param.json")

# --- King ----------------------------------------------------------------------
_W0S = (3.0, 5.0, 7.0, 9.0)
_KM, _KRT = 1.0, 10.0
_KR = numpy.geomspace(1e-3 * _KRT, 0.9 * _KRT, 80)
_HS_K = (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5)
_H_K_SHOW = 1e-3

# --- Osipkov-Merritt sampling ---------------------------------------------------
_A0, _RA, _M = 1.0, 1.5, 1.0
_RMAX = 1e4  # osipkovmerrittdf's default rmax
_NCH, _NPER = 16, 250000
_SEED = 20261005
_EDGES = numpy.geomspace(0.05, 30.0, 13)
_HS_S = (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6)
_H_S_SHOW = 1e-4
_NPP = 4  # chunks used for the per-particle error statistics


def _measure_king():
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import torch

    from galpy.backend import use
    from galpy.df import kingdf

    def lndens_jax(W0):
        with use("jax", force=True):
            return jnp.log(kingdf(W0=W0, M=_KM, rt=_KRT).dens(jnp.asarray(_KR)))

    def lndens_np(W0):
        return numpy.log(numpy.asarray(kingdf(W0=W0, M=_KM, rt=_KRT).dens(_KR)))

    def lndens_torch(W):
        with use("torch", force=True):
            d = kingdf(W0=W, M=_KM, rt=_KRT)
            return torch.log(d.dens(torch.as_tensor(_KR, dtype=torch.float64)))

    jit_jac = jax.jit(jax.jacfwd(lndens_jax))
    out = {"r": _KR.tolist(), "W0s": list(_W0S), "hs": list(_HS_K),
           "h_show": _H_K_SHOW, "M": _KM, "rt": _KRT, "W0": {}}
    for W0 in _W0S:
        lnr = lndens_np(W0)
        ad = numpy.asarray(jax.jacfwd(lndens_jax)(W0))
        adj = numpy.asarray(jit_jac(W0))
        Wt = torch.tensor(W0, dtype=torch.float64, requires_grad=True)
        adt = torch.autograd.functional.jacobian(lndens_torch, Wt).detach().numpy()
        fds = {h: (lndens_np(W0 + h) - lndens_np(W0 - h)) / (2.0 * h) for h in _HS_K}
        out["W0"][str(W0)] = {
            "lnrho": lnr.tolist(),
            "ad_eager": ad.tolist(),
            "ad_jit": adj.tolist(),
            "fd_show": fds[_H_K_SHOW].tolist(),
            "err_eager": [float(numpy.max(numpy.abs(fds[h] - ad))) for h in _HS_K],
            "err_jit": [float(numpy.max(numpy.abs(fds[h] - adj))) for h in _HS_K],
            "jit_vs_eager": float(numpy.max(numpy.abs(adj - ad))),
            "torch_vs_eager": float(numpy.max(numpy.abs(adt - ad))),
        }
        print(f"King W0={W0}: min_h err eager {min(out['W0'][str(W0)]['err_eager']):.2e}"
              f" jit {min(out['W0'][str(W0)]['err_jit']):.2e}"
              f" jit-vs-eager {out['W0'][str(W0)]['jit_vs_eager']:.1e}"
              f" torch-vs-eager {out['W0'][str(W0)]['torch_vs_eager']:.1e}", flush=True)
    return out


def _jeans_tables():
    """OM-Hernquist Jeans sigma_r^2(r; a) on a log-r grid at a0 and a0 +- da."""
    from scipy import integrate, interpolate

    def sig2(r, a):
        rho = lambda x: _M * a / (2.0 * numpy.pi * x * (x + a) ** 3)
        f = lambda x: rho(x) * (x * x + _RA * _RA) * _M / (x + a) ** 2
        I = integrate.quad(f, r, _RMAX, epsabs=0.0, epsrel=1e-12, limit=500)[0]
        return I / (rho(r) * (r * r + _RA * _RA))

    lr = numpy.linspace(-3.5, 2.5, 301)
    da = 1e-4
    tab = {s: numpy.array([sig2(10.0**x, _A0 + s * da) for x in lr]) for s in (-1, 0, 1)}
    S = interpolate.CubicSpline(lr, tab[0])
    dSda = interpolate.CubicSpline(lr, (tab[1] - tab[-1]) / (2.0 * da))
    return S, dSda


def _measure_sampling():
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from galpy.backend import random as grandom
    from galpy.backend import use
    from galpy.df import osipkovmerrittdf
    from galpy.potential import HernquistPotential

    def construct_and_sample(a, key):
        # the DF is BUILT from the potential inside the differentiated function
        with use("jax", force=True):
            df = osipkovmerrittdf(
                pot=HernquistPotential(amp=2.0 * _M, a=a), ra=_RA, rmin=0.0
            )
            R, vR, vT, z, vz, phi = df.sample(n=_NPER, key=key, return_orbit=False)
        r = jnp.sqrt(R**2 + z**2)
        return r, (R * vR + z * vz) / r

    f_jit = jax.jit(construct_and_sample)
    jvp_jit = jax.jit(
        lambda a, k: jax.jvp(lambda aa: construct_and_sample(aa, k), (a,), (jnp.asarray(1.0),))
    )
    S, dSda = _jeans_tables()
    nb = len(_EDGES) - 1
    # per chunk, per bin: sum vr^2, sum 2 vr dvr, sum E, sum dE, count, sum r
    acc = numpy.zeros((_NCH, nb, 6))
    accfd = numpy.zeros((_NCH, len(_HS_S), nb))  # sum (vr+^2 - vr-^2)/(2h)
    pp_med = numpy.zeros((_NPP, len(_HS_S)))
    pp_999 = numpy.zeros((_NPP, len(_HS_S)))
    n_kink = numpy.zeros((_NPP, len(_HS_S)))
    a0 = jnp.asarray(_A0)
    for k in range(_NCH):
        key = grandom.key(_SEED + k, "jax")
        (r, vr), (dr, dvr) = jvp_jit(a0, key)
        r, vr, dr, dvr = (numpy.asarray(x) for x in (r, vr, dr, dvr))
        idx = numpy.digitize(r, _EDGES) - 1
        masks = [idx == b for b in range(nb)]
        lx = numpy.log10(r)
        s2 = S(lx)
        ds2 = dSda(lx) + S(lx, 1) / (r * numpy.log(10.0)) * dr
        for b, m in enumerate(masks):
            acc[k, b] = [numpy.sum(vr[m] ** 2), numpy.sum(2 * vr[m] * dvr[m]),
                         numpy.sum(s2[m]), numpy.sum(ds2[m]), m.sum(), numpy.sum(r[m])]
        rms = numpy.sqrt(numpy.mean(dvr**2))
        for j, h in enumerate(_HS_S):
            _, vp = f_jit(jnp.asarray(_A0 + h), key)
            _, vm = f_jit(jnp.asarray(_A0 - h), key)
            vp, vm = numpy.asarray(vp), numpy.asarray(vm)
            for b, m in enumerate(masks):
                accfd[k, j, b] = numpy.sum(vp[m] ** 2 - vm[m] ** 2) / (2.0 * h)
            if k < _NPP:
                e = numpy.abs((vp - vm) / (2.0 * h) - dvr) / rms
                pp_med[k, j] = numpy.median(e)
                pp_999[k, j] = numpy.percentile(e, 99.9)
                n_kink[k, j] = numpy.mean(e > 1e-5)
        print(f"sampling chunk {k + 1}/{_NCH} done", flush=True)

    tot = acc.sum(0)
    n = tot[:, 4]
    v2, dv2, E, dE = (tot[:, i] / n for i in range(4))
    sig, sigE = numpy.sqrt(v2), numpy.sqrt(E)
    dsig, dsigE = dv2 / (2 * sig), dE / (2 * sigE)
    # MC error bars: scatter of the per-key estimates / sqrt(NCH)
    nk = acc[:, :, 4]
    sig_k = numpy.sqrt(acc[:, :, 0] / nk)
    dsig_k = acc[:, :, 1] / nk / (2 * sig_k)
    esig = sig_k.std(0, ddof=1) / numpy.sqrt(_NCH)
    edsig = dsig_k.std(0, ddof=1) / numpy.sqrt(_NCH)
    fd_dsig = accfd.sum(0) / n[None, :] / (2 * sig[None, :])  # (nh, nb)
    err_bin = numpy.max(numpy.abs(fd_dsig - dsig[None, :]) / numpy.abs(dsig[None, :]), axis=1)
    jshow = list(_HS_S).index(_H_S_SHOW)
    # a smooth Jeans sigma_r(r) curve for panel (d)
    rr = numpy.geomspace(_EDGES[0], _EDGES[-1], 120)
    return {
        "a0": _A0, "ra": _RA, "M": _M, "nch": _NCH, "nper": _NPER,
        "edges": _EDGES.tolist(), "r_med": (tot[:, 5] / n).tolist(), "n": n.tolist(),
        "sig": sig.tolist(), "sig_err": esig.tolist(), "sig_exp": sigE.tolist(),
        "dsig": dsig.tolist(), "dsig_err": edsig.tolist(), "dsig_exp": dsigE.tolist(),
        "dsig_fd_show": fd_dsig[jshow].tolist(), "h_show": _H_S_SHOW,
        "hs": list(_HS_S), "err_bin": err_bin.tolist(),
        "pp_med": numpy.median(pp_med, 0).tolist(), "pp_999": numpy.median(pp_999, 0).tolist(),
        "pp_kink_frac": numpy.mean(n_kink, 0).tolist(),
        "jeans_r": rr.tolist(), "jeans_sig": numpy.sqrt(S(numpy.log10(rr))).tolist(),
    }


def _measure():
    return {"king": _measure_king(), "sampling": _measure_sampling()}


_KCOL = ("#0072B2", "#009E73", "#E69F00", "#D55E00")
_C_S, _C_E, _C_FD = "#CC79A7", "#000000", "#0072B2"


@figure(
    name="gD4_spherical_df_param",
    track="F",
    seed=0,
    description="derivatives through spherical-DF construction and sampling: King d/dW0 through the ODE solve (eager graft & jit/diffrax) and Osipkov-Merritt d(sample)/d(scale) with common random numbers vs FD and Jeans (CPU float64) [measure->json, plot byte-stable]",
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
            json.dump(data, fh, indent=2, sort_keys=True)
            fh.write("\n")
    K, S = data["king"], data["sampling"]

    fig, axs = plt.subplots(2, 3, figsize=(13.2, 7.6))
    (ax_a, ax_b, ax_c), (ax_d, ax_e, ax_f) = axs

    # ---------------- King -----------------------------------------------------
    r = numpy.array(K["r"])
    sub = slice(2, None, 5)
    hs = numpy.array(K["hs"])
    for W0, col in zip(K["W0s"], _KCOL):
        d = K["W0"][str(W0)]
        ax_a.loglog(r, numpy.exp(d["lnrho"]), color=col, label=rf"$W_0={W0:g}$")
        ax_b.semilogx(r, d["ad_eager"], "-", color=col, lw=1.6)
        ax_b.semilogx(r, d["ad_jit"], "--", color="k", lw=0.8, dashes=(3, 3))
        ax_b.semilogx(r[sub], numpy.array(d["fd_show"])[sub], "o", mfc="none", mec=col, ms=5)
        ax_c.loglog(hs, d["err_eager"], "o-", color=col, ms=3.5, label=rf"$W_0={W0:g}$, eager")
        ax_c.loglog(hs, d["err_jit"], "s--", color=col, ms=3.5, mfc="none", lw=1.0)
    ax_a.set_xlabel(r"$r$")
    ax_a.set_ylabel(r"$\rho(r)$")
    ax_a.set_title(rf"(a) King models, $M={K['M']:g}$, $r_t={K['rt']:g}$")
    ax_a.legend(fontsize=7.5, loc="lower left")
    ax_a.set_ylim(1e-6, None)
    ax_b.set_xlabel(r"$r$")
    ax_b.set_ylabel(r"$\partial\ln\rho/\partial W_0$ at fixed $(M, r_t)$")
    ax_b.set_title(r"(b) $d/dW_0$ through the King ODE solve")
    ax_b.plot([], [], "-", color="0.3", label=r"AD, eager ($\tt jacfwd$; scipy + sensitivities)")
    ax_b.plot([], [], "--", color="k", lw=0.8, dashes=(3, 3), label=r"AD, $\tt jax.jit$ (diffrax solve)")
    ax_b.plot([], [], "o", mfc="none", mec="0.3", label=rf"FD, rebuilt on numpy ($h={K['h_show']:g}$)")
    ax_b.legend(fontsize=7, loc="lower left", framealpha=0.92)
    ax_c.set_xlabel(r"FD step $h$ in $W_0$")
    ax_c.set_ylabel(r"$\max_r\,|{\rm FD}-{\rm AD}|$ in $\partial\ln\rho/\partial W_0$")
    ax_c.set_title("(c) King: AD vs FD (solid eager, dashed jit)")
    ax_c.legend(fontsize=7, loc="upper center", ncol=2, framealpha=0.92)

    # ---------------- sampling -------------------------------------------------
    rm = numpy.array(S["r_med"])
    ax_d.semilogx(S["jeans_r"], S["jeans_sig"], "-", color="0.7", lw=2.6,
                  label="Jeans (Osipkov-Merritt)")
    ax_d.semilogx(rm, S["sig_exp"], "s", mfc="none", mec=_C_E, ms=6,
                  label="expectation for the drawn radii")
    ax_d.errorbar(rm, S["sig"], yerr=S["sig_err"], fmt="o", color=_C_S, ms=4.5,
                  capsize=2, label=rf"sample ($N={S['nch'] * S['nper'] / 1e6:g}\times10^6$)")
    ax_d.set_xlabel(r"$r$ (bin median)")
    ax_d.set_ylabel(r"$\sigma_r$ in enclosed-mass bins")
    ax_d.set_title(rf"(d) OM Hernquist, $a={S['a0']:g}$, $r_a={S['ra']:g}$")
    ax_d.legend(fontsize=7.5, loc="lower left", framealpha=0.92)

    dsig, dexp, err = (numpy.array(S[k]) for k in ("dsig", "dsig_exp", "dsig_err"))
    ax_e.semilogx(rm, dexp, "s", mfc="none", mec=_C_E, ms=7, label="expectation (Jeans)")
    ax_e.errorbar(rm, dsig, yerr=err, fmt="o", color=_C_S, ms=4.5, capsize=2,
                  label=r"$\tt jax.jvp$ through construct + sample")
    ax_e.semilogx(rm, S["dsig_fd_show"], "x", color=_C_FD, ms=7, mew=1.2,
                  label=rf"FD of the sample, same key ($h={S['h_show']:g}$)")
    pull = (dsig - dexp) / err
    fdv = numpy.array(S["dsig_fd_show"])
    ax_e.set_ylim(-0.175, -0.045)
    ax_e.set_xlim(0.04, 40.0)
    ins = ax_e.inset_axes([0.47, 0.07, 0.5, 0.3])
    ins.axhline(0.0, color="0.6", lw=0.8)
    ins.errorbar(rm, (dsig - dexp) / numpy.abs(dexp) * 100, yerr=err / numpy.abs(dexp) * 100,
                 fmt="o", color=_C_S, ms=3, capsize=1.5, elinewidth=0.9)
    ins.plot(rm, (fdv - dsig) / numpy.abs(dsig) * 100, "x", color=_C_FD, ms=4)
    ins.set_xscale("log")
    ins.set_xlim(0.04, 40.0)
    ins.set_ylim(-1.2, 1.2)
    ins.tick_params(labelsize=6)
    ins.set_ylabel(r"$\Delta$ (%)", fontsize=6.5)
    ins.set_title(f"sample $-$ expectation: |pull| max {numpy.max(numpy.abs(pull)):.1f}, "
                  f"median {numpy.median(numpy.abs(pull)):.1f}", fontsize=6.3)
    eb = numpy.array(S["err_bin"])
    hsS = numpy.array(S["hs"])
    jmin = int(numpy.argmin(eb))
    ax_e.text(0.03, 0.62,
              f"AD vs FD, same key (max over bins):\n"
              f"{eb[list(hsS).index(S['h_show'])]:.0e} at $h$={S['h_show']:g}, "
              f"{eb[jmin]:.0e} at $h$={hsS[jmin]:g}",
              transform=ax_e.transAxes, ha="left", va="top", fontsize=6.8, color="0.25")
    ax_e.set_xlabel(r"$r$ (bin median)")
    ax_e.set_ylabel(r"$d\sigma_r/da$ (bins fixed in enclosed mass)")
    ax_e.set_title(r"(e) $d/da$ through DF construction AND sampling")
    ax_e.legend(fontsize=7, loc="upper left", framealpha=0.92)

    hs = numpy.array(S["hs"])
    ax_f.loglog(hs, S["err_bin"], "o-", color=_C_S, ms=4, label=r"binned $d\sigma_r/da$ (max rel. over bins)")
    ax_f.loglog(hs, S["pp_med"], "s-", color=_C_FD, ms=4, label="per particle, median")
    ax_f.loglog(hs, S["pp_999"], "^--", color=_C_FD, ms=4, mfc="none", label="per particle, 99.9th pct")
    for c, p, lab in ((3e-1, 2, r"$\propto h^2$"), (6e-2, 1, r"$\propto h$")):
        ax_f.loglog(hs, c * hs**p, ":", color="0.6", lw=0.9)
        ax_f.text(hs[0] * 0.9, c * hs[0] ** p, lab, fontsize=6.5, color="0.5", ha="right", va="center")
    ax_f.set_xlabel(r"FD step $h$ in $a$")
    ax_f.set_ylabel(r"$|{\rm FD}-{\rm AD}|$ (per particle: / rms $dv_r/da$)")
    ax_f.set_title("(f) sampling: AD vs FD, same key")
    ax_f.legend(fontsize=7, loc="upper left", framealpha=0.92)

    fig.tight_layout()
    return fig
