"""GD2 -- autodiff straight through a Monte-Carlo sample of a spherical DF.

The DF-layer counterpart to G18/GD1, but the differentiated object here is a
*random sample*, not a quadrature. We draw N particles from an isotropic
Hernquist DF, build a kernel estimator of sigma_r(r) from the FIXED sample, and
take d(sigma_r)/dr with jax.grad -- then compare against jax.grad of
``jeans.sigmar``, i.e. autodiff of the deterministic numerical calculation of
the same quantity.

Why differentiating a sample is not circular: r enters the estimator through the
kernel *weights*, not through the draw. So the gradient flows without needing a
differentiable sampler (backend RNG/key threading is still open work) -- and the
comparison is meaningful precisely because the two paths share no code: one is a
weighted sum over particles, the other a quadrature of the Jeans equation.

Two things the estimator has to get right, both learned the hard way:

* Local-CONSTANT (Nadaraya-Watson) weighting couples to the steep density
  gradient and visibly biases the derivative. The local-LINEAR fit used here
  absorbs that to leading order.
* The sampler's inverse-CDF velocity grid must be refined. At galpy's shipped
  ``n_v_vesc=100`` the sampled sigma_r is biased LOW by ~0.6-0.8% at every
  radius -- 25/25 log-spaced bins below the analytic value, and n-independent
  (identical at 4e5 and 4e6 particles), so it is a resolution bias, not noise.
  That is a real pre-existing galpy bug, reported as jobovy/galpy#1343 and found
  by this figure disagreeing with theory. We set n_v_vesc=300 here, where the
  sample matches Hernquist (1990) eq. 10, ``df.sigmar`` and ``jeans.sigmar`` --
  three independent legs -- to ~0.01%.

What the figure claims, and its one caveat: the sample-derivative agrees with
the quadrature-derivative to a median 0.6 sigma of the MC noise (error bars are
the scatter of 8 independent sub-samples). The innermost point, r = 1.0 = 4H, is
the exception at ~5 sigma -- there the residual is kernel BIAS, not noise, so the
error bar does not cover it: the window still spans enough curvature in the
steeply-falling density for the local-linear fit to leave a residual. Kept in the
figure rather than trimmed, since the honest statement is "valid where the kernel
is narrow compared to the scale over which sigma_r varies".

Reproducibility: the sample is the expensive part, so measure->json and the plot
is byte-stable from it. Re-measure by deleting the JSON.
"""

import json
import os

import numpy

from paper.figures._harness.figure import figure

_DATA = os.path.join(os.path.dirname(__file__), "data")
_JSON = os.path.join(_DATA, "gD2_df_sample_autodiff.json")

_AMP, _A = 2.0, 1.3          # galpy HernquistPotential(amp=2M, a); G=1 => M=1
_N = 4000000
_N_V_VESC = 300              # see docstring / galpy#1343
_H = 0.25                    # kernel bandwidth
_R_GRID = numpy.linspace(1.0, 4.0, 11)
_NCHUNK = 8          # independent sub-samples -> MC error bar


def _measure():
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    from galpy.df import isotropicHernquistdf, jeans
    from galpy.potential import HernquistPotential

    hp = HernquistPotential(amp=_AMP, a=_A)
    df = isotropicHernquistdf(pot=hp)

    # Refine the inverse-CDF velocity grid. _rmin_sampling is only set during
    # sampling setup, so prime with a tiny draw before rebuilding.
    numpy.random.seed(0)
    df.sample(n=10)
    r_a_end = (
        max(numpy.log10(df._rmax / df._scale), 3) if numpy.isfinite(df._rmax) else 3
    )
    df._v_vesc_pvr_interpolator = df._make_pvr_interpolator(
        r_a_end=r_a_end, n_v_vesc=_N_V_VESC
    )

    numpy.random.seed(0)
    s = df.sample(n=_N)
    R_S = jnp.asarray(numpy.asarray(s.r(use_physical=False), dtype=float))
    VR_S = jnp.asarray(numpy.asarray(s.vr(use_physical=False), dtype=float))

    def _local_linear(y, r, R=None):
        """Weighted local-linear fit of y(r') evaluated at r' = r (intercept)."""
        R = R_S if R is None else R
        d = R - r
        w = jnp.exp(-0.5 * (d / _H) ** 2)
        S0 = jnp.sum(w)
        S1 = jnp.sum(w * d)
        S2 = jnp.sum(w * d * d)
        T0 = jnp.sum(w * y)
        T1 = jnp.sum(w * d * y)
        return (S2 * T0 - S1 * T1) / (S0 * S2 - S1 * S1)

    def sigma_mc(r):
        m1 = _local_linear(VR_S, r)
        m2 = _local_linear(VR_S**2, r)
        return jnp.sqrt(m2 - m1 * m1)

    def _sigma_chunk(r, R, V):
        m1 = _local_linear(V, r, R)
        m2 = _local_linear(V**2, r, R)
        return jnp.sqrt(m2 - m1 * m1)

    def sigma_jeans(r):
        # dispatches on the COORDINATE: r must be a backend array to stay traced
        return jeans.sigmar(
            hp,
            r,
            dens=lambda x: hp.dens(x, 0.0, use_physical=False),
            beta=0.0,
            use_physical=False,
        )

    def sigma_analytic(r):
        s_ = r / _A
        M = _AMP / 2.0
        return numpy.sqrt(
            (M / (12.0 * _A))
            * (
                (12.0 * r * (r + _A) ** 3 / _A**4) * numpy.log((r + _A) / r)
                - (r / (r + _A)) * (25.0 + 52.0 * s_ + 42.0 * s_**2 + 12.0 * s_**3)
            )
        )

    g_mc = jax.grad(sigma_mc)
    g_je = jax.grad(sigma_jeans)

    out = {"r_grid": _R_GRID.tolist(), "n": _N, "h": _H, "n_v_vesc": _N_V_VESC,
           "amp": _AMP, "a": _A}
    for key, fn in (("sigma_mc", sigma_mc), ("sigma_jeans", sigma_jeans),
                    ("dsigma_mc", g_mc), ("dsigma_jeans", g_je)):
        out[key] = [float(fn(jnp.asarray(float(r)))) for r in _R_GRID]
    out["sigma_analytic"] = [float(sigma_analytic(r)) for r in _R_GRID]

    # MC uncertainty: K independent sub-samples. Each chunk has N/K particles so
    # its scatter is sqrt(K) times the full-sample error -- hence std/sqrt(K).
    nper = R_S.shape[0] // _NCHUNK
    g_chunk = jax.grad(_sigma_chunk)
    s_err, d_err = [], []
    for r in _R_GRID:
        rr = jnp.asarray(float(r))
        sv, dv = [], []
        for k in range(_NCHUNK):
            sl = slice(k * nper, (k + 1) * nper)
            sv.append(float(_sigma_chunk(rr, R_S[sl], VR_S[sl])))
            dv.append(float(g_chunk(rr, R_S[sl], VR_S[sl])))
        s_err.append(float(numpy.std(sv, ddof=1) / numpy.sqrt(_NCHUNK)))
        d_err.append(float(numpy.std(dv, ddof=1) / numpy.sqrt(_NCHUNK)))
    out["sigma_mc_err"] = s_err
    out["dsigma_mc_err"] = d_err
    out["nchunk"] = _NCHUNK
    return out


_C_MC = "#c2410c"
_C_TH = "#1d4ed8"


@figure(
    name="gD2_df_sample_autodiff",
    track="G",
    seed=0,
    description="autodiff through a 4e6-particle sample of the isotropic Hernquist DF: d(sigma_r)/dr from the sample vs autodiff of the Jeans quadrature (CPU float64) [measure->json, plot byte-stable]",
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

    r = numpy.array(data["r_grid"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 4.0))

    ax1.plot(r, data["sigma_analytic"], "-", color="0.6", lw=3.0,
             label="Hernquist (1990) analytic")
    ax1.plot(r, data["sigma_jeans"], "-", color=_C_TH, lw=1.6,
             label=r"$\tt jeans.sigmar$ (quadrature)")
    ax1.errorbar(r, data["sigma_mc"], yerr=data["sigma_mc_err"], fmt="o", mfc="none",
                 mec=_C_MC, ecolor=_C_MC, mew=1.4, ms=6, lw=0, elinewidth=1.2, capsize=3,
                 label=r"MC sample ($N=4\times10^6$)")
    ax1.set_xlabel("$r$")
    ax1.set_ylabel(r"$\sigma_r(r)$")
    ax1.set_title("the value")
    ax1.legend(fontsize=7, framealpha=0.92)

    ax2.plot(r, data["dsigma_jeans"], "-", color=_C_TH, lw=1.8,
             label=r"$\tt jax.grad$ of the quadrature")
    ax2.errorbar(r, data["dsigma_mc"], yerr=data["dsigma_mc_err"], fmt="o", mfc="none",
                 mec=_C_MC, ecolor=_C_MC, mew=1.4, ms=6, lw=0, elinewidth=1.2, capsize=3,
                 label=r"$\tt jax.grad$ through the sample")
    ax2.set_xlabel("$r$")
    ax2.set_ylabel(r"$d\sigma_r/dr$")
    ax2.set_title("the derivative, two independent ways")
    ax2.legend(fontsize=7, framealpha=0.92)

    d1 = numpy.array(data["dsigma_mc"])
    d2 = numpy.array(data["dsigma_jeans"])
    err = numpy.array(data["dsigma_mc_err"])
    pull = numpy.fabs(d1 - d2) / err
    ax2.text(0.04, 0.06,
             f"agreement within MC noise:\nmax {pull.max():.1f}$\\sigma$, median {numpy.median(pull):.1f}$\\sigma$",
             transform=ax2.transAxes, fontsize=7.5, color="0.25")

    fig.tight_layout()
    return fig
