"""GD1 -- differentiable distribution functions: the Jeans equations and the
spherical DFs.

Track F made ``galpy.df`` backend-agnostic. This figure is the "derivatives
show-off" for the DF layer, in the same style as G18 for action-angle: autodiff
straight through a DF quantity, validated against a central finite difference of
the same quantity.

(a) *Jeans.* ``jeans.sigmar`` solves the spherical Jeans equation by quadrature.
    Sweeping the Hernquist scale radius a, we plot sigma_r(a) at fixed r together
    with the autodiff sensitivity d(sigma_r)/da (jax.grad THROUGH the quadrature),
    with the finite difference overplotted as open circles.

(b) *Spherical DF.* ``isotropicHernquistdf`` evaluates f(E) built from the
    potential; we sweep the potential amplitude and show f together with
    d f / d(amp) by autodiff, again against a finite difference.

Note on dispatch, which is the practical lesson for users: these functions
dispatch on the COORDINATE, not on the parameter. Passing r (or E) as a plain
float sends the call down the scipy/numpy path and a traced parameter then dies
with a ConcretizationTypeError; passing it as a backend array is what selects the
differentiable path.

Reproducibility: cheap enough to compute inline, but cached anyway --
data/gD1_df_sensitivity.json holds the measured curves and the plot is
byte-stable from it. Re-measure by deleting the JSON.
"""

import json
import os

import numpy

from paper.figures._harness.figure import figure

_DATA = os.path.join(os.path.dirname(__file__), "data")
_JSON = os.path.join(_DATA, "gD1_df_sensitivity.json")

_R_EVAL = 2.0        # radius at which sigma_r is evaluated
_A_GRID = numpy.linspace(0.8, 2.0, 13)
_AMP_GRID = numpy.linspace(1.4, 2.6, 13)
_E_EVAL = -0.4
_H = 1e-5            # central-FD step (the AD/FD agreement bottoms out near here)

_C_VAL, _C_DER = "#0072B2", "#56B4E9"
_C_VAL2, _C_DER2 = "#D55E00", "#E69F00"


def _measure():
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from galpy.df import isotropicHernquistdf, jeans
    from galpy.potential import HernquistPotential

    _CPU = jax.devices("cpu")[0]

    # --- (a) Jeans: sigma_r(r=2) as a function of the Hernquist scale radius a.
    # r MUST be a backend array: jeans.sigmar dispatches on the coordinate, and a
    # plain float would route to scipy.quad and reject the traced parameter.
    def sig_r(a):
        hp = HernquistPotential(amp=2.0, a=a)
        return jeans.sigmar(hp, jnp.asarray(_R_EVAL), use_physical=False)

    sig, dsig_ad, dsig_fd = [], [], []
    for a in _A_GRID:
        aa = jax.device_put(jnp.asarray(float(a)), _CPU)
        sig.append(float(sig_r(aa)))
        dsig_ad.append(float(jax.grad(sig_r)(aa)))
        dsig_fd.append((float(sig_r(aa + _H)) - float(sig_r(aa - _H))) / (2.0 * _H))

    # --- (b) spherical DF: f(E) of the isotropic Hernquist DF vs potential amp.
    # The single-argument call form takes a TUPLE (E, L, Lz).
    def fE(amp):
        hp = HernquistPotential(amp=amp, a=1.3)
        return isotropicHernquistdf(pot=hp)((jnp.asarray(_E_EVAL),))

    f, df_ad, df_fd = [], [], []
    for m in _AMP_GRID:
        mm = jax.device_put(jnp.asarray(float(m)), _CPU)
        f.append(float(fE(mm)))
        df_ad.append(float(jax.grad(fE)(mm)))
        df_fd.append((float(fE(mm + _H)) - float(fE(mm - _H))) / (2.0 * _H))

    return {
        "r_eval": _R_EVAL, "E_eval": _E_EVAL, "h": _H,
        "a_grid": _A_GRID.tolist(), "sigma_r": sig,
        "dsigma_ad": dsig_ad, "dsigma_fd": dsig_fd,
        "amp_grid": _AMP_GRID.tolist(), "fE": f,
        "df_ad": df_ad, "df_fd": df_fd,
    }


@figure(
    name="gD1_df_sensitivity",
    track="G",
    seed=0,
    description="differentiable DFs: jeans.sigmar and the isotropic Hernquist spherical DF, autodiff sensitivities vs finite difference (CPU float64) [measure->json, plot byte-stable]",
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 4.0))

    # (a) Jeans
    a = numpy.array(data["a_grid"])
    ax1.plot(a, data["sigma_r"], "-", color=_C_VAL, lw=1.8, label=r"$\sigma_r$")
    ax1.set_xlabel(r"Hernquist scale radius $a$")
    ax1.set_ylabel(rf"$\sigma_r(r={data['r_eval']})$", color=_C_VAL)
    ax1.tick_params(axis="y", labelcolor=_C_VAL)
    ax1b = ax1.twinx()
    ax1b.plot(a, data["dsigma_ad"], "-", color=_C_DER, lw=1.6, label=r"$d\sigma_r/da$ (AD)")
    ax1b.plot(a, data["dsigma_fd"], "o", mfc="none", mec="0.25", ms=5, lw=0,
              label="finite difference")
    ax1b.set_ylabel(r"$d\sigma_r/da$", color=_C_DER)
    ax1b.tick_params(axis="y", labelcolor=_C_DER)
    ax1.set_title(r"Jeans: $\sigma_r$ and its AD sensitivity")
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax1b.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=7, loc="best", framealpha=0.92)

    # (b) spherical DF
    m = numpy.array(data["amp_grid"])
    ax2.plot(m, data["fE"], "-", color=_C_VAL2, lw=1.8, label=r"$f(E)$")
    ax2.set_xlabel(r"potential amplitude")
    ax2.set_ylabel(rf"$f(E={data['E_eval']})$", color=_C_VAL2)
    ax2.tick_params(axis="y", labelcolor=_C_VAL2)
    ax2b = ax2.twinx()
    ax2b.plot(m, data["df_ad"], "-", color=_C_DER2, lw=1.6, label=r"$df/d\,$amp (AD)")
    ax2b.plot(m, data["df_fd"], "o", mfc="none", mec="0.25", ms=5, lw=0,
              label="finite difference")
    ax2b.set_ylabel(r"$df/d\,$amp", color=_C_DER2)
    ax2b.tick_params(axis="y", labelcolor=_C_DER2)
    ax2.set_title("isotropic Hernquist DF: $f$ and its AD sensitivity")
    h1, l1 = ax2.get_legend_handles_labels(); h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, fontsize=7, loc="best", framealpha=0.92)

    fig.tight_layout()
    return fig
