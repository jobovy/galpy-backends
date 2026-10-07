"""GS1 -- differentiable stream tracks: d(stream track)/d(halo flattening q).

The streamdf stream track is now assembled through the backend layer -- auxiliary
orbit integration, AD ``calcaAJac``, and a ``jax.lax.map`` over the track chunks --
so ``jax.grad``/``jax.jvp`` propagate straight through a *stream model* to the
potential parameters. This is the capability figure for that stack.

(a) The track itself, and what the derivative predicts. The solid curve is the
    stream track for a LogarithmicHalo with flattening q0; the dashed curves are
    the first-order AD prediction  track(q0) + dq * d(track)/dq  for +-dq, i.e. the
    derivative used the way a fitter would use it.

(b) The cross-check, on the figure rather than in a footnote. Every track
    component's AD derivative is compared against a central finite difference of
    the SAME backend track. Agreement is plotted per component, so the gradient
    the figure shows carries its own evidence.

Why an explicit sequential loop of ``jax.grad``: the orbit path is a
``jax.custom_vjp`` (reverse-mode only -- forward-mode ``jvp`` raises), so the
derivative of each shown component is taken with a separate, unbatched
``jax.grad``, and the chunk assembly uses ``jax.lax.map`` rather than
``jax.vmap``.

That is NOT because jacrev cannot be vmapped -- it can; measured 2026-08-19,
pure-JAX ``vmap(jacrev)`` under an outer ``grad`` agrees to 1e-10, and so does
galpy's own action-angle map when the potential parameter is a plain float
(4.5e-7). The real constraint is the ODE solver: ``orbit_ode`` integrates with an
ADAPTIVE step-size controller, and the second-order path this figure needs
(an outer d/dq through an inner ``jacrev``) requires diffrax's ``DirectAdjoint``,
which differentiates the solver's internal operations. Differentiating through
ADAPTIVELY CHOSEN steps is what corrupts the gradient: under ``jax.vmap`` batch
elements take different step sequences and the batched reverse pass is wrong by
2-13% (growing with integration time), while the forward value stays right to
1e-10 -- which is exactly why this was mis-diagnosed for so long as a jacrev
batching bug. Forcing identical steps (``ConstantStepSize``) restores agreement
to ~3e-7. Sequential ``lax.map`` sidesteps it because the steps are never
batched, which is what makes the numbers in panel (b) trustworthy.

Reproducibility: the track evaluations are expensive (~100 s each), so the
*measure* step writes data/gS1_stream_track_sensitivity.json and the *plot* step
is byte-stable from that JSON. Re-measure by deleting the JSON and re-running.
"""

import json
import os

import numpy

from paper.figures._harness.figure import figure

_DATA = os.path.join(os.path.dirname(__file__), "data")
_JSON = os.path.join(_DATA, "gS1_stream_track_sensitivity.json")

_Q0 = 0.9
_NCH = 3
_TINTJ = 8
_DELTA = 0.5
_H = 1e-3          # central-FD step in q
_DQ_SHOW = 0.02    # flattening perturbation drawn in panel (a)
# journal version (default): no suptitle, short panel titles, no stamps; the
# caption carries the explanations. GALPY_JAX_PAPER=0 restores the annotated figure.
PAPER = os.environ.get("GALPY_JAX_PAPER", "1") == "1"

_C_TRACK, _C_PLUS, _C_MINUS = "#0072B2", "#D55E00", "#009E73"


def _measure():
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from galpy.actionAngle import actionAngleIsochroneApprox
    from galpy.df import streamdf
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential
    from galpy.util import conversion

    _IC = [1.56148083, 0.35081535, -1.15481504, 0.88719443, -0.47713334, 0.12019596]
    tdisrupt = 4.5 / conversion.time_in_Gyr(220.0, 8.0)

    def _numpy_sdf(q):
        lp = LogarithmicHaloPotential(normalize=1.0, q=q)
        aA = actionAngleIsochroneApprox(pot=lp, b=0.8, tintJ=_TINTJ)
        return streamdf(
            0.365 / 220.0, progenitor=Orbit(numpy.array(_IC)), pot=lp, aA=aA,
            leading=True, nTrackChunks=_NCH, nTrackIterations=0,
            deltaAngleTrack=_DELTA, tdisrupt=tdisrupt, nospreadsetup=True,
            useInterp=False, interpTrack=False,
        )

    sdf = _numpy_sdf(_Q0)
    prog = numpy.asarray(sdf._progenitor.vxvv[0], dtype=float)

    def track(q, direct):
        """ObsTrack (nTrackChunks, 6) as a differentiable function of q.

        ``direct`` selects the diffrax adjoint: 'direct' scans a FIXED number of
        steps and is the only one reverse-differentiable here (the default
        recursive adjoint uses a dynamic-bound while_loop, which jax.grad
        rejects); the default is used for the finite-difference arm.
        """
        ikw = ({"adjoint": "direct", "max_steps": 20000} if direct
               else {"max_steps": 200000})
        lp = LogarithmicHaloPotential(normalize=1.0, q=q)
        sdf._pot = lp
        sdf._aA = actionAngleIsochroneApprox(
            pot=lp, b=0.8, tintJ=_TINTJ, integrate_method="diffrax",
            integrate_kwargs=ikw,
        )
        pb = Orbit(jnp.stack([jnp.asarray(v) for v in prog]))
        pb.turn_physical_off()
        sdf._progenitor = pb
        sdf._determine_stream_track_backend()
        return sdf._ObsTrack

    # R (col 0) and z (col 3) per chunk: 2*nch separate, UNBATCHED reverse-mode
    # gradients. No jacrev (it vmaps the backward pass) and no vmap anywhere.
    val = track(jnp.asarray(_Q0), True)
    cols = (0, 3)
    dad = numpy.zeros((_NCH, 6))
    for ci in cols:
        for k in range(_NCH):
            g = jax.grad(lambda q, ci=ci, k=k: track(q, True)[k, ci])(jnp.asarray(_Q0))
            dad[k, ci] = float(g)
    tp = numpy.asarray(track(jnp.asarray(_Q0 + _H), False), dtype=float)
    tm = numpy.asarray(track(jnp.asarray(_Q0 - _H), False), dtype=float)
    dfd = (tp - tm) / (2.0 * _H)

    return {
        "q0": _Q0, "h": _H, "dq_show": _DQ_SHOW,
        "track": numpy.asarray(val, dtype=float).tolist(),
        "d_ad": dad.tolist(),
        "cols_measured": [0, 3],
        "d_fd": dfd.tolist(),
    }


@figure(
    name="gS1_stream_track_sensitivity",
    track="G",
    seed=0,
    description="differentiable stream track: d(track)/d(halo q) by AD (jax.jvp) with an on-figure finite-difference cross-check [measure->json, plot byte-stable]",
)
def make():
    import matplotlib.pyplot as plt

    if os.path.exists(_JSON):
        data = json.load(open(_JSON))
    else:
        data = _measure()
        os.makedirs(_DATA, exist_ok=True)
        with open(_JSON, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")

    tr = numpy.array(data["track"])
    ad = numpy.array(data["d_ad"])
    fd = numpy.array(data["d_fd"])
    dq = data["dq_show"]

    # PAPER: final printed width of a figure* (7.0 in); text sizes are then the
    # printed sizes (9-10 pt labels/ticks, 10 pt titles, 8 pt legends).
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 3.3) if PAPER else (9.8, 4.0))
    _ts = {"fontsize": 10} if PAPER else {}
    _fs_leg = 8 if PAPER else 7

    # (a) track in the (R, z) plane + first-order AD prediction for +-dq
    R, z = tr[:, 0], tr[:, 3]
    ax1.plot(R, z, "-o", color=_C_TRACK, ms=4, lw=1.8, label=f"track, $q={data['q0']}$", zorder=3)
    ax1.plot(R + dq * ad[:, 0], z + dq * ad[:, 3], "--s", color=_C_PLUS, ms=3, lw=1.2,
             label=rf"AD: $q+{dq}$", zorder=2)
    ax1.plot(R - dq * ad[:, 0], z - dq * ad[:, 3], "--^", color=_C_MINUS, ms=3, lw=1.2,
             label=rf"AD: $q-{dq}$", zorder=2)
    ax1.set_xlabel(r"$R$"); ax1.set_ylabel(r"$z$")
    ax1.set_title("(a) track and AD prediction" if PAPER
                  else r"stream track and its AD sensitivity to halo $q$", **_ts)
    ax1.legend(fontsize=_fs_leg, framealpha=0.92)

    # (b) the cross-check: AD vs central FD, per track component
    lbl = [r"$R$", r"$v_R$", r"$v_T$", r"$z$", r"$v_z$", r"$\phi$"]
    rel = numpy.abs(ad - fd) / numpy.maximum(numpy.abs(fd), 1e-30)
    for j in data.get("cols_measured", [0, 3]):
        ax2.semilogy(numpy.arange(tr.shape[0]), rel[:, j], "o-", ms=4, lw=1.0, label=lbl[j])
    ax2.set_xlabel("track chunk"); ax2.set_ylabel(r"$|{\rm AD}-{\rm FD}|/|{\rm FD}|$")
    ax2.set_title("(b) AD vs finite difference" if PAPER
                  else rf"AD vs central FD ($h={data['h']}$)", **_ts)
    ax2.set_xticks(numpy.arange(tr.shape[0]))
    if PAPER:  # the y-range spans < 1 decade: label the minor ticks too
        from matplotlib.ticker import LogFormatterSciNotation
        ax2.yaxis.set_minor_formatter(LogFormatterSciNotation(minor_thresholds=(2, 0.5)))
        ax2.tick_params(axis="y", which="minor", labelsize=9)
    ax2.legend(fontsize=_fs_leg, ncol=2, framealpha=0.92)
    ax2.grid(True, which="both", ls=":", lw=0.5, alpha=0.5)

    fig.tight_layout()
    return fig
