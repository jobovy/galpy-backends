"""GS2 -- autodiff through a PARTICLE-SPRAY stream: d(stream)/d(halo flattening q).

gS1 differentiates the streamdf *track* (progenitor orbit -> action-angle -> track
chunks). This figure differentiates the other kind of stream model: streamspraydf,
which sprays individual particles off the progenitor at stripping times drawn from
a distribution and integrates each one. The gradient therefore flows through the
sampler AND every particle's orbit integration.

(a) The stream itself at two flattenings, from the SAME random key. Because the key
    is fixed (common random numbers), the two panels differ only through the
    physics, not through resampling -- which is exactly what makes a derivative of
    a Monte-Carlo stream meaningful.

(b) Why the gradient is trustworthy, on the figure rather than in a footnote. A
    single AD-vs-FD number would be weak evidence for a stochastic estimator, so
    this plots the finite difference CONVERGING onto the autodiff value as the step
    h shrinks, for several particle counts. AD is the h->0 limit; the large-h
    disagreement is finite-difference truncation, not a defect in the derivative.

Why the raw-array path: ``sample(return_orbit=True)`` wraps its output via
``as_numpy(out).T``, which severs the gradient at the Orbit boundary (a separate,
still-open item), so ``return_orbit=False`` is what carries the derivative today.

REQUIRES the streamspraydf theta-probe fix (galpy#1346). Before it, ``_sample_setup``
skipped the backend-parameter probe whenever the progenitor IC was a backend array,
so the progenitor integrated through the C-STM -- which carries d/d(IC) but NOT
d/d(theta) -- and jax.grad w.r.t. q died in the C parser. Measuring this figure
against a galpy without that fix raises TracerArrayConversionError rather than
silently producing a wrong number.

Reproducibility: the sampling+gradient sweep is the expensive part (~10 min), so
measure->json and the plot is byte-stable from it. Re-measure by deleting the JSON.
"""

import json
import os

import numpy

from paper.figures._harness.figure import figure

_DATA = os.path.join(os.path.dirname(__file__), "data")
_JSON = os.path.join(_DATA, "gS2_streamspray_sensitivity.json")

_IC = [1.0, 0.1, 1.1, 0.05, 0.03, 0.2]
_RO, _VO = 8.0, 220.0
_Q0 = 0.9
_DQ_SHOW = 0.05        # flattening difference between the two drawn streams
_NSHOW = 400           # particles in the morphology panel
_NS = (60, 120, 240)   # particle counts for the convergence panel
_HS = (1e-2, 1e-3, 1e-4, 1e-5)
_SEED = 20260707
# journal version (default): no suptitle, short panel titles, no stamps; the
# caption carries the explanations. GALPY_JAX_PAPER=0 restores the annotated figure.
PAPER = os.environ.get("GALPY_JAX_PAPER", "1") == "1"


def _measure():
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from galpy.backend import random as grandom
    from galpy.df import fardal15spraydf
    from galpy.orbit import Orbit
    from galpy.potential import LogarithmicHaloPotential
    from galpy.util import conversion

    mass = 2 * 10.0**4.0 / conversion.mass_in_msol(_VO, _RO)
    td = 2.0 / conversion.time_in_Gyr(_VO, _RO)

    def _spray(q, n, key):
        pot = LogarithmicHaloPotential(normalize=1.0, q=q)
        prog = Orbit(jnp.stack([jnp.asarray(float(v)) for v in _IC]))
        prog.turn_physical_off()
        spdf = fardal15spraydf(mass, progenitor=prog, pot=pot, tdisrupt=td)
        return spdf.sample(n=n, integrate=True, key=key, return_orbit=False)

    def stat(q, n, key):
        return jnp.mean(_spray(q, n, key)[0] ** 2)

    # (a) two streams, SAME key -> the difference is physics, not resampling
    key_show = grandom.key(_SEED, backend="jax")
    streams = {}
    for lab, q in (("q0", _Q0), ("q1", _Q0 + _DQ_SHOW)):
        out = _spray(jnp.asarray(q), _NSHOW, key_show)
        R, phi = numpy.asarray(out[0], dtype=float), numpy.asarray(out[5], dtype=float)
        streams[lab] = {
            "x": (R * numpy.cos(phi)).tolist(),
            "y": (R * numpy.sin(phi)).tolist(),
            "q": float(q),
        }

    # (b) FD -> AD convergence
    conv = []
    for n in _NS:
        key = grandom.key(_SEED, backend="jax")
        ad = float(jax.grad(lambda q: stat(q, n, key))(jnp.asarray(_Q0)))
        rels = []
        for h in _HS:
            fd = (
                float(stat(jnp.asarray(_Q0 + h), n, key))
                - float(stat(jnp.asarray(_Q0 - h), n, key))
            ) / (2 * h)
            rels.append(abs(ad - fd) / abs(ad))
        conv.append({"n": n, "ad": ad, "rel": rels})

    return {"streams": streams, "conv": conv, "hs": list(_HS), "q0": _Q0,
            "dq_show": _DQ_SHOW, "n_show": _NSHOW, "seed": _SEED}


_C0, _C1 = "#1d4ed8", "#c2410c"


@figure(
    name="gS2_streamspray_sensitivity",
    track="G",
    seed=_SEED,
    description="autodiff through a particle-spray stream: d(stream)/d(halo q), with the finite difference shown converging onto the AD value (CPU float64) [measure->json, plot byte-stable]",
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

    # Plotting the two streams on top of each other is useless here: with the key
    # held fixed the q=0.90 and q=0.95 clouds differ by ~0.003 in position and sit
    # exactly on top of one another. What is worth seeing is WHERE the stream
    # responds, so colour each particle by its own sensitivity |dr/dq|, obtained
    # from the same two common-random-number draws (finite difference over dq, so
    # this is a first-order sensitivity, not the AD gradient itself).
    s0, s1 = data["streams"]["q0"], data["streams"]["q1"]
    x0, y0 = numpy.array(s0["x"]), numpy.array(s0["y"])
    dq = s1["q"] - s0["q"]
    dr = numpy.hypot(numpy.array(s1["x"]) - x0, numpy.array(s1["y"]) - y0) / dq
    sc = ax1.scatter(x0, y0, c=dr, s=7, cmap="magma_r", linewidths=0)
    cb = fig.colorbar(sc, ax=ax1, pad=0.02)
    cb.set_label(r"$|{\rm d}\mathbf{r}/{\rm d}q|$", fontsize=9)
    cb.ax.tick_params(labelsize=7)
    ax1.set_xlabel("$x$")
    ax1.set_ylabel("$y$")
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.set_title("(a) per-particle sensitivity" if PAPER
                  else rf"per-particle sensitivity to $q$ ($N={data['n_show']}$)")

    hs = numpy.array(data["hs"])
    for i, row in enumerate(data["conv"]):
        ax2.loglog(hs, row["rel"], "o-", ms=4.5, lw=1.4,
                   color=plt.cm.viridis(0.15 + 0.35 * i),
                   label=rf"$N={row['n']}$")
    ax2.set_xlabel("finite-difference step $h$")
    ax2.set_ylabel(r"$|{\rm AD}-{\rm FD}(h)|\,/\,|{\rm AD}|$")
    ax2.set_title("(b) FD convergence to AD" if PAPER
                  else "the FD converges onto the AD value")
    ax2.legend(fontsize=7, framealpha=0.92)
    ax2.grid(True, alpha=0.25)

    fig.tight_layout()
    return fig
