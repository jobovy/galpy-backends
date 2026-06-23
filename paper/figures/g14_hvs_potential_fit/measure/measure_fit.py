"""Measure: the g14 HVS hybrid fit (C-STM IC fit + in-backend AD potential fit).

Runs the skeleton's two fits on a fixed seeded S5-HVS1-like mock and saves
everything make.py needs: the obs vs fitted backward orbits (x,y), the C-STM
IC-fit loss history, the in-backend q-loss landscape (grid) + Adam path, and the
recovered values. CPU + float64 + fixed seed -> deterministic, so the plot is
byte-stable. Run:  JAX_PLATFORMS=cpu PYTHONPATH=<#972-worktree> python measure/measure_fit.py
"""

import json
import pathlib

import numpy

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from skeleton import (  # noqa: E402
    cstm_integrate, fit_ic_cstm, fit_q_inbackend, mock_s5hvs1, potential, q_loss_curve,
    closest_r2_cstm,
)

Q_TRUE = 0.85
TS_FWD = numpy.linspace(0.0, 0.50, 150)
TS_BACK = numpy.linspace(0.0, -0.50, 150)
SIGMA = numpy.array([0.03, 0.05, 0.05, 0.02, 0.05, 0.02])


def _xy(ic, q):  # backward orbit in the disc plane (x,y), Orbit order -> Cartesian
    orb = numpy.asarray(cstm_integrate(potential(q), jnp.asarray(ic), TS_BACK, method="dop853_c"))
    R, phi = orb[:, 0], orb[:, 5]
    return (R * numpy.cos(phi)).tolist(), (R * numpy.sin(phi)).tolist()


truth = mock_s5hvs1(Q_TRUE, TS_FWD)
rng = numpy.random.default_rng(1)
obs = truth + rng.normal(size=6) * SIGMA

# (1) C-STM IC fit (potential fixed at the truth)
ic_fit, ic_hist = fit_ic_cstm(obs, Q_TRUE, TS_BACK, SIGMA)
ic_loss = [h[1] for h in ic_hist]  # closest-approach^2 per Gauss-Newton iteration
obs_x, obs_y = _xy(obs, Q_TRUE)
fit_x, fit_y = _xy(ic_fit, Q_TRUE)

# (2) in-backend AD potential (flattening q) fit
q_fit, q_path = fit_q_inbackend(truth, TS_BACK, q0=0.6, n_steps=60)
qs = numpy.linspace(0.6, 1.15, 23)
q_losses = q_loss_curve(truth, TS_BACK, qs)

out = {
    "meta": {"q_true": Q_TRUE, "tspan_back": -0.50, "sigma": SIGMA.tolist(),
             "potential": "cored flattened LogarithmicHalo", "note": "S5-HVS1-like mock"},
    "orbit_obs": {"x": obs_x, "y": obs_y},
    "orbit_fit": {"x": fit_x, "y": fit_y},
    "ic_fit": {"loss_hist": ic_loss,
               "r_obs": float(numpy.sqrt(ic_loss[0])), "r_fit": float(numpy.sqrt(ic_loss[-1]))},
    "q_fit": {"qs": qs.tolist(), "losses": q_losses.tolist(),
              "path_q": q_path[:, 0].tolist(), "path_loss": q_path[:, 1].tolist(),
              "recovered": float(q_fit)},
}
dpath = pathlib.Path(__file__).resolve().parent.parent / "data"
dpath.mkdir(exist_ok=True)
(dpath / "fit.json").write_text(json.dumps(out, indent=2) + "\n")
print(f"IC-fit closest approach r: {out['ic_fit']['r_obs']:.4f} -> {out['ic_fit']['r_fit']:.4f}")
print(f"q fit: true={Q_TRUE} recovered={q_fit:.3f}")
print("wrote", dpath / "fit.json")
