"""g14 skeleton -- "The Great Escape": constrain the Milky Way with a hypervelocity
star, using BOTH differentiable-orbit mechanisms (Koposov et al. 2019, S5-HVS1,
arXiv:1907.11725).

A star ejected from Sgr A* must, when integrated *backward* through the correct
Galactic potential, return to the Galactic Centre (r -> 0). We use the closest
approach r^2 of the backward orbit as a differentiable objective and fit two things:

  (1) the star's present-day phase-space WITHIN its observational uncertainties --
      an INITIAL-CONDITION fit. Gradients dx(t)/dx0 come from galpy's compiled C
      variational integrator (the state-transition matrix), so this is fast.
  (2) a potential parameter (here the halo flattening q) -- a PARAMETER fit. C-STM
      carries no d/dtheta, so this uses the in-backend ODE solve (diffrax) and
      reverse-mode autodiff, which differentiates w.r.t. the potential.

The two mechanisms compose: C-STM for the (cheap, frequent) IC gradients, in-backend
AD for the parameter gradient. make.py imports these functions, so the paper listing
is exactly the code that ran. Needs jax + diffrax + galpy(#972 worktree).
"""

import numpy

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from galpy.backend._jax.orbit_stm import integrate as cstm_integrate  # C-STM (IC grads)
from galpy.backend._reference.inbackend_ode import integrate_orbit  # in-backend AD (param)
from galpy.potential import LogarithmicHaloPotential

# Flattened, cored logarithmic halo; the cored centre keeps the backward orbit
# regular as it approaches the Galactic Centre. The knob we fit is the flattening q.
def potential(q):
    return LogarithmicHaloPotential(amp=1.0, q=q, core=0.1)


def _r2_to_gc(orbit):
    return orbit[:, 0] ** 2 + orbit[:, 3] ** 2  # R^2 + z^2 along the trajectory


# ---- C-STM: backward closest approach, differentiable in the initial conditions ----
def closest_r2_cstm(present_ic, q, ts_back):
    orbit = cstm_integrate(potential(q), jnp.asarray(present_ic), ts_back, method="dop853_c")
    return jnp.min(_r2_to_gc(orbit))


# ---- in-backend AD: same objective, differentiable in the potential parameter q ----
def closest_r2_inbackend(present_ic, q, ts_back):
    orbit = integrate_orbit(potential(q), jnp.asarray(present_ic), jnp.asarray(ts_back))
    return jnp.min(_r2_to_gc(orbit))


def mock_s5hvs1(q_true, ts_fwd):
    """An S5-HVS1-like mock: eject a ~1700 km/s star from the Galactic Centre, nearly
    radially (small angular momentum) but off-plane, and integrate it FORWARD through
    the true potential to 'today'. Launching from ~r=0 makes the GC-return a sharp
    constraint: the backward closest approach reflects the angular-momentum-set
    pericenter (sensitive to the IC) and the z-deflection (sensitive to q)."""
    launch = jnp.array([0.004, 2.4, 0.03, 0.002, 1.5, 0.6])  # R,vR,vT,z,vz,phi
    return numpy.asarray(cstm_integrate(potential(q_true), launch, ts_fwd, method="dop853_c")[-1])


# ---- (1) IC fit via C-STM: recover the phase-space that returns to the GC ----------
def fit_ic_cstm(obs_ic, q, ts_back, sigma, n_steps=6, damp=1e-3, sigma_gc=0.004):
    """Gauss-Newton on [GC-return residual + (ic-obs)/sigma prior]. The GC-return
    term sqrt(r2)/sigma_gc encodes the strong prior that the star was ejected from
    Sgr A* (must pass within ~sigma_gc of the Centre); the (ic-obs)/sigma prior keeps
    the fit within the observational uncertainties. Only the GC-return row needs the
    C state-transition matrix (the prior block is the constant I/sigma), so one
    value_and_grad of sqrt(r2) gives the row in a single C-STM solve."""
    ic = jnp.asarray(obs_ic)
    obs = jnp.asarray(obs_ic)
    sig = jnp.asarray(sigma)
    sqrt_r2_and_grad = jax.value_and_grad(lambda x: jnp.sqrt(closest_r2_cstm(x, q, ts_back)))
    prior_block = jnp.diag(1.0 / sig)  # constant Jacobian of the (ic-obs)/sigma rows

    hist = []
    for _ in range(n_steps):
        s, gtop = sqrt_r2_and_grad(ic)  # GC-return residual value + its IC gradient (the STM)
        hist.append((numpy.asarray(ic), float(s) ** 2))
        r = jnp.concatenate([jnp.array([s / sigma_gc]), (ic - obs) / sig])  # (7,)
        J = jnp.vstack([(gtop / sigma_gc)[None, :], prior_block])  # (7,6)
        step = jnp.linalg.solve(J.T @ J + damp * jnp.eye(6), J.T @ r)
        ic = ic - step
    hist.append((numpy.asarray(ic), float(closest_r2_cstm(ic, q, ts_back))))
    return numpy.asarray(ic), hist


# ---- (2) parameter fit via in-backend AD: recover the halo flattening q ------------
# diffrax's adaptive solver uses a while_loop, so reverse-over-reverse (a Hessian)
# is unavailable -> first-order. Adam is robust to the (small) gradient scale.
def fit_q_inbackend(present_ic, ts_back, q0, lr=0.04, n_steps=40):
    present_ic, ts_back = jnp.asarray(present_ic), jnp.asarray(ts_back)
    # jit value_and_grad: the diffrax solve compiles ONCE and is reused across Adam
    # steps (q is a traced arg) instead of re-tracing every step -- the intended
    # "user jits their own galpy-using code" pattern.
    vg = jax.jit(jax.value_and_grad(lambda q: closest_r2_inbackend(present_ic, q, ts_back)))
    q, m, v, hist = q0, 0.0, 0.0, []
    for t in range(1, n_steps + 1):
        l, g = vg(jnp.asarray(q))
        hist.append((float(q), float(l)))
        g = float(g)
        m, v = 0.9 * m + 0.1 * g, 0.999 * v + 0.001 * g * g
        mhat, vhat = m / (1 - 0.9**t), v / (1 - 0.999**t)
        q = float(jnp.clip(q - lr * mhat / (vhat**0.5 + 1e-12), 0.3, 1.6))
    hist.append((q, float(vg(jnp.asarray(q))[0])))
    return q, numpy.array(hist)


def q_loss_curve(present_ic, ts_back, qs):
    """closest-approach^2 vs q (a grid scan) -- the GC-return constraint landscape."""
    present_ic, ts_back = jnp.asarray(present_ic), jnp.asarray(ts_back)
    loss = jax.jit(lambda q: closest_r2_inbackend(present_ic, q, ts_back))
    return numpy.array([float(loss(jnp.asarray(q))) for q in qs])


if __name__ == "__main__":
    q_true = 0.85
    ts_fwd = numpy.linspace(0.0, 0.50, 150)
    ts_back = numpy.linspace(0.0, -0.50, 150)
    truth = mock_s5hvs1(q_true, ts_fwd)
    rng = numpy.random.default_rng(1)
    sigma = numpy.array([0.03, 0.05, 0.05, 0.02, 0.05, 0.02])
    obs = truth + rng.normal(size=6) * sigma
    print("closest-approach r (obs, true q):", numpy.sqrt(float(closest_r2_cstm(obs, q_true, ts_back))))
    ic_fit, _ = fit_ic_cstm(obs, q_true, ts_back, sigma)
    print("after C-STM IC fit:", numpy.sqrt(float(closest_r2_cstm(ic_fit, q_true, ts_back))))
    q_fit, qh = fit_q_inbackend(truth, ts_back, q0=0.6)
    print(f"in-backend q fit: true={q_true} recovered={q_fit:.3f}")
