"""Train the Staeckel neural inverse (JAX), validate it (round-trip + orbit
reconstruction), and render the g19 figure. No analytic inverse exists, so the
validation is: (i) F(G(J,theta)) = (J,theta) round-trip, and (ii) G reconstructs
integrated orbits from their actions + angle evolution.
"""

import os

import numpy
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

import nn_staeckel_inverse_jax as M

HERE = os.path.dirname(os.path.abspath(__file__))
JAX_C = "#2a6f97"
ORB_C = ["#22577a", "#e07a5f", "#3d5a40"]


def predict(params, model, JR, Jz, thR, thz):
    return numpy.asarray(model.apply(params, M.feats(
        jnp.asarray(JR), jnp.asarray(Jz), jnp.asarray(thR), jnp.asarray(thz))))


def integrated_orbit(R0, vR0, z0, vz0, n_t=600, n_orbit=6.0):
    """Integrate an orbit in MWPotential2014 at L_z=LZ; return (R,z,vR,vz)(t)."""
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    vT0 = M.LZ / R0
    o = Orbit([R0, vR0, vT0, z0, vz0, 0.0])
    ts = numpy.linspace(0.0, n_orbit * 2.0 * numpy.pi, n_t)
    o.integrate(ts, MWPotential2014)
    return (numpy.asarray(o.R(ts)), numpy.asarray(o.z(ts)),
            numpy.asarray(o.vR(ts)), numpy.asarray(o.vz(ts)))


def main():
    params, model, hist, train_s = M.train()

    # ---- (b) round-trip quality on IN-DISTRIBUTION tori: draw phase-space
    # points the same way the training data was drawn, forward them to
    # (JR,Jz,thR,thz), then measure F_J(G(.)) closure (a fair test -- uniform
    # (JR,Jz) would include physically-unreachable high-JR+high-Jz pairs). ----
    Rt, vRt, zt, vzt, JR, Jz, thR, thz = (numpy.asarray(a) for a in
                                          M.gen_dataset(6000, seed=7))
    pred = predict(params, model, JR, Jz, thR, thz)
    JRh, Jzh, thRh, thzh = (numpy.asarray(a) for a in M.forward(
        jnp.asarray(pred[:, 0]), jnp.asarray(pred[:, 1]),
        jnp.asarray(pred[:, 2]), jnp.asarray(pred[:, 3])))
    ok = numpy.isfinite(JRh) & numpy.isfinite(Jzh)

    # ---- (a,c) orbit reconstruction ----
    ICs = [(1.0, 0.06, 0.12, 0.10), (0.85, 0.03, 0.20, 0.06), (1.15, 0.10, 0.06, 0.14)]
    orbits = []
    for (R0, vR0, z0, vz0) in ICs:
        R, z, vR, vz = integrated_orbit(R0, vR0, z0, vz0)
        JRo, Jzo, thRo, thzo = (numpy.asarray(a) for a in M.forward(
            jnp.asarray(R), jnp.asarray(vR), jnp.asarray(z), jnp.asarray(vz)))
        rec = predict(params, model, JRo, Jzo, thRo, thzo)
        orbits.append((R, z, rec[:, 0], rec[:, 2], numpy.median(JRo), numpy.median(Jzo)))

    # ---- figure ----
    fig, (axo, axl, axr) = plt.subplots(1, 3, figsize=(15.5, 4.7))

    # (a) meridional-plane orbit reconstruction
    for k, (R, z, Rr, zr, JRm, Jzm) in enumerate(orbits):
        c = ORB_C[k]
        axo.plot(R, z, "-", color=c, lw=1.4, alpha=0.55,
                 label=f"orbit $J_R$={JRm:.3f},$J_z$={Jzm:.3f}")
        axo.plot(Rr[::7], zr[::7], "o", color=c, ms=3.5, mfc="white", mew=1.0)
    axo.plot([], [], "o", color="0.35", mfc="white", mew=1.0, label="NN inverse $G(J,\\theta)$")
    axo.set_xlabel(r"$R$  [$R_0$]"); axo.set_ylabel(r"$z$  [$R_0$]")
    axo.set_title("(a) orbit reconstruction from actions+angles\n(line=integrated, ○=neural inverse)")
    axo.legend(fontsize=7.5, loc="upper right")
    axo.grid(alpha=0.25)

    # (b) training curve
    h = numpy.array(hist)
    axl.semilogy(h[:, 0], h[:, 1], "-o", color=JAX_C, ms=4, label="total")
    axl.semilogy(h[:, 0], h[:, 2], "--", color="#7aa5c2", label="data")
    axl.semilogy(h[:, 0], h[:, 3], ":", color="#e07a5f", label="physics cycle")
    axl.set_xlabel("training step")
    axl.set_ylabel("loss")
    axl.set_title(f"(b) backprop through Staeckel AA (JAX)\ntrained in {train_s:.0f}s")
    axl.legend(fontsize=8.5)
    axl.grid(alpha=0.25, which="both")

    # (c) round-trip: recovered vs input action
    axr.plot([0, 0.15], [0, 0.15], "k-", lw=0.8, alpha=0.5)
    axr.plot(JR[ok], JRh[ok], ".", color=JAX_C, ms=2, alpha=0.4, label="$J_R$")
    axr.plot(Jz[ok], Jzh[ok], ".", color="#e07a5f", ms=2, alpha=0.4, label="$J_z$")
    rms = numpy.sqrt(numpy.mean((JR[ok] - JRh[ok]) ** 2 + (Jz[ok] - Jzh[ok]) ** 2))
    axr.set_xlabel("input action  $J$")
    axr.set_ylabel(r"recovered action  $F_J(G(J,\theta))$")
    axr.set_title(f"(c) round-trip closure\nRMS $\\Delta J$ = {rms:.2e}")
    axr.legend(fontsize=9, markerscale=4)
    axr.grid(alpha=0.25)

    fig.suptitle(
        "Neural inverse of the Staeckel action-angle map for MWPotential2014 "
        "(no analytic inverse) -- learned by differentiating through galpy",
        fontsize=12.5, y=1.02,
    )
    fig.tight_layout()
    out = os.path.join(HERE, "g20_neural_staeckel_inverse.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"round-trip RMS dJ = {rms:.3e}; wrote {out}")


if __name__ == "__main__":
    main()
