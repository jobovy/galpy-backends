"""Pilot: differentiable orbit integration with diffrax (jax) + torchdiffeq (torch),
through galpy's backend-agnostic force evaluators, validated vs the C integrator.

State y = [R, vR, phi, Omega(=dphi/dt), z, vz]  (galpy's _EOM ordering).
"""
import numpy

# ---- backend-agnostic cylindrical EOM (mirrors galpy._EOM, xp via force eval) ----
def rhs_components(xp_funcs, y):
    evalR, evalphi, evalz = xp_funcs
    R, vR, phi, Om, z, vz = y[0], y[1], y[2], y[3], y[4], y[5]
    Lz2 = (R**2 * Om) ** 2
    aR = Lz2 / R**3 + evalR(R, z, phi)
    dOm = (evalphi(R, z, phi) - 2.0 * R * vR * Om) / R**2
    az = evalz(R, z, phi)
    return vR, aR, Om, dOm, vz, az


# initial conditions (galpy Orbit convention [R, vR, vT, z, vz, phi]) + time grid
R0, vR0, vT0, z0, vz0, phi0 = 1.0, 0.1, 0.9, 0.2, 0.05, 0.3
T, NT = 8.0, 200
ts = numpy.linspace(0.0, T, NT)
y0_np = numpy.array([R0, vR0, phi0, vT0 / R0, z0, vz0])


def galpy_c_reference(b=0.6):
    from galpy.orbit import Orbit
    from galpy.potential import PlummerPotential
    pot = PlummerPotential(amp=1.0, b=b)
    o = Orbit([R0, vR0, vT0, z0, vz0, phi0])
    o.integrate(ts, pot, method="dop853_c")
    tf = ts[-1]
    return numpy.array([o.R(tf), o.vR(tf), o.phi(tf), o.vT(tf) / o.R(tf), o.z(tf), o.vz(tf)])


def jax_diffrax(b=0.6, y0=None):
    import jax, jax.numpy as jnp
    import diffrax
    from galpy.potential import (PlummerPotential, evaluateRforces,
                                 evaluatephitorques, evaluatezforces)
    if y0 is None:
        y0 = jnp.asarray(y0_np)
    pot = PlummerPotential(amp=1.0, b=b)
    fns = (lambda R, z, phi: evaluateRforces(pot, R, z, phi=phi),
           lambda R, z, phi: evaluatephitorques(pot, R, z, phi=phi),
           lambda R, z, phi: evaluatezforces(pot, R, z, phi=phi))

    def field(t, y, args):
        return jnp.stack(rhs_components(fns, y))

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(field), diffrax.Dopri8(), t0=0.0, t1=T, dt0=0.01, y0=y0,
        saveat=diffrax.SaveAt(ts=jnp.asarray(ts)),
        stepsize_controller=diffrax.PIDController(rtol=1e-11, atol=1e-11),
        max_steps=100000,
    )
    return sol.ys[-1]


def torch_torchdiffeq(b=0.6):
    import torch
    from torchdiffeq import odeint
    from galpy.potential import (PlummerPotential, evaluateRforces,
                                 evaluatephitorques, evaluatezforces)
    torch.set_default_dtype(torch.float64)
    pot = PlummerPotential(amp=1.0, b=b)
    fns = (lambda R, z, phi: evaluateRforces(pot, R, z, phi=phi),
           lambda R, z, phi: evaluatephitorques(pot, R, z, phi=phi),
           lambda R, z, phi: evaluatezforces(pot, R, z, phi=phi))

    def field(t, y):
        return torch.stack(rhs_components(fns, y))

    out = odeint(field, torch.as_tensor(y0_np), torch.as_tensor(ts),
                 method="dopri8", rtol=1e-11, atol=1e-11)
    return out[-1].detach().numpy()


if __name__ == "__main__":
    ref = galpy_c_reference()
    print("galpy C final:", numpy.array2string(ref, precision=8))

    import jax
    jax.config.update("jax_enable_x64", True)
    yj = numpy.asarray(jax_diffrax())
    print("jax/diffrax  :", numpy.array2string(yj, precision=8),
          " max|d vs C| =", float(numpy.max(numpy.abs(yj - ref))))

    yt = torch_torchdiffeq()
    print("torch/tde    :", numpy.array2string(yt, precision=8),
          " max|d vs C| =", float(numpy.max(numpy.abs(yt - ref))))

    # ---- differentiability: d(final R)/d(initial vR), autodiff vs finite-diff ----
    import jax.numpy as jnp

    def finalR_of_vR(vR):
        y0 = jnp.array([R0, vR, phi0, vT0 / R0, z0, vz0])
        return jax_diffrax(y0=y0)[0]

    g_ic = float(jax.grad(finalR_of_vR)(jnp.asarray(vR0)))
    eps = 1e-6
    fd_ic = (float(finalR_of_vR(vR0 + eps)) - float(finalR_of_vR(vR0 - eps))) / (2 * eps)
    print(f"\nd(final R)/d(vR0):  jax.grad={g_ic:.8f}  FD={fd_ic:.8f}  ok={abs(g_ic-fd_ic)<1e-5}")

    # ---- parameter gradient: d(final R)/d(Plummer b), autodiff vs finite-diff ----
    def finalR_of_b(b):
        return jax_diffrax(b=b)[0]

    g_b = float(jax.grad(finalR_of_b)(jnp.asarray(0.6)))
    fd_b = (float(finalR_of_b(0.6 + eps)) - float(finalR_of_b(0.6 - eps))) / (2 * eps)
    print(f"d(final R)/d(b):    jax.grad={g_b:.8f}  FD={fd_b:.8f}  ok={abs(g_b-fd_b)<1e-5}")
