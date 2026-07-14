"""Torch twin of nn_vertical_inverse_jax.py: the SAME hybrid (data + physics-
informed) neural inverse of the 1D anharmonic vertical action-angle map,
backpropagating through galpy's differentiable forward AA under PyTorch autograd.
Demonstrates backend-agnosticism -- identical physics/loss, torch.nn + torch.optim
instead of flax + optax.
"""

import os
import warnings

import numpy
import torch

torch.set_default_dtype(torch.float64)
warnings.filterwarnings("ignore", message=".*requires_grad now defaults.*")

from galpy.actionAngle import actionAngleVertical
from galpy.potential import MWPotential2014, toVerticalPotential

R0 = 1.0
_vpot = toVerticalPotential(MWPotential2014, R0)
_aAV = actionAngleVertical(pot=_vpot)

Z_BOX, VZ_BOX = 1.6, 0.95


def forward_Jtheta(z, vz):
    J, _Omega, theta = _aAV._actionsFreqsAngles(z, vz)
    return J, theta


def feats(J, th):
    sj = torch.sqrt(torch.clamp(J, min=0.0))
    c, s = torch.cos(th), torch.sin(th)
    return torch.stack([J, sj * c, sj * s, c, s], dim=-1)


class InverseMLP(torch.nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(5, width), torch.nn.Tanh(),
            torch.nn.Linear(width, width), torch.nn.Tanh(),
            torch.nn.Linear(width, width), torch.nn.Tanh(),
            torch.nn.Linear(width, 2),
        )
        self.register_buffer("box", torch.tensor([Z_BOX, VZ_BOX]))

    def forward(self, x):
        return self.box * torch.tanh(self.net(x))


def gen_dataset(n, Jlo, Jhi, seed=0, device="cpu"):
    rng = numpy.random.RandomState(seed)
    scales = numpy.concatenate([numpy.full(n // 3, s) for s in (1.0, 0.45, 0.2)])[:n]
    scales = numpy.concatenate([scales, numpy.ones(n - scales.shape[0])])
    z = scales * Z_BOX * (2 * rng.rand(n) - 1)
    vz = scales * VZ_BOX * (2 * rng.rand(n) - 1)
    with torch.no_grad():
        J, th = forward_Jtheta(torch.tensor(z), torch.tensor(vz))
    J = J.cpu().numpy(); th = th.cpu().numpy()
    m = (J > Jlo) & (J < Jhi)
    to = lambda a: torch.tensor(a[m], device=device)
    return to(z), to(vz), to(J), to(th)


def loss_fn(model, z, vz, J, th, lam):
    pred = model(feats(J, th))
    w = 1.0 / (J + 0.015)
    se = (pred[:, 0] - z) ** 2 + (pred[:, 1] - vz) ** 2
    sup = torch.sum(w * se) / torch.sum(w)
    Jh, thh = forward_Jtheta(pred[:, 0], pred[:, 1])
    cyc_J = torch.mean(((Jh - J) / (J + 0.015)) ** 2)
    cyc_a = torch.mean(1.0 - torch.cos(thh - th))
    return sup + lam * (cyc_J + cyc_a), sup, cyc_J, cyc_a


def main(n_steps=1200, batch=256, lr=2e-3, Jlo=0.008, Jhi=0.14, lam=1.0,
         seed=0, width=128, device="cpu"):
    z, vz, J, th = gen_dataset(60000, Jlo, Jhi, seed=seed, device=device)
    ndat = int(z.shape[0])
    print(f"dataset: {ndat} bound samples, J in [{float(J.min()):.3f},{float(J.max()):.3f}]")

    torch.manual_seed(seed)
    model = InverseMLP(width).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_steps, eta_min=lr * 0.05)
    hist = []
    for i in range(n_steps):
        idx = torch.randint(0, ndat, (batch,), device=device)
        opt.zero_grad()
        loss, s, cJ, ca = loss_fn(model, z[idx], vz[idx], J[idx], th[idx], lam)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if i % 250 == 0 or i == n_steps - 1:
            hist.append((i, float(loss), float(s), float(cJ), float(ca)))
            print(
                f"step {i:4d}  loss={float(loss):.3e}  sup={float(s):.3e}  "
                f"cyc_J={float(cJ):.3e}  cyc_ang={float(ca):.3e}"
            )
    return model, hist


def validate_vs_numpy(model, Es=(0.02, 0.05, 0.10), n_ang=64, device="cpu"):
    from galpy.actionAngle import actionAngleVerticalInverse

    aAVi = actionAngleVerticalInverse(
        pot=_vpot, Es=list(numpy.linspace(0.005, 0.16, 12)), nta=128, setup_interp=True
    )
    th = numpy.linspace(0.0, 2.0 * numpy.pi, n_ang, endpoint=False)
    out = {}
    for E in Es:
        Jv = float(aAVi.J(E))
        z_ref, vz_ref = aAVi(numpy.array([Jv] * n_ang), th)
        Jt = torch.tensor(numpy.full(n_ang, Jv), device=device)
        tht = torch.tensor(th, device=device)
        with torch.no_grad():
            zv = model(feats(Jt, tht)).cpu().numpy()
        rms = float(
            numpy.sqrt(numpy.mean((zv[:, 0] - z_ref) ** 2 + (zv[:, 1] - vz_ref) ** 2))
        )
        out[E] = (Jv, rms)
    return out


if __name__ == "__main__":
    dev = "cuda" if (torch.cuda.is_available() and os.environ.get("USE_GPU")) else "cpu"
    print(f"device: {dev}")
    model, hist = main(device=dev)
    print("\n=== validation vs numpy actionAngleVerticalInverse (RMS |dz,dvz|) ===")
    for E, (Jv, rms) in validate_vs_numpy(model, device=dev).items():
        print(f"  E={E:.3f}  J={Jv:.4f}  RMS(z,vz) inverse error = {rms:.3e}")
