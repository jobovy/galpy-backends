"""Train ONE backend's neural AA-inverse and dump everything the figure needs to
an .npz (run each backend in its own process to avoid jax/torch both grabbing the
GPU). Separates 'measure' (this script) from 'plot' (make_figure.py).

Usage:
  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python train_and_dump.py jax
  USE_GPU=1 CUDA_VISIBLE_DEVICES=0 python train_and_dump.py torch
"""

import os
import sys
import time

import numpy

HERE = os.path.dirname(os.path.abspath(__file__))
PLOT_ES = [0.025, 0.07, 0.13]  # tori shown in panel (a) (within trained J range)
ERR_ES = list(numpy.linspace(0.022, 0.16, 24))  # energies -> J within [~0.01, ~0.13]
N_ANG = 200


def _numpy_ref(vpot, Es, n_ang):
    from galpy.actionAngle import actionAngleVerticalInverse

    aAVi = actionAngleVerticalInverse(
        pot=vpot, Es=list(numpy.linspace(0.005, 0.16, 12)), nta=128, setup_interp=True
    )
    th = numpy.linspace(0.0, 2.0 * numpy.pi, n_ang)
    Js, zs, vzs = [], [], []
    for E in Es:
        Jv = float(aAVi.J(E))
        z, vz = aAVi(numpy.array([Jv] * n_ang), th)
        Js.append(Jv); zs.append(numpy.asarray(z)); vzs.append(numpy.asarray(vz))
    return th, numpy.array(Js), numpy.array(zs), numpy.array(vzs)


def main(backend):
    t0 = time.time()
    if backend == "jax":
        import nn_vertical_inverse_jax as M
        import jax.numpy as jnp

        params, model, hist = M.main()

        def predict(J, th):
            return numpy.asarray(
                model.apply(params, M.feats(jnp.asarray(J), jnp.asarray(th)))
            )

    elif backend == "torch":
        import nn_vertical_inverse_torch as M
        import torch

        dev = "cuda" if (torch.cuda.is_available() and os.environ.get("USE_GPU")) else "cpu"
        model, hist = M.main(device=dev)

        def predict(J, th):
            Jt = torch.tensor(numpy.asarray(J), device=dev)
            tht = torch.tensor(numpy.asarray(th), device=dev)
            with torch.no_grad():
                return model(M.feats(Jt, tht)).cpu().numpy()

    else:
        raise SystemExit("backend must be jax or torch")

    train_s = time.time() - t0

    # panel (a): predicted tori at PLOT_ES
    th_a, Ja, za, vza = _numpy_ref(M._vpot, PLOT_ES, N_ANG)
    pred_a = numpy.array([predict(numpy.full(N_ANG, Jv), th_a) for Jv in Ja])

    # panel (c): RMS error vs numpy inverse across the action range
    th_c = numpy.linspace(0, 2 * numpy.pi, 128, endpoint=False)
    _, Jc, zc, vzc = _numpy_ref(M._vpot, ERR_ES, 128)
    errs = []
    for k, Jv in enumerate(Jc):
        p = predict(numpy.full(128, Jv), th_c)
        errs.append(
            numpy.sqrt(numpy.mean((p[:, 0] - zc[k]) ** 2 + (p[:, 1] - vzc[k]) ** 2))
        )

    out = os.path.join(HERE, f"results_{backend}.npz")
    numpy.savez(
        out,
        backend=backend,
        hist=numpy.array(hist),
        train_seconds=train_s,
        th_a=th_a, J_a=Ja, z_ref_a=za, vz_ref_a=vza, pred_a=pred_a,
        J_c=Jc, err_c=numpy.array(errs),
    )
    print(f"[{backend}] trained in {train_s:.1f}s; wrote {out}")
    print(f"[{backend}] inverse RMS vs numpy: "
          f"min={min(errs):.2e} max={max(errs):.2e}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "jax")
