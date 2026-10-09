"""Train the Staeckel neural inverse (JAX), validate it (round-trip + orbit
reconstruction), and dump everything the g20 figure needs to
data/results_staeckel_jax.npz. 'Measure' (this script, GPU) is separate from
'plot' (make_staeckel_figure.py, plot-only, reads the dump).

No analytic inverse exists, so the validation is: (i) F(G(J,theta)) = (J,theta)
round-trip on in-distribution tori, and (ii) G reconstructs integrated orbits
from their actions + angle evolution.

galpy comes from $GALPY_WORKTREE (prepended to sys.path) or else PYTHONPATH;
the paper's data are on galpy feat/backends 34bca2d1d, stamped as galpy_commit.

Compile cost: each jit of the Staeckel forward takes minutes and recompiles per
input shape, so the forward runs in fixed 512-point chunks (one executable) and
the training step is compiled once, ahead of time; both are timed into the dump.
A persistent compilation cache (--cache-dir or $JAX_COMPILATION_CACHE_DIR) makes
a re-run skip them. --ckpt checkpoints the training every --ckpt-every steps and
--resume continues a timed-out run from it (bit-identical to an uninterrupted one).

Usage (any cwd; normally via tools/trillium_stage.sh g20 on the H100):
  XLA_PYTHON_CLIENT_PREALLOCATE=false GALPY_WORKTREE=<galpy> python train_and_dump.py \
      [--cache-dir DIR] [--ckpt FILE [--resume]]
  ... python train_and_dump.py --smoke      # 20 steps on a small dataset; writes
                                            # data/results_staeckel_jax_smoke.npz (not for the figure)
"""

import argparse
import os
import platform
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if os.environ.get("GALPY_WORKTREE"):
    sys.path.insert(0, os.environ["GALPY_WORKTREE"])
sys.path.insert(0, HERE)

import numpy  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402


def _set_cache(d):
    if d:
        os.makedirs(d, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", d)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    return d


import nn_staeckel_inverse_jax as M  # noqa: E402

# panel (a): integrated orbits (R, vR, z, vz) at L_z = M.LZ
ICS = [(1.0, 0.06, 0.12, 0.10), (0.85, 0.03, 0.20, 0.06), (1.15, 0.10, 0.06, 0.14)]


def predict(params, model, JR, Jz, thR, thz):
    return numpy.asarray(model.apply(params, M.feats(
        jnp.asarray(JR), jnp.asarray(Jz), jnp.asarray(thR), jnp.asarray(thz))))


def integrated_orbit(R0, vR0, z0, vz0, n_t=600, n_orbit=6.0):
    """Integrate an orbit in MWPotential2014 at L_z=LZ; return (R,z,vR,vz)(t)."""
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014

    o = Orbit([R0, vR0, M.LZ / R0, z0, vz0, 0.0])
    ts = numpy.linspace(0.0, n_orbit * 2.0 * numpy.pi, n_t)
    o.integrate(ts, MWPotential2014)
    return (numpy.asarray(o.R(ts)), numpy.asarray(o.z(ts)),
            numpy.asarray(o.vR(ts)), numpy.asarray(o.vz(ts)))


def _galpy_commit():
    import galpy

    d = os.path.dirname(os.path.dirname(os.path.abspath(galpy.__file__)))
    try:
        sha = subprocess.check_output(["git", "-C", d, "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", d, "status", "--porcelain", "-uno"],
                                        text=True).strip()
        return sha, bool(dirty), galpy.__version__
    except Exception:  # not a git checkout
        return "unknown", False, galpy.__version__


def _peak_gpu_bytes():
    try:
        return int(jax.devices()[0].memory_stats().get("peak_bytes_in_use", -1))
    except Exception:  # CPU device: no memory_stats
        return -1


def main(n_steps=3200, smoke=False, out=None, n_data=None, n_rt=None, n_t=None,
         ckpt=None, ckpt_every=200, resume=False, log_every=50, stop_after=None, cache_dir=None):
    d_data, d_rt, d_t = (3000, 500, 120) if smoke else (40000, 6000, 600)
    n_data, n_rt, n_t = n_data or d_data, n_rt or d_rt, n_t or d_t
    if smoke:
        n_steps = min(n_steps, 20)
    hp = dict(n_steps=n_steps, batch=512, lr=2e-3, warmup=400, seed=0, n_data=n_data)
    t_all = time.time()
    print(f"{hp}; device {jax.devices()[0].device_kind}; jax {jax.__version__}; "
          f"compilation cache {cache_dir or 'off'}", flush=True)
    params, model, hist, train_s = M.train(**hp, ckpt=ckpt, ckpt_every=ckpt_every, resume=resume,
                                           log_every=log_every, stop_after=stop_after)
    t_eval = time.time()

    # ---- (c) round-trip on IN-DISTRIBUTION tori: draw phase-space points the
    # way the training data was drawn, forward them to (JR,Jz,thR,thz), then
    # measure F(G(.)) closure (uniform (JR,Jz) would include unreachable pairs).
    _, _, _, _, JR, Jz, thR, thz = (numpy.asarray(a) for a in M.gen_dataset(n_rt, seed=7))
    pred = predict(params, model, JR, Jz, thR, thz)
    # jitted (chunked): an unbound proposal returns the 9999.99 sentinel / NaN
    # (masked below) instead of raising UnboundError as the eager forward does
    JRh, Jzh, thRh, thzh = M.forward_chunked(*(pred[:, k] for k in range(4)))
    ok = numpy.isfinite(JRh) & numpy.isfinite(Jzh) & (JRh < 1.0) & (Jzh < 1.0)
    rms = numpy.sqrt(numpy.mean((JR[ok] - JRh[ok]) ** 2 + (Jz[ok] - Jzh[ok]) ** 2))
    dth = numpy.concatenate([numpy.angle(numpy.exp(1j * (thRh[ok] - thR[ok]))),
                             numpy.angle(numpy.exp(1j * (thzh[ok] - thz[ok])))])
    rms_th = numpy.sqrt(numpy.mean(dth ** 2))

    # ---- (a) orbit reconstruction ----
    orb_R, orb_z, rec_R, rec_z, orb_JR, orb_Jz = [], [], [], [], [], []
    for (R0, vR0, z0, vz0) in ICS:
        R, z, vR, vz = integrated_orbit(R0, vR0, z0, vz0, n_t=n_t)
        JRo, Jzo, thRo, thzo = M.forward_chunked(R, vR, z, vz)
        rec = predict(params, model, JRo, Jzo, thRo, thzo)
        orb_R.append(R); orb_z.append(z); rec_R.append(rec[:, 0]); rec_z.append(rec[:, 2])
        orb_JR.append(numpy.median(JRo)); orb_Jz.append(numpy.median(Jzo))

    sha, dirty, gversion = _galpy_commit()
    dev = jax.devices()[0]
    if out is None:
        out = os.path.join(HERE, "data", "results_staeckel_jax_smoke.npz" if smoke
                           else "results_staeckel_jax.npz")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    numpy.savez(
        out,
        hist=numpy.array(hist),  # columns: step, total, data (sup), cycle
        hist_columns=numpy.array(["step", "total", "data", "cycle"]),
        train_seconds=train_s,  # training-loop wall time, summed over resumed runs; excl. compile
        total_seconds=time.time() - t_all,  # this run only
        eval_seconds=time.time() - t_eval,
        # trace + lower + compile (the persistent cache saves only the XLA compile part)
        compile_step_seconds=M.TIMINGS["compile_step"],
        compile_forward_seconds=M.TIMINGS[f"compile_forward_{M.CHUNK}"],
        dataset_seconds=M.TIMINGS["dataset_seconds"],
        resumed_from_step=M.TIMINGS["resumed_from_step"], n_runs=M.TIMINGS["n_runs"],
        compilation_cache=str(cache_dir or ""),
        hyperparams=numpy.array(repr(hp)), smoke=smoke,
        JR_in=JR, Jz_in=Jz, thR_in=thR, thz_in=thz,
        JR_rt=JRh, Jz_rt=Jzh, thR_rt=thRh, thz_rt=thzh, rt_ok=ok,
        rms_dJ=rms, rms_dtheta=rms_th, n_rt_ok=int(ok.sum()),
        orbit_ics=numpy.array(ICS), orbit_R=numpy.array(orb_R), orbit_z=numpy.array(orb_z),
        recon_R=numpy.array(rec_R), recon_z=numpy.array(rec_z),
        orbit_JR=numpy.array(orb_JR), orbit_Jz=numpy.array(orb_Jz),
        delta=M.DELTA, Lz=M.LZ,
        galpy_commit=sha, galpy_dirty=dirty, galpy_version=gversion,
        jax_version=jax.__version__, device=dev.platform, device_kind=dev.device_kind,
        peak_gpu_bytes=_peak_gpu_bytes(), host=platform.node(),
        python_version=sys.version.split()[0],
        created=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    print(f"compile: forward {M.TIMINGS[f'compile_forward_{M.CHUNK}']:.0f}s, "
          f"train step {M.TIMINGS['compile_step']:.0f}s; dataset {M.TIMINGS['dataset_seconds']:.0f}s; "
          f"eval {time.time() - t_eval:.0f}s", flush=True)
    print(f"trained {n_steps} steps in {train_s:.1f}s on {dev.device_kind}; galpy {sha}"
          f"{' (dirty)' if dirty else ''}; jax {jax.__version__}; "
          f"peak GPU memory {_peak_gpu_bytes() / 2**30:.2f} GiB" if _peak_gpu_bytes() >= 0
          else "no GPU", flush=True)
    print(f"round-trip RMS dJ = {rms:.3e}, RMS dtheta = {rms_th:.3e} rad "
          f"({ok.sum()}/{ok.size} finite); wrote {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--steps", type=int, default=3200)
    ap.add_argument("--smoke", action="store_true", help="20 steps, small dataset")
    ap.add_argument("--out", default=None, help="dump path (default: data/results_staeckel_jax[_smoke].npz)")
    ap.add_argument("--n-data", type=int, help="override the dataset size (smoke-testing only)")
    ap.add_argument("--n-rt", type=int, help="override the round-trip sample size (smoke-testing only)")
    ap.add_argument("--n-t", type=int, help="override the orbit sampling (smoke-testing only)")
    ap.add_argument("--cache-dir", default=os.environ.get("JAX_COMPILATION_CACHE_DIR"),
                    help="persistent jax compilation cache (default $JAX_COMPILATION_CACHE_DIR)")
    ap.add_argument("--ckpt", default=None, help="checkpoint file (params, optimizer state, step)")
    ap.add_argument("--ckpt-every", type=int, default=200)
    ap.add_argument("--resume", action="store_true", help="continue from --ckpt if it exists")
    ap.add_argument("--log-every", type=int, default=50, help="progress line every N steps")
    ap.add_argument("--stop-after", type=int, default=None,
                    help="end after N training steps of this run (tests --resume)")
    a = ap.parse_args()
    if a.resume and not a.ckpt:
        ap.error("--resume needs --ckpt")
    main(a.steps, a.smoke, a.out, a.n_data, a.n_rt, a.n_t, a.ckpt, a.ckpt_every, a.resume,
         a.log_every, a.stop_after, _set_cache(a.cache_dir))
