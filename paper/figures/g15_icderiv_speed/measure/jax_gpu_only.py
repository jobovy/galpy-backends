"""Re-measure g15's jax (diffrax) GPU series ALONE -- the single-orbit IC gradient
and the batch scaling at every N already in the JSON -- with the protocol of
measure_icderiv.py / measure_scaling.py / measure_largeN.py (the same IC batch,
time_call median of 3 for N <= 4096, best of 2 above), and splice them in.

Usage: PYTHONPATH=<galpy worktree> python jax_gpu_only.py [path/to/icderiv.json]
(default: data/icderiv.json; the H100 copy is data/h100/icderiv.json). Written for
the device-aware incomplete-gamma switch (galpy branch backend/special-gpu-gate),
which changes only the compiled jax series on a GPU; the galpy commit is stamped in
galpy_commit_parts["gpu_jax"]."""
import datetime, json, os, subprocess, sys, time, pathlib
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
import numpy, jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from paper.figures._harness.bench import time_call  # noqa: E402
from galpy.backend._reference.inbackend_ode import integrate_orbit  # noqa: E402
from galpy.potential import MWPotential2014  # noqa: E402

_GPU = next(d for d in jax.devices() if d.platform == "gpu")
_IC = numpy.array([1.0, 0.1, 0.9, 0.05, 0.1, 0.2])
_TS = numpy.linspace(0.0, 10.0, 100)
ts_j = jnp.asarray(_TS)
_F = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else
                  pathlib.Path(__file__).resolve().parent.parent / "data" / "icderiv.json")
out = json.loads(_F.read_text())


def _batch(n):  # measure_scaling/largeN's batch (== measure_icderiv's for n <= 64)
    return numpy.stack([_IC * (1 + 0.001 * (k % 97)) for k in range(n)])


def _save():
    _F.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")


with jax.default_device(_GPU):
    dg = jax.jit(jax.grad(lambda ic: integrate_orbit(MWPotential2014, ic, ts_j)[-1, 0]))
    icj = jnp.asarray(_IC)
    out["single"]["diffrax"]["gpu"] = {"grad": time_call(lambda: jax.block_until_ready(dg(icj)), sync="jax")["median"]}
    _save()
    print("single diffrax gpu:", out["single"]["diffrax"]["gpu"], flush=True)
    gd = jax.jit(jax.grad(lambda v: jax.vmap(lambda ic: integrate_orbit(MWPotential2014, ic, ts_j)[-1, 0])(v).sum()))
    for n in sorted((int(k) for k, v in out["scaling"].items() if "diffrax_gpu" in v)):
        icb = jnp.asarray(_batch(n))
        if n <= 4096:
            t = time_call(lambda: jax.block_until_ready(gd(icb)), sync="jax", repeats=3)["median"]
        else:  # measure_largeN: warm-up, then the best of 2
            jax.block_until_ready(gd(icb))
            t = 1e18
            for _ in range(2):
                t0 = time.perf_counter(); jax.block_until_ready(gd(icb)); t = min(t, time.perf_counter() - t0)
        out["scaling"][str(n)]["diffrax_gpu"] = t
        _save()
        print(f"N={n}: diffrax_gpu {t:.4g} s", flush=True)
wt = os.environ["PYTHONPATH"].split(os.pathsep)[0]
sha = subprocess.run(["git", "-C", wt, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
out.setdefault("galpy_commit_parts", {})["gpu_jax"] = {
    "sha": sha, "worktree": wt, "date": datetime.datetime.now().isoformat(timespec="seconds"),
    "note": "diffrax GPU series re-measured alone (jax_gpu_only.py) with the device-aware incomplete-gamma switch"}
_save()
print("PATCH " + json.dumps({"single_gpu": out["single"]["diffrax"]["gpu"],
                             "scaling_gpu": {k: v["diffrax_gpu"] for k, v in out["scaling"].items() if "diffrax_gpu" in v},
                             "commit": out["galpy_commit_parts"]["gpu_jax"]}), flush=True)
print("written", _F)
