"""Payoff: recover the halo mass by gradient descent on the GENERATIVE stream
track (xv(theta) flows). Uses the jittable M2 pipeline. Observed = track at
amp=1.0; fit from a mismatched start. Frozen structure+noise (common random
numbers) so the loss is smooth; the gradient is the physical d(track)/d(amp)."""
import sys, time; sys.path.insert(0, "/var/tmp/cint-param")
import numpy
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import m2_generative as m2   # importing does NOT run its __main__

AMP_TRUE, AMP_START = 1.0, 1.08
LAM = 1.0

# frozen structure + lambda at the TRUE amp (the "observed" stream's reconstruction)
idx0, B0 = jax.jit(m2._structure)(jnp.asarray(AMP_TRUE))
idx0 = jax.lax.stop_gradient(idx0); B0 = jax.lax.stop_gradient(B0)
target = jax.lax.stop_gradient(jax.jit(lambda a: m2.track_given(a, idx0, B0, LAM)[0])(jnp.asarray(AMP_TRUE)))


def loss(amp):
    tr = m2.track_given(amp, idx0, B0, LAM)[0]
    return jnp.mean((tr - target) ** 2)


vg = jax.jit(jax.value_and_grad(loss))
# scale-aware step: gradient is large (sensitive stream) -> use Adam
amp = jnp.asarray(AMP_START)
# simple hand-rolled Adam
m, v = 0.0, 0.0
lr, b1, b2, eps = 3e-3, 0.9, 0.999, 1e-12
print(f"start amp={float(amp):.4f}", flush=True)
t0 = time.time()
for step in range(60):
    l, g = vg(amp); l = float(l); g = float(g)
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g * g
    mhat = m / (1 - b1 ** (step + 1)); vhat = v / (1 - b2 ** (step + 1))
    amp = amp - lr * mhat / (jnp.sqrt(vhat) + eps)
    if step % 10 == 0 or step == 59:
        print(f"  step {step:2d}: amp={float(amp):.5f} loss={l:.3e} grad={g:.2e}", flush=True)
print(f"RESULT: amp {AMP_START} -> {float(amp):.5f} (true {AMP_TRUE})  in {time.time()-t0:.0f}s", flush=True)
