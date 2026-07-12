"""Generative halo-mass fit: descend d(track)/d(amp) with the particles xv(amp)
FLOWING (re-integrated each step). Structure frozen at the observed stream."""
import sys, time, json; sys.path.insert(0, "/var/tmp/cint-param")
sys.path.insert(0, "/home/bovy/Repos/galpy-jax/logs/backend_theta_sampler")
import numpy, jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import m2_generative as m2
OUT="/tmp/claude-21516/-home-bovy-Repos-galpy/e83d6165-0836-4adf-9f62-52251f942f9d/scratchpad/trackg"
AMP_TRUE, AMP_START, LAM = 1.0, 1.08, 1.0
idx0, B0 = jax.jit(m2._structure)(jnp.asarray(AMP_TRUE))
idx0 = jax.lax.stop_gradient(idx0); B0 = jax.lax.stop_gradient(B0)
trk = jax.jit(lambda a: m2.track_given(a, idx0, B0, LAM)[0])
target = jax.lax.stop_gradient(trk(jnp.asarray(AMP_TRUE)))
def loss(amp): return jnp.mean((m2.track_given(amp, idx0, B0, LAM)[0]-target)**2)
vg = jax.jit(jax.value_and_grad(loss))
amp = jnp.asarray(AMP_START); m=v=0.0; lr,b1,b2,eps=3e-3,0.9,0.999,1e-12
losses, amps = [], [float(amp)]; t0=time.time()
for step in range(60):
    l,g = vg(amp); l=float(l); g=float(g)
    m=b1*m+(1-b1)*g; v=b2*v+(1-b2)*g*g
    amp = amp - lr*(m/(1-b1**(step+1)))/(jnp.sqrt(v/(1-b2**(step+1)))+eps)
    losses.append(l); amps.append(float(amp))
    if step%10==0 or step==79: print(f"  step {step:2d}: amp={float(amp):.5f} loss={l:.3e}", flush=True)
tr_start=numpy.asarray(trk(jnp.asarray(AMP_START)))
tr_obs=numpy.asarray(target); tr_final=numpy.asarray(trk(jnp.asarray(amps[-1])))
numpy.savez(f"{OUT}/fit_data.npz", losses=numpy.array(losses), amps=numpy.array(amps),
            tr_start=tr_start, tr_obs=tr_obs, tr_final=tr_final)
J={"fit_amp_start":AMP_START,"fit_amp_true":AMP_TRUE,"fit_amp_final":float(amps[-1]),
   "fit_loss_start":float(losses[0]),"fit_loss_final":float(losses[-1]),"fit_n_steps":len(losses)}
json.dump(J, open(f"{OUT}/fit_numbers.json","w"), indent=2)
print(f"RESULT amp {AMP_START} -> {float(amps[-1]):.5f} loss {losses[0]:.2e} -> {losses[-1]:.2e} in {time.time()-t0:.0f}s", flush=True)
