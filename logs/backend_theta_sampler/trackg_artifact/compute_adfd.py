"""AD vs FD comparison: the exact autodiff gradient of the production streamTrack
vs a finite difference (= running streamTrack twice at amp +/- h) as a function of
the step size h. Shows the gradient is exact (FD -> AD at small h) while the FD is
unreliable (nonlinear at large h; discrete-assignment noise at intermediate h)."""
import sys, time, json; sys.path.insert(0,"/var/tmp/cint-param")
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp, numpy
from galpy.orbit import Orbit
from galpy.potential import LogarithmicHaloPotential
from galpy.df import fardal15spraydf
from galpy.util import conversion
from galpy.backend import random as grandom
RO,VO=8.0,220.0; Q0=0.9; IC=[1.2,0.15,0.85,0.08,0.05,0.0]
M=3*10.0**4.0/conversion.mass_in_msol(VO,RO)
key=grandom.key(20260711, backend="jax"); N=150
TD=1.0/conversion.time_in_Gyr(VO,RO); AMP0=1.05
def loss(amp):
    sp=fardal15spraydf(M,progenitor=Orbit(IC),pot=LogarithmicHaloPotential(amp=amp,q=Q0),tdisrupt=TD)
    tr=sp.streamTrack(n=N,tail="leading",velocity_weight=1.0,order=1,key=key)
    return tr._track_xyz.sum()
t0=time.time()
AD=float(jax.jit(jax.grad(loss))(jnp.asarray(AMP0)))
lj=jax.jit(loss)
print(f"AD gradient = {AD:.6e}  (compile {time.time()-t0:.0f}s)", flush=True)
hs=numpy.logspace(-7.0,-1.5,24)
fds=[]
for h in hs:
    fd=(float(lj(jnp.asarray(AMP0+h)))-float(lj(jnp.asarray(AMP0-h))))/(2*h)
    fds.append(fd); print(f"  h={h:.2e}: FD={fd:.6e} REL={abs(fd-AD)/abs(AD):.2e}", flush=True)
json.dump({"AD":AD,"hs":hs.tolist(),"fds":fds,"TD_gyr":1.0,"amp0":AMP0},
          open("/tmp/claude-21516/-home-bovy-Repos-galpy/e83d6165-0836-4adf-9f62-52251f942f9d/scratchpad/trackg/adfd.json","w"),indent=2)
print("saved adfd.json", flush=True)
