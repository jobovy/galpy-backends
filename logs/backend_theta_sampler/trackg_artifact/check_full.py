"""Check the figure's frozen-structure deformation against just running the FULL
production streamTrack at different parameters (recomputes structure each time)."""
import sys, numpy; sys.path.insert(0,"/var/tmp/cint-param")
from galpy.orbit import Orbit
from galpy.potential import LogarithmicHaloPotential
from galpy.df import fardal15spraydf
from galpy.util import conversion
RO,VO=8.0,220.0; AMP0,Q0=1.0,0.9; IC=[1.2,0.15,0.85,0.08,0.05,0.0]; SEED=20260711
M=3*10.0**4.0/conversion.mass_in_msol(VO,RO); TD=4.5/conversion.time_in_Gyr(VO,RO)
def full_track(amp,q,ic):  # FULL production streamTrack: structure recomputed
    sp=fardal15spraydf(M,progenitor=Orbit(list(ic)),pot=LogarithmicHaloPotential(amp=amp,q=q),tdisrupt=TD)
    numpy.random.seed(SEED); xv,_=sp._sample_tail(600,True,leading=True)
    return sp.streamTrack(particles=numpy.asarray(xv,float),tail="leading",velocity_weight=1.0)
def ic_pert(c,d): x=list(IC); x[c]+=d; return x
tr0=full_track(AMP0,Q0,IC)
D={"amp":(AMP0+0.30,Q0,IC),"q":(AMP0,Q0+0.11,IC),"vR":(AMP0,Q0,ic_pert(1,0.10)),"vz":(AMP0,Q0,ic_pert(4,0.10))}
g0=numpy.asarray(tr0.tp_grid())
print("full streamTrack vs figure (frozen-structure) deformation, max|Δ(x,z)| over common tp:")
figvals={"amp":0.194,"q":0.0216,"vR":0.2305,"vz":0.1096}
for k,(a,q,ic) in D.items():
    trp=full_track(a,q,ic)
    gp=numpy.asarray(trp.tp_grid())
    lo,hi=max(g0[0],gp[0]),min(g0[-1],gp[-1])
    tp=numpy.linspace(lo+1e-3,hi-1e-3,120)
    d0=numpy.column_stack([numpy.asarray(tr0.x(tp)),numpy.asarray(tr0.z(tp))])
    dp=numpy.column_stack([numpy.asarray(trp.x(tp)),numpy.asarray(trp.z(tp))])
    dmax=float(numpy.abs(dp-d0).max())
    print(f"  {k}: FULL streamTrack={dmax:.4f}   figure(frozen-struct)={figvals[k]:.4f}   ratio={dmax/figvals[k]:.2f}")
