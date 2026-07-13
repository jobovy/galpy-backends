"""Is the closest-point assignment rougher than it should be? Compare how the
per-particle tp assignment ta(amp) changes with amp for HARD (argmin, current)
vs SOFT (softmax) vs SUBGRID (parabolic) assignment."""
import sys, numpy; sys.path.insert(0,"/var/tmp/cint-param")
from galpy.orbit import Orbit
from galpy.potential import LogarithmicHaloPotential
from galpy.df import fardal15spraydf
from galpy.util import conversion, coords
RO,VO=8.0,220.0; Q0=0.9; IC=[1.2,0.15,0.85,0.08,0.05,0.0]; SEED=20260711
M=3*10.0**4.0/conversion.mass_in_msol(VO,RO); TD=4.5/conversion.time_in_Gyr(VO,RO)
T,HD=2.0,1201
t_fwd=numpy.linspace(0.0,T,HD); t_back=numpy.linspace(0.0,-T,HD)
tg=numpy.concatenate([t_back[::-1],t_fwd[1:]])
def curve(amp):
    pot=LogarithmicHaloPotential(amp=amp,q=Q0)
    of=Orbit(IC); of.turn_physical_off(); of.integrate(t_fwd,pot,method="dop853_c")
    ob=Orbit(IC); ob.turn_physical_off(); ob.integrate(t_back,pot,method="dop853_c")
    cart=lambda o,ts: numpy.column_stack([o.x(ts),o.y(ts),o.z(ts)])
    return numpy.concatenate([cart(ob,t_back)[::-1],cart(of,t_fwd)[1:]],axis=0)
def sample(amp):
    sp=fardal15spraydf(M,progenitor=Orbit(IC),pot=LogarithmicHaloPotential(amp=amp,q=Q0),tdisrupt=TD)
    numpy.random.seed(SEED); xv,_=sp._sample_tail(600,True,leading=True)
    return coords.galcencyl_to_galcenrect(*numpy.asarray(xv,float))[:, :3]
def dist2(p,c): return numpy.sum((p[:,None,:]-c[None,:,:])**2,axis=-1)  # (N,M)
def ta_hard(p,c): return tg[numpy.argmin(dist2(p,c),axis=1)]
def ta_soft(p,c,tau):
    d2=dist2(p,c); d2=d2-d2.min(axis=1,keepdims=True)
    w=numpy.exp(-d2/tau); w/=w.sum(axis=1,keepdims=True); return w@tg
def ta_subgrid(p,c):
    d2=dist2(p,c); j=numpy.argmin(d2,axis=1); j=numpy.clip(j,1,c.shape[0]-2)
    f0=d2[numpy.arange(len(p)),j-1]; f1=d2[numpy.arange(len(p)),j]; f2=d2[numpy.arange(len(p)),j+1]
    denom=(f0-2*f1+f2); delta=numpy.where(numpy.abs(denom)>1e-30,0.5*(f0-f2)/denom,0.0)
    delta=numpy.clip(delta,-1,1); dtg=tg[1]-tg[0]
    return tg[j]+delta*dtg
p0=sample(1.0); c0=curve(1.0)
for name,fn in [("HARD (argmin, current)",ta_hard),("SUBGRID (parabolic)",ta_subgrid),
                ("SOFT tau=0.02",lambda p,c: ta_soft(p,c,0.02)),("SOFT tau=0.1",lambda p,c: ta_soft(p,c,0.1))]:
    ta0=fn(p0,c0)
    print(f"{name}:")
    for h in (1e-4,1e-3,1e-2):
        tap=fn(sample(1.0+h),curve(1.0+h))
        slope=float(numpy.median(numpy.abs(tap-ta0)))/h
        njump=int(numpy.sum(numpy.abs(tap-ta0)>3*(tg[1]-tg[0])))  # particles jumping >3 grid cells
        print(f"   h={h:g}: median|Δta|/h={slope:8.3f}   #big-jumps={njump}/600")
