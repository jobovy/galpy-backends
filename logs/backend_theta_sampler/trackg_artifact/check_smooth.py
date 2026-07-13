"""Is the FULL production streamTrack smooth in amp? Measure the track deformation
per unit amp at shrinking h -- constant => smooth; growing => jumpy reconstruction."""
import sys, numpy; sys.path.insert(0,"/var/tmp/cint-param")
from galpy.orbit import Orbit
from galpy.potential import LogarithmicHaloPotential
from galpy.df import fardal15spraydf
from galpy.util import conversion
RO,VO=8.0,220.0; Q0=0.9; IC=[1.2,0.15,0.85,0.08,0.05,0.0]; SEED=20260711
M=3*10.0**4.0/conversion.mass_in_msol(VO,RO); TD=4.5/conversion.time_in_Gyr(VO,RO)
def track(amp,T=None):
    sp=fardal15spraydf(M,progenitor=Orbit(IC),pot=LogarithmicHaloPotential(amp=amp,q=Q0),tdisrupt=TD)
    numpy.random.seed(SEED); xv,_=sp._sample_tail(600,True,leading=True)
    kw=dict(tail="leading",velocity_weight=1.0)
    if T is not None: kw["track_time_range"]=T
    return sp.streamTrack(particles=numpy.asarray(xv,float),**kw)
for T,lbl in [(None,"auto extent"),(2.0,"T=2.0")]:
    tr0=track(1.0,T); g0=numpy.asarray(tr0.tp_grid())
    print(f"--- {lbl}: tp_range=[{g0[0]:.3f},{g0[-1]:.3f}] ---")
    for h in (1e-4,1e-3,1e-2):
        trp=track(1.0+h,T); gp=numpy.asarray(trp.tp_grid())
        lo,hi=max(g0[0],gp[0]),min(g0[-1],gp[-1]); tp=numpy.linspace(lo+1e-3,hi-1e-3,150)
        d0=numpy.column_stack([numpy.asarray(tr0.x(tp)),numpy.asarray(tr0.z(tp))])
        dp=numpy.column_stack([numpy.asarray(trp.x(tp)),numpy.asarray(trp.z(tp))])
        slope_max=float(numpy.abs(dp-d0).max())/h; slope_med=float(numpy.median(numpy.abs(dp-d0)))/h
        print(f"   h={h:g}: |Δtrack|/h  max={slope_max:.3f}  median={slope_med:.3f}  (constant => smooth)")
