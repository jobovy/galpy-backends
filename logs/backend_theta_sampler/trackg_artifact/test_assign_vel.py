"""Does using VELOCITIES (6D, as production does) make the closest-point
assignment smoother than position-only (3D)?"""
import sys, numpy; sys.path.insert(0,"/var/tmp/cint-param")
from galpy.orbit import Orbit
from galpy.potential import LogarithmicHaloPotential
from galpy.df import fardal15spraydf
from galpy.util import conversion, coords
RO,VO=8.0,220.0; Q0=0.9; IC=[1.2,0.15,0.85,0.08,0.05,0.0]; SEED=20260711
M=3*10.0**4.0/conversion.mass_in_msol(VO,RO); TD=4.5/conversion.time_in_Gyr(VO,RO)
T,HD=2.0,1201
t_fwd=numpy.linspace(0.0,T,HD); t_back=numpy.linspace(0.0,-T,HD)
tg=numpy.concatenate([t_back[::-1],t_fwd[1:]]); dtg=tg[1]-tg[0]
def curve6(amp):
    pot=LogarithmicHaloPotential(amp=amp,q=Q0)
    of=Orbit(IC); of.turn_physical_off(); of.integrate(t_fwd,pot,method="dop853_c")
    ob=Orbit(IC); ob.turn_physical_off(); ob.integrate(t_back,pot,method="dop853_c")
    cart=lambda o,ts: numpy.column_stack([o.x(ts),o.y(ts),o.z(ts),o.vx(ts),o.vy(ts),o.vz(ts)])
    return numpy.concatenate([cart(ob,t_back)[::-1],cart(of,t_fwd)[1:]],axis=0)
def sample6(amp):
    sp=fardal15spraydf(M,progenitor=Orbit(IC),pot=LogarithmicHaloPotential(amp=amp,q=Q0),tdisrupt=TD)
    numpy.random.seed(SEED); xv,_=sp._sample_tail(600,True,leading=True)
    return coords.galcencyl_to_galcenrect(*numpy.asarray(xv,float))
def ta_hard(p,c,cols):
    d2=numpy.sum((p[:,None,:][:,:,cols]-c[None,:,:][:,:,cols])**2,axis=-1)
    return tg[numpy.argmin(d2,axis=1)]
p0=sample6(1.0); c0=curve6(1.0)
COLS={"position only (3D)":[0,1,2], "position+velocity (6D, production)":[0,1,2,3,4,5]}
for name,cols in COLS.items():
    ta0=ta_hard(p0,c0,cols)
    print(f"{name}:")
    for h in (1e-4,1e-3,1e-2):
        tap=ta_hard(sample6(1.0+h),curve6(1.0+h),cols)
        slope=float(numpy.median(numpy.abs(tap-ta0)))/h
        nz=int(numpy.sum(numpy.abs(tap-ta0)<0.5*dtg))
        njump=int(numpy.sum(numpy.abs(tap-ta0)>3*dtg))
        print(f"   h={h:g}: median|dta|/h={slope:8.3f}  #unchanged={nz}/600  #big-jumps={njump}/600")
