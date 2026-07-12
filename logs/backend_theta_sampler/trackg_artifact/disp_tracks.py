import sys, numpy; sys.path.insert(0,"/var/tmp/cint-param")
from galpy.orbit import Orbit
from galpy.potential import LogarithmicHaloPotential
from galpy.df import fardal15spraydf
from galpy.util import conversion
RO,VO=8.0,220.0; Q0=0.9; IC=[1.2,0.15,0.85,0.08,0.05,0.0]; SEED=20260711
M=3*10.0**4.0/conversion.mass_in_msol(VO,RO); TD=20.0
def track(amp):
    sp=fardal15spraydf(M,progenitor=Orbit(IC),pot=LogarithmicHaloPotential(amp=amp,q=Q0),tdisrupt=TD)
    numpy.random.seed(SEED); xv,_=sp._sample_tail(600,True,leading=True)
    tr=sp.streamTrack(particles=numpy.asarray(xv,float),tail="leading",velocity_weight=1.0)
    return numpy.asarray(tr._track_xyz)
import json
amps=json.load(open("fit_numbers.json"))
ts=track(1.08); to=track(1.00); tf=track(amps["fit_amp_final"])
numpy.savez("disp_tracks.npz", tr_start=ts, tr_obs=to, tr_final=tf)
print("start x-range",float(ts[:,0].min()),float(ts[:,0].max()),"| obs",float(to[:,0].min()),float(to[:,0].max()))
print("saved disp_tracks.npz")
