"""Honest sensitivity: perturbed tracks from ACTUALLY running the production
streamTrack at each parameter (structure recomputed each time, stars re-sampled)."""
import sys, numpy, base64, io; sys.path.insert(0,"/var/tmp/cint-param")
from galpy.orbit import Orbit
from galpy.potential import LogarithmicHaloPotential
from galpy.df import fardal15spraydf
from galpy.util import conversion
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT="/tmp/claude-21516/-home-bovy-Repos-galpy/e83d6165-0836-4adf-9f62-52251f942f9d/scratchpad/trackg"
RO,VO=8.0,220.0; AMP0,Q0=1.0,0.9; IC=[1.2,0.15,0.85,0.08,0.05,0.0]; SEED=20260711
M=3*10.0**4.0/conversion.mass_in_msol(VO,RO); TD=4.5/conversion.time_in_Gyr(VO,RO)
def full(amp,q,ic):
    sp=fardal15spraydf(M,progenitor=Orbit(list(ic)),pot=LogarithmicHaloPotential(amp=amp,q=q),tdisrupt=TD)
    numpy.random.seed(SEED); xv,_=sp._sample_tail(600,True,leading=True)
    tr=sp.streamTrack(particles=numpy.asarray(xv,float),tail="leading",velocity_weight=1.0)
    return tr
def ic_pert(c,d): x=list(IC); x[c]+=d; return x
tr0=full(AMP0,Q0,IC); xyz0=numpy.asarray(tr0._track_xyz); g0=numpy.asarray(tr0.tp_grid())
P={"amp":(AMP0+0.30,Q0,IC),"q":(AMP0,Q0+0.11,IC),"vR":(AMP0,Q0,ic_pert(1,0.10)),"vz":(AMP0,Q0,ic_pert(4,0.10))}
PT={}; WH={}
for k,(a,q,ic) in P.items():
    trp=full(a,q,ic); PT[k]=numpy.asarray(trp._track_xyz)
    gp=numpy.asarray(trp.tp_grid()); lo,hi=max(g0[0],gp[0]),min(g0[-1],gp[-1])
    tpw=numpy.linspace(lo+1e-3,hi-1e-3,22)
    WH[k]=(numpy.column_stack([numpy.asarray(tr0.x(tpw)),numpy.asarray(tr0.z(tpw))]),
           numpy.column_stack([numpy.asarray(trp.x(tpw)),numpy.asarray(trp.z(tpw))]))
    print(f"  {k}: full-streamTrack deform max|Δ(x,z)|={float(numpy.abs(WH[k][1]-WH[k][0]).max()):.3f}")
# FIGURE
BG="#0b1020"; FG="#e8ecf6"; MUT="#8b93a7"; GRID="#1c2438"; C_BASE="#6b7699"
COL={"amp":"#ff7a5c","q":"#6ee7ff","vR":"#a78bfa","vz":"#34d399"}
plt.rcParams.update({"figure.facecolor":BG,"axes.facecolor":BG,"savefig.facecolor":BG,
 "text.color":FG,"axes.labelcolor":FG,"xtick.color":MUT,"ytick.color":MUT,
 "axes.edgecolor":GRID,"grid.color":GRID,"font.size":12,"axes.titlecolor":FG,"figure.dpi":130})
fig,axes=plt.subplots(1,2,figsize=(11.6,4.6))
def panel(ax,title,entries):
    ax.plot(xyz0[:,0],xyz0[:,2],color=C_BASE,lw=3.0,zorder=2,solid_capstyle="round",label="observed track")
    for key,lab in entries:
        p=PT[key]; col=COL[key]; wb,wp=WH[key]
        for i in range(len(wb)): ax.plot([wb[i,0],wp[i,0]],[wb[i,1],wp[i,1]],color=col,lw=0.8,alpha=0.35,zorder=3)
        ax.plot(p[:,0],p[:,2],color=col,lw=2.0,zorder=4,label=lab)
    ax.set_title(title,loc="left",fontsize=12.5,pad=8); ax.set_xlabel("x  [galactocentric, natural units]")
    ax.legend(frameon=False,fontsize=9.5,labelcolor=FG,loc="upper left")
    ax.grid(True,lw=0.5,alpha=0.5); [ax.spines[s].set_visible(False) for s in ("top","right")]
panel(axes[0],r"$\partial\,\mathrm{track}\,/\,\partial\,\theta$   (potential)",
      [("amp","heavier halo  (+30% mass)"),("q","rounder halo  (q: 0.90 → 1.01)")])
panel(axes[1],r"$\partial\,\mathrm{track}\,/\,\partial\,\mathrm{progenitor}$   (initial conditions)",
      [("vR",r"radial kick  ($v_R$ +0.10)"),("vz",r"vertical kick  ($v_z$ +0.10)")])
axes[0].set_ylabel("z")
fig.suptitle("Turn any knob and re-run streamTrack — each track is the actual production reconstruction of the re-formed stream",
             x=0.012,ha="left",y=1.015,fontsize=10.5,color=MUT)
buf=io.BytesIO(); fig.savefig(buf,format="png",bbox_inches="tight",facecolor=BG); plt.close(fig)
b64=base64.b64encode(buf.getvalue()).decode()
open(f"{OUT}/fig_sens.b64","w").write(b64); open(f"{OUT}/fig_sens.png","wb").write(base64.b64decode(b64))
print("fig_sens (FULL streamTrack) written:",len(b64)//1024,"KB")
