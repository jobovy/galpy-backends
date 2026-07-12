import numpy, base64, io, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT="/tmp/claude-21516/-home-bovy-Repos-galpy/e83d6165-0836-4adf-9f62-52251f942f9d/scratchpad/trackg"
d=numpy.load(f"{OUT}/sens_tracks.npz"); base=d["base"]
pert={k:d[f"pert_{k}"] for k in ("amp","q","vR","vz")}
BG="#0b1020"; FG="#e8ecf6"; MUT="#8b93a7"; GRID="#1c2438"; C_BASE="#6b7699"
COL={"amp":"#ff7a5c","q":"#6ee7ff","vR":"#a78bfa","vz":"#34d399"}
plt.rcParams.update({"figure.facecolor":BG,"axes.facecolor":BG,"savefig.facecolor":BG,
 "text.color":FG,"axes.labelcolor":FG,"xtick.color":MUT,"ytick.color":MUT,
 "axes.edgecolor":GRID,"grid.color":GRID,"font.size":12,"axes.titlecolor":FG,"figure.dpi":130})
fig,axes=plt.subplots(1,2,figsize=(11.6,4.6))
CONN=numpy.linspace(0,base.shape[0]-1,22).astype(int)
def panel(ax,title,entries):
    ax.plot(base[:,0],base[:,2],color=C_BASE,lw=3.0,zorder=2,solid_capstyle="round",label="observed track")
    for key,lab in entries:
        p=pert[key]; col=COL[key]
        for i in CONN: ax.plot([base[i,0],p[i,0]],[base[i,2],p[i,2]],color=col,lw=0.8,alpha=0.35,zorder=3)
        ax.plot(p[:,0],p[:,2],color=col,lw=2.0,zorder=4,label=lab)
    ax.set_title(title,loc="left",fontsize=12.5,pad=8)
    ax.set_xlabel("x  [galactocentric, natural units]"); ax.legend(frameon=False,fontsize=9.5,labelcolor=FG,loc="upper left")
    ax.grid(True,lw=0.5,alpha=0.5); [ax.spines[s].set_visible(False) for s in ("top","right")]
panel(axes[0],r"$\partial\,\mathrm{track}\,/\,\partial\,\theta$   (potential)",
      [("amp","heavier halo  (+30% mass)"),("q","rounder halo  (q: 0.90 → 1.01)")])
panel(axes[1],r"$\partial\,\mathrm{track}\,/\,\partial\,\mathrm{progenitor}$   (initial conditions)",
      [("vR",r"radial kick  ($v_R$ +0.10)"),("vz",r"vertical kick  ($v_z$ +0.10)")])
axes[0].set_ylabel("z")
fig.suptitle("Turn any knob and the WHOLE stream re-forms — every particle re-integrates, so the track deforms generatively",
             x=0.012,ha="left",y=1.015,fontsize=10.5,color=MUT)
buf=io.BytesIO(); fig.savefig(buf,format="png",bbox_inches="tight",facecolor=BG); plt.close(fig)
b64=base64.b64encode(buf.getvalue()).decode()
open(f"{OUT}/fig_sens.b64","w").write(b64); open(f"{OUT}/fig_sens.png","wb").write(base64.b64decode(b64))
print("fig_sens (GENERATIVE) written:",len(b64)//1024,"KB")
