import numpy, base64, io, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT="/tmp/claude-21516/-home-bovy-Repos-galpy/e83d6165-0836-4adf-9f62-52251f942f9d/scratchpad/trackg"
d=numpy.load(f"{OUT}/fit_data.npz"); losses=numpy.array(d["losses"]); amps=numpy.array(d["amps"])
best=numpy.minimum.accumulate(losses)  # best fit-error so far (monotone, honest)
BG="#0b1020"; FG="#e8ecf6"; MUT="#8b93a7"; GRID="#1c2438"
plt.rcParams.update({"figure.facecolor":BG,"axes.facecolor":BG,"savefig.facecolor":BG,
 "text.color":FG,"axes.labelcolor":FG,"xtick.color":MUT,"ytick.color":MUT,
 "axes.edgecolor":GRID,"grid.color":GRID,"font.size":12,"axes.titlecolor":FG,"figure.dpi":130})
fig,ax=plt.subplots(figsize=(11.0,4.5)); ax2=ax.twinx()
st=range(len(losses))
ax.plot(st,best,color="#6ee7ff",lw=2.6,marker="o",ms=3.5,label="fit error (best so far)")
ax2.plot(range(len(amps)),amps,color="#ff7a5c",lw=2.4,marker="s",ms=3.2,label="halo mass  (amp)")
ax2.axhline(1.0,color="#8b93a7",lw=1.0,ls=":")
ax2.text(len(amps)*0.72,1.001,"true halo mass",color=MUT,fontsize=10,va="bottom")
ax.set_yscale("log"); ax.set_xlabel("gradient-descent step (Adam)")
ax.set_ylabel("track mean-squared error",color="#6ee7ff"); ax2.set_ylabel("recovered halo mass  (amp)",color="#ff7a5c")
ax.set_title("Recover the halo mass by descending the generative gradient",loc="left",fontsize=13.5,pad=10)
ax2.tick_params(colors="#ff7a5c"); ax.tick_params(axis="y",colors="#6ee7ff")
[ax.spines[s].set_visible(False) for s in ("top",)]; [ax2.spines[s].set_visible(False) for s in ("top",)]
ax.grid(True,lw=0.5,alpha=0.4)
l1,la1=ax.get_legend_handles_labels(); l2,la2=ax2.get_legend_handles_labels()
ax.legend(l1+l2,la1+la2,frameon=False,fontsize=10,labelcolor=FG,loc="center right")
buf=io.BytesIO(); fig.savefig(buf,format="png",bbox_inches="tight",facecolor=BG); plt.close(fig)
b64=base64.b64encode(buf.getvalue()).decode()
open(f"{OUT}/fig_fit.b64","w").write(b64); open(f"{OUT}/fig_fit.png","wb").write(base64.b64decode(b64))
print("fig_fit (convergence) written:",len(b64)//1024,"KB")
