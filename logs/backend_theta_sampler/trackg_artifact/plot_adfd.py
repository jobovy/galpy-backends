import json, numpy, base64, io, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT="/tmp/claude-21516/-home-bovy-Repos-galpy/e83d6165-0836-4adf-9f62-52251f942f9d/scratchpad/trackg"
d=json.load(open(f"{OUT}/adfd.json")); AD=d["AD"]; hs=numpy.array(d["hs"]); fds=numpy.array(d["fds"])
BG="#0b1020"; FG="#e8ecf6"; MUT="#8b93a7"; GRID="#1c2438"; GOLD="#ffd166"; CYAN="#6ee7ff"; CORAL="#ff7a5c"
plt.rcParams.update({"figure.facecolor":BG,"axes.facecolor":BG,"savefig.facecolor":BG,
 "text.color":FG,"axes.labelcolor":FG,"xtick.color":MUT,"ytick.color":MUT,
 "axes.edgecolor":GRID,"grid.color":GRID,"font.size":12,"axes.titlecolor":FG,"figure.dpi":130})
fig,ax=plt.subplots(figsize=(11.0,4.7))
rel=numpy.abs(fds-AD)/abs(AD); ok=rel<0.05
# danger-zone shading (where FD is >50% off)
bad=rel>0.5
if bad.any():
    ax.axvspan(hs[bad].min()*0.8, hs[bad].max()*1.25, color=CORAL, alpha=0.08, zorder=0)
ax.axhline(AD, color=GOLD, lw=2.4, ls="--", zorder=4)
ax.text(2.5e-3, AD*4.2, "autodiff — exact, one pass", color=GOLD, fontsize=12, va="bottom", ha="center", fontweight="bold")
ax.plot(hs, fds, color=CYAN, lw=1.6, zorder=3, alpha=0.8)
ax.scatter(hs[ok], fds[ok], color=CYAN, s=34, zorder=5, label="finite diff = run streamTrack twice (reliable)")
ax.scatter(hs[~ok], fds[~ok], color=CORAL, s=34, zorder=5, label="finite diff (unreliable — assignment flips)")
ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=1e3)
ax.set_xlabel("finite-difference step size  $h$"); ax.set_ylabel(r"$\partial\,\mathrm{track}\,/\,\partial\,(\mathrm{halo\ mass})$")
ax.set_title("Autodiff gives the exact gradient; finite-differencing streamTrack does not",loc="left",fontsize=13.5,pad=10)
ax.annotate("wrong by 10×,\neven wrong sign", xy=(3e-6,-4.5e4), xytext=(1.3e-5,-3.2e4),
            color=CORAL, fontsize=10.5, ha="left",
            arrowprops=dict(arrowstyle="->", color=CORAL, lw=1.2))
ax.legend(frameon=False,fontsize=10,labelcolor=FG,loc="lower right")
ax.grid(True,lw=0.5,alpha=0.4,which="both"); [ax.spines[s].set_visible(False) for s in ("top","right")]
buf=io.BytesIO(); fig.savefig(buf,format="png",bbox_inches="tight",facecolor=BG); plt.close(fig)
b64=base64.b64encode(buf.getvalue()).decode()
open(f"{OUT}/fig_adfd.b64","w").write(b64); open(f"{OUT}/fig_adfd.png","wb").write(base64.b64decode(b64))
print("fig_adfd written:",len(b64)//1024,"KB")
