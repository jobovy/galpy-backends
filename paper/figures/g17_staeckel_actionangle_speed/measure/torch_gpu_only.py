"""Re-measure g17's torch (GPU) and torch (GPU, +transfer) series ALONE (no jax in
the process), with the same grid, timing helper and N list as make.py's _measure.

Usage: PYTHONPATH=<galpy worktree> python torch_gpu_only.py [path/to/g17.json]
(default: the figure's data/g17_staeckel_actionangle_speed.json; the H100 copy is
data/h100/g17_staeckel_actionangle_speed.json). In one process jax's retained GPU
memory pool starves torch and the N>=1e5 points go OOM (logs/h100/diag on the
gpu-h100 branch); run alone, torch completes them."""
import json, sys, importlib
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
import torch
torch.set_default_dtype(torch.float64)
m = importlib.import_module("paper.figures.g17_staeckel_actionangle_speed.make")  # noqa: E402
from galpy.actionAngle import actionAngleStaeckel
from galpy.potential import MWPotential2014
aF = actionAngleStaeckel(pot=MWPotential2014, delta=0.45, c=False)
JSON = sys.argv[1] if len(sys.argv) > 1 else m._JSON
d = json.load(open(JSON))
Ns_gpu = d["Ns_gpu"]
def safe(fn):
    try:
        return fn()
    except Exception as e:
        print("  skip:", str(e)[:90], flush=True); torch.cuda.empty_cache(); return float("nan")
c, e2e = [], []
for N in Ns_gpu:
    def t(N=N):
        g = [torch.as_tensor(b, device="cuda") for b in m._grid(N)]
        aF.actionsFreqsAngles(*g); torch.cuda.synchronize()
        return m._med(lambda g=g: aF.actionsFreqsAngles(*g), warmup=2, repeats=3, sync="torch")
    def te(N=N):
        gn = m._grid(N)
        return m._med(lambda: aF.actionsFreqsAngles(*[torch.as_tensor(b, device="cuda") for b in gn])[0].cpu().numpy(),
                      warmup=2, repeats=3)
    c.append(safe(t)); e2e.append(safe(te))
    print(f"N={N}: {c[-1]:.4g} s  +transfer {e2e[-1]:.4g} s", flush=True)
d["series"]["torch (GPU)"] = c
d["series"]["torch (GPU, +transfer)"] = e2e
d.setdefault("env", {})["torch_gpu_remeasured"] = "torch GPU series re-measured alone (torch_gpu_only.py)"
json.dump(d, open(JSON, "w"), indent=2)
print("written", JSON)
