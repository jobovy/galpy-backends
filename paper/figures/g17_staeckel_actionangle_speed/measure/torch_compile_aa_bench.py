"""torch.compile of galpy's vectorized Staeckel transformation (c=False), compiled.

    python torch_compile_aa_bench.py <device> N [N ...]
device: cpu or cuda. Same phase-space grid as the g17 figure / gate_bench.py
(actionAngleStaeckel(pot=MWPotential2014, delta=0.45, c=False).actionsFreqsAngles),
float64. For each N: the first call (compile, dynamic=False so every N is its own
static compile) and the warm call (median of 5), or the error.
"""
import sys, time, traceback, numpy, torch
torch.set_default_dtype(torch.float64)
device, Ns = sys.argv[1], [int(float(v)) for v in sys.argv[2:]]
from galpy.actionAngle import actionAngleStaeckel
from galpy.potential import MWPotential2014
aa = actionAngleStaeckel(pot=MWPotential2014, delta=0.45, c=False)
Rg=numpy.array([0.7,0.9,1.1,1.3]);vRg=numpy.array([-0.3,-0.1,0.1,0.3]);vTg=numpy.array([-0.8,0.6,0.9,1.2]);zg=numpy.array([-0.3,-0.1,0.1,0.3]);vzg=numpy.array([-0.2,0.0,0.15])
G=numpy.meshgrid(Rg,vRg,vTg,zg,vzg,indexing='ij');B=[g.ravel() for g in G];B.append(numpy.linspace(0.1,6.2,B[0].size))
name = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
print(f"# torch.compile aa: torch {torch.__version__} on {name}, threads {torch.get_num_threads()}", flush=True)
f = torch.compile(lambda *a: aa.actionsFreqsAngles(*a), dynamic=False)
g_eager = lambda *a: aa.actionsFreqsAngles(*a)


def args(N):
    reps = (N // B[0].size) + 1
    return tuple(torch.as_tensor(numpy.tile(b, reps)[:N], device=device) for b in B)


def sync():
    if device == "cuda":
        torch.cuda.synchronize()


for N in Ns:
    g = args(N)
    try:
        t0 = time.perf_counter(); ref = g_eager(*g); sync(); te = time.perf_counter() - t0
        t0 = time.perf_counter(); out = f(*g); sync(); tc = time.perf_counter() - t0
        ts = []
        for _ in range(5):
            t0 = time.perf_counter(); out = f(*g); sync(); ts.append(time.perf_counter() - t0)
        err = max(float(torch.max(torch.abs(o - r))) for o, r in zip(out, ref))
        print(f"torch.compile aa {device} N={N:7d} eager {te*1e3:10.2f} ms  compile {tc:8.2f} s  "
              f"warm {sorted(ts)[2]*1e3:10.2f} ms  max|compiled-eager| {err:.2e}", flush=True)
    except Exception as e:
        print(f"torch.compile aa {device} N={N}: ERROR {type(e).__name__}: {str(e)[:400]}", flush=True)
        traceback.print_exc(limit=3)
        break
