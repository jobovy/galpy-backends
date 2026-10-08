"""g17 C (c=True) Staeckel actionsFreqsAngles on K OpenMP threads (multi-core CPU
reference for the GPU series). The C action-angle loop threads over the points with
OpenMP (omp_get_max_threads), so the thread count is OMP_NUM_THREADS; pin the cores.

  OMP_NUM_THREADS=16 taskset -c 16-31 python measure/c_omp_threads.py check  # N=1e5 only
  OMP_NUM_THREADS=16 taskset -c 16-31 python measure/c_omp_threads.py        # full N grid
    -> data/g17_c_16threads.json ({"Ns", "C (c=True, 16 threads)", env stamps})

Same grid (_grid), timing helper (_med) and protocol (warmup 1, median of 5 for
N <= 1e5 else 3) as make.py's "C (c=True)" series.
"""
import json
import os
import platform
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
nthr = int(os.environ["OMP_NUM_THREADS"])  # must be set before galpy's libgomp loads

from paper.figures.g17_staeckel_actionangle_speed.make import _DATA, _grid, _med  # noqa: E402
from galpy.actionAngle import actionAngleStaeckel  # noqa: E402
from galpy.potential import MWPotential2014  # noqa: E402

Ns = [1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000, 300000, 1000000]
if sys.argv[1:] == ["check"]:
    Ns = [100000]
aC = actionAngleStaeckel(pot=MWPotential2014, delta=0.45, c=True)
name = f"C (c=True, {nthr} threads)"
ts = []
for N in Ns:
    g = _grid(N)
    t = _med(lambda g=g: aC.actionsFreqsAngles(*g), warmup=1, repeats=5 if N <= 100000 else 3)
    ts.append(t)
    print(f"{name} N={N:8d}: {t:.4g} s  ({1e3 * t / N:.5f} ms/point)", flush=True)
if sys.argv[1:] != ["check"]:
    import galpy
    import numpy

    env = {"omp_num_threads": nthr, "cpu_affinity": len(os.sched_getaffinity(0)),
           "loadavg_1_5_15": list(os.getloadavg()), "python": platform.python_version(),
           "numpy": numpy.__version__, "galpy": galpy.__version__, "galpy_file": galpy.__file__}
    try:
        with open("/proc/cpuinfo") as fh:
            env["cpu"] = next(l.split(":", 1)[1].strip() for l in fh if l.startswith("model name"))
    except Exception:
        pass
    out = os.path.join(_DATA, f"g17_c_{nthr}threads.json")
    with open(out, "w") as fh:
        json.dump({"Ns": Ns, name: ts, "env": env}, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print("written", out)
