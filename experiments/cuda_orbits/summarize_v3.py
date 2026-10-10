"""Matched-accuracy table for the v3 experiments: accuracy (CPU scan, accuracy_*.json)
joined with GPU timings (bench.py / bench_symp.py JSONs in a directory).

    python summarize_v3.py data/titan/v3b [--md]

Accuracy = max over the 300 reference orbits of |dx|/|x| after 2 Gyr vs galpy dop853_c at
1e-13; dE = max |E(t)-E(0)|/|E(0)| over orbits and saves.  DP-pipe instructions/orbit for
the FP64-efficiency column: force evaluation 2052 (galpy arithmetic) / 545 (fast) DP
instructions (flopcount.cpp per-force counts x sass_ops.cu weights) + integrator work
(dop853 ~140/feval from the per-orbit counts; symplectic drift+kick 12/feval).
"""
import glob
import json
import os
import sys

D = sys.argv[1]
MD = "--md" in sys.argv
acc = []
for f in glob.glob("data/accuracy_*.json"):
    acc += json.load(open(f))["rows"]
PEAK = {"TITAN": 30 * 4 * 1.582e9, "H100": 132 * 64 * 1.98e9}
FORCE = {"exact": 2052.0, "fast": 545.0}
OVERHEAD = {"dop853": 140.0, "symp": 12.0}


def find_acc(d):
    if "scheme" in d:  # symplectic
        if d["dt_mode"] == "common":
            m = [r for r in acc if r["scheme"] == d["scheme"] and r.get("n_per_output") == d["n_per_output"]]
        else:
            m = [r for r in acc if r["scheme"] == d["scheme"] and r["mode"] == "per_orbit_dt"
                 and abs(r["tol"] / d["tol"] - 1) < 1e-6]
    else:
        m = [r for r in acc if r["scheme"] == "dop853" and abs(r["tol"] / d["rtol"] - 1) < 1e-6]
    return m[0] if m else None


rows = []
for f in sorted(glob.glob(os.path.join(D, "*.json"))):
    d = json.load(open(f))
    if not d.get("results"):
        continue
    arith = "fast" if "fast" in d["variant"] else "exact"
    gpu = "H100" if "H100" in d["device"] else "TITAN"
    if "scheme" in d:
        name = d["scheme"] + (f" n={d['n_per_output']}" if d["dt_mode"] == "common" else f" per-orbit tol={d['tol']:g}")
        if d.get("orbits_per_thread", 1) == 2:
            name += " 2/thread"
        name += f" minb={d['minb']}"
        info = d.get("kernel_info", {})
        kind = "symp"
    else:
        name = f"dop853 tol={d['rtol']:g}" + (f" smem{d['nsh']}" if d.get("nsh") else (" refill" if d.get("refill") else " v1"))
        info = d.get("kernel_info_faithful", {})
        kind = "dop853"
    a = find_acc(d)
    for r in d["results"]:
        k_us = r["kernel_ms"] / r["N"] * 1e3
        dp = r["fevals_per_s_kernel"] * (FORCE[arith] + OVERHEAD[kind])
        rows.append(dict(file=os.path.basename(f), gpu=gpu, arith=arith, cfg=name, N=r["N"],
                         kernel_us=k_us, wall_us=r["per_orbit_ms"] * 1e3, nfev=r["nfev_mean"],
                         fev_s=r["fevals_per_s_kernel"], regs=info.get("regs"),
                         occ=info.get("occupancy"), local=info.get("local_bytes"),
                         fp64_eff=dp / PEAK[gpu],
                         pos_err=a["pos_err"] if a else None, dE=a["energy_err"] if a else None,
                         excl={True: "yes", False: "SHARED", None: "?"}[r.get("gpu_exclusive")]))
hdr = ["gpu", "arith", "cfg", "N", "pos_err", "dE", "nfev", "kernel_us", "wall_us", "fev_s", "regs", "occ", "local", "fp64_eff", "excl"]
fmt = {"pos_err": "{:.1e}", "dE": "{:.1e}", "nfev": "{:.0f}", "kernel_us": "{:.2f}", "wall_us": "{:.2f}",
       "fev_s": "{:.3g}", "occ": "{:.0%}", "fp64_eff": "{:.0%}"}
def cell(r, h):
    v = r[h]
    return "" if v is None else (fmt[h].format(v) if h in fmt else str(v))
if MD:
    print("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr))
    for r in rows:
        print("| " + " | ".join(cell(r, h) for h in hdr) + " |")
else:
    for r in rows:
        print("  ".join(f"{cell(r, h):>9s}" if h != "cfg" else f"{cell(r, h):34s}" for h in hdr))
json.dump(rows, open(os.path.join(D, "summary_v3.json"), "w"), indent=1)
