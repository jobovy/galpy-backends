"""Markdown tables from run_all.sh's JSON (seconds; best CPU vs GPU)."""
import glob
import json
import os
import sys

res = {}
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.json"))):
    d = json.load(open(f))
    res[(d["backend"], d["device"])] = d["cases"]
cols = [c for c in [("numpy", "cpu"), ("jax", "cpu"), ("jax", "gpu"), ("torch", "cpu"), ("torch", "gpu")] if c in res]
for case in ("orbits", "forces", "forcegrad", "sample"):
    ns = sorted({int(n) for c in cols for n in res[c].get(case, {})})
    if not ns:
        continue
    print(f"\n### {case}\n")
    print("| n | " + " | ".join(f"{b} {d}" for b, d in cols) + " |")
    print("|---|" + "---|" * len(cols))
    for n in ns:
        row = []
        for c in cols:
            v = res[c].get(case, {}).get(str(n))
            row.append("--" if v is None else (v if isinstance(v, str) else f"{v:.3g}"))
        print(f"| {n:.0e} | " + " | ".join(row) + " |")
