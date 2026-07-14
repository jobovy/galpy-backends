import json
d = json.load(open("raw_results.json"))
def cell(s):
    if s.startswith("backend=True finite=True"): return "backend-array, finite"
    if s.startswith("backend=False"): return "**NUMPY** (backend=False)"
    if s.startswith("CRASH"): return "**" + s.replace("CRASH ", "CRASH: ")[:44] + "**"
    if s == "": return "-"
    return s[:44]
order = {"CRASH":0,"NUMPY-FALLTHROUGH":1,"EAGER-ONLY":2,"NOT-INSTANTIATED":3,"OUT-OF-SCOPE":4,"FULLY-BACKEND":5}
rows = sorted(d.items(), key=lambda kv: (order.get(kv[1]["classification"], 9), kv[0]))
out = []
for n, r in rows:
    out.append("| " + " | ".join([n, cell(r["jax_eager"]), cell(r["jax_traced"]),
                                   cell(r["torch"]), r["classification"]]) + " |")
open("table_rows.txt", "w").write("\n".join(out) + "\n")
print("wrote", len(out), "rows")
