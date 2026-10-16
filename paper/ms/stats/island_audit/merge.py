import json, glob, collections, sys
bk = sys.argv[1]
ev = collections.defaultdict(collections.Counter); tests = collections.defaultdict(set); allt = {}
for f in glob.glob(f"/var/tmp/claude-probes/audit/out2_{bk}/*.json"):
    d = json.load(open(f))
    for s, kinds in d["events"].items(): ev[s].update(kinds)
    for k, t in d["tests"].items(): tests[k].update(t)
    allt.update(d["all"])
print(f"{bk}: tests {len(allt)} ({collections.Counter(allt.values())}), sites {len(ev)}")
site_tests = collections.defaultdict(set)
for k, t in tests.items():
    s, kind = k.rsplit("|", 1); site_tests[s] |= t
rows = sorted(ev.items(), key=lambda kv: -len(site_tests[kv[0]]))
json.dump({"sites": {s: {"kinds": dict(k), "ntests": len(site_tests[s])} for s, k in ev.items()},
           "site_tests": {s: sorted(t) for s, t in site_tests.items()}, "all": allt},
          open(f"/var/tmp/claude-probes/audit/merged_{bk}.json", "w"))
for s, k in rows:
    print(f"{len(site_tests[s]):5d} tests  {dict(k)}  {s}")
