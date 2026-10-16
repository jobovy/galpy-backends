import json, collections, sys
sys.argv = [sys.argv[0], sys.argv[1]]
exec(open("/var/tmp/claude-probes/audit/classify.py").read().split("ORDER = ")[0])
bk = sys.argv[1]
m = json.load(open(f"/var/tmp/claude-probes/audit/merged_{bk}.json"))
G = collections.OrderedDict([
 ("isochroneapprox", lambda f: f.startswith("actionAngle/actionAngleIsochroneApprox")),
 ("streamdf", lambda f: f.startswith("df/streamdf") or f.startswith("df/streamgapdf")),
 ("diskdf", lambda f: f.startswith("df/diskdf")),
 ("scipy roots", lambda f: f.startswith("potential/Potential.py") or f.startswith("potential/TwoPowerSphericalPotential.py") or f in ("orbit/Orbits.py:rguiding", "orbit/Orbits.py:rE", "orbit/Orbits.py:LcE")),
 ("multipole at point", lambda f: f.startswith("potential/MultipoleExpansionPotential") or f.startswith("util/special")),
 ("qdf estimates", lambda f: f.startswith("df/quasiisothermaldf")),
 ("orbit E fallback", lambda f: f == "orbit/Orbits.py:E"),
 ("align_to_orbit", lambda f: f.startswith("util/coords")),
 ("movingobject", lambda f: f.startswith("potential/MovingObject")),
 ("other", lambda f: True),
])
gt = collections.defaultdict(set)
for s, ts in m["site_tests"].items():
    if category(s) != "ISL": continue
    f, line, func = s[len("galpy/"):].rsplit(":", 2); key = f"{f}:{func}"
    for g, p in G.items():
        if p(key): gt[g] |= set(ts); break
allisl = set().union(*gt.values()) if gt else set()
print(bk, "island tests", len(allisl))
for g in G:
    print(f"  {g:20s} {len(gt[g]):4d}")
print("  other funcs:", sorted({s[len('galpy/'):].rsplit(':',1)[0].rsplit(':',1)[0] + ':' + s.rsplit(':',1)[1] for s in m['site_tests'] if category(s)=='ISL' and not any(p(s[len('galpy/'):].rsplit(':',2)[0]+':'+s.rsplit(':',1)[1]) for g,p in list(G.items())[:-1])}))
