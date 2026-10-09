"""classify.py <junit.xml> <out.tsv>: per failure: nodeid, side (galpy|test|third-party),
deepest galpy frame, one-line error. Prints summaries."""
import sys, re, collections, xml.etree.ElementTree as ET
rows = []
for tc in ET.parse(sys.argv[1]).getroot().iter("testcase"):
    e = tc.find("failure"); e = e if e is not None else tc.find("error")
    if e is None: continue
    txt = e.text or ""; msg = (e.get("message") or "?").splitlines()[0] if e.get("message") else "?"
    frames = re.findall(r"^(\S+\.py):(\d+): in (\S+)", txt, re.M)
    g = [f for f in frames if f[0].startswith("galpy/") or "/galpy-fb/galpy/" in f[0]]
    last = frames[-1][0] if frames else ""
    side = "galpy" if last.startswith("galpy/") or "/galpy-fb/galpy/" in last else ("test" if last.startswith("tests/") else ("third-party" if last else "?"))
    gf = (g[-1][0].split("galpy-fb/")[-1] + ":" + g[-1][1] + " " + g[-1][2]) if g else ""
    rows.append((tc.get("classname").replace(".", "/") + ".py::" + tc.get("name"), side, gf, msg[:200]))
with open(sys.argv[2], "w") as fh:
    for r in rows: fh.write("\t".join(r) + "\n")
print(len(rows), "failures; by side:", dict(collections.Counter(r[1] for r in rows)))
c = collections.Counter((r[1], re.sub(r"\d+", "#", r[3])[:90]) for r in rows)
for (s, m), n in c.most_common(18): print(f"{n:5d} {s:11s} {m}")
print("== top deepest galpy frames")
for k, n in collections.Counter(r[2] for r in rows if r[2]).most_common(15): print(f"{n:5d} {k}")
